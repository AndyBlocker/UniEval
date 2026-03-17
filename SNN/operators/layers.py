"""SNN layer implementations: LLConv2d, LLLinear, Spiking_LayerNorm, SpikeMaxPooling."""

import math

import torch
import torch.nn as nn

from .base import SNNOperator
from .neurons import IFNeuron, STBIFNeuron

class LLConv2d(nn.Module, SNNOperator):
    """Leaky-integration Conv2d wrapper for SNN temporal inference.

    Handles bias leakage across timesteps and zero-input propagation.

    Args:
        conv: The original nn.Conv2d module.
        neuron_type: Neuron type string (e.g. "IF", "ST-BIF").
        level: Quantization level.
    """

    def __init__(self, conv, neuron_type="ST-BIF", level=16):
        super().__init__()
        self.conv = conv
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = neuron_type
        self.level = level
        self.steps = 1
        self.realize_time = self.steps

    def reset(self):
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self, input):
        x = input
        N, C, H, W = x.shape
        F_h, F_w = self.conv.kernel_size
        S_h, S_w = self.conv.stride
        P_h, P_w = self.conv.padding
        C_out = self.conv.out_channels
        H_out = math.floor((H - F_h + 2 * P_h) / S_h) + 1
        W_out = math.floor((W - F_w + 2 * P_w) / S_w) + 1

        if self.zero_output is None:
            self.zero_output = torch.zeros(
                size=(N, C_out, H_out, W_out), device=x.device, dtype=x.dtype
            )

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x == 0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (
                    self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) / self.steps
                    if self.conv.bias is not None else 0.0
                )
                self.realize_time -= 1
                self.is_work = True
                return output
            return self.zero_output

        output = self.conv(x)

        if self.neuron_type == "IF":
            pass
        else:
            if self.conv.bias is not None:
                output = output - self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                if self.realize_time > 0:
                    output = output + (
                        self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) / self.steps
                    )
                    self.realize_time -= 1

        self.is_work = True
        self.first = False
        return output

    def forward_multistep(self, x_seq):
        """Sequential multi-step with pre-allocated output (avoids list+stack)."""
        T = x_seq.shape[0]
        out0 = self.forward(x_seq[0])
        result = torch.empty(T, *out0.shape, device=out0.device, dtype=out0.dtype)
        result[0] = out0
        for t in range(1, T):
            result[t] = self.forward(x_seq[t])
        return result


class LLLinear(nn.Module, SNNOperator):
    """Leaky-integration Linear wrapper for SNN temporal inference.

    Handles bias leakage across timesteps and zero-input propagation.

    Args:
        linear: The original nn.Linear module.
        neuron_type: Neuron type string (e.g. "IF", "ST-BIF").
        level: Quantization level.
    """

    def __init__(self, linear, neuron_type="ST-BIF", level=16):
        super().__init__()
        self.linear = linear
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = neuron_type
        self.level = level
        self.steps = 1
        self.realize_time = self.steps

    def reset(self):
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self, input):
        x = input
        if x.dim() == 3:
            B, N, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, N, D)
        elif x.dim() == 2:
            B, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, D)
        else:
            shape_new = x.shape[:-1] + (self.linear.out_features,)

        if self.zero_output is None:
            self.zero_output = torch.zeros(
                size=shape_new, device=x.device, dtype=x.dtype
            )

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x == 0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (
                    self.linear.bias.data.unsqueeze(0) / self.steps
                    if self.linear.bias is not None else 0.0
                )
                self.realize_time -= 1
                self.is_work = True
                return output
            return self.zero_output

        output = self.linear(x)

        if self.neuron_type == "IF":
            pass
        else:
            if self.linear.bias is not None:
                output = output - self.linear.bias.data.unsqueeze(0)
                if self.realize_time > 0:
                    output = output + (
                        self.linear.bias.data.unsqueeze(0) / self.steps
                    )
                    self.realize_time -= 1

        self.is_work = True
        self.first = False
        return output

    def forward_multistep(self, x_seq):
        """Sequential multi-step with pre-allocated output (avoids list+stack)."""
        T = x_seq.shape[0]
        out0 = self.forward(x_seq[0])
        result = torch.empty(T, *out0.shape, device=out0.device, dtype=out0.dtype)
        result[0] = out0
        for t in range(1, T):
            result[t] = self.forward(x_seq[t])
        return result


class Spiking_LayerNorm(nn.Module, SNNOperator):
    """Spiking LayerNorm: accumulates input, outputs differential normalized values.

    Args:
        dim: Normalized shape dimension.
    """

    participates_in_early_stop = False

    def __init__(self, dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.X = 0.0
        self.Y_pre = None

    def reset(self):
        self.X = 0.0
        self.Y_pre = None

    def forward(self, input):
        self.X = self.X + input
        Y = self.layernorm(self.X)
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

    def forward_multistep(self, x_seq):
        """Vectorized multi-step: cumsum + layernorm + diff.

        Args:
            x_seq: [T, B, N, D]
        Returns:
            output: [T, B, N, D]
        """
        X_cum = x_seq.cumsum(dim=0) + self.X
        Y = self.layernorm(X_cum)
        if self.Y_pre is not None:
            Y_prev = self.Y_pre.detach().clone().unsqueeze(0)
        else:
            Y_prev = torch.zeros_like(Y[:1])
        Y_shifted = torch.cat([Y_prev, Y[:-1]], dim=0)
        output = Y - Y_shifted
        self.X = X_cum[-1]
        self.Y_pre = Y[-1]
        return output


class SpikeMaxPooling(nn.Module, SNNOperator):
    """Temporal max pooling for SNNs: accumulates and outputs differential pooled values.

    Args:
        maxpool: The original MaxPool module.
    """

    participates_in_early_stop = False

    def __init__(self, maxpool):
        super().__init__()
        self.maxpool = maxpool
        self.accumulation = None

    def reset(self):
        self.accumulation = None

    def forward(self, x):
        old_accu = self.accumulation
        if self.accumulation is None:
            self.accumulation = x
        else:
            self.accumulation = self.accumulation + x

        if old_accu is None:
            output = self.maxpool(self.accumulation)
        else:
            output = self.maxpool(self.accumulation) - self.maxpool(old_accu)

        return output

    def forward_multistep(self, x_seq):
        """Vectorized multi-step: cumsum + maxpool + diff.

        Args:
            x_seq: [T, B, C, H, W]
        Returns:
            output: [T, B, C, H', W']
        """
        T, B = x_seq.shape[:2]
        if self.accumulation is not None:
            accu = x_seq.cumsum(dim=0) + self.accumulation
            pooled_prev = self.maxpool(self.accumulation).unsqueeze(0)
        else:
            accu = x_seq.cumsum(dim=0)
            pooled_prev = torch.zeros_like(
                self.maxpool(accu[0])
            ).unsqueeze(0)
        pooled = self.maxpool(accu.reshape(T * B, *accu.shape[2:]))
        pooled = pooled.reshape(T, B, *pooled.shape[1:])
        pooled_shifted = torch.cat([pooled_prev, pooled[:-1]], dim=0)
        output = pooled - pooled_shifted
        self.accumulation = accu[-1]
        return output


class SpikeConv2dFuseBN(t.nn.Conv2d):
    def __init__(self, m: QuanConv2dFuseBN, last_quan_out_fn, is_first=False, name="act"):
        assert type(m) == QuanConv2dFuseBN
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=True if m.bias is not None else False,
                    padding_mode=m.padding_mode)

        self.is_first = is_first
        self.m = m  
        self.name = name 
        self.last_quan_out_fn = last_quan_out_fn
        self.thd_neg = m.quan_out_fn.thd_neg
        self.thd_pos = m.quan_out_fn.thd_pos

        running_std = torch.sqrt(self.m.running_var + self.m.eps)
        self.weight.data = m.weight.cuda()
        # print(self.weight.mean())
        weight = self.weight * reshape_to_weight(self.m.gamma / running_std)
        bias = reshape_to_bias(self.m.beta - self.m.gamma * self.m.running_mean / running_std)

        self.first = True
        self.neuron = 
    
    def forward(self,x):
        quantized_weight = self.quan_w_fn(weight)
        if self.is_first:
            x = self.quan_a_fn(x)
        out = self._conv_forward(x, quantized_weight,bias = None)        
        quantized_bias = self.quan_out_fn(bias)


        quantized_out = torch.clip(self.quan_out_fn(out,clip=False) + quantized_bias.reshape(1,-1,1,1),min=self.quan_out_fn.s*self.quan_out_fn.thd_neg,max=self.quan_out_fn.s*self.quan_out_fn.thd_pos)

        return x
