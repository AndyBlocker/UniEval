"""SNN layer implementations: LLConv2d, LLLinear, Spiking_LayerNorm, SpikeMaxPooling."""

import math

import torch
import torch.nn as nn
import torch as t

from .base import SNNOperator
from .accumulating_transform import AccumulatingTransform
from .neurons import STBIFNeuron
from unieval.qann.operators import QuanConv2dFuseBN, QuanLinear, QuanAvgPool, AdditionQuan
from unieval.qann.quantization import LsqQuan, LsqQuanAct




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


class Spiking_LayerNorm(AccumulatingTransform):
    """Spiking LayerNorm: accumulates input, outputs differential normalized values.

    Args:
        dim: Normalized shape dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self._transform_attr = "layernorm"


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


def reshape_to_activation(inputs):
    return inputs.reshape(1, -1, 1, 1)

def reshape_to_weight(inputs):
    return inputs.reshape(-1, 1, 1, 1)

def reshape_to_bias(inputs):
    return inputs.reshape(-1)

class SpikeConv2dFuseBN(t.nn.Conv2d, SNNOperator):
    def __init__(self, m: QuanConv2dFuseBN, relu = True, name=f"act", T = 32):
        assert type(m) == QuanConv2dFuseBN
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=True if m.bias is not None else False,
                    padding_mode=m.padding_mode)

        self.m = m
        self.name = name
        self.is_work = False
        self.is_first = self.m.is_first
        self.thd_neg = m.quan_out_fn.thd_neg
        self.thd_pos = m.quan_out_fn.thd_pos

        running_std = torch.sqrt(self.m.running_var + self.m.eps)
        self.weight.data = m.weight.detach().clone()
        # print(self.weight.mean())
        weight = self.weight * reshape_to_weight(self.m.gamma / running_std)
        self.spike = True
        # print("SpikeConv2dFuseBN bias", m.bias)

        if self.bias is not None:
            bias = m.bias.to(self.m.gamma.device) * self.m.gamma / running_std + reshape_to_bias(self.m.beta - self.m.gamma * self.m.running_mean / running_std)
        else:
            bias = reshape_to_bias(self.m.beta - self.m.gamma * self.m.running_mean / running_std)

        
        self.weight = nn.Parameter(m.quan_w_fn(weight.detach()), requires_grad = False)
        self.bias = nn.Parameter(m.quan_out_fn(bias.detach()), requires_grad = False)


        self.neuron = STBIFNeuron(q_threshold=self.m.quan_out_fn.s, level=2**self.m.quan_out_fn.bit, sym=self.m.quan_out_fn.symmetric)
        self.neuron.pos_max = self.m.quan_out_fn.thd_pos
        if relu:
            self.neuron.neg_min = self.neuron.neg_min * 0
        else:
            self.neuron.neg_min = self.m.quan_out_fn.thd_neg
        self.neuron.q_threshold = self.m.quan_out_fn.s
        
        # if self.is_first:
        #     self.neuron_first = STBIFNeuron(q_threshold=self.m.quan_a_fn.s, level=2**self.m.quan_a_fn.bit, sym=self.m.quan_a_fn.symmetric)
        #     self.neuron_first.pos_max = self.m.quan_a_fn.thd_pos
        #     if relu:
        #         self.neuron_first.neg_min = self.neuron_first.neg_min * 0
        #     else:
        #         self.neuron_first.neg_min = self.m.quan_a_fn.thd_neg
        #     self.neuron_first.q_threshold = self.m.quan_a_fn.s            
            
        self.t = 0
        self.accu = 0.0
        self.accu1 = 0.0
        self.accu2 = 0.0
    
    def reset(self):
        self.t = 0
        self.is_work = False
        if hasattr(self, "neuron") and hasattr(self.neuron, "reset"):
            self.neuron.reset()
    
    def forward(self,x):
        # if self.is_first:
        #     x = self.neuron_first(x)
        
        # self.accu1 = self.accu1 + x
        # if self.t == 63:
        #     print("SpikeConv2dFuseBN Input",self.accu1.abs().mean())

        out = self._conv_forward(x, self.weight, bias = None)      
        # self.accu2 = self.accu2 +  out 
        # if self.t == 63:
        #     print("SpikeConv2dFuseBN _conv_forward",self.accu2.abs().mean(), "self.weight", self.weight.abs().mean(),"self.bias",self.bias.abs().mean())
        # print("SpikeConv2dFuseBN","x",x.abs().mean(), "self.weight", self.weight.abs().mean(),"out",out.abs().mean())
        if self.t == 0:
            # print("out.shape",out.shape,"self.bias.shape",self.bias.shape)
            spike_out = self.neuron(out + self.bias.reshape(1,-1,1,1))
        else:
            spike_out = self.neuron(out)
        # print("SpikeConv2dFuseBN", spike_out)

        self.t = self.t + 1
        self.is_work = bool(getattr(self.neuron, "is_work", True))
        # self.accu = self.accu + spike_out
        # if self.t == 64:
        #     print("SpikeConv2dFuseBN Output",self.accu.abs().mean())
        
        return spike_out
    
class SpikeLinear(t.nn.Linear, SNNOperator):
    def __init__(self, m: QuanLinear, name=f"act", T = 32, directlyOut = False):
        assert type(m) == QuanLinear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.m = m
        self.first = True
        self.spike = True
        self.name = name
        self.is_work = False
        self.thd_neg = m.quan_out_fn.thd_neg
        self.thd_pos = m.quan_out_fn.thd_pos

        self.weight = t.nn.Parameter(m.quan_w_fn(m.weight.detach()),requires_grad=False)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.quan_out_fn(m.bias.detach()),requires_grad=False)

        if directlyOut == False:
            self.neuron = STBIFNeuron(q_threshold=self.m.quan_out_fn.s, level=2**self.m.quan_out_fn.bit, sym=self.m.quan_out_fn.symmetric)
            self.neuron.pos_max = self.m.quan_out_fn.thd_pos
            self.neuron.neg_min = self.m.quan_out_fn.thd_neg
            self.neuron.q_threshold = self.m.quan_out_fn.s

        self.directlyOut = directlyOut
        self.t = 0
    
    def reset(self):
        self.t = 0
        self.is_work = False
        if hasattr(self, "neuron") and hasattr(self.neuron, "reset"):
            self.neuron.reset()

    def forward(self,x):
        
        if self.t == 0:
            out = t.nn.functional.linear(x, self.weight) + self.bias
        else:
            out = t.nn.functional.linear(x, self.weight)
        if self.directlyOut == False:
            out = self.neuron(out)

        self.t = self.t + 1
        self.is_work = bool(getattr(self.neuron, "is_work", True)) if (self.directlyOut == False) else True
        return out

class SpikeResidualAdd(t.nn.Module, SNNOperator):
    def __init__(self, m:AdditionQuan, name=f"Residual", T = 32):
        super(SpikeResidualAdd,self).__init__()
        self.neuron = STBIFNeuron(m.quan_a_fn.s.data, 2**m.quan_a_fn.bit, sym=m.quan_a_fn.symmetric)
        self.neuron.q_threshold = m.quan_a_fn.s.data
        self.neuron.pos_max = m.quan_a_fn.thd_pos
        self.neuron.neg_min = m.quan_a_fn.thd_neg
        self.first = True
        self.spike = True
        self.name = name
        self.is_work = False
        self.accu = 0.0
        self.accu1 = 0.0
        self.accu2 = 0.0
        self.t = 0
    
    def reset(self):
        self.is_work = False
        if hasattr(self, "neuron") and hasattr(self.neuron, "reset"):
            self.neuron.reset()
        self.t = 0

    def forward(self,input1,input2):
        # self.accu = self.accu + input1
        # self.accu2 = self.accu2 + input2
        # if self.t == 63:
        #     print("SpikeResidualAdd input1:",self.accu.abs().mean(),"input2:",self.accu2.abs().mean())
        output = self.neuron(input1+input2)
        self.is_work = bool(getattr(self.neuron, "is_work", True))
        # self.accu1 = self.accu1 + output
        # if self.t == 63:
        #     print("SpikeResidualAdd Output:",self.accu1.abs().mean())
        self.t = self.t + 1
        return output

class SpikeInferAvgPool(t.nn.Module, SNNOperator):
    def __init__(self, m: QuanAvgPool, name=f"AvgPool", T = 32):
        super(SpikeInferAvgPool,self).__init__()
        self.m = m
        self.thd_pos = m.quan_out_fn.thd_pos
        self.thd_neg = m.quan_out_fn.thd_neg
        self.neuron = STBIFNeuron(1.0, self.thd_pos - self.thd_neg, self.m.quan_out_fn.symmetric)
        self.neuron.q_threshold = m.quan_out_fn.s
        self.neuron.pos_max = m.quan_out_fn.thd_pos
        self.neuron.neg_min = m.quan_out_fn.thd_neg

        self.first = True 
        self.spike = True
        self.name = name
        self.is_work = False

    def reset(self):
        self.is_work = False
        if hasattr(self, "neuron") and hasattr(self.neuron, "reset"):
            self.neuron.reset()

    def forward(self,x):
        x = self.m.m(x)
        y = self.neuron(x)
        self.is_work = bool(getattr(self.neuron, "is_work", True))
        return y
