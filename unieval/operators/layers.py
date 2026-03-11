"""SNN layer implementations: LLConv2d, LLLinear, Spiking_LayerNorm, SpikeMaxPooling."""

import math

import torch
import torch.nn as nn

from .base import SNNOperator


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


class Spiking_LayerNorm(nn.Module, SNNOperator):
    """Spiking LayerNorm: accumulates input, outputs differential normalized values.

    Args:
        dim: Normalized shape dimension.
    """

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


class SpikeMaxPooling(nn.Module, SNNOperator):
    """Temporal max pooling for SNNs: accumulates and outputs differential pooled values.

    Args:
        maxpool: The original MaxPool module.
    """

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
