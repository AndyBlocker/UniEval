"""Integrate-and-Fire neuron implementations."""

import torch
import torch.nn as nn

from .base import SNNOperator
from ..registry import NEURON_REGISTRY


def _sequential_multistep(module, x_seq):
    """Efficient sequential multi-step: pre-allocate output to avoid list+stack."""
    T = x_seq.shape[0]
    out0 = module(x_seq[0])
    result = torch.empty(T, *out0.shape, device=out0.device, dtype=out0.dtype)
    result[0] = out0
    for t in range(1, T):
        result[t] = module(x_seq[t])
    return result


@NEURON_REGISTRY.register("IF")
class IFNeuron(nn.Module, SNNOperator):
    """Enhanced ST-BIF neuron with symmetric/asymmetric spike levels.


    Args:
        q_threshold: Firing threshold (learnable scale from quantization).
        level: Quantization level.
        sym: If True, symmetric range [-level//2, level//2-1]; else [0, level-1].
    """

    def __init__(self, q_threshold, level, sym=False):
        super().__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        self.level = torch.tensor(level)
        self.sym = sym
        self.neuron_type = "IF"
        self.is_init = True
        if sym:
            self.pos_max = torch.tensor(level // 2 - 1)
            self.neg_min = torch.tensor(-level // 2)
        else:
            self.pos_max = torch.tensor(level - 1)
            self.neg_min = torch.tensor(0)
        self.eps = 0

    def reset(self):
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False

    def forward(self, input):
        x = input / self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and \
           (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x * self.q_threshold

        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape, dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape, dtype=torch.float32).to(x.device)
            self.q = torch.zeros(x.shape, dtype=torch.float32).to(x.device) + 0.5

        self.is_work = True

        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0) & (self.acc_q < self.pos_max)
        neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        if (x == 0).all() and (self.cur_output == 0).all():
            self.is_work = False

        return self.cur_output * self.q_threshold

    def forward_multistep(self, x_seq):
        return _sequential_multistep(self, x_seq)


@NEURON_REGISTRY.register("ORIIF")
class ORIIFNeuron(nn.Module, SNNOperator):
    """Original Integrate-and-Fire neuron (unsigned, no negative spikes).

    Args:
        q_threshold: Firing threshold.
        level: Quantization level.
        sym: Unused, kept for interface compatibility.
    """

    def __init__(self, q_threshold, level, sym=False):
        super().__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        self.level = torch.tensor(level)
        self.sym = sym
        self.neuron_type = "ORIIF"
        self.is_init = True
        self.pos_max = torch.tensor(level - 1)
        self.neg_min = torch.tensor(0)
        self.eps = 0

    def reset(self):
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False

    def forward(self, input):
        x = input / self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and \
           (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x

        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape, dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape, dtype=torch.float32).to(x.device)
            self.q = torch.zeros(x.shape, dtype=torch.float32).to(x.device) + 0.5

        self.is_work = True

        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1

        if (x == 0).all() and (self.cur_output == 0).all():
            self.is_work = False

        return self.cur_output * self.q_threshold

    def forward_multistep(self, x_seq):
        return _sequential_multistep(self, x_seq)
