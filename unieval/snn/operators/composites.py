"""Composite SNN Operators: SConv2d, SLinear.

Each composite bundles a connection layer (LLConv2d/LLLinear) with an
IFNeuron, forming a complete SNN operator (spike in → spike out).

Design decisions:
- participates_in_early_stop = False: the internal IFNeuron already
  participates, so Judger won't double-check the composite.
- Internal LLConv2d/LLLinear and IFNeuron retain SNNOperator identity,
  so reset_model() traversal is safe (idempotent reset).
- OpsCounter hooks are registered at the inner nn.Conv2d/nn.Linear
  level, so no double-counting occurs.
"""

import torch
import torch.nn as nn

from .base import SNNOperator, CompositeSNNModule
from .neurons import _sequential_multistep


class SConv2d(CompositeSNNModule):
    """Spike-in → LLConv2d → IFNeuron → spike-out.

    Args:
        conv: LLConv2d instance (connection layer).
        neuron: IFNeuron instance (spiking neuron).
    """

    def __init__(self, conv, neuron):
        super().__init__()
        self.conv = conv
        self.neuron = neuron

    def forward(self, x):
        return self.neuron(self.conv(x))

    def forward_multistep(self, x_seq):
        return _sequential_multistep(self, x_seq)


class SLinear(CompositeSNNModule):
    """Spike-in → LLLinear → IFNeuron → spike-out.

    Args:
        linear: LLLinear instance (connection layer).
        neuron: IFNeuron instance (spiking neuron).
    """

    def __init__(self, linear, neuron):
        super().__init__()
        self.linear = linear
        self.neuron = neuron

    def forward(self, x):
        return self.neuron(self.linear(x))

    def forward_multistep(self, x_seq):
        return _sequential_multistep(self, x_seq)
