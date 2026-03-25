"""Spiking UniAffine Attention with GQA and UniAffine activation.

Spiking_UniAffineAct: accumulate-diff operator for UniAffine score activation.
SpikeUniAffineAttention: thin subclass of DecoderSpikingAttentionBase.
"""

import torch
import torch.nn as nn

from .base import SNNOperator
from .neurons import STBIFNeuron
from .decoder_attention_base import DecoderSpikingAttentionBase


class Spiking_UniAffineAct(nn.Module, SNNOperator):
    """Spiking UniAffine activation: accumulate scores, apply activation, output diff.

    Pattern:
        S_acc += input
        Y = gamma * clamp(relu(act_a * S_acc + act_b), 0, 1)
        output = Y - Y_pre

    Args:
        core: UniAffineCore module (carries act_a, act_b, gamma).
    """

    participates_in_early_stop = False

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.X = 0.0
        self.Y_pre = None

    def reset(self):
        self.X = 0.0
        self.Y_pre = None

    def forward(self, input, mask=None):
        self.X = self.X + input
        scores = self.X + mask if mask is not None else self.X
        Y = self.core(scores)
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

    def forward_multistep(self, x_seq, mask=None):
        """Vectorized: cumsum + uniaffine_act + diff."""
        X_cum = x_seq.cumsum(dim=0) + self.X
        scores = X_cum + mask if mask is not None else X_cum
        Y = self.core(scores)
        if self.Y_pre is not None:
            Y_prev = self.Y_pre.detach().clone().unsqueeze(0)
        else:
            Y_prev = torch.zeros_like(Y[:1])
        Y_shifted = torch.cat([Y_prev, Y[:-1]], dim=0)
        output = Y - Y_shifted
        self.X = X_cum[-1]
        self.Y_pre = Y[-1]
        return output


class SpikeUniAffineAttention(DecoderSpikingAttentionBase):
    """Spiking UniAffine Attention with GQA.

    Uses UniAffine activation instead of softmax for score normalization.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of Q heads.
        num_kv_heads: Number of K/V heads.
        head_dim: Per-head dimension.
        core: UniAffineCore from the ANN model.
        rope: RotaryEmbedding from the ANN model.
        neuron_layer: Neuron class (default: STBIFNeuron).
        level: Quantization level.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        core,
        rope,
        neuron_layer=STBIFNeuron,
        level=2,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope=rope,
            neuron_layer=neuron_layer,
            level=level,
        )
        self.core = core
        # score activation registered under original name for state_dict compat
        self.s_uniaffine = Spiking_UniAffineAct(core)

    @property
    def scale(self):
        """Read from core.scale at forward time (buffer moves with .to(device))."""
        return self.core.scale

    @property
    def score_act(self):
        """Alias for base class forward() — preserves s_uniaffine state_dict keys."""
        return self.s_uniaffine
