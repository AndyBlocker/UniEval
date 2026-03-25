"""Spiking attention module and helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import SNNOperator, CompositeSNNModule
from .neurons import IFNeuron, _sequential_multistep


def multi(x1_t, x2_t, x1_sum_t, x2_sum_t):
    """Efficient temporal matrix multiplication for Q*K^T."""
    return (x1_sum_t @ x2_t.transpose(-2, -1)
            + x1_t @ x2_sum_t.transpose(-2, -1)
            - x1_t @ x2_t.transpose(-2, -1))


def multi1(x1_t, x2_t, x1_sum_t, x2_sum_t):
    """Efficient temporal matrix multiplication for Attn*V."""
    return x1_sum_t @ x2_t + x1_t @ x2_sum_t - x1_t @ x2_t


class spiking_softmax(nn.Module, SNNOperator):
    """Spiking softmax: accumulates input and outputs differential softmax."""

    participates_in_early_stop = False

    def __init__(self):
        super().__init__()
        self.X = 0.0
        self.Y_pre = 0.0

    def reset(self):
        self.X = 0.0
        self.Y_pre = 0.0

    def forward(self, input, mask=None):
        """Accumulate scores and output differential softmax.

        Args:
            input: Score increment for this timestep.
            mask: Optional additive mask (e.g. causal mask with -inf).
                  Applied at each step before softmax but NOT accumulated.
        """
        self.X = input + self.X
        scores = self.X + mask if mask is not None else self.X
        Y = F.softmax(scores, dim=-1)
        if torch.is_tensor(self.Y_pre):
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = self.Y_pre
        self.Y_pre = Y
        return Y - Y_pre

    def forward_multistep(self, x_seq, mask=None):
        """Vectorized multi-step: cumsum + softmax + diff.

        Args:
            x_seq: [T, B, H, N, N]
            mask: Optional additive mask [N, N] or broadcastable.
        Returns:
            output: [T, B, H, N, N]
        """
        X_cum = x_seq.cumsum(dim=0) + self.X
        scores = X_cum + mask if mask is not None else X_cum
        Y = F.softmax(scores, dim=-1)
        if torch.is_tensor(self.Y_pre):
            Y_prev = self.Y_pre.detach().clone().unsqueeze(0)
        else:
            Y_prev = torch.zeros_like(Y[:1])
        Y_shifted = torch.cat([Y_prev, Y[:-1]], dim=0)
        output = Y - Y_shifted
        self.X = X_cum[-1]
        self.Y_pre = Y[-1]
        return output


class SAttention(CompositeSNNModule):
    """Spiking multi-head self-attention.

    Replaces QAttention with temporal spiking dynamics using IFNeurons
    for Q, K, V, attention scores, and projection.

    Args:
        dim: Total embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias: If True, add bias to QKV linear.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
        neuron_layer: Neuron class to use (default: IFNeuron).
        level: Quantization level.
        is_softmax: Whether to use spiking softmax.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        neuron_layer=IFNeuron,
        level=2,
        is_softmax=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.neuron_layer = neuron_layer
        self.level = level
        self.is_softmax = is_softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(
            q_threshold=torch.tensor(1.0), level=self.level, sym=True
        )
        self.k_IF = self.neuron_layer(
            q_threshold=torch.tensor(1.0), level=self.level, sym=True
        )
        self.v_IF = self.neuron_layer(
            q_threshold=torch.tensor(1.0), level=self.level, sym=True
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_IF = self.neuron_layer(
            q_threshold=torch.tensor(1.0), level=self.level, sym=False
        )
        self.after_attn_IF = self.neuron_layer(
            q_threshold=torch.tensor(1.0), level=self.level, sym=True
        )
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_IF = self.neuron_layer(
            q_threshold=torch.tensor(1.0), level=self.level, sym=True
        )
        if self.is_softmax:
            self.Ssoftmax = spiking_softmax()
        self.T = 0

    def reset_local_state(self):
        self.T = 0

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)

        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold
        attn = multi(
            q, k,
            q_acc.float(),
            (self.k_IF.acc_q * self.k_IF.q_threshold).float(),
        )

        if self.is_softmax:
            attn = self.Ssoftmax(attn)

        attn = self.attn_IF(attn)
        if not self.is_softmax:
            attn = attn / N
            acc_attn = self.attn_IF.acc_q * self.attn_IF.q_threshold / N

        attn = self.attn_drop(attn)

        if not self.is_softmax:
            x = multi1(
                attn, v,
                acc_attn.float(),
                (self.v_IF.acc_q * self.v_IF.q_threshold).float(),
            )
        else:
            x = multi1(
                attn, v,
                (self.attn_IF.acc_q * self.attn_IF.q_threshold).float(),
                (self.v_IF.acc_q * self.v_IF.q_threshold).float(),
            )

        x = self.after_attn_IF(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.proj_IF(x)

        self.T += 1
        return x

    def forward_multistep(self, x_seq):
        """Sequential multi-step with pre-allocated output.

        SAttention is inherently sequential (IF neurons maintain running
        state, multi() depends on accumulated values).  Pre-allocation
        avoids list+stack overhead.
        """
        return _sequential_multistep(self, x_seq)
