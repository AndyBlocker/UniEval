"""Spiking UniAffine Attention with GQA and UniAffine activation.

Follows the SAttention pattern but adapted for:
- GQA (Grouped Query Attention): num_heads != num_kv_heads
- UniAffine activation: gamma * clamp(relu(act_a * scores + act_b), 0, 1)
- No softmax path
- RoPE applied to accumulated Q,K (same position per token across timesteps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SNNOperator
from .neurons import STBIFNeuron, _sequential_multistep
from ...ann.operators.rope import apply_rotary_pos_emb


def multi(x1_t, x2_t, x1_sum_t, x2_sum_t):
    """Temporal matmul for Q*K^T: d(QK^T) = Q_acc*dK^T + dQ*K_acc^T - dQ*dK^T."""
    return (x1_sum_t @ x2_t.transpose(-2, -1)
            + x1_t @ x2_sum_t.transpose(-2, -1)
            - x1_t @ x2_t.transpose(-2, -1))


def multi1(x1_t, x2_t, x1_sum_t, x2_sum_t):
    """Temporal matmul for Attn*V: d(AV) = A_acc*dV + dA*V_acc - dA*dV."""
    return x1_sum_t @ x2_t + x1_t @ x2_sum_t - x1_t @ x2_t


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


class SpikeUniAffineAttention(nn.Module, SNNOperator):
    """Spiking UniAffine Attention with GQA.

    Adapted from SAttention for UniAffine models:
    - Separate Q, K, V from fused QKV projection
    - GQA: K,V heads expanded to match Q heads
    - UniAffine activation instead of softmax/ReLU
    - RoPE applied to accumulated Q,K

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

    participates_in_early_stop = False

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
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.core = core
        self.rope = rope

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        qkv_dim = q_dim + 2 * kv_dim
        self.qkv_proj = nn.Linear(hidden_size, qkv_dim, bias=False)

        self.q_IF = neuron_layer(
            q_threshold=torch.tensor(1.0), level=level, sym=True
        )
        self.k_IF = neuron_layer(
            q_threshold=torch.tensor(1.0), level=level, sym=True
        )
        self.v_IF = neuron_layer(
            q_threshold=torch.tensor(1.0), level=level, sym=True
        )
        self.attn_IF = neuron_layer(
            q_threshold=torch.tensor(1.0), level=level, sym=False
        )
        self.after_attn_IF = neuron_layer(
            q_threshold=torch.tensor(1.0), level=level, sym=True
        )

        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)
        self.proj_IF = neuron_layer(
            q_threshold=torch.tensor(1.0), level=level, sym=True
        )

        self.s_uniaffine = Spiking_UniAffineAct(core)
        self.T = 0

    def reset(self):
        self.q_IF.reset()
        self.k_IF.reset()
        self.v_IF.reset()
        self.attn_IF.reset()
        self.after_attn_IF.reset()
        self.proj_IF.reset()
        self.s_uniaffine.reset()
        if hasattr(self.qkv_proj, "reset"):
            self.qkv_proj.reset()
        if hasattr(self.o_proj, "reset"):
            self.o_proj.reset()
        self.T = 0

    def forward(self, x, causal_mask=None):
        B, S, _ = x.shape

        qkv = self.qkv_proj(x)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_dim].reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_dim:q_dim + kv_dim].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim:].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)

        # GQA expand
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        # RoPE on accumulated Q, K
        q_acc = self.q_IF.acc_q * self.q_IF.q_threshold
        k_acc = self.k_IF.acc_q * self.k_IF.q_threshold
        if self.num_kv_groups > 1:
            k_acc = k_acc.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        cos, sin = self.rope(q, seq_len=S)
        q_rot, _ = apply_rotary_pos_emb(q, k, cos, sin)
        _, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        q_acc_rot, k_acc_rot = apply_rotary_pos_emb(q_acc, k_acc, cos, sin)

        # Temporal Q*K^T
        scale = self.core.scale
        attn_diff = multi(
            q_rot * scale, k_rot,
            (q_acc_rot * scale).float(),
            k_acc_rot.float(),
        )

        # Spiking UniAffine activation (causal mask applied inside to accumulated scores)
        attn = self.s_uniaffine(attn_diff, mask=causal_mask)

        attn = self.attn_IF(attn)
        attn_acc = self.attn_IF.acc_q * self.attn_IF.q_threshold

        v_acc = self.v_IF.acc_q * self.v_IF.q_threshold
        if self.num_kv_groups > 1:
            v_acc = v_acc.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        x = multi1(attn, v, attn_acc.float(), v_acc.float())
        x = self.after_attn_IF(x)
        x = x.transpose(1, 2).reshape(B, S, -1)
        x = self.o_proj(x)
        x = self.proj_IF(x)

        self.T += 1
        return x

    def forward_multistep(self, x_seq, causal_mask=None):
        """Sequential multi-step with causal mask support."""
        T = x_seq.shape[0]
        out0 = self.forward(x_seq[0], causal_mask=causal_mask)
        result = torch.empty(T, *out0.shape, device=out0.device, dtype=out0.dtype)
        result[0] = out0
        for t in range(1, T):
            result[t] = self.forward(x_seq[t], causal_mask=causal_mask)
        return result
