"""Base class for decoder spiking attention with GQA and RoPE.

Extracts the common forward logic shared by SQwen3Attention and
SpikeUniAffineAttention.  Subclasses provide self.scale and
self.score_act (as attribute or property).

SAttention (ViT) is intentionally NOT covered here -- its structure
differs too much (cls_token, no GQA, no RoPE, dropout paths).
"""

import torch
import torch.nn as nn

from .base import CompositeSNNModule
from .neurons import STBIFNeuron
from .attention import multi, multi1
from ...ann.operators.rope import apply_rotary_pos_emb


class DecoderSpikingAttentionBase(CompositeSNNModule):
    """Decoder GQA spiking attention template.

    Provides the full forward() pipeline:
        QKV split -> IF neurons -> GQA expand -> RoPE on accumulated Q,K
        -> temporal Q*K^T (multi) -> score_act -> attn_IF
        -> temporal Attn*V (multi1) -> after_attn_IF -> o_proj -> proj_IF

    Subclasses must provide (via attribute or property):
        1. self.scale — float or Tensor (use property for buffer-backed scales)
        2. self.score_act — nn.Module with forward(input, mask=None)

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of Q heads.
        num_kv_heads: Number of K/V heads.
        head_dim: Per-head dimension.
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
        rope,
        neuron_layer=STBIFNeuron,
        level=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.rope = rope

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, q_dim + 2 * kv_dim, bias=False)

        self.q_IF = neuron_layer(q_threshold=torch.tensor(1.0), level=level, sym=True)
        self.k_IF = neuron_layer(q_threshold=torch.tensor(1.0), level=level, sym=True)
        self.v_IF = neuron_layer(q_threshold=torch.tensor(1.0), level=level, sym=True)
        self.attn_IF = neuron_layer(q_threshold=torch.tensor(1.0), level=level, sym=False)
        self.after_attn_IF = neuron_layer(q_threshold=torch.tensor(1.0), level=level, sym=True)

        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)
        self.proj_IF = neuron_layer(q_threshold=torch.tensor(1.0), level=level, sym=True)

        self.T = 0
        # Subclass must provide: self.scale, self.score_act (attribute or property)

    def reset_local_state(self):
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

        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        q_acc = self.q_IF.acc_q * self.q_IF.q_threshold
        k_acc = self.k_IF.acc_q * self.k_IF.q_threshold
        if self.num_kv_groups > 1:
            k_acc = k_acc.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        cos, sin = self.rope(q, seq_len=S)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        q_acc_rot, k_acc_rot = apply_rotary_pos_emb(q_acc, k_acc, cos, sin)

        attn_diff = multi(
            q_rot * self.scale, k_rot,
            (q_acc_rot * self.scale).float(),
            k_acc_rot.float(),
        )

        attn = self.score_act(attn_diff, mask=causal_mask)
        attn = self.attn_IF(attn)

        v_acc = self.v_IF.acc_q * self.v_IF.q_threshold
        if self.num_kv_groups > 1:
            v_acc = v_acc.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        x = multi1(
            attn, v,
            (self.attn_IF.acc_q * self.attn_IF.q_threshold).float(),
            v_acc.float(),
        )

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
