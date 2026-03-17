"""PTQ placement rules for Qwen3 decoder blocks.

Places PTQQuan at key points in Qwen3Block:
- After RMSNorm (norm1, norm2)
- 6 independent quantizers inside QQwen3Attention (Q, K, V, attn, after_attn, proj)
- Before SiLU gate activation (sym=True)
- After up/down projections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QuantPlacementRule
from ..operators.composites import QLinear as QCompLinear, QNorm
from ..operators.ptq import PTQQuan
from ...protocols import is_qwen3_block_like, is_rmsnorm_like
from ...ann.operators.rope import apply_rotary_pos_emb


class QQwen3Attention(nn.Module):
    """Quantized Qwen3 Attention with 6 internal quantizers.

    Mirrors ViT's QAttention pattern: independent quantizers after Q/K/V split,
    after softmax, after attn*V, and after output projection.

    Args:
        num_heads: Number of Q heads.
        num_kv_heads: Number of K/V heads (GQA).
        head_dim: Per-head dimension.
        rope: RotaryEmbedding from the ANN model.
        level: Quantization level.
        quan_cls: Quantization module class (default: PTQQuan).
    """

    def __init__(self, num_heads, num_kv_heads, head_dim, rope, level, quan_cls=PTQQuan):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_kv_groups = num_heads // num_kv_heads
        self.rope = rope

        # Projections injected by _apply_qwen3_block
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.qkv_proj = nn.Linear(q_dim + 2 * kv_dim, q_dim + 2 * kv_dim, bias=False)  # placeholder
        self.o_proj = nn.Linear(q_dim, q_dim, bias=False)  # placeholder

        # 6 independent quantizers
        self.quan_q = quan_cls(level, sym=True)
        self.quan_k = quan_cls(level, sym=True)
        self.quan_v = quan_cls(level, sym=True)
        self.attn_quan = quan_cls(level, sym=False)    # softmax output [0,1]
        self.after_attn_quan = quan_cls(level, sym=True)
        self.quan_proj = quan_cls(level, sym=True)

    def forward(self, x, causal_mask=None):
        B, S, _ = x.shape

        # 1. Fused QKV projection
        qkv = self.qkv_proj(x)

        # 2. Split Q, K, V
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_dim].reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_dim:q_dim + kv_dim].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim:].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Independent quantization
        q = self.quan_q(q)
        k = self.quan_k(k)
        v = self.quan_v(v)

        # 4. RoPE
        cos, sin = self.rope(q, seq_len=S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 5. GQA expand
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        # 6. Attention: scores -> causal_mask -> softmax -> quantize
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_quan(attn)

        # 7. attn @ V -> quantize
        x = torch.matmul(attn, v)
        x = self.after_attn_quan(x)

        # 8. Output projection -> quantize
        x = x.transpose(1, 2).reshape(B, S, -1)
        x = self.o_proj(x)
        x = self.quan_proj(x)
        return x


def _match_qwen3_block(name, child, parent):
    return is_qwen3_block_like(child)


def _apply_qwen3_block(name, child, parent, level, **kw):
    """Insert PTQQuan in a Qwen3Block."""
    block = parent._modules[name]

    # Norms
    block.norm1 = QNorm(block.norm1, PTQQuan(level, sym=True))
    block.norm2 = QNorm(block.norm2, PTQQuan(level, sym=True))

    # Replace attention with quantized version
    attn = block.attn
    q_attn = QQwen3Attention(
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
        rope=attn.rope,
        level=level,
    )
    q_attn.qkv_proj = attn.qkv_proj  # original Linear
    q_attn.o_proj = attn.o_proj       # original Linear
    block.attn = q_attn

    # MLP
    mlp = block.mlp
    mlp.act = nn.Sequential(PTQQuan(level, sym=True), mlp.act)
    mlp.up_proj = QCompLinear(mlp.up_proj, PTQQuan(level, sym=True))
    mlp.down_proj = QCompLinear(mlp.down_proj, PTQQuan(level, sym=True))


def _match_rmsnorm_standalone(name, child, parent):
    """Match standalone RMSNorm (e.g. final_norm)."""
    return is_rmsnorm_like(child)


def _apply_rmsnorm_standalone(name, child, parent, level, **kw):
    parent._modules[name] = QNorm(child, PTQQuan(level, sym=True))


QWEN3_PTQ_RULES = [
    QuantPlacementRule("qwen3_block", _match_qwen3_block, _apply_qwen3_block),
    QuantPlacementRule("rmsnorm_standalone", _match_rmsnorm_standalone, _apply_rmsnorm_standalone),
]
