"""PTQ placement rules for UniAffine decoder blocks.

Places PTQQuan at key points in UniAffineBlock:
- After UnifiedClipNorm (norm1, norm2)
- 6 independent quantizers inside QUniAffineAttention (Q, K, V, attn, after_attn, proj)
- Before ReLU gate activation (sym=False for non-negative output)
- After up/down projections

UniAffine learnable parameters (gamma, act_a, act_b) are NOT quantized.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QuantPlacementRule
from ..operators.composites import QLinear as QCompLinear, QNorm
from ..operators.ptq import PTQQuan
from ...ann.models.uniaffine import UniAffineBlock, UnifiedClipNorm
from ...ann.operators.rope import apply_rotary_pos_emb


class QUniAffineAttention(nn.Module):
    """Quantized UniAffine Attention with 6 internal quantizers.

    Mirrors ViT's QAttention pattern but uses UniAffineCore instead of softmax.
    attn_quan uses sym=False because core output is in [0, 1].

    Args:
        num_heads: Number of Q heads.
        num_kv_heads: Number of K/V heads (GQA).
        head_dim: Per-head dimension.
        core: UniAffineCore module.
        rope: RotaryEmbedding from the ANN model.
        level: Quantization level.
        quan_cls: Quantization module class (default: PTQQuan).
    """

    def __init__(self, num_heads, num_kv_heads, head_dim, core, rope, level, quan_cls=PTQQuan):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.core = core
        self.rope = rope

        # Projections injected by _apply_uniaffine_block
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.qkv_proj = nn.Linear(q_dim + 2 * kv_dim, q_dim + 2 * kv_dim, bias=False)  # placeholder
        self.o_proj = nn.Linear(q_dim, q_dim, bias=False)  # placeholder

        # 6 independent quantizers
        self.quan_q = quan_cls(level, sym=True)
        self.quan_k = quan_cls(level, sym=True)
        self.quan_v = quan_cls(level, sym=True)
        self.attn_quan = quan_cls(level, sym=False)    # core output >= 0
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

        # 6. Attention: scores -> causal_mask -> UniAffineCore -> quantize
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.core.scale
        if causal_mask is not None:
            scores = scores + causal_mask
        attn = self.core(scores)
        attn = self.attn_quan(attn)

        # 7. attn @ V -> quantize
        x = torch.matmul(attn, v)
        x = self.after_attn_quan(x)

        # 8. Output projection -> quantize
        x = x.transpose(1, 2).reshape(B, S, -1)
        x = self.o_proj(x)
        x = self.quan_proj(x)
        return x


def _match_uniaffine_block(name, child, parent):
    return isinstance(child, UniAffineBlock)


def _apply_uniaffine_block(name, child, parent, level, **kw):
    """Insert PTQQuan in a UniAffineBlock."""
    block = parent._modules[name]

    # Norms
    block.norm1 = QNorm(block.norm1, PTQQuan(level, sym=True))
    block.norm2 = QNorm(block.norm2, PTQQuan(level, sym=True))

    # Replace attention with quantized version
    attn = block.attn
    q_attn = QUniAffineAttention(
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
        core=attn.core,
        rope=attn.rope,
        level=level,
    )
    q_attn.qkv_proj = attn.qkv_proj  # original Linear
    q_attn.o_proj = attn.o_proj       # original Linear
    block.attn = q_attn

    # MLP: quantize gate before ReLU (sym=False since ReLU output >= 0)
    mlp = block.mlp
    mlp.act = nn.Sequential(PTQQuan(level, sym=False), mlp.act)
    mlp.up_proj = QCompLinear(mlp.up_proj, PTQQuan(level, sym=True))
    mlp.down_proj = QCompLinear(mlp.down_proj, PTQQuan(level, sym=True))


def _match_uclip_standalone(name, child, parent):
    """Match standalone UnifiedClipNorm (e.g. final_norm)."""
    return isinstance(child, UnifiedClipNorm)


def _apply_uclip_standalone(name, child, parent, level, **kw):
    parent._modules[name] = QNorm(child, PTQQuan(level, sym=True))


UNIAFFINE_PTQ_RULES = [
    QuantPlacementRule("uniaffine_block", _match_uniaffine_block, _apply_uniaffine_block),
    QuantPlacementRule("uclip_standalone", _match_uclip_standalone, _apply_uclip_standalone),
]
