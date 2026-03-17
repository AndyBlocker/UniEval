"""Shared RoPE (Rotary Position Embedding) utilities.

Extracted from models/uniaffine.py so that operators/ can use RoPE
without importing from models/.
"""

import torch


def _rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to Q and K tensors.

    Args:
        q: [B, num_heads, S, head_dim]
        k: [B, num_kv_heads, S, head_dim]
        cos, sin: [S, head_dim]
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed
