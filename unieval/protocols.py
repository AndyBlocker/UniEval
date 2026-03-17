"""Duck-typing predicates for structural matching.

Replaces isinstance checks against models/ classes, allowing framework
core layers (operators/, quantization/, conversion/, evaluation/) to
match arbitrary user models by structure rather than by type.
"""

import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility: unwrap Sequential wrappers (PTQ wraps modules in Sequential)
# ---------------------------------------------------------------------------

def _unwrap(m):
    """If m is nn.Sequential, return its first element; if QNorm, return .norm; otherwise return m."""
    if isinstance(m, nn.Sequential) and len(m) > 0:
        return m[0]
    if hasattr(m, "norm") and hasattr(m, "quan") and not hasattr(m, "blocks"):
        # QNorm composite: unwrap to the inner norm
        return m.norm
    return m


# ---------------------------------------------------------------------------
# Norm predicates
# ---------------------------------------------------------------------------

def is_uclip_like(m):
    """UnifiedClipNorm-like: learnable alpha, gamma, beta with clip bounds."""
    return (hasattr(m, "alpha") and hasattr(m, "gamma")
            and hasattr(m, "beta")
            and hasattr(m, "clip_min") and hasattr(m, "clip_max"))


def is_rmsnorm_like(m):
    """RMSNorm-like: weight + eps, but NOT alpha/clip_min (excludes UClip)."""
    return (hasattr(m, "weight") and hasattr(m, "eps")
            and not hasattr(m, "alpha") and not hasattr(m, "clip_min"))


# ---------------------------------------------------------------------------
# Attention predicates
# ---------------------------------------------------------------------------

def is_uniaffine_attn_like(m):
    """UniAffine attention: qkv_proj + o_proj + core with act_a."""
    return (hasattr(m, "qkv_proj") and hasattr(m, "o_proj")
            and hasattr(m, "core") and hasattr(getattr(m, "core", None), "act_a"))


def is_softmax_decoder_attn_like(m):
    """Softmax decoder attention: qkv_proj + o_proj + scale, but NO core."""
    return (hasattr(m, "qkv_proj") and hasattr(m, "o_proj")
            and hasattr(m, "scale") and not hasattr(m, "core"))


# ---------------------------------------------------------------------------
# MLP predicates
# ---------------------------------------------------------------------------

def _unwrap_act(act):
    """Unwrap activation from Sequential(PTQQuan, act) added by PTQ."""
    if isinstance(act, nn.Sequential) and len(act) > 0:
        return act[-1]
    return act


def is_reglu_mlp_like(m):
    """ReGLU MLP: gate_proj + up_proj + down_proj + ReLU activation.

    Uses _unwrap_act to see through Sequential wrappers added by PTQ.
    """
    return (hasattr(m, "gate_proj") and hasattr(m, "up_proj")
            and hasattr(m, "down_proj") and hasattr(m, "act")
            and isinstance(_unwrap_act(m.act), nn.ReLU))


def is_swiglu_mlp_like(m):
    """SwiGLU MLP: gate_proj + up_proj + down_proj + SiLU activation.

    Uses _unwrap_act to see through Sequential wrappers added by PTQ.
    """
    return (hasattr(m, "gate_proj") and hasattr(m, "up_proj")
            and hasattr(m, "down_proj") and hasattr(m, "act")
            and isinstance(_unwrap_act(m.act), nn.SiLU))


# ---------------------------------------------------------------------------
# Block predicates (compose norm + attn + mlp)
# ---------------------------------------------------------------------------

def is_uniaffine_block_like(m):
    """UniAffine decoder block: UClip norms + UniAffine attn + MLP.

    Uses _unwrap to see through Sequential wrappers added by PTQ.
    """
    return (hasattr(m, "norm1") and is_uclip_like(_unwrap(m.norm1))
            and hasattr(m, "attn") and is_uniaffine_attn_like(m.attn)
            and hasattr(m, "mlp"))


def is_qwen3_block_like(m):
    """Qwen3 decoder block: RMSNorm + softmax attn + MLP.

    Uses _unwrap to see through Sequential wrappers added by PTQ.
    """
    return (hasattr(m, "norm1") and is_rmsnorm_like(_unwrap(m.norm1))
            and hasattr(m, "attn") and is_softmax_decoder_attn_like(m.attn)
            and hasattr(m, "mlp"))


# ---------------------------------------------------------------------------
# Model-level predicates
# ---------------------------------------------------------------------------

def is_decoder_model_like(m):
    """Decoder-only model: embedding + blocks + final_norm."""
    return (hasattr(m, "embedding") and hasattr(m, "blocks")
            and hasattr(m, "final_norm"))


def _first_block(m):
    """Return first block from blocks list, or None."""
    blocks = getattr(m, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        return blocks[0]
    return None


def is_uniaffine_model_like(m):
    """UniAffine decoder model: decoder model with UniAffine blocks."""
    blk = _first_block(m)
    return blk is not None and is_uniaffine_block_like(blk)


def is_qwen3_model_like(m):
    """Qwen3 decoder model: decoder model with Qwen3 blocks."""
    blk = _first_block(m)
    return blk is not None and is_qwen3_block_like(blk)


# ---------------------------------------------------------------------------
# Profile predicates
# ---------------------------------------------------------------------------

def is_decoder_profile(p):
    """Decoder model profile: has seq_len, head_dim, num_kv_heads."""
    return (hasattr(p, "seq_len") and hasattr(p, "head_dim")
            and hasattr(p, "num_kv_heads"))


# ---------------------------------------------------------------------------
# Spiking attention predicates (for energy discovery)
# ---------------------------------------------------------------------------

def is_spiking_decoder_attn_like(m):
    """Spiking decoder attention: has q_IF, k_IF, v_IF neurons."""
    return (hasattr(m, "q_IF") and hasattr(m, "k_IF") and hasattr(m, "v_IF")
            and hasattr(m, "qkv_proj") and hasattr(m, "o_proj"))


def is_spiking_attention_like(m):
    """Any spiking attention (ViT or decoder): has q_IF, k_IF, v_IF neurons."""
    return hasattr(m, "q_IF") and hasattr(m, "k_IF") and hasattr(m, "v_IF")


# ---------------------------------------------------------------------------
# ViT attention predicate (for ops_counter — informational, not used yet)
# ---------------------------------------------------------------------------

def is_vit_attention_like(m):
    """ViT attention: qkv + proj + num_heads, but NOT qkv_proj."""
    return (hasattr(m, "qkv") and hasattr(m, "proj")
            and hasattr(m, "num_heads") and not hasattr(m, "qkv_proj"))
