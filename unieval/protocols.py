"""Type-based predicates for module matching.

Each predicate uses isinstance checks against concrete model/operator classes.
Function signatures are preserved for backward compatibility — callers continue
to ``from unieval.protocols import is_xxx`` without changes.
"""

from .ann.models.vit import Attention
from .ann.models.uniaffine import (
    UnifiedClipNorm, UniAffineAttention, ReGLUMlp,
    UniAffineBlock, UniAffineModel,
)
from .ann.models.qwen3 import (
    RMSNorm, Qwen3Attention, SwiGLUMlp, Qwen3Block, Qwen3Model,
)
from .ann.models.base import DecoderModelProfile
from .qann.quantization.uniaffine_rules import QUniAffineAttention
from .qann.quantization.qwen3_rules import QQwen3Attention
from .snn.operators.attention import SAttention
from .snn.operators.uniaffine_attention import SpikeUniAffineAttention
from .snn.operators.qwen3_attention import SQwen3Attention


# ---------------------------------------------------------------------------
# Norm predicates
# ---------------------------------------------------------------------------

def is_uclip_like(m):
    """UnifiedClipNorm-like: learnable alpha, gamma, beta with clip bounds."""
    return isinstance(m, UnifiedClipNorm)


def is_rmsnorm_like(m):
    """RMSNorm-like: weight + eps, but NOT alpha/clip_min (excludes UClip)."""
    return isinstance(m, RMSNorm)


# ---------------------------------------------------------------------------
# Attention predicates
# ---------------------------------------------------------------------------

def is_uniaffine_attn_like(m):
    """UniAffine attention (ANN or quantized)."""
    return isinstance(m, (UniAffineAttention, QUniAffineAttention))


def is_softmax_decoder_attn_like(m):
    """Softmax decoder attention (ANN or quantized)."""
    return isinstance(m, (Qwen3Attention, QQwen3Attention))


# ---------------------------------------------------------------------------
# MLP predicates
# ---------------------------------------------------------------------------

def is_reglu_mlp_like(m):
    """ReGLU MLP: gate_proj + up_proj + down_proj + ReLU activation."""
    return isinstance(m, ReGLUMlp)


def is_swiglu_mlp_like(m):
    """SwiGLU MLP: gate_proj + up_proj + down_proj + SiLU activation."""
    return isinstance(m, SwiGLUMlp)


# ---------------------------------------------------------------------------
# Block predicates (compose norm + attn + mlp)
# ---------------------------------------------------------------------------

def is_uniaffine_block_like(m):
    """UniAffine decoder block."""
    return isinstance(m, UniAffineBlock)


def is_qwen3_block_like(m):
    """Qwen3 decoder block."""
    return isinstance(m, Qwen3Block)


# ---------------------------------------------------------------------------
# Model-level predicates
# ---------------------------------------------------------------------------

def is_decoder_model_like(m):
    """Decoder-only model (UniAffine or Qwen3)."""
    return isinstance(m, (UniAffineModel, Qwen3Model))


def is_uniaffine_model_like(m):
    """UniAffine decoder model."""
    return isinstance(m, UniAffineModel)


def is_qwen3_model_like(m):
    """Qwen3 decoder model."""
    return isinstance(m, Qwen3Model)


# ---------------------------------------------------------------------------
# Profile predicates
# ---------------------------------------------------------------------------

def is_decoder_profile(p):
    """Decoder model profile."""
    return isinstance(p, DecoderModelProfile)


# ---------------------------------------------------------------------------
# Spiking attention predicates (for energy discovery)
# ---------------------------------------------------------------------------

def is_spiking_decoder_attn_like(m):
    """Spiking decoder attention (UniAffine or Qwen3)."""
    return isinstance(m, (SpikeUniAffineAttention, SQwen3Attention))


def is_spiking_attention_like(m):
    """Any spiking attention (ViT or decoder)."""
    return isinstance(m, (SAttention, SpikeUniAffineAttention, SQwen3Attention))


# ---------------------------------------------------------------------------
# ViT attention predicate
# ---------------------------------------------------------------------------

def is_vit_attention_like(m):
    """ViT attention: qkv + proj + num_heads."""
    return isinstance(m, Attention)
