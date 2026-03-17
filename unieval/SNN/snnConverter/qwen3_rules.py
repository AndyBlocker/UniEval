"""ANN-to-SNN conversion rules for Qwen3 baseline model.

Handles:
- QQwen3Attention -> SQwen3Attention (spiking softmax + GQA + threshold transfer)
- SwiGLUMlp -> Spiking_SwiGLUMlp (temporal product decomposition with Spiking_SiLU)
- RMSNorm -> Spiking_RMSNorm
- SiLU -> Spiking_SiLU
- Reuses default rules: quan->STBIFNeuron, Linear->LLLinear, ReLU->Identity
"""

import torch
import torch.nn as nn

from .rules import ConversionRule, DEFAULT_CONVERSION_RULES
from .threshold import transfer_threshold
from ..operators.neurons import STBIFNeuron, IFNeuron
from ..operators.layers import LLLinear
from ..operators.decoder_layers import Spiking_RMSNorm, Spiking_SiLU, Spiking_SwiGLUMlp
from ..operators.qwen3_attention import SQwen3Attention
from ...protocols import is_softmax_decoder_attn_like, is_rmsnorm_like, is_swiglu_mlp_like
from ...QANN.operators.lsq import MyQuan
from ...QANN.operators.ptq import PTQQuan
from ...QANN.operators.composites import QNorm


# ---------------------------------------------------------------------------
# Match functions
# ---------------------------------------------------------------------------

def _match_qwen3_attn(name, child, parent):
    """Match quantized Qwen3 attention (QQwen3Attention with quan_q)."""
    return is_softmax_decoder_attn_like(child) and hasattr(child, "quan_q")


def _match_rmsnorm(name, child, parent):
    return is_rmsnorm_like(child)


def _match_qnorm_rmsnorm(name, child, parent):
    """Match QNorm wrapping a RMSNorm."""
    return isinstance(child, QNorm) and is_rmsnorm_like(child.norm)


def _match_swiglu_mlp(name, child, parent):
    return is_swiglu_mlp_like(child)


def _match_silu(name, child, parent):
    return isinstance(child, nn.SiLU)


# ---------------------------------------------------------------------------
# Convert functions
# ---------------------------------------------------------------------------

def _convert_qwen3_attn(name, child, parent, level, neuron_type, **kw):
    """Convert QQwen3Attention to SQwen3Attention with threshold transfer."""
    attn = child

    s_attn = SQwen3Attention(
        hidden_size=attn.num_heads * attn.head_dim,
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
        rope=attn.rope,
        neuron_layer=STBIFNeuron,
        level=level,
    )

    # Transfer projections (original Linear, not wrapped in Sequential)
    s_attn.qkv_proj = LLLinear(attn.qkv_proj, neuron_type="ST-BIF", level=level)
    s_attn.o_proj = LLLinear(attn.o_proj, neuron_type="ST-BIF", level=level)

    # Transfer 6 calibrated thresholds -> IF neurons
    transfer_threshold(attn.quan_q, s_attn.q_IF, neuron_type, level)
    transfer_threshold(attn.quan_k, s_attn.k_IF, neuron_type, level)
    transfer_threshold(attn.quan_v, s_attn.v_IF, neuron_type, level)
    transfer_threshold(attn.attn_quan, s_attn.attn_IF, neuron_type, level)
    transfer_threshold(attn.after_attn_quan, s_attn.after_attn_IF, neuron_type, level)
    transfer_threshold(attn.quan_proj, s_attn.proj_IF, neuron_type, level)

    parent._modules[name] = s_attn


def _convert_rmsnorm(name, child, parent, **kw):
    parent._modules[name] = Spiking_RMSNorm(child)


def _convert_qnorm_rmsnorm(name, child, parent, level, neuron_type, **kw):
    """Convert QNorm(RMSNorm) -> Sequential(Spiking_RMSNorm, IFNeuron)."""
    spiking_norm = Spiking_RMSNorm(child.norm)
    neuron = IFNeuron(
        q_threshold=torch.tensor(1.0), sym=child.quan.sym, level=child.quan.pos_max
    )
    transfer_threshold(child.quan, neuron, neuron_type, level)
    parent._modules[name] = nn.Sequential(spiking_norm, neuron)


def _convert_swiglu_mlp(name, child, parent, neuron_type, level, **kw):
    """Convert SwiGLUMlp to Spiking_SwiGLUMlp.

    First converts sub-modules (Linear->LLLinear, PTQQuan->STBIFNeuron,
    SiLU->Spiking_SiLU), then wraps in Spiking_SwiGLUMlp.
    """
    from .converter import SNNConverter

    mlp = child
    silu_rule = ConversionRule("silu_to_spiking_silu", _match_silu, _convert_silu, priority=85)
    sub_converter = SNNConverter(rules=[silu_rule] + DEFAULT_CONVERSION_RULES)
    sub_converter.convert(mlp, level=level, neuron_type=neuron_type, is_softmax=True)
    parent._modules[name] = Spiking_SwiGLUMlp(mlp)


def _convert_silu(name, child, parent, **kw):
    parent._modules[name] = Spiking_SiLU()


# ---------------------------------------------------------------------------
# Qwen3 conversion rule set
# ---------------------------------------------------------------------------

QWEN3_CONVERSION_RULES = [
    ConversionRule("qwen3_attn_to_spike", _match_qwen3_attn, _convert_qwen3_attn, priority=100),
    ConversionRule("swiglu_to_spiking_swiglu", _match_swiglu_mlp, _convert_swiglu_mlp, priority=95),
    ConversionRule("qnorm_rmsnorm_to_spiking", _match_qnorm_rmsnorm, _convert_qnorm_rmsnorm, priority=92),
    ConversionRule("rmsnorm_to_spiking_rmsnorm", _match_rmsnorm, _convert_rmsnorm, priority=90),
    ConversionRule("silu_to_spiking_silu", _match_silu, _convert_silu, priority=85),
]
