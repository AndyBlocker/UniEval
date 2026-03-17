"""ANN-to-SNN conversion rules for UniAffine models.

Handles:
- QUniAffineAttention -> SpikeUniAffineAttention (GQA + UniAffine activation + threshold transfer)
- ReGLUMlp -> Spiking_ReGLUMlp (temporal product decomposition)
- UnifiedClipNorm -> Spiking_UnifiedClipNorm
- Reuses default rules: quan->STBIFNeuron, Linear->LLLinear, ReLU->Identity
"""

import torch
import torch.nn as nn

from .rules import ConversionRule, DEFAULT_CONVERSION_RULES
from .threshold import transfer_threshold
from ..operators.neurons import STBIFNeuron, IFNeuron
from ..operators.layers import LLLinear
from ..operators.uniaffine_layers import Spiking_UnifiedClipNorm, Spiking_ReGLUMlp
from ..operators.uniaffine_attention import SpikeUniAffineAttention
from ...protocols import is_uniaffine_attn_like, is_uclip_like, is_reglu_mlp_like
from ...qann.operators.lsq import MyQuan
from ...qann.operators.ptq import PTQQuan
from ...qann.operators.composites import QNorm


# ---------------------------------------------------------------------------
# Match functions
# ---------------------------------------------------------------------------

def _match_uniaffine_attn(name, child, parent):
    """Match quantized UniAffine attention (QUniAffineAttention with quan_q)."""
    return is_uniaffine_attn_like(child) and hasattr(child, "quan_q")


def _match_uclip(name, child, parent):
    return is_uclip_like(child)


def _match_qnorm_uclip(name, child, parent):
    """Match QNorm wrapping a UnifiedClipNorm."""
    return isinstance(child, QNorm) and is_uclip_like(child.norm)


def _match_reglu_mlp(name, child, parent):
    return is_reglu_mlp_like(child)


# ---------------------------------------------------------------------------
# Convert functions
# ---------------------------------------------------------------------------

def _convert_uniaffine_attn(name, child, parent, level, neuron_type, **kw):
    """Convert QUniAffineAttention to SpikeUniAffineAttention with threshold transfer."""
    attn = child

    sgpt = SpikeUniAffineAttention(
        hidden_size=attn.num_heads * attn.head_dim,
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
        core=attn.core,
        rope=attn.rope,
        neuron_layer=STBIFNeuron,
        level=level,
    )

    # Transfer projections (original Linear, not wrapped in Sequential)
    sgpt.qkv_proj = LLLinear(attn.qkv_proj, neuron_type="ST-BIF", level=level)
    sgpt.o_proj = LLLinear(attn.o_proj, neuron_type="ST-BIF", level=level)

    # Transfer 6 calibrated thresholds -> IF neurons
    transfer_threshold(attn.quan_q, sgpt.q_IF, neuron_type, level)
    transfer_threshold(attn.quan_k, sgpt.k_IF, neuron_type, level)
    transfer_threshold(attn.quan_v, sgpt.v_IF, neuron_type, level)
    transfer_threshold(attn.attn_quan, sgpt.attn_IF, neuron_type, level)
    transfer_threshold(attn.after_attn_quan, sgpt.after_attn_IF, neuron_type, level)
    transfer_threshold(attn.quan_proj, sgpt.proj_IF, neuron_type, level)

    parent._modules[name] = sgpt


def _convert_uclip(name, child, parent, **kw):
    parent._modules[name] = Spiking_UnifiedClipNorm(child)


def _convert_qnorm_uclip(name, child, parent, level, neuron_type, **kw):
    """Convert QNorm(UClip) -> Sequential(Spiking_UnifiedClipNorm, IFNeuron)."""
    spiking_norm = Spiking_UnifiedClipNorm(child.norm)
    neuron = IFNeuron(
        q_threshold=torch.tensor(1.0), sym=child.quan.sym, level=child.quan.pos_max
    )
    transfer_threshold(child.quan, neuron, neuron_type, level)
    parent._modules[name] = nn.Sequential(spiking_norm, neuron)


def _convert_reglu_mlp(name, child, parent, neuron_type, level, **kw):
    """Convert ReGLUMlp to Spiking_ReGLUMlp.

    First converts internal sub-modules (Linear->LLLinear, PTQQuan->STBIFNeuron,
    ReLU->Identity), then wraps in Spiking_ReGLUMlp for temporal product.
    """
    from .converter import SNNConverter

    mlp = child
    sub_converter = SNNConverter(rules=DEFAULT_CONVERSION_RULES)
    sub_converter.convert(mlp, level=level, neuron_type=neuron_type, is_softmax=False)
    parent._modules[name] = Spiking_ReGLUMlp(mlp)


# ---------------------------------------------------------------------------
# UniAffine conversion rule set
# ---------------------------------------------------------------------------

UNIAFFINE_CONVERSION_RULES = [
    ConversionRule("uniaffine_attn_to_spike", _match_uniaffine_attn, _convert_uniaffine_attn, priority=100),
    ConversionRule("reglu_to_spiking_reglu", _match_reglu_mlp, _convert_reglu_mlp, priority=95),
    ConversionRule("qnorm_uclip_to_spiking", _match_qnorm_uclip, _convert_qnorm_uclip, priority=92),
    ConversionRule("uclip_to_spiking_uclip", _match_uclip, _convert_uclip, priority=90),
    # Default rules (quan->STBIFNeuron, Linear->LLLinear, ReLU->Identity)
    # are appended in the converter via UNIAFFINE_CONVERSION_RULES + DEFAULT_CONVERSION_RULES
]
