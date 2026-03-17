"""Conversion rules for ANN-to-SNN conversion."""

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from ..operators.neurons import IFNeuron
from ..operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm
from ..operators.composites import SConv2d, SLinear
from ..operators.attention import SAttention
from ...qann.operators.lsq import MyQuan, QAttention, QuanConv2d, QuanLinear
from ...qann.operators.ptq import PTQQuan
from ...qann.operators.composites import (
    QConv2d as QCompConv2d, QLinear as QCompLinear, QNorm,
)
from .threshold import transfer_threshold


@dataclass
class ConversionRule:
    """A rule for converting ANN modules to SNN modules.

    Args:
        name: Human-readable rule name.
        match_fn: (name, module, parent) -> bool.
        convert_fn: (name, module, parent, **kwargs) -> None, in-place replacement.
        priority: Higher priority rules are checked first.
    """
    name: str
    match_fn: Callable
    convert_fn: Callable
    priority: int = 0


# ---------------------------------------------------------------------------
# Default conversion rule implementations
# ---------------------------------------------------------------------------

def _match_qattention(name, child, parent):
    return isinstance(child, QAttention)


def _convert_qattention(name, child, parent, level, neuron_type, is_softmax, **kw):
    from .wrapper import attn_convert
    sattn = SAttention(
        dim=child.num_heads * child.head_dim,
        num_heads=child.num_heads,
        level=level,
        is_softmax=is_softmax,
        neuron_layer=IFNeuron,
    )
    attn_convert(QAttn=child, SAttn=sattn, level=level, neuron_type=neuron_type)
    parent._modules[name] = sattn


def _match_quan(name, child, parent):
    """Match any quantization module (MyQuan or PTQQuan)."""
    return isinstance(child, (MyQuan, PTQQuan))


def _convert_quan_to_neuron(name, child, parent, level, neuron_type, **kw):
    """Convert MyQuan / PTQQuan -> IFNeuron with transferred thresholds."""
    neurons = IFNeuron(
        q_threshold=torch.tensor(1.0), sym=child.sym, level=child.pos_max
    )
    neurons.q_threshold = child.s.data
    neurons.neuron_type = neuron_type
    neurons.level = level
    neurons.pos_max = child.pos_max
    neurons.neg_min = child.neg_min
    neurons.is_init = False
    parent._modules[name] = neurons


def _match_conv2d(name, child, parent):
    return isinstance(child, (nn.Conv2d, QuanConv2d))


def _convert_conv2d(name, child, parent, neuron_type, level, **kw):
    parent._modules[name] = LLConv2d(child, neuron_type=neuron_type, level=level)


def _match_linear(name, child, parent):
    return isinstance(child, (nn.Linear, QuanLinear))


def _convert_linear(name, child, parent, neuron_type, level, **kw):
    parent._modules[name] = LLLinear(child, neuron_type=neuron_type, level=level)


def _match_layernorm(name, child, parent):
    return isinstance(child, nn.LayerNorm)


def _convert_layernorm(name, child, parent, **kw):
    snn_ln = Spiking_LayerNorm(child.normalized_shape[0])
    if child.elementwise_affine:
        snn_ln.layernorm.weight.data = child.weight.data
        snn_ln.layernorm.bias.data = child.bias.data
    parent._modules[name] = snn_ln


def _match_relu(name, child, parent):
    return isinstance(child, nn.ReLU)


def _convert_relu(name, child, parent, **kw):
    parent._modules[name] = nn.Identity()


# ---------------------------------------------------------------------------
# Composite QANN -> SNN conversion rules (1:1 mapping)
# ---------------------------------------------------------------------------

def _match_qconv2d(name, child, parent):
    return isinstance(child, QCompConv2d)


def _convert_qconv2d(name, child, parent, level, neuron_type, **kw):
    """Convert QConv2d -> SConv2d with threshold transfer."""
    neuron = IFNeuron(
        q_threshold=torch.tensor(1.0), sym=child.quan.sym, level=child.quan.pos_max
    )
    transfer_threshold(child.quan, neuron, neuron_type, level)
    ll_conv = LLConv2d(child.conv, neuron_type=neuron_type, level=level)
    parent._modules[name] = SConv2d(ll_conv, neuron)


def _match_qlinear(name, child, parent):
    return isinstance(child, QCompLinear)


def _convert_qlinear(name, child, parent, level, neuron_type, **kw):
    """Convert QLinear -> SLinear with threshold transfer."""
    neuron = IFNeuron(
        q_threshold=torch.tensor(1.0), sym=child.quan.sym, level=child.quan.pos_max
    )
    transfer_threshold(child.quan, neuron, neuron_type, level)
    ll_linear = LLLinear(child.linear, neuron_type=neuron_type, level=level)
    parent._modules[name] = SLinear(ll_linear, neuron)


def _match_qnorm_layernorm(name, child, parent):
    """Match QNorm wrapping a LayerNorm (direct conversion to Spiking_LayerNorm + IFNeuron).

    Non-LayerNorm QNorms (UClip, RMSNorm) are NOT matched here -- the converter
    recurses into QNorm's children, allowing norm-specific rules (e.g. _match_uclip,
    _match_rmsnorm from decoder rule sets) to handle the inner norm.
    """
    return isinstance(child, QNorm) and isinstance(child.norm, nn.LayerNorm)


def _convert_qnorm_layernorm(name, child, parent, level, neuron_type, **kw):
    """Convert QNorm(LayerNorm) -> Sequential(Spiking_LayerNorm, IFNeuron)."""
    norm = child.norm
    quan = child.quan
    snn_ln = Spiking_LayerNorm(norm.normalized_shape[0])
    if norm.elementwise_affine:
        snn_ln.layernorm.weight.data = norm.weight.data
        snn_ln.layernorm.bias.data = norm.bias.data
    neuron = IFNeuron(
        q_threshold=torch.tensor(1.0), sym=quan.sym, level=quan.pos_max
    )
    transfer_threshold(quan, neuron, neuron_type, level)
    parent._modules[name] = nn.Sequential(snn_ln, neuron)


# ---------------------------------------------------------------------------
# Default rule set (ordered by priority, highest first)
# ---------------------------------------------------------------------------

DEFAULT_CONVERSION_RULES = [
    ConversionRule("qattention_to_sattention", _match_qattention, _convert_qattention, priority=100),
    ConversionRule("quan_to_ifneuron", _match_quan, _convert_quan_to_neuron, priority=90),
    # Composite Q* -> S* rules (priority > old atomic rules)
    ConversionRule("qconv2d_to_sconv2d", _match_qconv2d, _convert_qconv2d, priority=55),
    ConversionRule("qlinear_to_slinear", _match_qlinear, _convert_qlinear, priority=55),
    ConversionRule("qnorm_layernorm_to_spiking", _match_qnorm_layernorm, _convert_qnorm_layernorm, priority=45),
    # Old atomic rules (fallback for decoder MLP sub-conversion, etc.)
    ConversionRule("conv2d_to_llconv2d", _match_conv2d, _convert_conv2d, priority=50),
    ConversionRule("linear_to_lllinear", _match_linear, _convert_linear, priority=50),
    ConversionRule("layernorm_to_spiking", _match_layernorm, _convert_layernorm, priority=40),
    ConversionRule("relu_to_identity", _match_relu, _convert_relu, priority=30),
]
