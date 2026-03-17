"""LSQ Quantizer: rule-based learned step-size quantization strategy."""

import torch.nn as nn

from .base import BaseQuantizer, QuantPlacementRule
from ..operators.lsq import MyQuan, QAttention, QuanConv2d, QuanLinear
from ..operators.composites import QConv2d as QCompConv2d, QLinear as QCompLinear, QNorm
from ...registry import QUANTIZER_REGISTRY


# ---------------------------------------------------------------------------
# Default placement rules for LSQ quantization
# ---------------------------------------------------------------------------

def _match_transformer_block(name, child, parent):
    """Duck-typed match: any module with .attn and .mlp submodules."""
    return hasattr(child, "attn") and hasattr(child, "mlp")


def _apply_transformer_block(name, child, parent, level, is_softmax=True, **kw):
    """Replace attention and add MyQuan to norms and MLP."""
    attn = child.attn
    qattn = QAttention(
        dim=attn.num_heads * attn.head_dim,
        num_heads=attn.num_heads,
        level=level,
        is_softmax=is_softmax,
    )
    qattn.qkv = attn.qkv
    qattn.attn_drop = attn.attn_drop
    qattn.proj = attn.proj
    qattn.proj_drop = attn.proj_drop
    parent._modules[name].attn = qattn
    parent._modules[name].norm1 = QNorm(child.norm1, MyQuan(level, sym=True))
    parent._modules[name].norm2 = QNorm(child.norm2, MyQuan(level, sym=True))
    parent._modules[name].mlp.act = nn.Sequential(
        MyQuan(level, sym=False), child.mlp.act
    )
    parent._modules[name].mlp.fc2 = QCompLinear(child.mlp.fc2, MyQuan(level, sym=True))


def _match_conv2d(name, child, parent):
    return isinstance(child, nn.Conv2d) and not isinstance(child, QuanConv2d)


def _apply_conv2d(name, child, parent, level, **kw):
    parent._modules[name] = QCompConv2d(child, MyQuan(level, sym=True))


def _match_layernorm(name, child, parent):
    return isinstance(child, nn.LayerNorm)


def _apply_layernorm(name, child, parent, level, **kw):
    parent._modules[name] = QNorm(child, MyQuan(level, sym=True))


DEFAULT_LSQ_RULES = [
    QuantPlacementRule("transformer_block", _match_transformer_block, _apply_transformer_block),
    QuantPlacementRule("conv2d", _match_conv2d, _apply_conv2d),
    QuantPlacementRule("layernorm", _match_layernorm, _apply_layernorm),
]


# ---------------------------------------------------------------------------
# LSQ Quantizer
# ---------------------------------------------------------------------------

@QUANTIZER_REGISTRY.register("lsq")
class LSQQuantizer(BaseQuantizer):
    """Learned Step-size Quantizer with rule-based placement.

    Args:
        level: Quantization level.
        weight_bit: Weight quantization bit-width (32 = no weight quant).
        is_softmax: Whether attention uses softmax.
        rules: List of QuantPlacementRule (defaults to DEFAULT_LSQ_RULES).
    """

    def __init__(self, level=16, weight_bit=32, is_softmax=True, rules=None):
        self.level = level
        self.weight_bit = weight_bit
        self.is_softmax = is_softmax
        self.rules = rules or DEFAULT_LSQ_RULES

    def quantize_model(self, model):
        """Apply LSQ quantization to model in-place."""
        self._apply_rules(
            model, self.rules,
            level=self.level,
            is_softmax=self.is_softmax,
        )
        if self.weight_bit < 32:
            self._weight_quantization(model, self.weight_bit)
        return model

    def _weight_quantization(self, model, weight_bit):
        """Recursively quantize Conv2d/Linear weights."""
        children = list(model.named_children())
        for name, child in children:
            if type(child) == nn.Conv2d:
                model._modules[name] = QuanConv2d(
                    m=child, quan_w_fn=MyQuan(level=2 ** weight_bit, sym=True)
                )
            elif type(child) == nn.Linear:
                model._modules[name] = QuanLinear(
                    m=child, quan_w_fn=MyQuan(level=2 ** weight_bit, sym=True)
                )
            else:
                self._weight_quantization(child, weight_bit)
