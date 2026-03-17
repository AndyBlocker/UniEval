"""PTQ Quantizer: rule-based post-training quantization strategy."""

import torch
import torch.nn as nn

from .base import BaseQuantizer, QuantPlacementRule
from ..operators.lsq import MyQuan, QAttention, QuanConv2d, QuanLinear
from ..operators.ptq import PTQQuan
from ..operators.composites import QConv2d as QCompConv2d, QLinear as QCompLinear, QNorm
from ...registry import QUANTIZER_REGISTRY


@QUANTIZER_REGISTRY.register("ptq")
class PTQQuantizer(BaseQuantizer):
    """Post-Training Quantizer with rule-based placement.

    Uses PTQQuan (KL-div calibration) instead of learnable MyQuan.

    Args:
        level: Quantization level.
        is_softmax: Whether attention uses softmax.
        rules: List of QuantPlacementRule.
    """

    def __init__(self, level=16, is_softmax=True, rules=None):
        self.level = level
        self.is_softmax = is_softmax
        self.rules = rules or self._default_rules()

    def _default_rules(self):
        """Build default PTQ placement rules using PTQQuan."""

        def match_block(name, child, parent):
            return hasattr(child, "attn") and hasattr(child, "mlp")

        def apply_block(name, child, parent, level, is_softmax=True, **kw):
            attn = child.attn
            qattn = QAttention(
                dim=attn.num_heads * attn.head_dim,
                num_heads=attn.num_heads,
                level=level,
                is_softmax=is_softmax,
            )
            # Replace MyQuan inside QAttention with PTQQuan
            qattn.quan_q = PTQQuan(level, sym=True)
            qattn.quan_k = PTQQuan(level, sym=True)
            qattn.quan_v = PTQQuan(level, sym=True)
            qattn.attn_quan = PTQQuan(level, sym=False)
            qattn.after_attn_quan = PTQQuan(level, sym=True)
            qattn.quan_proj = PTQQuan(level, sym=True)
            qattn.qkv = attn.qkv
            qattn.attn_drop = attn.attn_drop
            qattn.proj = attn.proj
            qattn.proj_drop = attn.proj_drop
            parent._modules[name].attn = qattn
            parent._modules[name].norm1 = QNorm(
                child.norm1, PTQQuan(level, sym=True)
            )
            parent._modules[name].norm2 = QNorm(
                child.norm2, PTQQuan(level, sym=True)
            )
            parent._modules[name].mlp.act = nn.Sequential(
                PTQQuan(level, sym=False), child.mlp.act
            )
            parent._modules[name].mlp.fc2 = QCompLinear(
                child.mlp.fc2, PTQQuan(level, sym=True)
            )

        def match_conv2d(name, child, parent):
            return isinstance(child, nn.Conv2d) and not isinstance(child, QuanConv2d)

        def apply_conv2d(name, child, parent, level, **kw):
            parent._modules[name] = QCompConv2d(
                child, PTQQuan(level, sym=True)
            )

        def match_layernorm(name, child, parent):
            return isinstance(child, nn.LayerNorm)

        def apply_layernorm(name, child, parent, level, **kw):
            parent._modules[name] = QNorm(
                child, PTQQuan(level, sym=True)
            )

        return [
            QuantPlacementRule("transformer_block", match_block, apply_block),
            QuantPlacementRule("conv2d", match_conv2d, apply_conv2d),
            QuantPlacementRule("layernorm", match_layernorm, apply_layernorm),
        ]

    def quantize_model(self, model):
        """Apply PTQ quantization to model in-place."""
        self._apply_rules(
            model, self.rules,
            level=self.level,
            is_softmax=self.is_softmax,
        )
        return model

    def calibrate(self, model, dataloader, num_batches=10):
        """Run calibration by forwarding data through the model.

        PTQQuan modules automatically calibrate on their first forward pass.

        Args:
            model: The quantized model.
            dataloader: Calibration data.
            num_batches: Number of batches for calibration.
        """
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for i, (batch, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                batch = batch.to(device)
                model(batch)
