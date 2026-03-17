"""Explicit QANN composite operators: QConv2d, QLinear, QNorm.

These replace the ad-hoc Sequential(module, quantizer) pattern with
named composites that can be 1:1 matched by conversion rules.

Each composite bundles a connection/normalization module with a
quantizer, forming a complete QANN operator:
- QConv2d: Conv2d + Quantizer
- QLinear: Linear + Quantizer
- QNorm: LayerNorm/RMSNorm/UClipNorm + Quantizer
"""

import torch.nn as nn


class QConv2d(nn.Module):
    """Conv2d + Quantizer composite.

    Replaces Sequential(Conv2d, MyQuan/PTQQuan) with a named type
    that conversion rules can match via isinstance.

    Args:
        conv: The original nn.Conv2d module.
        quan: Quantization module (MyQuan or PTQQuan).
    """

    def __init__(self, conv, quan):
        super().__init__()
        self.conv = conv
        self.quan = quan

    def forward(self, x):
        return self.quan(self.conv(x))


class QLinear(nn.Module):
    """Linear + Quantizer composite.

    Replaces Sequential(Linear, MyQuan/PTQQuan) with a named type
    that conversion rules can match via isinstance.

    Args:
        linear: The original nn.Linear module.
        quan: Quantization module (MyQuan or PTQQuan).
    """

    def __init__(self, linear, quan):
        super().__init__()
        self.linear = linear
        self.quan = quan

    def forward(self, x):
        return self.quan(self.linear(x))


class QNorm(nn.Module):
    """Norm + Quantizer composite.

    Replaces Sequential(LayerNorm/RMSNorm/UClipNorm, MyQuan/PTQQuan)
    with a named type that conversion rules can match via isinstance.

    Args:
        norm: The original normalization module.
        quan: Quantization module (MyQuan or PTQQuan).
    """

    def __init__(self, norm, quan):
        super().__init__()
        self.norm = norm
        self.quan = quan

    def forward(self, x):
        return self.quan(self.norm(x))
