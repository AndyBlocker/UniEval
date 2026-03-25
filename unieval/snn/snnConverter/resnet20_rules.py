"""ANN-to-SNN conversion rules for ResNet20 / CIFAR-10 CNN models.

Handles:
- BasicBlockCifar -> SpikeBasicBlock (block-level, replaces entire block)
- QuanConv2dFuseBN -> SpikeConv2dFuseBN (standalone conv outside blocks)
- QuanLinear -> SpikeLinear (classifier)
- AdditionQuan -> SpikeResidualAdd (standalone residual)
- QuanAvgPool -> SpikeInferAvgPool (pooling)
- ReLU -> Identity
"""

import torch.nn as nn

from .rules import ConversionRule
from ..operators.cnn_blocks import SpikeBasicBlock
from ..operators.layers import (
    SpikeConv2dFuseBN, SpikeLinear, SpikeResidualAdd, SpikeInferAvgPool,
)
from ...qann.operators.quanAddition import AdditionQuan
from ...qann.operators.quanAvgPool import QuanAvgPool
from ...qann.operators.quanConv2d import QuanConv2dFuseBN
from ...qann.operators.quanLinear import QuanLinear
from ...ann.models.resnet_cifar10 import BasicBlockCifar


# ---------------------------------------------------------------------------
# Match functions
# ---------------------------------------------------------------------------

def _match_basicblock(name, child, parent):
    return isinstance(child, BasicBlockCifar)


def _match_quanconvfusebn(name, child, parent):
    return isinstance(child, QuanConv2dFuseBN)


def _match_quanlinear(name, child, parent):
    """Match CNN-specific QuanLinear (with activation quantization).

    Weight-only QuanLinear (ViT/decoder via LSQ) has quan_out_fn=None
    and should fall through to default leaf rules.
    """
    return isinstance(child, QuanLinear) and getattr(child, 'quan_out_fn', None) is not None


def _match_quanaddition(name, child, parent):
    return isinstance(child, AdditionQuan)


def _match_quanavgpool(name, child, parent):
    return isinstance(child, QuanAvgPool)


def _match_relu(name, child, parent):
    return isinstance(child, nn.ReLU)


# ---------------------------------------------------------------------------
# Convert functions
# ---------------------------------------------------------------------------

def _convert_basicblock(name, child, parent, ctx, time_step=64, **kw):
    """Replace entire BasicBlockCifar with SpikeBasicBlock."""
    parent._modules[name] = SpikeBasicBlock(
        block=child, ctx=ctx, time_step=time_step,
    )


def _convert_quanconvfusebn(name, child, parent, ctx, time_step=64, **kw):
    parent._modules[name] = SpikeConv2dFuseBN(
        m=child, relu=True,
        name=f"Conv2dFuseBN_act{ctx.next_index('conv')}", T=time_step,
    )


def _convert_quanlinear(name, child, parent, ctx, time_step=64, **kw):
    parent._modules[name] = SpikeLinear(
        m=child, name=f"SpikeLinear_act{ctx.next_index('linear')}",
        T=time_step, directlyOut=False,
    )


def _convert_quanaddition(name, child, parent, ctx, time_step=64, **kw):
    parent._modules[name] = SpikeResidualAdd(
        m=child, name=f"ResidualAdd_act{ctx.next_index('res')}", T=time_step,
    )


def _convert_quanavgpool(name, child, parent, ctx, time_step=64, **kw):
    parent._modules[name] = SpikeInferAvgPool(
        m=child, name=f"InferAvgPool_act{ctx.next_index('pool')}", T=time_step,
    )


def _convert_relu(name, child, parent, **kw):
    parent._modules[name] = nn.Identity()


# ---------------------------------------------------------------------------
# ResNet20 conversion rule set
# ---------------------------------------------------------------------------

RESNET20_CONVERSION_RULES = [
    ConversionRule("basicblock_to_spike", _match_basicblock, _convert_basicblock, priority=100),
    ConversionRule("quanlinear_to_spiking_linear", _match_quanlinear, _convert_quanlinear, priority=95),
    ConversionRule("quanconvbn_to_spiking_convbn", _match_quanconvfusebn, _convert_quanconvfusebn, priority=92),
    ConversionRule("quanaddition_to_spiking_addition", _match_quanaddition, _convert_quanaddition, priority=90),
    ConversionRule("quanavgpool_to_spiking_avgpool", _match_quanavgpool, _convert_quanavgpool, priority=85),
    ConversionRule("relu_to_identity", _match_relu, _convert_relu, priority=85),
]
