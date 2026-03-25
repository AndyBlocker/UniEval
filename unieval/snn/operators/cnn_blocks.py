"""SNN block replacements for CNN structural containers.

Each class is the SNN equivalent of a quantized ANN block, with a rewritten
forward() that uses spiking sub-modules directly — no hasattr hacks or
conditional branching between ANN/SNN modes.

Design principle: any block whose forward() contains inter-module topology
(residual add, concatenation, gating) needs an explicit SNN block replacement.
"""

import torch.nn as nn

from .base import CompositeSNNModule
from .layers import SpikeConv2dFuseBN, SpikeLinear, SpikeResidualAdd, SpikeInferAvgPool


class SpikeBasicBlock(CompositeSNNModule):
    """SNN replacement for BasicBlockCifar.

    Converts all internal sub-modules (conv, BN-fused conv, residual add)
    to their spiking equivalents during __init__.  The forward() is a clean
    SNN-only path with no conditional logic.

    Args:
        block: The quantized BasicBlockCifar (with QuanConv2dFuseBN children).
        ctx: ConversionContext for layer numbering.
        time_step: Number of SNN timesteps.
    """

    def __init__(self, block, ctx, time_step=64, **kw):
        super().__init__()
        self.conv1 = SpikeConv2dFuseBN(
            m=block.conv1, relu=True,
            name=f"Conv2dFuseBN_act{ctx.next_index('conv')}", T=time_step,
        )
        self.conv2 = SpikeConv2dFuseBN(
            m=block.conv2, relu=True,
            name=f"Conv2dFuseBN_act{ctx.next_index('conv')}", T=time_step,
        )

        if isinstance(block.downsample, nn.Sequential):
            self.downsample = SpikeConv2dFuseBN(
                m=block.downsample[0], relu=True,
                name=f"Conv2dFuseBN_act{ctx.next_index('conv')}", T=time_step,
            )
        else:
            self.downsample = None

        self.residual = SpikeResidualAdd(
            m=block.ResidualAdd,
            name=f"ResidualAdd_act{ctx.next_index('res')}", T=time_step,
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.residual(out, identity)
