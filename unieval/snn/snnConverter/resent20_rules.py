import torch
import torch.nn as nn

from .rules import ConversionRule, DEFAULT_CONVERSION_RULES
from .threshold import transfer_threshold
from ..operators.neurons import STBIFNeuron
from ..operators.layers import SpikeConv2dFuseBN, SpikeInferAvgPool, SpikeLinear, SpikeResidualAdd
from ...qann.operators.quanAddition import AdditionQuan
from ...qann.operators.quanAvgPool import QuanAvgPool
from ...qann.operators.quanConv2d import QuanConv2dFuseBN
from ...qann.operators.quanLinear import QuanLinear
from ...ann.models.resnet_cifar10 import BasicBlockCifar

def _match_quanconvfusebn(name, child, parent):
    """Match quantized Qwen3 attention (QQwen3Attention)."""
    return isinstance(child, QuanConv2dFuseBN)


def _match_quanlinear(name, child, parent):
    return isinstance(child, QuanLinear)


def _match_quanaddition(name, child, parent):
    """Match QNorm wrapping a RMSNorm."""
    return isinstance(child, AdditionQuan)


def _match_quanavgpool(name, child, parent):
    return isinstance(child, QuanAvgPool)


def _match_relu(name, child, parent):
    return isinstance(child, nn.ReLU)


def _match_basicblock(name, child, parent):
    return isinstance(child, BasicBlockCifar)

index3 = 0

def _convert_basicblock(name, child, parent, level, neuron_type, **kw):
    """Convert QQwen3Attention to SQwen3Attention with threshold transfer."""
    global index3
    parent._modules[name].conv1 = SpikeConv2dFuseBN(m=child.conv1, relu=True, name=f"Conv2dFuseBN_act{index3}",T=kw["time_step"])
    index3 = index3 + 1
    parent._modules[name].conv2 = SpikeConv2dFuseBN(m=child.conv2, relu=True, name=f"Conv2dFuseBN_act{index3}",T=kw["time_step"])
    index3 = index3 + 1
    if isinstance(child.downsample,torch.nn.Sequential):
        child.downsample[0] = SpikeConv2dFuseBN(m=child.downsample[0], relu=True, name=f"Conv2dFuseBN_act{index3}",T=kw["time_step"])
        child.downsample[2] = nn.Identity()
        index3 = index3 + 1    
    parent._modules[name].relu1 = nn.Identity()
    parent._modules[name].relu2 = nn.Identity()
    parent._modules[name].spikeResidual = SpikeResidualAdd(m=child.ResidualAdd, name=f"ResidualAdd_act{index3}",T=kw["time_step"])
    index3 = index3 + 1    
    parent._modules[name].ResidualAdd = None

def _convert_quanconvfusebn(name, child, parent, level, neuron_type, **kw):
    parent._modules[name] = SpikeConv2dFuseBN(m=child, relu=True, name=f"Conv2dFuseBN_act{index3}",T=kw["time_step"])

def _convert_quanlinear(name, child, parent, level, neuron_type, **kw):
    parent._modules[name] = SpikeLinear(m=child, name=f"SpikeLinear_act{index3}",T=kw["time_step"], directlyOut=False)

def _convert_quanaddition(name, child, parent, level, neuron_type, **kw):
    parent._modules[name] = SpikeResidualAdd(m=child, name=f"ResidualAdd_act{index3}", T=kw["time_step"])

def _convert_quanavgpool(name, child, parent, level, neuron_type, **kw):
    parent._modules[name] = SpikeInferAvgPool(m=child, name=f"InferAvgPool_act{index3}", T=kw["time_step"])
    
def _convert_relu(name, child, parent, level, neuron_type, **kw):
    parent._modules[name] = nn.Identity()


RESNET20_CONVERSION_RULES = [
    ConversionRule("basicblock_to_spike", _match_basicblock, _convert_basicblock, priority=100),
    ConversionRule("quanlinear_to_spiking_linear", _match_quanlinear, _convert_quanlinear, priority=95),
    ConversionRule("quanconvbn_to_spiking_convbn", _match_quanconvfusebn, _convert_quanconvfusebn, priority=92),
    ConversionRule("quanaddition_to_spiking_addition", _match_quanaddition, _convert_quanaddition, priority=90),
    ConversionRule("quanavgpool_to_spiking_avgpool", _match_quanavgpool, _convert_quanavgpool, priority=85),
    ConversionRule("relu_to_identity", _match_relu, _convert_relu, priority=85)
]



