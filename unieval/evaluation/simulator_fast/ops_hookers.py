"""Hook-based SYOPS (Synaptic Operations) counter.

Provides per-layer operation counting with spike rate detection.
New operator hooks can be registered via the OPS_HOOK_REGISTRY.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..feasibility.spike_utils import spike_rate
from ...snn.operators.neurons import IFNeuron, ORIIFNeuron, STBIFNeuron
from ...snn.operators.layers import LLLinear, LLConv2d, Spiking_LayerNorm
from ...snn.operators.layers import SpikeLinear, SpikeConv2dFuseBN, SpikeInferAvgPool, SpikeResidualAdd
from ...snn.operators.composites import SConv2d, SLinear
from ...snn.operators.decoder_layers import Spiking_RMSNorm, Spiking_SiLU, Spiking_SwiGLUMlp
from ...snn.operators.uniaffine_layers import Spiking_UnifiedClipNorm, Spiking_ReGLUMlp
from ...snn.operators.uniaffine_attention import Spiking_UniAffineAct
from ...qann.operators.lsq import MyQuan, QAttention, QuanConv2d
from ...qann.operators.quanConv2d import QuanConv2dFuseBN
from ...qann.operators.quanLinear import QuanLinear
from ...qann.operators.quanAddition import AdditionQuan
from ...qann.operators.quanAvgPool import QuanAvgPool

from ...qann.operators.ptq import PTQQuan
from ...qann.operators.composites import (
    QConv2d as QCompConv2d, QLinear as QCompLinear, QNorm,
)
from ...ann.models.vit import Attention
from ...registry import Registry



# ---------------------------------------------------------------------------
# SYOPS counting hooks
# ---------------------------------------------------------------------------

def empty_syops_hardware_fast_hook(module, input, output):
    module.__syops__ += np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 六个数分别为： #SOP， #Spike，#F_out, #Weight, #membrane, #tracer


def conv_syops_hardware_fast_hook(module, input, output):
    inp = input[0]

    spike, rate, spkhistc = spike_rate(inp)

    batch_size = inp.shape[0]
    output_dims = list(output.shape[2:])
    kernel_dims = list(module.kernel_size)
    in_channels = module.in_channels
    out_channels = module.out_channels
    groups = module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_syops = out_channels * active_elements_count if module.bias is not None else 0
    overall_syops = overall_conv_syops + bias_syops

    module.__syops__[0] += int(overall_syops * rate)
    module.__syops__[1] += int(np.prod(inp.shape) * rate)
    module.__syops__[2] += out_channels // groups
    module.__syops__[3] += int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    module.__syops__[4] += 0
    module.__syops__[5] += 0

    module.__spkhistc__ = spkhistc


def linear_syops_hardware_fast_hook(module, input, output):
    inp = input[0]
    spike, rate, spkhistc = spike_rate(inp)

    batch_size = inp.shape[0]
    input_last_dim = inp.shape[-1]
    output_last_dim = output.shape[-1]
    bias_syops = output_last_dim * batch_size if module.bias is not None else 0
    overall_syops = int(np.prod(inp.shape) * output_last_dim + bias_syops)

    # If this Linear is inside LLLinear/LLConv2d (SNN path), force AC
    # regardless of spike_rate detection. In hardware, all post-IF signals
    # are spike trains; software simulation outputs continuous differentials
    # from norms but these map to AC on neuromorphic hardware.
    force_ac = getattr(module, "__snn_ac_forced__", False)

    #SOP， #Spike，#F_out, #Weight, #membrane, #tracer
    module.__syops__[0] += int(overall_syops * rate)
    module.__syops__[1] += int(np.prod(inp.shape) * rate)
    module.__syops__[2] += output_last_dim
    module.__syops__[3] += int(input_last_dim * output_last_dim)
    module.__syops__[4] += 0
    module.__syops__[5] += 0

    module.__spkhistc__ = spkhistc


def IF_syops_hardware_fast_hook(module, input, output):
    active_elements_count = input[0].numel()
    #SOP， #Spike，#F_out, #Weight, #membrane, #tracer
    module.__syops__[0] += int(active_elements_count)
    module.__syops__[1] += 0
    module.__syops__[2] += 1
    module.__syops__[3] += 0
    module.__syops__[4] += int(active_elements_count)
    module.__syops__[5] += 0
    module.__spkhistc__ = True

def STBIF_syops_hardware_fast_hook(module, input, output):
    # 考虑Spike tracer，memrbane和spike tracer的数量相同，因此直接乘2
    active_elements_count = input[0].numel()
    #SOP， #Spike，#F_out, #Weight, #membrane, #tracer
    module.__syops__[0] += int(active_elements_count) * 2
    module.__syops__[1] += 0
    module.__syops__[2] += 1
    module.__syops__[3] += 0
    module.__syops__[4] += int(active_elements_count)
    module.__syops__[5] += int(active_elements_count)
    module.__spkhistc__ = True

def pool_syops_hardware_fast_hook(module, input, output):
    #SOP， #Spike，#F_out, #Weight, #membrane, #tracer

    inp = input[0]

    spike, rate, spkhistc = spike_rate(inp)

    batch_size = inp.shape[0]
    output_dims = list(output.shape[2:])
    in_channels = out_channels = groups = inp.shape[-1]
    kernel_dims = (module.kernel_size, module.kernel_size)

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_syops = conv_per_position_syops * active_elements_count

    module.__syops__[0] += int(overall_syops * rate)
    module.__syops__[1] += int(np.prod(inp.shape) * rate)
    module.__syops__[2] += out_channels // groups
    module.__syops__[3] += 0
    module.__syops__[4] += 0
    module.__syops__[5] += 0

    module.__spkhistc__ = spkhistc

def multihead_attention_hardware_fast_hook(multihead_attention_module, input, output):
    #SOP， #Spike，#F_out, #Weight, #membrane, #tracer
    x1_t,x2_t,x1_sum_t,x2_sum_t = input[0],input[1],input[2],input[3]
    spike1, rate1, spkhistc1 = spike_rate(x1_t)
    spike2, rate2, spkhistc2 = spike_rate(x2_t)
    
    batchsize1, num_head1, dim11, dim12 = x1_t.shape
    batchsize2, num_head2, dim21, dim22 = x1_t.shape
    
    overall_syops = 0
    overall_syops_sparsity = 0
    
    overall_syops = 2 * batchsize1 * num_head1 * dim11 * dim22 * dim12 + batchsize1 * num_head1 * dim11 * dim22
    overall_syops_sparsity = batchsize1 * num_head1 * dim11 * dim22 * dim12 * (rate1 + rate2) + batchsize1 * num_head1 * dim11 * dim22
    
    multihead_attention_module.__syops__[0] += int(overall_syops_sparsity)
    multihead_attention_module.__syops__[1] += int(x1_t * rate1 + x2_t * rate2)
    
    # 对于不对称的attention算子，例如attn_weight和value乘积，f_ont需要看spike是从哪边来的，这边只能勉强根据spike rate算一个加权平均值
    multihead_attention_module.__syops__[2] += batchsize1 * num_head2 * (dim22*rate1 + dim11*rate2) / (rate1 + rate2) 
    multihead_attention_module.__syops__[3] += 0
    multihead_attention_module.__syops__[4] += batchsize1 * num_head1 * dim11 * dim22
    multihead_attention_module.__syops__[5] += batchsize1 * num_head1 * dim11 * dim22
    
    multihead_attention_module.__spkhistc__ = spkhistc1
    

def spiking_norm_syops_hardware_fast_hook(module, input, output):
    """Hook for Spiking_RMSNorm, Spiking_UnifiedClipNorm, Spiking_LayerNorm.

    These accumulate input and apply norm to the accumulator, then output
    a differential. The ops are element-wise (affine transform + norm).
    """
    #SOP， #Spike，#F_out, #Weight, #membrane, #tracer
    
    inp = input[0]
    spike, rate, spkhistc = spike_rate(inp)
    batch_syops = int(np.prod(inp.shape)) * 2  # multiply + add for affine
    
    module.__syops__[0] += int(batch_syops * rate)
    module.__syops__[1] += int(np.prod(inp.shape) * rate)
    module.__syops__[2] += 1
    module.__syops__[3] += 0
    module.__syops__[4] += 0
    module.__syops__[5] += 0
    module.__spkhistc__ = spkhistc


def spiking_activation_syops_hardware_fast_hook(module, input, output):
    """Hook for Spiking_SiLU, Spiking_UniAffineAct.

    Activation ops: element-wise on accumulated input.
    """
    inp = input[0]
    spike, rate, spkhistc = spike_rate(inp)
    active_elements = inp.numel()
    module.__syops__[0] += int(active_elements * rate)
    module.__syops__[1] += int(np.prod(inp.shape) * rate)
    module.__syops__[2] += 1
    module.__syops__[3] += 0
    module.__syops__[4] += 0
    module.__syops__[5] += 0
    module.__spkhistc__ = spkhistc

def spiking_residual_syops_hardware_fast_hook(module, input, output):
    inp1 = input[0]
    inp2 = input[1]
    spike1, rate1, spkhistc1 = spike_rate(inp1)
    spike2, rate2, spkhistc2 = spike_rate(inp2)
       
    active_elements = 2 * inp1.numel()
    module.__syops__[0] += int(active_elements * rate1 + active_elements * rate2)
    module.__syops__[1] += int(inp1.numel() * rate1 + inp2.numel() * rate2)
    module.__syops__[2] += 1
    module.__syops__[3] += 0
    module.__syops__[4] += 0
    module.__syops__[5] += 0
    module.__spkhistc__ = spkhistc1 and spkhistc2

# ---------------------------------------------------------------------------
# Default module mapping
# ---------------------------------------------------------------------------

def _build_default_modules_hardware_fast_mapping():
    """Build the default mapping of module types to SYOPS hooks."""
    mapping = {
        nn.Conv1d: conv_syops_hardware_fast_hook,
        nn.Conv2d: conv_syops_hardware_fast_hook,
        QuanConv2d: conv_syops_hardware_fast_hook,
        QuanConv2dFuseBN: conv_syops_hardware_fast_hook,
        nn.Conv3d: conv_syops_hardware_fast_hook,
        nn.ReLU: spiking_activation_syops_hardware_fast_hook,
        MyQuan: spiking_activation_syops_hardware_fast_hook,
        nn.PReLU: spiking_activation_syops_hardware_fast_hook,
        nn.ELU: spiking_activation_syops_hardware_fast_hook,
        nn.LeakyReLU: spiking_activation_syops_hardware_fast_hook,
        nn.ReLU6: spiking_activation_syops_hardware_fast_hook,
        nn.MaxPool1d: pool_syops_hardware_fast_hook,
        nn.AvgPool1d: pool_syops_hardware_fast_hook,
        nn.AvgPool2d: pool_syops_hardware_fast_hook,
        nn.MaxPool2d: pool_syops_hardware_fast_hook,
        nn.MaxPool3d: pool_syops_hardware_fast_hook,
        nn.AvgPool3d: pool_syops_hardware_fast_hook,
        nn.AdaptiveMaxPool1d: pool_syops_hardware_fast_hook,
        nn.AdaptiveAvgPool1d: pool_syops_hardware_fast_hook,
        nn.AdaptiveMaxPool2d: pool_syops_hardware_fast_hook,
        nn.AdaptiveAvgPool2d: pool_syops_hardware_fast_hook,
        nn.AdaptiveMaxPool3d: pool_syops_hardware_fast_hook,
        nn.AdaptiveAvgPool3d: pool_syops_hardware_fast_hook,
        IFNeuron: IF_syops_hardware_fast_hook,
        nn.Linear: linear_syops_hardware_fast_hook,
        QuanLinear: linear_syops_hardware_fast_hook,
        nn.Upsample: empty_syops_hardware_fast_hook,
        nn.ConvTranspose1d: conv_syops_hardware_fast_hook,
        nn.ConvTranspose2d: conv_syops_hardware_fast_hook,
        nn.ConvTranspose3d: conv_syops_hardware_fast_hook,
        nn.MultiheadAttention: multihead_attention_hardware_fast_hook,
        Attention: multihead_attention_hardware_fast_hook,
        QAttention: multihead_attention_hardware_fast_hook,
        # SNN layer wrappers (delegates to inner Linear/Conv2d but needs hook for spike rate)
        LLLinear: empty_syops_hardware_fast_hook,    # inner nn.Linear already counted
        LLConv2d: empty_syops_hardware_fast_hook,    # inner nn.Conv2d already counted
        # ResNet:
        SpikeResidualAdd: spiking_residual_syops_hardware_fast_hook,
        SpikeConv2dFuseBN: conv_syops_hardware_fast_hook,
        SpikeLinear: linear_syops_hardware_fast_hook,
        SpikeInferAvgPool: empty_syops_hardware_fast_hook, # (sub-modules already counted)

        # Decoder SNN operators
        ORIIFNeuron: IF_syops_hardware_fast_hook,
        STBIFNeuron: STBIF_syops_hardware_fast_hook,
        PTQQuan: spiking_activation_syops_hardware_fast_hook,
        Spiking_RMSNorm: spiking_norm_syops_hardware_fast_hook,
        Spiking_UnifiedClipNorm: spiking_norm_syops_hardware_fast_hook,
        Spiking_SiLU: spiking_activation_syops_hardware_fast_hook,
        Spiking_UniAffineAct: spiking_activation_syops_hardware_fast_hook,
        # Container modules: use empty hook (sub-modules already counted)
        Spiking_SwiGLUMlp: empty_syops_hardware_fast_hook,
        Spiking_ReGLUMlp: empty_syops_hardware_fast_hook,
        # Composite SNN operators (sub-modules already counted)
        SConv2d: empty_syops_hardware_fast_hook,
        SLinear: empty_syops_hardware_fast_hook,
        # Composite QANN operators (sub-modules already counted if eval at QANN stage)
        QCompConv2d: empty_syops_hardware_fast_hook,
        QCompLinear: empty_syops_hardware_fast_hook,
        QNorm: empty_syops_hardware_fast_hook,
    }
    if hasattr(nn, "GELU"):
        mapping[nn.GELU] = spiking_activation_syops_hardware_fast_hook
    return mapping


# ---------------------------------------------------------------------------
# OpsCounter
# ---------------------------------------------------------------------------

class OpsCounterFast:
    """Hook-based synaptic operations counter.

    Registers forward hooks on supported modules to count operations
    during inference. New operator hooks can be registered via
    register_hook().

    Args:
        custom_hooks: Optional dict of {module_type: hook_fn} to add.
        time_step: Timestep count per forward pass (used for firing rate
            normalization). Defaults to 15 for backward compatibility with
            the original SpikeZIP-TF code.
    """

    def __init__(self, custom_hooks=None, time_step=15):
        self.modules_mapping = _build_default_modules_hardware_fast_mapping()
        if custom_hooks:
            self.modules_mapping.update(custom_hooks)
        self._handles = []
        self._batch_handle = None
        self.time_step = time_step

    def register_hook(self, module_type, hook_fn):
        """Register a SYOPS counting hook for a new module type."""
        self.modules_mapping[module_type] = hook_fn

    def is_supported(self, module):
        return type(module) in self.modules_mapping

    def attach(self, model):
        """Attach counting hooks to all supported modules in the model.

        Also attaches a batch counter to the top-level module.
        Marks nn.Linear/nn.Conv2d inside LLLinear/LLConv2d as SNN-path
        (force AC classification) since their inputs are spike trains
        on neuromorphic hardware, even though software simulation
        produces continuous differentials from spiking norms.
        """
        self._reset_counters(model)
        self._add_batch_counter(model)

        for module in model.modules():
            if self.is_supported(module):
                handle = module.register_forward_hook(
                    self.modules_mapping[type(module)]
                )
                module.__syops_handle__ = handle
                self._handles.append(handle)

    def detach(self, model):
        """Remove all counting hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._batch_handle is not None:
            self._batch_handle.remove()
            self._batch_handle = None

    def _reset_counters(self, model):
        """Initialize or reset counters on all supported modules."""
        model.__batch_counter__ = 0
        model.__times_counter__ = 0
        for module in model.modules():
            if self.is_supported(module):
                module.__syops__ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                module.__params__ = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                module.__spkhistc__ = None

    def _add_batch_counter(self, model):
        """Add a forward hook that counts batches."""
        time_step = self.time_step

        def batch_counter_hook(module, input, output):
            batch_size = 1
            if len(input) > 0:
                batch_size = len(input[0])
            module.__batch_counter__ += batch_size
            module.__times_counter__ += time_step

        self._batch_handle = model.register_forward_hook(batch_counter_hook)

    def compute_total(self, model):
        """Compute total SYOPS across all modules.

        Returns:
            Tuple of (syops_array, total_params).
        """
        syops_sum = self._accumulate(model)
        if model.__batch_counter__ > 0:
            syops_sum = np.array([
                item / model.__batch_counter__ for item in syops_sum
            ])
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return syops_sum, params

    def _accumulate(self, module):
        """Recursively accumulate SYOPS from module tree."""
        if self.is_supported(module):
            return module.__syops__
        total = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for child in module.children():
            total += self._accumulate(child)
        return total

    def get_per_layer_stats(self, model):
        """Get per-layer statistics for all supported modules.

        Returns:
            List of (name, module, accumulated_syops) tuples.
        """
        stats = []
        batch_counter = max(model.__batch_counter__, 1)
        times_counter = max(model.__times_counter__, 1)

        for name, module in model.named_modules():
            if self.is_supported(module):
                syops = self._accumulate(module).copy()
                syops[0] /= batch_counter
                syops[1] /= batch_counter
                syops[2] /= batch_counter
                if syops[2] > 0: # 如果含mac操作则强制为100.0
                    syops[3] = 100.0
                else:
                    syops[3] /= times_counter
                    
                # print("times_counter",times_counter)
                stats.append((name, module, syops))
        return stats
