"""Hook-based SYOPS (Synaptic Operations) counter.

Provides per-layer operation counting with spike rate detection.
New operator hooks can be registered via the OPS_HOOK_REGISTRY.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spike_utils import spike_rate
from ..operators.neurons import IFNeuron
from ..quantization.lsq import MyQuan, QAttention, QuanConv2d, QuanLinear
from ..models.vit import Attention
from ..registry import OPS_HOOK_REGISTRY


# ---------------------------------------------------------------------------
# SYOPS counting hooks
# ---------------------------------------------------------------------------

def empty_syops_counter_hook(module, input, output):
    module.__syops__ += np.array([0.0, 0.0, 0.0, 0.0])


def conv_syops_counter_hook(module, input, output):
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

    module.__syops__[0] += int(overall_syops)
    if spike:
        module.__syops__[1] += int(overall_syops) * rate
    else:
        module.__syops__[2] += int(overall_syops)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def linear_syops_counter_hook(module, input, output):
    inp = input[0]
    spike, rate, spkhistc = spike_rate(inp)

    batch_size = inp.shape[0]
    output_last_dim = output.shape[-1]
    bias_syops = output_last_dim * batch_size if module.bias is not None else 0

    module.__syops__[0] += int(np.prod(inp.shape) * output_last_dim + bias_syops)
    if spike:
        module.__syops__[1] += int(np.prod(inp.shape) * output_last_dim + bias_syops) * rate
    else:
        module.__syops__[2] += int(np.prod(inp.shape) * output_last_dim + bias_syops)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def IF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)
    spike, rate, spkhistc = spike_rate(output)
    module.__syops__[1] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def relu_syops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__syops__[0] += int(active_elements_count)
    spike, rate, _ = spike_rate(output)
    if spike:
        module.__syops__[1] += int(active_elements_count) * rate
    else:
        module.__syops__[2] += int(active_elements_count)
    module.__syops__[3] += rate * 100


def pool_syops_counter_hook(module, input, output):
    inp = input[0]
    spike, rate, spkhistc = spike_rate(inp)
    module.__syops__[0] += int(np.prod(inp.shape))
    if spike:
        module.__syops__[1] += int(np.prod(inp.shape)) * rate
    else:
        module.__syops__[2] += int(np.prod(inp.shape))
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def bn_syops_counter_hook(module, input, output):
    inp = input[0]
    spike, rate, spkhistc = spike_rate(inp)
    batch_syops = np.prod(inp.shape)
    if module.affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)
    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def ln_syops_counter_hook(module, input, output):
    inp = input[0]
    spike, rate, spkhistc = spike_rate(inp)
    batch_syops = np.prod(inp.shape)
    if module.elementwise_affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)
    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def multihead_attention_counter_hook(module, input, output):
    q = k = v = input[0]
    batch_size = q.shape[0]
    qdim = q.shape[2]
    qlen = q.shape[1]
    klen = k.shape[1]
    vlen = v.shape[1]
    vdim = qdim
    num_heads = module.num_heads

    syops = qlen * qdim  # Q scaling
    syops += (qlen * qdim * qdim + klen * qdim * qdim + vlen * vdim * vdim)  # projections
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads
    head_syops = (qlen * klen * qk_head_dim + qlen * klen + qlen * klen * v_head_dim)
    syops += num_heads * head_syops
    syops += qlen * vdim * (vdim + 1)
    syops *= batch_size

    module.__syops__[0] += int(syops)
    module.__syops__[2] += int(syops)


# ---------------------------------------------------------------------------
# Default module mapping
# ---------------------------------------------------------------------------

def _build_default_modules_mapping():
    """Build the default mapping of module types to SYOPS hooks."""
    mapping = {
        nn.Conv1d: conv_syops_counter_hook,
        nn.Conv2d: conv_syops_counter_hook,
        QuanConv2d: conv_syops_counter_hook,
        nn.Conv3d: conv_syops_counter_hook,
        nn.ReLU: relu_syops_counter_hook,
        MyQuan: relu_syops_counter_hook,
        nn.PReLU: relu_syops_counter_hook,
        nn.ELU: relu_syops_counter_hook,
        nn.LeakyReLU: relu_syops_counter_hook,
        nn.ReLU6: relu_syops_counter_hook,
        nn.MaxPool1d: pool_syops_counter_hook,
        nn.AvgPool1d: pool_syops_counter_hook,
        nn.AvgPool2d: pool_syops_counter_hook,
        nn.MaxPool2d: pool_syops_counter_hook,
        nn.MaxPool3d: pool_syops_counter_hook,
        nn.AvgPool3d: pool_syops_counter_hook,
        nn.AdaptiveMaxPool1d: pool_syops_counter_hook,
        nn.AdaptiveAvgPool1d: pool_syops_counter_hook,
        nn.AdaptiveMaxPool2d: pool_syops_counter_hook,
        nn.AdaptiveAvgPool2d: pool_syops_counter_hook,
        nn.AdaptiveMaxPool3d: pool_syops_counter_hook,
        nn.AdaptiveAvgPool3d: pool_syops_counter_hook,
        nn.BatchNorm1d: bn_syops_counter_hook,
        nn.BatchNorm2d: bn_syops_counter_hook,
        nn.BatchNorm3d: bn_syops_counter_hook,
        nn.LayerNorm: ln_syops_counter_hook,
        IFNeuron: IF_syops_counter_hook,
        nn.InstanceNorm1d: bn_syops_counter_hook,
        nn.InstanceNorm2d: bn_syops_counter_hook,
        nn.InstanceNorm3d: bn_syops_counter_hook,
        nn.GroupNorm: bn_syops_counter_hook,
        nn.Linear: linear_syops_counter_hook,
        QuanLinear: linear_syops_counter_hook,
        nn.Upsample: empty_syops_counter_hook,
        nn.ConvTranspose1d: conv_syops_counter_hook,
        nn.ConvTranspose2d: conv_syops_counter_hook,
        nn.ConvTranspose3d: conv_syops_counter_hook,
        nn.MultiheadAttention: multihead_attention_counter_hook,
        Attention: multihead_attention_counter_hook,
        QAttention: multihead_attention_counter_hook,
    }
    if hasattr(nn, "GELU"):
        mapping[nn.GELU] = relu_syops_counter_hook
    return mapping


# ---------------------------------------------------------------------------
# OpsCounter
# ---------------------------------------------------------------------------

class OpsCounter:
    """Hook-based synaptic operations counter.

    Registers forward hooks on supported modules to count operations
    during inference. New operator hooks can be registered via
    register_hook().

    Args:
        custom_hooks: Optional dict of {module_type: hook_fn} to add.
    """

    def __init__(self, custom_hooks=None):
        self.modules_mapping = _build_default_modules_mapping()
        if custom_hooks:
            self.modules_mapping.update(custom_hooks)
        self._handles = []
        self._batch_handle = None

    def register_hook(self, module_type, hook_fn):
        """Register a SYOPS counting hook for a new module type."""
        self.modules_mapping[module_type] = hook_fn

    def is_supported(self, module):
        return type(module) in self.modules_mapping

    def attach(self, model):
        """Attach counting hooks to all supported modules in the model.

        Also attaches a batch counter to the top-level module.
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
                module.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
                module.__params__ = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                module.__spkhistc__ = None

    def _add_batch_counter(self, model):
        """Add a forward hook that counts batches."""
        def batch_counter_hook(module, input, output):
            batch_size = 1
            if len(input) > 0:
                batch_size = len(input[0])
            module.__batch_counter__ += batch_size
            module.__times_counter__ += 15  # default timestep count

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
        total = np.array([0.0, 0.0, 0.0, 0.0])
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
                syops[3] /= times_counter
                stats.append((name, module, syops))
        return stats
