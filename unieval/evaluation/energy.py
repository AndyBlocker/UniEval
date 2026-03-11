"""Energy consumption evaluator with configurable energy model."""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseEvaluator, EvalResult
from .ops_counter import OpsCounter
from ..operators.neurons import IFNeuron
from ..operators.attention import SAttention
from ..quantization.lsq import MyQuan
from ..conversion.wrapper import SNNWrapper
from ..config import EnergyConfig
from ..models.base import ModelProfile
from ..registry import EVALUATOR_REGISTRY


def _find_sattention_modules(model):
    """Dynamically discover all SAttention modules in the model tree."""
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, SAttention):
            modules.append((name, module))
    return modules


@EVALUATOR_REGISTRY.register("energy")
class EnergyEvaluator(BaseEvaluator):
    """Evaluator for SNN energy consumption.

    Computes energy based on configurable E_mac/E_ac parameters and
    dynamically discovered SAttention modules.

    Args:
        energy_config: EnergyConfig with e_mac, e_ac, nspks_max.
        model_profile: ModelProfile with depth, num_heads, embed_dim, etc.
        ops_counter: Optional OpsCounter instance.
        num_batches: Number of batches to evaluate.
    """

    def __init__(self, energy_config=None, model_profile=None,
                 ops_counter=None, num_batches=5):
        self.energy_config = energy_config or EnergyConfig()
        self.profile = model_profile
        self.ops_counter = ops_counter or OpsCounter()
        self.num_batches = num_batches

    def evaluate(self, model, dataloader, **kwargs) -> EvalResult:
        """Run energy evaluation.

        Args:
            model: The SNN model (possibly wrapped in DataParallel).
            dataloader: Data to evaluate on.

        Returns:
            EvalResult with energy metrics.
        """
        self.ops_counter.attach(model)
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for i, (batch, target) in enumerate(dataloader):
                if i >= self.num_batches:
                    break
                batch = batch.float().to(device)
                output = model(batch)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # Reset SNN state
                inner = model.module if hasattr(model, "module") else model
                if isinstance(inner, SNNWrapper):
                    inner.reset()

        # Compute per-layer stats
        layer_stats = self.ops_counter.get_per_layer_stats(model)
        energy_result = self._compute_energy(model, layer_stats)

        self.ops_counter.detach(model)
        return energy_result

    def _compute_energy(self, model, layer_stats):
        """Compute energy from per-layer SYOPS stats."""
        e_mac = self.energy_config.e_mac
        e_ac = self.energy_config.e_ac

        total_mac_ops = 0.0
        total_ac_ops = 0.0
        layer_details = []

        for name, module, syops in layer_stats:
            if isinstance(module, MyQuan):
                continue

            if abs(syops[3] - 100) < 1e-4:  # firing rate ~100% -> MAC
                total_mac_ops += syops[2]
            else:
                total_ac_ops += syops[1]

            layer_details.append({
                "name": name,
                "total_ops": syops[0],
                "ac_ops": syops[1],
                "mac_ops": syops[2],
                "firing_rate": syops[3] / 100.0,
            })

        # SSA energy from dynamically discovered SAttention modules
        inner = model.module if hasattr(model, "module") else model
        if isinstance(inner, SNNWrapper) and self.profile is not None:
            ssa_ac = self._compute_ssa_energy(inner, self.profile)
            total_ac_ops += ssa_ac

        # Convert to energy (pJ -> mJ)
        total_mac_ops_g = total_mac_ops / 1e9
        total_ac_ops_g = total_ac_ops / 1e9
        e_mac_total = total_mac_ops_g * e_mac  # mJ
        e_ac_total = total_ac_ops_g * e_ac     # mJ
        e_total = e_mac_total + e_ac_total

        return EvalResult(
            metrics={
                "energy_mJ": e_total,
                "e_mac_mJ": e_mac_total,
                "e_ac_mJ": e_ac_total,
                "mac_ops_G": total_mac_ops_g,
                "ac_ops_G": total_ac_ops_g,
            },
            details={"layers": layer_details},
        )

    def _compute_ssa_energy(self, wrapper, profile):
        """Compute SSA (Spiking Self-Attention) energy dynamically.

        Instead of using eval() with hardcoded paths, discovers SAttention
        modules and reads their firing rates directly.
        """
        depth = profile.depth
        num_heads = profile.num_heads
        embed_dim = profile.embed_dim
        patch_size = profile.patch_size
        time_steps = profile.time_steps

        embed_per_head = embed_dim // num_heads
        ssa_base = time_steps * num_heads * (patch_size ** 2) * (embed_per_head ** 2)

        # Find all SAttention modules
        model = wrapper.model if hasattr(wrapper, "model") else wrapper
        sa_modules = _find_sattention_modules(model)

        total_ssa_ac = 0.0
        for _, sa_module in sa_modules[:depth]:
            # Read firing rates from IF neurons
            q_fr = sa_module.q_IF.__syops__[3] / 100.0 if hasattr(sa_module.q_IF, "__syops__") else 0.5
            k_fr = sa_module.k_IF.__syops__[3] / 100.0 if hasattr(sa_module.k_IF, "__syops__") else 0.5
            v_fr = sa_module.v_IF.__syops__[3] / 100.0 if hasattr(sa_module.v_IF, "__syops__") else 0.5

            # Q*K^T and Attn*V operations with sparsity
            t_ac = ssa_base * (q_fr + k_fr + min(q_fr, k_fr))
            t_ac += ssa_base * (v_fr + min(q_fr, k_fr, v_fr) + min(q_fr, k_fr))
            total_ssa_ac += t_ac

        return total_ssa_ac
