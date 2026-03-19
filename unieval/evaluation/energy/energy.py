"""Energy consumption evaluator with configurable energy model.

Energy model (45nm technology node, Horowitz 2014):
    E_MAC = 4.6 pJ per multiply-accumulate
    E_AC  = 0.9 pJ per accumulate (addition only)

For ternary-output neurons ({-threshold, 0, +threshold}), the threshold
can be fused into weights, so multiplication degenerates to conditional
addition (AC).  Layers whose input firing rate == 100% are treated as
dense (MAC); all others are sparse (AC).

SSA (Spiking Self-Attention) energy follows the SpikeZIP-TF formula
which accounts for Q, K, V firing rates in the temporal matmul
decomposition (multi / multi1 functions).
"""

import numpy as np
import torch
import torch.nn as nn

from ..benchmarks.base import BaseEvaluator, EvalResult
from .ops_counter import OpsCounter
from ...snn.operators.neurons import IFNeuron
from ...snn.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm
from ...snn.operators.composites import SConv2d, SLinear
from ...snn.operators.attention import SAttention
from ...snn.operators.decoder_layers import Spiking_RMSNorm, Spiking_SiLU
from ...snn.operators.uniaffine_layers import Spiking_UnifiedClipNorm
from ...snn.operators.uniaffine_attention import Spiking_UniAffineAct, SpikeUniAffineAttention
from ...snn.operators.qwen3_attention import SQwen3Attention
from ...qann.operators.lsq import MyQuan
from ...snn.snnConverter.wrapper import SNNWrapper
from ...config import EnergyConfig
from ...ann.models.base import ModelProfile, DecoderModelProfile, CNNModelProfile
from ...registry import EVALUATOR_REGISTRY


# ---------------------------------------------------------------------------
# Layer filtering — matches original SpikeZIP-TF flops_counter.py logic
# ---------------------------------------------------------------------------

def _is_energy_relevant(name, module):
    """Check if a module should be included in energy calculation.

    Uses isinstance checks instead of name-based heuristics for reliability.
    Includes: Conv2d, Linear (and their SNN wrappers), LayerNorm, IFNeuron,
    and decoder SNN norms/activations.
    Excludes: MyQuan, composite wrappers (sub-modules already counted).
    """
    if isinstance(module, MyQuan):
        return False
    # Composite SNN wrappers — skip (sub-modules already counted)
    if isinstance(module, (SConv2d, SLinear)):
        return False
    # Core connection layers (inside LLConv2d/LLLinear or standalone)
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return True
    # SNN layer wrappers
    if isinstance(module, (LLConv2d, LLLinear)):
        return True
    # Norms
    if isinstance(module, (nn.LayerNorm, Spiking_LayerNorm)):
        return True
    # Neurons
    if isinstance(module, IFNeuron):
        return True
    # Decoder SNN norms and activations
    if isinstance(module, (Spiking_RMSNorm, Spiking_UnifiedClipNorm,
                           Spiking_SiLU, Spiking_UniAffineAct)):
        return True
    # Fallback: name-based for backward compat with unexpected module types
    if "conv" in name or "linear" in name:
        return True
    return False


def _is_conv_layer(name, module):
    """Conv layers need Tsteps correction on firing rate."""
    if isinstance(module, (nn.Conv2d, LLConv2d)):
        return True
    return "conv" in name


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@EVALUATOR_REGISTRY.register("energy")
class EnergyEvaluator(BaseEvaluator):
    """Evaluator for SNN energy consumption.

    Computes energy based on configurable E_mac/E_ac parameters and
    dynamically discovered SAttention modules for SSA energy.

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
                batch = batch.to(device)
                if batch.is_floating_point():
                    pass  # images etc. already float
                elif batch.dtype in (torch.long, torch.int):
                    pass  # token IDs for decoder models — keep as int
                else:
                    batch = batch.float()
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
        """Compute energy from per-layer SYOPS stats.

        Decision logic (matching original SpikeZIP-TF flops_counter.py):
        1. For each energy-relevant layer, read its firing rate %.
        2. Conv layers: firing_rate *= Tsteps (the hook accumulates across
           timesteps, so the raw rate needs rescaling).
        3. If firing_rate ≈ 100% → dense input → count as MAC.
        4. Otherwise → sparse/spike input → count as AC.

        AC means addition-only because the ternary spike output {-1,0,+1}
        turns multiplication into conditional addition (threshold fused
        into weights).
        """
        e_mac = self.energy_config.e_mac
        e_ac = self.energy_config.e_ac
        time_steps = self.profile.time_steps if self.profile else self.ops_counter.time_step

        total_mac_ops = 0.0
        total_ac_ops = 0.0
        layer_details = []

        for name, module, syops in layer_stats:
            # NOTE:
            # OpsCounter.get_per_layer_stats already normalizes syops[3] by
            # times_counter (= time_step per forward). Therefore syops[3] is
            # already an average firing-rate percentage per timestep.
            # Do NOT multiply by time_steps again; otherwise firing_rate can
            # exceed 100% and firing_rate (0~1) can exceed 1.0.
            firing_rate_pct = syops[3]

            if abs(firing_rate_pct - 100) < 1e-4:  # fr ≈ 100% → MAC
                total_mac_ops += syops[2]
            else:                                    # sparse → AC
                total_ac_ops += syops[1]

            layer_details.append({
                "name": name,
                "total_ops": syops[0],
                "ac_ops": syops[1],
                "mac_ops": syops[2],
                # firing_rate is a ratio in [0, 1] (not percentage)
                "firing_rate": firing_rate_pct / 100.0,
            })

        # SSA energy from dynamically discovered SAttention / SpikeUniAffineAttention modules
        inner = model.module if hasattr(model, "module") else model
        times_counter = max(model.__times_counter__, 1) if hasattr(model, "__times_counter__") else 1
        if isinstance(inner, SNNWrapper) and self.profile is not None:
            if isinstance(self.profile, DecoderModelProfile):
                ssa_ac, ssa_qkv_fr = self._compute_decoder_ssa_energy(
                    inner, self.profile, times_counter
                )
            elif isinstance(self.profile, CNNModelProfile):
                ssa_ac = 0.0
            else:
                ssa_ac, ssa_qkv_fr = self._compute_ssa_energy(
                    inner, self.profile, times_counter
                )
            total_ac_ops += ssa_ac
        else:
            ssa_qkv_fr = []

        # Convert raw op counts to Giga-ops, then to energy (mJ)
        total_mac_ops_g = total_mac_ops / 1e9
        total_ac_ops_g = total_ac_ops / 1e9
        e_mac_total = total_mac_ops_g * e_mac  # mJ
        e_ac_total = total_ac_ops_g * e_ac     # mJ
        e_total = e_mac_total + e_ac_total

        if isinstance(self.profile, CNNModelProfile):
            return EvalResult(
                metrics={
                    "energy_mJ": e_total,
                    "e_mac_mJ": e_mac_total,
                    "e_ac_mJ": e_ac_total,
                    "mac_ops_G": total_mac_ops_g,
                    "ac_ops_G": total_ac_ops_g,
                },
                details={
                    "layers": layer_details,
                },
            )
        else:
            return EvalResult(
                metrics={
                    "energy_mJ": e_total,
                    "e_mac_mJ": e_mac_total,
                    "e_ac_mJ": e_ac_total,
                    "mac_ops_G": total_mac_ops_g,
                    "ac_ops_G": total_ac_ops_g,
                },
                details={
                    "layers": layer_details,
                    "ssa_qkv_firing_rates": ssa_qkv_fr,
                },
            )

    def _compute_ssa_energy(self, wrapper, profile, times_counter):
        """Compute SSA (Spiking Self-Attention) energy.

        Fixes a normalization bug inherited from SpikeZIP-TF: the original
        code reads raw accumulated __syops__[3] without dividing by
        times_counter, making SSA energy dependent on num_batches.

        The temporal matmul decomposition (multi/multi1 functions) yields:
          Q*K^T:  base * (q_fr + k_fr + min(q_fr, k_fr))
          Attn*V: base * (v_fr + min(q_fr, k_fr, v_fr) + min(q_fr, k_fr))

        where base = Tsteps * Nheads * patchSize^2 * (embSize_per_head)^2.

        Returns:
            (total_ssa_ac_ops, qkv_firing_rates_per_block)
        """
        depth = profile.depth
        num_heads = profile.num_heads
        embed_dim = profile.embed_dim
        patch_size = profile.patch_size
        time_steps = profile.time_steps

        embed_per_head = embed_dim // num_heads
        ssa_base = (time_steps * num_heads
                    * (patch_size ** 2)
                    * (embed_per_head ** 2))

        # Find all spiking attention modules
        model = wrapper.model
        sa_modules = []
        for _, module in model.named_modules():
            if isinstance(module, (SAttention, SpikeUniAffineAttention, SQwen3Attention)):
                sa_modules.append(module)

        total_ssa_ac = 0.0
        qkv_fr_list = []
        for sa in sa_modules[:depth]:
            # Read firing rates from IF neurons' __syops__ counters,
            # normalized by times_counter (fixes SpikeZIP-TF bug)
            q_fr = self._get_neuron_fr(sa.q_IF, times_counter)
            k_fr = self._get_neuron_fr(sa.k_IF, times_counter)
            v_fr = self._get_neuron_fr(sa.v_IF, times_counter)
            qkv_fr_list.append([q_fr, k_fr, v_fr])

            # Q*K^T term
            t_ac = ssa_base * (q_fr + k_fr + min(q_fr, k_fr))
            # Attn*V term
            t_ac += ssa_base * (v_fr + min(q_fr, k_fr, v_fr) + min(q_fr, k_fr))
            total_ssa_ac += t_ac

        return total_ssa_ac, qkv_fr_list

    def _compute_decoder_ssa_energy(self, wrapper, profile, times_counter):
        """Compute SSA energy for decoder models with GQA.

        Adapts the ViT SSA formula for GPT:
        - Uses seq_len instead of patch_size^2
        - GQA-aware: K/V have fewer heads, expanded for matmul
        - Separate accounting for Q heads vs K/V heads

        The temporal matmul decomposition is the same:
          Q*K^T:  base * (q_fr + k_fr + min(q_fr, k_fr))
          Attn*V: base * (v_fr + min(q_fr, k_fr, v_fr) + min(q_fr, k_fr))

        But base = Tsteps * num_heads * seq_len * head_dim^2
        (num_heads for Q; K/V are expanded via GQA repeat)

        Returns:
            (total_ssa_ac_ops, qkv_firing_rates_per_block)
        """
        depth = profile.depth
        num_heads = profile.num_heads
        head_dim = profile.head_dim
        seq_len = profile.seq_len
        time_steps = profile.time_steps

        # SSA base: uses seq_len instead of patch_size^2
        ssa_base = time_steps * num_heads * seq_len * (head_dim ** 2)

        model = wrapper.model
        sa_modules = []
        for _, module in model.named_modules():
            if isinstance(module, (SQwen3Attention, SpikeUniAffineAttention)):
                sa_modules.append(module)

        total_ssa_ac = 0.0
        qkv_fr_list = []
        for sa in sa_modules[:depth]:
            q_fr = self._get_neuron_fr(sa.q_IF, times_counter)
            k_fr = self._get_neuron_fr(sa.k_IF, times_counter)
            v_fr = self._get_neuron_fr(sa.v_IF, times_counter)
            qkv_fr_list.append([q_fr, k_fr, v_fr])

            # Q*K^T term (GQA: K is expanded, but firing rate is same)
            t_ac = ssa_base * (q_fr + k_fr + min(q_fr, k_fr))
            # Attn*V term
            t_ac += ssa_base * (v_fr + min(q_fr, k_fr, v_fr) + min(q_fr, k_fr))
            total_ssa_ac += t_ac

        return total_ssa_ac, qkv_fr_list

    @staticmethod
    def _get_neuron_fr(neuron, times_counter=1):
        """Read normalized firing rate from a neuron's __syops__ counter.

        Divides by times_counter to get per-timestep average firing rate,
        fixing the normalization bug inherited from SpikeZIP-TF.

        Falls back to 0.0 if OpsCounter hooks were not attached.
        """
        if hasattr(neuron, "__syops__"):
            return neuron.__syops__[3] / (100.0 * times_counter)
        return 0.0
