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

import torch
import torch.nn as nn

from ..benchmarks.base import BaseEvaluator, EvalResult
from .ops_hookers import OpsCounterFast
from ...snn.operators.neurons import IFNeuron
from ...snn.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm
from ...snn.operators.composites import SConv2d, SLinear
from ...snn.operators.attention import SAttention
from ...snn.operators.decoder_layers import Spiking_RMSNorm, Spiking_SiLU
from ...snn.operators.uniaffine_layers import Spiking_UnifiedClipNorm
from ...snn.operators.uniaffine_attention import Spiking_UniAffineAct
from ...qann.operators.lsq import MyQuan
from ...snn.snnConverter.wrapper import SNNWrapper
from ...config import EnergyConfig
from ...ann.models.base import ModelProfile, DecoderModelProfile, CNNModelProfile
from ...snn.operators.uniaffine_attention import SpikeUniAffineAttention
from ...snn.operators.qwen3_attention import SQwen3Attention
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

@EVALUATOR_REGISTRY.register("fastsimulator")
class FastSimulator(BaseEvaluator):
    """Evaluator for SNN energy consumption.

    Computes energy based on configurable E_mac/E_ac parameters and
    dynamically discovered SAttention modules for SSA energy.

    Args:
        energy_config: EnergyConfig with e_mac, e_ac, nspks_max.
        model_profile: ModelProfile with depth, num_heads, embed_dim, etc.
        ops_counter: Optional OpsCounter instance.
        num_batches: Number of batches to evaluate.
    """

    def __init__(self, hardware_config=None, model_profile=None,
                 ops_counter=None, num_batches=5):
        self.profile = model_profile
        self.ops_counter = ops_counter or OpsCounterFast()
        self.num_batches = num_batches

        # 请帮我load /home/kang_you/UniEval/configs/hardware_param.yaml
        self.hardware_param = hardware_config

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
        result = self._compute_energy_latency_area(model, layer_stats)

        self.ops_counter.detach(model)
        return result
    
    def _compute_energy_for_a_layer(self, stats):
        
        # stats: #SOP， #Spike，#F_out, #Weight, #membrane, #tracer
        e_buffer = self.hardware_param["hardware_modules"]["buffer"]["energy"]["e_buffer"]
        e_pe = self.hardware_param["hardware_modules"]["pe"]["energy"]["e_pe"]
        e_neuron = self.hardware_param["hardware_modules"]["neuron"]["energy"]["e_neuron"]
        e_router = self.hardware_param["hardware_modules"]["router"]["energy"]["e_router"]
        e_wire = self.hardware_param["hardware_modules"]["noc"]["energy"]["e_wire"]
        B_aer = self.hardware_param["hardware_modules"]["noc"]["packet"]["B_AER"]
        N_hop = 1
        
        E_core = stats[1] * (stats[2] * (e_buffer + e_pe + e_neuron) + e_router)
        E_noc = stats[1] * e_wire * B_aer * N_hop
        return E_core + E_noc
    
    def _compute_total_area(self, weight_num, membrane_num, spike_tracer_num):
        b_weight = self.hardware_param["hardware_modules"]["buffer"]["area"]["B_weight"]
        b_mem = self.hardware_param["hardware_modules"]["buffer"]["area"]["B_mem"]
        b_tracer = self.hardware_param["hardware_modules"]["buffer"]["area"]["B_tracer"]
        
        b_sram = self.hardware_param["hardware_modules"]["buffer"]["area"]["B_SRAM"]
        a_sram = self.hardware_param["hardware_modules"]["buffer"]["area"]["A_SRAM"]
        
        N_core = self.hardware_param["system_config"]["N_core"]
        C_topo = self.hardware_param["system_config"]["C_topo"]
        a_wire = self.hardware_param["hardware_modules"]["noc"]["area"]["A_wire"]                
        
        a_pe = self.hardware_param["hardware_modules"]["pe"]["area"]["A_PE"]
        a_neuron = self.hardware_param["hardware_modules"]["neuron"]["area"]["A_neuron"]
        a_router = self.hardware_param["hardware_modules"]["router"]["area"]["A_router"]                
        
        area = (weight_num * b_weight + membrane_num * b_mem + spike_tracer_num * b_tracer)/b_sram*a_sram + \
            N_core * (a_pe + a_neuron + a_router + a_wire * C_topo)
            
        return area

    def _compute_total_latency(self, total_sop_num, total_spike_num):
        latency_pe = self.hardware_param["hardware_modules"]["pe"]["latency"]["L_PE"]
        latency_buffer = self.hardware_param["hardware_modules"]["buffer"]["latency"]["L_buffer"]
        latency_neuron = self.hardware_param["hardware_modules"]["neuron"]["latency"]["L_neuron"]
        b_aer = self.hardware_param["hardware_modules"]["noc"]["packet"]["B_AER"]
        w_noc = self.hardware_param["hardware_modules"]["noc"]["bandwidth"]["W_NOC"]
        w_pe = self.hardware_param["hardware_modules"]["pe"]["compute"]["W_PE"]
        N_core = self.hardware_param["system_config"]["N_core"]
        
        latency = total_sop_num/(w_pe * N_core) * (latency_pe + latency_buffer + latency_neuron) + total_spike_num * b_aer / w_noc
        return latency

    def _compute_energy_latency_area(self, model, layer_stats):
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
        energy_results = {}
        total_weight_num = 0
        total_mem_num = 0
        total_tracer_num = 0
        total_sop_num = 0        
        total_spike_num = 0
        total_energy = 0.0
        # stats: #SOP， #Spike，#F_out, #Weight, #membrane, #tracer
        for name, module, stats in layer_stats:
            energy_layer = self._compute_energy_for_a_layer(stats)
            energy_results[name] = {
                "energy": energy_layer,
            }
            total_spike_num = total_spike_num + stats[1]
            total_weight_num = total_weight_num + stats[3]
            total_mem_num = total_mem_num + stats[4]
            total_tracer_num = total_tracer_num + stats[5]
            total_sop_num = total_sop_num + stats[0]
            total_energy = total_energy + energy_layer
        
        total_area = self._compute_total_area(total_weight_num, total_mem_num, total_tracer_num)
        total_latency = self._compute_total_latency(total_sop_num, total_spike_num)
        
        clock_frequency = self.hardware_param["system_config"]["clock_frequency"]
        
        results = {
            "total_energy": float(total_energy) * 1e-9,
            "total_area": float(total_area) * 1e-6,
            "total_latency": float(total_latency/clock_frequency) * 1000
        }        
        
        return results
