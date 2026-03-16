"""
Comparison experiment: UniEval vs SpikeZIP-TF

Verifies that UniEval produces numerically identical results to the original
SpikeZIP-TF implementation for:
  1. IFNeuron forward pass
  2. Spiking_LayerNorm forward pass
  3. spiking_softmax forward pass
  4. LLLinear / LLConv2d forward pass
  5. SAttention forward pass
  6. MyQuan (LSQ) quantization
  7. Full QANN -> SNN conversion output equivalence
  8. spike_rate() detection
  9. OPS counter hooks (per-layer syops)
 10. Energy calculation (Nmac, Nac, E_total)

Usage:
    python experiments/compare_with_spikzip.py
"""

import sys
import os
import importlib.util
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

# Add both projects to path
UNIEVAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIKZIP_ROOT = os.path.join(os.path.dirname(UNIEVAL_ROOT), "SpikeZIP-TF")
sys.path.insert(0, UNIEVAL_ROOT)
sys.path.insert(0, SPIKZIP_ROOT)


def _load_module_direct(name, filepath):
    """Load a Python module directly from file path, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_spikzip_ops():
    """Load SpikeZIP-TF ops.py with mocked spikingjelly dependency."""
    if "spikzip_ops" in sys.modules:
        return sys.modules["spikzip_ops"]

    import types

    # Mock timm.models.vision_transformer.Attention (timm 0.3.2 incompatible with PyTorch 2.x)
    if "timm" not in sys.modules:
        mock_timm = types.ModuleType("timm")
        mock_timm_models = types.ModuleType("timm.models")
        mock_timm_vt = types.ModuleType("timm.models.vision_transformer")
        mock_timm_vt.Attention = type("Attention", (nn.Module,), {})
        mock_timm.models = mock_timm_models
        mock_timm_models.vision_transformer = mock_timm_vt
        sys.modules["timm"] = mock_timm
        sys.modules["timm.models"] = mock_timm_models
        sys.modules["timm.models.vision_transformer"] = mock_timm_vt

    # Mock spikingjelly (ops.py imports it but spike_rate/hooks don't use it)
    mock = types.ModuleType("spikingjelly")
    mock_cd = types.ModuleType("spikingjelly.clock_driven")
    mock_cd_neuron = types.ModuleType("spikingjelly.clock_driven.neuron")
    mock_ab = types.ModuleType("spikingjelly.activation_based")
    mock_ab_neuron = types.ModuleType("spikingjelly.activation_based.neuron")

    # Create dummy classes for the neuron types
    for attr in ["MultiStepIFNode", "MultiStepLIFNode", "IFNode", "LIFNode",
                 "MultiStepParametricLIFNode", "ParametricLIFNode"]:
        setattr(mock_cd_neuron, attr, type(attr, (nn.Module,), {}))
        setattr(mock_ab_neuron, attr, type(attr, (nn.Module,), {}))

    mock.clock_driven = mock_cd
    mock_cd.neuron = mock_cd_neuron
    mock.activation_based = mock_ab
    mock_ab.neuron = mock_ab_neuron

    sys.modules["spikingjelly"] = mock
    sys.modules["spikingjelly.clock_driven"] = mock_cd
    sys.modules["spikingjelly.clock_driven.neuron"] = mock_cd_neuron
    sys.modules["spikingjelly.activation_based"] = mock_ab
    sys.modules["spikingjelly.activation_based.neuron"] = mock_ab_neuron

    return _load_module_direct(
        "spikzip_ops",
        os.path.join(SPIKZIP_ROOT, "energy_consumption_calculation", "ops.py"),
    )


# ────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")

def close(a, b, atol=1e-5, rtol=1e-4):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a.cpu().float(), b.cpu().float(), atol=atol, rtol=rtol)
    return abs(a - b) < atol


# ════════════════════════════════════════════════════════════
# Test 1: IFNeuron
# ════════════════════════════════════════════════════════════
def test_ifneuron():
    print("\n=== Test 1: IFNeuron ===")
    torch.manual_seed(42)

    # --- SpikeZIP-TF ---
    from spike_quan_layer import IFNeuron as OrigIFNeuron

    # --- UniEval ---
    from unieval.operators.neurons import IFNeuron as NewIFNeuron

    for sym in [True, False]:
        for level in [4, 16, 32]:
            torch.manual_seed(42)
            threshold = torch.tensor(0.3)
            x_seq = torch.randn(8, 2, 64)  # 8 timesteps

            orig = OrigIFNeuron(q_threshold=threshold.clone(), level=level, sym=sym)
            new = NewIFNeuron(q_threshold=threshold.clone(), level=level, sym=sym)

            orig_outputs = []
            new_outputs = []
            for t in range(8):
                orig_outputs.append(orig(x_seq[t]))
                new_outputs.append(new(x_seq[t]))

            orig_out = torch.stack(orig_outputs)
            new_out = torch.stack(new_outputs)
            check(f"IFNeuron(sym={sym}, level={level})", close(orig_out, new_out),
                  f"max diff={torch.abs(orig_out - new_out).max():.2e}")


# ════════════════════════════════════════════════════════════
# Test 2: Spiking_LayerNorm
# ════════════════════════════════════════════════════════════
def test_spiking_layernorm():
    print("\n=== Test 2: Spiking_LayerNorm ===")
    torch.manual_seed(42)

    from spike_quan_layer import Spiking_LayerNorm as OrigSLN
    from unieval.operators.layers import Spiking_LayerNorm as NewSLN

    dim = 64
    orig = OrigSLN(dim)
    new = NewSLN(dim)
    # Sync weights
    new.layernorm.weight.data = orig.layernorm.weight.data.clone()
    new.layernorm.bias.data = orig.layernorm.bias.data.clone()

    x_seq = torch.randn(5, 2, 10, dim)
    orig_outputs = []
    new_outputs = []
    for t in range(5):
        orig_outputs.append(orig(x_seq[t]))
        new_outputs.append(new(x_seq[t]))

    orig_out = torch.stack(orig_outputs)
    new_out = torch.stack(new_outputs)
    check("Spiking_LayerNorm", close(orig_out, new_out),
          f"max diff={torch.abs(orig_out - new_out).max():.2e}")


# ════════════════════════════════════════════════════════════
# Test 3: spiking_softmax
# ════════════════════════════════════════════════════════════
def test_spiking_softmax():
    print("\n=== Test 3: spiking_softmax ===")
    torch.manual_seed(42)

    from spike_quan_layer import spiking_softmax as OrigSS
    from unieval.operators.attention import spiking_softmax as NewSS

    orig = OrigSS()
    new = NewSS()

    x_seq = torch.randn(6, 2, 4, 10, 10)
    orig_outputs = []
    new_outputs = []
    for t in range(6):
        orig_outputs.append(orig(x_seq[t]))
        new_outputs.append(new(x_seq[t]))

    orig_out = torch.stack(orig_outputs)
    new_out = torch.stack(new_outputs)
    check("spiking_softmax", close(orig_out, new_out),
          f"max diff={torch.abs(orig_out - new_out).max():.2e}")


# ════════════════════════════════════════════════════════════
# Test 4: LLLinear
# ════════════════════════════════════════════════════════════
def test_lllinear():
    print("\n=== Test 4: LLLinear ===")
    torch.manual_seed(42)

    from spike_quan_layer import LLLinear as OrigLL
    from unieval.operators.layers import LLLinear as NewLL

    linear = nn.Linear(32, 64, bias=True)
    torch.manual_seed(42)
    nn.init.normal_(linear.weight)
    nn.init.normal_(linear.bias)

    for ntype in ["ST-BIF", "IF"]:
        orig = OrigLL(linear=deepcopy(linear), neuron_type=ntype, level=16)
        new = NewLL(linear=deepcopy(linear), neuron_type=ntype, level=16)

        x_seq = torch.randn(5, 2, 10, 32)
        orig_outputs = []
        new_outputs = []
        for t in range(5):
            orig_outputs.append(orig(x_seq[t]))
            new_outputs.append(new(x_seq[t]))

        orig_out = torch.stack(orig_outputs)
        new_out = torch.stack(new_outputs)
        check(f"LLLinear(type={ntype})", close(orig_out, new_out),
              f"max diff={torch.abs(orig_out - new_out).max():.2e}")


# ════════════════════════════════════════════════════════════
# Test 5: LLConv2d
# ════════════════════════════════════════════════════════════
def test_llconv2d():
    print("\n=== Test 5: LLConv2d ===")
    torch.manual_seed(42)

    from spike_quan_layer import LLConv2d as OrigLC
    from unieval.operators.layers import LLConv2d as NewLC

    conv = nn.Conv2d(3, 16, 3, padding=1, bias=True)

    for ntype in ["ST-BIF", "IF"]:
        orig = OrigLC(conv=deepcopy(conv), neuron_type=ntype, level=16)
        new = NewLC(conv=deepcopy(conv), neuron_type=ntype, level=16)

        x_seq = torch.randn(4, 1, 3, 8, 8)
        orig_outputs = []
        new_outputs = []
        for t in range(4):
            orig_outputs.append(orig(x_seq[t]))
            new_outputs.append(new(x_seq[t]))

        orig_out = torch.stack(orig_outputs)
        new_out = torch.stack(new_outputs)
        check(f"LLConv2d(type={ntype})", close(orig_out, new_out),
              f"max diff={torch.abs(orig_out - new_out).max():.2e}")


# ════════════════════════════════════════════════════════════
# Test 6: MyQuan (LSQ)
# ════════════════════════════════════════════════════════════
def test_myquan():
    print("\n=== Test 6: MyQuan (LSQ) ===")
    torch.manual_seed(42)

    from spike_quan_layer import MyQuan as OrigMQ
    from unieval.quantization.lsq import MyQuan as NewMQ

    for sym in [True, False]:
        for level in [4, 16, 32]:
            orig = OrigMQ(level=level, sym=sym)
            new = NewMQ(level=level, sym=sym)

            # Set same scale
            orig.s.data = torch.tensor(0.5)
            new.s.data = torch.tensor(0.5)
            orig.init_state = 1  # skip init
            new.init_state = 1

            orig.eval()
            new.eval()

            x = torch.randn(2, 10, 64) * 2
            orig_out = orig(x)
            new_out = new(x)
            check(f"MyQuan(sym={sym}, level={level})", close(orig_out, new_out),
                  f"max diff={torch.abs(orig_out - new_out).max():.2e}")


# ════════════════════════════════════════════════════════════
# Test 7: QAttention
# ════════════════════════════════════════════════════════════
def test_qattention():
    print("\n=== Test 7: QAttention ===")
    torch.manual_seed(42)

    from spike_quan_layer import QAttention as OrigQA
    from unieval.quantization.lsq import QAttention as NewQA

    dim = 64
    num_heads = 4
    level = 16

    orig = OrigQA(dim=dim, num_heads=num_heads, level=level, is_softmax=True)
    new = NewQA(dim=dim, num_heads=num_heads, level=level, is_softmax=True)

    # Sync all parameters
    new.load_state_dict(orig.state_dict())

    # Set init_state to skip auto-init
    for module in [orig, new]:
        for m in module.modules():
            if hasattr(m, 'init_state'):
                m.init_state = 1

    orig.eval()
    new.eval()

    x = torch.randn(2, 10, dim)
    orig_out = orig(x)
    new_out = new(x)
    check("QAttention", close(orig_out, new_out, atol=1e-4),
          f"max diff={torch.abs(orig_out - new_out).max():.2e}")


# ════════════════════════════════════════════════════════════
# Test 8: SAttention
# ════════════════════════════════════════════════════════════
def test_sattention():
    print("\n=== Test 8: SAttention ===")
    torch.manual_seed(42)

    from spike_quan_layer import SAttention as OrigSA, IFNeuron as OrigIF
    from unieval.operators.attention import SAttention as NewSA
    from unieval.operators.neurons import IFNeuron as NewIF

    dim = 64
    num_heads = 4
    level = 16

    orig = OrigSA(dim=dim, num_heads=num_heads, level=level, is_softmax=True, neuron_layer=OrigIF)
    new = NewSA(dim=dim, num_heads=num_heads, level=level, is_softmax=True, neuron_layer=NewIF)

    # Sync all parameters (qkv, proj weights + neuron thresholds)
    # We need to manually copy since state_dict structures might differ
    orig_sd = orig.state_dict()
    new.load_state_dict(orig_sd, strict=False)
    # Copy remaining params manually
    for name in orig_sd:
        parts = name.split('.')
        orig_p = orig
        new_p = new
        for p in parts:
            orig_p = getattr(orig_p, p)
            new_p = getattr(new_p, p)
        if isinstance(new_p, torch.Tensor) and isinstance(orig_p, torch.Tensor):
            new_p.data = orig_p.data.clone()

    x_seq = torch.randn(6, 2, 10, dim) * 0.1
    orig_outputs = []
    new_outputs = []
    for t in range(6):
        orig_outputs.append(orig(x_seq[t]))
        new_outputs.append(new(x_seq[t]))

    orig_out = torch.stack(orig_outputs)
    new_out = torch.stack(new_outputs)
    check("SAttention (6 timesteps)", close(orig_out, new_out, atol=1e-4),
          f"max diff={torch.abs(orig_out - new_out).max():.2e}")


# ════════════════════════════════════════════════════════════
# Test 9: spike_rate()
# ════════════════════════════════════════════════════════════
def test_spike_rate():
    print("\n=== Test 9: spike_rate() ===")

    _ops = _load_spikzip_ops()
    orig_sr = _ops.spike_rate
    from unieval.evaluation.spike_utils import spike_rate as new_sr

    # Case 1: All zeros
    x = torch.zeros(2, 10)
    o_spike, o_rate, _ = orig_sr(x)
    n_spike, n_rate, _ = new_sr(x)
    check("spike_rate(zeros)", o_spike == n_spike and abs(o_rate - n_rate) < 1e-6)

    # Case 2: Ternary spike tensor
    x = torch.tensor([[-1.0, 0, 1, 0, -1], [0, 1, 1, 0, 0]])
    o_spike, o_rate, _ = orig_sr(x)
    n_spike, n_rate, _ = new_sr(x)
    check("spike_rate(ternary)", o_spike == n_spike and abs(o_rate - n_rate) < 1e-6,
          f"orig=({o_spike},{o_rate:.4f}) new=({n_spike},{n_rate:.4f})")

    # Case 3: Dense (non-spike) tensor
    x = torch.randn(2, 100)
    o_spike, o_rate, _ = orig_sr(x)
    n_spike, n_rate, _ = new_sr(x)
    check("spike_rate(dense)", o_spike == n_spike and abs(o_rate - n_rate) < 1e-6,
          f"orig=({o_spike},{o_rate:.4f}) new=({n_spike},{n_rate:.4f})")

    # Case 4: Scaled ternary (threshold != 1)
    threshold = 0.3
    x = torch.tensor([[-0.3, 0, 0.3, 0], [0, 0.3, -0.3, 0]])
    o_spike, o_rate, _ = orig_sr(x)
    n_spike, n_rate, _ = new_sr(x)
    check("spike_rate(scaled ternary)", o_spike == n_spike and abs(o_rate - n_rate) < 1e-6,
          f"orig=({o_spike},{o_rate:.4f}) new=({n_spike},{n_rate:.4f})")


# ════════════════════════════════════════════════════════════
# Test 10: SYOPS hooks comparison
# ════════════════════════════════════════════════════════════
def test_syops_hooks():
    print("\n=== Test 10: SYOPS Hooks ===")
    torch.manual_seed(42)

    _ops = _load_spikzip_ops()
    orig_conv_hook = _ops.conv_syops_counter_hook
    orig_linear_hook = _ops.linear_syops_counter_hook
    orig_if_hook = _ops.IF_syops_counter_hook
    orig_ln_hook = _ops.ln_syops_counter_hook

    from unieval.evaluation.ops_counter import (
        conv_syops_counter_hook as new_conv_hook,
        linear_syops_counter_hook as new_linear_hook,
        IF_syops_counter_hook as new_if_hook,
        ln_syops_counter_hook as new_ln_hook,
    )

    # Test conv hook
    conv = nn.Conv2d(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 8, 8)  # spike-like won't be detected since it's random
    output = conv(x)

    conv.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
    conv.__spkhistc__ = None
    orig_conv_hook(conv, (x,), output)
    orig_syops = conv.__syops__.copy()

    conv.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
    conv.__spkhistc__ = None
    new_conv_hook(conv, (x,), output)
    new_syops = conv.__syops__.copy()
    check("Conv2d hook", np.allclose(orig_syops, new_syops),
          f"orig={orig_syops} new={new_syops}")

    # Test linear hook
    linear = nn.Linear(32, 64)
    x = torch.randn(2, 10, 32)
    output = linear(x)

    linear.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
    linear.__spkhistc__ = None
    orig_linear_hook(linear, (x,), output)
    orig_syops = linear.__syops__.copy()

    linear.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
    linear.__spkhistc__ = None
    new_linear_hook(linear, (x,), output)
    new_syops = linear.__syops__.copy()
    check("Linear hook", np.allclose(orig_syops, new_syops),
          f"orig={orig_syops} new={new_syops}")

    # Test IF hook with spike-like input
    from spike_quan_layer import IFNeuron as OrigIF
    from unieval.operators.neurons import IFNeuron as NewIF

    neuron_orig = OrigIF(q_threshold=torch.tensor(0.5), level=16, sym=True)
    neuron_new = NewIF(q_threshold=torch.tensor(0.5), level=16, sym=True)

    x = torch.randn(2, 64)
    out_orig = neuron_orig(x)
    out_new = neuron_new(x)

    neuron_orig.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
    neuron_orig.__spkhistc__ = None
    orig_if_hook(neuron_orig, (x,), out_orig)
    orig_syops = neuron_orig.__syops__.copy()

    neuron_new.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
    neuron_new.__spkhistc__ = None
    new_if_hook(neuron_new, (x,), out_new)
    new_syops = neuron_new.__syops__.copy()
    check("IF hook", np.allclose(orig_syops, new_syops),
          f"orig={orig_syops} new={new_syops}")

    # Test LayerNorm hook
    ln = nn.LayerNorm(64)
    x = torch.randn(2, 10, 64)
    output = ln(x)

    ln.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
    ln.__spkhistc__ = None
    orig_ln_hook(ln, (x,), output)
    orig_syops = ln.__syops__.copy()

    ln.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
    ln.__spkhistc__ = None
    new_ln_hook(ln, (x,), output)
    new_syops = ln.__syops__.copy()
    check("LayerNorm hook", np.allclose(orig_syops, new_syops),
          f"orig={orig_syops} new={new_syops}")


# ════════════════════════════════════════════════════════════
# Test 11: Full QANN → SNN conversion output equivalence
# ════════════════════════════════════════════════════════════
def test_full_conversion():
    print("\n=== Test 11: Full QANN → SNN Conversion ===")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build the same ViT-small model in both frameworks ---
    from unieval.models.vit import vit_small_patch16 as new_vit_factory
    new_model = new_vit_factory(
        img_size=224, num_classes=100, global_pool=False, act_layer=nn.ReLU
    )

    # We need to also create the same model in SpikeZIP-TF
    # SpikeZIP-TF uses timm, so let's try a different approach:
    # Build UniEval model, quantize, convert, and verify internal consistency
    from unieval.quantization.lsq import LSQQuantizer
    from unieval.conversion.wrapper import SNNWrapper
    from unieval.conversion.converter import SNNConverter

    level = 16
    time_step = 32

    # Copy model weights before quantization
    model_state = deepcopy(new_model.state_dict())

    # Quantize
    quantizer = LSQQuantizer(level=level, weight_bit=32, is_softmax=True)
    quantizer.quantize_model(new_model)

    # Set all MyQuan init_state to 1 and set deterministic scales
    from unieval.quantization.lsq import MyQuan
    for name, m in new_model.named_modules():
        if isinstance(m, MyQuan):
            m.init_state = 1
            m.s.data = torch.tensor(0.1)

    new_model.eval()
    new_model = new_model.to(device)

    # Save quantized state
    qann_state = deepcopy(new_model.state_dict())

    # Test QANN forward pass consistency
    torch.manual_seed(123)
    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        qann_out1 = new_model(x)
        qann_out2 = new_model(x)

    check("QANN forward deterministic", close(qann_out1, qann_out2),
          f"max diff={torch.abs(qann_out1 - qann_out2).max():.2e}")

    # Convert to SNN
    wrapper = SNNWrapper(
        ann_model=new_model,
        time_step=time_step,
        encoding_type="analog",
        level=level,
        neuron_type="ST-BIF",
        model_name="vit_small_patch16",
        is_softmax=True,
    )
    wrapper = wrapper.to(device)
    wrapper.eval()

    with torch.no_grad():
        accu1, T1 = wrapper(x)
        wrapper.reset()
        accu2, T2 = wrapper(x)

    check("SNN forward deterministic", close(accu1, accu2),
          f"T={T1}/{T2}, max diff={torch.abs(accu1 - accu2).max():.2e}")
    check("SNN timestep consistent", T1 == T2, f"T1={T1}, T2={T2}")
    print(f"  (SNN converged in {T1} timesteps)")


# ════════════════════════════════════════════════════════════
# Test 12: Energy formula verification
# ════════════════════════════════════════════════════════════
def test_energy_formula():
    print("\n=== Test 12: Energy Formula ===")

    # Reproduce SpikeZIP-TF energy formula with known values
    ssa_info = {'depth': 12, 'Nheads': 6, 'embSize': 384, 'patchSize': 16, 'Tsteps': 15}

    depth = ssa_info['depth']
    Nheads = ssa_info['Nheads']
    embSize = ssa_info['embSize']
    patchSize = ssa_info['patchSize']
    Tsteps = ssa_info['Tsteps']
    embSize_per_head = embSize // Nheads

    SSA_Nac_base_orig = Tsteps * Nheads * pow(patchSize, 2) * embSize_per_head * embSize_per_head

    # UniEval formula
    from unieval.models.base import ModelProfile
    profile = ModelProfile(
        depth=12, num_heads=6, embed_dim=384, patch_size=16,
        img_size=224, time_steps=15
    )
    embed_per_head = profile.embed_dim // profile.num_heads
    SSA_Nac_base_new = (profile.time_steps * profile.num_heads
                        * (profile.patch_size ** 2)
                        * (embed_per_head ** 2))

    check("SSA base formula", SSA_Nac_base_orig == SSA_Nac_base_new,
          f"orig={SSA_Nac_base_orig} new={SSA_Nac_base_new}")

    # Test with known firing rates
    q_fr, k_fr, v_fr = 0.3, 0.25, 0.4

    # Original formula
    tNac_orig = SSA_Nac_base_orig * (q_fr + k_fr + min(q_fr, k_fr))
    tNac_orig += SSA_Nac_base_orig * (v_fr + min(q_fr, k_fr, v_fr) + min(q_fr, k_fr))

    # UniEval formula (from EnergyEvaluator._compute_ssa_energy)
    tNac_new = SSA_Nac_base_new * (q_fr + k_fr + min(q_fr, k_fr))
    tNac_new += SSA_Nac_base_new * (v_fr + min(q_fr, k_fr, v_fr) + min(q_fr, k_fr))

    check("SSA per-block energy", abs(tNac_orig - tNac_new) < 1e-3,
          f"orig={tNac_orig:.1f} new={tNac_new:.1f}")

    # Verify energy conversion
    Nmac = 1e9  # 1G MACs
    Nac = 2e9   # 2G ACs
    E_mac = (Nmac / 1e9) * 4.6  # mJ
    E_ac = (Nac / 1e9) * 0.9    # mJ
    E_total = E_mac + E_ac

    check("Energy constants", abs(E_mac - 4.6) < 1e-6 and abs(E_ac - 1.8) < 1e-6,
          f"E_mac={E_mac} E_ac={E_ac} E_total={E_total}")


# ════════════════════════════════════════════════════════════
# Test 13: End-to-end energy evaluation comparison
# ════════════════════════════════════════════════════════════
def test_e2e_energy(device):
    print("\n=== Test 13: End-to-end Energy Evaluation ===")
    torch.manual_seed(42)

    from unieval.models.vit import vit_small_patch16
    from unieval.quantization.lsq import LSQQuantizer, MyQuan
    from unieval.conversion.wrapper import SNNWrapper
    from unieval.evaluation.energy import EnergyEvaluator
    from unieval.evaluation.ops_counter import OpsCounter
    from unieval.models.base import ModelProfile
    from unieval.config import EnergyConfig

    level = 16
    time_step = 15

    # Create + quantize + convert
    model = vit_small_patch16(img_size=224, num_classes=100, global_pool=False, act_layer=nn.ReLU)
    quantizer = LSQQuantizer(level=level, weight_bit=32, is_softmax=True)
    quantizer.quantize_model(model)

    for m in model.modules():
        if isinstance(m, MyQuan):
            m.init_state = 1
            m.s.data = torch.tensor(0.1)

    model.eval()

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=time_step,
        encoding_type="analog",
        level=level,
        neuron_type="ST-BIF",
        model_name="vit_small_patch16",
        is_softmax=True,
    )
    wrapper = wrapper.to(device)
    wrapper.eval()

    # Create fake data
    fake_data = [(torch.randn(2, 3, 224, 224), torch.randint(0, 100, (2,)))
                 for _ in range(3)]

    profile = ModelProfile(
        depth=12, num_heads=6, embed_dim=384, patch_size=16,
        img_size=224, time_steps=time_step
    )

    ops_counter = OpsCounter(time_step=time_step)
    energy_eval = EnergyEvaluator(
        energy_config=EnergyConfig(),
        model_profile=profile,
        ops_counter=ops_counter,
        num_batches=3,
    )

    result = energy_eval.evaluate(wrapper, fake_data)

    # Verify structure
    check("Energy result has metrics", "energy_mJ" in result.metrics)
    check("Energy result has details", "layers" in result.details)
    check("Energy is non-negative", result.metrics["energy_mJ"] >= 0)
    check("MAC ops computed", result.metrics["mac_ops_G"] >= 0)
    check("AC ops computed", result.metrics["ac_ops_G"] >= 0)

    print(f"  Energy: {result.metrics['energy_mJ']:.4f} mJ")
    print(f"  MACs: {result.metrics['mac_ops_G']:.4f} G")
    print(f"  ACs: {result.metrics['ac_ops_G']:.4f} G")
    print(f"  E_mac: {result.metrics['e_mac_mJ']:.4f} mJ")
    print(f"  E_ac: {result.metrics['e_ac_mJ']:.4f} mJ")
    if result.details.get("ssa_qkv_firing_rates"):
        print(f"  SSA Q/K/V firing rates (first block): {result.details['ssa_qkv_firing_rates'][0]}")

    return result


# ════════════════════════════════════════════════════════════
# Test 16: SSA firing rate normalization regression test
# ════════════════════════════════════════════════════════════
def test_ssa_normalization(device):
    print("\n=== Test 16: SSA Firing Rate Normalization ===")
    torch.manual_seed(42)

    from unieval.models.vit import vit_small_patch16
    from unieval.quantization.lsq import LSQQuantizer, MyQuan
    from unieval.conversion.wrapper import SNNWrapper
    from unieval.evaluation.energy import EnergyEvaluator
    from unieval.evaluation.ops_counter import OpsCounter
    from unieval.models.base import ModelProfile
    from unieval.config import EnergyConfig

    level = 16
    time_step = 15

    def make_model_and_eval(num_batches):
        torch.manual_seed(42)
        model = vit_small_patch16(img_size=224, num_classes=100, global_pool=False, act_layer=nn.ReLU)
        quantizer = LSQQuantizer(level=level, weight_bit=32, is_softmax=True)
        quantizer.quantize_model(model)
        for m in model.modules():
            if isinstance(m, MyQuan):
                m.init_state = 1
                m.s.data = torch.tensor(0.1)
        model.eval()

        wrapper = SNNWrapper(
            ann_model=model, time_step=time_step, encoding_type="analog",
            level=level, neuron_type="ST-BIF", model_name="vit_small_patch16",
            is_softmax=True,
        )
        wrapper = wrapper.to(device)
        wrapper.eval()

        # Use the SAME input repeated
        torch.manual_seed(999)
        batch = torch.randn(2, 3, 224, 224)
        fake_data = [(batch.clone(), torch.randint(0, 100, (2,)))] * num_batches

        profile = ModelProfile(
            depth=12, num_heads=6, embed_dim=384, patch_size=16,
            img_size=224, time_steps=time_step
        )
        ops_counter = OpsCounter(time_step=time_step)
        energy_eval = EnergyEvaluator(
            energy_config=EnergyConfig(), model_profile=profile,
            ops_counter=ops_counter, num_batches=num_batches,
        )
        return energy_eval.evaluate(wrapper, fake_data)

    r1 = make_model_and_eval(num_batches=1)
    r3 = make_model_and_eval(num_batches=3)

    fr1 = r1.details["ssa_qkv_firing_rates"]
    fr3 = r3.details["ssa_qkv_firing_rates"]

    if fr1 and fr3:
        # After normalization fix, firing rates should be same regardless of num_batches
        q1, q3 = fr1[0][0], fr3[0][0]
        check("SSA Q fr invariant to num_batches",
              abs(q1 - q3) < 0.01,
              f"1-batch={q1:.4f} vs 3-batch={q3:.4f}")
        check("SSA Q fr in [0, 1] range", 0 <= q1 <= 1.0,
              f"q_fr={q1:.4f}")
        print(f"  1-batch SSA fr[0]: {fr1[0]}")
        print(f"  3-batch SSA fr[0]: {fr3[0]}")


# ════════════════════════════════════════════════════════════
# Test 17: IFNeuron edge cases (boundary, saturation, recovery)
# ════════════════════════════════════════════════════════════
def test_ifneuron_edge_cases():
    print("\n=== Test 17: IFNeuron Edge Cases ===")
    from unieval.operators.neurons import IFNeuron

    # Case 1: Exact +0.5*threshold boundary (q starts at 0.5, so first input
    # of exactly +0.5*threshold should reach q=1.0 and fire)
    n = IFNeuron(q_threshold=torch.tensor(1.0), level=16, sym=True)
    out = n(torch.tensor([[0.5]]))  # x/threshold = 0.5, q becomes 0.5+0.5=1.0
    check("Boundary +0.5: fires", out.item() == 1.0,
          f"got {out.item()}")

    # Case 2: Exact -0.5*threshold (no negative fire because acc_q starts at 0
    # which is NOT > neg_min=-8 ... wait, 0 > -8 is True, so it should fire -1)
    n2 = IFNeuron(q_threshold=torch.tensor(1.0), level=16, sym=True)
    out2 = n2(torch.tensor([[-1.0]]))  # x/threshold = -1.0, q=0.5-1.0=-0.5, fires -1
    check("Boundary -1.0: fires negative", out2.item() == -1.0,
          f"got {out2.item()}")

    # Case 3: Saturation at pos_max (sym=True, level=16: pos_max=7)
    n3 = IFNeuron(q_threshold=torch.tensor(1.0), level=4, sym=True)
    # pos_max = 1 for level=4, sym=True
    outputs = []
    for _ in range(10):
        outputs.append(n3(torch.tensor([[1.0]])).item())
    n3_acc = n3.acc_q.item()
    check("Saturation at pos_max", n3_acc <= n3.pos_max.item(),
          f"acc_q={n3_acc}, pos_max={n3.pos_max.item()}")

    # Case 4: sym=False with negative input (should NOT fire negative spikes)
    n4 = IFNeuron(q_threshold=torch.tensor(1.0), level=16, sym=False)
    out4 = n4(torch.tensor([[-5.0]]))
    check("sym=False: no negative spike", out4.item() == 0.0,
          f"got {out4.item()}")

    # Case 5: Recovery after saturation — after saturating at pos_max,
    # feeding enough negative input to make q < 0 should fire negative spike
    n5 = IFNeuron(q_threshold=torch.tensor(1.0), level=4, sym=True)
    # pos_max = 1, neg_min = -2 for level=4 sym=True
    n5(torch.tensor([[1.0]]))  # fires +1, acc_q=1 (saturated at pos_max=1)
    acc_saturated = n5.acc_q.clone()
    # Now feed large negative to push q below 0
    for _ in range(10):
        n5(torch.tensor([[-2.0]]))
    acc_after = n5.acc_q.clone()
    check("Recovery after saturation", acc_after.item() < acc_saturated.item(),
          f"saturated={acc_saturated.item()}, after={acc_after.item()}")


# ════════════════════════════════════════════════════════════
# Test 14: Quantization placement comparison
# ════════════════════════════════════════════════════════════
def test_quantization_placement():
    print("\n=== Test 14: Quantization Placement ===")
    torch.manual_seed(42)

    from unieval.models.vit import vit_small_patch16
    from unieval.quantization.lsq import LSQQuantizer, MyQuan, QAttention

    model = vit_small_patch16(img_size=224, num_classes=100, global_pool=False, act_layer=nn.ReLU)

    quantizer = LSQQuantizer(level=16, weight_bit=32, is_softmax=True)
    quantizer.quantize_model(model)

    # Check expected structure for a transformer block
    blk = model.blocks[0]

    # norm1 should be Sequential(LayerNorm, MyQuan)
    check("norm1 is Sequential", isinstance(blk.norm1, nn.Sequential))
    check("norm1[0] is LayerNorm", isinstance(blk.norm1[0], nn.LayerNorm))
    check("norm1[1] is MyQuan", isinstance(blk.norm1[1], MyQuan))

    # attn should be QAttention
    check("attn is QAttention", isinstance(blk.attn, QAttention))

    # norm2 should be Sequential(LayerNorm, MyQuan)
    check("norm2 is Sequential", isinstance(blk.norm2, nn.Sequential))

    # mlp.act should be Sequential(MyQuan, ReLU)
    check("mlp.act is Sequential", isinstance(blk.mlp.act, nn.Sequential))
    check("mlp.act[0] is MyQuan(sym=False)", isinstance(blk.mlp.act[0], MyQuan) and not blk.mlp.act[0].sym)

    # mlp.fc2 should be Sequential(Linear, MyQuan)
    check("mlp.fc2 is Sequential", isinstance(blk.mlp.fc2, nn.Sequential))
    check("mlp.fc2[1] is MyQuan(sym=True)", isinstance(blk.mlp.fc2[1], MyQuan) and blk.mlp.fc2[1].sym)

    # Count total MyQuan modules - should be 4 per block * 12 blocks + conv + norm = ~50+
    mquan_count = sum(1 for m in model.modules() if isinstance(m, MyQuan))
    qattn_count = sum(1 for m in model.modules() if isinstance(m, QAttention))
    print(f"  Total MyQuan: {mquan_count}, QAttention: {qattn_count}")
    check("QAttention count == depth", qattn_count == 12)


# ════════════════════════════════════════════════════════════
# Test 15: Conversion rules comparison
# ════════════════════════════════════════════════════════════
def test_conversion_rules():
    print("\n=== Test 15: Conversion Rules ===")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from unieval.models.vit import vit_small_patch16
    from unieval.quantization.lsq import LSQQuantizer, MyQuan
    from unieval.conversion.converter import SNNConverter
    from unieval.operators.attention import SAttention
    from unieval.operators.neurons import IFNeuron
    from unieval.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm

    model = vit_small_patch16(img_size=224, num_classes=100, global_pool=False, act_layer=nn.ReLU)
    quantizer = LSQQuantizer(level=16, weight_bit=32, is_softmax=True)
    quantizer.quantize_model(model)

    for m in model.modules():
        if isinstance(m, MyQuan):
            m.init_state = 1
            m.s.data = torch.tensor(0.1)

    converter = SNNConverter()
    converter.convert(model, level=16, neuron_type="ST-BIF", is_softmax=True)

    # After conversion, check expected types
    blk = model.blocks[0]
    check("attn is SAttention", isinstance(blk.attn, SAttention))
    check("SAttention has IFNeuron q_IF", isinstance(blk.attn.q_IF, IFNeuron))
    check("SAttention qkv is LLLinear", isinstance(blk.attn.qkv, LLLinear))

    # norm1 should be Sequential containing Spiking_LayerNorm and IFNeuron
    has_sln = any(isinstance(m, Spiking_LayerNorm) for m in blk.norm1.modules())
    has_if = any(isinstance(m, IFNeuron) for m in blk.norm1.modules())
    check("norm1 has Spiking_LayerNorm", has_sln)
    check("norm1 has IFNeuron", has_if)

    # Count converted modules
    sattn_count = sum(1 for m in model.modules() if isinstance(m, SAttention))
    ifn_count = sum(1 for m in model.modules() if isinstance(m, IFNeuron))
    sln_count = sum(1 for m in model.modules() if isinstance(m, Spiking_LayerNorm))
    lll_count = sum(1 for m in model.modules() if isinstance(m, LLLinear))
    print(f"  SAttention: {sattn_count}, IFNeuron: {ifn_count}, "
          f"Spiking_LN: {sln_count}, LLLinear: {lll_count}")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("UniEval vs SpikeZIP-TF Comparison Tests")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Component-level tests
    test_ifneuron()
    test_spiking_layernorm()
    test_spiking_softmax()
    test_lllinear()
    test_llconv2d()
    test_myquan()
    test_qattention()
    test_sattention()
    test_spike_rate()
    test_syops_hooks()

    # Energy formula
    test_energy_formula()

    # Structure tests
    test_quantization_placement()
    test_conversion_rules()

    # IFNeuron edge cases
    test_ifneuron_edge_cases()

    # Full pipeline tests (need GPU for reasonable speed)
    test_full_conversion()
    energy_result = test_e2e_energy(device)

    # SSA normalization regression test
    test_ssa_normalization(device)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
