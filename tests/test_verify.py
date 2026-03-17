#!/usr/bin/env python3
"""UniEval Verification Test Script.

This script must be run in an environment with CUDA and PyTorch installed.
It verifies the correctness of the UniEval framework by testing:

1. All module imports (including adapter)
2. Registry system
3. Config system
4. Operator instantiation and forward pass
5. Quantization pipeline (LSQ & PTQ)
6. SNN conversion pipeline
7. SNNWrapper temporal inference
8. Evaluation components (OpsCounter, spike_rate)
9. Model creation
10. forward_multistep equivalence (strong semantic contract)
11. Judger pre-cache and reset_model flat traversal
12. New SNNWrapper API (step_encoded, encode_sequence, run_auto)
13. Full pipeline integration

Usage:
    python tests/test_verify.py

Requirements:
    - Python >= 3.8
    - PyTorch >= 1.10 (with CUDA)
    - numpy, scipy
"""

import os
import sys
import traceback

# Add project root to sys.path so `import unieval` works
# regardless of where the script is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PASS = 0
FAIL = 0
ERRORS = []


def run_test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL += 1
        ERRORS.append((name, e))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


# ===== Test 1: Core Imports =====

def test_core_imports():
    from unieval.registry import Registry, QUANTIZER_REGISTRY, NEURON_REGISTRY, MODEL_PROFILE_REGISTRY
    from unieval.config import UniEvalConfig, QuantConfig, ConversionConfig, EnergyConfig, EvalConfig
    assert isinstance(QUANTIZER_REGISTRY, Registry)
    assert isinstance(NEURON_REGISTRY, Registry)


def test_operator_imports():
    from unieval.snn.operators.base import SNNOperator
    from unieval.snn.operators.neurons import IFNeuron, ORIIFNeuron
    from unieval.snn.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm, SpikeMaxPooling
    from unieval.snn.operators.attention import SAttention, spiking_softmax, multi, multi1


def test_quantization_imports():
    from unieval.qann.quantization.base import BaseQuantizer, QuantPlacementRule
    from unieval.qann.operators.lsq import MyQuan, QAttention, QuanConv2d, QuanLinear
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.qann.operators.ptq import PTQQuan
    from unieval.qann.quantization.ptq import PTQQuantizer


def test_conversion_imports():
    from unieval.snn.snnConverter.rules import ConversionRule, DEFAULT_CONVERSION_RULES
    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.snnConverter.wrapper import SNNWrapper, Judger, reset_model, attn_convert
    from unieval.snn.snnConverter.adapter import (
        ADAPTER_REGISTRY, ModelExecutionAdapter, ViTExecutionAdapter, DefaultExecutionAdapter,
    )
    assert "vit" in ADAPTER_REGISTRY
    assert "default" in ADAPTER_REGISTRY


def test_evaluation_imports():
    from unieval.evaluation.benchmarks.base import EvalResult, BaseEvaluator
    from unieval.evaluation.feasibility.spike_utils import spike_rate
    from unieval.evaluation.energy.ops_counter import OpsCounter
    from unieval.evaluation.benchmarks.accuracy import AccuracyEvaluator
    from unieval.evaluation.energy.energy import EnergyEvaluator


def test_model_imports():
    from unieval.ann.models.base import ModelProfile
    from unieval.ann.models.vit import VisionTransformer, VisionTransformerDVS
    from unieval.ann.models.vit import vit_small_patch16, vit_base_patch16


def test_api_imports():
    from unieval.qann import quantize, calibrate_ptq
    from unieval.snn import convert
    from unieval.evaluation import evaluate_accuracy, evaluate_energy, evaluate_perplexity


# ===== Test 2: Registry System =====

def test_registry_basic():
    from unieval.registry import Registry
    reg = Registry("test")
    @reg.register("foo")
    class Foo:
        pass
    assert reg.get("foo") is Foo
    assert "foo" in reg
    assert reg.list_keys() == ["foo"]


def test_registry_duplicate_error():
    from unieval.registry import Registry
    reg = Registry("test_dup")
    reg.register_obj("bar", 42)
    try:
        reg.register_obj("bar", 99)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_registry_missing_error():
    from unieval.registry import Registry
    reg = Registry("test_miss")
    try:
        reg.get("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_neuron_registry():
    from unieval.registry import NEURON_REGISTRY
    keys = NEURON_REGISTRY.list_keys()
    assert "IF" in keys, f"IFNeuron not registered, keys={keys}"
    assert "ORIIF" in keys, f"ORIIFNeuron not registered, keys={keys}"


def test_quantizer_registry():
    from unieval.registry import QUANTIZER_REGISTRY
    keys = QUANTIZER_REGISTRY.list_keys()
    assert "lsq" in keys, f"LSQ not registered, keys={keys}"
    assert "ptq" in keys, f"PTQ not registered, keys={keys}"


def test_model_profile_registry():
    from unieval.registry import MODEL_PROFILE_REGISTRY
    keys = MODEL_PROFILE_REGISTRY.list_keys()
    assert "vit_small" in keys
    assert "vit_base" in keys
    assert "vit_large" in keys
    assert "vit_small_dvs" in keys
    profile = MODEL_PROFILE_REGISTRY.get("vit_small")
    assert profile.depth == 12
    assert profile.num_heads == 6
    assert profile.embed_dim == 384


# ===== Test 3: Config System =====

def test_config_defaults():
    from unieval.config import UniEvalConfig
    cfg = UniEvalConfig()
    assert cfg.model_name == "vit_small"
    assert cfg.quant.level == 16
    assert cfg.conversion.encoding_type == "analog"
    assert cfg.energy.e_mac == 4.6
    assert cfg.energy.e_ac == 0.9
    assert cfg.evaluation.topk == [1, 5]


def test_config_custom():
    from unieval.config import UniEvalConfig, QuantConfig
    cfg = UniEvalConfig(
        model_name="vit_base",
        quant=QuantConfig(level=32),
    )
    assert cfg.model_name == "vit_base"
    assert cfg.quant.level == 32


# ===== Test 4: Operators =====

def test_ifneuron_forward():
    import torch
    from unieval.snn.operators.neurons import IFNeuron

    neuron = IFNeuron(q_threshold=torch.tensor(0.5), level=16, sym=True)
    x = torch.randn(2, 4)
    y = neuron(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    # Test reset
    neuron.reset()
    assert neuron.is_work == False
    assert neuron.q == 0.0


def test_oriifneuron_forward():
    import torch
    from unieval.snn.operators.neurons import ORIIFNeuron

    neuron = ORIIFNeuron(q_threshold=torch.tensor(0.5), level=16)
    x = torch.randn(2, 4).abs()  # unsigned
    y = neuron(x)
    assert y.shape == x.shape


def test_spiking_layernorm():
    import torch
    from unieval.snn.operators.layers import Spiking_LayerNorm

    sln = Spiking_LayerNorm(dim=8)
    x = torch.randn(2, 4, 8)
    y1 = sln(x)
    y2 = sln(x)
    assert y1.shape == x.shape
    assert y2.shape == x.shape
    sln.reset()
    assert sln.Y_pre is None


def test_lllinear():
    import torch
    import torch.nn as nn
    from unieval.snn.operators.layers import LLLinear

    linear = nn.Linear(8, 4)
    ll = LLLinear(linear, neuron_type="ST-BIF", level=16)
    x = torch.randn(2, 3, 8)
    y = ll(x)
    assert y.shape == (2, 3, 4)

    # Test zero input
    ll.reset()
    z = torch.zeros(2, 3, 8)
    y_zero = ll(z)
    assert y_zero.shape == (2, 3, 4)


def test_llconv2d():
    import torch
    import torch.nn as nn
    from unieval.snn.operators.layers import LLConv2d

    conv = nn.Conv2d(3, 16, 3, padding=1)
    ll = LLConv2d(conv, neuron_type="ST-BIF", level=16)
    x = torch.randn(1, 3, 8, 8)
    y = ll(x)
    assert y.shape == (1, 16, 8, 8)

    ll.reset()
    assert ll.is_work == False


def test_spiking_softmax():
    import torch
    from unieval.snn.operators.attention import spiking_softmax

    ssm = spiking_softmax()
    x = torch.randn(2, 4, 4)
    y1 = ssm(x)
    y2 = ssm(x)
    assert y1.shape == x.shape
    ssm.reset()


def test_sattention():
    import torch
    from unieval.snn.operators.attention import SAttention

    sa = SAttention(dim=64, num_heads=4, level=16)
    x = torch.randn(1, 8, 64)
    y = sa(x)
    assert y.shape == (1, 8, 64)


def test_spike_max_pooling():
    import torch
    import torch.nn as nn
    from unieval.snn.operators.layers import SpikeMaxPooling

    pool = SpikeMaxPooling(nn.MaxPool2d(2))
    x = torch.randn(1, 3, 4, 4)
    y = pool(x)
    assert y.shape == (1, 3, 2, 2)
    pool.reset()


# ===== Test 5: Quantization =====

def test_myquan_full_precision():
    import torch
    from unieval.qann.operators.lsq import MyQuan

    mq = MyQuan(level=512, sym=True)
    x = torch.randn(2, 4)
    y = mq(x)
    assert torch.allclose(x, y), "Full precision MyQuan should be identity"


def test_myquan_quantize():
    import torch
    from unieval.qann.operators.lsq import MyQuan

    mq = MyQuan(level=16, sym=True)
    mq.eval()  # skip init_state logic
    mq.init_state = 1  # pretend already initialized
    mq.s.data = torch.tensor(0.1)
    x = torch.randn(2, 4)
    y = mq(x)
    assert y.shape == x.shape


def test_qattention():
    import torch
    from unieval.qann.operators.lsq import QAttention

    qa = QAttention(dim=64, num_heads=4, qkv_bias=True, level=16)
    qa.eval()
    # Initialize all MyQuan s parameters
    for name, m in qa.named_modules():
        if hasattr(m, 'init_state'):
            m.init_state = 1
            m.s.data = torch.tensor(0.1)
    x = torch.randn(1, 8, 64)
    y = qa(x)
    assert y.shape == (1, 8, 64)


def test_lsq_quantizer():
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.ann.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10,
        global_pool=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model = quantizer.quantize_model(model)

    # Verify QAttention was placed in blocks
    from unieval.qann.operators.lsq import QAttention, MyQuan
    found_qattn = False
    found_myquan = False
    for name, m in model.named_modules():
        if isinstance(m, QAttention):
            found_qattn = True
        if isinstance(m, MyQuan):
            found_myquan = True
    assert found_qattn, "QAttention not found after quantization"
    assert found_myquan, "MyQuan not found after quantization"


def test_ptq_quan():
    import torch
    from unieval.qann.operators.ptq import PTQQuan

    pq = PTQQuan(level=16, sym=True)
    assert pq.calibrated == False
    # PTQ calibrates on first forward (needs valid data for KL-div)
    # Just check construction works
    assert pq.level == 16


# ===== Test 6: Conversion =====

def test_conversion_rules():
    from unieval.snn.snnConverter.rules import DEFAULT_CONVERSION_RULES
    assert len(DEFAULT_CONVERSION_RULES) == 9
    names = [r.name for r in DEFAULT_CONVERSION_RULES]
    assert "qattention_to_sattention" in names
    assert "quan_to_ifneuron" in names
    assert "relu_to_identity" in names
    # New composite Q*→S* rules
    assert "qconv2d_to_sconv2d" in names
    assert "qlinear_to_slinear" in names
    assert "qnorm_layernorm_to_spiking" in names


def test_snn_converter():
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.operators.neurons import IFNeuron
    from unieval.snn.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm
    from unieval.snn.operators.attention import SAttention
    from unieval.ann.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10,
        global_pool=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model = quantizer.quantize_model(model)

    converter = SNNConverter()
    converter.convert(model, level=16, neuron_type="ST-BIF", is_softmax=True)

    # Verify conversion results
    found_sattn = False
    found_if = False
    found_lllinear = False
    found_spiking_ln = False
    for name, m in model.named_modules():
        if isinstance(m, SAttention):
            found_sattn = True
        if isinstance(m, IFNeuron):
            found_if = True
        if isinstance(m, LLLinear):
            found_lllinear = True
        if isinstance(m, Spiking_LayerNorm):
            found_spiking_ln = True

    assert found_sattn, "SAttention not found after conversion"
    assert found_if, "IFNeuron not found after conversion"
    assert found_lllinear, "LLLinear not found after conversion"
    assert found_spiking_ln, "Spiking_LayerNorm not found after conversion"


# ===== Test 7: SNNWrapper =====

def test_snn_wrapper_analog():
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.snn.snnConverter.wrapper import SNNWrapper
    from unieval.ann.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10,
        global_pool=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model = quantizer.quantize_model(model)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=4,  # small for testing
        encoding_type="analog",
        level=16,
        neuron_type="ST-BIF",
        model_name="vit_small",
        is_softmax=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = wrapper.to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    output, timesteps = wrapper(x)
    assert output.shape == (1, 10), f"Output shape mismatch: {output.shape}"
    assert timesteps <= 4

    wrapper.reset()


# ===== Test 8: Evaluation =====

def test_spike_rate_spiking():
    import torch
    from unieval.evaluation.feasibility.spike_utils import spike_rate

    # Spiking input: only {-1, 0, 1}
    x = torch.tensor([[-1.0, 0, 1, 0], [0, 0, 1, -1]])
    is_spike, rate, _ = spike_rate(x)
    assert is_spike == True
    assert 0 < rate < 1


def test_spike_rate_non_spiking():
    import torch
    from unieval.evaluation.feasibility.spike_utils import spike_rate

    # Non-spiking input: continuous values
    x = torch.randn(4, 8)
    is_spike, rate, _ = spike_rate(x)
    assert is_spike == False
    assert rate == 1


def test_spike_rate_zero():
    import torch
    from unieval.evaluation.feasibility.spike_utils import spike_rate

    x = torch.zeros(4, 8)
    is_spike, rate, _ = spike_rate(x)
    assert is_spike == True
    assert rate == 0


def test_ops_counter():
    import torch
    import torch.nn as nn
    from unieval.evaluation.energy.ops_counter import OpsCounter

    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    counter = OpsCounter()
    counter.attach(model)

    x = torch.randn(2, 8)
    _ = model(x)

    syops, params = counter.compute_total(model)
    assert syops[0] > 0, "Total ops should be > 0"
    counter.detach(model)


def test_eval_result():
    from unieval.evaluation.benchmarks.base import EvalResult
    result = EvalResult(
        metrics={"top1": 75.5, "top5": 92.3},
        details={"layers": []},
    )
    assert result.metrics["top1"] == 75.5
    s = repr(result)
    assert "75.5" in s


# ===== Test 9: Model Creation =====

def test_vit_small_creation():
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.ann.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10,
        global_pool=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    assert hasattr(model, 'patch_embed')
    assert hasattr(model, 'blocks')
    assert len(model.blocks) == 12


# ===== Test 10: forward_multistep Equivalence =====

def test_spiking_layernorm_multistep():
    """Spiking_LayerNorm.forward_multistep ≡ sequential single-step."""
    import torch
    from unieval.snn.operators.layers import Spiking_LayerNorm

    sln1 = Spiking_LayerNorm(dim=8)
    sln2 = Spiking_LayerNorm(dim=8)
    sln2.load_state_dict(sln1.state_dict())

    T = 5
    x_seq = torch.randn(T, 2, 4, 8)

    # Single-step
    outputs_single = []
    for t in range(T):
        outputs_single.append(sln1(x_seq[t]))
    outputs_single = torch.stack(outputs_single)

    # Multi-step
    outputs_multi = sln2.forward_multistep(x_seq)

    assert torch.allclose(outputs_single, outputs_multi, atol=1e-5), \
        f"Output diff: {(outputs_single - outputs_multi).abs().max()}"
    assert torch.allclose(sln1.X, sln2.X, atol=1e-5), "State X mismatch"
    assert torch.allclose(sln1.Y_pre, sln2.Y_pre, atol=1e-5), "State Y_pre mismatch"


def test_spiking_layernorm_multistep_from_state():
    """forward_multistep works from non-initial state (strong contract)."""
    import torch
    from unieval.snn.operators.layers import Spiking_LayerNorm

    sln1 = Spiking_LayerNorm(dim=8)
    sln2 = Spiking_LayerNorm(dim=8)
    sln2.load_state_dict(sln1.state_dict())

    # Run a few single steps to build up state
    warmup = torch.randn(3, 2, 4, 8)
    for t in range(3):
        sln1(warmup[t])
        sln2(warmup[t])

    # Now test multistep from this non-initial state
    x_seq = torch.randn(4, 2, 4, 8)

    outputs_single = torch.stack([sln1(x_seq[t]) for t in range(4)])
    outputs_multi = sln2.forward_multistep(x_seq)

    assert torch.allclose(outputs_single, outputs_multi, atol=1e-5), \
        f"From-state diff: {(outputs_single - outputs_multi).abs().max()}"


def test_spike_max_pooling_multistep():
    """SpikeMaxPooling.forward_multistep ≡ sequential single-step."""
    import torch
    import torch.nn as nn
    from unieval.snn.operators.layers import SpikeMaxPooling

    pool1 = SpikeMaxPooling(nn.MaxPool2d(2))
    pool2 = SpikeMaxPooling(nn.MaxPool2d(2))

    T = 5
    x_seq = torch.randn(T, 2, 3, 4, 4)

    # Single-step
    outputs_single = torch.stack([pool1(x_seq[t]) for t in range(T)])

    # Multi-step
    outputs_multi = pool2.forward_multistep(x_seq)

    assert torch.allclose(outputs_single, outputs_multi, atol=1e-5), \
        f"Output diff: {(outputs_single - outputs_multi).abs().max()}"
    assert torch.allclose(pool1.accumulation, pool2.accumulation, atol=1e-5)


def test_spiking_softmax_multistep():
    """spiking_softmax.forward_multistep ≡ sequential single-step."""
    import torch
    from unieval.snn.operators.attention import spiking_softmax

    ssm1 = spiking_softmax()
    ssm2 = spiking_softmax()

    T = 5
    x_seq = torch.randn(T, 2, 4, 6, 6)

    # Single-step
    outputs_single = torch.stack([ssm1(x_seq[t]) for t in range(T)])

    # Multi-step
    outputs_multi = ssm2.forward_multistep(x_seq)

    assert torch.allclose(outputs_single, outputs_multi, atol=1e-5), \
        f"Output diff: {(outputs_single - outputs_multi).abs().max()}"


def test_ifneuron_multistep_fallback():
    """IFNeuron.forward_multistep uses default for-loop (inherits from SNNOperator)."""
    import torch
    from unieval.snn.operators.neurons import IFNeuron

    n1 = IFNeuron(q_threshold=torch.tensor(0.5), level=16, sym=True)
    n2 = IFNeuron(q_threshold=torch.tensor(0.5), level=16, sym=True)

    T = 4
    x_seq = torch.randn(T, 2, 4)

    outputs_single = torch.stack([n1(x_seq[t]) for t in range(T)])
    outputs_multi = n2.forward_multistep(x_seq)

    assert torch.allclose(outputs_single, outputs_multi, atol=1e-6), \
        f"Output diff: {(outputs_single - outputs_multi).abs().max()}"


def test_participates_in_early_stop_flags():
    """Verify participates_in_early_stop is set correctly per operator type."""
    import torch
    from unieval.snn.operators.neurons import IFNeuron, ORIIFNeuron
    from unieval.snn.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm, SpikeMaxPooling
    from unieval.snn.operators.attention import SAttention, spiking_softmax
    import torch.nn as nn

    # Should participate in early stop
    assert IFNeuron(torch.tensor(1.0), 16, True).participates_in_early_stop == True
    assert ORIIFNeuron(torch.tensor(1.0), 16).participates_in_early_stop == True
    assert LLConv2d(nn.Conv2d(3, 3, 3)).participates_in_early_stop == True
    assert LLLinear(nn.Linear(8, 4)).participates_in_early_stop == True

    # Should NOT participate in early stop
    assert Spiking_LayerNorm(8).participates_in_early_stop == False
    assert SpikeMaxPooling(nn.MaxPool2d(2)).participates_in_early_stop == False
    assert spiking_softmax().participates_in_early_stop == False
    assert SAttention(dim=64, num_heads=4, level=16).participates_in_early_stop == False


def test_working_property_defensive():
    """working property raises if participates_in_early_stop=True but no is_work."""
    from unieval.snn.operators.base import SNNOperator

    class BadOperator(SNNOperator):
        participates_in_early_stop = True
        def reset(self): pass

    op = BadOperator()
    try:
        _ = op.working
        assert False, "Should raise AttributeError"
    except AttributeError:
        pass


# ===== Test 11: Judger and reset_model =====

def test_judger_precache():
    """Judger caches early-stop operators at init time."""
    import torch
    import torch.nn as nn
    from unieval.snn.operators.neurons import IFNeuron
    from unieval.snn.operators.layers import Spiking_LayerNorm
    from unieval.snn.snnConverter.wrapper import Judger

    model = nn.Sequential(
        IFNeuron(torch.tensor(0.5), level=16, sym=True),
        Spiking_LayerNorm(dim=8),
    )
    judger = Judger(model)
    # Only IFNeuron should be in early_stop_ops (Spiking_LayerNorm has flag=False)
    assert len(judger._early_stop_ops) == 1


def test_reset_model_flat():
    """reset_model uses flat traversal and resets all SNNOperator instances."""
    import torch
    import torch.nn as nn
    from unieval.snn.operators.neurons import IFNeuron
    from unieval.snn.operators.layers import Spiking_LayerNorm
    from unieval.snn.operators.attention import spiking_softmax
    from unieval.snn.snnConverter.wrapper import reset_model

    model = nn.Sequential(
        IFNeuron(torch.tensor(0.5), level=16, sym=True),
        Spiking_LayerNorm(dim=8),
        spiking_softmax(),
    )
    # Run a forward to build state
    x = torch.randn(2, 8)
    model[0](x.narrow(-1, 0, 4))  # IFNeuron
    model[1](x.unsqueeze(1))      # Spiking_LayerNorm
    model[2](x.unsqueeze(1))      # spiking_softmax

    reset_model(model)

    # All should be in reset state
    assert model[0].is_work == False
    assert model[1].Y_pre is None
    assert isinstance(model[2].X, float) and model[2].X == 0.0


# ===== Test 12: New SNNWrapper API =====

def test_encode_sequence_analog():
    """encode_sequence with analog encoding: [x, 0, 0, ...]."""
    import torch
    from unieval.snn.snnConverter.wrapper import SNNWrapper
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.ann.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10, global_pool=True,
        act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model = quantizer.quantize_model(model)

    wrapper = SNNWrapper(
        ann_model=model, time_step=4, encoding_type="analog",
        level=16, neuron_type="ST-BIF", model_name="vit_small",
    )

    x = torch.randn(2, 3, 224, 224)
    x_seq = wrapper.encode_sequence(x, T=4)
    assert x_seq.shape == (4, 2, 3, 224, 224)
    assert torch.allclose(x_seq[0], x)
    assert (x_seq[1:] == 0).all()


def test_encode_sequence_rate():
    """encode_sequence with rate encoding: uniform division."""
    import torch
    from unieval.snn.snnConverter.wrapper import SNNWrapper
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.ann.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10, global_pool=True,
        act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model = quantizer.quantize_model(model)

    wrapper = SNNWrapper(
        ann_model=model, time_step=4, encoding_type="rate",
        level=16, neuron_type="ST-BIF", model_name="vit_small",
    )

    x = torch.randn(2, 3, 224, 224)
    x_seq = wrapper.encode_sequence(x, T=8, encoding_type="rate")
    assert x_seq.shape == (8, 2, 3, 224, 224)
    # Sum should reconstruct original
    assert torch.allclose(x_seq.sum(dim=0), x, atol=1e-5)


def test_step_encoded_api():
    """step_encoded runs a single timestep on pre-encoded input."""
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.snn.snnConverter.wrapper import SNNWrapper
    from unieval.ann.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10, global_pool=True,
        act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model = quantizer.quantize_model(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = SNNWrapper(
        ann_model=model, time_step=4, encoding_type="analog",
        level=16, neuron_type="ST-BIF", model_name="vit_small",
    ).to(device).eval()

    x = torch.randn(1, 3, 224, 224).to(device)
    wrapper.reset()

    # step 0: full input
    out0 = wrapper.step_encoded(x)
    assert out0.shape == (1, 10)

    # step 1: zero input
    out1 = wrapper.step_encoded(torch.zeros_like(x))
    assert out1.shape == (1, 10)
    assert wrapper._current_t == 2


def test_step_encoded_vs_run_auto():
    """Manual step_encoded loop should match run_auto for analog encoding."""
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.snn.snnConverter.wrapper import SNNWrapper
    from unieval.ann.models.vit import VisionTransformer

    model = VisionTransformer(
        img_size=32, patch_size=8, in_chans=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.,
        qkv_bias=True, act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=True,
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model.train()
    with torch.no_grad():
        model(torch.randn(4, 3, 32, 32))
    model.eval()
    model = quantizer.quantize_model(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run auto
    from copy import deepcopy
    model1 = deepcopy(model)
    wrapper1 = SNNWrapper(
        ann_model=model1, time_step=8, encoding_type="analog",
        level=16, neuron_type="ST-BIF", model_name="vit_tiny",
    ).to(device).eval()

    x = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        accu_auto, T_auto = wrapper1(x)

    # Manual step_encoded
    model2 = deepcopy(model)
    wrapper2 = SNNWrapper(
        ann_model=model2, time_step=8, encoding_type="analog",
        level=16, neuron_type="ST-BIF", model_name="vit_tiny",
    ).to(device).eval()

    wrapper2.reset()
    x_seq = wrapper2.encode_sequence(x, T=T_auto)
    accu_manual = torch.zeros_like(accu_auto)
    with torch.no_grad():
        for t in range(T_auto):
            out_t = wrapper2.step_encoded(x_seq[t])
            accu_manual += out_t

    assert torch.allclose(accu_auto, accu_manual, atol=1e-5), \
        f"Max diff: {(accu_auto - accu_manual).abs().max()}"


# ===== Test 13: Bug Fix Verification =====

def test_myquan_device_agnostic():
    """MyQuan init_state should use x.device, not hardcoded .cuda()."""
    import torch
    from unieval.qann.operators.lsq import MyQuan

    mq = MyQuan(level=16, sym=True)
    mq.train()
    x = torch.randn(4, 8)  # CPU tensor
    y = mq(x)
    assert y.shape == x.shape
    # s should be on same device as x
    assert str(mq.s.device) == str(x.device)


def test_sattention_reset_before_conversion():
    """SAttention.reset() should not crash when qkv/proj are still nn.Linear."""
    import torch
    from unieval.snn.operators.attention import SAttention

    sa = SAttention(dim=64, num_heads=4, level=16)
    x = torch.randn(1, 8, 64)
    _ = sa(x)
    # Should NOT raise AttributeError
    sa.reset()


def test_ops_counter_configurable_timestep():
    """OpsCounter should use configurable time_step, not hardcoded 15."""
    import torch
    import torch.nn as nn
    from unieval.evaluation.energy.ops_counter import OpsCounter

    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
    counter = OpsCounter(time_step=32)
    counter.attach(model)

    x = torch.randn(2, 8)
    _ = model(x)

    # times_counter should be 32, not 15
    assert model.__times_counter__ == 32
    counter.detach(model)


def test_spiking_softmax_no_deepcopy():
    """spiking_softmax should use detach().clone() instead of deepcopy."""
    import torch
    from unieval.snn.operators.attention import spiking_softmax

    ssm = spiking_softmax()
    x = torch.randn(2, 4, 4)
    y1 = ssm(x)
    assert y1.shape == x.shape
    # Y_pre should be a tensor after first forward
    assert torch.is_tensor(ssm.Y_pre)
    y2 = ssm(x)
    assert y2.shape == x.shape


def test_model_profile_num_patches():
    """ModelProfile should compute num_patches from img_size and patch_size."""
    from unieval.ann.models.base import ModelProfile

    profile = ModelProfile(
        depth=12, num_heads=6, embed_dim=384,
        patch_size=16, img_size=224,
    )
    assert profile.num_patches == 196  # (224/16)^2

    profile14 = ModelProfile(
        depth=12, num_heads=6, embed_dim=384,
        patch_size=14, img_size=224,
    )
    assert profile14.num_patches == 256  # (224/14)^2


def test_config_img_size():
    """UniEvalConfig should have img_size field."""
    from unieval.config import UniEvalConfig
    cfg = UniEvalConfig()
    assert cfg.img_size == 224


def test_remove_softmax():
    """remove_softmax should replace Attention with Attention_no_softmax."""
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.ann.models.vit import (
        vit_small_patch16, Attention, Attention_no_softmax, remove_softmax,
    )

    model = vit_small_patch16(
        num_classes=10, global_pool=True,
        act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    # Before: should have Attention modules
    has_attn = any(isinstance(m, Attention) for m in model.modules())
    assert has_attn

    remove_softmax(model)

    # After: should have Attention_no_softmax, no Attention
    has_attn = any(isinstance(m, Attention) for m in model.modules())
    has_relu_attn = any(isinstance(m, Attention_no_softmax) for m in model.modules())
    assert not has_attn, "Standard Attention should be replaced"
    assert has_relu_attn, "Attention_no_softmax should be present"


def test_wrapper_no_debug_print(capsys=None):
    """SNNWrapper.run_auto should not print when verbose=False."""
    import torch
    import torch.nn as nn
    from functools import partial
    from io import StringIO
    import sys
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.snn.snnConverter.wrapper import SNNWrapper
    from unieval.ann.models.vit import VisionTransformer

    model = VisionTransformer(
        img_size=32, patch_size=8, in_chans=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.,
        qkv_bias=True, act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=True,
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model.train()
    with torch.no_grad():
        model(torch.randn(4, 3, 32, 32))
    model.eval()
    model = quantizer.quantize_model(model)

    wrapper = SNNWrapper(
        ann_model=model, time_step=4, encoding_type="analog",
        level=16, neuron_type="ST-BIF", model_name="vit_tiny",
    ).eval()

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            wrapper(x, verbose=False)
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert "Time Step" not in output, f"Debug output leaked: {output!r}"


# ===== Test 14: New Fix Verification =====

def test_ifneuron_default_attributes():
    """IFNeuron should have neuron_type and is_init attributes by default."""
    import torch
    from unieval.snn.operators.neurons import IFNeuron, ORIIFNeuron

    n = IFNeuron(q_threshold=torch.tensor(0.5), level=16, sym=True)
    assert n.neuron_type == "IF"
    assert n.is_init == True

    n2 = ORIIFNeuron(q_threshold=torch.tensor(0.5), level=16)
    assert n2.neuron_type == "ORIIF"
    assert n2.is_init == True


def test_llconv2d_multistep():
    """LLConv2d.forward_multistep uses pre-allocated output."""
    import torch
    import torch.nn as nn
    from unieval.snn.operators.layers import LLConv2d

    conv = nn.Conv2d(3, 8, 3, padding=1)
    ll = LLConv2d(conv, neuron_type="ST-BIF", level=16)

    T = 3
    x_seq = torch.randn(T, 1, 3, 8, 8)

    # Multi-step
    result = ll.forward_multistep(x_seq)
    assert result.shape == (T, 1, 8, 8, 8)

    # Verify equivalence with sequential
    ll2 = LLConv2d(conv, neuron_type="ST-BIF", level=16)
    ll2.load_state_dict(ll.state_dict(), strict=False)
    ll2.reset()
    ll.reset()

    x_seq2 = torch.randn(T, 1, 3, 8, 8)
    single = torch.stack([ll(x_seq2[t]) for t in range(T)])
    ll2.reset()
    multi = ll2.forward_multistep(x_seq2)
    assert torch.allclose(single, multi, atol=1e-6)


def test_lllinear_multistep():
    """LLLinear.forward_multistep uses pre-allocated output."""
    import torch
    import torch.nn as nn
    from unieval.snn.operators.layers import LLLinear

    linear = nn.Linear(8, 4)
    ll = LLLinear(linear, neuron_type="ST-BIF", level=16)

    T = 3
    x_seq = torch.randn(T, 2, 3, 8)

    result = ll.forward_multistep(x_seq)
    assert result.shape == (T, 2, 3, 4)


def test_sattention_multistep():
    """SAttention.forward_multistep uses pre-allocated output."""
    import torch
    from unieval.snn.operators.attention import SAttention

    sa = SAttention(dim=32, num_heads=2, level=16)
    T = 3
    x_seq = torch.randn(T, 1, 4, 32)
    result = sa.forward_multistep(x_seq)
    assert result.shape == (T, 1, 4, 32)


def test_adapter_forward_multistep_sequential():
    """ViTExecutionAdapter.forward_multistep should handle nn.Sequential proj."""
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.lsq import LSQQuantizer
    from unieval.snn.snnConverter.wrapper import SNNWrapper
    from unieval.ann.models.vit import VisionTransformer

    model = VisionTransformer(
        img_size=32, patch_size=8, in_chans=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.,
        qkv_bias=True, act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=True,
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model.train()
    with torch.no_grad():
        model(torch.randn(4, 3, 32, 32))
    model.eval()
    model = quantizer.quantize_model(model)

    wrapper = SNNWrapper(
        ann_model=model, time_step=4, encoding_type="analog",
        level=16, neuron_type="ST-BIF", model_name="vit_tiny",
    ).eval()

    # After conversion, patch_embed.proj should be SConv2d (composite operator)
    from unieval.snn.operators.composites import SConv2d
    assert isinstance(wrapper.model.patch_embed.proj, SConv2d), \
        f"Expected SConv2d, got {type(wrapper.model.patch_embed.proj)}"

    # forward_encoded should NOT crash (this was the bug)
    wrapper.reset()
    x = torch.randn(1, 3, 32, 32)
    x_seq = wrapper.encode_sequence(x, T=3)
    with torch.no_grad():
        output_seq = wrapper.forward_encoded(x_seq)
    assert output_seq.shape[0] == 3
    assert output_seq.shape[1] == 1
    assert output_seq.shape[2] == 10


def test_dvs_model_creation():
    """DVS model should be creatable directly."""
    import torch.nn as nn
    from functools import partial
    from unieval.ann.models.vit import vit_small_patch16_dvs

    model = vit_small_patch16_dvs(
        num_classes=10,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    assert hasattr(model, 'align')  # DVS-specific alignment conv


def test_ptq_conversion():
    """PTQ quantized model should convert correctly to SNN."""
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.qann.quantization.ptq import PTQQuantizer
    from unieval.qann.operators.ptq import PTQQuan
    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.operators.neurons import IFNeuron
    from unieval.ann.models.vit import VisionTransformer

    model = VisionTransformer(
        img_size=32, patch_size=8, in_chans=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.,
        qkv_bias=True, act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=True,
    )
    quantizer = PTQQuantizer(level=16, is_softmax=True)
    model = quantizer.quantize_model(model)

    # Verify PTQQuan modules exist
    found_ptq = any(isinstance(m, PTQQuan) for m in model.modules())
    assert found_ptq, "PTQQuan not found after PTQ quantization"

    # Convert to SNN — the merged rule should handle PTQQuan
    converter = SNNConverter()
    converter.convert(model, level=16, neuron_type="ST-BIF", is_softmax=True)

    found_if = any(isinstance(m, IFNeuron) for m in model.modules())
    assert found_if, "IFNeuron not found after PTQ conversion"


# ===== Test 15: Full Pipeline =====

def test_full_pipeline():
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.ann.models.vit import vit_small_patch16
    from unieval.qann import quantize
    from unieval.snn import convert

    # Create model
    model = vit_small_patch16(
        num_classes=10, img_size=224, global_pool=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    assert model is not None

    # Quantize
    model = quantize(model, method="lsq", level=16)

    # Convert
    wrapper = convert(model, time_step=4, level=16, encoding_type="analog")
    assert isinstance(wrapper, torch.nn.Module)

    # Verify forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = wrapper.to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    output, timesteps = wrapper(x)
    assert output.shape[1] == 10
    wrapper.reset()


# ===== Main =====

if __name__ == "__main__":
    print("=" * 60)
    print("UniEval Verification Tests")
    print("=" * 60)

    print("\n--- Core Imports ---")
    run_test("core imports", test_core_imports)
    run_test("operator imports", test_operator_imports)
    run_test("quantization imports", test_quantization_imports)
    run_test("conversion imports", test_conversion_imports)
    run_test("evaluation imports", test_evaluation_imports)
    run_test("model imports", test_model_imports)
    run_test("engine imports", test_engine_imports)

    print("\n--- Registry System ---")
    run_test("registry basic", test_registry_basic)
    run_test("registry duplicate error", test_registry_duplicate_error)
    run_test("registry missing error", test_registry_missing_error)
    run_test("neuron registry", test_neuron_registry)
    run_test("quantizer registry", test_quantizer_registry)
    run_test("model profile registry", test_model_profile_registry)

    print("\n--- Config System ---")
    run_test("config defaults", test_config_defaults)
    run_test("config custom", test_config_custom)

    print("\n--- Operators ---")
    run_test("IFNeuron forward", test_ifneuron_forward)
    run_test("ORIIFNeuron forward", test_oriifneuron_forward)
    run_test("Spiking_LayerNorm", test_spiking_layernorm)
    run_test("LLLinear", test_lllinear)
    run_test("LLConv2d", test_llconv2d)
    run_test("spiking_softmax", test_spiking_softmax)
    run_test("SAttention", test_sattention)
    run_test("SpikeMaxPooling", test_spike_max_pooling)

    print("\n--- Quantization ---")
    run_test("MyQuan full precision", test_myquan_full_precision)
    run_test("MyQuan quantize", test_myquan_quantize)
    run_test("QAttention", test_qattention)
    run_test("LSQQuantizer on ViT", test_lsq_quantizer)
    run_test("PTQQuan construction", test_ptq_quan)

    print("\n--- Conversion ---")
    run_test("conversion rules", test_conversion_rules)
    run_test("SNNConverter on quantized ViT", test_snn_converter)

    print("\n--- SNNWrapper ---")
    run_test("SNNWrapper analog encoding", test_snn_wrapper_analog)

    print("\n--- Evaluation ---")
    run_test("spike_rate spiking", test_spike_rate_spiking)
    run_test("spike_rate non-spiking", test_spike_rate_non_spiking)
    run_test("spike_rate zero", test_spike_rate_zero)
    run_test("OpsCounter", test_ops_counter)
    run_test("EvalResult", test_eval_result)

    print("\n--- Model Creation ---")
    run_test("ViT small creation", test_vit_small_creation)

    print("\n--- forward_multistep Equivalence ---")
    run_test("Spiking_LayerNorm multistep", test_spiking_layernorm_multistep)
    run_test("Spiking_LayerNorm multistep from state", test_spiking_layernorm_multistep_from_state)
    run_test("SpikeMaxPooling multistep", test_spike_max_pooling_multistep)
    run_test("spiking_softmax multistep", test_spiking_softmax_multistep)
    run_test("IFNeuron multistep fallback", test_ifneuron_multistep_fallback)
    run_test("participates_in_early_stop flags", test_participates_in_early_stop_flags)
    run_test("working property defensive", test_working_property_defensive)

    print("\n--- Judger and reset_model ---")
    run_test("Judger precache", test_judger_precache)
    run_test("reset_model flat traversal", test_reset_model_flat)

    print("\n--- New SNNWrapper API ---")
    run_test("encode_sequence analog", test_encode_sequence_analog)
    run_test("encode_sequence rate", test_encode_sequence_rate)
    run_test("step_encoded API", test_step_encoded_api)
    run_test("step_encoded vs run_auto", test_step_encoded_vs_run_auto)

    print("\n--- Bug Fix Verification ---")
    run_test("MyQuan device-agnostic", test_myquan_device_agnostic)
    run_test("SAttention reset before conversion", test_sattention_reset_before_conversion)
    run_test("OpsCounter configurable timestep", test_ops_counter_configurable_timestep)
    run_test("spiking_softmax no deepcopy", test_spiking_softmax_no_deepcopy)
    run_test("ModelProfile num_patches", test_model_profile_num_patches)
    run_test("Config img_size", test_config_img_size)
    run_test("remove_softmax utility", test_remove_softmax)
    run_test("SNNWrapper no debug print", test_wrapper_no_debug_print)

    print("\n--- New Fix Verification ---")
    run_test("IFNeuron default attributes", test_ifneuron_default_attributes)
    run_test("LLConv2d multistep", test_llconv2d_multistep)
    run_test("LLLinear multistep", test_lllinear_multistep)
    run_test("SAttention multistep", test_sattention_multistep)
    run_test("adapter forward_multistep Sequential", test_adapter_forward_multistep_sequential)
    run_test("DVS model creation", test_dvs_model_creation)
    run_test("PTQ conversion", test_ptq_conversion)

    print("\n--- Full Pipeline ---")
    run_test("full pipeline", test_full_pipeline)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)

    if ERRORS:
        print("\nFailed tests:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")

    sys.exit(0 if FAIL == 0 else 1)
