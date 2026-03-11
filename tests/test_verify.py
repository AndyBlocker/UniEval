#!/usr/bin/env python3
"""UniEval Verification Test Script.

This script must be run in an environment with CUDA, PyTorch, and timm installed.
It verifies the correctness of the UniEval framework by testing:

1. All module imports
2. Registry system
3. Config system
4. Operator instantiation and forward pass
5. Quantization pipeline (LSQ & PTQ)
6. SNN conversion pipeline
7. SNNWrapper temporal inference
8. Evaluation components (OpsCounter, spike_rate)
9. Full pipeline integration

Usage:
    python tests/test_verify.py

Requirements:
    - Python >= 3.8
    - PyTorch >= 1.10 (with CUDA)
    - timm == 0.3.2
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
    from unieval.operators.base import SNNOperator
    from unieval.operators.neurons import IFNeuron, ORIIFNeuron
    from unieval.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm, SpikeMaxPooling
    from unieval.operators.attention import SAttention, spiking_softmax, multi, multi1


def test_quantization_imports():
    from unieval.quantization.base import BaseQuantizer, QuantPlacementRule
    from unieval.quantization.lsq import MyQuan, QAttention, QuanConv2d, QuanLinear, LSQQuantizer
    from unieval.quantization.ptq import PTQQuan, PTQQuantizer


def test_conversion_imports():
    from unieval.conversion.rules import ConversionRule, DEFAULT_CONVERSION_RULES
    from unieval.conversion.converter import SNNConverter
    from unieval.conversion.wrapper import SNNWrapper, Judger, reset_model, attn_convert


def test_evaluation_imports():
    from unieval.evaluation.base import EvalResult, BaseEvaluator
    from unieval.evaluation.spike_utils import spike_rate
    from unieval.evaluation.ops_counter import OpsCounter
    from unieval.evaluation.accuracy import AccuracyEvaluator
    from unieval.evaluation.energy import EnergyEvaluator


def test_model_imports():
    from unieval.models.base import ModelProfile
    from unieval.models.vit import VisionTransformer, VisionTransformerDVS
    from unieval.models.vit import vit_small_patch16, vit_base_patch16


def test_engine_imports():
    from unieval.engine.runner import UniEvalRunner


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
    from unieval.operators.neurons import IFNeuron

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
    from unieval.operators.neurons import ORIIFNeuron

    neuron = ORIIFNeuron(q_threshold=torch.tensor(0.5), level=16)
    x = torch.randn(2, 4).abs()  # unsigned
    y = neuron(x)
    assert y.shape == x.shape


def test_spiking_layernorm():
    import torch
    from unieval.operators.layers import Spiking_LayerNorm

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
    from unieval.operators.layers import LLLinear

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
    from unieval.operators.layers import LLConv2d

    conv = nn.Conv2d(3, 16, 3, padding=1)
    ll = LLConv2d(conv, neuron_type="ST-BIF", level=16)
    x = torch.randn(1, 3, 8, 8)
    y = ll(x)
    assert y.shape == (1, 16, 8, 8)

    ll.reset()
    assert ll.is_work == False


def test_spiking_softmax():
    import torch
    from unieval.operators.attention import spiking_softmax

    ssm = spiking_softmax()
    x = torch.randn(2, 4, 4)
    y1 = ssm(x)
    y2 = ssm(x)
    assert y1.shape == x.shape
    ssm.reset()


def test_sattention():
    import torch
    from unieval.operators.attention import SAttention

    sa = SAttention(dim=64, num_heads=4, level=16)
    x = torch.randn(1, 8, 64)
    y = sa(x)
    assert y.shape == (1, 8, 64)


def test_spike_max_pooling():
    import torch
    import torch.nn as nn
    from unieval.operators.layers import SpikeMaxPooling

    pool = SpikeMaxPooling(nn.MaxPool2d(2))
    x = torch.randn(1, 3, 4, 4)
    y = pool(x)
    assert y.shape == (1, 3, 2, 2)
    pool.reset()


# ===== Test 5: Quantization =====

def test_myquan_full_precision():
    import torch
    from unieval.quantization.lsq import MyQuan

    mq = MyQuan(level=512, sym=True)
    x = torch.randn(2, 4)
    y = mq(x)
    assert torch.allclose(x, y), "Full precision MyQuan should be identity"


def test_myquan_quantize():
    import torch
    from unieval.quantization.lsq import MyQuan

    mq = MyQuan(level=16, sym=True)
    mq.eval()  # skip init_state logic
    mq.init_state = 1  # pretend already initialized
    mq.s.data = torch.tensor(0.1)
    x = torch.randn(2, 4)
    y = mq(x)
    assert y.shape == x.shape


def test_qattention():
    import torch
    from unieval.quantization.lsq import QAttention

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
    from unieval.quantization.lsq import LSQQuantizer
    from unieval.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10,
        global_pool=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    model = quantizer.quantize_model(model)

    # Verify QAttention was placed in blocks
    from unieval.quantization.lsq import QAttention, MyQuan
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
    from unieval.quantization.ptq import PTQQuan

    pq = PTQQuan(level=16, sym=True)
    assert pq.calibrated == False
    # PTQ calibrates on first forward (needs valid data for KL-div)
    # Just check construction works
    assert pq.level == 16


# ===== Test 6: Conversion =====

def test_conversion_rules():
    from unieval.conversion.rules import DEFAULT_CONVERSION_RULES
    assert len(DEFAULT_CONVERSION_RULES) == 7
    names = [r.name for r in DEFAULT_CONVERSION_RULES]
    assert "qattention_to_sattention" in names
    assert "myquan_to_ifneuron" in names
    assert "relu_to_identity" in names


def test_snn_converter():
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.quantization.lsq import LSQQuantizer
    from unieval.conversion.converter import SNNConverter
    from unieval.operators.neurons import IFNeuron
    from unieval.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm
    from unieval.operators.attention import SAttention
    from unieval.models.vit import vit_small_patch16

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
    from unieval.quantization.lsq import LSQQuantizer
    from unieval.conversion.wrapper import SNNWrapper
    from unieval.models.vit import vit_small_patch16

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
    from unieval.evaluation.spike_utils import spike_rate

    # Spiking input: only {-1, 0, 1}
    x = torch.tensor([[-1.0, 0, 1, 0], [0, 0, 1, -1]])
    is_spike, rate, _ = spike_rate(x)
    assert is_spike == True
    assert 0 < rate < 1


def test_spike_rate_non_spiking():
    import torch
    from unieval.evaluation.spike_utils import spike_rate

    # Non-spiking input: continuous values
    x = torch.randn(4, 8)
    is_spike, rate, _ = spike_rate(x)
    assert is_spike == False
    assert rate == 1


def test_spike_rate_zero():
    import torch
    from unieval.evaluation.spike_utils import spike_rate

    x = torch.zeros(4, 8)
    is_spike, rate, _ = spike_rate(x)
    assert is_spike == True
    assert rate == 0


def test_ops_counter():
    import torch
    import torch.nn as nn
    from unieval.evaluation.ops_counter import OpsCounter

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
    from unieval.evaluation.base import EvalResult
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
    from unieval.models.vit import vit_small_patch16

    model = vit_small_patch16(
        num_classes=10,
        global_pool=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    assert hasattr(model, 'patch_embed')
    assert hasattr(model, 'blocks')
    assert len(model.blocks) == 12


# ===== Test 10: Full Pipeline =====

def test_full_pipeline():
    import torch
    import torch.nn as nn
    from functools import partial
    from unieval.config import UniEvalConfig, QuantConfig, ConversionConfig, EvalConfig
    from unieval.engine.runner import UniEvalRunner

    cfg = UniEvalConfig(
        model_name="vit_small",
        num_classes=10,
        quant=QuantConfig(level=16),
        conversion=ConversionConfig(level=16, time_step=4, encoding_type="analog"),
        evaluation=EvalConfig(num_batches=1),
    )
    runner = UniEvalRunner(cfg)

    # Create model
    model = runner.create_model(
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    assert model is not None

    # Quantize
    model = runner.quantize(model, quantizer_name="lsq")

    # Convert
    wrapper = runner.convert(model)
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
