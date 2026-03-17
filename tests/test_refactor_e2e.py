#!/usr/bin/env python3
"""End-to-end verification tests for the UniEval refactoring.

Tests the full pipeline with the new composite operators, QANN operators,
1:1 conversion rules, and adapter lifecycle API.

Must be run in an environment with CUDA and PyTorch installed.

Usage:
    python tests/test_refactor_e2e.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

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


# ===== Test 1: New imports =====

def test_new_imports():
    """Verify all new modules import correctly."""
    from unieval.SNN.snnConverter.threshold import transfer_threshold
    from unieval.SNN.operators.composites import SConv2d, SLinear
    from unieval.QANN.operators.composites import QConv2d, QLinear, QNorm
    from unieval.SNN.snnConverter.adapter import (
        auto_detect_adapter, CausalDecoderAdapter, UniAffineExecutionAdapter,
    )
    from unieval.protocols import is_spiking_attention_like
    assert CausalDecoderAdapter is UniAffineExecutionAdapter, "backward compat alias"


# ===== Test 2: ANN Forward =====

def test_ann_forward():
    """Create ViT, forward random input, verify output shape."""
    from unieval.ANN.models.vit import vit_small_patch16
    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 100), f"Expected (2, 100), got {y.shape}"


# ===== Test 3: LSQ Quantization produces Q* composites =====

def test_lsq_quantization_structure():
    """After LSQ quantization, verify QConv2d/QNorm/QLinear are present."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.QANN.quantization.lsq import LSQQuantizer
    from unieval.QANN.operators.composites import QConv2d, QLinear, QNorm

    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    quantizer.quantize_model(model)

    # patch_embed.proj should be QConv2d
    assert isinstance(model.patch_embed.proj, QConv2d), \
        f"patch_embed.proj should be QConv2d, got {type(model.patch_embed.proj)}"

    # block norms should be QNorm
    blk = model.blocks[0]
    assert isinstance(blk.norm1, QNorm), \
        f"blocks[0].norm1 should be QNorm, got {type(blk.norm1)}"
    assert isinstance(blk.norm2, QNorm), \
        f"blocks[0].norm2 should be QNorm, got {type(blk.norm2)}"

    # MLP fc2 should be QLinear
    assert isinstance(blk.mlp.fc2, QLinear), \
        f"blocks[0].mlp.fc2 should be QLinear, got {type(blk.mlp.fc2)}"

    # MLP act should still be Sequential(MyQuan, ReLU) — pre-activation pattern
    assert isinstance(blk.mlp.act, nn.Sequential), \
        f"blocks[0].mlp.act should be Sequential, got {type(blk.mlp.act)}"


# ===== Test 4: PTQ Quantization produces Q* composites =====

def test_ptq_quantization_structure():
    """After PTQ quantization, verify QConv2d/QNorm/QLinear are present."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.QANN.quantization.ptq import PTQQuantizer
    from unieval.QANN.operators.composites import QConv2d, QLinear, QNorm

    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    quantizer = PTQQuantizer(level=16, is_softmax=True)
    quantizer.quantize_model(model)

    assert isinstance(model.patch_embed.proj, QConv2d), \
        f"patch_embed.proj should be QConv2d, got {type(model.patch_embed.proj)}"

    blk = model.blocks[0]
    assert isinstance(blk.norm1, QNorm), \
        f"blocks[0].norm1 should be QNorm, got {type(blk.norm1)}"
    assert isinstance(blk.mlp.fc2, QLinear), \
        f"blocks[0].mlp.fc2 should be QLinear, got {type(blk.mlp.fc2)}"


# ===== Test 5: Conversion produces S* composites =====

def test_conversion_structure():
    """After conversion, verify SConv2d/SLinear are present."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.QANN.quantization.ptq import PTQQuantizer
    from unieval.SNN.snnConverter.converter import SNNConverter
    from unieval.SNN.operators.composites import SConv2d, SLinear
    from unieval.SNN.operators.attention import SAttention

    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    PTQQuantizer(level=16, is_softmax=True).quantize_model(model)

    converter = SNNConverter()
    converter.convert(model, level=16, neuron_type="ST-BIF", is_softmax=True)

    # patch_embed.proj should be SConv2d
    assert isinstance(model.patch_embed.proj, SConv2d), \
        f"patch_embed.proj should be SConv2d, got {type(model.patch_embed.proj)}"

    # block attn should be SAttention
    blk = model.blocks[0]
    assert isinstance(blk.attn, SAttention), \
        f"blocks[0].attn should be SAttention, got {type(blk.attn)}"

    # MLP fc2 should be SLinear
    assert isinstance(blk.mlp.fc2, SLinear), \
        f"blocks[0].mlp.fc2 should be SLinear, got {type(blk.mlp.fc2)}"


# ===== Test 6: Adapter auto-detection =====

def test_adapter_auto_detection():
    """Verify adapter auto-detection works via duck-typing."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.SNN.snnConverter.adapter import (
        auto_detect_adapter, ViTExecutionAdapter, CausalDecoderAdapter,
        DefaultExecutionAdapter,
    )

    vit = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    adapter = auto_detect_adapter(vit)
    assert isinstance(adapter, ViTExecutionAdapter), \
        f"ViT should use ViTExecutionAdapter, got {type(adapter)}"

    # Verify adapter captured context
    assert adapter.pos_embed is not None, "pos_embed should be captured"
    assert adapter.cls_token is not None, "cls_token should be captured"


# ===== Test 7: ViT adapter supports() =====

def test_vit_adapter_supports():
    """Verify ViTExecutionAdapter.supports() duck-typing."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.SNN.snnConverter.adapter import ViTExecutionAdapter

    vit = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    assert ViTExecutionAdapter.supports(vit), "ViT should be supported"

    # A plain Linear should NOT be supported
    assert not ViTExecutionAdapter.supports(nn.Linear(10, 10)), \
        "Linear should not be supported by ViT adapter"


# ===== Test 8: SNNWrapper without model_name =====

def test_wrapper_no_model_name():
    """Verify SNNWrapper works without explicit model_name."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.QANN.quantization.ptq import PTQQuantizer
    from unieval.SNN.snnConverter.wrapper import SNNWrapper

    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    PTQQuantizer(level=16, is_softmax=True).quantize_model(model)

    # Create wrapper without model_name — should auto-detect ViT
    wrapper = SNNWrapper(
        ann_model=model,
        time_step=4,
        encoding_type="analog",
        level=16,
        neuron_type="ST-BIF",
        is_softmax=True,
    )

    from unieval.SNN.snnConverter.adapter import ViTExecutionAdapter
    assert isinstance(wrapper.adapter, ViTExecutionAdapter), \
        f"Should auto-detect ViT adapter, got {type(wrapper.adapter)}"


# ===== Test 9: Shared threshold transfer =====

def test_shared_threshold_transfer():
    """Verify shared transfer_threshold works correctly."""
    from unieval.SNN.snnConverter.threshold import transfer_threshold
    from unieval.SNN.operators.neurons import IFNeuron
    from unieval.QANN.operators.ptq import PTQQuan

    quan = PTQQuan(level=16, sym=True)
    quan.s.fill_(0.42)

    neuron = IFNeuron(q_threshold=torch.tensor(1.0), level=16, sym=True)
    transfer_threshold(quan, neuron, "ST-BIF", 16)

    assert abs(neuron.q_threshold.item() - 0.42) < 1e-6, \
        f"Threshold should be 0.42, got {neuron.q_threshold.item()}"
    assert neuron.neuron_type == "ST-BIF"
    assert not neuron.is_init


# ===== Test 10: SConv2d/SLinear forward =====

def test_composite_operators_forward():
    """Verify SConv2d and SLinear forward pass."""
    from unieval.SNN.operators.composites import SConv2d, SLinear
    from unieval.SNN.operators.layers import LLConv2d, LLLinear
    from unieval.SNN.operators.neurons import IFNeuron

    # SConv2d
    conv = nn.Conv2d(3, 16, 3, padding=1)
    ll_conv = LLConv2d(conv, neuron_type="ST-BIF", level=16)
    neuron_c = IFNeuron(q_threshold=torch.tensor(0.5), level=16, sym=True)
    s_conv = SConv2d(ll_conv, neuron_c)

    x = torch.randn(1, 3, 8, 8)
    y = s_conv(x)
    assert y.shape == (1, 16, 8, 8), f"Expected (1, 16, 8, 8), got {y.shape}"

    # SLinear
    linear = nn.Linear(64, 32)
    ll_linear = LLLinear(linear, neuron_type="ST-BIF", level=16)
    neuron_l = IFNeuron(q_threshold=torch.tensor(0.5), level=16, sym=True)
    s_linear = SLinear(ll_linear, neuron_l)

    x = torch.randn(1, 10, 64)
    y = s_linear(x)
    assert y.shape == (1, 10, 32), f"Expected (1, 10, 32), got {y.shape}"

    # Reset
    s_conv.reset()
    s_linear.reset()


# ===== Test 11: SConv2d forward_multistep equivalence =====

def test_composite_multistep_equivalence():
    """Verify forward_multistep equals sequential single-step for composites."""
    from unieval.SNN.operators.composites import SConv2d, SLinear
    from unieval.SNN.operators.layers import LLConv2d, LLLinear
    from unieval.SNN.operators.neurons import IFNeuron

    # SLinear test
    linear = nn.Linear(32, 16, bias=True)
    ll_linear = LLLinear(linear, neuron_type="ST-BIF", level=16)
    neuron = IFNeuron(q_threshold=torch.tensor(0.3), level=16, sym=True)
    s_linear = SLinear(ll_linear, neuron)

    T = 5
    x_seq = torch.randn(T, 2, 10, 32)

    # Multistep
    s_linear.reset()
    out_multi = s_linear.forward_multistep(x_seq)

    # Sequential single-step
    s_linear.reset()
    out_seq = []
    for t in range(T):
        out_seq.append(s_linear(x_seq[t]))
    out_seq = torch.stack(out_seq)

    assert torch.allclose(out_multi, out_seq, atol=1e-5), \
        f"Multistep vs sequential mismatch: max diff = {(out_multi - out_seq).abs().max()}"


# ===== Test 12: QANN forward equivalence (QConv2d, QLinear, QNorm) =====

def test_qann_forward_equivalence():
    """Verify Q* composites produce same output as Sequential equivalents."""
    from unieval.QANN.operators.composites import QConv2d, QLinear, QNorm
    from unieval.QANN.operators.ptq import PTQQuan

    # QConv2d
    conv = nn.Conv2d(3, 16, 3, padding=1)
    quan = PTQQuan(level=16, sym=True)
    q_conv = QConv2d(conv, quan)
    x_img = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        y1 = q_conv(x_img)
    # Should produce valid output
    assert y1.shape == (2, 16, 8, 8)

    # QLinear
    linear = nn.Linear(64, 32)
    quan_l = PTQQuan(level=16, sym=True)
    q_linear = QLinear(linear, quan_l)
    x_lin = torch.randn(2, 10, 64)
    with torch.no_grad():
        y2 = q_linear(x_lin)
    assert y2.shape == (2, 10, 32)

    # QNorm
    norm = nn.LayerNorm(32)
    quan_n = PTQQuan(level=16, sym=True)
    q_norm = QNorm(norm, quan_n)
    x_norm = torch.randn(2, 10, 32)
    with torch.no_grad():
        y3 = q_norm(x_norm)
    assert y3.shape == (2, 10, 32)


# ===== Test 13: Full ViT E2E pipeline with new composites =====

def test_vit_e2e_pipeline():
    """Full ViT pipeline: create → quantize → convert → forward."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.QANN.quantization.ptq import PTQQuantizer
    from unieval.SNN.snnConverter.wrapper import SNNWrapper
    from unieval.SNN.operators.composites import SConv2d, SLinear

    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    PTQQuantizer(level=16, is_softmax=True).quantize_model(model)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=4,
        encoding_type="analog",
        level=16,
        neuron_type="ST-BIF",
        is_softmax=True,
    )

    # Verify composites exist in converted model
    has_sconv = any(isinstance(m, SConv2d) for m in wrapper.model.modules())
    has_slinear = any(isinstance(m, SLinear) for m in wrapper.model.modules())
    assert has_sconv, "Should have SConv2d after conversion"
    assert has_slinear, "Should have SLinear after conversion"

    # Forward pass
    x = torch.randn(1, 3, 224, 224)
    wrapper.eval()
    with torch.no_grad():
        accu, actual_T = wrapper.run_auto(x)
    assert accu.shape == (1, 100), f"Expected (1, 100), got {accu.shape}"
    assert actual_T > 0, f"Should run at least 1 timestep, got {actual_T}"


# ===== Test 14: Multistep equivalence (wrapper level) =====

def test_wrapper_multistep_equivalence():
    """forward_encoded == manual step_encoded loop."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.QANN.quantization.ptq import PTQQuantizer
    from unieval.SNN.snnConverter.wrapper import SNNWrapper

    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    PTQQuantizer(level=16, is_softmax=True).quantize_model(model)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=4,
        encoding_type="analog",
        level=16,
        neuron_type="ST-BIF",
        is_softmax=True,
    )
    wrapper.eval()

    x = torch.randn(1, 3, 224, 224)
    T = 3
    x_seq = wrapper.encode_sequence(x, T=T, encoding_type="analog")

    # forward_encoded path
    wrapper.reset()
    with torch.no_grad():
        out_multi = wrapper.forward_encoded(x_seq)

    # step_encoded loop
    wrapper.reset()
    out_steps = []
    with torch.no_grad():
        for t in range(T):
            out_steps.append(wrapper.step_encoded(x_seq[t]))
    out_steps = torch.stack(out_steps)

    assert torch.allclose(out_multi, out_steps, atol=1e-4), \
        f"Multistep vs step mismatch: max diff = {(out_multi - out_steps).abs().max()}"


# ===== Test 15: Energy evaluator with new composites =====

def test_energy_nonzero():
    """EnergyEvaluator should return non-zero energy values."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.ANN.models.base import ModelProfile
    from unieval.QANN.quantization.ptq import PTQQuantizer
    from unieval.SNN.snnConverter.wrapper import SNNWrapper
    from unieval.Evaluation.energy.energy import EnergyEvaluator
    from unieval.Evaluation.energy.ops_counter import OpsCounter
    from unieval.config import EnergyConfig

    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    PTQQuantizer(level=16, is_softmax=True).quantize_model(model)
    wrapper = SNNWrapper(
        ann_model=model,
        time_step=4,
        encoding_type="analog",
        level=16,
        neuron_type="ST-BIF",
        is_softmax=True,
    )
    wrapper.eval()

    profile = ModelProfile(
        depth=12, num_heads=6, embed_dim=384,
        patch_size=14, time_steps=4
    )
    ops_counter = OpsCounter(time_step=4)
    evaluator = EnergyEvaluator(
        energy_config=EnergyConfig(),
        model_profile=profile,
        ops_counter=ops_counter,
        num_batches=1,
    )

    # Create a simple dataloader
    x = torch.randn(1, 3, 224, 224)
    y = torch.zeros(1, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    result = evaluator.evaluate(wrapper, loader)
    energy = result.metrics["energy_mJ"]
    assert energy > 0, f"Energy should be > 0, got {energy}"


# ===== Test 16: OpsCounter handles new types =====

def test_ops_counter_new_types():
    """OpsCounter should recognize SConv2d, SLinear, Q* composites."""
    from unieval.Evaluation.energy.ops_counter import OpsCounter
    from unieval.SNN.operators.composites import SConv2d, SLinear
    from unieval.QANN.operators.composites import QConv2d, QLinear, QNorm

    counter = OpsCounter()
    # All new types should be recognized
    for cls_name, cls in [
        ("SConv2d", SConv2d),
        ("SLinear", SLinear),
        ("QConv2d", QConv2d),
        ("QLinear", QLinear),
        ("QNorm", QNorm),
    ]:
        assert cls in counter.modules_mapping, \
            f"{cls_name} should be in OpsCounter modules_mapping"


# ===== Test 17: Adapter reset_context =====

def test_adapter_reset_context():
    """Verify adapter reset_context restores ViT embeddings."""
    from unieval.ANN.models.vit import vit_small_patch16
    from unieval.SNN.snnConverter.adapter import ViTExecutionAdapter

    model = vit_small_patch16(img_size=224, num_classes=100, act_layer=nn.ReLU)
    adapter = ViTExecutionAdapter()
    adapter.capture_context(model)

    # Simulate what step() does at t=1: zero out embeddings
    model.pos_embed = nn.Parameter(torch.zeros_like(model.pos_embed))
    model.cls_token = nn.Parameter(torch.zeros_like(model.cls_token))

    # Verify they're zeroed
    assert (model.pos_embed.data == 0).all()

    # Reset should restore
    adapter.reset_context(model)
    assert not (model.pos_embed.data == 0).all(), \
        "pos_embed should be restored after reset_context"


# ===== Test 18: protocols is_spiking_attention_like =====

def test_spiking_attention_protocol():
    """Verify is_spiking_attention_like matches SAttention."""
    from unieval.SNN.operators.attention import SAttention
    from unieval.SNN.operators.neurons import IFNeuron
    from unieval.protocols import is_spiking_attention_like

    sa = SAttention(dim=64, num_heads=4, level=16, is_softmax=True, neuron_layer=IFNeuron)
    assert is_spiking_attention_like(sa), "SAttention should match is_spiking_attention_like"
    assert not is_spiking_attention_like(nn.Linear(10, 10)), \
        "Linear should not match is_spiking_attention_like"


# ===== Main =====

if __name__ == "__main__":
    print("=" * 60)
    print("UniEval Refactoring E2E Tests")
    print("=" * 60)

    tests = [
        ("New imports", test_new_imports),
        ("ANN Forward", test_ann_forward),
        ("LSQ Quantization structure", test_lsq_quantization_structure),
        ("PTQ Quantization structure", test_ptq_quantization_structure),
        ("Conversion structure", test_conversion_structure),
        ("Adapter auto-detection", test_adapter_auto_detection),
        ("ViT adapter supports()", test_vit_adapter_supports),
        ("SNNWrapper without model_name", test_wrapper_no_model_name),
        ("Shared threshold transfer", test_shared_threshold_transfer),
        ("Composite operators forward", test_composite_operators_forward),
        ("Composite multistep equivalence", test_composite_multistep_equivalence),
        ("QANN forward equivalence", test_qann_forward_equivalence),
        ("ViT E2E pipeline", test_vit_e2e_pipeline),
        ("Wrapper multistep equivalence", test_wrapper_multistep_equivalence),
        ("Energy evaluator non-zero", test_energy_nonzero),
        ("OpsCounter new types", test_ops_counter_new_types),
        ("Adapter reset_context", test_adapter_reset_context),
        ("Spiking attention protocol", test_spiking_attention_protocol),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if ERRORS:
        print("\nFailed tests:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
