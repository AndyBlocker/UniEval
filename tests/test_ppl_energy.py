#!/usr/bin/env python3
"""PPL & Energy evaluation for three model types: ViT, Qwen3, UniAffine.

Uses small random-weight models with synthetic data to verify the full
PPL + Energy pipeline works end-to-end. Real PPL values are meaningless
on random weights — this tests correctness of the evaluation plumbing.

Usage:
    python tests/test_ppl_energy.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from functools import partial

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


# ===== Helpers =====

def _small_qwen3():
    from unieval.ANN.models.qwen3 import Qwen3Model, Qwen3Config
    cfg = Qwen3Config(
        vocab_size=256, num_layers=2, hidden_size=64,
        ffn_hidden_size=128, num_heads=4, num_kv_heads=2,
        head_dim=16, max_seq_len=32,
    )
    return Qwen3Model(cfg)


def _small_uniaffine():
    from unieval.ANN.models.uniaffine import UniAffineModel, UniAffineConfig
    cfg = UniAffineConfig(
        vocab_size=256, num_layers=2, hidden_size=64,
        ffn_hidden_size=128, num_heads=4, num_kv_heads=2,
        head_dim=16, max_seq_len=32,
    )
    return UniAffineModel(cfg)


def _small_vit():
    from unieval.ANN.models.vit import vit_small_patch16
    return vit_small_patch16(
        num_classes=10,
        global_pool=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def _decoder_dataloader(vocab_size=256, seq_len=16, batch_size=2, num_batches=3):
    """Synthetic dataloader for decoder PPL: yields (input_ids, input_ids)."""
    data = []
    for _ in range(num_batches):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append((ids, ids))
    return data


def _vit_dataloader(batch_size=2, num_batches=3):
    """Synthetic dataloader for ViT: yields (images, labels)."""
    data = []
    for _ in range(num_batches):
        imgs = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 10, (batch_size,))
        data.append((imgs, labels))
    return data


# ===== Test 1: Decoder PPL (ANN) =====

def test_qwen3_ann_ppl():
    from unieval.Evaluation.benchmarks.perplexity import PerplexityEvaluator

    model = _small_qwen3().eval()
    dl = _decoder_dataloader()
    evaluator = PerplexityEvaluator(num_batches=3)
    result = evaluator.evaluate(model, dl)

    ppl = result.metrics["perplexity"]
    loss = result.metrics["avg_loss"]
    print(f"    Qwen3 ANN: PPL={ppl:.2f}, loss={loss:.4f}")
    assert ppl > 1.0, f"PPL should be > 1, got {ppl}"
    assert loss > 0, f"Loss should be > 0, got {loss}"


def test_uniaffine_ann_ppl():
    from unieval.Evaluation.benchmarks.perplexity import PerplexityEvaluator

    model = _small_uniaffine().eval()
    dl = _decoder_dataloader()
    evaluator = PerplexityEvaluator(num_batches=3)
    result = evaluator.evaluate(model, dl)

    ppl = result.metrics["perplexity"]
    loss = result.metrics["avg_loss"]
    print(f"    UniAffine ANN: PPL={ppl:.2f}, loss={loss:.4f}")
    assert ppl > 1.0
    assert loss > 0


# ===== Test 2: Decoder PPL (QANN) =====

def test_qwen3_qann_ppl():
    from unieval.Evaluation.benchmarks.perplexity import PerplexityEvaluator
    from unieval.QANN.quantization.base import BaseQuantizer
    from unieval.QANN.quantization.qwen3_rules import QWEN3_PTQ_RULES

    model = _small_qwen3().eval()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    dl = _decoder_dataloader()
    evaluator = PerplexityEvaluator(num_batches=3)
    result = evaluator.evaluate(model, dl)

    ppl = result.metrics["perplexity"]
    loss = result.metrics["avg_loss"]
    print(f"    Qwen3 QANN: PPL={ppl:.2f}, loss={loss:.4f}")
    assert ppl > 1.0
    assert loss > 0


def test_uniaffine_qann_ppl():
    from unieval.Evaluation.benchmarks.perplexity import PerplexityEvaluator
    from unieval.QANN.quantization.base import BaseQuantizer
    from unieval.QANN.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES

    model = _small_uniaffine().eval()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, UNIAFFINE_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    dl = _decoder_dataloader()
    evaluator = PerplexityEvaluator(num_batches=3)
    result = evaluator.evaluate(model, dl)

    ppl = result.metrics["perplexity"]
    loss = result.metrics["avg_loss"]
    print(f"    UniAffine QANN: PPL={ppl:.2f}, loss={loss:.4f}")
    assert ppl > 1.0
    assert loss > 0


# ===== Test 3: Decoder PPL (SNN via SNNWrapper) =====

def test_qwen3_snn_ppl():
    from unieval.Evaluation.benchmarks.perplexity import PerplexityEvaluator
    from unieval.QANN.quantization.base import BaseQuantizer
    from unieval.QANN.quantization.qwen3_rules import QWEN3_PTQ_RULES
    from unieval.SNN.snnConverter.wrapper import SNNWrapper

    model = _small_qwen3().eval()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    # Calibrate
    dl = _decoder_dataloader()
    with torch.no_grad():
        for ids, _ in dl:
            model(ids)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=32,
        encoding_type="analog",
        level=32,
        neuron_type="ST-BIF",
        model_name="qwen3",
        is_softmax=True,
    )

    dl2 = _decoder_dataloader()
    evaluator = PerplexityEvaluator(num_batches=2)
    result = evaluator.evaluate(wrapper, dl2)

    ppl = result.metrics["perplexity"]
    loss = result.metrics["avg_loss"]
    print(f"    Qwen3 SNN: PPL={ppl:.2f}, loss={loss:.4f}")
    assert ppl > 1.0


def test_uniaffine_snn_ppl():
    from unieval.Evaluation.benchmarks.perplexity import PerplexityEvaluator
    from unieval.QANN.quantization.base import BaseQuantizer
    from unieval.QANN.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES
    from unieval.SNN.snnConverter.wrapper import SNNWrapper

    model = _small_uniaffine().eval()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, UNIAFFINE_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    dl = _decoder_dataloader()
    with torch.no_grad():
        for ids, _ in dl:
            model(ids)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=32,
        encoding_type="analog",
        level=32,
        neuron_type="ST-BIF",
        model_name="uniaffine",
        is_softmax=False,
    )

    dl2 = _decoder_dataloader()
    evaluator = PerplexityEvaluator(num_batches=2)
    result = evaluator.evaluate(wrapper, dl2)

    ppl = result.metrics["perplexity"]
    loss = result.metrics["avg_loss"]
    print(f"    UniAffine SNN: PPL={ppl:.2f}, loss={loss:.4f}")
    assert ppl > 1.0


# ===== Test 4: Energy & OpsCounter =====

def test_qwen3_snn_energy():
    from unieval.QANN.quantization.base import BaseQuantizer
    from unieval.QANN.quantization.qwen3_rules import QWEN3_PTQ_RULES
    from unieval.SNN.snnConverter.wrapper import SNNWrapper
    from unieval.Evaluation.energy.ops_counter import OpsCounter
    from unieval.Evaluation.energy.energy import EnergyEvaluator
    from unieval.config import EnergyConfig
    from unieval.ANN.models.base import DecoderModelProfile

    model = _small_qwen3().eval()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    dl = _decoder_dataloader()
    with torch.no_grad():
        for ids, _ in dl:
            model(ids)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=32,
        encoding_type="analog",
        level=32,
        neuron_type="ST-BIF",
        model_name="qwen3",
        is_softmax=True,
    )

    profile = DecoderModelProfile(
        depth=2, num_heads=4, embed_dim=64,
        num_kv_heads=2, seq_len=16, head_dim=16,
        ffn_hidden_size=128, time_steps=32,
        patch_size=1, img_size=1,
    )

    ops_counter = OpsCounter(time_step=32)
    energy_cfg = EnergyConfig()
    evaluator = EnergyEvaluator(
        energy_config=energy_cfg,
        model_profile=profile,
        ops_counter=ops_counter,
        num_batches=2,
    )

    dl2 = _decoder_dataloader()
    result = evaluator.evaluate(wrapper, dl2)

    mac_g = result.metrics["mac_ops_G"]
    ac_g = result.metrics["ac_ops_G"]
    e_mac = result.metrics["e_mac_mJ"]
    e_ac = result.metrics["e_ac_mJ"]
    e_total = result.metrics["energy_mJ"]

    print(f"    Qwen3 SNN Energy:")
    print(f"      MAC: {mac_g:.6f} G-ops, E_MAC: {e_mac:.6f} mJ")
    print(f"      AC (SOP): {ac_g:.6f} G-ops, E_AC: {e_ac:.6f} mJ")
    print(f"      Total: {e_total:.6f} mJ")
    print(f"      SSA QKV firing rates: {result.details.get('ssa_qkv_firing_rates', [])}")

    assert mac_g >= 0
    assert ac_g >= 0
    assert e_total >= 0
    # At least some ops should be counted
    assert mac_g + ac_g > 0, "No ops counted at all!"


def test_uniaffine_snn_energy():
    from unieval.QANN.quantization.base import BaseQuantizer
    from unieval.QANN.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES
    from unieval.SNN.snnConverter.wrapper import SNNWrapper
    from unieval.Evaluation.energy.ops_counter import OpsCounter
    from unieval.Evaluation.energy.energy import EnergyEvaluator
    from unieval.config import EnergyConfig
    from unieval.ANN.models.base import DecoderModelProfile

    model = _small_uniaffine().eval()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, UNIAFFINE_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    dl = _decoder_dataloader()
    with torch.no_grad():
        for ids, _ in dl:
            model(ids)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=32,
        encoding_type="analog",
        level=32,
        neuron_type="ST-BIF",
        model_name="uniaffine",
        is_softmax=False,
    )

    profile = DecoderModelProfile(
        depth=2, num_heads=4, embed_dim=64,
        num_kv_heads=2, seq_len=16, head_dim=16,
        ffn_hidden_size=128, time_steps=32,
        patch_size=1, img_size=1,
    )

    ops_counter = OpsCounter(time_step=32)
    energy_cfg = EnergyConfig()
    evaluator = EnergyEvaluator(
        energy_config=energy_cfg,
        model_profile=profile,
        ops_counter=ops_counter,
        num_batches=2,
    )

    dl2 = _decoder_dataloader()
    result = evaluator.evaluate(wrapper, dl2)

    mac_g = result.metrics["mac_ops_G"]
    ac_g = result.metrics["ac_ops_G"]
    e_mac = result.metrics["e_mac_mJ"]
    e_ac = result.metrics["e_ac_mJ"]
    e_total = result.metrics["energy_mJ"]

    print(f"    UniAffine SNN Energy:")
    print(f"      MAC: {mac_g:.6f} G-ops, E_MAC: {e_mac:.6f} mJ")
    print(f"      AC (SOP): {ac_g:.6f} G-ops, E_AC: {e_ac:.6f} mJ")
    print(f"      Total: {e_total:.6f} mJ")
    print(f"      SSA QKV firing rates: {result.details.get('ssa_qkv_firing_rates', [])}")

    assert mac_g >= 0
    assert ac_g >= 0
    assert e_total >= 0
    assert mac_g + ac_g > 0, "No ops counted at all!"


def test_vit_snn_energy():
    from unieval.QANN.quantization.lsq import LSQQuantizer
    from unieval.SNN.snnConverter.wrapper import SNNWrapper
    from unieval.Evaluation.energy.ops_counter import OpsCounter
    from unieval.Evaluation.energy.energy import EnergyEvaluator
    from unieval.config import EnergyConfig
    from unieval.ANN.models.base import ModelProfile

    model = _small_vit().eval()
    quantizer = LSQQuantizer(level=16, is_softmax=True)
    quantizer.quantize_model(model)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=4,
        encoding_type="analog",
        level=16,
        neuron_type="ST-BIF",
        model_name="vit_small",
        is_softmax=True,
    )

    profile = ModelProfile(
        depth=12, num_heads=6, embed_dim=384,
        patch_size=16, time_steps=4,
    )

    ops_counter = OpsCounter(time_step=4)
    evaluator = EnergyEvaluator(
        energy_config=EnergyConfig(),
        model_profile=profile,
        ops_counter=ops_counter,
        num_batches=1,
    )

    dl = _vit_dataloader(batch_size=1, num_batches=1)
    result = evaluator.evaluate(wrapper, dl)

    mac_g = result.metrics["mac_ops_G"]
    ac_g = result.metrics["ac_ops_G"]
    e_total = result.metrics["energy_mJ"]

    print(f"    ViT SNN Energy:")
    print(f"      MAC: {mac_g:.6f} G-ops")
    print(f"      AC (SOP): {ac_g:.6f} G-ops")
    print(f"      Total: {e_total:.6f} mJ")

    assert e_total >= 0
    assert mac_g + ac_g > 0


# ===== Test 5: OpsCounter hook coverage =====

def test_decoder_ops_hook_coverage():
    """Verify that OpsCounter can attach hooks to all decoder SNN module types."""
    from unieval.QANN.quantization.base import BaseQuantizer
    from unieval.QANN.quantization.qwen3_rules import QWEN3_PTQ_RULES
    from unieval.SNN.snnConverter.wrapper import SNNWrapper
    from unieval.Evaluation.energy.ops_counter import OpsCounter

    model = _small_qwen3().eval()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    dl = _decoder_dataloader()
    with torch.no_grad():
        for ids, _ in dl:
            model(ids)

    wrapper = SNNWrapper(
        ann_model=model,
        time_step=32,
        encoding_type="analog",
        level=32,
        neuron_type="ST-BIF",
        model_name="qwen3",
        is_softmax=True,
    )

    counter = OpsCounter(time_step=32)

    # Check which modules are supported
    supported_types = set()
    unsupported_types = set()
    for name, module in wrapper.model.named_modules():
        if counter.is_supported(module):
            supported_types.add(type(module).__name__)
        else:
            t = type(module).__name__
            # Skip container types and trivial modules
            if t not in ("Module", "ModuleList", "Sequential", "Identity",
                         "Qwen3Model", "Qwen3Block", "SQwen3Attention",
                         "spiking_softmax", "RotaryEmbedding",
                         "Embedding", "RMSNorm"):
                unsupported_types.add(t)

    print(f"    Supported: {sorted(supported_types)}")
    if unsupported_types:
        print(f"    Unsupported (non-trivial): {sorted(unsupported_types)}")

    # Key types that must be covered
    assert "STBIFNeuron" in supported_types, "STBIFNeuron should have hook"
    assert "Linear" in supported_types, "nn.Linear should have hook"
    assert "Spiking_RMSNorm" in supported_types, "Spiking_RMSNorm should have hook"
    assert "Spiking_SwiGLUMlp" in supported_types, "Spiking_SwiGLUMlp should have hook"
    assert "Spiking_SiLU" in supported_types, "Spiking_SiLU should have hook"


# ===== Main =====

if __name__ == "__main__":
    print("=" * 60)
    print("PPL & Energy Evaluation Tests")
    print("=" * 60)

    print("\n--- OpsCounter Hook Coverage ---")
    run_test("Decoder ops hook coverage", test_decoder_ops_hook_coverage)

    print("\n--- Decoder ANN PPL ---")
    run_test("Qwen3 ANN PPL", test_qwen3_ann_ppl)
    run_test("UniAffine ANN PPL", test_uniaffine_ann_ppl)

    print("\n--- Decoder QANN PPL ---")
    run_test("Qwen3 QANN PPL", test_qwen3_qann_ppl)
    run_test("UniAffine QANN PPL", test_uniaffine_qann_ppl)

    print("\n--- Decoder SNN PPL ---")
    run_test("Qwen3 SNN PPL", test_qwen3_snn_ppl)
    run_test("UniAffine SNN PPL", test_uniaffine_snn_ppl)

    print("\n--- SNN Energy (MAC vs SOP) ---")
    run_test("Qwen3 SNN Energy", test_qwen3_snn_energy)
    run_test("UniAffine SNN Energy", test_uniaffine_snn_energy)
    run_test("ViT SNN Energy", test_vit_snn_energy)

    print()
    print("=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)

    if ERRORS:
        print("\nFailed tests:")
        for name, e in ERRORS:
            print(f"  - {name}: {e}")
        sys.exit(1)
