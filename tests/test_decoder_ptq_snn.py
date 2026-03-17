#!/usr/bin/env python3
"""Test decoder PTQ->SNN conversion chain: quantizer placement + threshold transfer.

Verifies:
1. QQwen3Attention / QUniAffineAttention have 6 internal quantizers after PTQ
2. ANN vs QANN output similarity (quantization quality)
3. QANN vs SNN threshold transfer (all 6 thresholds != 1.0)
4. QANN vs SNN output similarity (conversion equivalence)
5. Protocols still match after PTQ replacement

Usage:
    python tests/test_decoder_ptq_snn.py
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


# ===== Helper: small model configs for fast testing =====

def _small_qwen3():
    from unieval.ann.models.qwen3 import Qwen3Model, Qwen3Config
    cfg = Qwen3Config(
        vocab_size=256, num_layers=2, hidden_size=64,
        ffn_hidden_size=128, num_heads=4, num_kv_heads=2,
        head_dim=16, max_seq_len=32,
    )
    return Qwen3Model(cfg)


def _small_uniaffine():
    from unieval.ann.models.uniaffine import UniAffineModel, UniAffineConfig
    cfg = UniAffineConfig(
        vocab_size=256, num_layers=2, hidden_size=64,
        ffn_hidden_size=128, num_heads=4, num_kv_heads=2,
        head_dim=16, max_seq_len=32,
    )
    return UniAffineModel(cfg)


# ===== Test 1: PTQ places 6 quantizers inside attention =====

def test_qwen3_ptq_quantizers():
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.qwen3_rules import QWEN3_PTQ_RULES, QQwen3Attention
    from unieval.qann.operators.ptq import PTQQuan

    model = _small_qwen3()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    # Check block 0 attention is now QQwen3Attention
    attn = model.blocks[0].attn
    assert isinstance(attn, QQwen3Attention), f"Expected QQwen3Attention, got {type(attn)}"
    # Check 6 quantizers exist
    for qname in ["quan_q", "quan_k", "quan_v", "attn_quan", "after_attn_quan", "quan_proj"]:
        q = getattr(attn, qname)
        assert isinstance(q, PTQQuan), f"{qname} is {type(q)}, expected PTQQuan"
    # Check projections are raw Linear (not Sequential)
    assert isinstance(attn.qkv_proj, nn.Linear), f"qkv_proj should be Linear, got {type(attn.qkv_proj)}"
    assert isinstance(attn.o_proj, nn.Linear), f"o_proj should be Linear, got {type(attn.o_proj)}"


def test_uniaffine_ptq_quantizers():
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES, QUniAffineAttention
    from unieval.qann.operators.ptq import PTQQuan

    model = _small_uniaffine()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, UNIAFFINE_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    attn = model.blocks[0].attn
    assert isinstance(attn, QUniAffineAttention), f"Expected QUniAffineAttention, got {type(attn)}"
    for qname in ["quan_q", "quan_k", "quan_v", "attn_quan", "after_attn_quan", "quan_proj"]:
        q = getattr(attn, qname)
        assert isinstance(q, PTQQuan), f"{qname} is {type(q)}, expected PTQQuan"
    assert isinstance(attn.qkv_proj, nn.Linear)
    assert isinstance(attn.o_proj, nn.Linear)
    # Check core is preserved
    assert hasattr(attn, "core"), "QUniAffineAttention should have core"


# ===== Test 2: Correct types after PTQ =====

def test_types_after_qwen3_ptq():
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.qwen3_rules import QWEN3_PTQ_RULES, QQwen3Attention

    model = _small_qwen3()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    assert isinstance(model.blocks[0].attn, QQwen3Attention), \
        "Attention should be QQwen3Attention after PTQ"


def test_types_after_uniaffine_ptq():
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES, QUniAffineAttention

    model = _small_uniaffine()

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, UNIAFFINE_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    assert isinstance(model.blocks[0].attn, QUniAffineAttention), \
        "Attention should be QUniAffineAttention after PTQ"


# ===== Test 3: ANN vs QANN forward (quantized model runs correctly) =====

def test_qwen3_ann_vs_qann():
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.qwen3_rules import QWEN3_PTQ_RULES
    from copy import deepcopy

    model = _small_qwen3().eval()
    input_ids = torch.randint(0, 256, (1, 16))

    with torch.no_grad():
        ann_out = model(input_ids)

    qmodel = deepcopy(model)

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(qmodel)
    qmodel.eval()

    with torch.no_grad():
        qann_out = qmodel(input_ids)

    cos_sim = torch.nn.functional.cosine_similarity(
        ann_out.flatten().unsqueeze(0),
        qann_out.flatten().unsqueeze(0),
    ).item()
    print(f"    Qwen3 ANN vs QANN cos_sim = {cos_sim:.4f}")
    assert cos_sim > 0.90, f"Qwen3 QANN too different from ANN: cos_sim={cos_sim:.4f}"


def test_uniaffine_ann_vs_qann():
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES
    from copy import deepcopy

    model = _small_uniaffine().eval()
    input_ids = torch.randint(0, 256, (1, 16))

    with torch.no_grad():
        ann_out = model(input_ids)

    qmodel = deepcopy(model)

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, UNIAFFINE_PTQ_RULES, level=32)
            return m

    Q().quantize_model(qmodel)
    qmodel.eval()

    with torch.no_grad():
        qann_out = qmodel(input_ids)

    cos_sim = torch.nn.functional.cosine_similarity(
        ann_out.flatten().unsqueeze(0),
        qann_out.flatten().unsqueeze(0),
    ).item()
    print(f"    UniAffine ANN vs QANN cos_sim = {cos_sim:.4f}")
    assert cos_sim > 0.90, f"UniAffine QANN too different from ANN: cos_sim={cos_sim:.4f}"


# ===== Test 4: QANN -> SNN threshold transfer =====

def test_qwen3_threshold_transfer():
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.qwen3_rules import QWEN3_PTQ_RULES
    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.snnConverter.qwen3_rules import QWEN3_CONVERSION_RULES
    from unieval.snn.snnConverter.rules import DEFAULT_CONVERSION_RULES
    from unieval.snn.operators.qwen3_attention import SQwen3Attention

    model = _small_qwen3().eval()
    input_ids = torch.randint(0, 256, (1, 16))

    # PTQ
    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    # Calibrate (PTQQuan calibrates on first forward)
    with torch.no_grad():
        model(input_ids)

    # Verify calibration happened
    attn = model.blocks[0].attn
    assert attn.quan_q.calibrated, "quan_q should be calibrated after forward"
    cal_s = attn.quan_q.s.item()
    assert cal_s != 1.0, f"quan_q.s should be calibrated, got {cal_s}"

    # Convert to SNN
    converter = SNNConverter(rules=QWEN3_CONVERSION_RULES + DEFAULT_CONVERSION_RULES)
    converter.convert(model, level=32, neuron_type="ST-BIF", is_softmax=True)

    # Verify attention was converted
    s_attn = model.blocks[0].attn
    assert isinstance(s_attn, SQwen3Attention), f"Expected SQwen3Attention, got {type(s_attn)}"

    # Verify all 6 thresholds were transferred (none should be default 1.0)
    for if_name in ["q_IF", "k_IF", "v_IF", "attn_IF", "after_attn_IF", "proj_IF"]:
        neuron = getattr(s_attn, if_name)
        th = neuron.q_threshold.item()
        assert th != 1.0, f"{if_name}.q_threshold = {th}, should be calibrated (not 1.0)"
        print(f"    {if_name}: threshold={th:.6f}")


def test_uniaffine_threshold_transfer():
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES
    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.snnConverter.uniaffine_rules import UNIAFFINE_CONVERSION_RULES
    from unieval.snn.snnConverter.rules import DEFAULT_CONVERSION_RULES
    from unieval.snn.operators.uniaffine_attention import SpikeUniAffineAttention

    model = _small_uniaffine().eval()
    input_ids = torch.randint(0, 256, (1, 16))

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, UNIAFFINE_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    with torch.no_grad():
        model(input_ids)

    attn = model.blocks[0].attn
    assert attn.quan_q.calibrated, "quan_q should be calibrated"

    converter = SNNConverter(rules=UNIAFFINE_CONVERSION_RULES + DEFAULT_CONVERSION_RULES)
    converter.convert(model, level=32, neuron_type="ST-BIF", is_softmax=False)

    s_attn = model.blocks[0].attn
    assert isinstance(s_attn, SpikeUniAffineAttention), f"Expected SpikeUniAffineAttention, got {type(s_attn)}"

    for if_name in ["q_IF", "k_IF", "v_IF", "attn_IF", "after_attn_IF", "proj_IF"]:
        neuron = getattr(s_attn, if_name)
        th = neuron.q_threshold.item()
        assert th != 1.0, f"{if_name}.q_threshold = {th}, should be calibrated (not 1.0)"
        print(f"    {if_name}: threshold={th:.6f}")


# ===== Test 5: QANN vs SNN output similarity =====

def test_qwen3_qann_vs_snn():
    """End-to-end: calibrate QANN, convert to SNN, compare outputs."""
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.qwen3_rules import QWEN3_PTQ_RULES
    from unieval.snn.snnConverter.wrapper import SNNWrapper, reset_model
    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.snnConverter.qwen3_rules import QWEN3_CONVERSION_RULES
    from unieval.snn.snnConverter.rules import DEFAULT_CONVERSION_RULES
    from copy import deepcopy

    torch.manual_seed(42)
    model = _small_qwen3().eval()
    input_ids = torch.randint(0, 256, (1, 16))

    # PTQ
    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, QWEN3_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    # Calibrate
    with torch.no_grad():
        qann_out = model(input_ids)

    # Deep copy for SNN
    snn_model = deepcopy(model)

    # Convert to SNN
    converter = SNNConverter(rules=QWEN3_CONVERSION_RULES + DEFAULT_CONVERSION_RULES)
    converter.convert(snn_model, level=32, neuron_type="ST-BIF", is_softmax=True)

    # Run SNN for T steps
    T = 32
    reset_model(snn_model)
    x = model.embedding(input_ids)
    S = x.shape[1]
    causal_mask = model.causal_mask[:S, :S]

    # Run QANN blocks
    with torch.no_grad():
        qann_hidden = x.clone()
        for blk in model.blocks:
            qann_hidden = blk(qann_hidden, causal_mask=causal_mask)

    # Run SNN blocks
    with torch.no_grad():
        snn_acc = torch.zeros_like(x)
        for t in range(T):
            step_out = x / T
            for blk in snn_model.blocks:
                step_out = blk(step_out, causal_mask=causal_mask)
            snn_acc += step_out

    cos_sim = torch.nn.functional.cosine_similarity(
        qann_hidden.flatten().unsqueeze(0),
        snn_acc.flatten().unsqueeze(0),
    ).item()
    print(f"    Qwen3 QANN vs SNN cos_sim = {cos_sim:.4f} (T={T})")
    # With proper threshold transfer, should be much better than -0.18
    assert cos_sim > 0.5, f"Qwen3 QANN vs SNN too low: cos_sim={cos_sim:.4f}"


def test_uniaffine_qann_vs_snn():
    """End-to-end: calibrate QANN, convert to SNN, compare outputs."""
    from unieval.qann.quantization.base import BaseQuantizer
    from unieval.qann.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES
    from unieval.snn.snnConverter.wrapper import SNNWrapper, reset_model
    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.snnConverter.uniaffine_rules import UNIAFFINE_CONVERSION_RULES
    from unieval.snn.snnConverter.rules import DEFAULT_CONVERSION_RULES
    from copy import deepcopy

    torch.manual_seed(42)
    model = _small_uniaffine().eval()
    input_ids = torch.randint(0, 256, (1, 16))

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, UNIAFFINE_PTQ_RULES, level=32)
            return m

    Q().quantize_model(model)

    with torch.no_grad():
        qann_out = model(input_ids)

    snn_model = deepcopy(model)

    converter = SNNConverter(rules=UNIAFFINE_CONVERSION_RULES + DEFAULT_CONVERSION_RULES)
    converter.convert(snn_model, level=32, neuron_type="ST-BIF", is_softmax=False)

    T = 32
    reset_model(snn_model)
    x = model.embedding(input_ids)
    S = x.shape[1]
    causal_mask = model.causal_mask[:S, :S]

    with torch.no_grad():
        qann_hidden = x.clone()
        for blk in model.blocks:
            qann_hidden = blk(qann_hidden, causal_mask=causal_mask)

    with torch.no_grad():
        snn_acc = torch.zeros_like(x)
        for t in range(T):
            step_out = x / T
            for blk in snn_model.blocks:
                step_out = blk(step_out, causal_mask=causal_mask)
            snn_acc += step_out

    cos_sim = torch.nn.functional.cosine_similarity(
        qann_hidden.flatten().unsqueeze(0),
        snn_acc.flatten().unsqueeze(0),
    ).item()
    print(f"    UniAffine QANN vs SNN cos_sim = {cos_sim:.4f} (T={T})")
    assert cos_sim > 0.5, f"UniAffine QANN vs SNN too low: cos_sim={cos_sim:.4f}"


# ===== Main =====

if __name__ == "__main__":
    print("=" * 60)
    print("Decoder PTQ -> SNN Conversion Tests")
    print("=" * 60)

    print("\n--- PTQ Quantizer Placement ---")
    run_test("Qwen3 PTQ 6 quantizers", test_qwen3_ptq_quantizers)
    run_test("UniAffine PTQ 6 quantizers", test_uniaffine_ptq_quantizers)

    print("\n--- Protocol Matching After PTQ ---")
    run_test("Qwen3 types after PTQ", test_types_after_qwen3_ptq)
    run_test("UniAffine types after PTQ", test_types_after_uniaffine_ptq)

    print("\n--- ANN vs QANN Forward ---")
    run_test("Qwen3 ANN vs QANN", test_qwen3_ann_vs_qann)
    run_test("UniAffine ANN vs QANN", test_uniaffine_ann_vs_qann)

    print("\n--- Threshold Transfer ---")
    run_test("Qwen3 threshold transfer", test_qwen3_threshold_transfer)
    run_test("UniAffine threshold transfer", test_uniaffine_threshold_transfer)

    print("\n--- QANN vs SNN Equivalence ---")
    run_test("Qwen3 QANN vs SNN", test_qwen3_qann_vs_snn)
    run_test("UniAffine QANN vs SNN", test_uniaffine_qann_vs_snn)

    print()
    print("=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)

    if ERRORS:
        print("\nFailed tests:")
        for name, e in ERRORS:
            print(f"  - {name}: {e}")
        sys.exit(1)
