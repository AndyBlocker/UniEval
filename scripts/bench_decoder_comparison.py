#!/usr/bin/env python3
"""Benchmark: UniAffine vs Qwen3 — PPL, Energy, OP Counts across ANN/QANN/SNN.

Usage:
    python scripts/bench_decoder_comparison.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from copy import deepcopy

# ===== Model configs =====
# Small models for fast benchmarking (random weights — PPL values are meaningless
# in absolute terms but valid for cross-domain comparison).

NUM_LAYERS = 4
HIDDEN = 128
FFN_HIDDEN = 256
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 16
VOCAB_SIZE = 512
SEQ_LEN = 64
TIME_STEP = 32
LEVEL = 32
NUM_BATCHES = 5
BATCH_SIZE = 4


def build_qwen3():
    from unieval.ann.models.qwen3 import Qwen3Model, Qwen3Config
    cfg = Qwen3Config(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, hidden_size=HIDDEN,
        ffn_hidden_size=FFN_HIDDEN, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, max_seq_len=SEQ_LEN,
    )
    return Qwen3Model(cfg)


def build_uniaffine():
    from unieval.ann.models.uniaffine import UniAffineModel, UniAffineConfig
    cfg = UniAffineConfig(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, hidden_size=HIDDEN,
        ffn_hidden_size=FFN_HIDDEN, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, max_seq_len=SEQ_LEN,
    )
    return UniAffineModel(cfg)


def make_dataloader():
    data = []
    for _ in range(NUM_BATCHES):
        ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        data.append((ids, ids))
    return data


# ===== Pipeline helpers =====

def quantize_model(model, model_name):
    from unieval.qann.quantization.base import BaseQuantizer
    if model_name == "qwen3":
        from unieval.qann.quantization.qwen3_rules import QWEN3_PTQ_RULES
        rules = QWEN3_PTQ_RULES
    else:
        from unieval.qann.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES
        rules = UNIAFFINE_PTQ_RULES

    class Q(BaseQuantizer):
        def quantize_model(self, m):
            self._apply_rules(m, rules, level=LEVEL)
            return m

    Q().quantize_model(model)
    return model


def calibrate(model, dl):
    model.eval()
    with torch.no_grad():
        for ids, _ in dl:
            model(ids)


def wrap_snn(model, model_name, conversion_rules):
    from unieval.snn.snnConverter.wrapper import SNNWrapper
    is_softmax = (model_name == "qwen3")
    return SNNWrapper(
        ann_model=model,
        time_step=TIME_STEP,
        encoding_type="analog",
        level=LEVEL,
        neuron_type="ST-BIF",
        is_softmax=is_softmax,
        conversion_rules=conversion_rules,
    )


def eval_ppl(model, dl, num_batches=NUM_BATCHES):
    from unieval.evaluation.benchmarks.perplexity import PerplexityEvaluator
    evaluator = PerplexityEvaluator(num_batches=num_batches)
    result = evaluator.evaluate(model, dl)
    return result.metrics["perplexity"], result.metrics["avg_loss"]


def eval_energy(wrapper, model_name):
    from unieval.evaluation.energy.ops_counter import OpsCounter
    from unieval.evaluation.energy.energy import EnergyEvaluator
    from unieval.config import EnergyConfig
    from unieval.ann.models.base import DecoderModelProfile

    profile = DecoderModelProfile(
        depth=NUM_LAYERS, num_heads=NUM_HEADS, embed_dim=HIDDEN,
        num_kv_heads=NUM_KV_HEADS, seq_len=SEQ_LEN, head_dim=HEAD_DIM,
        ffn_hidden_size=FFN_HIDDEN, time_steps=TIME_STEP,
        patch_size=1, img_size=1,
    )
    ops_counter = OpsCounter(time_step=TIME_STEP)
    evaluator = EnergyEvaluator(
        energy_config=EnergyConfig(),
        model_profile=profile,
        ops_counter=ops_counter,
        num_batches=NUM_BATCHES,
    )
    dl = make_dataloader()
    result = evaluator.evaluate(wrapper, dl)
    m = result.metrics
    return {
        "mac_G": m["mac_ops_G"],
        "ac_G": m["ac_ops_G"],
        "e_mac_mJ": m["e_mac_mJ"],
        "e_ac_mJ": m["e_ac_mJ"],
        "energy_mJ": m["energy_mJ"],
        "ssa_fr": result.details.get("ssa_qkv_firing_rates", []),
    }


# ===== Run benchmark for one model =====

def bench_one(model_name, build_fn, conversion_rules):
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()}")
    print(f"{'='*60}")
    results = {}

    # --- ANN ---
    torch.manual_seed(42)
    ann = build_fn().eval()
    dl = make_dataloader()
    t0 = time.time()
    ppl, loss = eval_ppl(ann, dl)
    dt = time.time() - t0
    results["ann"] = {"ppl": ppl, "loss": loss, "time": dt}
    print(f"  ANN  PPL={ppl:.2f}  loss={loss:.4f}  ({dt:.2f}s)")

    # --- QANN ---
    torch.manual_seed(42)
    qann = build_fn().eval()
    quantize_model(qann, model_name)
    # Calibrate PTQ
    cal_dl = make_dataloader()
    calibrate(qann, cal_dl)
    dl = make_dataloader()
    t0 = time.time()
    ppl, loss = eval_ppl(qann, dl)
    dt = time.time() - t0
    results["qann"] = {"ppl": ppl, "loss": loss, "time": dt}
    print(f"  QANN PPL={ppl:.2f}  loss={loss:.4f}  ({dt:.2f}s)")

    # --- SNN ---
    torch.manual_seed(42)
    snn_base = build_fn().eval()
    quantize_model(snn_base, model_name)
    cal_dl = make_dataloader()
    calibrate(snn_base, cal_dl)
    wrapper = wrap_snn(snn_base, model_name, conversion_rules)

    dl = make_dataloader()
    t0 = time.time()
    ppl, loss = eval_ppl(wrapper, dl)
    dt = time.time() - t0
    results["snn"] = {"ppl": ppl, "loss": loss, "time": dt}
    print(f"  SNN  PPL={ppl:.2f}  loss={loss:.4f}  ({dt:.2f}s, T={TIME_STEP})")

    # --- Energy & OPs (SNN) ---
    # Re-wrap for clean energy eval (need fresh ops counters)
    torch.manual_seed(42)
    e_base = build_fn().eval()
    quantize_model(e_base, model_name)
    cal_dl = make_dataloader()
    calibrate(e_base, cal_dl)
    e_wrapper = wrap_snn(e_base, model_name, conversion_rules)

    energy = eval_energy(e_wrapper, model_name)
    results["energy"] = energy
    print(f"  Energy: {energy['energy_mJ']:.6f} mJ  "
          f"(MAC={energy['mac_G']:.4f} G, AC={energy['ac_G']:.4f} G)")
    if energy["ssa_fr"]:
        avg_q = sum(fr[0] for fr in energy["ssa_fr"]) / len(energy["ssa_fr"])
        avg_k = sum(fr[1] for fr in energy["ssa_fr"]) / len(energy["ssa_fr"])
        avg_v = sum(fr[2] for fr in energy["ssa_fr"]) / len(energy["ssa_fr"])
        print(f"  SSA avg firing rates: Q={avg_q:.4f}  K={avg_k:.4f}  V={avg_v:.4f}")

    return results


# ===== Main =====

def main():
    print("=" * 60)
    print("  UniAffine vs Qwen3 — PPL / Energy / OPs Benchmark")
    print(f"  layers={NUM_LAYERS}, hidden={HIDDEN}, heads={NUM_HEADS}, "
          f"seq={SEQ_LEN}, T={TIME_STEP}, L={LEVEL}")
    print(f"  batch={BATCH_SIZE}, batches={NUM_BATCHES}, vocab={VOCAB_SIZE}")
    print("=" * 60)

    from unieval.snn.snnConverter.uniaffine_rules import UNIAFFINE_CONVERSION_RULES
    from unieval.snn.snnConverter.qwen3_rules import QWEN3_CONVERSION_RULES
    from unieval.snn.snnConverter.rules import DEFAULT_CONVERSION_RULES

    r_ua = bench_one("uniaffine", build_uniaffine,
                      UNIAFFINE_CONVERSION_RULES + DEFAULT_CONVERSION_RULES)
    r_qw = bench_one("qwen3", build_qwen3,
                      QWEN3_CONVERSION_RULES + DEFAULT_CONVERSION_RULES)

    # --- Summary table ---
    print("\n")
    print("=" * 72)
    print("  COMPARISON TABLE")
    print("=" * 72)

    header = f"{'Metric':<28} {'UniAffine':>18} {'Qwen3':>18}"
    print(header)
    print("-" * 72)

    for domain in ["ann", "qann", "snn"]:
        tag = domain.upper()
        ua_ppl = r_ua[domain]["ppl"]
        qw_ppl = r_qw[domain]["ppl"]
        ua_loss = r_ua[domain]["loss"]
        qw_loss = r_qw[domain]["loss"]
        ua_t = r_ua[domain]["time"]
        qw_t = r_qw[domain]["time"]
        print(f"  {tag} PPL                    {ua_ppl:>14.2f}     {qw_ppl:>14.2f}")
        print(f"  {tag} Loss                   {ua_loss:>14.4f}     {qw_loss:>14.4f}")
        print(f"  {tag} Time (s)               {ua_t:>14.2f}     {qw_t:>14.2f}")
        print()

    # Energy
    ua_e = r_ua["energy"]
    qw_e = r_qw["energy"]
    print(f"  {'SNN MAC Ops (G)':<28} {ua_e['mac_G']:>14.6f}     {qw_e['mac_G']:>14.6f}")
    print(f"  {'SNN AC Ops (G)':<28} {ua_e['ac_G']:>14.6f}     {qw_e['ac_G']:>14.6f}")
    print(f"  {'SNN Total Ops (G)':<28} {ua_e['mac_G']+ua_e['ac_G']:>14.6f}     {qw_e['mac_G']+qw_e['ac_G']:>14.6f}")
    print(f"  {'E_MAC (mJ)':<28} {ua_e['e_mac_mJ']:>14.6f}     {qw_e['e_mac_mJ']:>14.6f}")
    print(f"  {'E_AC (mJ)':<28} {ua_e['e_ac_mJ']:>14.6f}     {qw_e['e_ac_mJ']:>14.6f}")
    print(f"  {'Energy Total (mJ)':<28} {ua_e['energy_mJ']:>14.6f}     {qw_e['energy_mJ']:>14.6f}")

    # SSA firing rates
    if ua_e["ssa_fr"] and qw_e["ssa_fr"]:
        print()
        print(f"  {'SSA Firing Rates':<28} {'UniAffine':>18} {'Qwen3':>18}")
        print("-" * 72)
        for i, (ua_fr, qw_fr) in enumerate(zip(ua_e["ssa_fr"], qw_e["ssa_fr"])):
            print(f"  Block {i} Q               {ua_fr[0]:>14.4f}     {qw_fr[0]:>14.4f}")
            print(f"  Block {i} K               {ua_fr[1]:>14.4f}     {qw_fr[1]:>14.4f}")
            print(f"  Block {i} V               {ua_fr[2]:>14.4f}     {qw_fr[2]:>14.4f}")

    print()
    print("=" * 72)
    print("  NOTE: Random weights — absolute PPL values are meaningless.")
    print("  Focus on cross-domain deltas and energy/ops differences.")
    print("=" * 72)


if __name__ == "__main__":
    main()
