#!/usr/bin/env python3
"""QANN vs SNN Equivalence Verification.

Verifies that the SNN accumulated output matches the QANN output at every
layer and operator level.  This is the core theoretical guarantee of
SpikeZIP-TF: after enough timesteps the accumulated spiking output should
converge to the quantised ANN output.

The script:
  1. Creates a tiny ViT and applies LSQ quantization
  2. Calibrates MyQuan scales with a forward pass
  3. Deep-copies the QANN and runs a reference forward
  4. Converts the original model to SNN
  5. Runs SNN temporal inference, accumulating every layer's output
  6. Compares QANN vs SNN layer-by-layer and operator-by-operator

Usage:
    python experiments/verify_equivalence.py
    python experiments/verify_equivalence.py --time_step 64 --level 32
"""

import os
import sys
import argparse
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from functools import partial

from unieval.models.vit import VisionTransformer
from unieval.quantization.lsq import LSQQuantizer
from unieval.conversion.wrapper import SNNWrapper


# ── helpers ──────────────────────────────────────────────────────

def _flat_cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def compare(name, qann_out, snn_out):
    """Return a dict of comparison metrics."""
    if isinstance(qann_out, tuple):
        qann_out = qann_out[0]
    if isinstance(snn_out, tuple):
        snn_out = snn_out[0]

    diff = (qann_out - snn_out).float()
    l1 = diff.abs().mean().item()
    max_diff = diff.abs().max().item()

    qann_norm = qann_out.float().norm().item()
    rel = (diff.norm().item() / qann_norm * 100) if qann_norm > 0 else 0.0
    cos = _flat_cos(qann_out, snn_out)

    return dict(
        name=name, shape=tuple(qann_out.shape),
        l1=l1, max_diff=max_diff, rel_pct=rel, cos_sim=cos,
        qann_mean=qann_out.float().mean().item(),
        snn_mean=snn_out.float().mean().item(),
    )


def print_table(title, rows):
    print(f"\n{'─' * 74}")
    print(f"  {title}")
    print(f"{'─' * 74}")
    print(f"  {'Layer':<22} {'Shape':<18} {'L1':>10} {'Rel%':>9} {'Cos':>10}")
    print(f"  {'─' * 68}")
    for r in rows:
        print(
            f"  {r['name']:<22} {str(r['shape']):<18} "
            f"{r['l1']:>10.2e} {r['rel_pct']:>8.4f}% {r['cos_sim']:>10.6f}"
        )


def attach_hooks(targets, storage, mode):
    """Register forward hooks on *targets* (dict name→module).

    mode="record"     → store output (single forward)
    mode="accumulate" → sum outputs across timesteps
    """
    handles = []
    for name, mod in targets.items():
        def _make(n):
            def _hook(_, __, out):
                o = out[0] if isinstance(out, tuple) else out
                o = o.detach().clone()
                if mode == "record":
                    storage[n] = o
                else:
                    storage[n] = storage.get(n, 0) + o
            return _hook
        handles.append(mod.register_forward_hook(_make(name)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── main ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--time_step", type=int, default=32)
    p.add_argument("--level", type=int, default=16)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    args = p.parse_args()
    device = args.device

    num_patches = (args.img_size // args.patch_size) ** 2

    print("=" * 74)
    print("  QANN vs SNN Equivalence Verification")
    print("=" * 74)
    print(f"  ViT  : depth={args.depth}  dim={args.embed_dim}  heads={args.num_heads}")
    print(f"  Quant: level={args.level}")
    print(f"  SNN  : T_max={args.time_step}  encoding=analog")
    print(f"  Image: {args.img_size}x{args.img_size}  patch={args.patch_size}  "
          f"patches={num_patches}")
    print(f"  Device: {device}")

    # ── 1  create & quantize ─────────────────────────────────────
    print("\n[1/6] Creating and quantizing model ...")
    model = VisionTransformer(
        img_size=args.img_size, patch_size=args.patch_size, in_chans=3,
        num_classes=10, embed_dim=args.embed_dim, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=2., qkv_bias=True,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=True,
    ).to(device)

    quantizer = LSQQuantizer(level=args.level, is_softmax=True)
    model = quantizer.quantize_model(model)

    # ── 2  calibrate MyQuan scales ───────────────────────────────
    print("[2/6] Calibrating quantization scales (1 training forward) ...")
    model.train()
    with torch.no_grad():
        model(torch.randn(4, 3, args.img_size, args.img_size, device=device))
    model.eval()

    # ── 3  QANN reference forward ────────────────────────────────
    print("[3/6] Running QANN forward ...")
    qann = deepcopy(model).eval()

    x = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    # hook targets for QANN
    q_targets = {}
    for i, blk in enumerate(qann.blocks):
        q_targets[f"block.{i}"] = blk
        q_targets[f"block.{i}.norm1"] = blk.norm1
        q_targets[f"block.{i}.attn"] = blk.attn
        q_targets[f"block.{i}.norm2"] = blk.norm2
        q_targets[f"block.{i}.mlp"] = blk.mlp

    q_out = {}
    hs = attach_hooks(q_targets, q_out, "record")
    with torch.no_grad():
        qann_final = qann(x)
    remove_hooks(hs)
    q_out["final"] = qann_final.detach().clone()

    print(f"     output shape : {qann_final.shape}")
    print(f"     output mean  : {qann_final.mean().item():.6f}")

    # ── 4  convert to SNN ────────────────────────────────────────
    print("[4/6] Converting to SNN ...")
    wrapper = SNNWrapper(
        ann_model=model,
        time_step=args.time_step,
        encoding_type="analog",
        level=args.level,
        neuron_type="ST-BIF",
        model_name="vit_tiny",
        is_softmax=True,
    ).to(device).eval()

    # hook targets for SNN (fresh references after conversion)
    s_targets = {}
    snn_m = wrapper.model
    for i, blk in enumerate(snn_m.blocks):
        s_targets[f"block.{i}"] = blk
        s_targets[f"block.{i}.norm1"] = blk.norm1
        s_targets[f"block.{i}.attn"] = blk.attn
        s_targets[f"block.{i}.norm2"] = blk.norm2
        s_targets[f"block.{i}.mlp"] = blk.mlp

    # ── 5  SNN temporal inference ────────────────────────────────
    print("[5/6] Running SNN temporal inference ...")
    s_out = {}
    hs = attach_hooks(s_targets, s_out, "accumulate")
    with torch.no_grad():
        snn_final, actual_T = wrapper(x)
    remove_hooks(hs)
    s_out["final"] = snn_final.detach().clone()

    print(f"     actual T : {actual_T}")

    # ── 6  compare ───────────────────────────────────────────────
    print("[6/6] Comparing QANN vs SNN ...\n")

    # — final output —
    fc = compare("final_output", q_out["final"], s_out["final"])
    print("=" * 74)
    print("  Final Output")
    print("=" * 74)
    print(f"  Shape      : {fc['shape']}")
    print(f"  QANN mean  : {fc['qann_mean']:+.6f}    SNN mean : {fc['snn_mean']:+.6f}")
    print(f"  L1 error   : {fc['l1']:.2e}")
    print(f"  Max |diff| : {fc['max_diff']:.2e}")
    print(f"  Rel error  : {fc['rel_pct']:.4f}%")
    print(f"  Cos sim    : {fc['cos_sim']:.6f}")

    # — per-block —
    blk_rows = []
    for i in range(args.depth):
        k = f"block.{i}"
        if k in q_out and k in s_out:
            blk_rows.append(compare(k, q_out[k], s_out[k]))
    print_table("Per-Block Output", blk_rows)

    # — per-operator per-block —
    all_op_rows = []
    for i in range(args.depth):
        op_rows = []
        for op in ("norm1", "attn", "norm2", "mlp"):
            k = f"block.{i}.{op}"
            if k in q_out and k in s_out:
                r = compare(op, q_out[k], s_out[k])
                op_rows.append(r)
        if op_rows:
            print_table(f"Per-Operator  (Block {i})", op_rows)
            all_op_rows.extend(op_rows)

    # ── verdict ──────────────────────────────────────────────────
    everything = [fc] + blk_rows + all_op_rows
    max_rel = max(r["rel_pct"] for r in everything)
    min_cos = min(r["cos_sim"] for r in everything)

    print(f"\n{'=' * 74}")
    if max_rel < 5.0 and min_cos > 0.95:
        print(f"  PASS   max rel err = {max_rel:.4f}%   min cos = {min_cos:.6f}")
    else:
        print(f"  WARN   max rel err = {max_rel:.4f}%   min cos = {min_cos:.6f}")
        print(f"         Try increasing --time_step for better convergence.")
    print(f"{'=' * 74}")

    wrapper.reset()


if __name__ == "__main__":
    main()
