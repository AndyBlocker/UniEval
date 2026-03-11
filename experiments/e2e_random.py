#!/usr/bin/env python3
"""End-to-end experiment with a small randomly initialized ViT.

Runs the full UniEval pipeline on synthetic data:
  1. Create a tiny ViT (2 blocks, dim=64)
  2. LSQ quantize
  3. Convert to SNN
  4. Wrap for temporal inference
  5. Evaluate accuracy + energy on random data

Usage:
    python experiments/e2e_random.py [--device cuda]
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader, TensorDataset

from unieval.config import UniEvalConfig, QuantConfig, ConversionConfig, EnergyConfig, EvalConfig
from unieval.models.vit import VisionTransformer, vit_small_patch16
from unieval.models.base import ModelProfile
from unieval.quantization.lsq import LSQQuantizer
from unieval.conversion.converter import SNNConverter
from unieval.conversion.wrapper import SNNWrapper
from unieval.evaluation.accuracy import AccuracyEvaluator
from unieval.evaluation.energy import EnergyEvaluator
from unieval.evaluation.ops_counter import OpsCounter
from unieval.registry import MODEL_PROFILE_REGISTRY


def create_tiny_vit(num_classes=10, device="cuda"):
    """Create a tiny ViT for fast experimentation."""
    model = VisionTransformer(
        img_size=32,        # small images
        patch_size=8,       # 4x4 = 16 patches
        in_chans=3,
        num_classes=num_classes,
        embed_dim=64,
        depth=2,            # only 2 blocks
        num_heads=4,
        mlp_ratio=2.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.ReLU,
        global_pool=True,
    )
    return model.to(device)


def create_fake_dataloader(num_samples=64, batch_size=16, img_size=32,
                           num_classes=10, device="cuda"):
    """Create a DataLoader with random images and labels."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--time_step", type=int, default=8)
    parser.add_argument("--level", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print("=" * 60)

    # ── 1. Create model ──────────────────────────────────────────
    print("\n[1/5] Creating tiny ViT ...")
    model = create_tiny_vit(num_classes=args.num_classes, device=device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"  Parameters: {num_params:.1f}K")
    print(f"  Blocks: {len(model.blocks)}, Embed dim: {model.embed_dim}")

    # Quick ANN forward check
    x = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"  ANN output shape: {y.shape}, sum: {y.sum().item():.4f}")

    # ── 2. Quantize ──────────────────────────────────────────────
    print("\n[2/5] LSQ quantization (level={}) ...".format(args.level))
    quantizer = LSQQuantizer(level=args.level, is_softmax=True)
    model = quantizer.quantize_model(model)

    # Show quantized structure (abbreviated)
    from unieval.quantization.lsq import QAttention, MyQuan
    n_qattn = sum(1 for m in model.modules() if isinstance(m, QAttention))
    n_myquan = sum(1 for m in model.modules() if isinstance(m, MyQuan))
    print(f"  QAttention modules: {n_qattn}")
    print(f"  MyQuan modules: {n_myquan}")

    # ── 3. Convert to SNN ────────────────────────────────────────
    print("\n[3/5] ANN -> SNN conversion ...")
    wrapper = SNNWrapper(
        ann_model=model,
        time_step=args.time_step,
        encoding_type="analog",
        level=args.level,
        neuron_type="ST-BIF",
        model_name="vit_tiny",
        is_softmax=True,
    )
    wrapper = wrapper.to(device)

    # Show converted structure
    from unieval.operators.neurons import IFNeuron
    from unieval.operators.layers import LLConv2d, LLLinear, Spiking_LayerNorm
    from unieval.operators.attention import SAttention
    counts = {
        "SAttention": sum(1 for m in wrapper.modules() if isinstance(m, SAttention)),
        "IFNeuron": sum(1 for m in wrapper.modules() if isinstance(m, IFNeuron)),
        "LLLinear": sum(1 for m in wrapper.modules() if isinstance(m, LLLinear)),
        "LLConv2d": sum(1 for m in wrapper.modules() if isinstance(m, LLConv2d)),
        "Spiking_LN": sum(1 for m in wrapper.modules() if isinstance(m, Spiking_LayerNorm)),
    }
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # ── 4. SNN Forward ──────────────────────────────────────────
    print(f"\n[4/5] SNN temporal inference (T_max={args.time_step}) ...")
    wrapper.eval()
    x = torch.randn(2, 3, 32, 32).to(device)
    t0 = time.time()
    with torch.no_grad():
        output, actual_T = wrapper(x)
    elapsed = time.time() - t0
    print(f"  Output shape: {output.shape}")
    print(f"  Actual timesteps: {actual_T}")
    print(f"  Wall time: {elapsed:.3f}s")
    wrapper.reset()

    # ── 5. Evaluate ──────────────────────────────────────────────
    print(f"\n[5/5] Evaluation on {args.num_samples} random samples ...")
    dataloader = create_fake_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        img_size=32,
        num_classes=args.num_classes,
        device=device,
    )

    # Accuracy
    print("\n  --- Accuracy ---")
    acc_eval = AccuracyEvaluator(topk=(1, 5), num_batches=None)
    acc_result = acc_eval.evaluate(wrapper, dataloader)
    for k, v in acc_result.metrics.items():
        print(f"  {k}: {v:.2f}%")

    # Energy
    print("\n  --- Energy ---")
    tiny_profile = ModelProfile(
        depth=2, num_heads=4, embed_dim=64, patch_size=8, time_steps=args.time_step,
    )
    energy_eval = EnergyEvaluator(
        model_profile=tiny_profile,
        num_batches=2,
    )
    energy_result = energy_eval.evaluate(wrapper, dataloader)
    for k, v in energy_result.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
