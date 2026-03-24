"""Test: verify new resnet20 block-level conversion produces same results as old in-place conversion.

Compares:
- Old: resent20_rules.py (global index3, in-place modification, hasattr hack)
- New: resnet20_rules.py (SpikeBasicBlock, ConversionContext, clean forward)

Both should produce numerically identical SNN outputs given the same QANN model.
"""
import copy
import sys
import os

import torch
import torch.nn as nn

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def build_qann_model(device="cpu"):
    """Build a quantized ResNet20 model with fused BN."""
    from unieval.ann.models.resnet_cifar10 import ResNet20
    from unieval.qann.quantization.model_quantization import quantized_train_model_fusebn
    from unieval.qann.quantization.bnFusion import fuse_module_train

    model = ResNet20(num_classes=10)
    quantized_train_model_fusebn(model, weightBit=4, actBit=8)
    fuse_module_train(model)
    model = model.to(device)
    model.eval()

    # Force is_init=True to skip calibration
    for m in model.modules():
        if hasattr(m, "is_init"):
            v = getattr(m, "is_init")
            if torch.is_tensor(v):
                v.data = torch.ones_like(v.data)
            else:
                setattr(m, "is_init", True)

    return model


def convert_old(qann_model, time_step=32, level=256):
    """Convert using the OLD resent20_rules (global index3, in-place)."""
    # Reset the global index
    import unieval.snn.snnConverter.resent20_rules as old_rules
    old_rules.index3 = 0

    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.snnConverter.wrapper import SNNWrapper

    converter = SNNConverter(rules=old_rules.RESNET20_CONVERSION_RULES)
    wrapper = SNNWrapper(
        ann_model=qann_model,
        time_step=time_step,
        level=level,
        encoding_type="analog",
        neuron_type="ST-BIF",
        is_softmax=False,
        converter=converter,
    )
    return wrapper


def convert_new(qann_model, time_step=32, level=256):
    """Convert using the NEW resnet20_rules (SpikeBasicBlock, ConversionContext)."""
    from unieval.snn.snnConverter.resnet20_rules import RESNET20_CONVERSION_RULES
    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.snnConverter.wrapper import SNNWrapper

    converter = SNNConverter(rules=RESNET20_CONVERSION_RULES)
    wrapper = SNNWrapper(
        ann_model=qann_model,
        time_step=time_step,
        level=level,
        encoding_type="analog",
        neuron_type="ST-BIF",
        is_softmax=False,
        converter=converter,
    )
    return wrapper


def convert_universal(qann_model, time_step=32, level=256):
    """Convert using UNIVERSAL_CONVERSION_RULES (auto-match)."""
    from unieval.snn import convert
    return convert(
        qann_model,
        time_step=time_step,
        level=level,
        encoding_type="analog",
        neuron_type="ST-BIF",
        is_softmax=False,
    )


@torch.no_grad()
def run_snn(wrapper, x, device="cpu"):
    """Run SNN inference, return (accumulated output, timesteps)."""
    wrapper = wrapper.to(device)
    wrapper.eval()
    accu, T = wrapper(x.to(device))
    return accu.cpu(), T


def test_old_vs_new():
    """Compare old in-place conversion vs new SpikeBasicBlock conversion."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(42)
    x = torch.randn(4, 3, 32, 32)

    time_step = 32
    level = 256  # 2^8

    # Build two identical QANN models
    print("[1/4] Building QANN models...")
    model_old = build_qann_model(device)
    model_new = copy.deepcopy(model_old)
    model_uni = copy.deepcopy(model_old)

    # QANN forward (baseline)
    print("[2/4] QANN forward pass...")
    qann_out = model_old(x.to(device)).cpu()
    print(f"  QANN output shape: {qann_out.shape}, mean: {qann_out.mean():.6f}")

    # Old conversion
    print("[3/4] Old conversion (resent20_rules, in-place)...")
    wrapper_old = convert_old(model_old, time_step=time_step, level=level)
    out_old, T_old = run_snn(wrapper_old, x, device)
    print(f"  Old SNN output shape: {out_old.shape}, T={T_old}, mean: {out_old.mean():.6f}")

    # New conversion
    print("[4/4] New conversion (resnet20_rules, SpikeBasicBlock)...")
    wrapper_new = convert_new(model_new, time_step=time_step, level=level)
    out_new, T_new = run_snn(wrapper_new, x, device)
    print(f"  New SNN output shape: {out_new.shape}, T={T_new}, mean: {out_new.mean():.6f}")

    # Compare
    diff = (out_old - out_new).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (out_old.abs() + 1e-8)).mean().item()

    print(f"\n--- Old vs New Comparison ---")
    print(f"  Max absolute diff:  {max_diff:.8f}")
    print(f"  Mean absolute diff: {mean_diff:.8f}")
    print(f"  Mean relative diff: {rel_diff:.8f}")

    if max_diff < 1e-4:
        print("  PASS: Old and New conversion produce identical results!")
    elif max_diff < 1e-2:
        print("  WARN: Small numerical difference (likely float precision)")
    else:
        print("  FAIL: Significant difference between old and new conversion")

    # Test UNIVERSAL rules
    print("\n[Bonus] UNIVERSAL rules (auto-match)...")
    wrapper_uni = convert_universal(model_uni, time_step=time_step, level=level)
    out_uni, T_uni = run_snn(wrapper_uni, x, device)
    diff_uni = (out_new - out_uni).abs()
    print(f"  UNIVERSAL SNN output mean: {out_uni.mean():.6f}")
    print(f"  New vs UNIVERSAL max diff: {diff_uni.max().item():.8f}")

    if diff_uni.max().item() < 1e-6:
        print("  PASS: UNIVERSAL rules match new resnet20 rules exactly!")
    else:
        print("  WARN: UNIVERSAL differs from explicit resnet20 rules")

    # Print model structures for comparison
    print("\n--- Old SNN model structure (top-level) ---")
    for name, module in wrapper_old.model.named_children():
        print(f"  {name}: {type(module).__name__}")
    print("\n--- New SNN model structure (top-level) ---")
    for name, module in wrapper_new.model.named_children():
        print(f"  {name}: {type(module).__name__}")

    return max_diff < 1e-2


def test_existing_tests():
    """Run existing test_verify.py to make sure nothing else broke."""
    print("\n" + "=" * 60)
    print("Running existing tests/test_verify.py...")
    print("=" * 60)
    import subprocess
    result = subprocess.run(
        [sys.executable, "tests/test_verify.py"],
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
    return result.returncode == 0


if __name__ == "__main__":
    print("=" * 60)
    print("ResNet20 Conversion Equivalence Test")
    print("=" * 60)

    ok1 = test_old_vs_new()

    ok2 = test_existing_tests()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Old vs New conversion: {'PASS' if ok1 else 'FAIL'}")
    print(f"  Existing tests:        {'PASS' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)
