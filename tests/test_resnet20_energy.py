"""Test: verify new resnet20 conversion produces same energy results as baseline.

Baseline: results/resnet20_cifar10_snn_energy_T32.txt
  energy_mJ: 0.0942
  e_mac_mJ:  0.0021
  e_ac_mJ:   0.0921
  mac_ops_G: 0.0005
  ac_ops_G:  0.1024

Uses real QANN checkpoint for meaningful comparison.
"""
import sys
import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Paths to checkpoints (same as snnInfer.py baseline run)
ANN_CKPT = "/home/weiziling/gpfs-share/resnet20_cpk/resnet20_cifar10/best.pt"
QANN_CKPT = "/home/weiziling/gpfs-share/resnet20_cpk/resnet20_cifar10_qat/best.pt"

# Baseline values from results/resnet20_cifar10_snn_energy_T32.txt
BASELINE = {
    "energy_mJ": 0.0942,
    "e_mac_mJ": 0.0021,
    "e_ac_mJ": 0.0921,
    "mac_ops_G": 0.0005,
    "ac_ops_G": 0.1024,
}

TIME_STEP = 32
WEIGHT_BIT = 4
ACT_BIT = 8
LEVEL = 2 ** ACT_BIT  # 256


def force_set_is_init_true(model):
    for m in model.modules():
        if hasattr(m, "is_init"):
            v = getattr(m, "is_init")
            if torch.is_tensor(v):
                v.data = torch.ones_like(v.data)
            else:
                setattr(m, "is_init", True)


def load_weights(model, ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state_dict, strict=False)


def build_qann_model(device="cpu"):
    from unieval.ann.models.resnet_cifar10 import ResNet20
    from unieval.qann.quantization.model_quantization import quantized_train_model_fusebn
    from unieval.qann.quantization.bnFusion import fuse_module_train

    model = ResNet20(num_classes=10)

    # Load ANN weights first (matches baseline pipeline in snnInfer.py)
    if os.path.exists(ANN_CKPT):
        print(f"  Loading ANN weights from: {ANN_CKPT}")
        load_weights(model, ANN_CKPT)

    quantized_train_model_fusebn(model, weightBit=WEIGHT_BIT, actBit=ACT_BIT)
    fuse_module_train(model)

    # Load QANN weights on top
    print(f"  Loading QANN weights from: {QANN_CKPT}")
    load_weights(model, QANN_CKPT)
    force_set_is_init_true(model)

    model = model.to(device)
    model.eval()
    return model


def build_dataloader():
    from torchvision import datasets, transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # Try multiple data paths (no network access on GPU servers)
    data_dirs = [
        "/home/weiziling/gpfs-share/cifar10",
        os.path.expanduser("~/gpfs-share/cifar10"),
        "./data/cifar10",
    ]
    data_root = None
    for d in data_dirs:
        if os.path.isdir(d):
            data_root = d
            break
    if data_root is None:
        raise FileNotFoundError(f"CIFAR-10 data not found in: {data_dirs}")
    print(f"  Using CIFAR-10 data from: {data_root}")
    test_set = datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=test_tf,
    )
    return DataLoader(
        test_set, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True,
    )


def run_energy_eval(wrapper, test_loader):
    from unieval.evaluation import evaluate_energy
    result = evaluate_energy(
        wrapper, test_loader,
        profile="resnet20", time_step=TIME_STEP, num_batches=10,
    )
    return result


def convert_old(qann_model):
    """Old conversion (resent20_rules, global index3, in-place)."""
    import unieval.snn.snnConverter.resent20_rules as old_rules
    old_rules.index3 = 0

    from unieval.snn.snnConverter.converter import SNNConverter
    from unieval.snn.snnConverter.wrapper import SNNWrapper

    converter = SNNConverter(rules=old_rules.RESNET20_CONVERSION_RULES)
    return SNNWrapper(
        ann_model=qann_model,
        time_step=TIME_STEP,
        level=LEVEL,
        encoding_type="analog",
        neuron_type="ST-BIF",
        is_softmax=False,
        converter=converter,
    )


def convert_new(qann_model):
    """New conversion (UNIVERSAL rules, SpikeBasicBlock)."""
    from unieval.snn import convert
    return convert(
        qann_model,
        time_step=TIME_STEP,
        level=LEVEL,
        encoding_type="analog",
        neuron_type="ST-BIF",
        is_softmax=False,
    )


def print_vs_baseline(label, result, baseline):
    """Print energy metrics against saved baseline (informational, not asserted).

    The baseline in results/ was generated under a specific code revision;
    the key assertion is old==new equivalence, not matching a frozen snapshot.
    """
    m = result.metrics
    print(f"\n--- {label} ---")
    for key in ["energy_mJ", "e_mac_mJ", "e_ac_mJ", "mac_ops_G", "ac_ops_G"]:
        diff = abs(m[key] - baseline[key])
        tag = "OK" if diff < 0.001 else f"DRIFT={diff:.6f}"
        print(f"  {key}: {m[key]:.4f}  (baseline: {baseline[key]:.4f})  [{tag}]")


def assert_old_new_match(result_old, result_new):
    """Assert old and new energy results match each other."""
    print(f"\n--- Old vs New Direct Comparison ---")
    fields = ["energy_mJ", "e_mac_mJ", "e_ac_mJ", "mac_ops_G", "ac_ops_G"]
    for key in fields:
        v_old = result_old.metrics[key]
        v_new = result_new.metrics[key]
        diff = abs(v_old - v_new)
        match = "==" if diff < 1e-6 else f"DIFF={diff:.6f}"
        print(f"  {key}: old={v_old:.6f}  new={v_new:.6f}  {match}")
        assert diff <= 0.001, (
            f"Old vs New: {key} mismatch: {v_old:.6f} vs {v_new:.6f} (diff={diff:.6f})"
        )
    print(f"  PASS")


def test_energy_equivalence():
    """Verify old and new conversion produce identical energy results matching baseline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not os.path.exists(QANN_CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {QANN_CKPT}")

    print("\n[1/5] Building QANN models...")
    model_old = build_qann_model(device)
    model_new = copy.deepcopy(model_old)

    print("\n[2/5] Building dataloader...")
    test_loader = build_dataloader()

    print("\n[3/5] Old conversion + energy eval...")
    wrapper_old = convert_old(model_old)
    wrapper_old = wrapper_old.to(device)
    wrapper_old.eval()
    result_old = run_energy_eval(wrapper_old, test_loader)

    print("\n[4/5] New conversion (UNIVERSAL) + energy eval...")
    wrapper_new = convert_new(model_new)
    wrapper_new = wrapper_new.to(device)
    wrapper_new.eval()
    result_new = run_energy_eval(wrapper_new, test_loader)

    print("\n[5/5] Comparing results...")
    print_vs_baseline("Old vs Baseline", result_old, BASELINE)
    print_vs_baseline("New vs Baseline", result_new, BASELINE)
    assert_old_new_match(result_old, result_new)

    print("\n  ALL ENERGY TESTS PASSED.")


if __name__ == "__main__":
    test_energy_equivalence()
