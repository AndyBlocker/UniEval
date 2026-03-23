from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def _project_paths() -> Tuple[str, str, str]:
    """
    返回关键路径：
    - benchmark_dir: 当前脚本目录
    - repo_root: UniEval 项目根目录（保证 `import unieval` 可用）
    - ann_dir: snn_framework/ANN（保证 `operators`/`models` 顶层导入可用）
    - snn_framework_dir: snn_framework 根目录（保证 `QANN` 顶层导入可用）
    """

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(benchmark_dir, "..", "..", "..", ".."))
    snn_framework_dir = os.path.abspath(os.path.join(benchmark_dir, "..", "..", ".."))
    ann_dir = os.path.join(snn_framework_dir, "ANN")
    return benchmark_dir, repo_root, ann_dir, snn_framework_dir


def _ensure_import_path() -> None:
    """
    说明：
    - 该脚本常见运行方式是 `python path/to/snnInfer.py`，此时不会自动把 UniEval 根目录加入 sys.path，
      因此需要显式加入以支持 `import unieval` 的绝对导入。
    - `ANN/models/resnet_cifar10.py` 内部通过 `from operators...` 引入算子；
      因此必须把 `.../snn_framework/ANN` 放进 sys.path。
    - `QANN/...` 使用 `from QANN...` 方式导入；
      因此必须把 `.../snn_framework` 根目录放进 sys.path。
    """

    _, repo_root, ann_dir, snn_framework_dir = _project_paths()
    for p in (repo_root, ann_dir, snn_framework_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def force_set_is_init_true(model: nn.Module) -> int:
    """
    强制把模型中所有子模块的 `is_init` 置为 True（用于跳过某些 QANN 模块的 init_from 流程）。

    返回：被修改的模块数量。
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "is_init"):
            try:
                v = getattr(m, "is_init")
                # Handle tensor/buffer/parameter style flags
                if torch.is_tensor(v):
                    v.data = torch.ones_like(v.data).to(v.device)
                else:
                    setattr(m, "is_init", True)
                n += 1
            except Exception:
                # Best-effort: do not crash inference due to an auxiliary debug utility
                continue
    return n


def _to_float(x) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return float("nan")


def filter_details_remove_zero_ops(details: dict) -> dict:
    """
    过滤掉 ops 为 0 的 layer 记录（用于打印/调试）。
    规则：total_ops、ac_ops、mac_ops 都为 0 的条目会被移除。
    """
    if not isinstance(details, dict):
        return details
    layers = details.get("layers")
    if not isinstance(layers, list):
        return details

    kept = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        total_ops = _to_float(layer.get("total_ops", 0.0))
        ac_ops = _to_float(layer.get("ac_ops", 0.0))
        mac_ops = _to_float(layer.get("mac_ops", 0.0))
        if (total_ops == 0.0) and (ac_ops == 0.0) and (mac_ops == 0.0):
            continue
        kept.append(layer)

    new_details = dict(details)
    new_details["layers"] = kept
    return new_details


def format_details_layers_table(details: dict, topk: int = 80) -> str:
    """
    将 details['layers']（已过滤）整理成易读表格：
    - 按 total_ops 降序
    - 输出 topk 行
    """
    if not isinstance(details, dict) or not isinstance(details.get("layers"), list):
        return str(details)

    layers = details.get("layers", [])
    layers_sorted = sorted(layers, key=lambda d: _to_float(d.get("total_ops", 0.0)), reverse=True)
    if topk is not None and topk > 0:
        layers_sorted = layers_sorted[:topk]

    name_w = min(64, max([len(str(d.get("name", ""))) for d in layers_sorted] + [4]))
    header = (
        f"{'name':<{name_w}}  "
        f"{'total_ops':>14}  {'mac_ops':>14}  {'ac_ops':>14}  {'firing_rate':>12}"
    )
    lines = [header, "-" * len(header)]
    for d in layers_sorted:
        name = str(d.get("name", ""))[:name_w]
        total_ops = _to_float(d.get("total_ops", 0.0))
        mac_ops = _to_float(d.get("mac_ops", 0.0))
        ac_ops = _to_float(d.get("ac_ops", 0.0))
        fr = _to_float(d.get("firing_rate", float("nan")))
        lines.append(
            f"{name:<{name_w}}  "
            f"{total_ops:>14.0f}  {mac_ops:>14.0f}  {ac_ops:>14.0f}  {fr:>12.6f}"
        )
    return "\n".join(lines)


@dataclass
class TrainState:
    epoch: int
    best_acc: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # print("QANN images",images.abs().mean())
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_seen += int(labels.size(0))
        # break

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": float(total_correct) / max(total_seen, 1),
    }


@torch.no_grad()
def evaluateSNN(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # print("SNN images",images.abs().mean())
        logits,_ = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_seen += int(labels.size(0))
        # break

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": float(total_correct) / max(total_seen, 1),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 100,
) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    start = time.time()
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_seen += int(bs)

        if log_interval > 0 and step % log_interval == 0:
            elapsed = time.time() - start
            print(
                f"[QAT] Epoch {epoch:03d} | step {step:04d}/{len(loader)} | "
                f"loss {(total_loss / max(total_seen, 1)):.4f} | "
                f"acc {(total_correct / max(total_seen, 1)):.4f} | "
                f"{(total_seen / max(elapsed, 1e-6)):.1f} samples/s"
            )

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": float(total_correct) / max(total_seen, 1),
    }


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, state: TrainState, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": state.epoch,
            "best_acc": state.best_acc,
            "WeightBit": int(args.WeightBit),
            "ActBit": int(args.ActBit),
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer) -> TrainState:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return TrainState(epoch=int(ckpt.get("epoch", 0)), best_acc=float(ckpt.get("best_acc", 0.0)))


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    workers: int,
) -> Tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def resolve_ann_checkpoint(path: str) -> str:
    """
    支持用户传入：
    - 具体 ckpt 文件路径（*.pt）
    - 或者一个目录路径（例如 runs/resnet20_cifar10），此时自动选择：
        1) best.pt（优先）
        2) last.pt
    """

    if not path:
        return ""

    path = os.path.abspath(path)
    if os.path.isdir(path):
        best = os.path.join(path, "best.pt")
        last = os.path.join(path, "last.pt")
        if os.path.exists(best):
            return best
        if os.path.exists(last):
            return last
        raise FileNotFoundError(f"ANN checkpoint directory has no best.pt/last.pt: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"ANN checkpoint not found: {path}")
    return path


def load_ann_weights(model: nn.Module, ckpt_path: str) -> None:
    """
    仅加载 ANN 的 model 权重（不加载 optimizer）。

    兼容两种常见保存格式：
    - {'model': state_dict, ...}
    - 直接保存的 state_dict
    """

    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[QAT] Warning: load ANN weights with strict=False")
        if missing:
            print(f"[QAT]   missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[QAT]   unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")


def resolve_qann_checkpoint(path: str) -> str:
    """
    QANN checkpoint 解析逻辑与 ANN 一致：
    - 支持传入具体文件（*.pt）
    - 支持传入目录：自动选择 best.pt（优先）或 last.pt
    """
    if not path:
        return ""

    path = os.path.abspath(path)
    if os.path.isdir(path):
        best = os.path.join(path, "best.pt")
        last = os.path.join(path, "last.pt")
        if os.path.exists(best):
            return best
        if os.path.exists(last):
            return last
        raise FileNotFoundError(f"QANN checkpoint directory has no best.pt/last.pt: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"QANN checkpoint not found: {path}")
    return path


def load_qann_weights(model: nn.Module, ckpt_path: str) -> None:
    """
    加载 QANN 权重到已经完成量化替换/BN fuse 的模型上。
    兼容：
    - {'model': state_dict, ...}
    - 直接保存的 state_dict
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[SNN] Warning: load QANN weights with strict=False")
        if missing:
            print(f"[SNN]   missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[SNN]   unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")


def main() -> None:
    _ensure_import_path()

    parser = argparse.ArgumentParser(description="QAT training for ResNet20 on CIFAR-10 (QANN)")
    parser.add_argument("--data-dir", type=str, default="/home/youkang/gpfs-share/cifar10", help="CIFAR-10 数据目录（自动下载）")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05, help="QAT 通常用更小的学习率")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="/home/kang_you/UniEval/unieval/evaluation/benchmarks/cifar10/runs/resnet20_cifar10_snn/")
    parser.add_argument("--resume", type=str, default="", help="checkpoint 路径（可选）")
    parser.add_argument("--eval-only", action="store_true", help="只评估，不训练")
    parser.add_argument("--WeightBit", type=int, default=4)
    parser.add_argument("--ActBit", type=int, default=8)
    parser.add_argument(
        "--ann-ckpt",
        type=str,
        default="/home/youkang/gpfs-share/framework/snn_framework/Evaluation/benchmarks/cifar10/runs/resnet20_cifar10",
        help="训练好的 ANN checkpoint（文件或目录）。目录会自动选择 best.pt/last.pt",
    )
    parser.add_argument(
        "--qann-ckpt",
        type=str,
        default="",
        help="训练好的 QANN checkpoint（文件或目录）。用于直接加载 QAT 后的量化模型权重（可选）。",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=32,
        help="SNN 推理的 time steps（T）。Spike 版本输出会在 T 步内累加逼近量化输出。",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # 1) 数据
    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.workers)

    # 2) 构建 ANN 模型
    # 说明：此脚本可能以 `python path/to/snnInfer.py` 方式直接运行，
    # 相对导入会报 “no known parent package”，因此统一使用绝对导入。
    from unieval.ann.models.resnet_cifar10 import ResNet20

    model = ResNet20(num_classes=10)

    # 3) 先加载你训练好的 ANN 权重，再做量化替换
    ann_ckpt = resolve_ann_checkpoint(args.ann_ckpt)
    if ann_ckpt:
        print(f"[QAT] Loading ANN weights from: {ann_ckpt}")
        load_ann_weights(model, ann_ckpt)

    # 4) 按你指定的量化模型生成方式进行 QAT 替换与 BN fuse
    from unieval.qann.quantization.model_quantization import quantized_train_model_fusebn
    from unieval.qann.quantization.bnFusion import fuse_module_train

    quantized_train_model_fusebn(model, weightBit=args.WeightBit, actBit=args.ActBit)
    fuse_module_train(model)
    # print(model)

    model = model.to(device)

    # 5) 优化器与学习率策略
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    state = TrainState(epoch=0, best_acc=0.0)
    qann_ckpt = resolve_qann_checkpoint(args.qann_ckpt)
    if qann_ckpt:
        # state = load_checkpoint(args.resume, model, optimizer)
        print(f"[QAT] Loading QANN weights from: {qann_ckpt}")
        load_qann_weights(model, qann_ckpt)        
        # print(f"Resumed from {args.resume} | epoch={state.epoch} | best_acc={state.best_acc:.4f}")

    force_set_is_init_true(model)
    # print(model)
    metrics = evaluate(model, test_loader, device)
    print(f"[QAT] Eval | loss={metrics['loss']:.4f} | acc={metrics['acc']:.4f}")

    from unieval.snn import convert
    wrapper = convert(model, time_step=args.time_step, level=2 ** args.ActBit, is_softmax=False)
    wrapper = wrapper.to(device)
    wrapper.eval()
    
    # print(wrapper)
    print("[SNN] 转换完成")    
    metrics = evaluateSNN(wrapper, test_loader, device)
    print(f"[SNN] Eval | loss={metrics['loss']:.4f} | acc={metrics['acc']:.4f}")        
    
    from unieval.evaluation.feasibility.check_hooker import Feasibility_checker, results_to_table
    
    temporal_size = 1
    spatial_dimension = 1
    x_global = torch.randn(2, 3, 32, 32).to(device)
    print("[SNN] 可行性评估")    
    results = Feasibility_checker(
        wrapper,
        temporal_size=temporal_size,
        spatial_dimension=spatial_dimension,
        x_global=x_global,
        model_name="resnet20",
        verbose=False,
        ignore_error_modules=True,
        atol=1e-2,
        rtol=1e-2,
    )
    results_to_table(
        results,
        include_skipped=True,
        exclude_skipped_reasons=["cannot infer or capture input"],
        csv_path=os.path.join(args.out_dir, "feasibility.csv"),
    )
    
    from unieval.evaluation import evaluate_energy
    print("[SNN] 能耗评估")    
    result = evaluate_energy(
        wrapper, test_loader,
        profile="resnet20", time_step=args.time_step, num_batches=10,
    )
    print(f"\n{result}")
    details = filter_details_remove_zero_ops(result.details)
    print("\n[Details] layers (non-zero ops), sorted by total_ops desc (top 80)")
    print(format_details_layers_table(details, topk=80))
    print("\n[Details] raw dict (filtered zero-ops)")
    print(details)
    
    print("[SNN] 硬件仿真器结果")
    from unieval.evaluation import simulater_fast_evaluate
    import yaml
    hw_config = yaml.load(open('/home/kang_you/UniEval/configs/hardware_param.yaml', 'r'), Loader=yaml.FullLoader)
    result = simulater_fast_evaluate(
        wrapper, test_loader,
        profile="resnet20", time_step=args.time_step, num_batches=10, hardware_config=hw_config
    )
    energy = result['total_energy']
    area = result['total_area']
    latency = result['total_latency']
    print(f"energy:{energy} mJ, area:{area} mm^2, latency:{latency} ms")    
    
    
if __name__ == "__main__":
    main()

