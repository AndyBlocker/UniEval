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
    - 该脚本常见运行方式是 `python path/to/trainQAT.py`，此时不会自动把 UniEval 根目录加入 sys.path，
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

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_seen += int(labels.size(0))

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


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, state: TrainState, args: argparse.Namespace, scheduler=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": state.epoch,
        "best_acc": state.best_acc,
        "WeightBit": int(args.WeightBit),
        "ActBit": int(args.ActBit),
    }
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    torch.save(data, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device = None, scheduler=None) -> TrainState:
    ckpt = torch.load(path, map_location="cpu")
    msg = model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if device is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print("msg", msg)
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



def main() -> None:
    _ensure_import_path()

    parser = argparse.ArgumentParser(description="QAT training for ResNet20 on CIFAR-10 (QANN)")
    parser.add_argument("--data-dir", type=str, default="./data/cifar10", help="CIFAR-10 数据目录（自动下载）")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05, help="QAT 通常用更小的学习率")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="./runs/resnet20_cifar10_qat_1")
    parser.add_argument("--resume", type=str, default="", help="checkpoint 路径（可选）")
    parser.add_argument("--eval-only", action="store_true", help="只评估，不训练")
    parser.add_argument("--WeightBit", type=int, default=4)
    parser.add_argument("--ActBit", type=int, default=8)
    parser.add_argument(
        "--ann-ckpt",
        type=str,
        default="",
        help="训练好的 ANN checkpoint（文件或目录）。目录会自动选择 best.pt/last.pt",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # 1) 数据
    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.workers)

    # 2) 构建 ANN 模型
    # 说明：此脚本可能以 `python path/to/trainQAT.py` 方式直接运行，
    # 相对导入会报 “no known parent package”，因此统一使用绝对导入。
    from unieval.ann.models.resnet_cifar10 import ResNet20

    model = ResNet20(num_classes=10)

    # 3) 先加载 ANN 权重（resume 时跳过，checkpoint 已包含完整权重）
    if not args.resume:
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
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, device=device, scheduler=scheduler)
        print(f"Resumed from {args.resume} | epoch={state.epoch} | best_acc={state.best_acc:.4f}")

    if args.eval_only:
        if args.resume:
            force_set_is_init_true(model)
        print(model)
        metrics = evaluate(model, test_loader, device)
        print(f"[QAT] Eval | loss={metrics['loss']:.4f} | acc={metrics['acc']:.4f}")
        return

    ckpt_last = os.path.join(args.out_dir, "last.pt")
    ckpt_best = os.path.join(args.out_dir, "best.pt")

    for epoch in range(state.epoch + 1, args.epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        te = evaluate(model, test_loader, device)

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[QAT] Epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr_now:.6f} | "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
            f"test loss={te['loss']:.4f} acc={te['acc']:.4f}"
        )

        state.epoch = epoch
        if te["acc"] > state.best_acc:
            state.best_acc = te["acc"]
            save_checkpoint(ckpt_best, model, optimizer, state, args, scheduler=scheduler)
            print(f"[QAT] New best acc: {state.best_acc:.4f} | saved to {ckpt_best}")
        save_checkpoint(ckpt_last, model, optimizer, state, args, scheduler=scheduler)


if __name__ == "__main__":
    main()

