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


def _project_paths() -> Tuple[str, str]:
    """
    返回两个关键路径：
    - cifar10_benchmark_dir: 当前脚本所在目录
    - ann_dir: snn_framework/ANN 目录（用于保证 resnet_cifar10.py 的 imports 可用）
    """

    cifar10_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    # .../snn_framework/Evaluation/benchmarks/cifar10/train.py
    # -> .../snn_framework/ANN
    ann_dir = os.path.abspath(os.path.join(cifar10_benchmark_dir, "..", "..", "..", "ANN"))
    return cifar10_benchmark_dir, ann_dir


def _ensure_import_path() -> None:
    """确保 repo root 在 sys.path 中，以支持 `import unieval` 绝对导入。"""

    benchmark_dir, ann_dir = _project_paths()
    repo_root = os.path.abspath(os.path.join(benchmark_dir, "..", "..", "..", ".."))
    for p in (repo_root, ann_dir):
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


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return float(correct) / float(targets.size(0))


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
                f"Epoch {epoch:03d} | step {step:04d}/{len(loader)} | "
                f"loss {(total_loss / max(total_seen, 1)):.4f} | "
                f"acc {(total_correct / max(total_seen, 1)):.4f} | "
                f"{(total_seen / max(elapsed, 1e-6)):.1f} samples/s"
            )

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": float(total_correct) / max(total_seen, 1),
    }


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, state: TrainState, scheduler=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": state.epoch,
        "best_acc": state.best_acc,
    }
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    torch.save(data, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device = None, scheduler=None) -> TrainState:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if device is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return TrainState(epoch=int(ckpt.get("epoch", 0)), best_acc=float(ckpt.get("best_acc", 0.0)))


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 标准增强：
    - train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
    - test : Normalize
    """

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


def main() -> None:
    _ensure_import_path()

    parser = argparse.ArgumentParser(description="Train ResNet20 on CIFAR-10 (ANN)")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 数据目录（自动下载）")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="./runs/resnet20_cifar10")
    parser.add_argument("--resume", type=str, default="", help="checkpoint 路径（可选）")
    parser.add_argument("--eval-only", action="store_true", help="只评估，不训练")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # 1) 数据
    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.workers)

    # 2) 模型
    from unieval.ann.models.resnet_cifar10 import ResNet20

    model = ResNet20(num_classes=10).to(device)

    # 3) 优化器与学习率策略
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

    # 只评估
    if args.eval_only:
        metrics = evaluate(model, test_loader, device)
        print(f"Eval | loss={metrics['loss']:.4f} | acc={metrics['acc']:.4f}")
        return

    # 4) 训练循环
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
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr_now:.6f} | "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
            f"test loss={te['loss']:.4f} acc={te['acc']:.4f}"
        )

        state.epoch = epoch
        if te["acc"] > state.best_acc:
            state.best_acc = te["acc"]
            save_checkpoint(ckpt_best, model, optimizer, state, scheduler=scheduler)
            print(f"New best acc: {state.best_acc:.4f} | saved to {ckpt_best}")
        save_checkpoint(ckpt_last, model, optimizer, state, scheduler=scheduler)


if __name__ == "__main__":
    main()

