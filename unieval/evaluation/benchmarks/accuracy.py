"""Accuracy evaluator with top-k and per-timestep tracking."""

import torch
import torch.nn as nn

from .base import BaseEvaluator, EvalResult
from ...snn.snnConverter.wrapper import SNNWrapper
from ...registry import EVALUATOR_REGISTRY


def _accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


class _AverageMeter:
    """Tracks running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@EVALUATOR_REGISTRY.register("accuracy")
class AccuracyEvaluator(BaseEvaluator):
    """Top-1 / Top-5 accuracy evaluator.

    Args:
        topk: Tuple of k values for top-k accuracy.
        num_batches: Max number of batches to evaluate (None for all).
    """

    def __init__(self, topk=(1, 5), num_batches=None):
        self.topk = topk
        self.num_batches = num_batches

    def evaluate(self, model, dataloader, **kwargs) -> EvalResult:
        model.eval()
        device = next(model.parameters()).device

        meters = {f"top{k}": _AverageMeter() for k in self.topk}

        with torch.no_grad():
            for i, (batch, target) in enumerate(dataloader):
                if self.num_batches is not None and i >= self.num_batches:
                    break

                batch = batch.to(device)
                if not batch.is_floating_point() and batch.dtype not in (torch.long, torch.int):
                    batch = batch.float()
                target = target.to(device)

                output = model(batch)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                accs = _accuracy(output, target, topk=self.topk)
                for k, acc in zip(self.topk, accs):
                    meters[f"top{k}"].update(acc, batch.size(0))

                # Reset SNN state
                inner = model.module if hasattr(model, "module") else model
                if isinstance(inner, SNNWrapper):
                    inner.reset()

        metrics = {f"top{k}": meters[f"top{k}"].avg for k in self.topk}
        return EvalResult(metrics=metrics)
