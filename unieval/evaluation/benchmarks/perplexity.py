"""Perplexity evaluator for language models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEvaluator, EvalResult
from ...snn.snnConverter.wrapper import SNNWrapper
from ...registry import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("perplexity")
class PerplexityEvaluator(BaseEvaluator):
    """Evaluator for language model perplexity.

    Computes cross-entropy loss on next-token prediction, then
    converts to perplexity: PPL = exp(avg_loss).

    Expects dataloader to yield (input_ids, target_ids) tuples where:
    - input_ids: [B, S] token IDs
    - target_ids: [B, S] shifted target token IDs (or same as input_ids
      for teacher forcing with internal shift)

    Args:
        num_batches: Number of batches to evaluate.
        shift_labels: If True, shift labels internally (input_ids[:, 1:]
            as target, logits[:, :-1] as prediction). Default True.
    """

    def __init__(self, num_batches=5, shift_labels=True):
        self.num_batches = num_batches
        self.shift_labels = shift_labels

    def evaluate(self, model, dataloader, **kwargs):
        """Compute perplexity over the given data.

        Args:
            model: The language model (ANN, quantized, or SNN wrapper).
            dataloader: Yields (input_ids, target_ids) or (input_ids, labels).

        Returns:
            EvalResult with metrics={'perplexity': float, 'avg_loss': float}.
        """
        model.eval()
        device = next(model.parameters()).device

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                if i >= self.num_batches:
                    break

                if isinstance(batch_data, (tuple, list)):
                    input_ids = batch_data[0].to(device)
                    labels = batch_data[1].to(device) if len(batch_data) > 1 else input_ids.clone()
                else:
                    input_ids = batch_data.to(device)
                    labels = input_ids.clone()

                # Forward pass
                output = model(input_ids)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                elif isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output

                # For SNN wrapper, logits might be (accu, T) tuple
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                # Shift for next-token prediction
                if self.shift_labels:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                else:
                    shift_logits = logits
                    shift_labels = labels

                # Cross-entropy loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                # Count non-padding tokens
                num_tokens = (shift_labels != -100).sum().item()
                if num_tokens == 0:
                    num_tokens = shift_labels.numel()

                total_loss += loss.item()
                total_tokens += num_tokens

                # Reset SNN state
                inner = model.module if hasattr(model, "module") else model
                if isinstance(inner, SNNWrapper):
                    inner.reset()

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))  # cap to avoid overflow

        return EvalResult(
            metrics={
                "perplexity": perplexity,
                "avg_loss": avg_loss,
                "total_tokens": total_tokens,
            },
            details={},
        )
