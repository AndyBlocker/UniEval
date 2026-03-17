"""Base evaluation classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch.nn as nn


@dataclass
class EvalResult:
    """Container for evaluation results.

    Attributes:
        metrics: Dictionary of metric name -> value.
        details: Optional dictionary of detailed per-layer or per-timestep info.
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        lines = ["EvalResult:"]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""

    @abstractmethod
    def evaluate(self, model: nn.Module, dataloader, **kwargs) -> EvalResult:
        """Run evaluation and return results."""
        ...
