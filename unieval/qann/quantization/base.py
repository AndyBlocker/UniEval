"""Base quantizer ABC and QuantPlacementRule system."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn

# Leaf module types that are expected to have no quantization rule.
_QUANT_SKIP_WARN_TYPES = (
    nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU,
    nn.Dropout, nn.Identity, nn.Embedding,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm,
    nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    nn.Flatten, nn.Softmax, nn.Tanh, nn.Sigmoid,
)


@dataclass
class QuantPlacementRule:
    """A rule for placing quantization modules in the model tree.

    Args:
        name: Human-readable rule name.
        match_fn: (name, module, parent) -> bool, duck-typed matching.
        apply_fn: (name, module, parent, level, is_softmax) -> None, in-place modification.
    """
    name: str
    match_fn: Callable
    apply_fn: Callable


class BaseQuantizer(ABC):
    """Abstract base class for quantizers.

    Subclasses must implement quantize_model() which takes a model
    and applies quantization in-place.
    """

    @abstractmethod
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply quantization to the model in-place and return it."""
        ...

    def _apply_rules(self, model, rules, **kwargs):
        """Recursively apply placement rules to the model tree.

        Traverses the module tree. For each (name, child), iterates rules
        in order. First match wins: apply_fn is called and recursion stops
        for that child.
        """
        children = list(model.named_children())
        for name, child in children:
            matched = False
            for rule in rules:
                if rule.match_fn(name, child, model):
                    rule.apply_fn(name, child, model, **kwargs)
                    matched = True
                    break
            if not matched:
                is_leaf = len(list(child.children())) == 0
                if is_leaf and not isinstance(child, _QUANT_SKIP_WARN_TYPES):
                    warnings.warn(
                        f"UniEval: 叶模块 '{name}' ({type(child).__name__}) "
                        f"未被任何量化规则匹配，将保持原样。",
                        stacklevel=2,
                    )
                self._apply_rules(child, rules, **kwargs)
