"""SNNConverter: rule-driven recursive ANN-to-SNN conversion engine."""

from typing import List, Optional

import torch.nn as nn

from .rules import ConversionRule, DEFAULT_CONVERSION_RULES


class SNNConverter:
    """Rule-based recursive model converter from ANN/QANN to SNN.

    Traverses the model tree and applies conversion rules in priority order.
    First matching rule wins for each module.

    Args:
        rules: List of ConversionRule instances. Defaults to DEFAULT_CONVERSION_RULES.
    """

    def __init__(self, rules: Optional[List[ConversionRule]] = None):
        self.rules = sorted(
            rules or DEFAULT_CONVERSION_RULES,
            key=lambda r: r.priority,
            reverse=True,
        )

    def convert(self, model: nn.Module, **kwargs) -> nn.Module:
        """Recursively convert model in-place.

        Args:
            model: The model to convert.
            **kwargs: Passed to convert_fn (level, neuron_type, is_softmax, etc.)

        Returns:
            The converted model (same object, modified in-place).
        """
        self._convert_recursive(model, **kwargs)
        return model

    def _convert_recursive(self, model: nn.Module, **kwargs):
        children = list(model.named_children())
        for name, child in children:
            matched = False
            for rule in self.rules:
                if rule.match_fn(name, child, model):
                    rule.convert_fn(name, child, model, **kwargs)
                    matched = True
                    break
            if not matched:
                self._convert_recursive(child, **kwargs)
