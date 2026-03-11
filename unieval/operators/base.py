"""SNNOperator mixin interface for all SNN operator modules."""

import torch.nn as nn


class SNNOperator:
    """Mixin interface for SNN operators.

    All SNN operator modules (neurons, layers, attention) should inherit from
    this mixin alongside nn.Module. Provides a standard interface for:
    - reset(): Reset temporal state between samples
    - is_work: Flag indicating if the operator is actively processing
    """

    def reset(self):
        """Reset temporal state. Must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def working(self):
        """Whether this operator is currently active."""
        return getattr(self, "is_work", False)
