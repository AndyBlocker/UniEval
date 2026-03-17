"""SNNOperator mixin interface for all SNN operator modules."""

import torch


class SNNOperator:
    """Mixin interface for SNN operators.

    All SNN operator modules (neurons, layers, attention) should inherit from
    this mixin alongside nn.Module. Provides a standard interface for:
    - reset(): Reset temporal state between samples
    - is_work: Flag indicating if the operator is actively processing
    - forward_multistep(): Process a sequence [T, B, ...] of inputs
    - participates_in_early_stop: Whether Judger should check this operator

    Strong semantic contract for forward_multistep:
        forward_multistep(x_seq) ≡ torch.stack([self(x_seq[t]) for t in range(T)])
        Both output AND final state must be identical.
    """

    participates_in_early_stop = True

    def reset(self):
        """Reset temporal state. Must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def working(self):
        """Whether this operator is currently active.

        Defensive: if participates_in_early_stop=True but is_work is not
        defined, raise AttributeError to prevent silent false-convergence.
        """
        if self.participates_in_early_stop:
            if not hasattr(self, "is_work"):
                raise AttributeError(
                    f"{type(self).__name__} has participates_in_early_stop=True "
                    f"but does not define 'is_work'. Either implement is_work "
                    f"or set participates_in_early_stop=False."
                )
            return self.is_work
        return False

    def forward_multistep(self, x_seq):
        """Process a temporal sequence of inputs.

        Default implementation: sequential for-loop calling self(x_t).
        Subclasses may override with vectorized implementations.

        Semantic contract:
            Output and final state must be identical to sequential execution.
            Implementations must read current state as initial condition and
            sync final state at the end.

        Args:
            x_seq: [T, B, ...] tensor of temporal inputs.

        Returns:
            output_seq: [T, B, ...] tensor of temporal outputs.
        """
        return torch.stack([self(x_seq[t]) for t in range(x_seq.shape[0])])
