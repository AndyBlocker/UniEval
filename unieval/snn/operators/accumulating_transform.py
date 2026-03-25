"""AccumulatingTransform: parameterized accumulate-diff SNN operator.

Eliminates duplicated forward/forward_multistep/reset logic across
Spiking_LayerNorm, Spiking_RMSNorm, Spiking_SiLU, Spiking_UnifiedClipNorm.

Usage:
    # Direct (new code):
    norm = Spiking_LayerNorm(dim=128)

    # Or compose directly:
    class MySpikingOp(AccumulatingTransform):
        def __init__(self, my_module):
            super().__init__()
            self.my_module = my_module
            self._transform_attr = "my_module"
"""

import torch
import torch.nn as nn
from typing import Callable, Optional

from .base import SNNOperator


class AccumulatingTransform(nn.Module, SNNOperator):
    """Accumulate-diff operator: X_acc += input, Y = transform(X_acc), output = Y - Y_pre.

    Provides vectorized forward_multistep (cumsum + transform + diff) out of the box.
    Subclasses should NOT override forward/forward_multistep/reset.

    Two transform backends:
    - module-backed: subclass registers an nn.Module attribute and sets _transform_attr
      to the attribute name (e.g. self.layernorm = nn.LayerNorm(dim); _transform_attr = "layernorm")
    - callable-backed: subclass sets self._transform_fn = F.silu in __init__

    Constraint: transform must be element-wise or transparent to arbitrary leading
    dimensions. forward_multistep calls transform on [T, B, ...] tensors. Transforms
    that depend on dim=0 semantics (e.g. BatchNorm), carry internal state, or require
    extra arguments (e.g. mask) are NOT suitable for this class.
    """

    participates_in_early_stop = False
    _transform_attr: Optional[str] = None
    _transform_fn: Optional[Callable] = None

    def __init__(self):
        super().__init__()
        self._acc: Optional[torch.Tensor] = None
        self._prev: Optional[torch.Tensor] = None

    def _apply_transform(self, x):
        if self._transform_attr is not None:
            if self._transform_fn is not None:
                raise TypeError(
                    f"{type(self).__name__}: set _transform_attr or _transform_fn, not both"
                )
            return getattr(self, self._transform_attr)(x)
        if self._transform_fn is not None:
            return self._transform_fn(x)
        raise NotImplementedError(
            f"{type(self).__name__}: set _transform_attr or _transform_fn in __init__"
        )

    def forward(self, input):
        self._acc = input if self._acc is None else self._acc + input
        y = self._apply_transform(self._acc)
        prev = self._prev.detach().clone() if self._prev is not None else 0.0
        self._prev = y
        return y - prev

    def forward_multistep(self, x_seq):
        """Vectorized multi-step: cumsum + transform + diff.

        Args:
            x_seq: [T, ...] tensor where dim=0 is the time axis.
        Returns:
            output: [T, ...] differential outputs.
        """
        if self._acc is not None:
            X_cum = x_seq.cumsum(dim=0) + self._acc
        else:
            X_cum = x_seq.cumsum(dim=0)
        Y = self._apply_transform(X_cum)
        if self._prev is not None:
            Y_prev = self._prev.detach().clone().unsqueeze(0)
        else:
            Y_prev = torch.zeros_like(Y[:1])
        Y_shifted = torch.cat([Y_prev, Y[:-1]], dim=0)
        output = Y - Y_shifted
        self._acc = X_cum[-1]
        self._prev = Y[-1]
        return output

    def reset(self):
        self._acc = None
        self._prev = None

    # --- Backward-compatible property aliases ---
    # External code (tests, energy evaluator) accesses .X and .Y_pre directly.

    @property
    def X(self):
        return self._acc if self._acc is not None else 0.0

    @X.setter
    def X(self, v):
        if torch.is_tensor(v):
            self._acc = v
        else:
            self._acc = None if v == 0.0 else v

    @property
    def Y_pre(self):
        return self._prev

    @Y_pre.setter
    def Y_pre(self, v):
        self._prev = v
