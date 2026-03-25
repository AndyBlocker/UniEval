"""SNNConverter: rule-driven recursive ANN-to-SNN conversion engine."""

import warnings
from typing import Callable, Dict, List, Optional, Type

import torch.nn as nn

from .rules import ConversionRule

# Leaf module types that are expected to have no conversion rule.
_CONVERT_SKIP_WARN_TYPES = (
    nn.Identity, nn.Dropout, nn.Embedding,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm,
    nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    nn.Flatten, nn.Softmax, nn.Tanh, nn.Sigmoid,
)

# ---------------------------------------------------------------------------
# Typed dispatch: exact-type registry for QANN -> SNN conversion
# ---------------------------------------------------------------------------
# O(1) lookup by type(child).  Checked before ConversionRule iteration.
# Use @snn_convertible(QANNType) to register.

_TYPED_CONVERTERS: Dict[Type[nn.Module], Callable] = {}


def snn_convertible(qann_type: Type[nn.Module]):
    """Decorator: register a QANN type -> SNN conversion function.

    The decorated function must have the same signature as a ConversionRule
    convert_fn: ``(name, child, parent, **kwargs) -> None``.

    Typed dispatch is checked before rule-based matching in SNNConverter.
    It uses exact type matching (not isinstance), so subclasses are not
    caught unless explicitly registered.

    Example::

        @snn_convertible(QQwen3Attention)
        def _convert_qwen3_attn(name, child, parent, level, neuron_type, **kw):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        if qann_type in _TYPED_CONVERTERS:
            warnings.warn(
                f"snn_convertible: overwriting existing converter for "
                f"{qann_type.__name__}",
                stacklevel=2,
            )
        _TYPED_CONVERTERS[qann_type] = fn
        return fn
    return decorator


class ConversionContext:
    """Mutable state for a single convert() call.

    Provides named counters for layer numbering, replacing global variables.
    A fresh context is created per convert() invocation, ensuring reentrancy.
    """

    def __init__(self):
        self.counters = {}

    def next_index(self, prefix="layer"):
        """Return the next index for *prefix* and increment the counter."""
        count = self.counters.get(prefix, 0)
        self.counters[prefix] = count + 1
        return count


class SNNConverter:
    """Rule-based recursive model converter from ANN/QANN to SNN.

    Traverses the model tree and applies conversion rules in priority order.
    First matching rule wins for each module.

    Args:
        rules: List of ConversionRule instances.
            Defaults to UNIVERSAL_CONVERSION_RULES (all registered rules).
    """

    def __init__(self, rules: Optional[List[ConversionRule]] = None):
        if rules is None:
            from .rules import UNIVERSAL_CONVERSION_RULES
            rules = UNIVERSAL_CONVERSION_RULES
        self.rules = sorted(
            rules,
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
        ctx = ConversionContext()
        self._convert_recursive(model, ctx=ctx, **kwargs)
        self._warn_surviving_quantizers(model)
        return model

    def _convert_recursive(self, model: nn.Module, **kwargs):
        children = list(model.named_children())
        for name, child in children:
            # Fast path: typed dispatch (exact type match, O(1))
            converter_fn = _TYPED_CONVERTERS.get(type(child))
            if converter_fn is not None:
                converter_fn(name, child, model, **kwargs)
                continue

            # Slow path: rule-based matching (priority order)
            matched = False
            for rule in self.rules:
                if rule.match_fn(name, child, model):
                    rule.convert_fn(name, child, model, **kwargs)
                    matched = True
                    break
            if not matched:
                is_leaf = len(list(child.children())) == 0
                if is_leaf and not isinstance(child, _CONVERT_SKIP_WARN_TYPES):
                    warnings.warn(
                        f"UniEval: 叶模块 '{name}' ({type(child).__name__}) "
                        f"未被任何转换规则匹配，将保持原样。",
                        stacklevel=2,
                    )
                self._convert_recursive(child, **kwargs)

    def _warn_surviving_quantizers(self, model: nn.Module):
        """Warn if any quantization modules survived conversion."""
        from ...qann.operators.lsq import MyQuan
        from ...qann.operators.ptq import PTQQuan
        for name, module in model.named_modules():
            if isinstance(module, (MyQuan, PTQQuan)):
                warnings.warn(
                    f"量化模块 {type(module).__name__} 在转换后仍残留于 "
                    f"'{name}'，可能没有转换规则匹配到它，"
                    f"SNN 模型可能产生错误结果。",
                    RuntimeWarning,
                    stacklevel=2,
                )
