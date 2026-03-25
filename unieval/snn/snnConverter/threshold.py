"""Shared threshold transfer utility for QANN->SNN conversion.

Extracts the common pattern of transferring quantizer scale parameters
to neuron firing thresholds, which was previously duplicated in
wrapper.py (attn_convert), uniaffine_rules.py, and qwen3_rules.py.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class QuantSpec:
    """Explicit neuron configuration extracted from a QANN quantizer.

    Captures the three parameters needed to configure an SNN neuron's
    firing behavior: threshold (step size), and spike tracer bounds
    (pos_max, neg_min).

    Contract: all fields are Tensors. String sentinels (e.g. "full")
    are rejected at construction time.
    """
    threshold: Tensor
    pos_max: Tensor
    neg_min: Tensor

    @classmethod
    def from_quantizer(cls, quan, *, non_negative=False):
        """Extract neuron config from a PTQQuan or MyQuan quantizer.

        Args:
            quan: Quantization module with .s, .pos_max, .neg_min attributes.
            non_negative: If True, clamp neg_min to zero (for ReLU-gated paths).

        Raises:
            ValueError: If pos_max or neg_min is not a Tensor (e.g. "full" sentinel).
        """
        threshold = quan.s.data.clone()
        pos_max = quan.pos_max
        neg_min = quan.neg_min

        if isinstance(pos_max, str) or isinstance(neg_min, str):
            raise ValueError(
                f"QuantSpec: pos_max/neg_min must be Tensor, "
                f"got pos_max={type(pos_max).__name__}, neg_min={type(neg_min).__name__}"
            )

        if non_negative:
            neg_min = torch.zeros_like(neg_min) if torch.is_tensor(neg_min) else torch.tensor(0)

        return cls(threshold=threshold, pos_max=pos_max, neg_min=neg_min)

    def apply_to(self, neuron):
        """Configure a neuron's firing parameters from this spec.

        Sets q_threshold, pos_max, neg_min, and marks the neuron as
        initialized (is_init=False means "calibrated, skip auto-init").
        """
        neuron.q_threshold.data = self.threshold
        neuron.pos_max = self.pos_max
        neuron.neg_min = self.neg_min
        neuron.is_init = False


def transfer_threshold(quan, neuron, neuron_type, level):
    """Transfer quantizer calibrated scale to IF neuron threshold.

    This is a thin wrapper around QuantSpec for backward compatibility.
    New code should prefer QuantSpec.from_quantizer().apply_to().

    Args:
        quan: Source quantization module (PTQQuan or MyQuan) with .s,
            .pos_max, .neg_min attributes.
        neuron: Target IF neuron with .q_threshold parameter and
            .pos_max, .neg_min, .neuron_type, .level, .is_init attributes.
        neuron_type: Neuron type string (e.g. "ST-BIF").
        level: Quantization level.
    """
    neuron.neuron_type = neuron_type
    neuron.level = level
    spec = QuantSpec.from_quantizer(quan)
    spec.apply_to(neuron)
