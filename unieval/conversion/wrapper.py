"""SNNWrapper, Judger, reset utilities, and attn_convert."""

import torch
import torch.nn as nn
from copy import deepcopy

from ..operators.base import SNNOperator
from ..operators.layers import LLLinear
from .converter import SNNConverter
from .adapter import ADAPTER_REGISTRY


def reset_model(model):
    """Reset all SNN operator states in the model via flat traversal."""
    for module in model.modules():
        if isinstance(module, SNNOperator):
            module.reset()


def attn_convert(QAttn, SAttn, level, neuron_type):
    """Transfer learned thresholds from QAttention to SAttention neurons.

    Maps quantization scale parameters to neuron firing thresholds.

    Args:
        QAttn: Source QAttention module.
        SAttn: Target SAttention module.
        level: Quantization level.
        neuron_type: Neuron type string.
    """
    SAttn.qkv = LLLinear(linear=QAttn.qkv, neuron_type="ST-BIF", level=level)
    SAttn.proj = LLLinear(linear=QAttn.proj, neuron_type="ST-BIF", level=level)

    # Transfer Q
    SAttn.q_IF.neuron_type = neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min
    SAttn.q_IF.is_init = False

    # Transfer K
    SAttn.k_IF.neuron_type = neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min
    SAttn.k_IF.is_init = False

    # Transfer V
    SAttn.v_IF.neuron_type = neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min
    SAttn.v_IF.is_init = False

    # Transfer attn
    SAttn.attn_IF.neuron_type = neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = QAttn.attn_quan.s.data
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min
    SAttn.attn_IF.is_init = False

    # Transfer after_attn
    SAttn.after_attn_IF.neuron_type = neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    SAttn.after_attn_IF.is_init = False

    # Transfer proj
    SAttn.proj_IF.neuron_type = neuron_type
    SAttn.proj_IF.level = level
    SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    SAttn.proj_IF.is_init = False

    # Transfer dropout
    SAttn.attn_drop = QAttn.attn_drop
    SAttn.proj_drop = QAttn.proj_drop


class Judger:
    """Determines when SNN inference is complete.

    Uses pre-cached module lists and attribute-driven checking via
    SNNOperator.participates_in_early_stop and SNNOperator.working.
    """

    def __init__(self, model):
        self.network_finish = True
        self._early_stop_ops = [
            m for m in model.modules()
            if isinstance(m, SNNOperator) and m.participates_in_early_stop
        ]

    def judge_finish(self):
        """Check if all early-stop operators have converged."""
        self.network_finish = all(
            not op.working for op in self._early_stop_ops
        )

    def should_stop(self):
        """Returns True if inference should stop (all operators converged)."""
        self.judge_finish()
        return self.network_finish

    def reset_network_finish_flag(self):
        self.network_finish = True


def get_subtensors(tensor, sample_grain=255):
    """Rate coding: divide input across timesteps."""
    pieces = []
    for _ in range(int(sample_grain)):
        pieces.append(tensor / sample_grain)
    return torch.stack(pieces, dim=0)


class SNNWrapper(nn.Module):
    """Wraps a quantized ANN model for temporal SNN inference.

    Provides three levels of API:
    - run_auto(x): High-level, auto encoding + Judger early stop
    - step_encoded(x_t): Low-level primitive, single step on pre-encoded input
    - forward_encoded(x_seq): Low-level, multi-step on pre-encoded sequence
    - forward(x): Legacy API, equivalent to run_auto

    Also provides:
    - encode_sequence(x, T): Convert raw input to explicit temporal sequence

    Args:
        ann_model: The quantized ANN model to convert.
        time_step: Maximum number of timesteps.
        encoding_type: "rate" or "analog".
        level: Quantization level.
        neuron_type: Neuron type string.
        model_name: Model name (used for adapter selection).
        is_softmax: Whether attention uses softmax.
        converter: Optional SNNConverter instance.
        adapter_name: Execution adapter name (default: auto-detect from model_name).
    """

    def __init__(
        self,
        ann_model,
        time_step=2000,
        encoding_type="rate",
        level=16,
        neuron_type="ST-BIF",
        model_name="vit",
        is_softmax=True,
        converter=None,
        adapter_name=None,
    ):
        super().__init__()
        self.T = time_step
        self.Encoding_type = encoding_type
        self.level = level
        self.neuron_type = neuron_type
        self.model = ann_model
        self.model_name = model_name
        self.is_softmax = is_softmax
        self.max_T = 0
        self._current_t = 0

        # Save ViT embeddings before conversion
        if "vit" in self.model_name or "deit" in self.model_name:
            self._saved_pos_embed = deepcopy(self.model.pos_embed.data)
            self._saved_cls_token = deepcopy(self.model.cls_token.data)

        # Convert using rule-based converter
        conv = converter or SNNConverter()
        conv.convert(
            self.model,
            level=self.level,
            neuron_type=self.neuron_type,
            is_softmax=self.is_softmax,
        )

        # Initialize adapter
        if adapter_name is None:
            adapter_name = self._detect_adapter_name()
        self._init_adapter(adapter_name)

        # Initialize Judger (after conversion, so all SNN ops exist)
        self.finish_judger = Judger(self.model)

    def _detect_adapter_name(self):
        """Auto-detect adapter based on model_name."""
        for prefix in ("vit", "deit"):
            if prefix in self.model_name.lower():
                return prefix
        return "default"

    def _init_adapter(self, adapter_name):
        """Initialize the execution adapter."""
        if adapter_name in ("vit", "deit"):
            adapter_cls = ADAPTER_REGISTRY.get(adapter_name)
            self.adapter = adapter_cls(
                pos_embed=self._saved_pos_embed,
                cls_token=self._saved_cls_token,
            )
        elif adapter_name in ADAPTER_REGISTRY:
            adapter_cls = ADAPTER_REGISTRY.get(adapter_name)
            self.adapter = adapter_cls()
        else:
            adapter_cls = ADAPTER_REGISTRY.get("default")
            self.adapter = adapter_cls()

    # ----------------------------------------------------------------
    # State management
    # ----------------------------------------------------------------

    def reset(self):
        """Reset model state for new sample."""
        reset_model(self.model)
        self._current_t = 0

    # ----------------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------------

    def encode_sequence(self, x, T, encoding_type=None):
        """Convert raw input to explicit temporal sequence.

        Args:
            x: Raw input tensor [B, ...].
            T: Number of timesteps.
            encoding_type: "analog" or "rate". Defaults to self.Encoding_type.

        Returns:
            x_seq: [T, B, ...] temporal sequence.
        """
        enc = encoding_type or self.Encoding_type
        if enc == "analog":
            x_seq = torch.zeros(T, *x.shape, device=x.device, dtype=x.dtype)
            x_seq[0] = x
            return x_seq
        elif enc == "rate":
            return get_subtensors(x, sample_grain=T)
        else:
            raise ValueError(f"Unknown encoding type: {enc}")

    # ----------------------------------------------------------------
    # Core low-level primitives
    # ----------------------------------------------------------------

    def step_encoded(self, x_t):
        """Process a single pre-encoded timestep.

        Does NOT do encoding or early-stop checking.
        User controls the loop, encoding, accumulation, and stopping.

        Args:
            x_t: Pre-encoded input for this timestep [B, ...].

        Returns:
            Differential output for this timestep [B, ...].
        """
        output = self.adapter.step(self.model, x_t, self._current_t)
        self._current_t += 1
        return output

    def forward_encoded(self, x_seq):
        """Process an explicit pre-encoded sequence.

        Uses adapter's multistep path if available.
        Returns per-step differential outputs; user sums to get final.

        Args:
            x_seq: [T, B, ...] pre-encoded temporal sequence.

        Returns:
            output_seq: [T, B, ...] differential outputs per step.
        """
        output_seq = self.adapter.forward_multistep(self.model, x_seq)
        self._current_t += x_seq.shape[0]
        return output_seq

    # ----------------------------------------------------------------
    # High-level convenience API
    # ----------------------------------------------------------------

    def run_auto(self, x, verbose=False):
        """Auto encoding + Judger early stop.

        Equivalent to the original forward() behavior.

        Args:
            x: Raw input tensor [B, ...].
            verbose: If True, return per-timestep accumulations.

        Returns:
            (accu, actual_T) or (accu, actual_T, accu_per_timestep) if verbose.
        """
        self.reset()
        accu = None
        count = 0
        accu_per_timestep = []

        if self.Encoding_type == "rate":
            x_seq = self.encode_sequence(x, T=self.level)

        while True:
            if count > 0 and self.finish_judger.should_stop():
                self.max_T = max(count, self.max_T)
                break
            if count >= self.T:
                self.max_T = max(count, self.max_T)
                break

            # Prepare input for this timestep
            if self.Encoding_type == "rate":
                if count < x_seq.shape[0]:
                    input_t = x_seq[count]
                else:
                    input_t = torch.zeros_like(x_seq[0])
            else:  # analog
                if count == 0:
                    input_t = x
                else:
                    input_t = torch.zeros_like(x)

            output = self.step_encoded(input_t)

            if count == 0:
                accu = output.clone()
            else:
                accu = accu + output

            if verbose:
                accu_per_timestep.append(accu.clone())

            count += 1
            if verbose and count % 100 == 0:
                print(count)

        if verbose:
            print(f"\nTime Step: {count}")

        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep, dim=0)
            return accu, count, accu_per_timestep
        else:
            return accu, count

    def forward(self, x, verbose=False):
        """Legacy API: equivalent to run_auto with auto-reset."""
        return self.run_auto(x, verbose=verbose)
