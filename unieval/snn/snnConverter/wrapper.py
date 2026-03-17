"""SNNWrapper, Judger, reset utilities, and attn_convert."""

import torch
import torch.nn as nn
from copy import deepcopy

from ..operators.base import SNNOperator
from ..operators.layers import LLLinear
from ...protocols import (
    is_decoder_model_like, is_uniaffine_model_like, is_qwen3_model_like,
)
from .converter import SNNConverter
from .adapter import ADAPTER_REGISTRY, auto_detect_adapter
from .threshold import transfer_threshold


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

    # Transfer 6 calibrated thresholds -> IF neurons
    transfer_threshold(QAttn.quan_q, SAttn.q_IF, neuron_type, level)
    transfer_threshold(QAttn.quan_k, SAttn.k_IF, neuron_type, level)
    transfer_threshold(QAttn.quan_v, SAttn.v_IF, neuron_type, level)
    transfer_threshold(QAttn.attn_quan, SAttn.attn_IF, neuron_type, level)
    transfer_threshold(QAttn.after_attn_quan, SAttn.after_attn_IF, neuron_type, level)
    transfer_threshold(QAttn.quan_proj, SAttn.proj_IF, neuron_type, level)

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
        model_name: DEPRECATED. Model name hint (adapter auto-detection
            now uses duck-typing via adapter.supports()). Kept for backward
            compatibility; ignored when adapter_name is specified or
            auto-detection succeeds.
        is_softmax: Whether attention uses softmax.
        converter: Optional SNNConverter instance.
        adapter_name: Execution adapter name (default: auto-detect via duck-typing).
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
        self.model_name = model_name  # kept for backward compat
        self.is_softmax = is_softmax
        self.max_T = 0
        self._current_t = 0

        # Initialize adapter via lifecycle API (before conversion)
        if adapter_name is not None:
            # Explicit adapter name
            self._init_adapter_by_name(adapter_name)
        else:
            # Auto-detect via duck-typing
            self.adapter = auto_detect_adapter(self.model)

        # Auto-detect converter based on model structure
        if converter is not None:
            conv = converter
        elif is_uniaffine_model_like(self.model):
            from .uniaffine_rules import UNIAFFINE_CONVERSION_RULES
            from .rules import DEFAULT_CONVERSION_RULES
            conv = SNNConverter(rules=UNIAFFINE_CONVERSION_RULES + DEFAULT_CONVERSION_RULES)
        elif is_qwen3_model_like(self.model):
            from .qwen3_rules import QWEN3_CONVERSION_RULES
            from .rules import DEFAULT_CONVERSION_RULES
            conv = SNNConverter(rules=QWEN3_CONVERSION_RULES + DEFAULT_CONVERSION_RULES)
        else:
            conv = SNNConverter()
        conv.convert(
            self.model,
            level=self.level,
            neuron_type=self.neuron_type,
            is_softmax=self.is_softmax,
        )

        # Initialize Judger (after conversion, so all SNN ops exist)
        self.finish_judger = Judger(self.model)

    def _init_adapter_by_name(self, adapter_name):
        """Initialize adapter by explicit name (backward compat path)."""
        if adapter_name in ADAPTER_REGISTRY:
            adapter_cls = ADAPTER_REGISTRY.get(adapter_name)
            self.adapter = adapter_cls()
            self.adapter.capture_context(self.model)
        else:
            from .adapter import DefaultExecutionAdapter
            self.adapter = DefaultExecutionAdapter()
            self.adapter.capture_context(self.model)

    # ----------------------------------------------------------------
    # State management
    # ----------------------------------------------------------------

    def reset(self):
        """Reset model state for new sample."""
        reset_model(self.model)
        self._current_t = 0
        # Delegate context restoration to adapter
        self.adapter.reset_context(self.model)

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

        Note: Must be called from reset state (_current_t == 0) when using
        ViT adapter, as the multistep path handles pos_embed/cls_token
        internally assuming t starts at 0.

        Args:
            x_seq: [T, B, ...] pre-encoded temporal sequence.

        Returns:
            output_seq: [T, B, ...] differential outputs per step.
        """
        if self._current_t != 0:
            # Fall back to step-by-step to respect _current_t offset
            outputs = []
            for t in range(x_seq.shape[0]):
                outputs.append(self.step_encoded(x_seq[t]))
            return torch.stack(outputs)
        output_seq = self.adapter.forward_multistep(self.model, x_seq)
        self._current_t += x_seq.shape[0]
        return output_seq

    # ----------------------------------------------------------------
    # High-level convenience API
    # ----------------------------------------------------------------

    def run_auto(self, x, verbose=False):
        """Auto encoding + Judger early stop.

        Equivalent to the original forward() behavior.
        Uses forward_encoded (multistep) path for decoder models when
        verbose=False and early_stop is not needed, giving a significant
        speedup by avoiding Python per-step overhead.

        For decoder models, x should be token IDs [B, S].
        The embedding is applied via adapter.prepare_input before temporal encoding.

        Args:
            x: Raw input tensor [B, ...] (images for ViT, token IDs for decoder).
            verbose: If True, return per-timestep accumulations.

        Returns:
            (accu, actual_T) or (accu, actual_T, accu_per_timestep) if verbose.
        """
        self.reset()

        # Pre-process input via adapter (e.g. token IDs -> embeddings for decoder)
        x = self.adapter.prepare_input(self.model, x)

        # Detect if this is a decoder adapter (has non-trivial prepare_input)
        _is_decoder = is_decoder_model_like(self.model)

        # Fast path: use forward_encoded for fixed-T without early stop or verbose
        # TODO: rate encoding 下此路径用 T=self.T 编码，而 step-by-step 路径
        # 用 self.level 个有效步 + 零填充到 self.T，语义不一致。
        if not verbose and _is_decoder:
            T = self.T
            x_seq = self.encode_sequence(x, T=T)
            output_seq = self.forward_encoded(x_seq)  # [T, B, ...]
            accu = output_seq.sum(dim=0)
            self.max_T = max(T, self.max_T)
            return accu, T

        # Legacy step-by-step path (ViT with early stop, or verbose mode)
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
