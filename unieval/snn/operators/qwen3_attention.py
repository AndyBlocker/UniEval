"""Spiking Qwen3 Attention with GQA and softmax.

Thin subclass of DecoderSpikingAttentionBase that uses spiking_softmax
as the score activation.
"""

from .neurons import STBIFNeuron
from .attention import spiking_softmax
from .decoder_attention_base import DecoderSpikingAttentionBase


class SQwen3Attention(DecoderSpikingAttentionBase):
    """Spiking Qwen3 Attention with softmax, GQA, and RoPE.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of Q heads.
        num_kv_heads: Number of K/V heads.
        head_dim: Per-head dimension.
        rope: RotaryEmbedding from the ANN model.
        neuron_layer: Neuron class (default: STBIFNeuron).
        level: Quantization level.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rope,
        neuron_layer=STBIFNeuron,
        level=2,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope=rope,
            neuron_layer=neuron_layer,
            level=level,
        )
        self.scale = self.head_dim ** -0.5
        # registered under original name for consistency (no state_dict impact)
        self.Ssoftmax = spiking_softmax()

    @property
    def score_act(self):
        """Alias for base class forward()."""
        return self.Ssoftmax
