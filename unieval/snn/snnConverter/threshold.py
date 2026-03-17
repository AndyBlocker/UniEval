"""Shared threshold transfer utility for QANN->SNN conversion.

Extracts the common pattern of transferring quantizer scale parameters
to neuron firing thresholds, which was previously duplicated in
wrapper.py (attn_convert), uniaffine_rules.py, and qwen3_rules.py.
"""


def transfer_threshold(quan, neuron, neuron_type, level):
    """Transfer quantizer calibrated scale to IF neuron threshold.

    Maps the quantizer's step-size parameter (.s) and clipping bounds
    (.pos_max, .neg_min) to the neuron's firing threshold and spike
    tracer bounds, establishing the QANN<->SNN equivalence for that
    activation point.

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
    neuron.q_threshold.data = quan.s.data
    neuron.pos_max = quan.pos_max
    neuron.neg_min = quan.neg_min
    neuron.is_init = False
