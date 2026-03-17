"""SNN: 脉冲神经网络算子和 ANN→SNN 转换。"""


def convert(qann_model, time_step=64, level=16, encoding_type="analog",
            neuron_type="ST-BIF", is_softmax=True,
            converter=None, adapter_name=None):
    """将量化模型转换为 SNN，返回 SNNWrapper。

    Args:
        qann_model: 量化后的 ANN 模型。
        time_step: 最大时间步数。
        level: 量化级数。
        encoding_type: "analog" 或 "rate"。
        neuron_type: 神经元类型 (如 "ST-BIF")。
        is_softmax: attention 是否使用 softmax。
        converter: 自定义 SNNConverter 实例（None 则自动检测）。
        adapter_name: 执行适配器名称（None 则自动检测）。

    Returns:
        SNNWrapper 实例。
    """
    from .snnConverter.wrapper import SNNWrapper

    return SNNWrapper(
        ann_model=qann_model,
        time_step=time_step,
        encoding_type=encoding_type,
        level=level,
        neuron_type=neuron_type,
        is_softmax=is_softmax,
        converter=converter,
        adapter_name=adapter_name,
    )
