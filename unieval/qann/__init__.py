"""QANN: 量化 ANN 模块和量化策略。"""

import torch


def quantize(model, method="ptq", level=16, is_softmax=True,
             weight_bit=32, rules=None, **kwargs):
    """对 ANN 模型施加量化，返回 QANN 模型。

    Args:
        model: 原始 ANN 模型。
        method: 量化方法 ("lsq" 或 "ptq")。
        level: 量化级数。
        is_softmax: attention 是否使用 softmax。
        weight_bit: 权重量化位宽 (仅 LSQ，32=不量化)。
        rules: 自定义 QuantPlacementRule 列表，覆盖默认规则。

    Returns:
        量化后的模型 (in-place 修改并返回)。
    """
    if method == "lsq":
        from .quantization.lsq import LSQQuantizer
        quantizer = LSQQuantizer(
            level=level, weight_bit=weight_bit,
            is_softmax=is_softmax, rules=rules,
        )
    elif method == "ptq":
        from .quantization.ptq import PTQQuantizer
        quantizer = PTQQuantizer(
            level=level, is_softmax=is_softmax, rules=rules,
        )
    else:
        from ..registry import QUANTIZER_REGISTRY
        quantizer_cls = QUANTIZER_REGISTRY.get(method)
        quantizer = quantizer_cls(level=level, rules=rules, **kwargs)

    return quantizer.quantize_model(model)


def calibrate_ptq(model, dataloader, num_batches=10):
    """运行 PTQ 校准（PTQQuan 在首次 forward 时自动校准）。

    Args:
        model: 已量化的模型。
        dataloader: 校准数据，yield (input, target)。
        num_batches: 校准使用的 batch 数量。
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (batch, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            batch = batch.to(device)
            model(batch)
