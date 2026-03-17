"""示例：QANN ViT-Small 加载、运行和能耗评估。

演示基本 pipeline：
1. 创建 ViT-Small 模型 (ANN)
2. PTQ 量化 (ANN → QANN)
3. 转换为 SNN (QANN → SNN)
4. 能耗评估

Usage:
    python samples/qann_vit_small_energy.py
"""

import torch
import torch.nn as nn

from unieval.ann.models.vit import vit_small_patch16
from unieval.qann import quantize, calibrate_ptq
from unieval.snn import convert
from unieval.evaluation import evaluate_energy


def make_dummy_dataloader(batch_size=4, img_size=224, num_batches=2):
    """创建用于测试的 dummy dataloader。"""
    return [
        (torch.randn(batch_size, 3, img_size, img_size),
         torch.randint(0, 1000, (batch_size,)))
        for _ in range(num_batches)
    ]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 创建 ANN 模型
    model = vit_small_patch16(num_classes=1000, global_pool=True, act_layer=nn.ReLU)
    print(f"[ANN] ViT-Small, {sum(p.numel() for p in model.parameters()):,} params")

    # 2. 量化 (ANN → QANN)
    model = quantize(model, method="ptq", level=16, is_softmax=False)
    print("[QANN] PTQ 量化完成")

    # 3. PTQ 校准
    model = model.to(device)
    calibrate_ptq(model, make_dummy_dataloader(), num_batches=1)
    model = model.cpu()
    print("[QANN] PTQ 校准完成")

    # 4. 转换为 SNN
    wrapper = convert(model, time_step=64, level=16, is_softmax=False)
    wrapper = wrapper.to(device)
    wrapper.eval()
    print("[SNN] 转换完成")

    # 5. 能耗评估
    result = evaluate_energy(
        wrapper, make_dummy_dataloader(),
        profile="vit_small", time_step=64, num_batches=2,
    )
    print(f"\n{result}")


if __name__ == "__main__":
    main()
