"""Example: QANN ViT-Small load, run, and energy evaluation.

Demonstrates the basic pipeline:
1. Create a ViT-Small model (ANN)
2. Apply PTQ quantization (ANN → QANN)
3. Convert to SNN (QANN → SNN)
4. Run energy evaluation

Usage:
    python samples/qann_vit_small_energy.py
"""

import torch
import torch.nn as nn

from unieval.config import UniEvalConfig
from unieval.engine.runner import UniEvalRunner


def make_dummy_dataloader(batch_size=4, img_size=224, num_batches=2):
    """Create a dummy dataloader for testing."""
    dataset = [
        (torch.randn(batch_size, 3, img_size, img_size),
         torch.randint(0, 1000, (batch_size,)))
        for _ in range(num_batches)
    ]
    return dataset


def main():
    # 1. Configure
    config = UniEvalConfig(
        model_name="vit_small",
        num_classes=1000,
        img_size=224,
        global_pool=True,
    )
    config.quant.level = 16
    config.quant.is_softmax = False
    config.conversion.time_step = 64
    config.conversion.encoding_type = "analog"
    config.conversion.level = 16
    config.conversion.neuron_type = "ST-BIF"
    config.conversion.is_softmax = False
    config.evaluation.num_batches = 2

    runner = UniEvalRunner(config)

    # 2. Create ANN model
    model = runner.create_model(act_layer=nn.ReLU)
    print(f"[ANN] Created {config.model_name} with {sum(p.numel() for p in model.parameters()):,} params")

    # 3. Quantize (ANN → QANN via PTQ)
    model = runner.quantize(model, quantizer_name="ptq")
    print("[QANN] PTQ quantization applied")

    # 4. Calibrate PTQ with dummy data
    dummy_loader = make_dummy_dataloader()
    model = model.to(config.device)
    runner.calibrate_ptq(model, dummy_loader, num_batches=1)
    model = model.cpu()
    print("[QANN] PTQ calibration done")

    # 5. Convert to SNN (QANN → SNN)
    wrapper = runner.convert(model)
    wrapper = wrapper.to(config.device)
    wrapper.eval()
    print("[SNN] Conversion and wrapping done")

    # 6. Run energy evaluation
    eval_loader = make_dummy_dataloader()
    energy_result = runner.evaluate_energy(wrapper, eval_loader)
    print(f"\n{energy_result}")


if __name__ == "__main__":
    main()
