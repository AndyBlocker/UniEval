"""UniEvalRunner: main entry point for the full evaluation pipeline."""

import torch
import torch.nn as nn

from ..config import UniEvalConfig
from ..registry import QUANTIZER_REGISTRY, MODEL_PROFILE_REGISTRY, EVALUATOR_REGISTRY
from ..quantization.lsq import LSQQuantizer
from ..quantization.ptq import PTQQuantizer
from ..conversion.converter import SNNConverter
from ..conversion.wrapper import SNNWrapper
from ..evaluation.accuracy import AccuracyEvaluator
from ..evaluation.energy import EnergyEvaluator
from ..evaluation.ops_counter import OpsCounter
from ..models.base import ModelProfile
from ..models.vit import (
    vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14,
)


# Model factory mapping
MODEL_FACTORIES = {
    "vit_small": vit_small_patch16,
    "vit_base": vit_base_patch16,
    "vit_large": vit_large_patch16,
    "vit_huge": vit_huge_patch14,
}


class UniEvalRunner:
    """Orchestrates the full UniEval pipeline.

    Pipeline: create model -> quantize -> convert -> wrap -> evaluate

    Args:
        config: UniEvalConfig with all settings.
    """

    def __init__(self, config: UniEvalConfig):
        self.config = config

    def create_model(self, act_layer=nn.ReLU, **kwargs):
        """Create a base model from config.

        Args:
            act_layer: Activation layer class.

        Returns:
            The created model.
        """
        factory = MODEL_FACTORIES.get(self.config.model_name)
        if factory is None:
            raise ValueError(
                f"Unknown model: {self.config.model_name}. "
                f"Available: {list(MODEL_FACTORIES.keys())}"
            )
        model = factory(
            num_classes=self.config.num_classes,
            global_pool=self.config.global_pool,
            act_layer=act_layer,
            **kwargs,
        )
        return model

    def quantize(self, model, quantizer_name="lsq", **kwargs):
        """Apply quantization to the model.

        Args:
            model: The model to quantize.
            quantizer_name: Registered quantizer name ("lsq" or "ptq").

        Returns:
            Quantized model.
        """
        qcfg = self.config.quant
        if quantizer_name == "lsq":
            quantizer = LSQQuantizer(
                level=qcfg.level,
                weight_bit=qcfg.weight_bit,
                is_softmax=qcfg.is_softmax,
            )
        elif quantizer_name == "ptq":
            quantizer = PTQQuantizer(
                level=qcfg.level,
                is_softmax=qcfg.is_softmax,
            )
        else:
            quantizer_cls = QUANTIZER_REGISTRY.get(quantizer_name)
            quantizer = quantizer_cls(level=qcfg.level, **kwargs)

        return quantizer.quantize_model(model)

    def convert(self, model, converter=None):
        """Convert quantized model to SNN wrapped model.

        Args:
            model: Quantized model.
            converter: Optional SNNConverter instance.

        Returns:
            SNNWrapper instance.
        """
        ccfg = self.config.conversion
        wrapper = SNNWrapper(
            ann_model=model,
            time_step=ccfg.time_step,
            encoding_type=ccfg.encoding_type,
            level=ccfg.level,
            neuron_type=ccfg.neuron_type,
            model_name=self.config.model_name,
            is_softmax=ccfg.is_softmax,
            converter=converter,
        )
        return wrapper

    def evaluate_accuracy(self, model, dataloader, **kwargs):
        """Run accuracy evaluation.

        Args:
            model: The SNN model.
            dataloader: Test data.

        Returns:
            EvalResult with accuracy metrics.
        """
        evaluator = AccuracyEvaluator(
            topk=tuple(self.config.evaluation.topk),
            num_batches=self.config.evaluation.num_batches,
        )
        return evaluator.evaluate(model, dataloader, **kwargs)

    def evaluate_energy(self, model, dataloader, **kwargs):
        """Run energy evaluation.

        Args:
            model: The SNN model.
            dataloader: Test data.

        Returns:
            EvalResult with energy metrics.
        """
        profile_name = self.config.model_name
        if profile_name in MODEL_PROFILE_REGISTRY:
            profile = MODEL_PROFILE_REGISTRY.get(profile_name)
            # Update time_steps from conversion config
            profile.time_steps = self.config.conversion.time_step
        else:
            profile = None

        evaluator = EnergyEvaluator(
            energy_config=self.config.energy,
            model_profile=profile,
            num_batches=self.config.evaluation.num_batches,
        )
        return evaluator.evaluate(model, dataloader, **kwargs)

    def run_full_pipeline(self, dataloader, act_layer=nn.ReLU,
                          quantizer_name="lsq", checkpoint_path=None):
        """Run the complete pipeline: create -> quantize -> convert -> evaluate.

        Args:
            dataloader: Test data.
            act_layer: Activation layer for model creation.
            quantizer_name: Quantizer to use.
            checkpoint_path: Optional checkpoint to load.

        Returns:
            Tuple of (accuracy_result, energy_result, model).
        """
        # 1. Create model
        model = self.create_model(act_layer=act_layer)

        # 2. Load checkpoint if provided
        cp = checkpoint_path or self.config.checkpoint_path
        if cp:
            state_dict = torch.load(cp, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)

        # 3. Quantize
        model = self.quantize(model, quantizer_name=quantizer_name)

        # 4. Convert to SNN
        wrapper = self.convert(model)

        # 5. Move to device
        device = self.config.device
        wrapper = wrapper.to(device)

        # 6. Evaluate
        acc_result = self.evaluate_accuracy(wrapper, dataloader)
        energy_result = self.evaluate_energy(wrapper, dataloader)

        return acc_result, energy_result, wrapper
