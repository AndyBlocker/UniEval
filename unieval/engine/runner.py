"""UniEvalRunner: main entry point for the full evaluation pipeline."""

import torch
import torch.nn as nn

from ..config import UniEvalConfig
from ..registry import QUANTIZER_REGISTRY, MODEL_PROFILE_REGISTRY, EVALUATOR_REGISTRY
from ..QANN.quantization.lsq import LSQQuantizer
from ..QANN.quantization.ptq import PTQQuantizer
from ..SNN.snnConverter.converter import SNNConverter
from ..SNN.snnConverter.wrapper import SNNWrapper
from ..Evaluation.benchmarks.accuracy import AccuracyEvaluator
from ..Evaluation.energy.energy import EnergyEvaluator
from ..Evaluation.energy.ops_counter import OpsCounter
from ..ANN.models.base import ModelProfile
from ..ANN.models.vit import (
    vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14,
    vit_small_patch16_dvs,
)
from ..ANN.models.uniaffine import uniaffine_model, UniAffineConfig
from ..ANN.models.qwen3 import qwen3_model, Qwen3Config


# Model factory mapping
MODEL_FACTORIES = {
    "vit_small": vit_small_patch16,
    "vit_base": vit_base_patch16,
    "vit_large": vit_large_patch16,
    "vit_huge": vit_huge_patch14,
    "vit_small_dvs": vit_small_patch16_dvs,
    "uniaffine": uniaffine_model,
    "qwen3": qwen3_model,
}

# Config classes for decoder models
_DECODER_CONFIG_CLASSES = {
    "uniaffine": UniAffineConfig,
    "qwen3": Qwen3Config,
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

        # Decoder models use their own config, not ViT-style kwargs
        if self.config.model_name in _DECODER_CONFIG_CLASSES:
            config_cls = _DECODER_CONFIG_CLASSES[self.config.model_name]
            ua_kwargs = {k: v for k, v in kwargs.items()
                          if k in config_cls.__dataclass_fields__}
            model = factory(**ua_kwargs)
        else:
            model = factory(
                img_size=self.config.img_size,
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
            # Use model-specific PTQ rules for decoder models
            rules = None
            if self.config.model_name == "uniaffine":
                from ..QANN.quantization.uniaffine_rules import UNIAFFINE_PTQ_RULES
                rules = UNIAFFINE_PTQ_RULES
            elif self.config.model_name == "qwen3":
                from ..QANN.quantization.qwen3_rules import QWEN3_PTQ_RULES
                rules = QWEN3_PTQ_RULES
            quantizer = PTQQuantizer(
                level=qcfg.level,
                is_softmax=qcfg.is_softmax,
                rules=rules,
            )
        else:
            quantizer_cls = QUANTIZER_REGISTRY.get(quantizer_name)
            quantizer = quantizer_cls(level=qcfg.level, **kwargs)

        return quantizer.quantize_model(model)

    def calibrate_ptq(self, model, dataloader, num_batches=10):
        """Run PTQ calibration by forwarding data through the quantized model.

        PTQQuan modules automatically calibrate on their first forward pass
        using KL-divergence threshold optimization.

        Args:
            model: The quantized model (with PTQQuan modules).
            dataloader: Calibration data.
            num_batches: Number of batches for calibration.
        """
        model.eval()
        device = next(model.parameters()).device
        is_decoder = self.config.model_name in _DECODER_CONFIG_CLASSES
        with torch.no_grad():
            for i, (batch, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                if is_decoder:
                    batch = batch.long().to(device)
                else:
                    batch = batch.float().to(device)
                model(batch)

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

        time_step = self.config.conversion.time_step
        ops_counter = OpsCounter(time_step=time_step)

        evaluator = EnergyEvaluator(
            energy_config=self.config.energy,
            model_profile=profile,
            ops_counter=ops_counter,
            num_batches=self.config.evaluation.num_batches,
        )
        return evaluator.evaluate(model, dataloader, **kwargs)

    def run_full_pipeline(self, dataloader, act_layer=nn.ReLU,
                          quantizer_name="lsq", checkpoint_path=None,
                          calibration_dataloader=None):
        """Run the complete pipeline: create -> quantize -> convert -> evaluate.

        Args:
            dataloader: Test data.
            act_layer: Activation layer for model creation.
            quantizer_name: Quantizer to use.
            checkpoint_path: Optional checkpoint to load.
            calibration_dataloader: Optional separate dataloader for PTQ
                calibration. If None and quantizer is "ptq", uses the test
                dataloader (PTQQuan auto-calibrates on first forward).

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
            # Apply checkpoint key mapping for decoder models
            if self.config.model_name == "uniaffine":
                from ..ANN.models.uniaffine import convert_megatron_state_dict
                sample_key = next(iter(state_dict.keys()), "")
                if ("decoder.layers" in sample_key or
                        "embedding.word_embeddings" in sample_key):
                    state_dict = convert_megatron_state_dict(
                        state_dict, model.config
                    )
                    state_dict = {k: v for k, v in state_dict.items()
                                  if not k.startswith("_unmapped_")}
            elif self.config.model_name == "qwen3":
                from ..ANN.models.qwen3 import convert_hf_qwen3_state_dict
                sample_key = next(iter(state_dict.keys()), "")
                if ("layers." in sample_key or "embed_tokens" in sample_key):
                    state_dict = convert_hf_qwen3_state_dict(
                        state_dict, model.config
                    )
            model.load_state_dict(state_dict, strict=False)

        # 3. Quantize
        model = self.quantize(model, quantizer_name=quantizer_name)

        # 3b. PTQ calibration if needed
        if quantizer_name == "ptq" and calibration_dataloader is not None:
            device = self.config.device
            model = model.to(device)
            self.calibrate_ptq(model, calibration_dataloader)

        # 4. Convert to SNN
        wrapper = self.convert(model)

        # 5. Move to device and set eval mode
        device = self.config.device
        wrapper = wrapper.to(device)
        wrapper.eval()

        # 6. Evaluate
        acc_result = self.evaluate_accuracy(wrapper, dataloader)
        energy_result = self.evaluate_energy(wrapper, dataloader)

        return acc_result, energy_result, wrapper
