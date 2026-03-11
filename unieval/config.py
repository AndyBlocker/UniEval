"""Dataclass-based configuration system for UniEval."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EnergyConfig:
    """Energy model parameters."""
    e_mac: float = 4.6    # pJ per MAC operation
    e_ac: float = 0.9     # pJ per AC operation
    nspks_max: int = 2    # max spike levels for spike detection


@dataclass
class QuantConfig:
    """Quantization configuration."""
    level: int = 16
    weight_bit: int = 32
    sym: bool = True
    is_softmax: bool = True


@dataclass
class ConversionConfig:
    """ANN-to-SNN conversion configuration."""
    neuron_type: str = "ST-BIF"
    level: int = 16
    time_step: int = 64
    encoding_type: str = "analog"  # "analog" or "rate"
    is_softmax: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_batches: int = 5
    topk: List[int] = field(default_factory=lambda: [1, 5])


@dataclass
class UniEvalConfig:
    """Top-level configuration for UniEval pipeline."""
    model_name: str = "vit_small"
    num_classes: int = 1000
    global_pool: bool = True
    quant: QuantConfig = field(default_factory=QuantConfig)
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
