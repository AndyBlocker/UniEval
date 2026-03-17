"""UniEval: 通用 SNN 转换与评估框架。"""

from .registry import (
    Registry,
    QUANTIZER_REGISTRY,
    NEURON_REGISTRY,
    EVALUATOR_REGISTRY,
    MODEL_PROFILE_REGISTRY,
)
from .snn.snnConverter.adapter import ADAPTER_REGISTRY
from .config import (
    UniEvalConfig,
    QuantConfig,
    ConversionConfig,
    EnergyConfig,
    EvalConfig,
)

# 子包入口函数
from .qann import quantize, calibrate_ptq
from .snn import convert
from .evaluation import evaluate_accuracy, evaluate_perplexity, evaluate_energy
