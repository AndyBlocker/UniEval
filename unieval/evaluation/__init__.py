"""Evaluation: 准确率、能耗、困惑度、spike rate 评估。"""

from .benchmarks.base import EvalResult, BaseEvaluator
from .benchmarks.accuracy import AccuracyEvaluator
from .energy.energy import EnergyEvaluator
from .energy.ops_counter import OpsCounter
from .feasibility.spike_utils import spike_rate
from .simulator_fast.ops_hookers import OpsCounterFast as OpsCounterFastSim
from .simulator_fast.simulator import FastSimulator


def evaluate_accuracy(model, dataloader, topk=(1, 5), num_batches=None):
    """评估 top-k 准确率。

    Args:
        model: SNN 模型 (或任意可 forward 的模型)。
        dataloader: 测试数据，yield (input, target)。
        topk: 计算哪些 top-k 准确率。
        num_batches: 最多评估多少 batch (None=全部)。

    Returns:
        EvalResult，metrics 包含 top1, top5 等。
    """
    evaluator = AccuracyEvaluator(topk=topk, num_batches=num_batches)
    return evaluator.evaluate(model, dataloader)


def evaluate_perplexity(model, dataloader, num_batches=5, shift_labels=True):
    """评估语言模型困惑度。

    Args:
        model: 语言模型 (ANN/QANN/SNN)。
        dataloader: yield (input_ids, target_ids)。
        num_batches: 最多评估多少 batch。
        shift_labels: 是否内部 shift labels。

    Returns:
        EvalResult，metrics 包含 perplexity, avg_loss。
    """
    from .benchmarks.perplexity import PerplexityEvaluator
    evaluator = PerplexityEvaluator(
        num_batches=num_batches, shift_labels=shift_labels,
    )
    return evaluator.evaluate(model, dataloader)


def evaluate_energy(model, dataloader, profile=None, time_step=64,
                    energy_config=None, num_batches=5):
    """评估 SNN 能耗。

    Args:
        model: SNN 模型。
        dataloader: 测试数据。
        profile: ModelProfile 实例或注册名 (如 "vit_small")。
        time_step: 每次 forward 的时间步数。
        energy_config: EnergyConfig 实例（None 使用默认值）。
        num_batches: 最多评估多少 batch。

    Returns:
        EvalResult，metrics 包含 energy_mJ, mac_ops_G, ac_ops_G 等。
    """
    from ..config import EnergyConfig
    from ..registry import MODEL_PROFILE_REGISTRY

    if isinstance(profile, str):
        profile = MODEL_PROFILE_REGISTRY.get(profile)
        profile.time_steps = time_step

    ops_counter = OpsCounter(time_step=time_step)
    evaluator = EnergyEvaluator(
        energy_config=energy_config or EnergyConfig(),
        model_profile=profile,
        ops_counter=ops_counter,
        num_batches=num_batches,
    )
    return evaluator.evaluate(model, dataloader)


def simulater_fast_evaluate(model, dataloader, profile=None, time_step=64,
                    hardware_config=None, num_batches=5):
    """评估 SNN 硬件仿真能耗/面积/延迟。

    Args:
        model: SNN 模型。
        dataloader: 测试数据。
        profile: ModelProfile 实例或注册名 (如 "vit_small")。
        time_step: 每次 forward 的时间步数。
        hardware_config: 硬件参数字典（从 hardware_param.yaml 加载）。
        num_batches: 最多评估多少 batch。

    Returns:
        dict，包含 total_energy, total_area, total_latency 等。
    """
    from ..config import EnergyConfig
    from ..registry import MODEL_PROFILE_REGISTRY

    if isinstance(profile, str):
        profile = MODEL_PROFILE_REGISTRY.get(profile)
        profile.time_steps = time_step

    ops_counter = OpsCounterFastSim(time_step=time_step)
    evaluator = FastSimulator(
        hardware_config=hardware_config,
        model_profile=profile,
        ops_counter=ops_counter,
        num_batches=num_batches,
    )
    return evaluator.evaluate(model, dataloader)
