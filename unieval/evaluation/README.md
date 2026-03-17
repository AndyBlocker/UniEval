# Evaluation — 评估

观察式评估层，测量 SNN 性能，不参与模型表示变换。

## 结构

- `benchmarks/` — 任务指标（准确率、困惑度）
- `energy/` — 能耗分析（EnergyEvaluator、OpsCounter）
- `feasibility/` — spike 分析（spike_rate 检测）

## 约定

- Evaluation 可以依赖 ANN/QANN/SNN 的类型做检查，但**反向禁止**——领域层不应 import Evaluation
- `energy/` 需要引用具体的 SNN/QANN 算子类型来分类层和计算能耗，这是设计决策
- 新评估器继承 `BaseEvaluator`，用 `@EVALUATOR_REGISTRY.register("name")` 注册
- 新算子需在 `ops_counter.py` 的 `_build_default_modules_mapping()` 中注册 hook
