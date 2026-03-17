# unieval

通用 SNN 转换与评估框架，源自 SpikeZIP-TF (ICML 2024)。

## 流程

```
ANN 模型 → 量化 → QANN 模型 → [校准] → 转换 → SNN Wrapper → 评估
```

## 子包

| 目录 | 职责 |
|------|------|
| `ANN/` | 纯 ANN 模型定义 (ViT, Qwen3, UniAffine) 和共享算子 (RoPE) |
| `QANN/` | 量化模块和量化策略 |
| `SNN/` | SNN 算子和 ANN→SNN 转换引擎 |
| `Evaluation/` | 准确率、能耗、spike rate 评估 |
| `engine/` | `UniEvalRunner` 高层编排入口 |

## 包根文件

| 文件 | 职责 |
|------|------|
| `registry.py` | 通用注册表 |
| `config.py` | Dataclass 配置 |
| `protocols.py` | Duck-typing 结构谓词 (`is_*_like`) |

## 快速开始

```python
from unieval.config import UniEvalConfig
from unieval.engine.runner import UniEvalRunner

runner = UniEvalRunner(UniEvalConfig(model_name="vit_small"))
model = runner.create_model()
model = runner.quantize(model, quantizer_name="ptq")
wrapper = runner.convert(model)
```

完整示例见 `samples/`。

## 开发指南

根据你要做的事情，查阅对应的文档和目录：

| 我想要… | 需要改动的位置 | 参考文档 |
|---------|--------------|---------|
| **添加新模型族** (如新 decoder 架构) | `ANN/models/` 新建模型 → `protocols.py` 添加谓词 → `QANN/quantization/` PTQ rules → `SNN/snnConverter/` 转换 rules → `engine/runner.py` 注册 | [ANN/README](ANN/README.md)、[QANN/README](QANN/README.md)、[SNN/README](SNN/README.md)、[doc/architecture.md](../doc/architecture.md) §扩展 |
| **添加新量化方法** | `QANN/operators/` 新建量化模块 → `QANN/quantization/` 新建 Quantizer 类并注册 | [QANN/README](QANN/README.md) |
| **添加新 SNN 算子** | `SNN/operators/` 新建算子（继承 SNNOperator）→ `Evaluation/energy/ops_counter.py` 注册 hook | [SNN/README](SNN/README.md) |
| **添加新评估器** | `Evaluation/benchmarks/` 继承 BaseEvaluator 并用 `@EVALUATOR_REGISTRY.register()` 注册 | [Evaluation/README](Evaluation/README.md) |
| **添加新执行适配器** | `SNN/snnConverter/adapter.py` 继承 ModelExecutionAdapter 并用 `@ADAPTER_REGISTRY.register()` 注册 | [SNN/README](SNN/README.md) |
| **理解整体架构和依赖规则** | — | [doc/architecture.md](../doc/architecture.md) |
