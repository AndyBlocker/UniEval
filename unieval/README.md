# unieval

通用 SNN 转换与评估框架，源自 SpikeZIP-TF (ICML 2024)。

## 流程

```
ANN 模型 → 量化 → QANN 模型 → [校准] → 转换 → SNN Wrapper → 评估
```

## 子包

| 目录 | 职责 |
|------|------|
| `ann/` | 纯 ANN 模型定义 (ViT, Qwen3, UniAffine) 和共享算子 (RoPE) |
| `qann/` | 量化模块和量化策略 |
| `snn/` | SNN 算子和 ANN→SNN 转换引擎 |
| `evaluation/` | 准确率、能耗、spike rate 评估 |

## 包根文件

| 文件 | 职责 |
|------|------|
| `registry.py` | 通用注册表 |
| `config.py` | Dataclass 配置 |
| `protocols.py` | Duck-typing 结构谓词 (`is_*_like`) |

## 快速开始

```python
from unieval.ann.models.vit import vit_small_patch16
from unieval.qann import quantize, calibrate_ptq
from unieval.snn import convert
from unieval.evaluation import evaluate_energy

model = vit_small_patch16(num_classes=1000)
model = quantize(model, method="ptq", level=16)
calibrate_ptq(model.cuda(), dataloader)
wrapper = convert(model.cpu(), time_step=64, level=16)
result = evaluate_energy(wrapper.cuda(), dataloader, profile="vit_small")
```

完整示例见 `samples/`。

## 开发指南

根据你要做的事情，查阅对应的文档和目录：

| 我想要… | 需要改动的位置 | 参考文档 |
|---------|--------------|---------|
| **添加新模型族** | `ann/models/` 新建模型 → `protocols.py` 添加谓词 → `qann/quantization/` PTQ rules → `snn/snnConverter/` 转换 rules | [ann/README](ann/README.md)、[qann/README](qann/README.md)、[snn/README](snn/README.md) |
| **添加新量化方法** | `qann/operators/` 新建量化模块 → `qann/quantization/` 新建 Quantizer 类并注册 | [qann/README](qann/README.md) |
| **添加新 SNN 算子** | `snn/operators/` 新建算子（继承 SNNOperator）→ `evaluation/energy/ops_counter.py` 注册 hook | [snn/README](snn/README.md) |
| **添加新评估器** | `evaluation/benchmarks/` 继承 BaseEvaluator 并用 `@EVALUATOR_REGISTRY.register()` 注册 | [evaluation/README](evaluation/README.md) |
| **添加新执行适配器** | `snn/snnConverter/adapter.py` 继承 ModelExecutionAdapter 并用 `@ADAPTER_REGISTRY.register()` 注册 | [snn/README](snn/README.md) |
| **理解整体架构和依赖规则** | — | [doc/architecture.md](../doc/architecture.md) |
