# UniEval 架构文档

## 概述

UniEval 源自 SpikeZIP-TF (ICML 2024)，是一个通用的 SNN 转换与评估框架。

**核心流程**：

```
ANN 模型 → 量化 → QANN 模型 → [PTQ 校准] → 转换 → SNN Wrapper → 评估
```

## 分层结构

```
┌─────────────────────────────┐
│  evaluation/  (评估层)       │  观察 SNN/QANN，不参与变换
├─────────────────────────────┤
│  ann/ → qann/ → snn/        │  三种模型表示，单向依赖
├─────────────────────────────┤
│  registry / config / protocols│  跨层公共基础
└─────────────────────────────┘
```

### 依赖规则

**允许**：snn → qann、snn → ann/operators、qann → ann/operators、evaluation → 所有表示层

**禁止**：ann → qann/snn、qann → snn、基础层 → 子包

## API

每个子包暴露独立的入口函数，用户按需组合：

```python
from unieval.qann import quantize, calibrate_ptq
from unieval.snn import convert
from unieval.evaluation import evaluate_energy, evaluate_accuracy, evaluate_perplexity
```

不存在全局 "Runner" 或 "Pipeline" 对象——用户自己管理模型创建、checkpoint 加载和评估组合。

## 设计模式

1. **Registry** — `@registry.register("key")` 注册可替换组件（量化器、评估器、适配器等）
2. **规则匹配** — `QuantPlacementRule` / `ConversionRule`，遍历模型树，first-match-wins
3. **Duck-typing** — `protocols.py` 提供 `is_*_like()` 谓词，按结构匹配而非类型匹配
4. **Adapter** — `SNNWrapper` 委托执行给模型专属适配器（ViT / Decoder / Default）

## 关键契约

### SNNOperator

所有 SNN 算子继承 `SNNOperator` mixin：
- `reset()` — 清空时序状态
- `is_work` — 布尔字段，标记是否活跃（`Judger` 通过 `working` property 读取）
- `forward_multistep(x_seq)` — 语义等价于逐步 forward 的循环

### 阈值传递

`transfer_threshold(quan, neuron, ...)` 将量化器的 `s`/`pos_max`/`neg_min` 映射到神经元的发放阈值和脉冲边界。

### Composite 一对一映射

`QConv2d → SConv2d`，`QLinear → SLinear`，`QNorm → Spiking_*Norm + IFNeuron`

## 扩展

添加新模型族需要：ann 模型定义 → ModelProfile → duck-typing 谓词 → PTQ rule pack → Conversion rule pack → (可选) SNN 算子 / Adapter

添加新量化方法：qann/operators/ 新模块 → qann/quantization/ 新 Quantizer → (可选) SNN 转换规则

添加新评估器：继承 `BaseEvaluator`，用 `@EVALUATOR_REGISTRY.register()` 注册
