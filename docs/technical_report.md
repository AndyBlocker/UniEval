# UniEval Technical Report

## 1. 项目概述

UniEval 是从 SpikeZIP-TF（ICML 2024）中提取并重构的通用 SNN 转换与评估框架。原始代码中大量 hard-coded 逻辑（模型结构匹配、能耗常量、算子映射等）被重构为 Registry + Rule-based 架构，使框架可扩展到新模型、新量化方法、新评估指标。

**当前状态**: 阶段 1-4 已实现——forward_multistep 强语义契约、向量化算子 (Spiking_LayerNorm / SpikeMaxPooling / spiking_softmax)、新 API (step_encoded / forward_encoded / run_auto / encode_sequence)、Judger 预缓存 + 属性驱动、reset_model 平坦遍历、ModelExecutionAdapter 注册表 + ViTExecutionAdapter。需在 GPU 环境运行验证测试。

**依赖**: PyTorch >= 1.10, numpy, scipy, pyyaml。**不依赖 timm**——ViT 核心组件独立实现，属性名与 timm 0.3.2 一致以兼容已有 checkpoint。

---

## 2. 目录结构

```
UniEval/                          4220 行 Python + YAML
├── setup.py                      包安装配置
├── configs/
│   └── default.yaml              默认参数（level=16, T=64, e_mac=4.6pJ, ...）
├── unieval/
│   ├── __init__.py               顶层导出: Registry, Config 类
│   ├── registry.py          [60] 通用 Registry 类 + 6 个全局注册表
│   ├── config.py            [52] Dataclass 配置: UniEvalConfig 及子配置
│   ├── operators/                SNN 算子
│   │   ├── base.py          [22] SNNOperator mixin 接口
│   │   ├── neurons.py      [140] IFNeuron, ORIIFNeuron
│   │   ├── layers.py       [215] LLConv2d, LLLinear, Spiking_LayerNorm, SpikeMaxPooling
│   │   └── attention.py    [171] SAttention, spiking_softmax, multi(), multi1()
│   ├── quantization/             量化方法
│   │   ├── base.py          [52] BaseQuantizer ABC + QuantPlacementRule
│   │   ├── lsq.py          [446] MyQuan, QAttention, QuanConv2d/Linear, LSQQuantizer
│   │   └── ptq.py          [182] PTQQuan, PTQQuantizer
│   ├── conversion/               ANN → SNN 转换
│   │   ├── rules.py        [135] ConversionRule + 7 条默认规则
│   │   ├── converter.py     [50] SNNConverter 递归转换引擎
│   │   └── wrapper.py      [255] SNNWrapper, Judger, attn_convert, reset_model
│   ├── evaluation/               评估系统
│   │   ├── base.py          [37] EvalResult, BaseEvaluator ABC
│   │   ├── spike_utils.py   [31] spike_rate() 脉冲检测
│   │   ├── ops_counter.py  [330] OpsCounter (hook-based SYOPS 计数)
│   │   ├── energy.py       [164] EnergyEvaluator (可配置能耗模型)
│   │   └── accuracy.py      [83] AccuracyEvaluator (top-k)
│   ├── models/                   模型支持
│   │   ├── base.py          [44] ModelProfile + 4 个预注册 profile
│   │   └── vit.py          [355] 独立 ViT (DropPath/PatchEmbed/Attention/Mlp/Block)
│   └── engine/
│       └── runner.py       [195] UniEvalRunner 主入口
├── tests/
│   └── test_verify.py      [665] 38 个测试用例
└── experiments/
    ├── e2e_random.py        [184] 端到端 pipeline 实验
    └── verify_equivalence.py[268] QANN vs SNN 逐层等价性验证
```

---

## 3. 核心设计模式

### 3.1 Registry Pattern

所有扩展点由通用 `Registry` 类管理。通过装饰器注册，字符串键检索：

```python
@NEURON_REGISTRY.register("IF")
class IFNeuron(nn.Module, SNNOperator): ...

neuron_cls = NEURON_REGISTRY.get("IF")
```

6 个全局注册表:

| 注册表 | 内容 | 已注册键 |
|--------|------|---------|
| `QUANTIZER_REGISTRY` | 量化器 | `lsq`, `ptq` |
| `NEURON_REGISTRY` | 神经元 | `IF`, `ORIIF` |
| `EVALUATOR_REGISTRY` | 评估器 | `accuracy`, `energy` |
| `MODEL_PROFILE_REGISTRY` | 模型档案 | `vit_small/base/large/huge` |
| `OPS_HOOK_REGISTRY` | SYOPS 钩子 | (通过 OpsCounter 内部映射) |
| `CONVERSION_RULE_REGISTRY` | 转换规则 | (通过 DEFAULT_CONVERSION_RULES 列表) |

### 3.2 Rule-Based Quantization Placement

`QuantPlacementRule(name, match_fn, apply_fn)` 替代原始 hard-coded `myquan_replace()`。

**match_fn 使用 duck-typing** 而非 `isinstance(child, Block)`：

```python
def _match_transformer_block(name, child, parent):
    return hasattr(child, "attn") and hasattr(child, "mlp")
```

这使得量化规则可以适用于任何具有 `attn` + `mlp` 子模块的模型架构。

LSQ 默认规则（按匹配顺序）:

| 规则 | 匹配条件 | 操作 |
|------|---------|------|
| `transformer_block` | `hasattr(child, "attn") and hasattr(child, "mlp")` | 替换 attn→QAttention, 包裹 norm1/norm2/mlp |
| `conv2d` | `isinstance(child, nn.Conv2d)` | `Sequential(conv, MyQuan)` |
| `layernorm` | `isinstance(child, nn.LayerNorm)` | `Sequential(ln, MyQuan)` |

### 3.3 Rule-Based SNN Conversion

`ConversionRule(name, match_fn, convert_fn, priority)` 替代原始 `_replace_weight()`。规则按 priority 降序排列，first match wins。

7 条默认规则:

| Priority | 规则 | 转换 |
|----------|------|------|
| 100 | `qattention_to_sattention` | QAttention → SAttention |
| 90 | `myquan_to_ifneuron` | MyQuan → IFNeuron（迁移学到的 threshold） |
| 85 | `ptqquan_to_ifneuron` | PTQQuan → IFNeuron |
| 50 | `conv2d_to_llconv2d` | Conv2d/QuanConv2d → LLConv2d |
| 50 | `linear_to_lllinear` | Linear/QuanLinear → LLLinear |
| 40 | `layernorm_to_spiking` | LayerNorm → Spiking_LayerNorm（保留权重） |
| 30 | `relu_to_identity` | ReLU → Identity |

### 3.4 Temporal State Management

所有 SNN 算子继承 `SNNOperator` mixin，实现 `reset()` 方法和 `is_work` 标记：

```
SNNOperator (mixin)
├── IFNeuron       — state: q, acc_q, cur_output
├── ORIIFNeuron    — state: q, acc_q, cur_output
├── LLConv2d       — state: first, zero_output, realize_time
├── LLLinear       — state: first, zero_output, realize_time
├── Spiking_LayerNorm — state: X, Y_pre
├── spiking_softmax   — state: X, Y_pre
├── SpikeMaxPooling   — state: accumulation
└── SAttention        — 内含 6 个 IFNeuron + spiking_softmax + 2 个 LLLinear
```

**差分输出原则**: Spiking_LayerNorm、spiking_softmax、SpikeMaxPooling 输出 Y_t − Y_{t−1}，使得累积和 Σ(Y_t − Y_{t−1}) = Y_T − Y_0 = Y_T。

### 3.5 Configurable Energy Model

`EnergyConfig(e_mac, e_ac, nspks_max)` + `ModelProfile(depth, num_heads, embed_dim, ...)` 替代 hard-coded `ssa_info` 字典和 `eval()` 字符串路径访问。

**动态 SAttention 发现**替代 `eval(f'model.module.model.blocks[{d}].attn.q_IF.__syops__[3]')`：

```python
def _find_sattention_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, SAttention):
            modules.append((name, module))
```

---

## 4. 数据流

### 4.1 完整 Pipeline

```
              ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
 ViT Model ──→│ Quantize │──→│  Convert  │──→│ SNNWrapper │──→│ Evaluate │
 (ReLU)       │ (LSQ/PTQ)│   │(SNNConvert│   │ (temporal   │   │(Acc/Enrg)│
              │          │   │  er)      │   │  inference) │   │          │
              └──────────┘    └───────────┘    └────────────┘    └──────────┘
                  │                │                 │                │
             QuantPlacement   ConversionRule     Judger 早停      OpsCounter
             Rule (duck-     (priority-based)   + reset_model    hook-based
             typed match)                                        SYOPS 计数
```

### 4.2 模块转换对照

```
QANN (量化后)                          SNN (转换后)
─────────────────────────────         ─────────────────────────────
Block                                 Block (同一对象, 子模块替换)
├── norm1: Sequential                 ├── norm1: Sequential
│   ├── LayerNorm                     │   ├── Spiking_LayerNorm
│   └── MyQuan(sym=T)                 │   └── IFNeuron(sym=T)
├── attn: QAttention                  ├── attn: SAttention
│   ├── qkv: Linear                  │   ├── qkv: LLLinear
│   ├── quan_q: MyQuan               │   ├── q_IF: IFNeuron
│   ├── quan_k: MyQuan               │   ├── k_IF: IFNeuron
│   ├── quan_v: MyQuan               │   ├── v_IF: IFNeuron
│   ├── attn_quan: MyQuan            │   ├── attn_IF: IFNeuron
│   ├── after_attn_quan: MyQuan      │   ├── after_attn_IF: IFNeuron
│   ├── proj: Linear                 │   ├── proj: LLLinear
│   └── quan_proj: MyQuan            │   └── proj_IF: IFNeuron
├── norm2: Sequential                 ├── norm2: Sequential
│   ├── LayerNorm                     │   ├── Spiking_LayerNorm
│   └── MyQuan(sym=T)                 │   └── IFNeuron(sym=T)
└── mlp: Mlp                          └── mlp: Mlp (同一对象)
    ├── fc1: Linear                       ├── fc1: LLLinear
    ├── act: Sequential                   ├── act: Sequential
    │   ├── MyQuan(sym=F)                 │   ├── IFNeuron(sym=F)
    │   └── ReLU                          │   └── Identity
    ├── fc2: Sequential                   ├── fc2: Sequential
    │   ├── Linear                        │   ├── LLLinear
    │   └── MyQuan(sym=T)                 │   └── IFNeuron(sym=T)
    └── drop: Dropout                     └── drop: Dropout
```

### 4.3 时序推理流程 (SNNWrapper.forward)

```
t=0:  input=x, pos_embed=原值, cls_token=原值
      → model(input) → output_0
      → accu = output_0

t=1:  input=zeros, pos_embed=0, cls_token=0
      → model(input) → output_1  (内部神经元继续发放残余 spike)
      → accu = accu + output_1

...

t=T:  Judger 检测 all is_work==False → 早停
      或 t >= T_max → 强制停止

return accu, actual_T
```

---

## 5. 公开 API

### 5.1 高层接口 (UniEvalRunner)

```python
from unieval.config import UniEvalConfig, QuantConfig, ConversionConfig
from unieval.engine.runner import UniEvalRunner

config = UniEvalConfig(
    model_name="vit_small",
    quant=QuantConfig(level=16),
    conversion=ConversionConfig(time_step=64, encoding_type="analog"),
)
runner = UniEvalRunner(config)

# 分步执行
model = runner.create_model(act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6))
model = runner.quantize(model, quantizer_name="lsq")
wrapper = runner.convert(model)
acc = runner.evaluate_accuracy(wrapper, dataloader)
energy = runner.evaluate_energy(wrapper, dataloader)

# 或一键执行
acc, energy, wrapper = runner.run_full_pipeline(dataloader, checkpoint_path="ckpt.pth")
```

### 5.2 中层接口 (各组件独立使用)

```python
# 量化
from unieval.quantization.lsq import LSQQuantizer
quantizer = LSQQuantizer(level=16, weight_bit=32, is_softmax=True)
model = quantizer.quantize_model(model)

# 转换
from unieval.conversion.converter import SNNConverter
converter = SNNConverter()  # 使用默认规则
converter.convert(model, level=16, neuron_type="ST-BIF", is_softmax=True)

# 包装
from unieval.conversion.wrapper import SNNWrapper
wrapper = SNNWrapper(model, time_step=64, encoding_type="analog", ...)

# 评估
from unieval.evaluation.energy import EnergyEvaluator
from unieval.models.base import ModelProfile
evaluator = EnergyEvaluator(model_profile=ModelProfile(depth=12, ...))
result = evaluator.evaluate(wrapper, dataloader)
```

### 5.3 扩展接口 (注册新组件)

```python
from unieval.registry import NEURON_REGISTRY, QUANTIZER_REGISTRY
from unieval.operators.base import SNNOperator
from unieval.quantization.base import BaseQuantizer, QuantPlacementRule
from unieval.conversion.rules import ConversionRule

# 新神经元
@NEURON_REGISTRY.register("LIF")
class LIFNeuron(nn.Module, SNNOperator):
    def reset(self): ...
    def forward(self, input): ...

# 新量化器
@QUANTIZER_REGISTRY.register("dorefa")
class DoReFaQuantizer(BaseQuantizer):
    def quantize_model(self, model): ...

# 新转换规则
my_rule = ConversionRule(
    name="lif_convert",
    match_fn=lambda n, c, p: isinstance(c, LIFNeuron),
    convert_fn=lambda n, c, p, **kw: ...,
    priority=95,
)
converter = SNNConverter(rules=DEFAULT_CONVERSION_RULES + [my_rule])

# 新模型 profile
MODEL_PROFILE_REGISTRY.register_obj("swin_tiny", ModelProfile(
    depth=4, num_heads=3, embed_dim=96, patch_size=4, time_steps=32,
))
```

---

## 6. 源码映射 (SpikeZIP-TF → UniEval)

| 原始文件 | 原始函数/类 | UniEval 位置 |
|---------|------------|-------------|
| `spike_quan_layer.py` | `IFNeuron` | `operators/neurons.py` |
| | `ORIIFNeuron` | `operators/neurons.py` |
| | `Spiking_LayerNorm` | `operators/layers.py` |
| | `spiking_softmax` | `operators/attention.py` |
| | `SAttention` | `operators/attention.py` |
| | `LLConv2d`, `LLLinear` | `operators/layers.py` |
| | `SpikeMaxPooling` | `operators/layers.py` |
| | `MyQuan` | `quantization/lsq.py` |
| | `QAttention` | `quantization/lsq.py` |
| | `QuanConv2d`, `QuanLinear` | `quantization/lsq.py` |
| | `grad_scale`, `floor_pass`, `round_pass` | `quantization/lsq.py` |
| | `threshold_optimization` | `quantization/lsq.py` |
| `spike_quan_wrapper.py` | `myquan_replace()` | `quantization/lsq.py` → `LSQQuantizer` + rules |
| | `_replace_weight()` | `conversion/rules.py` + `converter.py` |
| | `SNNWrapper` | `conversion/wrapper.py` |
| | `Judger` | `conversion/wrapper.py` |
| | `attn_convert()` | `conversion/wrapper.py` |
| | `reset_model()` | `conversion/wrapper.py` |
| `energy_consumption_calculation/ops.py` | `spike_rate()` | `evaluation/spike_utils.py` |
| | `*_syops_counter_hook` | `evaluation/ops_counter.py` |
| | `MODULES_MAPPING` | `evaluation/ops_counter.py` |
| `energy_consumption_calculation/engine.py` | `get_syops_pytorch()` | `evaluation/ops_counter.py` → `OpsCounter` |
| `energy_consumption_calculation/flops_counter.py` | `get_energy_cost()` | `evaluation/energy.py` → `EnergyEvaluator` |
| | `ssa_info` (hard-coded dict) | `models/base.py` → `ModelProfile` + Registry |
| `models_vit.py` | `VisionTransformer` | `models/vit.py` (独立实现, 无 timm) |

---

## 7. 验证结果

### 7.1 单元测试 (38/38 PASS)

```
--- Core Imports ---           7 passed
--- Registry System ---        6 passed
--- Config System ---          2 passed
--- Operators ---              8 passed
--- Quantization ---           5 passed
--- Conversion ---             2 passed
--- SNNWrapper ---             1 passed
--- Evaluation ---             5 passed
--- Model Creation ---         1 passed
--- Full Pipeline ---          1 passed
```

### 7.2 QANN vs SNN 等价性验证 (level=8, T_max=32)

```
Final Output    : L1=8.20e-09  Rel=0.0000%  Cos=1.000000
Block 0         : L1=1.29e-08  Rel=0.0000%  Cos=1.000000
Block 1         : L1=1.75e-08  Rel=0.0000%  Cos=1.000000
Block 0 norm1   : L1=0.00e+00  Cos=1.000000
Block 0 attn    : L1=0.00e+00  Cos=1.000000
Block 0 norm2   : L1=0.00e+00  Cos=1.000000
Block 0 mlp     : L1=0.00e+00  Cos=1.000000
Block 1 norm1   : L1=2.19e-10  Cos=1.000000
Block 1 attn    : L1=0.00e+00  Cos=1.000000
Block 1 norm2   : L1=0.00e+00  Cos=1.000000
Block 1 mlp     : L1=0.00e+00  Cos=1.000000

Verdict: PASS — 所有误差在浮点精度范围内 (~1e-8)
SNN 在 T=7 自动收敛（早停生效, T_max=32）
```

---

## 8. Git 历史

```
c6cfece feat: add QANN vs SNN equivalence verification script
da9519b feat: add end-to-end random experiment script
7e9dec5 refactor: remove timm dependency with standalone ViT implementation
68beb03 fix: add project root to sys.path in test script
0b9c055 feat: initial UniEval framework implementation
```

---

## 9. 后续工作

- 在真实 checkpoint + ImageNet 数据上跑完整评估 pipeline
- 补充更多模型架构（Swin, DeiT 等）的 QuantPlacementRule / ConversionRule
- 添加 LIF (Leaky IF) 等新神经元类型
- 支持更多量化方法（DoReFa, PACT 等）
- PTQ 路径的端到端验证
- 修复 MyQuan 中 `.cuda()` 硬编码问题以支持 CPU-only 环境

---

## 10. Three Inference Modes 设计 (RFC)

> **状态**: Request for Comments — 设计方案征求意见中。部分问题已在外部架构 review 后达成共识（见 10.12），其余问题的结论已更新到各对应章节。

### 10.1 设计目标

当前 UniEval 只支持一种推理模式：SNNWrapper 内部 `while True` 循环 + Judger 早停（single-step auto）。这限制了：

1. **灵活性** — 用户无法在时间步之间插入自定义逻辑（如中间状态观测、自定义早停策略）
2. **性能** — 部分算子的 multi-step 计算可以向量化（cumsum → batch op → diff），避免 Python for-loop 开销
3. **可组合性** — 无法将 SNN 推理嵌入到更大的控制流中

目标：支持三种推理模式，满足不同使用场景：

| 模式 | 控制权 | 张量形状 | 早停 | 典型场景 |
|------|--------|---------|------|---------|
| **Single-step auto** | wrapper 内循环 | `[B, ...]` | ✓ Judger | 默认推理，开箱即用 |
| **Single-step manual** | 用户外循环 | `[B, ...]` | 用户自控 | 自定义调度、中间观测、调试 |
| **Multi-step batch** | 一次 forward | `[T, B, ...]` | ✗ | 性能优化（可向量化算子加速） |

三种模式应在给定相同输入和足够时间步数时产生**数值等价**的结果。

---

### 10.2 设计理念

#### 理念 1: 单一基类 + 属性声明

**放弃方案**: 为算子引入三个标记子类 `SNNNeuron` / `SNNLayer` / `SNNStateful` 做分类。

**放弃原因**: 对所有现有算子的行为分析表明，三分法本质上只区分了两种行为：

| 行为组 | 有 `is_work` | 参与早停 | 包含的算子 |
|--------|:---:|:---:|------|
| 组 A | ✓ | ✓ | IFNeuron, ORIIFNeuron, LLConv2d, LLLinear |
| 组 B | ✗ | ✗ | Spiking_LayerNorm, SpikeMaxPooling, spiking_softmax |

`SNNNeuron`（IFNeuron）和 `SNNLayer`（LLConv2d）在接口层面完全相同——都有 `is_work`，都参与早停。引入两个行为一致的标记类只增加了"该归哪类"的心智负担，没有提供可编程的差异。`SNNStateful` 这个名字也不准确——组 A 的算子同样有状态。

**采用方案**: 所有算子继承同一个 `SNNOperator`，行为差异通过**类属性**和**方法重写**表达，而非类型层级。

优势：
- 新算子只需 `class MyOp(nn.Module, SNNOperator)`，设 1-2 个属性即可集成
- 不需要纠结"我的算子属于 Neuron 还是 Layer 还是 Stateful"
- Judger / reset_model 统一用 `isinstance(child, SNNOperator)` + 属性检查，永远不会遗漏新算子

#### 理念 2: 状态变量形状不变 + 前态/终态同步

**关键决策**: `forward_multistep(x_seq)` **不改变**算子内部状态变量的形状。

以 IFNeuron 为例，膜电位 `q` 在 single-step 下为 `[B, C, H, W]`。在 multi-step 下，`q` 仍然是 `[B, C, H, W]`——不会变成 `[T, B, C, H, W]`。

原因：

```
如果允许 q 在两种模式间变形状：
  1. 交叉调用时形状错误 — 先 forward_multistep (q=[T,B,...]) 再 forward (期望 q=[B,...])
  2. reset() 要知道当前模式 — 不同模式的"初始状态"形状不同
  3. 算子内部处处加维度判断 — if q.dim() == 5: ... elif q.dim() == 4: ...
```

具体做法：
- **有序列依赖的算子**（IFNeuron, LLConv2d 等）：`forward_multistep` 内部仍是 for-loop over T，每步调用 `self.forward(x_seq[t])`，状态始终 `[B, ...]`
- **可向量化的算子**（Spiking_LayerNorm 等）：`forward_multistep` 从输入序列直接计算（`cumsum → transform → diff`），但**必须在开始时读取当前状态作为前缀，在结束时同步最终状态**到 `self.X`, `self.Y_pre` 等

> **与早期设计的差异**: 早期版本将可向量化算子的 `forward_multistep` 描述为"完全绕过状态变量，不读也不写"。经 review 后明确：这违背了强语义契约（见理念 4），向量化只是实现优化，不是语义上的状态脱离。正确做法是将当前状态（如 `self.Y_pre`）作为 shifted 序列的第一项，并在结束时将最终状态写回。

#### 理念 3: Multi-step 作为算子方法

**放弃方案**: 为每个算子创建独立的 `MultiStep*` 类（如 `MultiStepIFNeuron`, `MultiStepLLConv2d`），放在独立文件 `operators/multistep.py` 中。转换规则增加 `multistep_convert_fn` 字段，转换器增加 `mode` 参数。

**放弃原因**: 这种设计将同一算子的逻辑分散到两个类中，且引入了大量基础设施（新文件、新字段、新参数），收益仅仅是"multi-step 代码和 single-step 代码物理分离"。

**采用方案**: 在 `SNNOperator` 基类上定义 `forward_multistep()` 方法，提供默认的 for-loop fallback，各算子按需重写。

优势：
- 不需要 `operators/multistep.py`
- 不需要 `ConversionRule.multistep_convert_fn`
- 不需要 `SNNConverter.convert(mode=...)`
- 同一个算子对象同时支持 `op.forward(x)` 和 `op.forward_multistep(x_seq)`
- 新算子自动获得 multi-step 支持（默认 for-loop fallback），性能敏感时再重写

#### 理念 4: forward_multistep 强语义契约

**定义**: `forward_multistep(x_seq)` 的语义精确等价于：

```
从当前状态出发，顺序执行 forward(x_seq[0]), forward(x_seq[1]), ..., forward(x_seq[T-1])，
返回逐步输出堆叠为 [T, B, ...] 张量，留下完全相同的最终状态。
```

即：对于任何算子 `op`，以下两段代码必须产生 **bit-identical** 的输出和最终状态：

```python
# 路径 A: single-step 循环
state_before = copy(op.state)
outputs = [op.forward(x_seq[t]) for t in range(T)]
state_after_A = copy(op.state)
result_A = torch.stack(outputs)

# 路径 B: multi-step 调用
op.load_state(state_before)
result_B = op.forward_multistep(x_seq)
state_after_B = copy(op.state)

# 契约
assert torch.equal(result_A, result_B)
assert state_after_A == state_after_B
```

**关键推论**:

1. **向量化不是"绕过状态"** — 向量化实现是一种数学等价的高效计算路径，但语义上必须精确继承当前状态、留下正确的最终状态
2. **初始状态不一定为零** — 当前向量化公式中用 `zeros` 作为 shifted 序列的第一项，隐含假设初始状态为零。正确做法是用 `self.Y_pre`（当前状态）作为第一项
3. **终态必须同步** — `forward_multistep` 结束时必须将状态更新为 `Y[-1]`（最后一步的输出），使后续的 `forward()` 调用从正确的状态继续
4. **默认 for-loop fallback 天然满足契约** — 因为它就是逐步调用 `forward()`，状态由每步 `forward()` 自然更新

---

### 10.3 SNNOperator 统一接口

```python
import torch
import torch.nn as nn


class SNNOperator:
    """所有 SNN 算子的统一基类 (mixin)。

    与 nn.Module 配合使用：class MyOp(nn.Module, SNNOperator): ...

    子类必须实现:
        - reset(): 重置时序状态
        - forward(x): single-step 前向传播 (nn.Module 的 forward)

    子类可选实现:
        - forward_multistep(x_seq): multi-step 前向传播，默认 for-loop over forward()

    子类可选覆盖的类属性:
        - participates_in_early_stop: 是否参与 Judger 早停判断 (默认 True)
    """

    # ── 时序状态管理 ──

    def reset(self):
        """重置时序状态到初始值。每个算子必须实现。"""
        raise NotImplementedError

    @property
    def working(self) -> bool:
        """算子是否仍在产生非零输出（用于早停判断）。

        默认实现: 返回 self.is_work（如果存在），否则 False。

        防御性检查: 如果 participates_in_early_stop=True 但算子没有
        is_work 属性，说明算子声明参与早停却未实现工作状态追踪，
        应在初始化时报错而非静默返回 False（可能导致 Judger 误判收敛）。
        """
        if self.participates_in_early_stop and not hasattr(self, "is_work"):
            raise AttributeError(
                f"{type(self).__name__} has participates_in_early_stop=True "
                f"but does not define 'is_work'. Either set "
                f"participates_in_early_stop=False or implement is_work."
            )
        return getattr(self, "is_work", False)

    # ── 行为声明 ──

    participates_in_early_stop: bool = True
    """Judger 是否检查此算子的 working 状态来判断收敛。

    - True  (默认): IFNeuron, ORIIFNeuron, LLConv2d, LLLinear
    - False: Spiking_LayerNorm, SpikeMaxPooling, spiking_softmax, SAttention
    """

    # ── Multi-step 接口 ──

    def forward_multistep(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Multi-step forward: [T, B, ...] → [T, B, ...]

        强语义契约:
            forward_multistep(x_seq) ≡ 从当前状态出发，顺序执行
            forward(x_seq[0]), ..., forward(x_seq[T-1])，返回逐步输出
            堆叠为 [T, B, ...] 张量，留下完全相同的最终状态。

        即: 调用前后的状态变迁和输出必须与逐步调用 forward() 完全一致。
        向量化实现是允许的优化手段，但不得违反此契约。

        默认实现: for-loop over T，逐步调用 self.forward()，天然满足契约。

        重写指南:
            可向量化的算子应重写此方法以获得性能提升。重写时:
            1. 必须读取当前状态作为计算的初始条件（如 self.Y_pre 作为 shifted 的第一项）
            2. 必须在结束时将状态同步到最终时间步的值
            3. 状态变量形状保持 [B, ...] 不变
        """
        return torch.stack(
            [self(x_seq[t]) for t in range(x_seq.shape[0])],
            dim=0,
        )
```

各算子对 SNNOperator 接口的实现：

| 算子 | `participates_in_early_stop` | `forward_multistep` | 说明 |
|------|:---:|:---:|------|
| IFNeuron | `True` (默认) | 默认 for-loop | 膜电位有序列依赖，无法向量化 |
| ORIIFNeuron | `True` (默认) | 默认 for-loop | 同上 |
| LLConv2d | `True` (默认) | 默认 for-loop (见 Q2) | bias leakage 有序列依赖 |
| LLLinear | `True` (默认) | 默认 for-loop (见 Q2) | 同上 |
| Spiking_LayerNorm | **`False`** | **重写 (cumsum)** | 完全可向量化 |
| SpikeMaxPooling | **`False`** | **重写 (cumsum)** | 完全可向量化 |
| spiking_softmax | **`False`** | **重写 (cumsum)** | 完全可向量化 |
| SAttention | **`False`** | 默认 for-loop (见 Q3) | 复合算子 |

---

### 10.4 现有算子行为详解

为便于讨论 multi-step 实现，此处详细记录每个算子的 single-step 行为。

#### 10.4.1 IFNeuron / ORIIFNeuron — 脉冲发放

```
状态: q (膜电位), acc_q (累积发放计数), cur_output (当前输出)
形状: 初始为标量 0.0，首次接收张量输入后惰性扩展为 [B, ...] (与输入同形)

forward(x):
    x_scaled = x / q_threshold
    q += x_scaled                              # 膜电位积分
    if q >= 1:  cur_output = +1, q -= 1        # 正脉冲
    if q < 0:   cur_output = -1, q += 1        # 负脉冲 (IFNeuron only)
    return cur_output * q_threshold

序列依赖: q_t 依赖 q_{t-1} 的发放结果 → 无法跨时间步并行
```

#### 10.4.2 LLConv2d / LLLinear — 带 bias leakage 的线性变换

```
状态: realize_time (剩余 bias 释放步数), zero_output (缓存零张量)

forward(x):
    if x == 0:
        if realize_time > 0: return bias / steps, realize_time -= 1
        else: return zero_output
    output = conv(x)  或  linear(x)
    output -= bias                             # 减去完整 bias
    if realize_time > 0: output += bias/steps  # 逐步释放 bias
    return output

序列依赖: realize_time 是一个递减计数器，有顺序依赖
线性部分: conv(x) / linear(x) 无序列依赖 → 可批量化
```

#### 10.4.3 Spiking_LayerNorm — 差分归一化

```
状态: X (累积输入), Y_pre (上一步输出)

forward(x):
    X += x
    Y = layernorm(X)
    output = Y - Y_pre
    Y_pre = Y
    return output

数学等价:
    output_t = LN(Σ_{i=0}^{t} x_i) - LN(Σ_{i=0}^{t-1} x_i)
    即: output_t = LN(cumsum[t]) - LN(cumsum[t-1])

向量化: X_cum = cumsum(x_seq, dim=0); Y = LN(X_cum); output = Y - shift(Y)
```

#### 10.4.4 SpikeMaxPooling — 差分池化

```
状态: accumulation (累积输入)

forward(x):
    old_accu = accumulation
    accumulation += x
    output = maxpool(accumulation) - maxpool(old_accu)
    return output

数学等价:
    output_t = maxpool(cumsum[t]) - maxpool(cumsum[t-1])

向量化: accu = cumsum(x_seq); pooled = maxpool(accu); output = pooled - shift(pooled)
```

#### 10.4.5 spiking_softmax — 差分 softmax

```
状态: X (累积输入), Y_pre (上一步输出)

forward(x):
    X += x
    Y = softmax(X, dim=-1)
    output = Y - Y_pre
    Y_pre = Y
    return output

向量化: X_cum = cumsum(x_seq); Y = softmax(X_cum); output = Y - shift(Y)
```

#### 10.4.6 SAttention — 复合算子

SAttention 内部包含 6 个 IFNeuron、2 个 LLLinear、1 个 spiking_softmax，并使用特殊的 `multi()` / `multi1()` 高效时序矩阵乘法（利用累积值避免重复计算）：

```
multi(q_t, k_t, q_sum, k_sum)  = q_sum @ k_t^T + q_t @ k_sum^T - q_t @ k_t^T
multi1(a_t, v_t, a_sum, v_sum) = a_sum @ v_t   + a_t @ v_sum   - a_t @ v_t

其中 q_sum = Σ q_i (通过 IFNeuron.acc_q 获取)
```

---

### 10.5 各算子 forward_multistep 实现

#### 10.5.1 默认 for-loop fallback (IFNeuron, ORIIFNeuron)

不需要重写。基类默认实现即可：

```python
def forward_multistep(self, x_seq):  # [T, B, ...]
    return torch.stack([self(x_seq[t]) for t in range(x_seq.shape[0])])
    # 每步 self(x_seq[t]) 中，q/acc_q/cur_output 始终为 [B, ...] 形状
```

膜电位 `q_t` 依赖 `q_{t-1}` 的发放结果，无法跨时间步并行化。

#### 10.5.2 Spiking_LayerNorm (完全向量化)

等价性推导:
```
single-step (从任意当前状态 self.X, self.Y_pre 继续):
  t=0: X += x_0,         Y_0=LN(X),             out=Y_0 - Y_pre
  t=1: X += x_1,         Y_1=LN(X),             out=Y_1 - Y_0
  t=2: X += x_2,         Y_2=LN(X),             out=Y_2 - Y_1

forward_multistep (从当前状态继续):
  X_cum[t] = self.X + Σ_{i=0}^{t} x_i          # 累积在当前状态之上
  Y[t] = LN(X_cum[t])
  output[0] = Y[0] - self.Y_pre                 # 第一项用当前状态
  output[t] = Y[t] - Y[t-1]                     # 后续项用前一步
  结束时: self.X = X_cum[-1], self.Y_pre = Y[-1]  # 同步最终状态
```

```python
def forward_multistep(self, x_seq):  # [T, B, N, D]
    X_cum = x_seq.cumsum(dim=0) + self.X        # 在当前累积状态之上
    Y = self.layernorm(X_cum)
    Y_shifted = torch.cat([self.Y_pre.unsqueeze(0), Y[:-1]], dim=0)  # 前态作为第一项
    output = Y - Y_shifted
    # 同步最终状态（满足强语义契约）
    self.X = X_cum[-1]
    self.Y_pre = Y[-1]
    return output
```

#### 10.5.3 SpikeMaxPooling (完全向量化)

```python
def forward_multistep(self, x_seq):  # [T, B, C, H, W]
    T, B = x_seq.shape[:2]
    accu = x_seq.cumsum(dim=0) + self.accumulation  # 在当前累积状态之上
    pooled = self.maxpool(accu.reshape(T * B, *accu.shape[2:]))
    pooled = pooled.reshape(T, B, *pooled.shape[1:])
    # 前态: maxpool(当前累积) 作为 shifted 的第一项
    pooled_prev = self.maxpool(self.accumulation).unsqueeze(0)  # [1, B, ...]
    pooled_shifted = torch.cat([pooled_prev, pooled[:-1]], dim=0)
    output = pooled - pooled_shifted
    # 同步最终状态
    self.accumulation = accu[-1]
    return output
```

#### 10.5.4 spiking_softmax (完全向量化)

```python
def forward_multistep(self, x_seq):  # [T, B, H, N, N]
    X_cum = x_seq.cumsum(dim=0) + self.X             # 在当前累积状态之上
    Y = F.softmax(X_cum, dim=-1)
    Y_shifted = torch.cat([self.Y_pre.unsqueeze(0), Y[:-1]], dim=0)  # 前态作为第一项
    output = Y - Y_shifted
    # 同步最终状态
    self.X = X_cum[-1]
    self.Y_pre = Y[-1]
    return output
```

#### 10.5.5 LLConv2d / LLLinear — 默认 Fallback（结论：见 Q2）

线性变换 `conv(x)` / `linear(x)` 无序列依赖，可对 T 步批量执行。
但 bias leakage（`realize_time` 递减）有序列依赖。

**方案 A: 不重写 (使用默认 for-loop fallback)**

- 优势: 零代码修改
- 劣势: 无性能提升

**方案 B: 分离线性运算和 bias leakage**

```python
def forward_multistep(self, x_seq):  # [T, B, C, H, W]
    T, B = x_seq.shape[:2]

    # 批量线性运算 (无序列依赖)
    flat = x_seq.reshape(T * B, *x_seq.shape[2:])
    out_flat = self.conv(flat)                     # 一次 conv 处理 T*B 个样本
    out = out_flat.reshape(T, B, *out_flat.shape[1:])

    # bias leakage (有序列依赖，仍需顺序)
    if self.neuron_type != "IF" and self.conv.bias is not None:
        bias = self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        for t in range(T):
            out[t] = out[t] - bias
            if self.realize_time > 0:
                out[t] = out[t] + bias / self.steps
                self.realize_time -= 1

    return out
```

- 优势: Conv2d/Linear 通常是计算瓶颈，T*B batch 可提高 GPU 利用率
- 劣势: 实现更复杂；零输入的特殊路径需要处理；性能收益需 benchmark 验证
- **注意**: 方案 B 中零输入判断比 single-step 更复杂——需要逐步检测哪些时间步是全零输入

#### 10.5.6 SAttention — 默认 Fallback（结论：见 Q3）

SAttention 是复合算子，内部子算子的可向量化程度不同：

| SAttention 内部组件 | 可向量化? | 说明 |
|---|:---:|---|
| qkv (LLLinear) | fallback (见 Q2) | 线性部分可以，bias leakage 不行 |
| q_IF, k_IF, v_IF (IFNeuron) | ✗ | 膜电位序列依赖 |
| multi() — Q×K^T | ✗ | 依赖 IFNeuron 的 acc_q 累积值 |
| Ssoftmax (spiking_softmax) | ✓ | cumsum + softmax + diff |
| attn_IF (IFNeuron) | ✗ | 膜电位序列依赖 |
| multi1() — Attn×V | ✗ | 依赖 acc_q 累积值 |
| after_attn_IF, proj_IF (IFNeuron) | ✗ | 膜电位序列依赖 |
| proj (LLLinear) | fallback (见 Q2) | 同 LLLinear |

大部分组件无法向量化（IFNeuron 占主导）。

**方案 A: 使用默认 for-loop fallback**

```python
# 不重写，基类默认实现:
# for t in range(T): self(x_seq[t])
# 内部所有子算子走 single-step forward
```

- 优势: 零修改，逻辑完全复用
- 劣势: 不利用 spiking_softmax 的向量化能力

**方案 B: 重写 forward_multistep，混合向量化 + for-loop**

内部仍有 for-loop（IFNeuron 无法消除），但 spiking_softmax 部分提前向量化。需要：
- 重写 multi() / multi1() 的 multi-step 版本
- 处理 acc_q 在 for-loop 中逐步更新的逻辑
- 代码复杂度高，容易引入 bug

**当前倾向**: 方案 A。SAttention 中的性能瓶颈是 IFNeuron 和矩阵乘法，spiking_softmax 只占一小部分计算量。

---

### 10.6 推理模式设计（正交分解）

> **与早期设计的差异**: 早期版本将推理模式描述为三种互斥的 mode（single-step auto / single-step manual / multi-step batch）。经 review 后明确：这三者不是一个维度上的三个选项，而是**三件正交的事**。将它们拆解后，API 更清晰，组合也更灵活。

#### 10.6.1 三个正交关注点

| 关注点 | 选项 | 说明 |
|--------|------|------|
| **输入编码** | analog / rate / 用户自定义 | 原始输入如何变成时序脉冲序列 |
| **执行方式** | stepwise / batched | 逐步 for-loop 还是一次性向量化 |
| **停止策略** | Judger early stop / fixed T / 用户自定义 | 何时认为推理已收敛 |

这三件事可以自由组合。例如：
- `analog + stepwise + Judger` = 当前的 single-step auto
- `analog + stepwise + 用户自定义` = 当前的 single-step manual
- `analog + batched + fixed T` = multi-step batch
- `rate + stepwise + Judger` = rate encoding + 早停

#### 10.6.2 新 API 分层

**核心低层原语（吃已编码的输入）：**

```python
# 单步：吃已编码的单步输入，返回该步差分输出
def step_encoded(self, x_t: torch.Tensor) -> torch.Tensor:
    """处理一个已编码的时间步输入。

    不涉及编码逻辑，不涉及早停判断。
    使用前必须 reset()。用户完全控制循环、编码、累积和停止。
    """
    return self.model(x_t)

# 多步：吃 [T,B,...] 的显式序列，返回 [T,B,...] 差分输出序列
def forward_encoded(self, x_seq: torch.Tensor) -> torch.Tensor:
    """处理显式编码序列。

    内部走 adapter 的 multistep 路径（如果可用）或 stepwise fallback。
    返回逐步差分输出，用户自行 sum(dim=0) 得到最终输出。
    """
    ...
```

**便利高层接口（自动编码 + 自动停止）：**

```python
# 编码器：把原始输入变成显式时序序列
def encode_sequence(self, x: torch.Tensor, T: int, seed: int = None
                    ) -> torch.Tensor:
    """将原始输入编码为 [T, B, ...] 的显式时序序列。

    - analog: x_seq[0] = x, x_seq[1:] = 0
    - rate:   x_seq[t] = bernoulli(x, seed=seed+t)  (确定性，可重放)
    """
    ...

# 自动模式：内部编码 + Judger 早停
def run_auto(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """当前 forward() 的等价替代，自动编码 + Judger 早停。

    内部调用 step_encoded() + Judger，返回 (累积输出, actual_T)。
    """
    ...
```

**分层关系：**

```
run_auto(x)                     ← 便利接口，自动编码 + 早停
  ├── encode_sequence(x, T)     ← 编码
  └── step_encoded(x_t) loop    ← 核心原语 + Judger

forward_encoded(x_seq)          ← 核心原语，吃显式序列
  └── adapter.forward_multistep() 或 step_encoded() fallback
```

#### 10.6.3 Rate encoding 的等价性问题

**重要说明**: 对 stochastic rate encoding，三种执行路径**不自动等价**。

```
# 路径 A: run_auto — 内部编码，每步生成随机 bernoulli 样本
result_A = wrapper.run_auto(x)

# 路径 B: 显式编码 + stepwise
x_seq = wrapper.encode_sequence(x, T=64, seed=42)
result_B = sum(wrapper.step_encoded(x_seq[t]) for t in range(64))

# 路径 C: 显式编码 + batched
result_C = wrapper.forward_encoded(x_seq).sum(dim=0)

# result_B == result_C  ← 保证等价（消费同一份 x_seq）
# result_A != result_B  ← 不保证等价（不同的随机样本）
```

要使三种路径等价，必须**消费同一份显式 `x_seq`**。`encode_sequence` 的 `seed` 参数确保 rate encoding 可重放。

对 analog encoding，由于 `x_seq` 是确定性的（`[x, 0, 0, ...]`），三种路径天然等价。

#### 10.6.4 用法示例

```python
# 1. 开箱即用（当前用户无需改代码）
accu, T = wrapper.run_auto(x)

# 2. 自定义早停
wrapper.reset()
x_seq = wrapper.encode_sequence(x, T=64)
accu = 0
for t in range(64):
    accu += wrapper.step_encoded(x_seq[t])
    if my_custom_stop_condition(accu):
        break

# 3. 性能优化（固定 T，向量化）
wrapper.reset()
x_seq = wrapper.encode_sequence(x, T=64)
accu = wrapper.forward_encoded(x_seq).sum(dim=0)

# 4. 中间状态观测 / 调试
wrapper.reset()
x_seq = wrapper.encode_sequence(x, T=32)
for t in range(32):
    output_t = wrapper.step_encoded(x_seq[t])
    log_spike_rates(wrapper.model)
    log_membrane_potentials(wrapper.model)
```

---

### 10.7 ModelExecutionAdapter — 模型级执行适配

> **与早期设计的差异**: 早期版本将 model-specific 的 multi-step 路径直接放在 SNNWrapper 中（`_forward_multistep_vit` 等方法）。经 review 后明确：这样做会使 wrapper 膨胀为大泥球——每支持一个新模型就要在 wrapper 里加一套 `_forward_multistep_<model>` + `_forward_multistep_block_<model>` + `_forward_multistep_mlp_<model>` 方法。

**新设计**: 将 model-specific 逻辑抽离到 `ModelExecutionAdapter` registry 中。

#### 职责分离

| 组件 | 职责 | 不管什么 |
|------|------|---------|
| **SNNWrapper / executor** | 时序调度：编码、循环、早停、累积 | 模型内部结构 |
| **ModelExecutionAdapter** | 这个模型怎么走 step / multistep | 时序循环、早停策略 |

#### Adapter 接口

```python
class ModelExecutionAdapter:
    """模型执行适配器基类。

    每个 adapter 知道如何在特定模型架构上执行 single-step 和 multi-step forward。
    SNNWrapper 只管时序调度，不关心模型内部结构。
    """

    def step(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """执行单个时间步。

        处理 model-specific 的逻辑（如 ViT 的 pos_embed/cls_token 置零）。
        """
        raise NotImplementedError

    def forward_multistep(self, model: nn.Module, x_seq: torch.Tensor
                          ) -> torch.Tensor:
        """执行 multi-step forward。

        手动编排模型子模块的 forward_multistep 调用，
        绕过非 SNNOperator 模块的硬编码维度操作。
        """
        raise NotImplementedError
```

#### Adapter 注册

```python
ADAPTER_REGISTRY = Registry("model_execution_adapter")

@ADAPTER_REGISTRY.register("vit")
@ADAPTER_REGISTRY.register("deit")  # DeiT 与 ViT 共用同一 adapter
class ViTExecutionAdapter(ModelExecutionAdapter):
    """ViT/DeiT 执行适配器。"""

    def step(self, model, x_t, t):
        if t == 0:
            self._restore_embeddings(model)
        elif t == 1:
            self._zero_embeddings(model)
        return model(x_t)

    def forward_multistep(self, model, x_seq):
        # 手动编排 ViT 的 multi-step forward（见下方详细实现）
        ...

@ADAPTER_REGISTRY.register("swin")
class SwinExecutionAdapter(ModelExecutionAdapter):
    """Swin Transformer 执行适配器。"""
    ...
```

#### SNNWrapper 与 Adapter 的交互

```python
class SNNWrapper:
    def __init__(self, model, ..., adapter_name="vit"):
        self.adapter = ADAPTER_REGISTRY.get(adapter_name)()

    def step_encoded(self, x_t):
        return self.adapter.step(self.model, x_t, self._current_t)

    def forward_encoded(self, x_seq):
        return self.adapter.forward_multistep(self.model, x_seq)
```

#### ViT Adapter 的 forward_multistep 详细实现

```python
def forward_multistep(self, model, x_seq):
    """手动编排 ViT 的 multi-step forward。

    绕过 VisionTransformer.forward_features() 和 PatchEmbed.forward()
    中的硬编码维度操作，改为直接调用子模块的 forward_multistep。

    Args:
        x_seq: [T, B, C, H, W]
    Returns:
        output_seq: [T, B, num_classes]
    """
    T, B = x_seq.shape[:2]

    # 1. PatchEmbed — 手动处理 Conv2d + reshape
    x_seq = model.patch_embed.proj.forward_multistep(x_seq)  # → [T, B, C_out, H_out, W_out]
    x_seq = x_seq.flatten(3).permute(0, 1, 3, 2)             # → [T, B, N, D]
    #   对比 PatchEmbed.forward: x.flatten(2).transpose(1, 2)
    #   multi-step 下 dim 索引整体 +1

    # 2. cls_token — 仅 t=0 有值
    cls_seq = torch.zeros(T, B, 1, model.embed_dim, device=x_seq.device)
    cls_seq[0] = self._cls_token.expand(B, -1, -1)
    x_seq = torch.cat([cls_seq, x_seq], dim=2)               # → [T, B, N+1, D]

    # 3. pos_embed — 仅 t=0 有值
    pos_seq = torch.zeros(T, 1, x_seq.shape[2], model.embed_dim, device=x_seq.device)
    pos_seq[0] = self._pos_embed
    x_seq = x_seq + pos_seq                                   # → [T, B, N+1, D]

    # 4. Transformer Blocks
    for blk in model.blocks:
        x_seq = self._forward_multistep_block(blk, x_seq)

    # 5. Global pool + Head
    if model.global_pool:
        x_seq = x_seq[:, :, 1:, :].mean(dim=2)               # → [T, B, D]
    else:
        x_seq = model.norm.forward_multistep(x_seq)           # Spiking_LayerNorm
        x_seq = x_seq[:, :, 0]                                # → [T, B, D]

    x_seq = model.head.forward_multistep(x_seq)               # LLLinear → [T, B, num_classes]
    return x_seq
```

Block 级 multi-step forward:

```python
def _forward_multistep_block(self, blk, x_seq):
    """Block.forward 原始逻辑:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

    residual add 对 [T, B, ...] 天然兼容 (element-wise)。
    DropPath 在 eval 模式下是 identity。
    """
    # norm1 → attn → residual
    residual = x_seq
    x = blk.norm1.forward_multistep(x_seq)     # Spiking_LayerNorm (向量化)
    x = blk.attn.forward_multistep(x)          # SAttention
    x = residual + x

    # norm2 → mlp → residual
    residual = x
    x = blk.norm2.forward_multistep(x)         # Spiking_LayerNorm (向量化)
    x = self._forward_multistep_mlp(blk.mlp, x)
    x = residual + x
    return x
```

注意事项：
- `blk.norm1` / `blk.norm2` 在转换后可能是 `Sequential(Spiking_LayerNorm, IFNeuron)`，需要处理 Sequential 容器
- `blk.mlp` 内部包含 `fc1 (LLLinear)`, `act (Sequential(IFNeuron, Identity))`, `fc2 (Sequential(LLLinear, IFNeuron))`
- 每个子模块都需要显式调用其 `forward_multistep`

#### 与现有 Registry 的一致性

`ModelExecutionAdapter` registry 与现有的 `ModelProfile`、`ConversionRule` 风格一致：

| Registry | 注册什么 | 键 |
|----------|---------|-----|
| `MODEL_PROFILE_REGISTRY` | 模型硬件参数 | `vit_small`, `vit_base`, ... |
| `CONVERSION_RULE_REGISTRY` | ANN→SNN 转换规则 | (规则列表) |
| `ADAPTER_REGISTRY` | 模型执行适配器 | `vit`, `deit`, `swin`, ... |

---

### 10.8 Hook-based OpsCounter 与 multistep 的冲突

#### 问题

`OpsCounter` 使用 PyTorch 的 `register_forward_hook` 在 `nn.Module.__call__` 时拦截输入/输出，统计 SYOPS。但 `forward_multistep()` 是一个**普通方法调用**，不经过 `__call__`，因此 hooks **静默失效**。

```python
# stepwise 模式: 每步经过 __call__，hooks 正常触发
for t in range(T):
    output_t = module(x_seq[t])    # __call__ → hooks 触发 ✓

# multistep 模式: 直接调方法，hooks 不触发
output_seq = module.forward_multistep(x_seq)  # 普通方法 → hooks 不触发 ✗
```

#### 影响范围

| 评估指标 | stepwise | multistep | 说明 |
|----------|:---:|:---:|------|
| Accuracy | ✓ | ✓ | 不依赖 hooks，只看最终输出 |
| Energy/SYOPS | ✓ | ✗ | 依赖 hooks 统计运算量，multistep 下数据不可信 |
| Spike rate | ✓ | ✓ | 直接读算子状态，不依赖 hooks |

#### 需要明确的语义问题

`EnergyEvaluator` 统计的到底是什么？

- **语义 SNN 运算量**: 即使实际用向量化 cumsum 计算，也按"等价的逐步 SNN 执行"来计算 SYOPS。这是理论能耗。
- **实际 module 调用量**: 统计真正发生的 `forward` 调用次数和 FLOPs。这反映实际执行成本。

当前 hook-based 设计隐含的是"实际调用量"语义。在 stepwise 下两者一致；在 multistep 下两者分离。

#### 短期结论

**multistep 下的能耗统计标记为"尚未定义/未验证"。** 具体地：

- `AccuracyEvaluator`: multistep 下可信，正常使用
- `EnergyEvaluator`: multistep 下**不可信**，文档和 API 需明确标注
- 如需能耗统计，使用 stepwise 模式（`step_encoded` 或 `run_auto`）

长期方案候选：
1. 在 `forward_multistep` 内部手动触发 hooks（侵入性较强）
2. 为 multistep 提供独立的分析计数器（基于输入形状和算子参数推算，不依赖 hooks）
3. 要求用户先用 stepwise 跑一次获取统计，再用 multistep 跑性能

---

### 10.9 Judger 和 reset_model 更新

#### Judger — 统一类型检查

```python
# 当前 (hard-coded types):
if isinstance(child, (IFNeuron, LLLinear, LLConv2d)):
    self.network_finish &= not model._modules[name].is_work

# 改为 (属性驱动):
if isinstance(child, SNNOperator) and child.participates_in_early_stop:
    self.network_finish &= not child.working
```

**修复**: 当前实现遗漏了 `ORIIFNeuron`。改为属性驱动后，所有 `participates_in_early_stop=True` 的算子自动被检查，不再遗漏。

#### Judger — 预缓存模块列表

每次 `judge()` 调用都遍历 `model.named_children()` 递归查找算子，在推理热路径上开销不必要。应在 Judger 初始化后（或首次 reset 后）缓存两个列表：

```python
class Judger:
    def __init__(self, model, ...):
        # 初始化后缓存，避免每步重新遍历
        self._all_snn_ops = [
            m for m in model.modules() if isinstance(m, SNNOperator)
        ]
        self._early_stop_ops = [
            m for m in self._all_snn_ops if m.participates_in_early_stop
        ]

    def judge(self):
        self.network_finish = all(
            not op.working for op in self._early_stop_ops
        )
```

#### Judger — `working` 默认返回 False 的风险

如果一个新算子设置了 `participates_in_early_stop=True`（默认值）但忘记实现 `is_work` 属性，`working` property 将静默返回 `False`，导致 Judger 误判该算子已收敛，可能提前终止推理。

**防御措施**（已反映在 10.3 的 `working` property 更新中）：当 `participates_in_early_stop=True` 且 `is_work` 不存在时，抛出 `AttributeError` 而非静默返回 `False`。

#### reset_model — 统一类型检查

```python
# 当前 (hard-coded types, 递归时跳过已匹配节点):
def reset_model(model):
    for name, child in model.named_children():
        is_need = False
        if isinstance(child, (IFNeuron, LLConv2d, LLLinear, SAttention,
                              Spiking_LayerNorm, ORIIFNeuron)):
            child.reset()
            is_need = True
        if not is_need:
            reset_model(child)

# 改为 (平坦遍历):
def reset_model(model):
    for module in model.modules():
        if isinstance(module, SNNOperator):
            module.reset()
```

**修复**: 当前实现遗漏了 `spiking_softmax` 和 `SpikeMaxPooling`。

#### 与 SAttention.reset() 的交互（结论）

使用 `model.modules()` 平坦遍历后，SAttention 内部的 IFNeuron、LLLinear、spiking_softmax 都会被直接发现并各自 reset。而 SAttention.reset() 自身又手动调用了所有子算子的 reset()。这导致子算子被 reset 两次。

**结论**: 短期接受双重 reset（幂等，无害）。长期更干净的做法是将 `SAttention.reset()` 改为 local-only reset（只重置自身的 `self.T = 0`），子算子由 `reset_model` 的平坦遍历处理。但需注意：如果用户直接调用 `sattention.reset()` 而非 `reset_model()`，子算子不会被 reset——这需要在文档中明确说明。

---

### 10.10 `.cuda()` 硬编码修复

wrapper.py 中 4 处 `.cuda()` 需要替换为 `.to(device)`：

```python
# 位置 1: SNNWrapper.reset()
self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()      # ← .cuda()
self.model.cls_token.data = deepcopy(self.cls_token).cuda()       # ← .cuda()

# 位置 2-3: SNNWrapper.forward() 中 ViT embedding 置零
torch.zeros(...).to(x.device if torch.is_tensor(x) else "cuda")  # ← fallback "cuda"
```

修复方案：device 从输入张量推断，或在 `__init__` 中记录 device。

---

### 10.11 文件修改清单

| 文件 | 操作 | 改动内容 |
|------|------|---------|
| `operators/base.py` | 修改 | 添加 `participates_in_early_stop` 属性 + `forward_multistep()` 默认实现 + `working` 防御性检查 |
| `operators/neurons.py` | 不变 | 继承 SNNOperator，`participates_in_early_stop=True` 已是默认值 |
| `operators/layers.py` | 修改 | Spiking_LayerNorm/SpikeMaxPooling: 设 `participates_in_early_stop=False` + 重写 `forward_multistep`（含前态/终态同步） |
| `operators/attention.py` | 修改 | spiking_softmax: 设 `participates_in_early_stop=False` + 重写 `forward_multistep`（含前态/终态同步）；SAttention: 设 `participates_in_early_stop=False` |
| `conversion/wrapper.py` | 修改 | Judger 预缓存 + 属性驱动；reset_model 平坦遍历；新增 `step_encoded()` / `forward_encoded()` / `run_auto()` / `encode_sequence()`；修复 `.cuda()` |
| `conversion/adapter.py` | **新增** | `ModelExecutionAdapter` 基类 + `ViTExecutionAdapter` + `ADAPTER_REGISTRY` |
| `tests/test_verify.py` | 修改 | 新增: 各 multi-step 算子与 single-step 等价性测试（强契约验证）、`step_encoded()` API 测试、`forward_encoded()` 端到端测试 |
| `experiments/verify_equivalence.py` | 修改 | 多模式等价性验证 |

---

### 10.12 Q1-Q6 结论汇总

> 以下问题在早期版本中标记为 🔴 待讨论。经外部架构 review 后，已达成共识。

#### Q1: 模型级 Multi-Step 通路 — ✅ Adapter Registry

**结论**: 接受 model-specific 实现，但不放在 wrapper 中，而是通过 `ModelExecutionAdapter` registry（见 10.7）。

- SNNWrapper 不感知模型结构，只做时序调度
- 每个模型架构注册一个 adapter，adapter 知道如何编排该模型的 step / multistep
- ViT 和 DeiT 共用 `ViTExecutionAdapter`；Swin 单独注册
- 新增模型时只需写一个 adapter 并注册，不修改 wrapper

#### Q2: LLConv2d / LLLinear 的 forward_multistep — ✅ 默认 Fallback

**结论**: 先用默认 for-loop fallback，等 benchmark 证明这是瓶颈再优化。

理由：
- bias leakage 的序列依赖使向量化实现复杂度高
- 零输入的特殊路径在 multistep 下更难处理
- 在没有 throughput/memory benchmark 数据前，不值得引入复杂度
- 如果确认需要优化，方案 B（分离线性运算和 bias leakage）仍然可行

#### Q3: SAttention 的 forward_multistep — ✅ 默认 Fallback

**结论**: 使用默认 for-loop fallback。

理由：
- SAttention 中 IFNeuron 无法向量化，占主导计算量
- spiking_softmax 可向量化但占比小
- 混合向量化 + for-loop 实现复杂度高、易引入 bug
- 收益（一个 softmax 的向量化）不值得代价

#### Q4: SAttention.reset() 与 reset_model 的重复 reset — ✅ 短期双重 Reset

**结论**: 短期接受双重 reset（幂等，无害），长期 local-only reset 更干净。

具体：
- reset 操作是幂等的，双重 reset 不影响正确性
- 长期可将 `SAttention.reset()` 改为只重置 `self.T = 0`，子算子由 `reset_model` 平坦遍历处理
- 但需在文档中明确：直接调用 `sattention.reset()` 不会 reset 子算子

#### Q5: forward_multistep 后的状态一致性 — ✅ 强契约 + 状态同步

**结论**: 选方案 B 的升级版——**强语义契约**（见理念 4），`forward_multistep` 从任意当前状态继续，结束时同步最终状态。

具体：
- `forward_multistep` 的语义 ≡ 逐步调用 `forward()` 的输出和最终状态
- 向量化实现必须读取当前状态作为初始条件（`self.Y_pre` 作为 shifted 第一项）
- 向量化实现必须在结束时将状态更新为最终时间步的值
- 允许 `forward_multistep` 和 `forward()` 混用，因为状态始终保持一致
- 默认 for-loop fallback 天然满足契约

#### Q6: Multi-step 模式下的 early stop 替代 — ✅ Fixed Horizon + 可选 Chunked

**结论**: 不伪造 `actual_T`。full multistep = fixed-horizon 推理，可后续加 chunked mode。

具体：
- `forward_encoded(x_seq)` 处理完整的 T 步，返回 `[T, B, ...]`，不做早停
- `actual_T` 概念不适用于 full multistep——所有 T 步都被计算
- 如需兼顾性能和早停，可后续加 chunked mode：每次处理 chunk_size 步，检查早停条件后决定是否继续
- 用户如需精确的 `actual_T`，使用 `run_auto()`（stepwise + Judger）

---

### 10.13 已知问题与技术债

> 以下问题来自外部架构 review，目前不阻塞核心设计，但应在实现过程中逐步解决。

#### 10.13.1 API / 文档一致性漂移

Registry 命名存在不一致风险。例如：
- `MODEL_PROFILE_REGISTRY` vs `ADAPTER_REGISTRY` — 前者用全大写 + `_REGISTRY` 后缀，后者也应遵循同样的命名 (`MODEL_ADAPTER_REGISTRY`)
- 注册键的命名约定不统一（有的用下划线 `vit_small`，有的可能用连字符）

**建议**: 建立 registry 命名规范并在 CLAUDE.md 中记录。

#### 10.13.2 Duck-typing 量化规则缺乏 dry-run / explain 能力

当前 `QuantPlacementRule` 的 `match_fn` 使用 duck-typing（`hasattr(child, "attn") and hasattr(child, "mlp")`），灵活但不透明。用户无法预览"哪些模块会被量化"——只能跑一次看结果。

**建议**: 增加 `dry_run` / `explain` 模式，在不实际修改模型的情况下，列出所有匹配的模块和将要应用的规则。

```python
# 期望的 API
matches = quantizer.explain(model)
# → [("blocks.0", "transformer_block", QAttention),
#    ("blocks.1", "transformer_block", QAttention), ...]
```

#### 10.13.3 `.data` + `deepcopy` + 参数原地改写

`SNNWrapper` 中多处使用 `.data` 直接改写参数值（如 `self.model.pos_embed.data = deepcopy(...).cuda()`），以及 `LLConv2d` / `LLLinear` 中对 `conv.weight.data` 的原地缩放。

问题：
- `.data` 绕过 autograd 的版本计数，可能导致梯度计算错误（虽然当前场景是 eval-only，但不排除未来训练场景）
- `deepcopy` + `.data` 赋值的语义不清晰——是想创建独立副本还是共享参数？
- 原地改写使得"这个模块的权重到底是什么"难以追踪

**建议**: 短期接受（eval-only 场景无害），长期考虑用 `nn.Parameter` 正规赋值或 `load_state_dict` 替代。

#### 10.13.4 `participates_in_early_stop` 布尔属性不应继续膨胀

当前 `SNNOperator` 有一个布尔属性 `participates_in_early_stop`。如果未来需要更多行为标记（如 `supports_vectorization`, `has_temporal_dependency`, `requires_input_encoding` 等），逐个添加布尔属性会导致类接口膨胀。

**建议**: 如果布尔属性增长到 3 个以上，考虑改用 `capabilities: Set[str]` 或 `flags: IntFlag` 模式。当前只有一个属性，保持简单即可。

---

### 10.14 实现路线图

> 按外部 review 建议的落地顺序。每一步应独立可测试，不依赖后续步骤。

| 阶段 | 内容 | 依赖 | 可测试产出 |
|:---:|------|:---:|---------|
| **1** | 写 `forward_multistep` 强语义契约 + 等价性测试 | 无 | 每个算子的 multistep vs stepwise 等价性测试通过 |
| **2** | 落 `step_encoded()` / `should_stop()` / flat traversal `reset_model` | 1 | manual mode 用 `step_encoded` 跑通，结果与 `run_auto` 一致 |
| **3** | manual mode 做成真正低层原语（吃显式 `x_t`，不做编码） | 2 | 用户可用 `step_encoded` + 自定义早停跑完整推理 |
| **4** | 加 ViT adapter，向量化三个可向量化算子（Spiking_LayerNorm, SpikeMaxPooling, spiking_softmax）并正确处理前态/终态 | 1, 2 | `forward_encoded` 在 ViT 上跑通，结果与 stepwise 等价 |
| **5** | LLLinear / LLConv2d / SAttention 维持 for-loop fallback | 1 | 已由阶段 1 的等价性测试覆盖 |
| **6** | throughput + memory benchmark，决定是否需要 full multistep 优化 | 4 | benchmark 报告：stepwise vs batched 的延迟/显存对比 |
| **7** | energy / ops 统计跟上 multistep（在此之前不对外承诺 multistep 能耗数字） | 4, 6 | `EnergyEvaluator` 在 multistep 下的统计与 stepwise 一致 |

**关键原则**:
- 阶段 1-3 是基础设施，必须先落地
- 阶段 4-5 可并行开发
- 阶段 6 的 benchmark 结果决定后续优化投入方向
- 阶段 7 在能耗统计方案明确前，`EnergyEvaluator` 的 multistep 结果标记为 unreliable
