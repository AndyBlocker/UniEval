# UniEval Technical Report

## 1. 项目概述

UniEval 是从 SpikeZIP-TF（ICML 2024）中提取并重构的通用 SNN 转换与评估框架。原始代码中大量 hard-coded 逻辑（模型结构匹配、能耗常量、算子映射等）被重构为 Registry + Rule-based 架构，使框架可扩展到新模型、新量化方法、新评估指标。

**当前状态**: 38/38 单元测试通过，QANN→SNN 等价性验证通过（所有层 cosine similarity = 1.0）。

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
