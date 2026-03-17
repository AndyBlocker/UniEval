# SNN — 脉冲神经网络

SNN 算子和 ANN→SNN 转换引擎。

## 结构

- `operators/` — SNN 算子（神经元、层包装器、脉冲注意力等），都继承 `SNNOperator` mixin
- `snnConverter/` — 转换引擎（规则、转换器、SNNWrapper、执行适配器）

## 约定

### SNNOperator 契约

所有 SNN 算子必须：
1. `reset()` — 清空时序状态
2. 维护 `is_work` 字段 — 标记是否活跃（`Judger` 通过 `working` property 读取）
3. `forward_multistep(x_seq)` — 语义等价于逐步 forward 循环，输出和最终状态必须一致

### 阈值传递

`transfer_threshold(quan, neuron, ...)` 将量化器的 `s`/`pos_max`/`neg_min` 映射到神经元阈值和脉冲边界。

### Adapter 模式

`SNNWrapper` 不包含模型特定逻辑，委托给适配器（ViT / CausalDecoder / Default）处理 pos_embed、embedding、causal mask 等。
