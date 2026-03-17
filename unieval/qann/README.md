# QANN — 量化 ANN

量化模块（运行时算子）和量化策略（放置规则）。

## 结构

- `operators/` — 参与 forward 计算的量化 `nn.Module`（MyQuan, PTQQuan, QConv2d, QLinear, QNorm, QAttention）
- `quantization/` — 决定"在模型树哪里放置哪种量化算子"的策略和规则

## 约定

- **operators/ 是运行时算子，quantization/ 是放置策略**，两者分离
- `composites.py` 中的 QConv2d/QLinear/QNorm 与 SNN composites 一对一映射，用于确定性转换
- 模型专属量化算子（如 QQwen3Attention）与其 placement rules 同置于 `quantization/*_rules.py`
- 所有 `match_fn` 使用 `protocols.py` 的 duck-typing 谓词，不用 isinstance 匹配 ANN 模型类
