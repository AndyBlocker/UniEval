# engine — 应用层

`UniEvalRunner` 是面向用户的高层入口，编排完整 pipeline。

## 职责

- 模型工厂（model_name → 工厂函数）
- 量化策略选择（按模型选 PTQ rule pack）
- PTQ 校准
- SNN 转换（自动检测 converter 和 adapter）
- Checkpoint 加载和 key 映射
- 评估器装配

## 约定

- engine/ 依赖所有子包，**其他包不应反向 import engine/**
- 模型专属 dispatch（PTQ rules、converter 选择）目前硬编码在 `runner.py` 和 `wrapper.py` 中
- 添加新模型族要么修改 dispatch 逻辑，要么手动传 `PTQQuantizer(rules=...)` / `SNNConverter(rules=...)`
