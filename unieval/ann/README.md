# ANN — 纯神经网络模型

参考模型实现和共享 ANN 算子。

## 结构

- `models/` — 模型定义 + ModelProfile 注册（ViT、Qwen3、UniAffine）
- `operators/` — 共享算子（RoPE，供 ANN 模型和 SNN 算子共用）

## 约定

- models/ 是**参考实现**，不是框架核心。核心层 (QANN, SNN) 不应 import 这里的模型类
- 模型属性命名需与 `protocols.py` 的 duck-typing 谓词一致（如 decoder 需有 `embedding`, `blocks`, `final_norm`）
- `ModelProfile` 通过 `MODEL_PROFILE_REGISTRY` 注册，供能耗评估使用
- ViT 属性名兼容 timm 0.3.2 权重
