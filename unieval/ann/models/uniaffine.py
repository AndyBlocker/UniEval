"""Standalone UniAffine model (Qwen3-0.6B architecture baseline).

Softmax-free decoder-only transformer using:
- UnifiedClipNorm instead of LayerNorm
- UniAffine attention: gamma * clamp(relu(act_a * QK^T/scale + act_b), 0, 1)
- ReGLU MLP (ReLU-gated linear unit)
- RoPE positional encoding
- GQA (Grouped Query Attention)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UniAffineConfig:
    """Configuration for UniAffine model."""
    vocab_size: int = 151936
    num_layers: int = 28
    hidden_size: int = 1024
    ffn_hidden_size: int = 3072
    num_heads: int = 16          # Q heads
    num_kv_heads: int = 8        # K/V heads (GQA)
    head_dim: int = 128
    max_seq_len: int = 2048
    rope_theta: float = 1e6
    norm_clip_min: float = -1.0    # UnifiedClipNorm range for norms
    norm_clip_max: float = 1.0
    attn_clip_min: float = 0.0     # UniAffineCore range for attention
    attn_clip_max: float = 1.0
    tie_word_embeddings: bool = True


# ---------------------------------------------------------------------------
# UnifiedClipNorm
# ---------------------------------------------------------------------------

class UnifiedClipNorm(nn.Module):
    """y = gamma * clamp(alpha * x + beta, clip_min, clip_max)

    Standalone implementation of Megatron's UnifiedClipNorm,
    without MegatronModule or torch.compile dependencies.

    Args:
        hidden_size: Normalized dimension.
        clip_min: Lower clamp bound.
        clip_max: Upper clamp bound.
    """

    def __init__(self, hidden_size, clip_min=-1.0, clip_max=1.0,
                 init_alpha=0.8, init_gamma=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = nn.Parameter(torch.full((hidden_size,), init_alpha))
        self.gamma = nn.Parameter(torch.full((hidden_size,), init_gamma))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.register_buffer("clip_min", torch.tensor(clip_min))
        self.register_buffer("clip_max", torch.tensor(clip_max))

    def forward(self, x):
        y = self.alpha * x + self.beta
        y = torch.clamp(y, self.clip_min, self.clip_max)
        y = self.gamma * y
        return y

    def extra_repr(self):
        return (f"hidden_size={self.hidden_size}, "
                f"clip_min={self.clip_min.item()}, clip_max={self.clip_max.item()}")


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding.

    Args:
        head_dim: Per-head dimension.
        max_seq_len: Maximum sequence length.
        theta: RoPE base frequency.
    """

    def __init__(self, head_dim, max_seq_len=2048, theta=1e6):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype,
                         device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# RoPE utilities moved to operators/rope.py (canonical location).
# Re-exported here for backward compatibility (models/qwen3.py imports from here).
from ..operators.rope import _rotate_half, apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# UniAffine Attention Core
# ---------------------------------------------------------------------------

class UniAffineCore(nn.Module):
    """Core UniAffine activation: gamma * clamp(relu(act_a * scores + act_b), 0, 1).

    Learnable per-head parameters act_a, act_b, gamma.

    Args:
        num_heads: Number of Q attention heads.
        head_dim: Per-head dimension.
    """

    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Scale: 1 / (sqrt(head_dim) * 4096) from Triton kernel
        self.register_buffer(
            "scale", torch.tensor(1.0 / (math.sqrt(head_dim) * 4096))
        )
        self.act_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.act_b = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.gamma = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, scores):
        """Apply UniAffine activation to attention scores.

        Args:
            scores: [B, num_heads, S, S] raw Q*K^T (already scaled).
        """
        x = self.act_a * scores + self.act_b
        x = F.relu(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.gamma * x
        return x


# ---------------------------------------------------------------------------
# UniAffine Attention Block (with GQA)
# ---------------------------------------------------------------------------

class UniAffineAttention(nn.Module):
    """UniAffine attention: QKV proj -> RoPE -> UniAffineCore -> output proj.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of Q heads.
        num_kv_heads: Number of K/V heads (GQA).
        head_dim: Per-head dimension.
        rope: RotaryEmbedding instance (shared across layers).
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.rope = rope

        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_size, qkv_dim, bias=False)
        self.core = UniAffineCore(num_heads, head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, causal_mask=None):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_dim].reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_dim:q_dim + kv_dim].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim:].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, seq_len=S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.core.scale
        if causal_mask is not None:
            scores = scores + causal_mask
        attn_weights = self.core(scores)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# ReGLU MLP (ReLU-Gated Linear Unit)
# ---------------------------------------------------------------------------

class ReGLUMlp(nn.Module):
    """ReGLU MLP: down_proj(relu(gate_proj(x)) * up_proj(x)).

    Uses ReLU gating (not SiLU/SwiGLU). ReLU is naturally compatible
    with SNN conversion (ReLU -> Identity).

    Args:
        hidden_size: Input/output dimension.
        ffn_hidden_size: Intermediate dimension.
    """

    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# UniAffine Decoder Block
# ---------------------------------------------------------------------------

class UniAffineBlock(nn.Module):
    """Single decoder block: norm1 -> attn -> residual -> norm2 -> mlp -> residual.

    Args:
        config: UniAffineConfig.
        rope: Shared RotaryEmbedding instance.
    """

    def __init__(self, config, rope):
        super().__init__()
        self.norm1 = UnifiedClipNorm(config.hidden_size,
                                     config.norm_clip_min, config.norm_clip_max)
        self.attn = UniAffineAttention(
            config.hidden_size, config.num_heads, config.num_kv_heads,
            config.head_dim, rope,
        )
        self.norm2 = UnifiedClipNorm(config.hidden_size,
                                     config.norm_clip_min, config.norm_clip_max)
        self.mlp = ReGLUMlp(config.hidden_size, config.ffn_hidden_size)

    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# UniAffine Model
# ---------------------------------------------------------------------------

class UniAffineModel(nn.Module):
    """Softmax-free decoder-only transformer (Qwen3-0.6B architecture baseline).

    Args:
        config: UniAffineConfig with all model hyperparameters.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = UniAffineConfig()
        self.config = config
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.blocks = nn.ModuleList([
            UniAffineBlock(config, rope)
            for _ in range(config.num_layers)
        ])
        self.final_norm = UnifiedClipNorm(config.hidden_size,
                                          config.norm_clip_min, config.norm_clip_max)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        mask = torch.full((config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, input_ids):
        B, S = input_ids.shape
        x = self.embedding(input_ids)
        causal_mask = self.causal_mask[:S, :S]
        for blk in self.blocks:
            x = blk(x, causal_mask=causal_mask)
        x = self.final_norm(x)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embedding.weight)
        return logits


# ---------------------------------------------------------------------------
# Checkpoint Conversion: Megatron -> Standalone
# ---------------------------------------------------------------------------

def convert_megatron_state_dict(raw_state_dict, config=None):
    """Convert Megatron dist checkpoint state_dict to standalone model keys.

    Steps:
    1. Strip "model." prefix if present
    2. Map Megatron key names to standalone model key names
    3. Split fused linear_fc1 (2*ffn) into gate_proj + up_proj
    4. Map layer norm names (input_layernorm->norm1, pre_mlp_layernorm->norm2)
    5. Map core_attention parameters (act_a, act_b, gamma)

    Args:
        raw_state_dict: State dict from Megatron's model.state_dict().
        config: UniAffineConfig (defaults to UniAffineConfig()).
    Returns:
        Converted state dict compatible with UniAffineModel.
    """
    if config is None:
        config = UniAffineConfig()

    new_sd = {}

    for key, value in raw_state_dict.items():
        k = key
        if k.startswith("model."):
            k = k[len("model."):]

        if k == "embedding.word_embeddings.weight":
            new_sd["embedding.weight"] = value
            continue

        if k.startswith("decoder.final_layernorm."):
            suffix = k.split("decoder.final_layernorm.")[-1]
            new_sd[f"final_norm.{suffix}"] = value
            continue

        if k == "output_layer.weight":
            if not config.tie_word_embeddings:
                new_sd["lm_head.weight"] = value
            continue

        if k.startswith("decoder.layers."):
            parts = k.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])
            prefix = f"blocks.{layer_idx}"

            if rest.startswith("input_layernorm."):
                param = rest.split("input_layernorm.")[-1]
                new_sd[f"{prefix}.norm1.{param}"] = value
            elif rest.startswith("pre_mlp_layernorm."):
                param = rest.split("pre_mlp_layernorm.")[-1]
                new_sd[f"{prefix}.norm2.{param}"] = value
            elif rest == "self_attention.linear_qkv.weight":
                new_sd[f"{prefix}.attn.qkv_proj.weight"] = value
            elif rest == "self_attention.linear_proj.weight":
                new_sd[f"{prefix}.attn.o_proj.weight"] = value
            elif rest.startswith("self_attention.core_attention."):
                param = rest.split("self_attention.core_attention.")[-1]
                new_sd[f"{prefix}.attn.core.{param}"] = value
            elif rest == "mlp.linear_fc1.weight":
                ffn = config.ffn_hidden_size
                new_sd[f"{prefix}.mlp.gate_proj.weight"] = value[:ffn]
                new_sd[f"{prefix}.mlp.up_proj.weight"] = value[ffn:]
            elif rest == "mlp.linear_fc2.weight":
                new_sd[f"{prefix}.mlp.down_proj.weight"] = value
            elif "rotary_pos_emb" in rest:
                pass  # Shared RoPE, skip per-layer
            else:
                new_sd[f"_unmapped_.{key}"] = value
        elif k not in ("",):
            new_sd[f"_unmapped_.{key}"] = value

    return new_sd


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def uniaffine_model(config=None, **kwargs):
    """Create a UniAffineModel.

    Args:
        config: Optional UniAffineConfig. If None, uses defaults (Qwen3-0.6B).
    """
    if config is None:
        config = UniAffineConfig(**{k: v for k, v in kwargs.items()
                                    if k in UniAffineConfig.__dataclass_fields__})
    return UniAffineModel(config)
