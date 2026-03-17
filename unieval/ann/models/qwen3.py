"""Standalone Qwen3 baseline model (0.6B decoder-only transformer).

Standard softmax attention + SwiGLU MLP + RMSNorm + RoPE + GQA.
This is the baseline model before any UniAffine modifications.

Reuses RoPE and GQA infrastructure from the uniaffine module.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .uniaffine import RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 0.6B model."""
    vocab_size: int = 151936
    num_layers: int = 28
    hidden_size: int = 1024
    ffn_hidden_size: int = 3072
    num_heads: int = 16          # Q heads
    num_kv_heads: int = 8        # K/V heads (GQA)
    head_dim: int = 128
    max_seq_len: int = 2048
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        hidden_size: Normalized dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


# ---------------------------------------------------------------------------
# Standard Softmax Attention with GQA
# ---------------------------------------------------------------------------

class Qwen3Attention(nn.Module):
    """Standard softmax multi-head attention with GQA and RoPE.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of Q heads.
        num_kv_heads: Number of K/V heads (GQA).
        head_dim: Per-head dimension.
        rope: Shared RotaryEmbedding instance.
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.rope = rope

        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_size, qkv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, causal_mask=None):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        q = qkv[..., :q_dim].reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., q_dim:q_dim + kv_dim].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim:].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(q, seq_len=S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expand
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        # Scaled dot-product attention with softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            scores = scores + causal_mask
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMlp(nn.Module):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).

    Uses SiLU (Swish) gating — the standard Qwen3 / LLaMA MLP.

    Args:
        hidden_size: Input/output dimension.
        ffn_hidden_size: Intermediate dimension.
    """

    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Qwen3 Decoder Block
# ---------------------------------------------------------------------------

class Qwen3Block(nn.Module):
    """Qwen3 decoder block: RMSNorm -> attn -> res -> RMSNorm -> mlp -> res.

    Args:
        config: Qwen3Config.
        rope: Shared RotaryEmbedding instance.
    """

    def __init__(self, config, rope):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Qwen3Attention(
            config.hidden_size, config.num_heads, config.num_kv_heads,
            config.head_dim, rope,
        )
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMlp(config.hidden_size, config.ffn_hidden_size)

    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Qwen3 Model
# ---------------------------------------------------------------------------

class Qwen3Model(nn.Module):
    """Qwen3 0.6B baseline decoder-only model.

    Standard softmax attention + SwiGLU + RMSNorm.

    Args:
        config: Qwen3Config.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = Qwen3Config()
        self.config = config
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.blocks = nn.ModuleList([
            Qwen3Block(config, rope)
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
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
            return self.lm_head(x)
        return F.linear(x, self.embedding.weight)


# ---------------------------------------------------------------------------
# Checkpoint Conversion: HuggingFace Qwen3 -> Standalone
# ---------------------------------------------------------------------------

def convert_hf_qwen3_state_dict(hf_state_dict, config=None):
    """Convert HuggingFace Qwen3 state_dict to standalone model keys.

    Args:
        hf_state_dict: State dict from HuggingFace Qwen3 model.
        config: Qwen3Config.
    Returns:
        Converted state dict compatible with Qwen3Model.
    """
    if config is None:
        config = Qwen3Config()
    new_sd = {}
    for key, value in hf_state_dict.items():
        k = key
        if k.startswith("model."):
            k = k[len("model."):]

        if k == "embed_tokens.weight":
            new_sd["embedding.weight"] = value
        elif k == "norm.weight":
            new_sd["final_norm.weight"] = value
        elif k == "lm_head.weight":
            if not config.tie_word_embeddings:
                new_sd["lm_head.weight"] = value
        elif k.startswith("layers."):
            parts = k.split(".")
            idx = parts[1]
            rest = ".".join(parts[2:])
            prefix = f"blocks.{idx}"

            mapping = {
                "input_layernorm.weight": "norm1.weight",
                "post_attention_layernorm.weight": "norm2.weight",
                "self_attn.q_proj.weight": None,  # fused below
                "self_attn.k_proj.weight": None,
                "self_attn.v_proj.weight": None,
                "self_attn.o_proj.weight": "attn.o_proj.weight",
                "mlp.gate_proj.weight": "mlp.gate_proj.weight",
                "mlp.up_proj.weight": "mlp.up_proj.weight",
                "mlp.down_proj.weight": "mlp.down_proj.weight",
            }

            if rest in mapping and mapping[rest] is not None:
                new_sd[f"{prefix}.{mapping[rest]}"] = value
            elif rest == "self_attn.q_proj.weight":
                # Store for later fusing
                new_sd[f"_tmp_{prefix}.q_weight"] = value
            elif rest == "self_attn.k_proj.weight":
                new_sd[f"_tmp_{prefix}.k_weight"] = value
            elif rest == "self_attn.v_proj.weight":
                new_sd[f"_tmp_{prefix}.v_weight"] = value

    # Fuse Q, K, V into qkv_proj
    fuse_keys = set()
    for key in list(new_sd.keys()):
        if key.startswith("_tmp_") and key.endswith(".q_weight"):
            prefix = key[len("_tmp_"):].replace(".q_weight", "")
            q = new_sd.pop(f"_tmp_{prefix}.q_weight")
            k = new_sd.pop(f"_tmp_{prefix}.k_weight")
            v = new_sd.pop(f"_tmp_{prefix}.v_weight")
            new_sd[f"{prefix}.attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)

    return new_sd


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def qwen3_model(config=None, **kwargs):
    """Create a Qwen3Model."""
    if config is None:
        config = Qwen3Config(**{k: v for k, v in kwargs.items()
                                if k in Qwen3Config.__dataclass_fields__})
    return Qwen3Model(config)
