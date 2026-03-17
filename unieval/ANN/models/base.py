"""ModelProfile dataclass and model profile registry."""

from dataclasses import dataclass
from ...registry import MODEL_PROFILE_REGISTRY


@dataclass
class ModelProfile:
    """Architecture profile for energy evaluation.

    Replaces hard-coded ssa_info dictionaries.

    Attributes:
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        embed_dim: Embedding dimension.
        patch_size: Patch size (for ViT), used directly in SSA energy formula.
        img_size: Input image size (for computing num_patches).
        time_steps: Default number of SNN timesteps.
        mlp_ratio: MLP expansion ratio.
    """
    depth: int
    num_heads: int
    embed_dim: int
    patch_size: int = 16
    img_size: int = 224
    time_steps: int = 15
    mlp_ratio: float = 4.0

    @property
    def num_patches(self):
        """Number of patches: (img_size / patch_size) ^ 2."""
        return (self.img_size // self.patch_size) ** 2


# Register default ViT profiles
MODEL_PROFILE_REGISTRY.register_obj("vit_small", ModelProfile(
    depth=12, num_heads=6, embed_dim=384, patch_size=16, time_steps=15,
))

MODEL_PROFILE_REGISTRY.register_obj("vit_base", ModelProfile(
    depth=12, num_heads=12, embed_dim=768, patch_size=16, time_steps=64,
))

MODEL_PROFILE_REGISTRY.register_obj("vit_large", ModelProfile(
    depth=24, num_heads=16, embed_dim=1024, patch_size=16, time_steps=32,
))

MODEL_PROFILE_REGISTRY.register_obj("vit_huge", ModelProfile(
    depth=32, num_heads=16, embed_dim=1280, patch_size=14, time_steps=32,
))

MODEL_PROFILE_REGISTRY.register_obj("vit_small_dvs", ModelProfile(
    depth=12, num_heads=6, embed_dim=384, patch_size=16, time_steps=15,
))


@dataclass
class DecoderModelProfile(ModelProfile):
    """Extended profile for decoder-only models with GQA and sequence-based attention.

    Adds num_kv_heads for GQA-aware energy calculation, and seq_len
    which replaces patch_size^2 in the SSA energy formula.

    Attributes:
        num_kv_heads: Number of K/V attention heads (for GQA).
        seq_len: Sequence length (replaces patch_size^2 in SSA energy formula).
        head_dim: Per-head dimension.
        ffn_hidden_size: FFN intermediate dimension.
    """
    num_kv_heads: int = 8
    seq_len: int = 2048
    head_dim: int = 128
    ffn_hidden_size: int = 3072


MODEL_PROFILE_REGISTRY.register_obj("qwen3", DecoderModelProfile(
    depth=28, num_heads=16, embed_dim=1024,
    num_kv_heads=8, seq_len=2048, head_dim=128,
    ffn_hidden_size=3072, time_steps=64,
    patch_size=1, img_size=1,
))

MODEL_PROFILE_REGISTRY.register_obj("uniaffine", DecoderModelProfile(
    depth=28, num_heads=16, embed_dim=1024,
    num_kv_heads=8, seq_len=2048, head_dim=128,
    ffn_hidden_size=3072, time_steps=64,
    patch_size=1, img_size=1,  # Not applicable for decoder models
))
