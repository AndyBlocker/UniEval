"""ModelProfile dataclass and model profile registry."""

from dataclasses import dataclass
from ..registry import MODEL_PROFILE_REGISTRY


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
