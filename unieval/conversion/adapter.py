"""Model execution adapters for SNN temporal inference.

Each adapter knows how to execute single-step and multi-step forward
on a specific model architecture. SNNWrapper delegates execution to
the adapter, keeping itself free from model-specific knowledge.
"""

import torch
import torch.nn as nn

from ..registry import Registry

ADAPTER_REGISTRY = Registry("model_execution_adapters")


class ModelExecutionAdapter:
    """Base class for model execution adapters.

    Each adapter handles model-specific logic for:
    - step(): single timestep execution (e.g. ViT pos_embed/cls_token zeroing)
    - forward_multistep(): multi-step execution with manual sub-module orchestration
    """

    def step(self, model, x_t, t):
        """Execute a single timestep.

        Args:
            model: The SNN model.
            x_t: Input tensor for this timestep [B, ...].
            t: Current timestep index (0-based).
        Returns:
            Output tensor [B, ...].
        """
        raise NotImplementedError

    def forward_multistep(self, model, x_seq):
        """Execute multi-step forward.

        Manually orchestrates sub-module forward_multistep calls,
        bypassing non-SNNOperator modules' hardcoded dimension ops.

        Args:
            model: The SNN model.
            x_seq: Input sequence [T, B, ...].
        Returns:
            Output sequence [T, B, ...].
        """
        raise NotImplementedError


def _forward_multistep_sequential(seq, x_seq):
    """Run forward_multistep through an nn.Sequential container.

    Each child module's forward_multistep is called in order.
    For modules without forward_multistep (e.g. nn.Identity, nn.Dropout),
    falls back to applying the module per-timestep or element-wise.
    """
    for module in seq:
        if hasattr(module, "forward_multistep"):
            x_seq = module.forward_multistep(x_seq)
        else:
            T, B = x_seq.shape[:2]
            flat = x_seq.reshape(T * B, *x_seq.shape[2:])
            flat = module(flat)
            x_seq = flat.reshape(T, B, *flat.shape[1:])
    return x_seq


@ADAPTER_REGISTRY.register("vit")
@ADAPTER_REGISTRY.register("deit")
class ViTExecutionAdapter(ModelExecutionAdapter):
    """Execution adapter for ViT / DeiT architectures.

    Handles:
    - pos_embed / cls_token: full value at t=0, zero for t>0
    - Multi-step: manual orchestration of patch_embed, blocks, norm, head
    """

    def __init__(self, pos_embed, cls_token):
        """
        Args:
            pos_embed: Original pos_embed data (saved before conversion).
            cls_token: Original cls_token data (saved before conversion).
        """
        self.pos_embed = pos_embed
        self.cls_token = cls_token

    def step(self, model, x_t, t):
        device = x_t.device
        if t == 0:
            model.pos_embed.data = self.pos_embed.to(device)
            model.cls_token.data = self.cls_token.to(device)
        elif t == 1:
            model.pos_embed = nn.Parameter(
                torch.zeros_like(self.pos_embed, device=device)
            )
            model.cls_token = nn.Parameter(
                torch.zeros_like(self.cls_token, device=device)
            )
        return model(x_t)

    def forward_multistep(self, model, x_seq):
        """Manual orchestration of ViT multi-step forward.

        Args:
            x_seq: [T, B, C, H, W]
        Returns:
            output_seq: [T, B, num_classes]
        """
        T, B = x_seq.shape[:2]
        device = x_seq.device

        # 1. PatchEmbed — Conv2d + reshape
        x_seq = model.patch_embed.proj.forward_multistep(x_seq)
        x_seq = x_seq.flatten(3).permute(0, 1, 3, 2)  # [T, B, N, D]

        # 2. cls_token — only t=0 has value
        cls_seq = torch.zeros(
            T, B, 1, model.embed_dim, device=device, dtype=x_seq.dtype
        )
        cls_seq[0] = self.cls_token.to(device).expand(B, -1, -1)
        x_seq = torch.cat([cls_seq, x_seq], dim=2)  # [T, B, N+1, D]

        # 3. pos_embed — only t=0 has value
        pos_seq = torch.zeros(
            T, 1, x_seq.shape[2], model.embed_dim,
            device=device, dtype=x_seq.dtype
        )
        pos_seq[0] = self.pos_embed.to(device)
        x_seq = x_seq + pos_seq  # [T, B, N+1, D]

        # 4. Transformer Blocks
        for blk in model.blocks:
            x_seq = self._forward_multistep_block(blk, x_seq)

        # 5. Global pool + Head
        if model.global_pool:
            x_seq = x_seq[:, :, 1:, :].mean(dim=2)  # [T, B, D]
        else:
            x_seq = self._forward_multistep_submodule(model.norm, x_seq)
            x_seq = x_seq[:, :, 0]  # [T, B, D]

        x_seq = model.head.forward_multistep(x_seq)  # [T, B, num_classes]
        return x_seq

    def _forward_multistep_block(self, blk, x_seq):
        """Block-level multi-step forward.

        Original Block.forward:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        """
        # norm1 -> attn -> residual
        residual = x_seq
        x = self._forward_multistep_submodule(blk.norm1, x_seq)
        x = blk.attn.forward_multistep(x)
        x = residual + x

        # norm2 -> mlp -> residual
        residual = x
        x = self._forward_multistep_submodule(blk.norm2, x)
        x = self._forward_multistep_mlp(blk.mlp, x)
        x = residual + x
        return x

    def _forward_multistep_submodule(self, module, x_seq):
        """Forward multistep through a submodule that may be Sequential or direct."""
        if isinstance(module, nn.Sequential):
            return _forward_multistep_sequential(module, x_seq)
        elif hasattr(module, "forward_multistep"):
            return module.forward_multistep(x_seq)
        else:
            T, B = x_seq.shape[:2]
            flat = x_seq.reshape(T * B, *x_seq.shape[2:])
            flat = module(flat)
            return flat.reshape(T, B, *flat.shape[1:])

    def _forward_multistep_mlp(self, mlp, x_seq):
        """MLP multi-step: fc1 -> act -> fc2, each may be Sequential."""
        x_seq = self._forward_multistep_submodule(mlp.fc1, x_seq)
        x_seq = self._forward_multistep_submodule(mlp.act, x_seq)
        x_seq = self._forward_multistep_submodule(mlp.fc2, x_seq)
        return x_seq


@ADAPTER_REGISTRY.register("default")
class DefaultExecutionAdapter(ModelExecutionAdapter):
    """Fallback adapter: simple model(x_t) per step, no multistep support."""

    def step(self, model, x_t, t):
        return model(x_t)

    def forward_multistep(self, model, x_seq):
        return torch.stack([model(x_seq[t]) for t in range(x_seq.shape[0])])
