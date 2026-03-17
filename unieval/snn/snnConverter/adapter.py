"""Model execution adapters for SNN temporal inference.

Each adapter knows how to execute single-step and multi-step forward
on a specific model architecture. SNNWrapper delegates execution to
the adapter, keeping itself free from model-specific knowledge.

Lifecycle API:
- supports(model): class-level duck-typing detection
- capture_context(model): save embeddings/params before conversion
- prepare_input(model, x): pre-process raw input (e.g. token IDs -> embeddings)
- reset_context(model): restore state modified by step()
"""

from copy import deepcopy

import torch
import torch.nn as nn

from ...registry import Registry
from ...protocols import is_decoder_model_like

ADAPTER_REGISTRY = Registry("model_execution_adapters")


class ModelExecutionAdapter:
    """Base class for model execution adapters.

    Each adapter handles model-specific logic for:
    - Lifecycle: supports, capture_context, prepare_input, reset_context
    - Execution: step() for single timestep, forward_multistep() for vectorized
    """

    @classmethod
    def supports(cls, model):
        """Duck-typing detection: can this adapter handle the given model?

        Args:
            model: The ANN/QANN model (before conversion).
        Returns:
            True if this adapter can handle the model.
        """
        return False

    def capture_context(self, model):
        """Save model context before conversion (e.g. pos_embed, cls_token).

        Called once during SNNWrapper.__init__ before conversion happens.
        The adapter should save any data that conversion might modify.

        Args:
            model: The ANN/QANN model (before conversion).
        """
        pass

    def prepare_input(self, model, x):
        """Pre-process raw input before temporal encoding.

        For decoder models: convert token IDs to embeddings.
        For ViT: identity (images are already in correct format).

        Args:
            model: The SNN model (after conversion).
            x: Raw input tensor [B, ...].
        Returns:
            Processed input tensor [B, ...].
        """
        return x

    def reset_context(self, model):
        """Restore state that step() may have modified.

        For ViT: restore pos_embed/cls_token that step() zeroed after t=0.
        Called by SNNWrapper.reset().

        Args:
            model: The SNN model.
        """
        pass

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


def _forward_multistep_submodule(module, x_seq):
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


def auto_detect_adapter(model):
    """Auto-detect the appropriate adapter for a model using duck-typing.

    Tries each registered adapter's supports() method in priority order:
    1. ViTExecutionAdapter (pos_embed + patch_embed + blocks)
    2. CausalDecoderAdapter (embedding + blocks + final_norm)
    3. DefaultExecutionAdapter (fallback)

    Args:
        model: The ANN/QANN model.
    Returns:
        An instantiated adapter.
    """
    # Try specific adapters first
    for name in ("vit", "causal_decoder"):
        if name in ADAPTER_REGISTRY:
            adapter_cls = ADAPTER_REGISTRY.get(name)
            if adapter_cls.supports(model):
                adapter = adapter_cls()
                adapter.capture_context(model)
                return adapter

    # Fallback
    adapter = DefaultExecutionAdapter()
    adapter.capture_context(model)
    return adapter


@ADAPTER_REGISTRY.register("vit")
@ADAPTER_REGISTRY.register("deit")
class ViTExecutionAdapter(ModelExecutionAdapter):
    """Execution adapter for ViT / DeiT architectures.

    Handles:
    - pos_embed / cls_token: full value at t=0, zero for t>0
    - Multi-step: manual orchestration of patch_embed, blocks, norm, head
    """

    def __init__(self, pos_embed=None, cls_token=None):
        """
        Args:
            pos_embed: Original pos_embed data (saved before conversion).
                If None, will be captured via capture_context().
            cls_token: Original cls_token data (saved before conversion).
                If None, will be captured via capture_context().
        """
        self.pos_embed = pos_embed
        self.cls_token = cls_token

    @classmethod
    def supports(cls, model):
        """ViT-like: has pos_embed, patch_embed, and blocks."""
        return (hasattr(model, "pos_embed") and hasattr(model, "patch_embed")
                and hasattr(model, "blocks"))

    def capture_context(self, model):
        """Save pos_embed and cls_token before conversion."""
        if self.pos_embed is None and hasattr(model, "pos_embed"):
            self.pos_embed = deepcopy(model.pos_embed.data)
        if self.cls_token is None and hasattr(model, "cls_token"):
            self.cls_token = deepcopy(model.cls_token.data)

    def reset_context(self, model):
        """Restore pos_embed/cls_token that step() may have zeroed."""
        if self.pos_embed is not None:
            device = next(model.parameters()).device
            model.pos_embed.data = self.pos_embed.to(device)
            model.cls_token.data = self.cls_token.to(device)

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

        # 1. PatchEmbed -- Conv2d + reshape (proj may be SConv2d or Sequential after conversion)
        x_seq = _forward_multistep_submodule(model.patch_embed.proj, x_seq)
        x_seq = x_seq.flatten(3).permute(0, 1, 3, 2)  # [T, B, N, D]

        # 2. cls_token -- only t=0 has value
        cls_seq = torch.zeros(
            T, B, 1, model.embed_dim, device=device, dtype=x_seq.dtype
        )
        cls_seq[0] = self.cls_token.to(device).expand(B, -1, -1)
        x_seq = torch.cat([cls_seq, x_seq], dim=2)  # [T, B, N+1, D]

        # 3. pos_embed -- only t=0 has value
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
            x_seq = _forward_multistep_submodule(model.fc_norm, x_seq)
        else:
            x_seq = _forward_multistep_submodule(model.norm, x_seq)
            x_seq = x_seq[:, :, 0]  # [T, B, D]

        x_seq = _forward_multistep_submodule(model.head, x_seq)  # [T, B, num_classes]
        return x_seq

    def _forward_multistep_block(self, blk, x_seq):
        """Block-level multi-step forward.

        Original Block.forward:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        """
        # norm1 -> attn -> drop_path -> residual
        residual = x_seq
        x = _forward_multistep_submodule(blk.norm1, x_seq)
        x = blk.attn.forward_multistep(x)
        x = _forward_multistep_submodule(blk.drop_path, x)
        x = residual + x

        # norm2 -> mlp -> drop_path -> residual
        residual = x
        x = _forward_multistep_submodule(blk.norm2, x)
        x = self._forward_multistep_mlp(blk.mlp, x)
        x = _forward_multistep_submodule(blk.drop_path, x)
        x = residual + x
        return x

    def _forward_multistep_mlp(self, mlp, x_seq):
        """MLP multi-step: fc1 -> act -> drop -> fc2 -> drop."""
        x_seq = _forward_multistep_submodule(mlp.fc1, x_seq)
        x_seq = _forward_multistep_submodule(mlp.act, x_seq)
        x_seq = _forward_multistep_submodule(mlp.drop, x_seq)
        x_seq = _forward_multistep_submodule(mlp.fc2, x_seq)
        x_seq = _forward_multistep_submodule(mlp.drop, x_seq)
        return x_seq


@ADAPTER_REGISTRY.register("default")
class DefaultExecutionAdapter(ModelExecutionAdapter):
    """Fallback adapter: simple model(x_t) per step, no multistep support."""

    @classmethod
    def supports(cls, model):
        return True  # Accepts anything as fallback

    def step(self, model, x_t, t):
        return model(x_t)

    def forward_multistep(self, model, x_seq):
        return torch.stack([model(x_seq[t]) for t in range(x_seq.shape[0])])


@ADAPTER_REGISTRY.register("causal_decoder")
@ADAPTER_REGISTRY.register("uniaffine")  # backward compat alias
class CausalDecoderAdapter(ModelExecutionAdapter):
    """Execution adapter for causal decoder-only models (UniAffine, Qwen3, etc).

    Handles:
    - Embedding: pre-applied via prepare_input before temporal encoding
    - Causal mask: preserved across all timesteps
    - RoPE: applied per token position inside spiking attention

    Args:
        causal_mask: Precomputed causal mask tensor [max_S, max_S].
    """

    def __init__(self, causal_mask=None):
        self.causal_mask = causal_mask
        self._embedding = None

    @classmethod
    def supports(cls, model):
        """Decoder-like: has embedding, blocks, final_norm."""
        return is_decoder_model_like(model)

    def capture_context(self, model):
        """Save embedding layer reference and causal mask before conversion."""
        if hasattr(model, "embedding"):
            self._embedding = model.embedding
        if self.causal_mask is None and hasattr(model, "causal_mask"):
            self.causal_mask = model.causal_mask.clone()

    def prepare_input(self, model, x):
        """Convert token IDs to embeddings.

        Args:
            model: The SNN model.
            x: Token IDs [B, S] (long tensor).
        Returns:
            Embeddings [B, S, hidden_size].
        """
        if self._embedding is not None:
            return self._embedding(x)
        return x

    def step(self, model, x_t, t):
        """Execute a single timestep on pre-embedded input.

        x_t is already embedded (prepare_input handles embedding
        before temporal encoding). Bypasses model.forward() and calls
        blocks + final_norm + lm_head directly.
        """
        x = x_t
        S = x.shape[1] if x.dim() >= 3 else 1

        if self.causal_mask is not None:
            causal_mask = self.causal_mask[:S, :S].to(x.device)
        elif hasattr(model, "causal_mask"):
            causal_mask = model.causal_mask[:S, :S].to(x.device)
        else:
            causal_mask = None

        for blk in model.blocks:
            x = blk(x, causal_mask=causal_mask)

        x = model.final_norm(x)

        if model.lm_head is not None:
            x = model.lm_head(x)
        else:
            x = torch.nn.functional.linear(x, model.embedding.weight)

        return x

    def forward_multistep(self, model, x_seq):
        """Manual orchestration of decoder multi-step forward.

        Args:
            x_seq: [T, B, S, hidden_size] pre-embedded sequence.
        Returns:
            output_seq: [T, B, S, vocab_size]
        """
        S = x_seq.shape[2]
        device = x_seq.device
        if self.causal_mask is not None:
            causal_mask = self.causal_mask[:S, :S].to(device)
        elif hasattr(model, "causal_mask"):
            causal_mask = model.causal_mask[:S, :S].to(device)
        else:
            causal_mask = None

        for blk in model.blocks:
            x_seq = self._forward_multistep_block(blk, x_seq, causal_mask)

        x_seq = _forward_multistep_submodule(model.final_norm, x_seq)

        if model.lm_head is not None:
            x_seq = _forward_multistep_submodule(model.lm_head, x_seq)
        else:
            T_len, B_size = x_seq.shape[:2]
            flat = x_seq.reshape(T_len * B_size, *x_seq.shape[2:])
            flat = torch.nn.functional.linear(flat, model.embedding.weight)
            x_seq = flat.reshape(T_len, B_size, *flat.shape[1:])

        return x_seq

    def _forward_multistep_block(self, blk, x_seq, causal_mask=None):
        """Block multi-step: norm1->attn->res, norm2->mlp->res."""
        residual = x_seq
        x = _forward_multistep_submodule(blk.norm1, x_seq)
        x = blk.attn.forward_multistep(x, causal_mask=causal_mask)
        x = residual + x

        residual = x
        x = _forward_multistep_submodule(blk.norm2, x)
        # After conversion, mlp is Spiking_ReGLUMlp/Spiking_SwiGLUMlp with forward_multistep
        x = _forward_multistep_submodule(blk.mlp, x)
        x = residual + x
        return x


# Backward compatibility alias
UniAffineExecutionAdapter = CausalDecoderAdapter
