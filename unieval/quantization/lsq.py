"""LSQ (Learned Step-size Quantization) implementation.

Contains MyQuan, QAttention, QuanConv2d, QuanLinear, and LSQQuantizer.
"""

import math

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseQuantizer, QuantPlacementRule
from ..registry import QUANTIZER_REGISTRY


# ---------------------------------------------------------------------------
# STE helper functions
# ---------------------------------------------------------------------------

def grad_scale(x, scale):
    """Straight-Through Estimator with gradient scaling."""
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def floor_pass(x):
    """Floor with straight-through gradient."""
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    """Round with straight-through gradient."""
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def threshold_optimization(data, quantization_level=255, n_trial=300, eps=1e-10):
    """Find optimized clipping threshold using KL-divergence.

    Originated from post-training quantization (TensorRT style).

    Args:
        data: Activation data as numpy array.
        quantization_level: Number of quantization levels.
        n_trial: Number of search steps.
        eps: Numerical stability constant.

    Returns:
        Optimal threshold value.
    """
    n_lvl = quantization_level
    n_half_lvls = quantization_level // 2
    n_bin_edge = n_lvl * n_trial + 1

    data_max = np.max(np.abs(data))
    hist, bin_edge = np.histogram(
        data.flatten(),
        bins=np.linspace(-data_max, data_max, num=n_bin_edge),
    )

    mid_idx = int(len(hist) / 2)
    start_idx = 100
    kl_result = np.empty([len(range(start_idx, n_trial + 1)), 2])

    for i in range(start_idx, n_trial + 1):
        ref_dist = np.copy(
            hist[mid_idx - i * n_half_lvls:mid_idx + i * n_half_lvls]
        )
        ref_dist[0] += hist[:mid_idx - i * n_half_lvls].sum()
        ref_dist[-1] += hist[mid_idx + i * n_half_lvls:].sum()

        ref_dist_reshape = ref_dist.reshape(n_lvl, i)
        ref_dist_merged = ref_dist_reshape.sum(axis=1)
        nonzero_mask = ref_dist_reshape != 0
        average_bin_count = ref_dist_merged / (nonzero_mask.sum(1) + eps)
        expand_bin_count = np.expand_dims(average_bin_count, axis=1).repeat(i, axis=1)
        candidate_dist = (nonzero_mask * expand_bin_count).flatten()
        kl_div = scipy.stats.entropy(
            candidate_dist / candidate_dist.sum(),
            ref_dist / ref_dist.sum(),
        )
        current_th = np.abs(bin_edge[mid_idx - i * n_half_lvls])
        kl_result[i - start_idx, 0] = current_th
        kl_result[i - start_idx, 1] = kl_div

    th_sel = kl_result[kl_result[:, 1] == kl_result[:, 1].min()][0, 0]
    return th_sel


# ---------------------------------------------------------------------------
# Quantization modules
# ---------------------------------------------------------------------------

class MyQuan(nn.Module):
    """Learned step-size quantization module.

    Args:
        level: Number of quantization levels.
        sym: If True, symmetric quantization.
    """

    def __init__(self, level, sym=False):
        super().__init__()
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 512:
            self.pos_max = "full"
        else:
            if sym:
                self.pos_max = torch.tensor(float(level // 2 - 1))
                self.neg_min = torch.tensor(float(-level // 2))
            else:
                self.pos_max = torch.tensor(float(level - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0))
        self.batch_init = 20
        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"

    def __repr__(self):
        return (
            f"MyQuan(level={self.level}, sym={self.sym}, "
            f"pos_max={self.pos_max}, neg_min={getattr(self, 'neg_min', None)}, "
            f"s={self.s.data})"
        )

    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self, name, tfwriter, global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    def forward(self, x):
        if self.pos_max == "full":
            return x

        if str(self.neg_min.device) == "cpu":
            self.neg_min = self.neg_min.to(x.device)
        if str(self.pos_max.device) == "cpu":
            self.pos_max = self.pos_max.to(x.device)
        min_val = self.neg_min
        max_val = self.pos_max

        s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)

        if self.init_state == 0 and self.training:
            self.s.data = torch.tensor(
                x.detach().abs().mean() * 2
                / (self.pos_max.detach().abs().mean() ** 0.5),
                dtype=torch.float32,
            ).cuda()
            self.init_state += 1

        s_scale = grad_scale(self.s, s_grad_scale)
        output = torch.clamp(
            floor_pass(x / s_scale + 0.5), min=min_val, max=max_val
        ) * s_scale

        if self.debug and self.tfwriter is not None:
            self.tfwriter.add_histogram(
                tag=f"before_quan/{self.name}_data",
                values=x.detach().cpu(),
                global_step=self.global_step,
            )
            self.tfwriter.add_histogram(
                tag=f"after_quan/{self.name}_data",
                values=torch.clamp(
                    floor_pass(x / s_scale + 0.5), min=min_val, max=max_val
                ).detach().cpu(),
                global_step=self.global_step,
            )
            self.debug = False
            self.tfwriter = None
            self.name = ""
            self.global_step = 0.0

        return output


class QAttention(nn.Module):
    """Quantized multi-head attention.

    Quantizes Q, K, V, attention scores and projection output using MyQuan.

    Args:
        dim: Total embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias: If True, add bias to QKV.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
        level: Quantization level.
        is_softmax: Whether to use softmax attention.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        level=2,
        is_softmax=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.level = level
        self.is_softmax = is_softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.quan_q = MyQuan(self.level, sym=True)
        self.quan_k = MyQuan(self.level, sym=True)
        self.quan_v = MyQuan(self.level, sym=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.quan_proj = MyQuan(self.level, sym=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_quan = MyQuan(self.level, sym=False)
        self.after_attn_quan = MyQuan(self.level, sym=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.quan_q(q)
        k = self.quan_k(k)
        v = self.quan_v(v)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        if self.is_softmax:
            attn = attn.softmax(dim=-1)
            attn = self.attn_quan(attn)
        else:
            attn = self.attn_quan(attn) / N

        attn = self.attn_drop(attn)
        x = attn @ v
        x = self.after_attn_quan(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.quan_proj(x)
        return x


class QuanConv2d(torch.nn.Conv2d):
    """Conv2d with quantized weights."""

    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(
            m.in_channels, m.out_channels, m.kernel_size,
            stride=m.stride, padding=m.padding,
            dilation=m.dilation, groups=m.groups,
            bias=m.bias is not None,
            padding_mode=m.padding_mode,
        )
        self.quan_w_fn = quan_w_fn
        self.weight = torch.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return self._conv_forward(x, quantized_weight, self.bias)


class QuanLinear(torch.nn.Linear):
    """Linear with quantized weights."""

    def __init__(self, m: torch.nn.Linear, quan_w_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(
            m.in_features, m.out_features,
            bias=m.bias is not None,
        )
        self.quan_w_fn = quan_w_fn
        self.weight = torch.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return torch.nn.functional.linear(x, quantized_weight, self.bias)


class Attention_no_softmax(nn.Module):
    """Softmax-free attention using ReLU, divides by sequence length N."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_Relu = nn.ReLU(inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_Relu(attn) / N
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Default placement rules for LSQ quantization
# ---------------------------------------------------------------------------

def _match_transformer_block(name, child, parent):
    """Duck-typed match: any module with .attn and .mlp submodules."""
    return hasattr(child, "attn") and hasattr(child, "mlp")


def _apply_transformer_block(name, child, parent, level, is_softmax=True, **kw):
    """Replace attention and add MyQuan to norms and MLP."""
    attn = child.attn
    qattn = QAttention(
        dim=attn.num_heads * attn.head_dim,
        num_heads=attn.num_heads,
        level=level,
        is_softmax=is_softmax,
    )
    qattn.qkv = attn.qkv
    qattn.attn_drop = attn.attn_drop
    qattn.proj = attn.proj
    qattn.proj_drop = attn.proj_drop
    parent._modules[name].attn = qattn
    parent._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True))
    parent._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True))
    parent._modules[name].mlp.act = nn.Sequential(
        MyQuan(level, sym=False), child.mlp.act
    )
    parent._modules[name].mlp.fc2 = nn.Sequential(
        child.mlp.fc2, MyQuan(level, sym=True)
    )


def _match_conv2d(name, child, parent):
    return isinstance(child, nn.Conv2d) and not isinstance(child, QuanConv2d)


def _apply_conv2d(name, child, parent, level, **kw):
    parent._modules[name] = nn.Sequential(child, MyQuan(level, sym=True))


def _match_layernorm(name, child, parent):
    return isinstance(child, nn.LayerNorm)


def _apply_layernorm(name, child, parent, level, **kw):
    parent._modules[name] = nn.Sequential(child, MyQuan(level, sym=True))


DEFAULT_LSQ_RULES = [
    QuantPlacementRule("transformer_block", _match_transformer_block, _apply_transformer_block),
    QuantPlacementRule("conv2d", _match_conv2d, _apply_conv2d),
    QuantPlacementRule("layernorm", _match_layernorm, _apply_layernorm),
]


# ---------------------------------------------------------------------------
# LSQ Quantizer
# ---------------------------------------------------------------------------

@QUANTIZER_REGISTRY.register("lsq")
class LSQQuantizer(BaseQuantizer):
    """Learned Step-size Quantizer with rule-based placement.

    Args:
        level: Quantization level.
        weight_bit: Weight quantization bit-width (32 = no weight quant).
        is_softmax: Whether attention uses softmax.
        rules: List of QuantPlacementRule (defaults to DEFAULT_LSQ_RULES).
    """

    def __init__(self, level=16, weight_bit=32, is_softmax=True, rules=None):
        self.level = level
        self.weight_bit = weight_bit
        self.is_softmax = is_softmax
        self.rules = rules or DEFAULT_LSQ_RULES

    def quantize_model(self, model):
        """Apply LSQ quantization to model in-place."""
        self._apply_rules(
            model, self.rules,
            level=self.level,
            is_softmax=self.is_softmax,
        )
        if self.weight_bit < 32:
            self._weight_quantization(model, self.weight_bit)
        return model

    def _weight_quantization(self, model, weight_bit):
        """Recursively quantize Conv2d/Linear weights."""
        children = list(model.named_children())
        for name, child in children:
            if type(child) == nn.Conv2d:
                model._modules[name] = QuanConv2d(
                    m=child, quan_w_fn=MyQuan(level=2 ** weight_bit, sym=True)
                )
            elif type(child) == nn.Linear:
                model._modules[name] = QuanLinear(
                    m=child, quan_w_fn=MyQuan(level=2 ** weight_bit, sym=True)
                )
            else:
                self._weight_quantization(child, weight_bit)
