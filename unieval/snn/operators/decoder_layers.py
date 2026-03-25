"""Spiking operators shared between Qwen3 and UniAffine decoder models.

Spiking_RMSNorm: For Qwen3 baseline (accumulate -> RMSNorm -> differential).
Spiking_SiLU: For Qwen3 SwiGLU gate branch (accumulate -> silu -> differential).
Spiking_SwiGLUMlp: For Qwen3 SwiGLU with temporal multi1() decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SNNOperator, CompositeSNNModule
from .accumulating_transform import AccumulatingTransform
from .neurons import _sequential_multistep


class Spiking_RMSNorm(AccumulatingTransform):
    """Spiking RMSNorm: accumulate input, apply RMSNorm, output differential.

    Args:
        rmsnorm: The original RMSNorm module.
    """

    def __init__(self, rmsnorm):
        super().__init__()
        self.rmsnorm = rmsnorm
        self._transform_attr = "rmsnorm"


class Spiking_SiLU(AccumulatingTransform):
    """Spiking SiLU: accumulate input, apply SiLU, output differential.

    Used for the SwiGLU gate branch in Qwen3 baseline.
    """

    def __init__(self):
        super().__init__()
        self._transform_fn = F.silu


class Spiking_SwiGLUMlp(CompositeSNNModule):
    """Spiking SwiGLU MLP with temporal multi1() decomposition.

    SwiGLU: out = down_proj(silu(gate_proj(x)) * up_proj(x))

    The SiLU gate branch uses Spiking_SiLU (accumulate -> silu -> diff),
    and the element-wise product uses post-update multi1 decomposition:
        d(G * U)_t = G_acc_t * dU_t + dG_t * U_acc_t - dG_t * dU_t

    Args:
        mlp: The converted SwiGLUMlp (with LLLinear/Spiking_SiLU sub-modules).
    """

    def __init__(self, mlp):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.act = mlp.act
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.gate_acc = 0.0
        self.up_acc = 0.0

    def reset_local_state(self):
        self.gate_acc = 0.0
        self.up_acc = 0.0

    def forward(self, x):
        """Single timestep with temporal product decomposition.

        Uses post-update multi1: G_acc * dU + dG * U_acc - dG * dU
        where accumulators include current step's contribution.
        """
        # Gate branch: gate_proj -> Spiking_SiLU
        dG = self.gate_proj(x)
        dG = self.act(dG)

        # Up branch
        dU = self.up_proj(x)

        # Post-update accumulators (include current step)
        if torch.is_tensor(self.gate_acc):
            self.gate_acc = self.gate_acc + dG
            self.up_acc = self.up_acc + dU
        else:
            self.gate_acc = dG.clone()
            self.up_acc = dU.clone()

        # multi1: G_acc * dU + dG * U_acc - dG * dU
        product = self.gate_acc * dU + dG * self.up_acc - dG * dU

        return self.down_proj(product)

    def forward_multistep(self, x_seq):
        """Sequential (inherently sequential due to accumulator state)."""
        return _sequential_multistep(self, x_seq)
