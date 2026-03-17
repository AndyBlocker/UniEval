"""Spiking operators shared between Qwen3 and UniAffine decoder models.

Spiking_RMSNorm: For Qwen3 baseline (accumulate -> RMSNorm -> differential).
Spiking_SiLU: For Qwen3 SwiGLU gate branch (accumulate -> silu -> differential).
Spiking_SwiGLUMlp: For Qwen3 SwiGLU with temporal multi1() decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SNNOperator
from .neurons import _sequential_multistep


class Spiking_RMSNorm(nn.Module, SNNOperator):
    """Spiking RMSNorm: accumulate input, apply RMSNorm, output differential.

    Follows the Spiking_LayerNorm pattern:
        X_acc += input
        Y = RMSNorm(X_acc)
        output = Y - Y_pre

    Args:
        rmsnorm: The original RMSNorm module.
    """

    participates_in_early_stop = False

    def __init__(self, rmsnorm):
        super().__init__()
        self.rmsnorm = rmsnorm
        self.X = 0.0
        self.Y_pre = None

    def reset(self):
        self.X = 0.0
        self.Y_pre = None

    def forward(self, input):
        self.X = self.X + input
        Y = self.rmsnorm(self.X)
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

    def forward_multistep(self, x_seq):
        """Vectorized: cumsum + rmsnorm + diff."""
        X_cum = x_seq.cumsum(dim=0) + self.X
        Y = self.rmsnorm(X_cum)
        if self.Y_pre is not None:
            Y_prev = self.Y_pre.detach().clone().unsqueeze(0)
        else:
            Y_prev = torch.zeros_like(Y[:1])
        Y_shifted = torch.cat([Y_prev, Y[:-1]], dim=0)
        output = Y - Y_shifted
        self.X = X_cum[-1]
        self.Y_pre = Y[-1]
        return output


class Spiking_SiLU(nn.Module, SNNOperator):
    """Spiking SiLU: accumulate input, apply SiLU, output differential.

    Used for the SwiGLU gate branch in Qwen3 baseline.

    Pattern:
        X_acc += input
        Y = silu(X_acc)
        output = Y - Y_pre
    """

    participates_in_early_stop = False

    def __init__(self):
        super().__init__()
        self.X = 0.0
        self.Y_pre = None

    def reset(self):
        self.X = 0.0
        self.Y_pre = None

    def forward(self, input):
        self.X = self.X + input
        Y = F.silu(self.X)
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

    def forward_multistep(self, x_seq):
        """Vectorized: cumsum + silu + diff."""
        X_cum = x_seq.cumsum(dim=0) + self.X
        Y = F.silu(X_cum)
        if self.Y_pre is not None:
            Y_prev = self.Y_pre.detach().clone().unsqueeze(0)
        else:
            Y_prev = torch.zeros_like(Y[:1])
        Y_shifted = torch.cat([Y_prev, Y[:-1]], dim=0)
        output = Y - Y_shifted
        self.X = X_cum[-1]
        self.Y_pre = Y[-1]
        return output


class Spiking_SwiGLUMlp(nn.Module, SNNOperator):
    """Spiking SwiGLU MLP with temporal multi1() decomposition.

    SwiGLU: out = down_proj(silu(gate_proj(x)) * up_proj(x))

    The SiLU gate branch uses Spiking_SiLU (accumulate -> silu -> diff),
    and the element-wise product uses post-update multi1 decomposition:
        d(G * U)_t = G_acc_t * dU_t + dG_t * U_acc_t - dG_t * dU_t

    Args:
        mlp: The converted SwiGLUMlp (with LLLinear/Spiking_SiLU sub-modules).
    """

    participates_in_early_stop = False

    def __init__(self, mlp):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.act = mlp.act
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.gate_acc = 0.0
        self.up_acc = 0.0

    def reset(self):
        self.gate_acc = 0.0
        self.up_acc = 0.0
        for m in self.modules():
            if m is not self and isinstance(m, SNNOperator):
                m.reset()

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
