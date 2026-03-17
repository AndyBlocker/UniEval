"""Spiking operators for UniAffine-specific layers.

Spiking_UnifiedClipNorm: temporal accumulate + uclip + differential output.
Spiking_ReGLUMlp: temporal ReGLU with multi1() decomposition for element-wise product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SNNOperator
from .neurons import _sequential_multistep


class Spiking_UnifiedClipNorm(nn.Module, SNNOperator):
    """Spiking UnifiedClipNorm: accumulate input, apply uclip, output differential.

    Follows the Spiking_LayerNorm pattern:
        X_acc += input
        Y = gamma * clamp(alpha * X_acc + beta, clip_min, clip_max)
        output = Y - Y_pre

    Args:
        uclip: The original UnifiedClipNorm module.
    """

    participates_in_early_stop = False

    def __init__(self, uclip):
        super().__init__()
        self.uclip = uclip
        self.X = 0.0
        self.Y_pre = None

    def reset(self):
        self.X = 0.0
        self.Y_pre = None

    def forward(self, input):
        self.X = self.X + input
        Y = self.uclip(self.X)
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

    def forward_multistep(self, x_seq):
        """Vectorized multi-step: cumsum + uclip + diff."""
        X_cum = x_seq.cumsum(dim=0) + self.X
        Y = self.uclip(X_cum)
        if self.Y_pre is not None:
            Y_prev = self.Y_pre.detach().clone().unsqueeze(0)
        else:
            Y_prev = torch.zeros_like(Y[:1])
        Y_shifted = torch.cat([Y_prev, Y[:-1]], dim=0)
        output = Y - Y_shifted
        self.X = X_cum[-1]
        self.Y_pre = Y[-1]
        return output


class Spiking_ReGLUMlp(nn.Module, SNNOperator):
    """Spiking ReGLU MLP with temporal multi1() decomposition.

    ReGLU: out = down_proj(relu(gate_proj(x)) * up_proj(x))

    The element-wise product of gate and up branches requires temporal
    decomposition. Using post-update accumulators (matching multi1):
        d(G * U)_t = G_acc_t * dU_t + dG_t * U_acc_t - dG_t * dU_t
    where G_acc_t and U_acc_t include the current step's contribution.

    Since ReLU -> Identity after conversion, the gate branch is simply:
        gate_proj (LLLinear) -> [IFNeuron] -> Identity

    Args:
        mlp: The converted ReGLUMlp (with LLLinear/IFNeuron/Identity sub-modules).
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
        """Single timestep forward with temporal product decomposition.

        Uses post-update multi1 formula:
            d(G*U) = G_acc * dU + dG * U_acc - dG * dU
        where G_acc and U_acc include the current step (updated before product).
        """
        # Gate branch: gate_proj -> [IFNeuron ->] ReLU(=Identity)
        dG = self.gate_proj(x)
        dG = self.act(dG)

        # Up branch: up_proj -> [IFNeuron]
        dU = self.up_proj(x)

        # Update accumulators FIRST (post-update for correct multi1 formula)
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
        """Sequential multi-step (inherently sequential due to accumulator state)."""
        return _sequential_multistep(self, x_seq)
