"""Post-Training Quantization module: PTQQuan.

Uses min/max calibration with buffer-based step sizes.
"""

import numpy as np
import torch
import torch.nn as nn

from .lsq import threshold_optimization, floor_pass


class PTQQuan(nn.Module):
    """Post-Training Quantization module.

    Uses KL-divergence based threshold calibration on the first batch,
    then uses a fixed (non-learnable) step size stored as a buffer.

    Args:
        level: Number of quantization levels.
        sym: If True, symmetric quantization.
    """

    def __init__(self, level, sym=False):
        super().__init__()
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

        self.register_buffer("s", torch.tensor(1.0))
        self.calibrated = False

    def __repr__(self):
        return (
            f"PTQQuan(level={self.level}, sym={self.sym}, "
            f"pos_max={self.pos_max}, calibrated={self.calibrated})"
        )

    def forward(self, x):
        if self.pos_max == "full":
            return x

        if str(self.neg_min.device) == "cpu":
            self.neg_min = self.neg_min.to(x.device)
        if str(self.pos_max.device) == "cpu":
            self.pos_max = self.pos_max.to(x.device)

        # Calibrate on first forward pass
        if not self.calibrated:
            th = threshold_optimization(
                np.array(x.detach().cpu()),
                quantization_level=int(self.level),
            )
            s_val = th / self.level
            # Guard against zero/near-zero step size (causes div-by-zero)
            if s_val < 1e-10:
                s_val = x.detach().abs().max().item() / self.level
            self.s.fill_(max(s_val, 1e-12))
            self.calibrated = True

        min_val = self.neg_min
        max_val = self.pos_max
        output = torch.clamp(
            floor_pass(x / self.s + 0.5), min=min_val, max=max_val
        ) * self.s
        return output
