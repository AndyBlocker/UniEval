#!/usr/bin/env python3
"""Equivalence tests: Spike* layers vs Quan* layers.

Spike 版本：第一个 time-step 输入 x，后面 time-step 输入 0，对 T 步输出做累加.
Quan 版本：直接输入 x.
对拍条件：sum(spike_outputs over T steps) 应与 quan_output 一致（用于 debug 等效性）.

对应对：
  - SpikeConv2dFuseBN  <-> QuanConv2dFuseBN
  - QuanInferLinear    <-> QuanLinear
  - SpikeResidualAdd   <-> AdditionQuan
  - SpikeInferAvgPool  <-> QuanAvgPool
"""

import os
import sys

import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None:
    pytest.skip("torch not installed", allow_module_level=True)

# Project root only; use unieval.* so qann/snn stay under unieval (avoids ...registry beyond top-level)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Default timesteps for spike accumulation (should be enough for neuron to settle)
T_STEPS = 64
# Tolerance for float comparison
RTOL = 1e-4
ATOL = 1e-5


def _spike_multistep_sum(module, x, zero_like_x, steps=T_STEPS, reset_neuron=True):
    """Run spike module: first step input x, rest steps input 0; return sum of outputs."""
    if reset_neuron and hasattr(module, "neuron") and hasattr(module.neuron, "reset"):
        module.neuron.reset()
    out_sum = None
    for t in range(steps):
        inp = x if t == 0 else zero_like_x
        if hasattr(module, "forward"):
            out_t = module.forward(inp)
        else:
            raise AttributeError("module has no forward")
        if out_sum is None:
            out_sum = out_t.clone()
        else:
            out_sum = out_sum + out_t
    return out_sum


def _spike_residual_multistep_sum(module, x1, x2, zero1, zero2, steps=T_STEPS, reset_neuron=True):
    """Run SpikeResidualAdd: first step (x1,x2), rest (0,0); return sum of outputs."""
    if reset_neuron and hasattr(module, "neuron") and hasattr(module.neuron, "reset"):
        module.neuron.reset()
    out_sum = None
    for t in range(steps):
        a, b = (x1, x2) if t == 0 else (zero1, zero2)
        out_t = module.forward(a, b)
        if out_sum is None:
            out_sum = out_t.clone()
        else:
            out_sum = out_sum + out_t
    return out_sum


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (layers use .cuda())")
class TestSpikeQuanEquivalence:
    """Spike vs Quan layer equivalence (first step x, rest 0; compare sum(spike) vs quan(x))."""

    @pytest.fixture(autouse=True)
    def device(self):
        return torch.device("cuda")

    def test_spike_conv2d_fuse_bn_vs_quan_conv2d_fuse_bn(self, device):
        """SpikeConv2dFuseBN: first step x, rest 0; sum(outputs) == QuanConv2dFuseBN(x)."""
        from unieval.qann.operators import QuanConv2dFuseBN
        from unieval.qann.quantization import LsqQuan, LsqQuanAct
        from unieval.snn.operators.layers import SpikeConv2dFuseBN

        in_ch, out_ch = 4, 8
        k, stride, pad = 3, 1, 1
        conv = torch.nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=True)
        quan_w = LsqQuan(bit=8, all_positive=False, symmetric=False, per_channel=False)
        quan_out = LsqQuanAct(bit=4, all_positive=False, symmetric=False, per_channel=True)
        quan_conv = QuanConv2dFuseBN(conv, quan_w_fn=quan_w, quan_out_fn=quan_out, is_first=False)
        quan_conv = quan_conv.to(device)

        # Init quan by one forward (sets is_init and quan_out scale)
        x = torch.randn(2, in_ch, 2, 2, device=device)
        _ = quan_conv(x)
        out_quan = quan_conv(x)

        spike_conv = SpikeConv2dFuseBN(quan_conv, relu=False)
        zero_x = torch.zeros_like(x, device=device)
        out_spike_sum = _spike_multistep_sum(spike_conv, x, zero_x)

        torch.testing.assert_close(out_spike_sum, out_quan, rtol=RTOL, atol=ATOL)

    def test_spike_linear_vs_quan_linear(self, device):
        """QuanInferLinear: first step x, rest 0; sum(outputs) == QuanLinear(x)."""
        from unieval.qann.operators import QuanLinear
        from unieval.qann.quantization import LsqQuan, LsqQuanAct
        from unieval.snn.operators.layers import SpikeLinear

        in_f, out_f = 16, 32
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        quan_w = LsqQuan(bit=8, all_positive=False, symmetric=False, per_channel=False)
        quan_out = LsqQuanAct(bit=4, all_positive=False, symmetric=False, per_channel=True)
        quan_linear = QuanLinear(linear, quan_w_fn=quan_w, quan_out_fn=quan_out)
        quan_linear = quan_linear.to(device)

        x = torch.randn(2, in_f, device=device)
        _ = quan_linear(x)
        out_quan = quan_linear(x)

        spike_linear = SpikeLinear(quan_linear, directlyOut=False)
        zero_x = torch.zeros_like(x, device=device)
        out_spike_sum = _spike_multistep_sum(spike_linear, x, zero_x)

        torch.testing.assert_close(out_spike_sum, out_quan, rtol=RTOL, atol=ATOL)

    def test_spike_residual_add_vs_addition_quan(self, device):
        """SpikeResidualAdd: first step (x1,x2), rest (0,0); sum(outputs) == AdditionQuan(x1,x2)."""
        from unieval.qann.operators import AdditionQuan
        from unieval.qann.quantization import LsqQuanAct
        from unieval.snn.operators.layers import SpikeResidualAdd

        quan_a = LsqQuanAct(bit=4, all_positive=False, symmetric=False, per_channel=True)
        add_quan = AdditionQuan(quan_a_fn=quan_a)
        add_quan = add_quan.to(device)

        # Init by one forward
        x1 = torch.randn(2, 4, 8, 8, device=device)
        x2 = torch.randn(2, 4, 8, 8, device=device)
        _ = add_quan(x1, x2)
        out_quan = add_quan(x1, x2)

        spike_add = SpikeResidualAdd(add_quan)
        zero = torch.zeros_like(x1, device=device)
        out_spike_sum = _spike_residual_multistep_sum(spike_add, x1, x2, zero, zero)

        torch.testing.assert_close(out_spike_sum, out_quan, rtol=RTOL, atol=ATOL)

    def test_spike_infer_avg_pool_vs_quan_avg_pool(self, device):
        """SpikeInferAvgPool: first step x, rest 0; sum(outputs) == QuanAvgPool(x)."""
        from unieval.qann.operators import QuanAvgPool
        from unieval.qann.quantization import LsqQuanAct
        from unieval.snn.operators.layers import SpikeInferAvgPool

        pool = torch.nn.AdaptiveAvgPool2d(1)
        quan_out = LsqQuanAct(bit=4, all_positive=False, symmetric=False, per_channel=True)
        quan_pool = QuanAvgPool(pool, quan_out_fn=quan_out)
        quan_pool = quan_pool.to(device)

        x = torch.randn(2, 8, 4, 4, device=device)
        _ = quan_pool(x)
        out_quan = quan_pool(x)

        spike_pool = SpikeInferAvgPool(quan_pool)
        zero_x = torch.zeros_like(x, device=device)
        out_spike_sum = _spike_multistep_sum(spike_pool, x, zero_x)

        torch.testing.assert_close(out_spike_sum, out_quan, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
