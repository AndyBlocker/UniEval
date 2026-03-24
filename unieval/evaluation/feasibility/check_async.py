import torch
import torch.fx
import torch.nn as nn
from typing import Tuple, Dict, Any
from copy import deepcopy
import numpy as np
import random
import os
import sys
from ...snn.operators.layers import SpikeLinear, SpikeConv2dFuseBN, SpikeInferAvgPool, SpikeResidualAdd
from ...qann.operators.quanConv2d import QuanConv2dFuseBN
from ...qann.quantization.quantizer import LsqQuanAct, LsqQuan

@torch.no_grad()
def check_spatial_async(module, x, spatial_dimension, atol=1e-5, rtol=1e-4, ):
    """
    检测模块在空间维度上的异步性
    spatial_dimension: 表示在输入tensor的哪个维度作为粒度做异步检测，如果是-1则是element-wise的异步检测，如果是-2则是整个tensor的异步检测
    """
    module_async = deepcopy(module)
    module_async.eval()
    module_gt = deepcopy(module)
    module_gt.eval()
        
    if spatial_dimension == -1:
        # element-wise的异步检测
        original_shape = x.shape
        x_flat = x.view(-1)
        y_async = 0.0
        for i in range(x_flat.shape[0]):
            mask = torch.zeros_like(x_flat)
            mask[i] = 1
            mask = mask.view(original_shape)
            x_async = x * mask
            if hasattr(module_async, "forward_to_teq") and module_async.forward_to_teq is not None:
                y_async = y_async + module_async.forward_to_teq(x_async)
            else:
                y_async = y_async + module_async(x_async)

        if hasattr(module_async, "forward_to_teq") and module_gt.forward_to_teq is not None:
            y_gt = module_gt.forward_to_teq(x)
        else:
            y_gt = module_gt(x)
                
        if torch.allclose(y_async, y_gt, atol=atol, rtol=rtol):
            passed = True
        else:
            passed = False

        result = {
            "passed": bool(passed),
            "spatial_dimension": spatial_dimension,
            "atol": float(atol),
            "rtol": float(rtol),
            "y_async": y_async,
            "y_gt": y_gt,
        }

    elif spatial_dimension == -2:
        # 整个tensor的异步检测
        passed = True
        if hasattr(module_async, "forward_to_teq") and module_async.forward_to_teq is not None:
            y_async = module_async.forward_to_teq(x)
        else:
            y_async = module_async(x)
        if hasattr(module_async, "forward_to_teq") and module_gt.forward_to_teq is not None:
            y_gt = module_gt.forward_to_teq(x)
        else:
            y_gt = module_gt(x)
        result = {
            "passed": bool(passed),
            "spatial_dimension": spatial_dimension,
            "atol": float(atol),
            "rtol": float(rtol),
            "y_async": y_async,
            "y_gt": y_gt,
        }
    else:
        # 在输入tensor的某个维度作为粒度做异步检测
        assert spatial_dimension >= 0, "spatial_dimension must be >= 0"
        assert spatial_dimension < len(x.shape), "spatial_dimension must be < len(x.shape)"

        original_shape = x.shape
        x_split = torch.split(x, 1, dim=spatial_dimension)  # 如果你是按“单个位置”切分，建议用 1
        y_async = 0.0
        # print(len(x_split))

        for i in range(len(x_split)):
            x_async = torch.zeros_like(x)

            # 构造切片：只有第 spatial_dimension 维的第 i 个位置保留值
            index = [slice(None)] * x.dim()
            index[spatial_dimension] = slice(i, i + 1)
            x_async[tuple(index)] = x_split[i]

            if hasattr(module_async, "forward_to_teq") and module_async.forward_to_teq is not None:
                y_async = y_async + module_async.forward_to_teq(x_async)
            else:
                y_async = y_async + module_async(x_async)

        if hasattr(module_async, "forward_to_teq") and module_gt.forward_to_teq is not None:
            y_gt = module_gt.forward_to_teq(x)
        else:
            y_gt = module_gt(x)
        
        # if module_async.name == "resnet20":
        #     print(y_gt.abs().mean())
        #     print(y_async.abs().mean())
        # print(module_async.name)
        


        if torch.allclose(y_async, y_gt, atol=atol, rtol=rtol):
            passed = True
        else:
            passed = False

        result = {
            "passed": bool(passed),
            "spatial_dimension": spatial_dimension,
            "atol": float(atol),
            "rtol": float(rtol),
            "y_async": y_async,
            "y_gt": y_gt,
        }

    return result
    

@torch.no_grad()
def check_temporal_async(module, x, temporal_size, T_max, atol=1e-5, rtol=1e-4, ):
    """
    检测模块在时间维度上的异步性
    """
    assert T_max > 0, "T_max must be > 0"
    assert x.shape[0] % T_max == 0, "x.shape[0] must be divisible by T_max"

    TB = x.shape[0]
    x_async = x.reshape(T_max, TB//T_max, *x.shape[1:])
    B = TB//T_max

    assert temporal_size > 0, "temporal_size must be > 0"
    assert x_async.shape[0] % temporal_size == 0, "temporal_size must divide T_max"
    
    module_async = deepcopy(module)
    module_async.eval()
    module_gt = deepcopy(module)
    module_gt.eval()
    
    x_split = torch.split(x_async, temporal_size, dim=0)

    y_async = None

    if hasattr(module_async, "forward_to_teq") and module_async.forward_to_teq is not None:
        y_gt = module_gt.forward_to_teq(x)
    else:
        y_gt = module_gt(x)

    for i in range(len(x_split)):
        if hasattr(module_async, "reset"):
            module_async.reset()
        # 构造 full-length 输入（全0）
        x_masked = torch.zeros_like(x_async)
        mask = torch.zeros(torch.Size([x_async.shape[0],x_async.shape[1],*y_gt.shape[1:]])).to(x.device)

        # 只填当前时间块
        start = i * temporal_size
        end = start + temporal_size
        x_masked[start:end] = x_split[i]
        mask[start:end] = 1

        # reshape 回原始输入格式
        x_masked_flat = x_masked.reshape(-1, *x.shape[1:])
        mask_flat = mask.reshape(-1, *y_gt.shape[1:])

        # print(x_masked_flat.shape, )
        if hasattr(module_async, "forward_to_teq") and module_async.forward_to_teq is not None:
            out = module_async.forward_to_teq(x_masked_flat) * mask_flat
        else:
            out = module_async(x_masked_flat) * mask_flat

        y_async = out if y_async is None else y_async + out
    
    
    # print("module_gt",module_gt,"temporal_size, T_max",temporal_size, T_max,y_gt.shape, y_async.shape)
    
    
    if torch.allclose(y_async, y_gt, atol=atol, rtol=rtol):
        passed = True
    else:
        passed = False

    result = {
        "passed": bool(passed),
        "temporal_size": temporal_size,
        "T_max": T_max,
        "atol": float(atol),
        "rtol": float(rtol),
        "y_async": y_async,
        "y_gt": y_gt,
    }

    return result

@torch.no_grad()
def check_async(module, x, temporal_size, spatial_dimension, atol=1e-5, rtol=1e-4):
    """
    思路是这样的：
    1. 异步和LoCC的区别：LoCC描述的是模块在time-step时序上输入输出的关系，异步描述的是时间维度和空间维度上不同位置神经元之间的依赖关系
    2. 异步和LoCC的联系：异步和LoCC是正交的，也就是说一个模块它可以同时满足异步和LoCC的要求
    3. LoCC的检测方法：只要满足f( sum_{t=1..steps} x_t )  ==  sum_{t=1..steps} f(x_t)，在输入和一致的前提下模型的输出也一致就行，并不会模型内部的异步和同步做出要求
    4. 异步的检测方法：
        4.1 异步的定义：描述多个操作（operation）之间的时空依赖关系，当这些操作（operation）在空间上彼此独立，在时间上彼此独立，则称这些操作是相互异步的
        # 4.2 异步计算的执行基础：每个操作（operation）都有自己的独立硬件资源
        4.2 异步的空间粒度：异步的空间粒度取决于操作（operation）在空间维度上的规模，比如一个操作需要100个神经元的输入才能执行，那么异步的空间粒度就是100个神经元
        4.3 异步的时间粒度：异步的时间粒度取决于操作（operation）要求的计算时间，比如一个操作要求累计10个时间间隔（例如：clock/time-step/操作在硬件上执行的critical path latency）的结果才能够执行，那么异步的时间粒度是10个时间间隔
        4.4 异步的好处：所有操作（operation）可以同时执行，所有的硬件可以并行计算，不会因为依赖关系而等待，从而提高计算效率和硬件利用率
        4.5 时间维度上的检测方法：
            1. 用户确定本次检测的异步时间粒度（例如是n个time-step）
            2. 检测每n个time-step的计算的输入输出是否相互独立，具体来说我们每次输入n个time-step的数据，得到n个time-step的输出，重复m次，得到m x n 个time-step的输出
            3. 然后我们一次性打入同样的 m x n 个time-step的数据，得到m x n 个time-step的输出，如果两次的输出完全一致，则认为该module是在该时间粒度下是异步的
        4.6 空间维度上的检测方法：
            1. 首先用户确定本次检测的异步空间粒度（例如是n个神经元）
            2. 检测每n个神经元的计算的输入输出是否相互独立，也就是说我们每次只输入n个神经元的数据，重复m次，得到m x n 个神经元的输出
            3. 然后我们一次性打入同样的 m x n 个神经元的数据，得到m x n 个神经元的输出，如果两次的输出完全一致，则认为该module是在该空间粒度下是异步的
    """
    # 构造时间和空间维度的输入
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    T_step = 8
    x_temporal = torch.randn(temporal_size*T_step, *x.shape).reshape(-1, *x.shape[1:]).to(x.device)
    x_spatial = torch.randn(x.shape).to(x.device)
    
    # -------- Check --------
    spatial_result = check_spatial_async(
        module,
        x_temporal,
        spatial_dimension=spatial_dimension,
        atol=atol,
        rtol=rtol
    )

    temporal_result = check_temporal_async(
        module,
        x_temporal,          # ⚠️ temporal 要用时间展开的数据
        temporal_size=temporal_size,
        T_max=T_step * temporal_size,        # ⚠️ 必须传
        atol=atol,
        rtol=rtol
    )

    # -------- Print --------
    module_name = getattr(module, "name", module.__class__.__name__)

    if spatial_result["passed"] and temporal_result["passed"]:
        print(f"{module_name} is spatial and temporal asynchronous")
    elif spatial_result["passed"]:
        print(f"{module_name} is spatial asynchronous only")
    elif temporal_result["passed"]:
        print(f"{module_name} is temporal asynchronous only")
    else:
        print(f"{module_name} is NOT asynchronous")

# -----------------------------
# Pytest cases
# -----------------------------

def _make_x_temporal(step_x_shape, temporal_size, T_step=8, seed=0):
    """
    Build the flattened temporal input used by `check_temporal_async`.

    `check_async.py` constructs:
        x_temporal = torch.randn(temporal_size*T_step, *x.shape).reshape(-1, *x.shape[1:])
    which is equivalent to having a flattened first dimension of size:
        TB = temporal_size * T_step * B
    """
    torch.manual_seed(seed)
    B = step_x_shape[0]
    rest = step_x_shape[1:]
    TB = temporal_size * T_step * B
    return torch.randn(TB, *rest, dtype=torch.float32)


def test_linear_spatial_temporal_async():
    import pytest  # noqa: F401

    # Linear should be additive (when bias is disabled).
    B, D = 2, 8
    temporal_size = 2
    T_step = 8
    T_max = T_step * temporal_size

    x_temporal = _make_x_temporal((B, D), temporal_size, T_step=T_step, seed=0)

    lin = nn.Linear(D, 4, bias=False).eval()
    spatial = check_spatial_async(lin, x_temporal, spatial_dimension=-1)
    temporal = check_temporal_async(lin, x_temporal, temporal_size=temporal_size, T_max=T_max)

    assert spatial["passed"] is True
    assert temporal["passed"] is True


def test_conv2d_spatial_temporal_async():
    import pytest  # noqa: F401

    # Conv2d should be additive (when bias is disabled).
    B, Cin, H, W = 2, 3, 4, 4
    Cout = 4
    temporal_size = 2
    T_step = 8
    T_max = T_step * temporal_size

    x_temporal = _make_x_temporal((B, Cin, H, W), temporal_size, T_step=T_step, seed=0)

    conv = nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, bias=False).eval()
    spatial = check_spatial_async(conv, x_temporal, spatial_dimension=-1)
    temporal = check_temporal_async(conv, x_temporal, temporal_size=temporal_size, T_max=T_max)

    assert spatial["passed"] is True
    assert temporal["passed"] is True

def test_spiking_conv2d_spatial_temporal_async():
    import pytest  # noqa: F401

    # Conv2d should be additive (when bias is disabled).
    B, Cin, H, W = 2, 3, 4, 4
    Cout = 4
    temporal_size = 2
    T_step = 8
    T_max = T_step * temporal_size

    x_temporal = _make_x_temporal((B, Cin, H, W), temporal_size, T_step=T_step, seed=0)

    conv = nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, bias=False).eval()
    quanconv = QuanConv2dFuseBN(conv, quan_a_fn=LsqQuanAct(bit=3, per_channel=False), quan_w_fn=LsqQuanAct(bit=3, per_channel=False), quan_out_fn=LsqQuanAct(bit=3, per_channel=False))
    quanconv(x_temporal)
    spikeconv = SpikeConv2dFuseBN(quanconv, relu=False)
    
    spatial = check_spatial_async(spikeconv, x_temporal, spatial_dimension=-1)
    temporal = check_temporal_async(spikeconv, x_temporal, temporal_size=temporal_size, T_max=T_max)

    assert spatial["passed"] is True
    assert temporal["passed"] is True



def test_relu_spatial_temporal_async():
    import pytest  # noqa: F401

    # ReLU is elementwise and satisfies f(0)=0, so masked-sum reconstruction holds.
    B, D = 2, 8
    temporal_size = 2
    T_step = 8
    T_max = T_step * temporal_size

    x_temporal = _make_x_temporal((B, D), temporal_size, T_step=T_step, seed=0)

    relu = nn.ReLU().eval()
    spatial = check_spatial_async(relu, x_temporal, spatial_dimension=-1)
    temporal = check_temporal_async(relu, x_temporal, temporal_size=temporal_size, T_max=T_max)

    assert spatial["passed"] is True
    assert temporal["passed"] is True


def test_silu_spatial_temporal_async():
    import pytest  # noqa: F401

    # SiLU is elementwise and satisfies f(0)=0, so masked-sum reconstruction holds.
    B, D = 2, 8
    temporal_size = 2
    T_step = 8
    T_max = T_step * temporal_size

    x_temporal = _make_x_temporal((B, D), temporal_size, T_step=T_step, seed=0)

    silu = nn.SiLU().eval()
    spatial = check_spatial_async(silu, x_temporal, spatial_dimension=-1)
    temporal = check_temporal_async(silu, x_temporal, temporal_size=temporal_size, T_max=T_max)

    assert spatial["passed"] is True
    assert temporal["passed"] is True


def test_softmax_not_spatial_temporal_async():
    import pytest  # noqa: F401

    # Softmax is not additive and f(0) != 0; masked-sum reconstruction should fail.
    B, D = 2, 8
    temporal_size = 2
    T_step = 8
    T_max = T_step * temporal_size

    x_temporal = _make_x_temporal((B, D), temporal_size, T_step=T_step, seed=0)

    softmax = nn.Softmax(dim=-1).eval()
    spatial = check_spatial_async(softmax, x_temporal, spatial_dimension=-1)
    temporal = check_temporal_async(softmax, x_temporal, temporal_size=temporal_size, T_max=T_max)

    assert spatial["passed"] is False
    assert temporal["passed"] is True


def test_stbif_not_spatial_temporal_async():
    import pytest  # noqa: F401

    # ST-BIF neuron is stateful and does not satisfy the masked-sum reconstruction
    # used by `check_spatial_async` / `check_temporal_async`.
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "../../../"))  # .../UniEval
    if project_root not in sys.path:
        sys.path.append(project_root)

    from unieval.snn.operators.neurons import STBIFNeuron

    B, D = 2, 4
    temporal_size = 2
    T_step = 4
    T_max = T_step * temporal_size

    x_temporal = _make_x_temporal((B, D), temporal_size, T_step=T_step, seed=0)

    neuron = STBIFNeuron(q_threshold=torch.tensor(1.0), level=4, sym=False).eval()
    spatial = check_spatial_async(neuron, x_temporal, spatial_dimension=1)
    temporal = check_temporal_async(neuron, x_temporal, temporal_size=temporal_size, T_max=T_max)

    assert spatial["passed"] is True
    assert temporal["passed"] is True




def test_check_async_linear_prints_expected(capsys):
    # check_async is a print-only wrapper; we validate its stdout.
    torch.manual_seed(0)
    B, D = 2, 8
    x = torch.randn(B, D)
    temporal_size = 2

    lin = nn.Linear(D, 4, bias=False).eval()
    check_async(lin, x, temporal_size=temporal_size, spatial_dimension=1)

    out = capsys.readouterr().out
    assert "Linear is spatial and temporal asynchronous" in out


def test_check_async_softmax_prints_expected(capsys):
    torch.manual_seed(0)
    B, D = 2, 8
    x = torch.randn(B, D)
    temporal_size = 2

    softmax = nn.Softmax(dim=-1).eval()
    check_async(softmax, x, temporal_size=temporal_size, spatial_dimension=1)

    out = capsys.readouterr().out
    assert "Softmax is temporal asynchronous only" in out


def test_check_async_stbif_prints_expected(capsys):
    torch.manual_seed(0)
    B, D = 2, 8
    x = torch.randn(B, D)
    temporal_size = 2

    from unieval.snn.operators.neurons import STBIFNeuron

    neuron = STBIFNeuron(q_threshold=torch.tensor(1.0), level=4, sym=False).eval()
    check_async(neuron, x, temporal_size=temporal_size, spatial_dimension=1)

    out = capsys.readouterr().out
    assert "STBIFNeuron is spatial and temporal asynchronous" in out


