import torch
from copy import deepcopy


def _reset_if_possible(m):
    """Best-effort reset for stateful SNN operators.
    If the module has a callable reset(), call it; otherwise traverse all submodules
    and call reset() on any that have it.
    """
    if hasattr(m, "reset") and callable(getattr(m, "reset")):
        try:
            m.reset()
        except Exception:
            pass
        return
    for sub in m.modules():
        if sub is m:
            continue
        if hasattr(sub, "reset") and callable(getattr(sub, "reset")):
            try:
                sub.reset()
            except Exception:
                pass


def _infer_input(module, device, dtype=torch.float32, batch_size=2):
    """Infer a reasonable random input tensor for common modules."""
    # Linear-like
    if hasattr(module, "in_features"):
        in_features = int(getattr(module, "in_features"))
        return torch.randn(batch_size, in_features, device=device, dtype=dtype)
    # Conv2d-like
    if hasattr(module, "in_channels") and hasattr(module, "kernel_size"):
        in_channels = int(getattr(module, "in_channels"))
        # Keep spatial small but non-trivial
        return torch.randn(batch_size, in_channels, 8, 8, device=device, dtype=dtype)
    raise ValueError(
        f"Cannot infer input shape for module type {type(module).__name__}. "
        f"Please pass x explicitly."
    )


@torch.no_grad()
def check_locc(module, x, steps=16,mode="equal_split", atol=1e-5, rtol=1e-4, verbose=True):
    """
    检查 LoCC（linearity over temporal composition）：
        f( sum_{t=1..steps} x_t )  ?=  sum_{t=1..steps} f(x_t)

    说明：
    - 对 stateful module（例如 SNNOperator / neuron）会在两条路径上分别 reset。
    - 默认用等分方式构造序列：x_t = x / steps（保证 sum x_t = x）。

    Args:
        module: 任意 nn.Module。
        steps: 时间步数。
        x: 可选输入张量；若不提供会尝试根据 module 结构推断 shape 并随机生成。
        mode: "equal_split" 或 "random_split"（保证 sum x_t = x）。
        atol/rtol: 用于通过/失败判断。
        verbose: 是否打印简要统计。

    Returns:
        result: dict，包括误差、是否通过、以及两条路径输出。
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")

    # Snapshot for "restore": we do NOT run forwards on `module` at all.
    # Instead, we create two working copies for the two paths so the caller's
    # module state is preserved (important for stateful SNN operators).
    # Pick device/dtype from module params if possible
    device = x.device if x is not None else torch.device("cpu")
    dtype = x.dtype if x is not None else torch.float32

    x = x.to(device=device, dtype=dtype)

    # Build x_seq with sum(x_seq)=x
    if mode == "equal_split":
        x_seq = [x / steps for _ in range(steps)]
    elif mode == "random_split":
        # random positive weights then normalize, allow sign via x itself
        w = torch.rand(steps, device=device, dtype=dtype)
        w = w / w.sum()
        x_seq = [x * w[t] for t in range(steps)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Use two independent copies to avoid cross-path state leakage.
    mod_sum = deepcopy(module).eval()
    mod_seq = deepcopy(module).eval()

    # Path A: f(sum x_t)
    _reset_if_possible(mod_sum)
    if hasattr(mod_sum, "forward_to_teq"):
        y_a = mod_sum.forward_to_teq(sum(x_seq))
    else:
        y_a = mod_sum(sum(x_seq))
    if isinstance(y_a, (tuple, list)):
        y_a = y_a[0]

    # Path B: sum f(x_t)
    _reset_if_possible(mod_seq)
    y_b = None
    for xt in x_seq:
        if hasattr(mod_seq, "forward_to_teq"):
            yt = mod_seq.forward_to_teq(xt)
        else:
            yt = mod_seq(xt)
        if isinstance(yt, (tuple, list)):
            yt = yt[0]
        y_b = yt if y_b is None else (y_b + yt)

    # Compare
    if not torch.is_tensor(y_a):
        y_a = torch.tensor(y_a, device=x.device)
        
    if not torch.is_tensor(y_b):
        y_b = torch.tensor(y_b, device=x.device)

    
    diff = (y_a - y_b).detach()
    max_abs = float(diff.abs().max().item()) if diff.numel() > 0 else 0.0
    denom = float(y_a.detach().abs().max().item()) if y_a.numel() > 0 else 0.0
    rel = max_abs / (denom + 1e-12)
    passed = (max_abs <= atol + rtol * (denom + 1e-12))

    result = {
        "passed": bool(passed),
        "steps": int(steps),
        "mode": mode,
        "max_abs_diff": max_abs,
        "max_abs_ref": denom,
        "max_rel_diff": rel,
        "atol": float(atol),
        "rtol": float(rtol),
        "y_sum_then_f": y_a,
        "y_sum_f": y_b,
    }

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(
            f"[LoCC] {status} | steps={steps} mode={mode} "
            f"| max_abs_diff={max_abs:.6g} max_rel_diff={rel:.6g} "
            f"| ref_max={denom:.6g}"
        )
    return result


def test_check_locc_linear_equal_split_passes():
    lin = torch.nn.Linear(16, 32, bias=False).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16)
    result = check_locc(lin, x, steps=16, mode="equal_split", verbose=False)
    assert result["passed"], (
        f"LoCC failed: max_abs_diff={result['max_abs_diff']} "
        f"max_rel_diff={result['max_rel_diff']} ref_max={result['max_abs_ref']}"
    )



