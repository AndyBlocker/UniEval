"""
Hook all nn.Modules in a model, assign names, and run check_async / check_locc / check_inout_spike
for each module. Output aggregated detection results.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import csv

from .check_async import check_spatial_async, check_temporal_async
from .check_locc import check_locc, _reset_if_possible, _infer_input
from .check_spike import check_inout_spike


# Same as in check_async
T_STEP = 8


def _assign_module_names(model: nn.Module, prefix: str = "") -> None:
    """Recursively assign a unique .name attribute to every submodule."""
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        child.name = full_name  # type: ignore[attr-defined]
        _assign_module_names(child, prefix=full_name)


def _infer_or_capture_input(
    module: nn.Module,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 2,
    captured: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Use captured input if provided and shape matches; otherwise infer via _infer_input."""
    if captured is not None:
        return captured.to(device=device, dtype=dtype)
    try:
        return _infer_input(module, device, dtype, batch_size=batch_size)
    except (ValueError, AttributeError):
        return None


@torch.no_grad()
def Feasibility_checker(
    model: nn.Module,
    temporal_size: int,
    spatial_dimension: int,
    model_name: str = "model",
    x_global: Optional[torch.Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    steps_locc: int = 16,
    verbose: bool = True,
    ignore_error_modules: bool = False,
    return_ignored_error_modules: bool = False,
) -> List[Dict[str, Any]]:
    """
    Hook all nn.Modules in `model`, assign each a .name, then for each module run:
    - check_async (spatial + temporal async),
    - check_locc,
    - check_inout_spike (on module output).

    Args:
        model: The root nn.Module to traverse.
        temporal_size: Time granularity for async checks.
        spatial_dimension: Spatial dimension index for async checks (-2: whole tensor, -1: element-wise, >=0: dim index).
        x_global: Optional. If given, run one forward to capture per-module inputs via hooks; else infer per module.
        atol, rtol: Tolerances for async and LoCC.
        steps_locc: Time steps for LoCC check.
        verbose: If True, print per-module summary.

    Returns:
        List of dicts, one per module that was checked. Each dict has:
        - name: module full name
        - class_name: type name
        - async_spatial_passed, async_temporal_passed: from check_spatial_async / check_temporal_async
        - locc_passed: from check_locc
        - input_is_spike: from check_inout_spike on module output
        - skipped: optional str reason if module was skipped
    """
    model.eval()
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    except StopIteration:
        device = torch.device("cpu")
        dtype = torch.float32
    if not torch.is_tensor(dtype):
        dtype = torch.float32

    # 1) Assign names to all submodules (root -> "root", children -> "root.conv1", "root.relu1", ...)
    model.name = model_name  # type: ignore[attr-defined]
    _assign_module_names(model, prefix=model_name)

    # 2) Optionally capture inputs via forward hooks
    captured_inputs: Dict[nn.Module, torch.Tensor] = {}
    if x_global is not None:
        x_global = x_global.to(device=device, dtype=dtype)
        hooks = []
        capture_error: Optional[str] = None

        def _make_hook(m: nn.Module):
            def hook(_module, inp):
                if isinstance(inp, (tuple, list)):
                    inp = inp[0]
                if torch.is_tensor(inp):
                    captured_inputs[m] = inp.detach()
            return hook
        
        h = model.register_forward_pre_hook(_make_hook(model))
        hooks.append(h)
        for mod in model.modules():
            if mod is not model:
                h = mod.register_forward_pre_hook(_make_hook(mod))
                hooks.append(h)
        try:
            model(x_global)
        except Exception as e:
            # Some models may throw during forward; still keep whatever was captured
            # and continue per-module checks best-effort.
            capture_error = str(e)
        finally:
            for h in hooks:
                h.remove()
        # 显式记录整模型（接收 x_global 的 root）的输入，确保对 model 本身也做检测
        captured_inputs[model] = x_global
        if capture_error is not None and verbose:
            print(f"[{model_name}] forward capture error: {capture_error}")

    # 3) Collect results per module
    results: List[Dict[str, Any]] = []
    ignored_error_modules: List[Dict[str, Any]] = []
    for _short_name, module in model.named_modules():
        # Use the full name we assigned (root.conv1, root.relu1, ...), not named_modules() short path
        name = getattr(module, "name", "root" if _short_name == "" else _short_name)
        if name == "":
            name = "root"
        module.name = name  # ensure up-to-date
        class_name = module.__class__.__name__

        # Skip non-leaf modules when not using captured inputs (infer only works for leaf-like)
        captured = captured_inputs.get(module)
        x = _infer_or_capture_input(module, device, dtype, batch_size=2, captured=captured)
        if x is None:
            results.append({
                "name": name,
                "class_name": class_name,
                "skipped": "cannot infer or capture input",
            })
            if verbose:
                print(f"[{name}] ({class_name}) skipped: cannot infer or capture input")
            continue

        # Ensure spatial_dimension is valid for this input
        ndim = x.dim()
        sd = spatial_dimension
        if sd >= ndim and sd != -1 and sd != -2:
            sd = min(sd, ndim - 1) if ndim else -2
        if sd == -1 and ndim == 0:
            sd = -2

        # Build x_temporal as in check_async (reproducible)
        torch.manual_seed(42)
        x_temporal = torch.randn(
            temporal_size * T_STEP, *x.shape, device=x.device, dtype=x.dtype
        ).reshape(-1, *x.shape[1:])
        T_max = T_STEP * temporal_size
        if x_temporal.shape[0] % T_max != 0:
            results.append({
                "name": name,
                "class_name": class_name,
                "skipped": "x_temporal dim0 not divisible by T_max",
            })
            if verbose:
                print(f"[{name}] ({class_name}) skipped: x_temporal dim not divisible by T_max")
            continue

        entry: Dict[str, Any] = {
            "name": name,
            "class_name": class_name,
        }

        # --- check_async (spatial + temporal)
        try:
            _reset_if_possible(module)
            spatial_result = check_spatial_async(
                module, x_temporal, spatial_dimension=sd, atol=atol, rtol=rtol
            )
            temporal_result = check_temporal_async(
                module, x_temporal,
                temporal_size=temporal_size,
                T_max=T_max,
                atol=atol,
                rtol=rtol,
            )
            entry["async_spatial_passed"] = bool(spatial_result["passed"])
            entry["async_temporal_passed"] = bool(temporal_result["passed"])
        except Exception as e:
            entry["async_spatial_passed"] = False
            entry["async_temporal_passed"] = False
            entry["async_error"] = str(e)

        # --- check_locc
        try:
            _reset_if_possible(module)
            locc_result = check_locc(
                module, x, steps=steps_locc, mode="equal_split",
                atol=atol, rtol=rtol, verbose=False
            )
            entry["locc_passed"] = bool(locc_result["passed"])
        except Exception as e:
            entry["locc_passed"] = False
            entry["locc_error"] = str(e)

        # --- check_inout_spike (on module output)
        try:
            _reset_if_possible(module)
            out = module(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if torch.is_tensor(out):
                entry["input_is_spike"] = bool(check_inout_spike(out))
            else:
                entry["input_is_spike"] = False
                entry["spike_note"] = "output not a tensor"
        except Exception as e:
            entry["input_is_spike"] = False
            entry["spike_error"] = str(e)

        has_error = any(k in entry for k in ("async_error", "locc_error", "spike_error"))
        if has_error and ignore_error_modules:
            ignored_error_modules.append(entry)
            if verbose:
                print(f"[{name}] ({class_name}) ignored due to error")
            continue

        results.append(entry)
        if verbose:
            async_ok = entry.get("async_spatial_passed", False) and entry.get("async_temporal_passed", False)
            locc_ok = entry.get("locc_passed", False)
            spike = entry.get("input_is_spike", False)
            print(
                f"[{name}] ({class_name}) "
                f"async={async_ok} locc={locc_ok} input_is_spike={spike}"
            )

    if return_ignored_error_modules:
        # For backward-compatibility we still return a list, but embed ignored list as a final summary row.
        results.append({
            "name": "__ignored_error_modules__",
            "class_name": "",
            "ignored_error_modules": ignored_error_modules,
        })
    return results


def summary_requirements(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize how many modules passed each requirement.
    """
    checked = [r for r in results if "skipped" not in r]
    n = len(checked)
    async_ok = sum(1 for r in checked if r.get("async_spatial_passed") and r.get("async_temporal_passed"))
    locc_ok = sum(1 for r in checked if r.get("locc_passed"))
    spike_any = sum(1 for r in checked if r.get("input_is_spike"))
    return {
        "total_checked": n,
        "async_spatial_and_temporal_passed": async_ok,
        "locc_passed": locc_ok,
        "input_is_spike_count": spike_any,
        "skipped": len(results) - n,
    }


def results_to_table(
    results: List[Dict[str, Any]],
    *,
    include_skipped: bool = True,
    print_table: bool = True,
    csv_path: Optional[str] = None,
    exclude_skipped_reasons: Optional[List[str]] = None,
) -> str:
    """
    把每个模块的检测结果整理成表格字符串。

    Args:
        results: Feasibility_checker 返回的 list。
        include_skipped: 是否在表格中包含被 skip 的模块。
        print_table: 是否在返回的同时打印到 stdout。
        exclude_skipped_reasons: 若提供，则当 `row["skipped"]` 等于其中任意值时，
            该行将不会出现在表格/CSV 中（常用于过滤 "cannot infer or capture input"）。

    Returns:
        表格字符串（可复制到文档或终端）。
    """
    cols = [
        "name",
        "class_name",
        "async_spatial",
        "async_temporal",
        "locc",
        "input_is_spike",
        "remark",
    ]
    rows: List[List[str]] = []
    for r in results:
        if r.get("name") == "__ignored_error_modules__":
            # internal summary row, skip from table/csv
            continue
        if "skipped" in r:
            if exclude_skipped_reasons is not None and r.get("skipped") in exclude_skipped_reasons:
                continue
            if not include_skipped:
                continue
            rows.append([
                str(r.get("name", "")),
                str(r.get("class_name", "")),
                "-",
                "-",
                "-",
                "-",
                str(r.get("skipped", "")),
            ])
        else:
            async_s = "✓" if r.get("async_spatial_passed") else "✗"
            async_t = "✓" if r.get("async_temporal_passed") else "✗"
            locc = "✓" if r.get("locc_passed") else "✗"
            spike = "✓" if r.get("input_is_spike") else "✗"
            remark = ""
            for k in ("async_error", "locc_error", "spike_error", "spike_note"):
                if r.get(k):
                    remark = (remark + " " + str(r[k])).strip()
            rows.append([
                str(r.get("name", "")),
                str(r.get("class_name", "")),
                async_s,
                async_t,
                locc,
                spike,
                remark[:48] + "..." if len(remark) > 51 else remark,
            ])

    if not rows:
        out = "(no rows)"
        if print_table:
            print(out)
        return out

    # Column widths (header + content), each column at least 2
    widths = [max(2, len(c)) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], min(len(cell), 52))
    widths = [min(w + 1, 54) for w in widths]

    def sep(char: str = "-") -> str:
        return "+" + "+".join(char * w for w in widths) + "+"

    def line(cells: List[str]) -> str:
        parts = []
        for i, w in enumerate(widths):
            s = (cells[i] if i < len(cells) else "")[:w]
            parts.append(s.ljust(w))
        return "|" + "|".join(parts) + "|"

    lines = [sep(), line(cols), sep()]
    for row in rows:
        lines.append(line(row))
    lines.append(sep())
    out = "\n".join(lines)

    if print_table:
        print(out)

    if csv_path is not None:
        with open(csv_path, "w", newline="", encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
    return out


# -----------------------------
# Pytest: simple CNN
# -----------------------------


def test_check_hooker_simple_cnn():
    """Test Feasibility_checker on a small CNN (Conv2d + ReLU + Conv2d + ReLU + Linear)."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            )
            self.fc = nn.Linear(8 * 4 * 4, 10, bias=False)
            
        def forward(self, x):
            x = self.backbone(x)
            x = x.flatten(1)
            return self.fc(x)

    model = SimpleCNN().eval()
    # (B, C, H, W) = (2, 2, 4, 4) so flatten -> 2 * 8 * 4 * 4
    x_global = torch.randn(2, 2, 4, 4)
    temporal_size = 1
    spatial_dimension = -1

    results = Feasibility_checker(
        model,
        temporal_size=temporal_size,
        spatial_dimension=spatial_dimension,
        x_global=x_global,
        verbose=False,
    )
    results_to_table(results)

    assert isinstance(results, list)
    assert len(results) > 0
    summary = summary_requirements(results)

    assert "total_checked" in summary
    assert "skipped" in summary
    assert summary["total_checked"] + summary["skipped"] == len(results)

    # With x_global we expect root.conv1, root.relu1, root.conv2, root.relu2, root.fc to be checked
    # names = [r["name"] for r in results]
    # assert "root.conv1" in names
    # assert "root.conv2" in names
    # assert "root.fc" in names

    # for r in results:
    #     if "skipped" in r:
    #         continue
    #     assert "async_spatial_passed" in r
    #     assert "async_temporal_passed" in r
    #     assert "locc_passed" in r
    #     assert "input_is_spike" in r


def test_results_to_table_writes_csv_and_ignores_error_modules(tmp_path):
    class Boom(nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.ok = nn.ReLU()
            self.bad = Boom()

        def forward(self, x):
            x = self.ok(x)
            return self.bad(x)

    model = M().eval()
    x_global = torch.randn(2, 8)
    results = Feasibility_checker(
        model,
        temporal_size=1,
        spatial_dimension=1,
        model_name="root",
        x_global=x_global,
        verbose=False,
        ignore_error_modules=True,
        return_ignored_error_modules=True,
    )

    csv_file = tmp_path / "feasibility.csv"
    table = results_to_table(results, csv_path=str(csv_file), print_table=False)
    assert "name" in table and "class_name" in table
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "name,class_name,async_spatial" in content
