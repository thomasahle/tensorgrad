"""Suite runner: one training-step benchmark table, tensorgrad vs torch.

Discovers every module in benchmarks/ that exposes the suite contract
(module-level BENCH_NAME, make_tg_step(), make_torch_step(); the module
self-verifies correctness at import), runs each side SERIALLY -- never
concurrently, one step function at a time -- and prints a markdown table of
min-of-15 step times after 5 warmup steps.

    uv run python benchmarks/run.py                  # eager tg vs eager torch
    uv run python benchmarks/run.py --torch-compile  # tg.compile(torch_compile=True)
                                                     # vs torch.compile'd modules

Under --torch-compile the tg side recompiles every program through inductor
(tensorgrad.compile is rebound before the suite modules are imported, so
import-time tg.compile(...) calls are covered too), and each torch step's
nn.Module(s) -- found in the step closure -- are wrapped with torch.compile;
a purely functional torch step (no module) is compiled whole. Any side that
errors under compilation is skipped gracefully and shown as `skip`.
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
WARMUP, REPS = 5, 15
_MARKERS = ("BENCH_NAME", "def make_tg_step", "def make_torch_step")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ discovery
def discover() -> list[Path]:
    """Suite = every benchmarks/*.py whose SOURCE shows the contract. The
    check is static on purpose: importing a non-suite script here would run
    it (they benchmark at import / top level)."""
    return sorted(
        p
        for p in HERE.glob("*.py")
        if p.name != Path(__file__).name and all(m in p.read_text() for m in _MARKERS)
    )


def load(path: Path):
    """Import a suite module by file path (runs its import-time correctness
    gate). Registered in sys.modules under a bench_ prefix so suite modules
    can never shadow installed packages."""
    name = f"bench_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- torch.compile
def patch_tg_compile() -> None:
    """Rebind tensorgrad.compile so every suite program is built with
    torch_compile=True. Must run BEFORE suite modules are imported (some
    call tg.compile at import time)."""
    import tensorgrad

    orig = tensorgrad.compile

    def compile_with_inductor(torch_compile=False, print_info=False, **outputs):
        return orig(torch_compile=True, print_info=print_info, **outputs)

    tensorgrad.compile = compile_with_inductor


def compile_torch_step(step_fn):
    """Wrap the step's nn.Module(s) in torch.compile. The suite contract
    exposes only step functions, so the modules are found in the closure and
    compiled in place; a functional step with no module is compiled whole."""
    modules = []
    for cell in step_fn.__closure__ or ():
        try:
            obj = cell.cell_contents
        except ValueError:  # empty cell
            continue
        if isinstance(obj, torch.nn.Module):
            modules.append(obj)
    if not modules:
        return torch.compile(step_fn)
    for m in modules:
        m.compile()
    return step_fn


# ----------------------------------------------------------------------- timing
def bench_min(step_fn, warmup=WARMUP, reps=REPS) -> float:
    """min over `reps` individually-timed steps, after `warmup` untimed ones."""
    for _ in range(warmup):
        step_fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        step_fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


def bench_side(make_step, label: str, torch_side: bool, compiled: bool):
    """Build and time one side. Only compiled runs may skip: eager failures
    mean the suite is broken and must surface as a non-zero exit."""
    try:
        step = make_step()
        if compiled and torch_side:
            step = compile_torch_step(step)
        return bench_min(step)
    except Exception as e:  # noqa: BLE001 -- graceful skip under compile only
        if not compiled:
            raise
        _log(f"  note: {label} skipped under torch.compile: {type(e).__name__}: {e}")
        return None


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="tg.compile(torch_compile=True) vs torch.compile'd torch modules",
    )
    args = parser.parse_args()

    load1 = os.getloadavg()[0]
    if load1 > 3:
        print(f"WARNING: 1-min load average is {load1:.2f} (> 3) -- timings may be noisy.")

    if args.torch_compile:
        patch_tg_compile()

    rows = []
    for path in discover():
        _log(f"[{path.stem}] importing (runs its correctness gate) ...")
        try:
            mod = load(path)
        except Exception as e:  # noqa: BLE001 -- import may fail only under compile
            if not args.torch_compile:
                raise
            _log(f"  note: {path.stem} import failed under torch.compile: {type(e).__name__}: {e}")
            rows.append((path.stem, None, None))
            continue
        name = mod.BENCH_NAME
        _log(f"[{name}] timing tg step ...")
        t_tg = bench_side(mod.make_tg_step, f"{name} tg", torch_side=False, compiled=args.torch_compile)
        _log(f"[{name}] timing torch step ...")
        t_th = bench_side(mod.make_torch_step, f"{name} torch", torch_side=True, compiled=args.torch_compile)
        rows.append((name, t_tg, t_th))

    tg_col = "tg ms (torch.compile)" if args.torch_compile else "tg eager ms"
    th_col = "torch ms (compiled)" if args.torch_compile else "torch ms"
    fmt = lambda t: f"{t:.2f}" if t is not None else "skip"  # noqa: E731
    print(f"\n| benchmark | {tg_col} | {th_col} | ratio |")
    print("|---|---:|---:|---:|")
    for name, t_tg, t_th in rows:
        ratio = f"{t_tg / t_th:.2f}x" if t_tg is not None and t_th is not None else "-"
        print(f"| {name} | {fmt(t_tg)} | {fmt(t_th)} | {ratio} |")


if __name__ == "__main__":
    main()
