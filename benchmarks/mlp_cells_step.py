"""mlp_step with the fused F.adamw optimizer cell.

Identical model, data, loss and torch reference to mlp_step.py (loaded by
path -- one source of truth); the only change is the optimizer: the
adamw-as-algebra update is replaced by the multi-output F.adamw cell
(tensorgrad/compiler/cells.py), which consumes each gradient ONCE as a
single operand and emits w'/m'/v' from one fused call. The delta between
the two mlp rows prices the optimizer path in isolation, since everything
else is shared.

Importing runs the gate: the fused step's loss decreases over 20 steps,
and its first-step loss matches mlp_step's tg side at the shared init
(same seeds, same data) to rtol 1e-4.
"""

import importlib.util
import sys
from pathlib import Path

import torch

import tensorgrad as tg
import tensorgrad.functions as F

torch.set_num_threads(2)

BENCH_NAME = "mlp (+cells)"

GATE_STEPS = 20


def _load_mlp_step():
    path = Path(__file__).resolve().parent / "mlp_step.py"
    name = "bench_mlp_step"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_M = _load_mlp_step()


def make_tg_step():
    m = _M
    grads = tg.grad(m.loss, m.params)
    updates = {}
    for nm, p in m.params.items():
        mv, vv = m.moments[nm]
        updates[p], updates[mv], updates[vv] = F.adamw(
            p, grads[nm], mv, vv, m.c1, m.c2,
            beta1=m.B1, beta2=m.B2, lr=m.LR, eps=m.EPS, weight_decay=m.WD,
        )
    step = tg.compile(loss=m.loss, state=updates)
    xv, yv = m._make_data()
    holder = {"state": m._tg_init_state(), "t": 0}

    def step_fn() -> float:
        holder["t"] += 1
        t = holder["t"]
        out = step(holder["state"], dims=m.DIMS, x=xv, y=yv,
                   c1=1 / (1 - m.B1**t), c2=1 / (1 - m.B2**t))
        holder["state"] = out.state
        return out.loss.item()

    return step_fn


def make_torch_step():
    return _M.make_torch_step()


def _gate():
    fused = make_tg_step()
    plain = _M.make_tg_step()
    l_fused = fused()
    l_plain = plain()
    assert abs(l_fused - l_plain) < 1e-4 * max(1.0, abs(l_plain)), (l_fused, l_plain)
    losses = [fused() for _ in range(GATE_STEPS)]
    assert sum(losses[-3:]) < sum(losses[:3]), "fused loss did not decrease"


_gate()
print(f"[{BENCH_NAME}] correctness gate passed", file=sys.stderr)


def make_jax_step(seed=0):
    return _M.make_jax_step(seed)
