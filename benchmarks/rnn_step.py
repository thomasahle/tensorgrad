"""An unrolled Elman-RNN regression step: tensorgrad vs torch.

Model: sequence x (batch=64, seq=16, d=64) consumed step by step through
h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b) with SHARED weights across all 16
steps — the recursive-model stress test: one Wx/Wh pair appears 16 times in
the unrolled graph, so the gradient is a 16-term sum of increasingly deep
chain products (the BPTT structure), and the compiler's sharing (memoized
normalize, per-node cotangent accumulation, CSE) is what keeps compile and
runtime linear in seq instead of quadratic. Readout from the final state to
one scalar per example; MSE against fixed random targets; AdamW exactly as
in mlp_step.py, so the whole training step compiles into ONE program.

The torch reference runs the identical unrolled loop under autograd (no
cuDNN RNN kernel — same math, same shared-weight BPTT).

Importing this file runs the correctness gate: 3 steps from identical
initial weights in both implementations, losses asserted to rtol 1e-4.
"""

import math
import sys
import time

import torch
from sympy import symbols

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable, typed
from tensorgrad.tensor import Tensor

torch.set_grad_enabled(False)
torch.set_num_threads(2)

BENCH_NAME = "rnn_step"

BATCH, SEQ, D = 64, 16, 64
LR, B1, B2, WD, EPS = 1e-3, 0.9, 0.95, 0.1, 1e-8
N_CHECK_STEPS, RTOL = 3, 1e-4

batch, d = symbols("batch d")
DIMS = {batch: BATCH, d: D}

# ---------------------------------------------------------- tensorgrad model
params: dict[str, Variable] = {}


def param(name: str, **edges) -> Variable:
    return params.setdefault(name, Variable(name, **edges))


# One Variable per time step (a named slice of the sequence); the RECURRENCE
# is ordinary Python: h threads through 16 applications of the same weights.
x_vars = [Variable(f"x{t}", batch, d) for t in range(SEQ)]
y_var = Variable("y", batch)

wx = param("wx", din=d, d=d)
wh = param("wh", din=d, d=d)
b = param("b", d=d)

h: Tensor = F.tanh(x_vars[0].rename(d="din") @ wx + b)
for t in range(1, SEQ):
    h = F.tanh(x_vars[t].rename(d="din") @ wx + h.rename(d="din") @ wh + b)
pred = h @ param("w_out", d=d)
loss = F.sum((pred - y_var) * (pred - y_var)) / BATCH

# ---------------------------------------------------------- adamw as algebra
c1, c2 = Variable("c1"), Variable("c2")
moments = {n: (Variable(f"m.{n}", p.shape), Variable(f"v.{n}", p.shape)) for n, p in params.items()}


@typed
def adamw(w: Tensor, g: Tensor, m: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    decay = 1 - LR * WD if w.order >= 2 else 1
    return w * decay - LR * (c1 * m) / (F.sqrt(c2 * v) + EPS), m, v


_TG_PROGRAM = None


def _tg_program():
    global _TG_PROGRAM
    if _TG_PROGRAM is None:
        from tensorgrad.compiler.runtime import grad as tg_grad

        grads = tg_grad(loss, params)
        updates = {}
        for n, p in params.items():
            mv, vv = moments[n]
            updates[p], updates[mv], updates[vv] = adamw(p, grads[n], mv, vv)
        _TG_PROGRAM = tg.compile(loss=loss, state=updates)
    return _TG_PROGRAM


# ------------------------------------------------------------ torch reference
class ElmanRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.wx = torch.nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.wh = torch.nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.b = torch.nn.Parameter(torch.zeros(D))
        self.w_out = torch.nn.Parameter(torch.randn(D) / math.sqrt(D))

    def forward(self, xs):  # (batch, seq, d)
        h = torch.tanh(xs[:, 0] @ self.wx + self.b)
        for t in range(1, SEQ):
            h = torch.tanh(xs[:, t] @ self.wx + h @ self.wh + self.b)
        return h @ self.w_out


def _make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(BATCH, SEQ, D, generator=g), torch.randn(BATCH, generator=g)


def _tg_init_state() -> dict:
    model = ElmanRNN()
    state = {
        params["wx"]: model.wx.detach().clone(),
        params["wh"]: model.wh.detach().clone(),
        params["b"]: model.b.detach().clone(),
        params["w_out"]: model.w_out.detach().clone(),
    }
    state |= {mv: torch.zeros_like(state[params[n]]) for n in params for mv in moments[n]}
    return state


# --------------------------------------------------------------- step makers
def make_tg_step():
    step = _tg_program()
    xv, yv = _make_data()
    xs = {f"x{t}": xv[:, t].clone() for t in range(SEQ)}
    holder = {"state": _tg_init_state(), "t": 0}

    def step_fn() -> float:
        holder["t"] += 1
        t = holder["t"]
        out = step(holder["state"], dims=DIMS, y=yv, c1=1 / (1 - B1**t), c2=1 / (1 - B2**t), **xs)
        holder["state"] = out.state
        return out.loss.item()

    return step_fn


def make_torch_step():
    model = ElmanRNN()
    xv, yv = _make_data()
    opt = torch.optim.AdamW(
        [
            {"params": [model.wx, model.wh], "weight_decay": WD},
            {"params": [model.b, model.w_out], "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(B1, B2),
        eps=EPS,
    )

    def step_fn() -> float:
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss_t = ((model(xv) - yv) ** 2).mean()
            loss_t.backward()
        opt.step()
        return loss_t.item()

    return step_fn


# ------------------------------------------------------------ correctness gate
def _correctness_gate():
    tg_fn, torch_fn = make_tg_step(), make_torch_step()
    for i in range(1, N_CHECK_STEPS + 1):
        l_tg, l_torch = tg_fn(), torch_fn()
        assert math.isclose(l_tg, l_torch, rel_tol=RTOL), (
            f"step {i}: tg loss {l_tg!r} != torch loss {l_torch!r} (rtol {RTOL})"
        )


_correctness_gate()


# ------------------------------------------------------------------- timing
def _bench(step_fn, reps=10, warmup=3) -> float:
    for _ in range(warmup):
        step_fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        step_fn()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


if __name__ == "__main__":
    if "--check" in sys.argv:
        print(f"{BENCH_NAME}: correctness gate passed ({N_CHECK_STEPS} steps, rtol {RTOL})")
        sys.exit(0)
    t_tg = _bench(make_tg_step())
    t_torch = _bench(make_torch_step())
    print(f"{BENCH_NAME}: tg {t_tg:.2f}ms torch {t_torch:.2f}ms ({t_tg / t_torch:.2f}x)")
