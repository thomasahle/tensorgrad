"""A 6-layer residual-MLP regression step: tensorgrad vs torch.

Model: x (batch=256, d=128); 6 blocks of h = h + tanh(h @ W_i + b_i); a 1-D
readout w_out to one scalar prediction per example; MSE against fixed random
targets. AdamW is written exactly as in examples/mingpt.py -- four lines of
elementwise algebra, bias-corrected via the 0-dim inputs c1/c2 -- so the
whole training step (forward, all 13 gradients, optimizer) compiles into ONE
program: weights and moments in, loss and new weights and moments out.

The torch reference is the same model as an nn.Module, autograd backward and
torch.optim.AdamW with matching hyperparameters (decay on 2-D weights only,
mirroring the `w.order >= 2` rule below).

Stresses: deep shared chains (the residual spine threads all 6 blocks), the
residual-DAG gradient structure (each cotangent both flows through a block
and skips it), and mm-bound compute.

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

torch.set_grad_enabled(False)  # tensorgrad derives gradients symbolically
torch.set_num_threads(2)

BENCH_NAME = "mlp_step"

BATCH, D, N_BLOCKS = 256, 128, 6
LR, B1, B2, WD, EPS = 1e-3, 0.9, 0.95, 0.1, 1e-8
N_CHECK_STEPS, RTOL = 3, 1e-4

batch, d = symbols("batch d")
DIMS = {batch: BATCH, d: D}

# ---------------------------------------------------------- tensorgrad model
params: dict[str, Variable] = {}


def param(name: str, **edges) -> Variable:
    return params.setdefault(name, Variable(name, **edges))


x_var = Variable("x", batch, d)
y_var = Variable("y", batch)

h: Tensor = x_var
for i in range(N_BLOCKS):
    # W_i consumes the 'din' edge and emits a fresh 'd' edge (both sized d),
    # so the residual add is plain +; no axis bookkeeping anywhere.
    h = h + F.tanh(h.rename(d="din") @ param(f"w{i}", din=d, d=d) + param(f"b{i}", d=d))
pred = h @ param("w_out", d=d)  # (batch,): one scalar prediction per example
loss = F.sum((pred - y_var) * (pred - y_var)) / BATCH  # mean squared error

# ---------------------------------------------------------- adamw as algebra
c1, c2 = Variable("c1"), Variable("c2")  # 1/(1-B1^t), 1/(1-B2^t)
moments = {n: (Variable(f"m.{n}", p.shape), Variable(f"v.{n}", p.shape)) for n, p in params.items()}


@typed
def adamw(w: Tensor, g: Tensor, m: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """One AdamW step, written exactly as the update equations -> (w', m', v')."""
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    decay = 1 - LR * WD if w.order >= 2 else 1  # no decay on biases or the 1-D readout
    return w * decay - LR * (c1 * m) / (F.sqrt(c2 * v) + EPS), m, v


_TG_PROGRAM = None  # cached CompiledStep (compile once, share across step makers)


def _tg_program():
    """Loss + 13 gradients + AdamW, fused into one compiled program (cached)."""
    global _TG_PROGRAM
    if _TG_PROGRAM is None:
        grads = tg.grad(loss, params)
        updates = {}
        for n, p in params.items():
            mv, vv = moments[n]
            updates[p], updates[mv], updates[vv] = adamw(p, grads[n], mv, vv)
        _TG_PROGRAM = tg.compile(loss=loss, state=updates)
    return _TG_PROGRAM


# ------------------------------------------------------------ torch reference
class ResidualMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(D, D) for _ in range(N_BLOCKS))
        self.w_out = torch.nn.Parameter(torch.randn(D) / math.sqrt(D))

    def forward(self, x):
        h = x
        for lin in self.blocks:
            h = h + torch.tanh(lin(h))
        return h @ self.w_out


def _fresh_torch_model() -> ResidualMLP:
    torch.manual_seed(0)
    return ResidualMLP()


def _make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(BATCH, D, generator=g), torch.randn(BATCH, generator=g)


def _tg_init_state() -> dict:
    """Copy the torch init into the tensorgrad state (same seed as the torch
    step), plus zeroed moment estimates. nn.Linear stores (out, in), so the
    tensorgrad (din, d) weight is its transpose."""
    model = _fresh_torch_model()
    state = {}
    for i, lin in enumerate(model.blocks):
        state[params[f"w{i}"]] = lin.weight.detach().T.clone()
        state[params[f"b{i}"]] = lin.bias.detach().clone()
    state[params["w_out"]] = model.w_out.detach().clone()
    state |= {mv: torch.zeros_like(state[params[n]]) for n in params for mv in moments[n]}
    return state


# --------------------------------------------------------------- step makers
def make_tg_step():
    """One compiled call per step: (weights, moments, x, y, c1, c2) in,
    (loss, new weights, new moments) out. The loop is a dict reassignment."""
    step = _tg_program()
    xv, yv = _make_data()
    holder = {"state": _tg_init_state(), "t": 0}

    def step_fn() -> float:
        holder["t"] += 1
        t = holder["t"]
        out = step(holder["state"], dims=DIMS, x=xv, y=yv, c1=1 / (1 - B1**t), c2=1 / (1 - B2**t))
        holder["state"] = out.state
        return out.loss.item()

    return step_fn


def make_torch_step():
    model = _fresh_torch_model()
    xv, yv = _make_data()
    decay = [lin.weight for lin in model.blocks]
    no_decay = [lin.bias for lin in model.blocks] + [model.w_out]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}],
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
    print(f"{BENCH_NAME}: tg {t_tg:.2f}ms torch {t_torch:.2f}ms")


def make_jax_step(seed=0):
    """Architecture parity with the torch reference: 6 residual tanh blocks,
    1-D readout, MSE loss, hand-written decoupled AdamW (decay on dim>=2).
    Random init (timing, not correctness). jit end-to-end."""
    import jax
    import jax.numpy as jnp

    g = torch.Generator().manual_seed(0)
    P = {}
    for i in range(N_BLOCKS):
        P[f"w{i}"] = jnp.asarray((torch.randn(D, D, generator=g) / math.sqrt(D)).numpy())
        P[f"b{i}"] = jnp.zeros(D)
    P["w_out"] = jnp.asarray((torch.randn(D, generator=g) / math.sqrt(D)).numpy())
    xv, yv = _make_data()
    xj, yj = jnp.asarray(xv.numpy()), jnp.asarray(yv.numpy())

    def loss_fn(P):
        h = xj
        for i in range(N_BLOCKS):
            h = h + jnp.tanh(h @ P[f"w{i}"] + P[f"b{i}"])
        return ((h @ P["w_out"] - yj) ** 2).mean()

    @jax.jit
    def update(P, m, v, t):
        lv, gr = jax.value_and_grad(loss_fn)(P)
        m = jax.tree.map(lambda a, b: B1 * a + (1 - B1) * b, m, gr)
        v = jax.tree.map(lambda a, b: B2 * a + (1 - B2) * b * b, v, gr)
        c1, c2 = 1 / (1 - B1**t), 1 / (1 - B2**t)

        def upd(w, mm, vv):
            decay = 1 - LR * WD if w.ndim >= 2 else 1.0
            return w * decay - LR * (c1 * mm) / (jnp.sqrt(c2 * vv) + EPS)

        return jax.tree.map(upd, P, m, v), m, v, lv

    st = {"P": P, "m": jax.tree.map(jnp.zeros_like, P),
          "v": jax.tree.map(jnp.zeros_like, P), "t": 0}

    def step_fn() -> float:
        st["t"] += 1
        st["P"], st["m"], st["v"], lv = update(st["P"], st["m"], st["v"], st["t"])
        return float(lv)

    return step_fn
