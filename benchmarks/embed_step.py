"""Matrix-factorization / word2vec-style training step: tensorgrad vs torch.

Two embedding tables U, V (50_000 x 64). Each step draws 4096 (i, j) index
pairs and +-1 labels from a seeded torch.Generator, scores with
dot(U[i], V[j]), and takes the mean softplus(-label * score) loss
(softplus built from primitives: log(1 + exp(z))). Optimizer is plain SGD.

The tensorgrad step is ONE compiled program: loss + both gradients + the
SGD update, returning the new tables. The embedding lookup is the
contraction F.one_hot(idx, rows) @ table, which the compiler maps to
torch.index_select; the derived gradient becomes a zeros().index_add_
scatter — the dense (batch, rows) indicator never exists. This stresses
gather/scatter lowering, tiny-compute dispatch overhead, and
large-parameter small-touch updates (the SGD axpy sweeps all 6.4M
parameters while the batch touches at most 8192 rows).
"""

import time

import torch
from sympy import symbols

torch.set_grad_enabled(False)
torch.set_num_threads(2)

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable

BENCH_NAME = "embed_step"

N_ROWS, D, BATCH = 50_000, 64, 4096
LR = 0.5
INIT_SEED, DATA_SEED = 0, 1

rows, d, batch = symbols("rows d batch")
DIMS = {rows: N_ROWS, d: D, batch: BATCH}

# ------------------------------------------------------------- tensorgrad
U = Variable("U", rows=rows, d=d)
V = Variable("V", rows=rows, d=d)
i_idx = Variable("i", batch)  # integer row ids, carried as floats
j_idx = Variable("j", batch)
label = Variable("label", batch)  # +-1

u = F.one_hot(i_idx, rows) @ U  # (batch, d) — index_select, not a dense indicator
v = F.one_hot(j_idx, rows) @ V
score = F.dot(u, v, dim="d")  # (batch,)
softplus = lambda z: F.log(1 + F.exp(z))  # noqa: E731 — from primitives
loss = F.sum(softplus(-label * score)) / BATCH

grads = tg.grad(loss, {"U": U, "V": V})
updates = {U: U - LR * grads["U"], V: V - LR * grads["V"]}  # SGD, gradients are scatters
STEP = tg.compile(loss=loss, state=updates)


def init_tables(seed=INIT_SEED):
    gen = torch.Generator().manual_seed(seed)
    u0 = 0.02 * torch.randn(N_ROWS, D, generator=gen)
    v0 = 0.02 * torch.randn(N_ROWS, D, generator=gen)
    return u0, v0


def draw_batch(gen):
    iv = torch.randint(N_ROWS, (BATCH,), generator=gen)
    jv = torch.randint(N_ROWS, (BATCH,), generator=gen)
    lab = torch.randint(2, (BATCH,), generator=gen).float() * 2 - 1
    return iv, jv, lab


def make_tg_step(seed=DATA_SEED):
    u0, v0 = init_tables()
    state = {U: u0, V: v0}
    gen = torch.Generator().manual_seed(seed)

    def step_fn():
        nonlocal state
        iv, jv, lab = draw_batch(gen)
        out = STEP(state, dims=DIMS, i=iv.float(), j=jv.float(), label=lab)
        state = out.state
        return out.loss.item()

    return step_fn


# ------------------------------------------------------------------ torch
class Embeds(torch.nn.Module):
    def __init__(self, u0, v0):
        super().__init__()
        self.U = torch.nn.Parameter(u0.clone())
        self.V = torch.nn.Parameter(v0.clone())

    def forward(self, iv, jv, lab):
        s = (self.U[iv] * self.V[jv]).sum(-1)
        return torch.nn.functional.softplus(-lab * s).mean()


def make_torch_step(seed=DATA_SEED):
    u0, v0 = init_tables()
    with torch.enable_grad():
        model = Embeds(u0, v0)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)

    def step_fn():
        iv, jv, lab = draw_batch(gen)
        with torch.enable_grad():
            opt.zero_grad()
            lv = model(iv, jv, lab)
            lv.backward()
            opt.step()
        return lv.item()

    return step_fn


# -------------------------------------------------------- correctness gate
def _correctness_gate(steps=3, rtol=1e-4):
    """Same init, same data: per-step losses must agree between the fused
    tensorgrad program and the torch autograd + optim.SGD reference."""
    tg_step, torch_step = make_tg_step(), make_torch_step()
    for k in range(steps):
        a, b = tg_step(), torch_step()
        assert abs(a - b) <= rtol * abs(b), f"step {k}: tg {a} vs torch {b}"


_correctness_gate()


# ------------------------------------------------------------------ timing
def _bench(step_fn, warmup=3, reps=10):
    for _ in range(warmup):
        step_fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        step_fn()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


if __name__ == "__main__":
    t_tg = _bench(make_tg_step())
    t_torch = _bench(make_torch_step())
    print(f"{BENCH_NAME}: tg {t_tg:.1f}ms torch {t_torch:.1f}ms")


def make_jax_step(seed=DATA_SEED):
    """Architecture parity: same two 50k x 64 tables, dot-product score,
    mean softplus(-label*score) loss, plain SGD. jit end-to-end."""
    import jax
    import jax.numpy as jnp

    u0, v0 = init_tables()
    st = {"U": jnp.asarray(u0.numpy()), "V": jnp.asarray(v0.numpy())}

    def loss_fn(U, V, i, j, lab):
        s = (U[i] * V[j]).sum(-1)
        return jax.nn.softplus(-lab * s).mean()

    @jax.jit
    def update(U, V, i, j, lab):
        lv, (gU, gV) = jax.value_and_grad(loss_fn, argnums=(0, 1))(U, V, i, j, lab)
        return U - LR * gU, V - LR * gV, lv

    gen = torch.Generator().manual_seed(seed)

    def step_fn() -> float:
        iv, jv, lab = draw_batch(gen)
        out = update(st["U"], st["V"], jnp.asarray(iv.numpy()), jnp.asarray(jv.numpy()),
                     jnp.asarray(lab.numpy()))
        st["U"], st["V"], lv = out
        return float(lv)

    return step_fn
