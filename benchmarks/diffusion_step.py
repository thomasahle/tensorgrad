"""DDPM-style denoising training step: tensorgrad vs torch.

Model: a small time-conditioned MLP denoiser eps_theta(x_t, t). Fixed
synthetic data x0 (batch=128, dim=64) and a fixed batch of integer timesteps
t in [0, T) (T=100); fresh Gaussian noise eps each step. The noised input is
the standard DDPM forward process

    x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps

where abar is a fixed cosine/linear noise schedule (here linear-beta cumprod)
GATHERED per example by t: F.one_hot(t, T) @ abar. The sqrt coefficients stay
gathered per-example TENSORS -- not python floats -- so the schedule
participates in the graph, and the (batch,) coefficients broadcast over the
(batch, dim) data via plain `*`. A learned time-embedding table is gathered
the same way (F.one_hot(t, T) @ temb) and added to x_t, then 3 residual MLP
blocks (h = h + tanh(h @ W + b)) and a linear readout produce eps_theta. Loss
is the mean-squared error ||eps_theta(x_t, t) - eps||^2.

AdamW is written exactly as in examples/mingpt.py -- four lines of elementwise
algebra, bias-corrected via the 0-dim inputs c1/c2 -- so the whole training
step (forward, all 9 gradients, optimizer) compiles into ONE program: weights
and moments in, loss and new weights and moments out. The torch reference is
the same model as an nn.Module, autograd backward and torch.optim.AdamW with
matching hyperparameters (decay on 2-D weights only, mirroring `w.order >= 2`).

Stresses: gather (schedule + time-embedding lookups -> index_select forward,
scatter-add backward), broadcast of scalar schedule coefficients over the data,
and deep residual gradients + AdamW.

Importing this file runs the correctness gate: 3 steps from identical init and
data in both implementations, losses asserted to rtol 1e-4.
"""

import math
import time

import torch
from sympy import symbols

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable, typed
from tensorgrad.tensor import Tensor

torch.set_grad_enabled(False)  # tensorgrad derives gradients symbolically
torch.set_num_threads(2)

BENCH_NAME = "diffusion_step"

BATCH, DIM, T, N_BLOCKS = 128, 64, 100, 3
LR, B1, B2, WD, EPS = 1e-3, 0.9, 0.95, 0.1, 1e-8
N_CHECK_STEPS, RTOL = 3, 1e-4
INIT_SEED, DATA_SEED, NOISE_SEED = 0, 1, 2

batch, d, time_ = symbols("batch d time")
DIMS = {batch: BATCH, d: DIM, time_: T}


# ------------------------ fixed diffusion schedule -------------------------
def _make_abar() -> torch.Tensor:
    """Linear-beta DDPM schedule: abar[t] = prod_{s<=t} (1 - beta_s), a length-T
    vector in (0, 1). Gathered by t inside the graph, never a python float."""
    betas = torch.linspace(1e-4, 0.02, T)
    return torch.cumprod(1.0 - betas, dim=0)


ABAR = _make_abar()


# --------------------------------------------------------- tensorgrad model
params: dict[str, Variable] = {}


def param(name: str, **edges) -> Variable:
    return params.setdefault(name, Variable(name, **edges))


x0_var = Variable("x0", batch, d)  # fixed synthetic data
eps_var = Variable("eps", batch, d)  # fresh Gaussian noise each step
t_var = Variable("t", batch)  # integer timesteps carried as floats
abar_var = Variable("abar", time_)  # the schedule vector, gathered by t

# The one-hot indicator of t is the whole gather family: contracted with a
# table it is index_select, and its derived gradient is a scatter-add. Here it
# gathers BOTH the schedule coefficients and the learned time embedding.
onehot_t = F.one_hot(t_var, time_)  # (batch, time)

# Schedule coefficients, kept as gathered per-example tensors (not floats), so
# the schedule is part of the graph. The (batch,) coefficients broadcast over
# the (batch, dim) data through plain elementwise `*`.
abar_t = onehot_t @ abar_var  # (batch,)
sqrt_ab = F.sqrt(abar_t)  # (batch,)  sqrt(abar_t)
sqrt_1m = F.sqrt(1 - abar_t)  # (batch,)  sqrt(1 - abar_t)
x_t = sqrt_ab * x0_var + sqrt_1m * eps_var  # (batch, dim): the DDPM forward process

# Time conditioning: a learned embedding table gathered by t, added to x_t.
h: Tensor = x_t + onehot_t @ param("temb", time=time_, d=d)
for i in range(N_BLOCKS):
    # W_i consumes the 'din' edge and emits a fresh 'd' edge (both sized d),
    # so the residual add is plain +; no axis bookkeeping anywhere.
    h = h + F.tanh(h.rename(d="din") @ param(f"w{i}", din=d, d=d) + param(f"b{i}", d=d))
pred = h.rename(d="din") @ param("w_out", din=d, d=d) + param("b_out", d=d)  # eps_theta
loss = F.sum((pred - eps_var) * (pred - eps_var)) / (BATCH * DIM)  # MSE to the noise


# ---------------------------------------------------------- adamw as algebra
c1, c2 = Variable("c1"), Variable("c2")  # 1/(1-B1^t), 1/(1-B2^t)
moments = {n: (Variable(f"m.{n}", p.shape), Variable(f"v.{n}", p.shape)) for n, p in params.items()}


@typed
def adamw(w: Tensor, g: Tensor, m: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """One AdamW step, written exactly as the update equations -> (w', m', v')."""
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    decay = 1 - LR * WD if w.order >= 2 else 1  # decay tables/weights, not biases
    return w * decay - LR * (c1 * m) / (F.sqrt(c2 * v) + EPS), m, v


_TG_PROGRAM = None  # cached CompiledStep (compile once, share across step makers)


def _tg_program():
    """Loss + all 9 gradients + AdamW, fused into one compiled program (cached)."""
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
class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temb = torch.nn.Parameter(0.02 * torch.randn(T, DIM))  # time-embedding table
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(DIM, DIM) for _ in range(N_BLOCKS))
        self.readout = torch.nn.Linear(DIM, DIM)

    def forward(self, x0, eps, t_long, abar):
        abar_t = abar[t_long]  # (batch,) gather
        x_t = abar_t.sqrt()[:, None] * x0 + (1 - abar_t).sqrt()[:, None] * eps
        h = x_t + self.temb[t_long]  # time-embedding gather + add
        for lin in self.blocks:
            h = h + torch.tanh(lin(h))
        return self.readout(h)


def _fresh_torch_model() -> Denoiser:
    torch.manual_seed(INIT_SEED)
    return Denoiser()


def _fixed_data():
    """x0 and the integer timesteps t are drawn ONCE (seeded) and shared by
    both implementations; only eps is fresh per step."""
    g = torch.Generator().manual_seed(DATA_SEED)
    x0 = torch.randn(BATCH, DIM, generator=g)
    t = torch.randint(T, (BATCH,), generator=g)
    return x0, t


def _tg_init_state() -> dict:
    """Copy the torch init into the tensorgrad state (same seed as the torch
    step), plus zeroed moment estimates. nn.Linear stores (out, in), so the
    tensorgrad (din, d) weight is its transpose; the (T, dim) embedding table
    copies directly."""
    model = _fresh_torch_model()
    state = {params["temb"]: model.temb.detach().clone()}
    for i, lin in enumerate(model.blocks):
        state[params[f"w{i}"]] = lin.weight.detach().T.clone()
        state[params[f"b{i}"]] = lin.bias.detach().clone()
    state[params["w_out"]] = model.readout.weight.detach().T.clone()
    state[params["b_out"]] = model.readout.bias.detach().clone()
    state |= {mv: torch.zeros_like(state[params[n]]) for n in params for mv in moments[n]}
    return state


# --------------------------------------------------------------- step makers
def make_tg_step():
    """One compiled call per step: (weights, moments, x0, eps, t, abar, c1, c2)
    in, (loss, new weights, new moments) out. The loop is a dict reassignment."""
    step = _tg_program()
    x0, t = _fixed_data()
    tv = t.float()  # integer ids carried as floats, like argmax output
    gen = torch.Generator().manual_seed(NOISE_SEED)
    holder = {"state": _tg_init_state(), "n": 0}

    def step_fn() -> float:
        holder["n"] += 1
        n = holder["n"]
        eps = torch.randn(BATCH, DIM, generator=gen)
        out = step(
            holder["state"], dims=DIMS, x0=x0, eps=eps, t=tv, abar=ABAR,
            c1=1 / (1 - B1**n), c2=1 / (1 - B2**n),
        )
        holder["state"] = out.state
        return out.loss.item()

    return step_fn


def make_torch_step():
    model = _fresh_torch_model()
    x0, t = _fixed_data()
    decay = [model.temb] + [lin.weight for lin in model.blocks] + [model.readout.weight]
    no_decay = [lin.bias for lin in model.blocks] + [model.readout.bias]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
        betas=(B1, B2),
        eps=EPS,
    )
    gen = torch.Generator().manual_seed(NOISE_SEED)

    def step_fn() -> float:
        eps = torch.randn(BATCH, DIM, generator=gen)
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            pred = model(x0, eps, t, ABAR)
            loss_t = ((pred - eps) ** 2).mean()
            loss_t.backward()
        opt.step()
        return loss_t.item()

    return step_fn


# ------------------------------------------------------------ correctness gate
def _correctness_gate():
    """Same init, same data: per-step losses must agree between the fused
    tensorgrad program and the torch autograd + optim.AdamW reference."""
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
    t_tg = _bench(make_tg_step())
    t_torch = _bench(make_torch_step())
    print(f"{BENCH_NAME}: tg {t_tg:.1f}ms torch {t_torch:.1f}ms")


def make_jax_step(seed=NOISE_SEED):
    """Architecture parity: (T,dim) time-embedding table, 3 residual tanh
    blocks, linear readout, forward-corruption in-graph, MSE-to-noise loss,
    decoupled AdamW (decay on dim>=2). Random init (timing). jit end-to-end."""
    import jax
    import jax.numpy as jnp

    g = torch.Generator().manual_seed(INIT_SEED)
    P = {"temb": jnp.asarray((0.02 * torch.randn(T, DIM, generator=g)).numpy())}
    for i in range(N_BLOCKS):
        P[f"w{i}"] = jnp.asarray((torch.randn(DIM, DIM, generator=g) / math.sqrt(DIM)).numpy())
        P[f"b{i}"] = jnp.zeros(DIM)
    P["w_out"] = jnp.asarray((torch.randn(DIM, DIM, generator=g) / math.sqrt(DIM)).numpy())
    P["b_out"] = jnp.zeros(DIM)

    x0, t = _fixed_data()
    x0j, tj, abarj = jnp.asarray(x0.numpy()), jnp.asarray(t.numpy()), jnp.asarray(ABAR.numpy())

    def loss_fn(P, eps):
        abar_t = abarj[tj]
        x_t = jnp.sqrt(abar_t)[:, None] * x0j + jnp.sqrt(1 - abar_t)[:, None] * eps
        h = x_t + P["temb"][tj]
        for i in range(N_BLOCKS):
            h = h + jnp.tanh(h @ P[f"w{i}"] + P[f"b{i}"])
        return ((h @ P["w_out"] + P["b_out"] - eps) ** 2).mean()

    @jax.jit
    def update(P, m, v, t_, eps):
        lv, gr = jax.value_and_grad(loss_fn)(P, eps)
        m = jax.tree.map(lambda a, b: B1 * a + (1 - B1) * b, m, gr)
        v = jax.tree.map(lambda a, b: B2 * a + (1 - B2) * b * b, v, gr)
        c1, c2 = 1 / (1 - B1**t_), 1 / (1 - B2**t_)

        def upd(w, mm, vv):
            decay = 1 - LR * WD if w.ndim >= 2 else 1.0
            return w * decay - LR * (c1 * mm) / (jnp.sqrt(c2 * vv) + EPS)

        return jax.tree.map(upd, P, m, v), m, v, lv

    st = {"P": P, "m": jax.tree.map(jnp.zeros_like, P),
          "v": jax.tree.map(jnp.zeros_like, P), "t": 0}
    gen = torch.Generator().manual_seed(NOISE_SEED)

    def step_fn() -> float:
        eps = torch.randn(BATCH, DIM, generator=gen)
        st["t"] += 1
        st["P"], st["m"], st["v"], lv = update(
            st["P"], st["m"], st["v"], st["t"], jnp.asarray(eps.numpy()))
        return float(lv)

    return step_fn
