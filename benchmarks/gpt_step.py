"""One minGPT training step: tensorgrad vs torch autograd (correctness + timing).

The tensorgrad side IS examples/mingpt.py: this file loads that module from its
path (importlib.util.spec_from_file_location, no package install) and builds
the same one-program step its main() builds -- forward loss, all 53 parameter
gradients and the AdamW update fused into a single compiled callable, weights
and moments in, loss and new weights and moments out.

The torch reference is a faithful minimal GPT as an nn.Module -- same shape as
gpt-nano (3 layers, n_embd=48, 3 heads, seq 11, vocab 3), same architecture
(learned token + position embeddings, pre-LN blocks, causal softmax attention,
tanh-approximated gelu MLP, final LN, bias-free lm_head), same AdamW
hyperparameters (decay only on order>=2 weights), trained on the same sorting
task (random digits ++ sorted digits, first LENGTH-1 targets ignored).

The two inits are not bit-identical (torch layout vs mingpt's named-edge
layout), so per-step losses are NOT comparable across implementations; the
correctness gate at import instead checks that BOTH implementations' losses
decrease over 20 steps.
"""

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

BENCH_NAME = "gpt_step"

# ------------------------------------------------------- load examples/mingpt.py
_MINGPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "mingpt.py"


def _load_mingpt():
    spec = importlib.util.spec_from_file_location("bench_mingpt", _MINGPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_mingpt"] = mod
    spec.loader.exec_module(mod)  # builds the symbolic loss; main() is not run
    return mod


mingpt = _load_mingpt()

# mingpt.py already set torch.set_grad_enabled(False) and set_num_threads(2).
import tensorgrad as tg  # noqa: E402  (mingpt import must come first)

N_LAYER, N_HEAD, D = mingpt.N_LAYER, mingpt.N_HEAD, mingpt.N_EMBD
VOCAB, LENGTH, SEQ, BATCH = mingpt.VOCAB, mingpt.LENGTH, mingpt.SEQ, mingpt.BATCH
HS, D_MLP = D // N_HEAD, 4 * D
LR, B1, B2, WD, EPS = mingpt.LR, mingpt.B1, mingpt.B2, mingpt.WD, mingpt.EPS
GATE_STEPS = 20

# --------------------------------------------------------------- tensorgrad step
_TG_PROGRAM = None  # cached CompiledStep (compile once, share across step makers)


def _tg_program():
    """Loss + all gradients + AdamW fused into one program, exactly as
    mingpt.main() builds it (cached)."""
    global _TG_PROGRAM
    if _TG_PROGRAM is None:
        grads = tg.grad(mingpt.loss, mingpt.params)
        updates = {}
        for n, p in mingpt.params.items():
            mv, vv = mingpt.moments[n]
            updates[p], updates[mv], updates[vv] = mingpt.adamw(p, grads[n], mv, vv)
        _TG_PROGRAM = tg.compile(loss=mingpt.loss, state=updates)
    return _TG_PROGRAM


def make_tg_step(seed=0):
    """One compiled call per step; the training loop is a dict reassignment."""
    step = _tg_program()
    gen = torch.Generator().manual_seed(seed)
    state = mingpt.init_weights(gen)
    state |= {
        mv: torch.zeros_like(state[mingpt.params[n]]) for n in mingpt.params for mv in mingpt.moments[n]
    }
    holder = {"state": state, "t": 0}

    def step_fn() -> float:
        holder["t"] += 1
        t = holder["t"]
        out = step(
            holder["state"],
            dims=mingpt.DIMS,
            raw=mingpt.random_digits(BATCH, gen),
            c1=1 / (1 - B1**t),
            c2=1 / (1 - B2**t),
        )
        holder["state"] = out.state  # the whole optimizer step
        return out.loss.item()

    return step_fn


# ---------------------------------------------------------------- torch reference
class CausalSelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(D, D)
        self.wk = torch.nn.Linear(D, D)
        self.wv = torch.nn.Linear(D, D)
        self.wo = torch.nn.Linear(D, D)
        self.register_buffer("mask", torch.tril(torch.ones(SEQ, SEQ)).view(1, 1, SEQ, SEQ))

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, N_HEAD, HS).transpose(1, 2)  # (B, nh, T, hs)
        k = self.wk(x).view(B, T, N_HEAD, HS).transpose(1, 2)
        v = self.wv(x).view(B, T, N_HEAD, HS).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(HS)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        y = att.softmax(-1) @ v  # (B, nh, T, hs)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(D)
        self.attn = CausalSelfAttention()
        self.ln2 = torch.nn.LayerNorm(D)
        self.fc = torch.nn.Linear(D, D_MLP)
        self.proj = torch.nn.Linear(D_MLP, D)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = torch.nn.functional.gelu(self.fc(self.ln2(x)), approximate="tanh")  # minGPT's NewGELU
        return x + self.proj(h)


class GPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(VOCAB, D)
        self.wpe = torch.nn.Parameter(torch.zeros(SEQ, D))
        self.blocks = torch.nn.ModuleList(Block() for _ in range(N_LAYER))
        self.ln_f = torch.nn.LayerNorm(D)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)

    def forward(self, idx):  # (B, T) int64 -> (B, T, VOCAB)
        x = self.wte(idx) + self.wpe
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


def _init_torch(model: GPT, seed: int) -> GPT:
    """Same init POLICY as mingpt.init_weights: layer-norm gains 1, biases 0,
    everything else 0.02 * randn (not the same draws -- layouts differ)."""
    gen = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            p.zero_()
        elif ".ln" in name or name.startswith("ln"):  # LayerNorm gains
            p.fill_(1.0)
        else:
            p.copy_(0.02 * torch.randn(p.shape, generator=gen))
    return model


def _torch_batch(gen):
    """Same data pipeline as mingpt, host-side: raw digits ++ sorted digits,
    tokens = full[:SEQ], targets = full[1:SEQ+1]."""
    raw = torch.randint(VOCAB, (BATCH, LENGTH), generator=gen)
    full = torch.cat([raw, raw.sort(dim=1).values], dim=1)
    return full[:, :SEQ], full[:, 1 : SEQ + 1]


def make_torch_step(seed=0):
    model = _init_torch(GPT(), seed)
    decay = [p for p in model.parameters() if p.dim() >= 2]  # mirrors `w.order >= 2`
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
        betas=(B1, B2),
        eps=EPS,
    )
    gen = torch.Generator().manual_seed(seed)

    def step_fn() -> float:
        tokens, targets = _torch_batch(gen)
        with torch.enable_grad():
            logits = model(tokens)
            # minGPT's ignore_index=-1: only the last LENGTH targets count.
            loss = torch.nn.functional.cross_entropy(
                logits[:, LENGTH - 1 :].reshape(-1, VOCAB), targets[:, LENGTH - 1 :].reshape(-1)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
        opt.step()
        return loss.item()

    return step_fn


def make_jax_step(seed=0):
    """Architecture parity with the torch reference (LayerNorm gains+biases,
    qkvo/mlp biases, tanh-GELU, tied dims), hand-written decoupled AdamW with
    the same decay policy (dim >= 2), same host-side data pipeline."""
    import jax
    import jax.numpy as jnp

    gen = torch.Generator().manual_seed(seed)

    def rnd(*sh):
        return jnp.asarray((0.02 * torch.randn(*sh, generator=gen)).numpy())

    P = {"wte": rnd(VOCAB, D), "wpe": jnp.zeros((SEQ, D)), "lm": rnd(D, VOCAB),
         "lnfg": jnp.ones(D), "lnfb": jnp.zeros(D)}
    for i in range(N_LAYER):
        P[f"h{i}.ln1g"], P[f"h{i}.ln1b"] = jnp.ones(D), jnp.zeros(D)
        P[f"h{i}.ln2g"], P[f"h{i}.ln2b"] = jnp.ones(D), jnp.zeros(D)
        for nm, sh in [("wq", (D, D)), ("wk", (D, D)), ("wv", (D, D)), ("wo", (D, D)),
                       ("fc", (D, D_MLP)), ("proj", (D_MLP, D))]:
            P[f"h{i}.{nm}"] = rnd(*sh)
            P[f"h{i}.{nm}b"] = jnp.zeros(sh[1])
    mask = jnp.tril(jnp.ones((SEQ, SEQ)))

    def ln(x, g_, b_):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / jnp.sqrt(var + 1e-5) * g_ + b_

    def fwd(P, idx):
        x = P["wte"][idx] + P["wpe"]
        for i in range(N_LAYER):
            h = ln(x, P[f"h{i}.ln1g"], P[f"h{i}.ln1b"])

            def proj(nm):
                return (h @ P[f"h{i}.{nm}"] + P[f"h{i}.{nm}b"]).reshape(
                    BATCH, SEQ, N_HEAD, HS).transpose(0, 2, 1, 3)

            q, k, v = proj("wq"), proj("wk"), proj("wv")
            att = q @ k.transpose(0, 1, 3, 2) / math.sqrt(HS)
            att = jnp.where(mask == 0, -jnp.inf, att)
            y = jax.nn.softmax(att, -1) @ v
            y = y.transpose(0, 2, 1, 3).reshape(BATCH, SEQ, D)
            x = x + y @ P[f"h{i}.wo"] + P[f"h{i}.wob"]
            h2 = ln(x, P[f"h{i}.ln2g"], P[f"h{i}.ln2b"])
            u = jax.nn.gelu(h2 @ P[f"h{i}.fc"] + P[f"h{i}.fcb"], approximate=True)
            x = x + u @ P[f"h{i}.proj"] + P[f"h{i}.projb"]
        return ln(x, P["lnfg"], P["lnfb"]) @ P["lm"]

    def loss_fn(P, tokens, targets):
        logits = fwd(P, tokens)[:, LENGTH - 1 :, :]
        tgt = targets[:, LENGTH - 1 :]
        logp = jax.nn.log_softmax(logits, -1)
        return -jnp.take_along_axis(logp, tgt[..., None], -1).mean()

    @jax.jit
    def update(P, m, v, t, tokens, targets):
        lv, g_ = jax.value_and_grad(loss_fn)(P, tokens, targets)
        m = jax.tree.map(lambda a, b: B1 * a + (1 - B1) * b, m, g_)
        v = jax.tree.map(lambda a, b: B2 * a + (1 - B2) * b * b, v, g_)
        c1, c2 = 1 / (1 - B1**t), 1 / (1 - B2**t)

        def upd(w, mm, vv):
            decay = 1 - LR * WD if w.ndim >= 2 else 1.0  # mirrors `w.order >= 2`
            return w * decay - LR * (c1 * mm) / (jnp.sqrt(c2 * vv) + EPS)

        return jax.tree.map(upd, P, m, v), m, v, lv

    st = {"P": P, "m": jax.tree.map(jnp.zeros_like, P), "v": jax.tree.map(jnp.zeros_like, P), "t": 0}
    dgen = torch.Generator().manual_seed(seed)

    def step_fn() -> float:
        tokens, targets = _torch_batch(dgen)
        tj, gj = jnp.asarray(tokens.numpy()), jnp.asarray(targets.numpy())
        st["t"] += 1
        st["P"], st["m"], st["v"], lv = update(st["P"], st["m"], st["v"], st["t"], tj, gj)
        return float(lv)

    return step_fn


# ------------------------------------------------------------- correctness gate
def _correctness_gate():
    """Inits differ, so losses are not comparable across implementations; the
    gate is that each one's loss DECREASES over GATE_STEPS steps of training
    (mean of first 3 vs mean of last 3, robust to per-batch noise)."""
    makers = [("tensorgrad", make_tg_step), ("torch", make_torch_step)]
    try:
        import jax  # noqa: F401

        makers.append(("jax", make_jax_step))
    except ImportError:
        pass
    for label, make in makers:
        step = make()
        losses = [step() for _ in range(GATE_STEPS)]
        first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
        assert last < first, (
            f"{label}: loss did not decrease over {GATE_STEPS} steps ({first:.4f} -> {last:.4f})"
        )


_correctness_gate()


# ------------------------------------------------------------------------ timing
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
