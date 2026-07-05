"""minGPT (https://github.com/karpathy/minGPT) as a tensorgrad program.

What does a GPT look like when every tensor axis has a NAME and every gradient
is DERIVED symbolically instead of traced by autograd?

* No .view / .transpose / .reshape anywhere. Karpathy's

      k = k.view(B, T, nh, hs).transpose(1, 2)   # (B, nh, T, hs)

  is here just an edge named 'head' on the projection weights. The only
  axis bookkeeping in the whole model is semantic: attention keys/values get
  their sequence edge renamed 'seq' -> 'key', so scores naturally come out
  with (seq, key) edges.

* No autograd. torch.set_grad_enabled(False) below is global and permanent:
  loss.grad(p) builds a symbolic derivative, and compile_to_callable turns
  the loss plus one gradient per parameter into ONE fused straight-line
  program in which forward work is shared across outputs. PyTorch is used
  purely as a tensor library.

* No hand-written backward rules for composites. layer_norm below is three
  lines of mean/sqrt and its Jacobian is derived; softmax (and the
  log_softmax inside the loss) is fused and numerically stabilized by the
  compiler; gelu's tanh chain is differentiated symbolically too.

* Integer tokens stay integers: F.gather indexes the embedding table
  directly (index_select forward; the derived gradient is a scatter-add).
  No one-hot matmuls.

Running this file trains a small GPT on karpathy's sorting task ("given 6
digits, emit them sorted") to >90% held-out sequence accuracy in about four
minutes on CPU (nearly all of it torch.compile warmup on the ~600-op fused
step program; training itself is under a minute).
"""

import math
import time

import torch
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.tensor import set_lazy_rename

# The punchline: no gradient tape, ever. All gradients in this file are
# symbolic expressions, derived from the loss and compiled ahead of time.
torch.set_grad_enabled(False)
torch.set_num_threads(2)

# One pragmatic wart: deep residual nets reuse subtrees so much that eager
# edge-renaming would blow up exponentially; this makes renames O(1) nodes.
set_lazy_rename(True)

# ----------------------------------------------------------------- config
# One transformer block at gpt-nano width (n_embd=48). Every parameter's
# gradient here — including those under the softmax stack, the gelu and the
# LayerNorms — compiles into reverse-mode (cotangent-first) contraction
# order automatically: the adjoint pass (tensorgrad/compiler/adjoint.py)
# collapses the forward-mode-shaped Jacobian chains that symbolic
# differentiation produces. Before that pass, this exact width was
# infeasible (the mlp.w1 gradient materialized a multi-GB rank-5 Jacobian;
# n_embd=24 was the practical ceiling).
#
# KNOWN LIMITATION (why this is not 3-layer gpt-nano): a gradient that
# crosses SEVERAL stacked blocks needs cotangent contributions accumulated
# across different sums and different gradients. The adjoint pass currently
# re-merges branches only within one Sum, so at multi-block depth the
# branch count grows geometrically; it detects that and falls back to the
# (huge) forward-mode form. The frontier — true per-node adjoint
# accumulation — is documented in adjoint.py.

N_LAYER, N_HEAD, N_EMBD = 1, 3, 48
VOCAB, LENGTH = 3, 6                     # sort 6 digits from {0,1,2}
SEQ = 2 * LENGTH - 1                     # input digits + sorted digits, shifted
BATCH, LR, MAX_STEPS = 64, 1e-3, 400

batch, seq, vocab, d, head, hs, d_mlp = symbols("batch seq vocab d head hs d_mlp")
DIMS = {batch: BATCH, seq: SEQ, vocab: VOCAB, d: N_EMBD,
        head: N_HEAD, hs: N_EMBD // N_HEAD, d_mlp: 4 * N_EMBD}

params: dict[str, Variable] = {}


def param(name: str, **edges) -> Variable:
    return params.setdefault(name, Variable(name, **edges))


# -------------------------------------------------------------- the model
# Inputs. Tokens are integer ids (carried as floats); targets are one-hot
# rows, with all-zero rows where minGPT would use ignore_index=-1. The causal
# mask enters as data: 0 on/below the diagonal, -1e9 above (masked_fill).

tokens = Variable("tokens", batch, seq)
targets = Variable("targets", batch, seq, vocab)
causal_mask = Variable("causal_mask", seq=seq, key=seq)


def layer_norm(x, name):
    """Written from primitives -- tensorgrad DERIVES its backward pass."""
    x = x - F.mean(x, dim="d", keepdims=True)
    var = F.mean(x * x, dim="d", keepdims=True)
    return x / F.sqrt(var + 1e-5) * param(name + ".g", d=d) + param(name + ".b", d=d)


def attention(x, name):
    """Causal multi-head self-attention. 'head' is just an edge on the
    weights, and keys/values live on a renamed sequence edge -- that is the
    entire content of karpathy's four view/transpose lines."""
    q = x @ param(name + ".wq", d=d, head=head, hs=hs) + param(name + ".bq", head=head, hs=hs)
    k = x @ param(name + ".wk", d=d, head=head, hs=hs) + param(name + ".bk", head=head, hs=hs)
    v = x @ param(name + ".wv", d=d, head=head, hs=hs) + param(name + ".bv", head=head, hs=hs)
    k, v = k.rename(seq="key"), v.rename(seq="key")
    att = F.dot(q, k, dim="hs") / math.sqrt(DIMS[hs])   # (batch, head, seq, key)
    att = F.softmax(att + causal_mask, dim="key")       # stabilized by the compiler
    y = F.dot(att, v, dim="key")                        # (batch, head, seq, hs)
    return y @ param(name + ".wo", head=head, hs=hs, d=d) + param(name + ".bo", d=d)


def mlp(x, name):
    h = F.gelu(x @ param(name + ".w1", d=d, d_mlp=d_mlp) + param(name + ".b1", d_mlp=d_mlp),
               approximate="tanh")                      # minGPT's NewGELU, derivative derived
    return h @ param(name + ".w2", d_mlp=d_mlp, d=d) + param(name + ".b2", d=d)


def gpt(idx):
    """Token ids (batch, seq) -> logits (batch, seq, vocab)."""
    wte = param("wte", vocab=vocab, d=d)
    x = F.gather(wte, idx, dim="vocab") + param("wpe", seq=seq, d=d)  # real integer lookup
    for i in range(N_LAYER):
        x = x + attention(layer_norm(x, f"h{i}.ln1"), f"h{i}.attn")
        x = x + mlp(layer_norm(x, f"h{i}.ln2"), f"h{i}.mlp")
    return layer_norm(x, "ln_f") @ param("lm_head", d=d, vocab=vocab)


logits = gpt(tokens)
ce = F.cross_entropy(logits, targets, dim="vocab")      # (batch, seq); 0 at ignored positions
loss = F.sum(ce) / (BATCH * LENGTH)

# ------------------------------------------------------------- data + adamw


def sort_batch(n, gen):
    """x = "digits then their sorted order", y = x shifted left; the first
    LENGTH-1 targets are zeroed one-hot rows (= minGPT's ignore_index=-1)."""
    inp = torch.randint(VOCAB, (n, LENGTH), generator=gen)
    cat = torch.cat([inp, inp.sort(dim=1).values], dim=1)
    tgt = torch.nn.functional.one_hot(cat[:, 1:], VOCAB).float()
    tgt[:, : LENGTH - 1] = 0.0
    return cat[:, :-1].float(), tgt


def init_weights(gen):
    ws = {}
    for name, var in params.items():
        shape = [DIMS[s] for s in var.shape.values()]
        last = name.rsplit(".", 1)[-1]
        ws[name] = (torch.ones(shape) if last == "g"          # layer-norm gains
                    else torch.zeros(shape) if last[0] == "b"  # biases
                    else 0.02 * torch.randn(shape, generator=gen))
    return ws


def adamw_update(ws, opt_m, opt_v, grads, t, b1=0.9, b2=0.95, wd=0.1):
    for name, g in grads.items():
        w, m, v = ws[name], opt_m[name], opt_v[name]
        if w.dim() >= 2:
            w.mul_(1 - LR * wd)
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        w.sub_(LR * (m / (1 - b1**t)) / ((v / (1 - b2**t)).sqrt() + 1e-8))


# ---------------------------------------------------------------- training


def main():
    names = list(params)
    t0 = time.perf_counter()
    # THE line: loss + one symbolic gradient per parameter, compiled together.
    step = compile_to_callable(loss, *[loss.grad(params[n]) for n in names],
                               torch_compile=True)
    predict = compile_to_callable(logits)  # forward-only program for evaluation
    from tensorgrad.compiler.ir import ConstNode, InputNode, toposort
    n_ops = sum(not isinstance(n, (InputNode, ConstNode))
                for n in toposort([n for n, _ in step.outputs]))
    print(f"compiled loss + {len(names)} gradients into one program of "
          f"{n_ops} tensor ops ({time.perf_counter() - t0:.1f}s)")

    gen = torch.Generator().manual_seed(0)
    ws = init_weights(gen)
    opt_m = {n: torch.zeros_like(w) for n, w in ws.items()}
    opt_v = {n: torch.zeros_like(w) for n, w in ws.items()}
    feed = {params[n]: ws[n] for n in names}  # tensors updated in place by adamw
    feed[causal_mask] = torch.triu(torch.full((SEQ, SEQ), -1e9), diagonal=1)
    # (per-call inputs -- tokens/targets -- are merged in fresh each call)

    def held_out_accuracy(n=256, seed=1234):
        """Greedy autoregressive decode on fresh problems; exact-match rate."""
        g = torch.Generator().manual_seed(seed)
        inp = torch.randint(VOCAB, (n, LENGTH), generator=g)
        ctx = torch.zeros(n, SEQ, dtype=torch.long)
        ctx[:, :LENGTH] = inp
        for t in range(LENGTH - 1, SEQ):
            out = predict({**feed, tokens: ctx.float()})
            nxt = out.align_to("batch", "seq", "vocab").rename(None)[:, t].argmax(-1)
            if t + 1 < SEQ:
                ctx[:, t + 1] = nxt
        pred = torch.cat([ctx[:, LENGTH:], nxt.unsqueeze(1)], dim=1)
        return (pred == inp.sort(dim=1).values).all(dim=1).float().mean().item()

    acc, t_start = 0.0, time.perf_counter()
    for it in range(1, MAX_STEPS + 1):
        xs, ys = sort_batch(BATCH, gen)
        loss_val, *grad_vals = step({**feed, tokens: xs, targets: ys})
        grads = {n: g.align_to(*params[n].edges).rename(None)  # match param layout
                 for n, g in zip(names, grad_vals)}
        adamw_update(ws, opt_m, opt_v, grads, it)
        if it == 1:  # first call pays codegen + torch.compile; time the rest
            t_start, warmup = time.perf_counter(), time.perf_counter() - t_start
            print(f"first step (codegen + torch.compile warmup): {warmup:.1f}s")
        if it % 50 == 0:
            acc = held_out_accuracy()
            rate = (it - 1) / (time.perf_counter() - t_start)
            print(f"step {it:4d}  loss {loss_val.item():.4f}  "
                  f"held-out acc {acc:.3f}  ({rate:.1f} steps/s)")
            if acc >= 0.99:
                break
    print(f"final held-out accuracy: {acc:.3f} "
          f"({'PASS' if acc > 0.9 else 'FAIL'}, target 0.9)")


if __name__ == "__main__":
    main()
