"""Zero-gradient pruning (task #68, symbolic-value audit finding #1).

Gradients that are identically zero by symmetry — the canonical case being
the second bias inside softmax attention (softmax is shift-invariant per
row) — are proven zero at compile time (szfp, exact mod P, k=3) and their
Derivative nodes replaced by Zero, deleting the whole cotangent chain.
Measured on the real minGPT: exactly {h0,h1,h2}.attn.bk prune; training
gates pass; the pass costs ~1.5s at gpt scale (one batched szfp evaluation
over all 53 gradients' shared DAG).
"""

import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler.zerograd import prune_zero_gradients
from tensorgrad.tensor import Rename, Zero


def _unwrap(t):
    while isinstance(t, Rename):
        t = t.tensor
    return t


def _attention_loss():
    """Toy single-head attention with a key bias: d loss / d bk == 0 by
    softmax shift invariance; d loss / d bq is NOT zero."""
    n, d = symbols("n d")
    q = Variable("q", seq=n, hs=d)
    k = Variable("k", key=n, hs=d)
    v = Variable("v", key=n, dv=d)
    bk = Variable("bk", hs=d)
    bq = Variable("bq", hs=d)
    scores = F.dot(q + bq, k + bk.rename(hs="hs"), dim="hs")
    att = F.softmax(scores, dim="key")
    return F.sum(F.dot(att, v, dim="key")), q, bk, bq, (n, d)


def test_shift_invariant_bias_gradient_prunes_to_zero(monkeypatch):
    """The pass is default-OFF (the sz_cancel tier proves these zeros in
    plain normalize now); force it on to test the machinery itself."""
    from tensorgrad.compiler import zerograd

    monkeypatch.setattr(zerograd, "ZERO_PRUNE", True)
    loss, q, bk, bq, _ = _attention_loss()
    g_bk = loss.grad(bk)
    g_bq = loss.grad(bq)
    pruned = prune_zero_gradients((loss, g_bk, g_bq))
    assert isinstance(_unwrap(pruned[1]), Zero), "bk gradient must prove zero"
    assert not isinstance(_unwrap(pruned[2]), Zero), "bq gradient must NOT prune"


def test_compiled_program_returns_exact_zeros_and_correct_rest():
    loss, q, bk, bq, (n, d) = _attention_loss()
    prog = compile_to_callable(loss, loss.grad(bk), loss.grad(bq))
    N, D = 5, 4
    torch.manual_seed(0)
    vals = {
        v: torch.randn(*[{"seq": N, "key": N, "hs": D, "dv": D}[e] for e in v.edges],
                       dtype=torch.float64).rename(*v.edges)
        for v in prog.vars
    }
    lv, gbk, gbq = prog(vals, {n: N, d: D})
    assert torch.equal(gbk.rename(None), torch.zeros(D, dtype=torch.float64)), (
        "pruned gradient must be EXACT zeros, not roundoff"
    )
    # reference for the surviving gradient via autograd
    qr = vals[q].rename(None)
    bkr = vals[bk].rename(None)
    bqr = vals[bq].rename(None).clone().requires_grad_(True)
    kr = vals[[v for v in prog.vars if v.name == "k"][0]].rename(None)
    vr = vals[[v for v in prog.vars if v.name == "v"][0]].rename(None)
    with torch.enable_grad():
        s = (qr + bqr) @ (kr + bkr).T
        out = (torch.softmax(s, dim=1) @ vr).sum()
        (gr,) = torch.autograd.grad(out, bqr)
    torch.testing.assert_close(gbq.rename(None), gr, rtol=1e-9, atol=1e-12)


def test_plain_programs_untouched():
    n = symbols("n")
    x = Variable("x", i=n)
    w = Variable("w", i=n)
    loss = F.sum(F.tanh(x * w))
    g = loss.grad(w)
    pruned = prune_zero_gradients((loss, g))
    assert pruned == (loss, g) or pruned[1] is g  # nothing proves zero
