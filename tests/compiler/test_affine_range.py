"""Affine range rows (task #44, stage 1): inequality indicators as
structural tensors. [0 <= sum c_i*e_i + k <= X-1] rows on Affine give
tril/triu/causal masks as ONE constant instead of an O(n)-term Sum of
shifted diagonals. Stage 1 lowers any range-rowed Affine as a hoisted
dense indicator ("affine" ConstNode) — correct by construction; native
IR range constraints are the #46-era follow-up.
"""

import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler.affine import Affine, affine_ineq, affine_tril, affine_triu
from tensorgrad.extras.evaluate import evaluate


def test_tril_triu_evaluate():
    n = symbols("n")
    N = 5
    tril = evaluate(affine_tril(r=n, c=n), {}, {n: N}).rename(None)
    assert torch.equal(tril, torch.tril(torch.ones(N, N)))
    tril_s = evaluate(affine_tril(strict=True, r=n, c=n), {}, {n: N}).rename(None)
    assert torch.equal(tril_s, torch.tril(torch.ones(N, N), diagonal=-1))
    triu = evaluate(affine_triu(r=n, c=n), {}, {n: N}).rename(None)
    assert torch.equal(triu, torch.triu(torch.ones(N, N)))


def test_band_indicator_evaluate():
    """|i - j| <= 1: a tridiagonal band from one affine_ineq row."""
    n = symbols("n")
    N = 6
    band = affine_ineq({"i": 1, "j": -1}, -1, 1, i=n, j=n)
    got = evaluate(band, {}, {n: N}).rename(None)
    want = (torch.arange(N)[:, None] - torch.arange(N)[None, :]).abs() <= 1
    assert torch.equal(got, want.to(got.dtype))


def test_mixed_eq_and_range_rows():
    """Equality and range rows compose in one Affine: the strict lower
    triangle of a shifted diagonal-plane [i == j + 1 and i >= k]."""
    n = symbols("n")
    N = 4
    t = Affine(
        [({"i": 1, "j": -1}, 1), ("range", {"i": 1, "k": -1}, 0, n)],
        i=n, j=n, k=n,
    )
    got = evaluate(t, {}, {n: N}).rename(None)
    i, j, k = torch.meshgrid(*[torch.arange(N)] * 3, indexing="ij")
    want = (i == j + 1) & (i >= k)
    assert torch.equal(got, want.to(got.dtype))


def test_compiled_matches_evaluate():
    """Compile a program contracting a range-rowed mask: x masked-rowsum."""
    n = symbols("n")
    x = Variable("x", r=n, c=n)
    y = F.sum(x * affine_tril(r=n, c=n), dim="c")  # per-row prefix sums
    prog = compile_to_callable(y)
    N = 5
    torch.manual_seed(0)
    vals = {x: torch.randn(N, N).rename("r", "c")}
    out = prog(vals, {n: N})
    ref = evaluate(y.full_simplify(), dict(vals), {n: N})
    torch.testing.assert_close(
        out.rename(None), ref.align_to(*out.names).rename(None), rtol=1e-5, atol=1e-6
    )
    xr = vals[x].rename(None)
    want = (xr * torch.tril(torch.ones(N, N))).sum(dim=1)
    torch.testing.assert_close(out.rename(None), want, rtol=1e-5, atol=1e-6)


def test_causal_mask_gradients_match_diagonals_form():
    """The causal-masked softmax attention built with ONE range-rowed Affine
    computes the same loss and gradient as the diagonals()-Sum form that
    examples/mingpt.py uses (a Sum of seq shifted F.window diagonals)."""
    from tensorgrad.tensor import Sum

    n, d = symbols("n d")
    N, D = 5, 3
    q = Variable("q", seq=n, hs=d)
    k = Variable("k", key=n, hs=d)
    v = Variable("v", key=n, dv=d)

    def attn(mask):
        scores = F.dot(q, k, dim="hs") + (-1e9) * mask
        return F.sum(F.dot(F.softmax(scores, dim="key"), v, dim="key"))

    mask_range = affine_tril(strict=True, key=n, seq=n)  # [key > seq]
    mask_diags = Sum([F.window(start=kk, key=n, seq=n) for kk in range(1, N)])

    torch.manual_seed(0)
    vals = {
        q: torch.randn(N, D, dtype=torch.float64).rename("seq", "hs"),
        k: torch.randn(N, D, dtype=torch.float64).rename("key", "hs"),
        v: torch.randn(N, D, dtype=torch.float64).rename("key", "dv"),
    }
    outs = []
    for mask in (mask_range, mask_diags):
        loss = attn(mask)
        g = loss.grad(q)
        prog = compile_to_callable(loss, g)
        lv, gv = prog(dict(vals), {n: N, d: D})
        outs.append((lv.rename(None), gv.align_to("seq", "hs").rename(None)))
    torch.testing.assert_close(outs[0][0], outs[1][0], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(outs[0][1], outs[1][1], rtol=1e-9, atol=1e-12)


def test_rename_and_identity():
    n = symbols("n")
    t = affine_tril(r=n, c=n)
    r = t.rename(r="a", c="b")
    assert set(r.edges) == {"a", "b"}
    N = 4
    got = evaluate(r, {}, {n: N}).align_to("a", "b").rename(None)
    assert torch.equal(got, torch.tril(torch.ones(N, N)))
    # hash/equality distinguish different range rows
    assert hash(affine_tril(r=n, c=n)) != hash(affine_tril(strict=True, r=n, c=n))
    assert hash(affine_tril(r=n, c=n)) != hash(affine_triu(r=n, c=n))
