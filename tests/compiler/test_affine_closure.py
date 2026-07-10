"""The structural-tensor contraction closure (task #46).

contract_affines (registered from compiler/affine.py into
simplify.PAIR_RULES): contracting two Affine-representable factors
concatenates their row systems and eliminates the shared edges over the
integers. Eliminating a summed edge emits the boundary RANGE row
(sum_o [i = o + a] = [0 <= i - a <= X-1]) — the #44 vocabulary is what
makes the closure expressible. Unit-coefficient elimination only;
anything else returns None (missed simplification, never a wrong one).
"""

import random

import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Delta, Variable
from tensorgrad.compiler.affine import Affine, affine_shift, affine_tril
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.tensor import Product

n = symbols("n")
N = 7


def _dense(t, dims=None):
    out = evaluate(t, {}, dims or {n: N})
    return out.align_to(*sorted(out.names)).rename(None) if out.names else out.rename(None)


def test_window_composition_is_one_affine():
    w1 = F.window(start=2, i=n, o=n)
    w2 = F.window(start=3, o=n, p=n)
    comp = (w1 @ w2).simplify()
    assert isinstance(comp, Affine)
    assert torch.equal(
        _dense(comp),
        (evaluate(w1, {}, {n: N}).align_to("i", "o").rename(None)
         @ evaluate(w2, {}, {n: N}).align_to("o", "p").rename(None)),
    )


def test_sum_over_window_output_is_range_indicator():
    """sum_o [i == o + 2] = [0 <= i - 2 <= N-1]: the truncated-ones vector."""
    w = F.window(start=2, i=n, o=n)
    s = F.sum(w, dim="o").simplify()
    got = _dense(s)
    want = evaluate(w, {}, {n: N}).align_to("i", "o").rename(None).sum(dim=1)
    assert torch.equal(got, want)


def test_delta_through_shift():
    """Delta(n; a,b,c) contracted with a shift on c folds to one Affine."""
    d = Delta(n, "a", "b", "c")
    sh = affine_shift(1, c=n, o=n)
    comp = (d @ sh).simplify()
    got = _dense(comp)
    ref = (_dense_delta3() @ evaluate(sh, {}, {n: N}).align_to("c", "o").rename(None).reshape(N, N)).reshape(N, N, N)
    # reference the slow way: contract axis c of the dense delta with the shift
    dd = _dense_delta3()
    shm = evaluate(sh, {}, {n: N}).align_to("c", "o").rename(None)
    ref = torch.einsum("abc,co->abo", dd, shm)
    assert torch.equal(got.rename(None), ref)


def _dense_delta3():
    d = torch.zeros(N, N, N)
    i = torch.arange(N)
    d[i, i, i] = 1.0
    return d


def test_causal_mask_through_shift():
    """Range rows substitute correctly: tril contracted with a shift is the
    shifted triangle, still ONE Affine."""
    tril = affine_tril(seq=n, key=n)
    sh = affine_shift(1, key=n, k2=n)
    comp = (tril @ sh).simplify()
    assert isinstance(comp, Affine)
    got = evaluate(comp, {}, {n: N}).align_to("seq", "k2").rename(None)
    ref = torch.einsum(
        "sk,ko->so",
        evaluate(tril, {}, {n: N}).align_to("seq", "key").rename(None),
        evaluate(sh, {}, {n: N}).align_to("key", "k2").rename(None),
    )
    assert torch.equal(got, ref)


def test_free_shared_edge_gives_multiplicity():
    """Contracting rowless Affines (all-ones) over a shared edge yields the
    size as an order-0 Delta weight."""
    a1 = Affine([], a=n, b=n)
    a2 = Affine([], b=n, c=n)
    comp = (a1 @ a2).simplify()
    got = evaluate(comp, {}, {n: N}).align_to("a", "c").rename(None)
    assert torch.equal(got, torch.full((N, N), float(N)))


def test_non_unit_coefficient_declines():
    """[a == 2b] summed over b needs a divisibility condition: the rule must
    decline (stay a Product) and the value must be unchanged."""
    a1 = Affine([({"a": 1, "b": -2}, 0)], a=n, b=n)
    s = F.sum(a1, dim="b")
    simp = s.simplify()
    assert not isinstance(simp, Affine)
    got = _dense(simp)
    want = evaluate(a1, {}, {n: N}).align_to("a", "b").rename(None).sum(dim=1)
    assert torch.equal(got, want)


def test_fuzz_soundness_vs_dense():
    """Random Affine pairs, random contractions: whenever simplify rewrites,
    the dense value must be EXACTLY preserved (0/1 algebra, no tolerance)."""
    rng = random.Random(0)
    sizes = {n: 5}
    for trial in range(60):
        e1 = ["a", "b", "c"]
        e2 = ["b", "c", "d"] if rng.random() < 0.5 else ["b", "d"]

        def rand_rows(edges):
            rows = []
            for _ in range(rng.randrange(0, 3)):
                coeffs = {f: rng.choice([-2, -1, 1, 2]) for f in edges if rng.random() < 0.7}
                if not coeffs:
                    continue
                if rng.random() < 0.3:
                    rows.append(("range", coeffs, rng.randrange(-2, 3), rng.randrange(2, 7)))
                else:
                    rows.append((coeffs, rng.randrange(-2, 3)))
            return rows

        t1 = Affine(rand_rows(e1), **{f: n for f in e1})
        t2 = Affine(rand_rows(e2), **{f: n for f in e2})
        prod = Product([t1, t2])
        simp = prod.simplify()
        got = evaluate(simp, {}, dict(sizes))
        want = evaluate(prod, {}, dict(sizes))
        got = got.align_to(*sorted(got.names)).rename(None) if got.names else got.rename(None)
        want = want.align_to(*sorted(want.names)).rename(None) if want.names else want.rename(None)
        assert torch.equal(got, want), f"trial {trial}: {t1!r} x {t2!r}"


def test_compiled_program_with_closure_output():
    """A closure-produced Affine flows through the compiler: mask a variable
    with a composed window chain and compare against the dense computation."""
    x = Variable("x", i=n, p=n)
    w1 = F.window(start=2, i=n, o=n)
    w2 = F.window(start=3, o=n, p=n)
    y = F.sum(x * (w1 @ w2))
    from tensorgrad.compiler import compile_to_callable

    prog = compile_to_callable(y)
    torch.manual_seed(0)
    xt = torch.randn(N, N).rename("i", "p")
    out = prog({x: xt}, {n: N})
    dense = torch.einsum(
        "io,op->ip",
        evaluate(w1, {}, {n: N}).align_to("i", "o").rename(None),
        evaluate(w2, {}, {n: N}).align_to("o", "p").rename(None),
    )
    want = (xt.rename(None) * dense).sum()
    torch.testing.assert_close(out.rename(None), want, rtol=1e-6, atol=1e-7)
