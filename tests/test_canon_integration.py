"""Integration tests for the canon-backed hashing/equality wiring in tensor.py.

Covers (task #10):
  * __hash__ / is_isomorphic fast paths agree with the pure-nx implementation
  * deterministic simplify: identical expressions built with different
    commutative argument order (and under different PYTHONHASHSEED values)
    simplify to identical results
  * regression tests for three latent bugs found by the minGPT bridge:
      (a) lazy-Rename-wrapped Sum factors were invisible to Product expand
      (b) the expand recursion reset all other simplify flags to defaults
      (c) Tensor.substitute now is memoized and sharing-preserving
"""

import os
import subprocess
import sys
import textwrap

import networkx as nx
import pytest
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Delta, Product, Sum, Variable
from tensorgrad.tensor import Derivative, Rename, Tensor, set_lazy_rename

i, j, k, n = symbols("i j k n")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 1. Fast-path equivalence: is_isomorphic must agree with the pure nx test.
# ---------------------------------------------------------------------------


def _nx_is_isomorphic(a: Tensor, b: Tensor, match_edges: bool = False) -> bool:
    """The original (pre-fast-path) implementation, for cross-checking."""
    G1, _ = a.edge_structural_graph(match_edges=match_edges, edge_names={})
    G2, _ = b.edge_structural_graph(match_edges=match_edges, edge_names={})
    return nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name"))


def _corpus():
    # Reuse the randomized corpus from the canon test suite.
    from tests.compiler.test_canon import EXPRS, VARIANT_PAIRS

    pairs = list(VARIANT_PAIRS)
    # Add some organic (non-variant) pairs, including non-isomorphic ones.
    for a, b in zip(EXPRS[0::7], EXPRS[1::7]):
        pairs.append((a, b))
    return pairs


@pytest.mark.parametrize("match_edges", [False, True])
def test_is_isomorphic_agrees_with_nx(match_edges):
    for a, b in _corpus():
        expected = _nx_is_isomorphic(a, b, match_edges=match_edges)
        assert a.is_isomorphic(b, match_edges=match_edges) == expected, (a, b)


def test_hash_eq_contract_on_corpus():
    for a, b in _corpus():
        if a == b:
            assert hash(a) == hash(b)


def test_softmax_hessian_pair_not_combined():
    # Isomorphic but name-distinct: must be __eq__-equal yet match_edges-distinct.
    a = Product([Delta(n, "o", "i"), Delta(n, "p", "j", "k")])
    b = Product([Delta(n, "o", "j"), Delta(n, "p", "i", "k")])
    assert a == b
    assert hash(a) == hash(b)
    assert not a.is_isomorphic(b, match_edges=True)
    # And the Sum simplification keeps them as separate terms.
    assert len(Sum([a, b]).simplify().terms) == 2


# ---------------------------------------------------------------------------
# 2. Determinism: commutative construction order and PYTHONHASHSEED must not
#    leak into simplified results.
# ---------------------------------------------------------------------------

_DETERMINISM_SCRIPT = textwrap.dedent(
    """
    from sympy import symbols
    import tensorgrad.functions as F
    from tensorgrad import Variable, Product, Sum

    i, j, k = symbols("i j k")
    x = Variable("x", i=i, j=j)
    y = Variable("y", i=i, j=j)
    z = Variable("z", j=j, k=k)
    w = Variable("w", j=j, k=k)

    # Same expression, different commutative construction orders.
    a = Product([Sum([x, y], [2, 3]), z]).simplify({"expand": True})
    b = Product([z, Sum([y, x], [3, 2])]).simplify({"expand": True})
    # A wider sum-of-products, permuted.
    c = (x @ z + y @ w + x @ w + y @ z).simplify()
    d = (y @ z + x @ w + y @ w + x @ z).simplify()
    # A derivative-driven simplify (exercises fresh-name generation).
    e = F.frobenius2(x @ z).grad(x).simplify()
    for t in [a, b, c, d, e]:
        print(repr(t))
        print("@@@")
    """
)


def _run_determinism_script(seed: int) -> list[str]:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run(
        [sys.executable, "-c", _DETERMINISM_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return [part.strip() for part in out.stdout.split("@@@")]


def test_simplify_deterministic_across_seeds_and_argument_order():
    results = {seed: _run_determinism_script(seed) for seed in (0, 1, 2)}
    for seed, lines in results.items():
        assert lines[0] == lines[1], f"commutative Sum order leaked (seed {seed})"
        assert lines[2] == lines[3], f"commutative term order leaked (seed {seed})"
    assert results[0] == results[1] == results[2], "PYTHONHASHSEED leaked into simplify results"


def test_simplify_commutative_order_in_process():
    x = Variable("x", i=i, j=j)
    y = Variable("y", i=i, j=j)
    z = Variable("z", j=j, k=k)
    a = Product([Sum([x, y]), z]).simplify({"expand": True})
    b = Product([z, Sum([y, x])]).simplify({"expand": True})
    assert repr(a) == repr(b)
    # Product factor order is normalized too.
    p = Product([x, z]).simplify()
    q = Product([z, x]).simplify()
    assert repr(p) == repr(q)


# ---------------------------------------------------------------------------
# 3a. Lazy Rename-wrapped Sum factors must not block Product expand.
# ---------------------------------------------------------------------------


def test_expand_through_lazy_rename_wrapped_sum():
    prev = set_lazy_rename(True)
    try:
        x = Variable("x", i=i)
        y = Variable("y", i=i)
        z = Variable("z", i=i)
        s = (x + y).rename(i="j")  # Rename(Sum) under lazy renaming
        assert isinstance(s, Rename) and isinstance(s.tensor, Sum)
        p = Product([s, z.rename(i="j")])
        res = p.simplify({"expand": True})
        assert isinstance(res, Sum), f"expand failed to distribute: {res!r}"
        assert len(res.terms) == 2
        # And the result matches the eager-rename result.
        set_lazy_rename(False)
        expected = Product([(x + y).rename(i="j"), z.rename(i="j")]).simplify({"expand": True})
        assert res == expected
    finally:
        set_lazy_rename(prev)


# ---------------------------------------------------------------------------
# 3b. The expand recursion must preserve the caller's simplify args.
# ---------------------------------------------------------------------------


def _contains_derivative(t: Tensor) -> bool:
    if isinstance(t, Derivative):
        return True
    children = []
    if isinstance(t, Sum):
        children = t.terms
    elif isinstance(t, Product):
        children = t.factors
    elif isinstance(t, Rename):
        children = [t.tensor]
    elif hasattr(t, "inputs"):
        children = t.inputs
    return any(_contains_derivative(c) for c in children)


def test_expand_preserves_caller_args():
    u = Variable("u", i=i)
    a = Variable("a", i=i)
    b = Variable("b", i=i)
    d = Derivative(u, u)  # kept unevaluated iff grad_steps == 0
    p = Product([Sum([a, b]), d])
    res = p.simplify({"expand": True, "grad_steps": 0})
    assert isinstance(res, Sum) and len(res.terms) == 2, repr(res)
    # Before the fix, the expand recursion reset grad_steps to inf and the
    # Derivative was evaluated away.
    assert _contains_derivative(res), f"grad_steps=0 was dropped by the expand recursion: {res!r}"


# ---------------------------------------------------------------------------
# 3c. substitute: memoized, sharing-preserving, and behavior-identical.
# ---------------------------------------------------------------------------


def test_substitute_equivalence():
    x = Variable("x", i=i, j=j)
    y = Variable("y", i=i, j=j)
    z = Variable("z", j=j, k=k)
    repl = Sum([y, y.rename(i="j", j="i")]) if False else (2 * y)

    expr = Sum([Product([x, z]), Product([y, z])], [1, -1])
    res = expr.substitute(x, repl)
    expected = Sum([Product([repl, z]), Product([y, z])], [1, -1])
    assert res == expected
    assert res.shape == expr.shape

    # Substituting through Function / Rename nodes.
    f = F.softmax(x.rename(i="a"), dim="j")
    assert f.substitute(x, y) == F.softmax(y.rename(i="a"), dim="j")

    # A variable that doesn't occur: the tensor is returned unchanged (and,
    # with memoization, as the very same object).
    q = Variable("q", i=i)
    assert expr.substitute(q, Delta(i, "i")) is expr

    # Name-shadowing: a different variable with the same name is not replaced.
    x_other = Variable("x", i=i)  # same name, different shape => not equal
    assert x_other.substitute(x, repl) is x_other
    assert x.substitute(x_other, q) is x


def test_substitute_preserves_sharing_and_is_fast():
    # A doubling DAG: tree-size 2^60, DAG size 61. The old substitute
    # recursion would never terminate; the memoized one is instant.
    x = Variable("x", i=i)
    t = x
    for _ in range(60):
        t = Sum([t, t])
    y = Variable("y", i=i)
    res = t.substitute(x, y)
    assert isinstance(res, Sum)
    assert res.terms[0] is res.terms[1], "sharing was not preserved"
    leaf = res
    while isinstance(leaf, Sum):
        leaf = leaf.terms[0]
    assert leaf is y
