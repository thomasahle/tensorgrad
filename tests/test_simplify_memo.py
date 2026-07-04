"""Cross-subtree simplify memoization and the simplify_for_compile lite path.

Without memoization, a node reachable by K distinct DAG paths is re-simplified
K times; residual structure (x = x + f(x)) doubles the path count per layer, so
deep-model simplify is exponential in depth. Memoization (opt-in, keyed by
structural fingerprint + free edges) makes it linear, and simplify_for_compile
keeps the symbolic tree small by leaving the algebra to the compiler's IR
passes.
"""

import torch
from sympy import symbols

from tensorgrad import Variable
from tensorgrad.tensor import Derivative
import tensorgrad.functions as F
from tensorgrad.compiler import compile_to_callable


def _has_derivative(t, seen=None):
    seen = seen or set()
    if id(t) in seen:
        return False
    seen.add(id(t))
    if isinstance(t, Derivative):
        return True
    children = (
        list(getattr(t, "factors", []))
        + list(getattr(t, "terms", []))
        + list(getattr(t, "inputs", []))
        + ([t.tensor] if hasattr(t, "tensor") else [])
    )
    return any(_has_derivative(c, seen) for c in children)


def _dag_nodes(t, seen=None):
    seen = seen if seen is not None else set()
    if id(t) in seen:
        return seen
    seen.add(id(t))
    for c in (
        list(getattr(t, "factors", []))
        + list(getattr(t, "terms", []))
        + list(getattr(t, "inputs", []))
        + ([t.tensor] if hasattr(t, "tensor") else [])
    ):
        _dag_nodes(c, seen)
    return seen


def test_memoize_matches_non_memoized():
    """Opt-in memoization must produce an identical (isomorphic) result."""
    b, i, j = symbols("b i j")
    x = Variable("x", b, i)
    W = Variable("W", i, j)
    c = Variable("c", b, j)
    h = F.relu(x @ W + c)  # shared subtree
    loss = F.sum(h * h)
    g = loss.grad(W)

    r_memo = g.simplify({"grad_steps": float("inf"), "memoize": True})
    r_plain = g.simplify({"grad_steps": float("inf"), "memoize": False})
    assert r_memo == r_plain  # isomorphism

    # And the compiled outputs agree numerically.
    dims = {b: 4, i: 5, j: 6}
    torch.manual_seed(0)
    vals = {
        x: torch.randn(4, 5).refine_names("b", "i"),
        W: torch.randn(5, 6).refine_names("i", "j"),
        c: torch.randn(4, 6).refine_names("b", "j"),
    }
    ra = compile_to_callable(r_memo)(dict(vals), dims)
    rb = compile_to_callable(r_plain)(dict(vals), dims)
    torch.testing.assert_close(
        ra.rename(None), rb.align_to(*ra.names).rename(None), rtol=1e-5, atol=1e-6
    )


def test_memoize_with_expand_matches():
    b, i, j = symbols("b i j")
    x = Variable("x", b, i)
    W = Variable("W", i, j)
    loss = F.sum(F.softmax(x @ W, dim="j"))
    g = loss.grad(W)
    a = g.simplify({"grad_steps": float("inf"), "expand": True, "memoize": True})
    p = g.simplify({"grad_steps": float("inf"), "expand": True, "memoize": False})
    assert a == p


def test_simplify_for_compile_is_derivative_free_and_correct():
    """The lite path eliminates Derivatives and compiles to code matching the
    fully-simplified path."""
    b, i, j, k = symbols("b i j k")
    x = Variable("x", b, i)
    W1 = Variable("W1", i, j)
    W2 = Variable("W2", j, k)
    y = Variable("y", b, k)
    loss = F.sum((F.relu(x @ W1) @ W2 - y) ** 2)
    g = loss.grad(W1)

    lite = g.simplify_for_compile()
    assert not _has_derivative(lite)

    full = g.full_simplify()
    dims = {b: 4, i: 5, j: 6, k: 3}
    torch.manual_seed(0)
    vals = {
        x: torch.randn(4, 5).refine_names("b", "i"),
        W1: torch.randn(5, 6).refine_names("i", "j"),
        W2: torch.randn(6, 3).refine_names("j", "k"),
        y: torch.randn(4, 3).refine_names("b", "k"),
    }
    r_lite = compile_to_callable(lite)(dict(vals), dims)
    r_full = compile_to_callable(full)(dict(vals), dims)
    torch.testing.assert_close(
        r_lite.rename(None), r_full.align_to(*r_lite.names).rename(None), rtol=1e-4, atol=1e-6
    )


def test_deep_chain_grad_is_bounded_and_correct():
    """A deep feed-forward chain's first-layer gradient has a linear (adjoint)
    structure; simplify_for_compile keeps it small and compiles correctly.
    (Real transformers are in this benign regime — minGPT's per-layer softmax/
    layernorm keep the back-propagated adjoint shared, measured linear through
    6 layers. A bare residual with a squared-error loss can instead expand into
    genuinely-distinct terms that no memoization collapses; that is expected.)
    """
    b, i = symbols("b i")
    x = Variable("x", b, i)

    def grad_nodes(depth):
        h = x
        weights = []
        for n in range(depth):
            W = Variable(f"W{n}", i=i, o=i)
            weights.append(W)
            h = F.relu(h @ W).rename(o="i")  # feed-forward (no residual)
        loss = F.sum(h * Variable("t", b, i))  # linear read-out -> single adjoint
        g = loss.grad(weights[0]).simplify_for_compile()
        assert not _has_derivative(g)
        return len(_dag_nodes(g))

    n2, n4, n8 = grad_nodes(2), grad_nodes(4), grad_nodes(8)
    # Grows linearly, not exponentially: doubling depth must not >4x the nodes.
    assert n4 < 4 * n2 and n8 < 4 * n4, f"superlinear: {n2}, {n4}, {n8}"


def test_compile_to_callable_accepts_raw_derivatives():
    """compile_to_callable resolves Derivative nodes internally (shared memo
    across outputs), so raw loss.grad(p) can be passed directly."""
    b, i, j, k = symbols("b i j k")
    x = Variable("x", b, i)
    W1 = Variable("W1", i, j)
    W2 = Variable("W2", j, k)
    y = Variable("y", b, k)
    loss = F.sum((F.relu(x @ W1) @ W2 - y) ** 2)
    params = [W1, W2]

    # The killer-demo one-liner: raw Derivative nodes straight in.
    f_raw = compile_to_callable(loss, *[loss.grad(p) for p in params])
    # Explicit route for comparison.
    f_exp = compile_to_callable(loss.simplify_for_compile(),
                                *[loss.grad(p).simplify_for_compile() for p in params])

    dims = {b: 4, i: 5, j: 6, k: 3}
    torch.manual_seed(0)
    vals = {
        x: torch.randn(4, 5).refine_names("b", "i"),
        W1: torch.randn(5, 6).refine_names("i", "j"),
        W2: torch.randn(6, 3).refine_names("j", "k"),
        y: torch.randn(4, 3).refine_names("b", "k"),
    }
    outs_raw = f_raw(dict(vals), dims)
    outs_exp = f_exp(dict(vals), dims)
    for a, e in zip(outs_raw, outs_exp):
        torch.testing.assert_close(
            a.rename(None), e.align_to(*a.names).rename(None), rtol=1e-4, atol=1e-6
        )
