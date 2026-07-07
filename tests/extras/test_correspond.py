"""Correspondence matcher tests on real simplify-step pairs.

Each test hand-checks the classification the animation compiler will
consume: moved (Transform), copies (TransformFromCopy), merges,
births/deaths. match() itself audits total classification, so every test
also proves no atom was silently dropped.
"""

from sympy import symbols

from tensorgrad import Variable
from tensorgrad.tensor import Product, Sum
from tensorgrad.extras.correspond import match

n = symbols("n")


def test_identical_steps_all_moved():
    A = Variable("A", i=n, j=n)
    B = Variable("B", j=n, k=n)
    t = A @ B
    m = match(t, t)
    assert len(m.moved) == 2
    assert not m.copies and not m.merges and not m.births and not m.deaths
    # exact identity: paths map 1:1
    assert all(b.path == a.path for b, a in m.moved)


def test_distribution_creates_copies():
    # A (B + C)  ->  A B + A C : the rule reuses the SAME A leaf object in
    # both terms, so one occurrence is moved and the surplus is a copy.
    A = Variable("A", i=n, j=n)
    B = Variable("B", j=n, k=n)
    C = Variable("C", j=n, k=n)
    before = Product([A, Sum([B, C])])
    after = before.simplify({"expand": True})
    assert isinstance(after, Sum) and len(after.terms) == 2

    m = match(before, after)
    labels_moved = sorted(b.label for b, _ in m.moved)
    # B and C move; A moves once
    assert labels_moved == ["A", "B", "C"]
    # ...and appears once more as a copy
    assert [b.label for b, _ in m.copies] == ["A"]
    assert not m.births and not m.deaths and not m.merges


def test_gradient_step_classifies_everything():
    # d(x^T A x)/dx -> A x + A^T x (shapes vary by simplification), with
    # A and x duplicated across terms. We assert full classification and
    # that no Variable dies: differentiation only rearranges/copies them.
    i = symbols("i")
    x = Variable("x", i=i)
    A = Variable("A", i=i, j=i)
    q = x.rename(i="j") @ A @ x
    before = q.grad(x, {"i": "p"})
    after = before.simplify()

    m = match(before, after)
    # the quadratic's atoms (x, A, x) all survive into the two terms
    dead_vars = [d for d in m.deaths if d.kind == "var"]
    assert not dead_vars
    # something was duplicated by the product rule
    assert m.copies or m.merges


def test_factor_out_of_product_keeps_identity():
    # (A B) C -> associativity/merge inside simplify: atoms all survive.
    A = Variable("A", i=n, j=n)
    B = Variable("B", j=n, k=n)
    C = Variable("C", k=n, l=n)
    before = Product([Product([A, B]), C])
    after = before.simplify()
    m = match(before, after)
    assert sorted(b.label for b, _ in m.moved) == ["A", "B", "C"]
    assert not m.births and not m.deaths


def test_audit_counts_are_totals():
    A = Variable("A", i=n, j=n)
    B = Variable("B", j=n, k=n)
    C = Variable("C", j=n, k=n)
    before = Product([A, Sum([B, C])])
    after = before.simplify({"expand": True})
    m = match(before, after)
    # 3 atoms before; 4 after (A twice)
    assert len(m.moved) + len(m.merges) + len(m.deaths) == 3
    assert (len(m.moved) + len(m.copies) + len(m.merges)
            + len(m.births)) == 4
