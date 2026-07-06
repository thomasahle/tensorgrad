"""Structural tests for the book-grammar layout engine."""

import pytest
from sympy import symbols

from tensorgrad.tensor import Delta, Derivative, Product, Sum, Variable
from tensorgrad.extras.book_layout import (
    extract_graph,
    layout_tensor,
    to_book_tikz,
)

n = symbols("n")


def _A(name="A", **kw):
    return Variable(name, **(kw or dict(i=n, j=n)))


def wires_by_kind(layout):
    out = {}
    for w in layout.wires:
        out.setdefault(w.kind, []).append(w)
    return out


# ---------------------------------------------------------------- extraction


def test_extract_matrix_product():
    g = extract_graph(Product([_A(), Variable("B", j=n, k=n)]))
    assert len(g.atoms) == 2
    internal = [w for w, e in g.wires.items() if len(e) == 2]
    free = [w for w in g.free_wires]
    assert len(internal) == 1 and len(free) == 2


def test_extract_identity_joins_wires():
    # Tr(A) via an explicit identity: A(i,j) contracted with Delta(i,j)
    g = extract_graph(Product([_A(), Delta(n, "i", "j")]))
    assert len(g.atoms) == 1  # the Delta is a wire, not an atom
    (wid,) = [w for w, e in g.wires.items() if len(e) == 2]
    assert g.wires[wid] == [0, 0]  # self-loop on A


def test_extract_bare_identity():
    g = extract_graph(Product([Delta(n, "i", "j")]))
    assert not g.atoms and g.bare_wires


def test_extract_copydot_orders():
    g = extract_graph(Product([Variable("v", m=n), Delta(n, "m", "i", "j")]))
    kinds = sorted(a.kind for a in g.atoms)
    assert kinds == ["copydot", "var"]
    assert len(g.free_wires) == 2


def test_rename_swap():
    A = _A()
    swapped = A.rename(i="j", j="i")
    g = extract_graph(Product([swapped, Variable("B", j=n, k=n)]))
    # after the swap, A's original *i* connects to B
    internal = [e for e in g.wires.values() if len(e) == 2]
    assert internal == [[0, 1]]


def test_derivative_draws_penrose_loop():
    x = Variable("x", i=n)
    A = _A()
    expr = Derivative(Product([x, A]), x, {"i": "i_"})
    tex = to_book_tikz(expr)
    assert "ellipse" in tex and "fit=" in tex  # the loop
    assert tex.count("circle (1.4pt)") == 1  # the boundary dot
    assert "i'" in tex  # labeled whisker (i_ renders as i-prime)


def test_contracted_derivative_edge_raises():
    x = Variable("x", i=n)
    A = _A()
    d = Derivative(Product([x, A]), x, {"i": "i_"})
    closed = Product([d, Variable("w", j=n, i_=n)])
    with pytest.raises(NotImplementedError):
        to_book_tikz(closed)


# -------------------------------------------------------------------- layout


def test_chain_spine_is_horizontal_and_ordered():
    a = Variable("a", i=n)
    B = Variable("B", j=n, k=n)
    b = Variable("b", k=n)
    lay = layout_tensor(Product([a, _A(), B, b]))
    ys = {round(nd.y, 4) for nd in lay.nodes}
    assert ys == {0.0}
    xs = [nd.x for nd in sorted(lay.nodes, key=lambda m: m.x)]
    assert xs == sorted(xs)
    labels = [nd.label for nd in sorted(lay.nodes, key=lambda m: m.x)]
    assert labels == ["a", "A", "B", "b"]


def test_trace_reads_in_expression_order():
    A = _A()
    B = Variable("B", j=n, k=n)
    C = Variable("C", k=n, l=n)
    D = Variable("D", l=n, i=n)
    lay = layout_tensor(Product([A, B, C, D]))
    labels = [nd.label for nd in sorted(lay.nodes, key=lambda m: m.x)]
    assert labels == ["A", "B", "C", "D"]
    arcs = wires_by_kind(lay).get("arc", [])
    assert len(arcs) == 1 and arcs[0].span == 3


def test_single_matrix_has_two_stubs():
    lay = layout_tensor(_A())
    stubs = wires_by_kind(lay)["stub"]
    assert sorted(s.direction for s in stubs) == ["left", "right"]


def test_outer_product_stubs_point_outward():
    lay = layout_tensor(Product([Variable("x", i=n), Variable("x", j=n)]))
    stubs = wires_by_kind(lay)["stub"]
    assert sorted(s.direction for s in stubs) == ["left", "right"]


def test_pendant_hangs_below():
    # Tr(A diag(v) B): v must hang below the spine, not join it
    expr = Product(
        [
            _A(),
            Delta(n, "j", "k", "m"),
            Variable("v", m=n),
            Variable("B", k=n, i=n),
        ]
    )
    lay = layout_tensor(expr)
    v_node = next(nd for nd in lay.nodes if nd.label == "v")
    assert v_node.y < 0
    spine_labels = [nd.label for nd in lay.nodes if nd.y == 0]
    assert "v" not in spine_labels
    assert wires_by_kind(lay).get("pendant")


def test_left_right_override():
    lay = layout_tensor(_A(), left="j", right="i")
    # with the override, the spine still has one left and one right stub
    stubs = wires_by_kind(lay)["stub"]
    assert sorted(s.direction for s in stubs) == ["left", "right"]


def test_scalar_component_comes_first():
    trace = Product([Variable("C", i=n, j=n), Delta(n, "i", "j")])
    expr = Product([trace, _A(name="A", a=n, b=n)])
    lay = layout_tensor(expr)
    c = next(nd for nd in lay.nodes if nd.label == "C")
    a = next(nd for nd in lay.nodes if nd.label == "A")
    assert c.x < a.x


# ------------------------------------------------------------------ emission


def test_tikz_smoke():
    tex = to_book_tikz(Product([_A(), Variable("B", j=n, k=n)]))
    assert tex.startswith(r"\begin{tikzpicture}")
    assert tex.endswith(r"\end{tikzpicture}")
    assert tex.count(r"\node") == 2


def test_sum_signs():
    A, B = _A(), _A("B")
    tex = to_book_tikz(Sum([A, B], [2, -1]))
    assert "$2\\," in tex and "$-" in tex


def test_arcs_use_book_style():
    tex = to_book_tikz(Product([_A(), Delta(n, "i", "j")]))
    assert "loop" in tex and "out=160, in=20" in tex


# --------------------------------------------- workflow-verified regressions


def test_wire_conservation_pendant_two_attachments():
    # v connects to BOTH B and D: no wire may be silently dropped
    expr = Product(
        [
            Variable("A", i=n, e1=n),
            Variable("B", e1=n, e2=n, p1=n),
            Variable("C", e2=n, e3=n),
            Variable("D", e3=n, e4=n, p2=n),
            Variable("E", e4=n, o=n),
            Variable("v", p1=n, p2=n),
        ]
    )
    g = extract_graph(expr)
    internal = sum(1 for e in g.wires.values() if len(e) == 2)
    lay = layout_tensor(expr)
    drawn = sum(
        1 for w in lay.wires if w.kind in ("segment", "pendant", "arc", "loop", "extra")
    )
    assert drawn == internal


def test_group_trace_compiles_to_valid_tikz():
    # Tr(A+B): the closure on the group must reference the paren nodes
    tex = to_book_tikz(Product([Sum([_A(), _A("B")]), Delta(n, "i", "j")]))
    import re

    defined = set(re.findall(r"\\node[^()]*\(([^)]+)\)", tex))
    used = set(re.findall(r"\((\w+[LR]?)\)(?:[^{]|$)", tex))
    for m in re.finditer(r"edge\s*\[[^]]*\]\s*\((\w+)\)", tex):
        assert m.group(1) in defined, f"undefined node {m.group(1)} in: {tex}"
    for m in re.finditer(r"\\path\[?[^(]*\((\w+)\)", tex):
        assert m.group(1) in defined, f"undefined node {m.group(1)}"


def test_identity_ring_renders_circle():
    # Tr(I) = n: a closed ring of identities must draw a circle, not nothing
    t = Product([Delta(n, "i", "j"), Delta(n, "i", "j")])
    g = extract_graph(t)
    assert g.rings == 1
    assert "circle" in to_book_tikz(t)


def test_sum_of_group_products_no_overlap():
    x_ = Variable("x", i=n)
    A, B, C = _A(), _A("B"), _A("C")
    s2 = Sum([Product([x_, Sum([A, B])]), Product([x_, C])])
    from tensorgrad.extras.book_layout import layout_any

    lay = layout_any(s2)
    # term boundaries must not interleave: every node of term 0 left of term 1
    xs0 = [nd.x + nd.width / 2 for nd in lay.nodes if nd.id < 1000 and nd.kind != "sign"]
    xs1 = [nd.x - nd.width / 2 for nd in lay.nodes if 1000 <= nd.id < 2000]
    assert max(xs0) < min(xs1)


def test_double_partial_trace_two_visible_loops():
    T = Variable("T", a=n, b=n, c=n, d=n)
    tex = to_book_tikz(Product([T, Delta(n, "a", "b"), Delta(n, "c", "d")]))
    import re

    loops = [l for l in tex.splitlines() if "loop" in l]
    assert len(loops) == 2 and len(set(loops)) == 2  # distinct sizes


def test_dense_component_terminates_quickly():
    import time

    vs = []
    names = "ABCDEFGH"
    edges = {}
    for i, a in enumerate(names):
        sh = {}
        for j, b in enumerate(names):
            if i < j:
                sh[f"e{i}{j}"] = n
            elif i > j:
                sh[f"e{j}{i}"] = n
        vs.append(Variable(a, **sh))
    t0 = time.time()
    layout_tensor(Product(vs))
    assert time.time() - t0 < 10


def test_multiple_up_stubs_fan_out():
    T = Variable("T", i=n, j=n, k=n, l=n)
    lay = layout_tensor(T)
    ups = [w for w in lay.wires if w.kind == "stub" and w.direction == "up"]
    assert len(ups) == 2
    assert len({w.x for w in ups}) == 2  # distinct horizontal offsets


def test_rotation_only_on_free_edge_matrices():
    # a matrix contracted on BOTH sides (internal chain node) must not be
    # drawn rotated, even if its declared port order is reversed vs the chain
    import tensorgrad.functions as F

    Q = Variable("Q", s=n, d=n)
    K = Variable("K", t=n, d=n)
    V = Variable("V", t=n, e=n)
    from tensorgrad.extras.book_layout import layout_any

    lay = layout_any(F.softmax(Q @ K, dim="t") @ V)
    assert all(not nd.rotated for nd in lay.nodes if nd.kind == "var")


def test_rotation_still_marks_transpose_with_free_edge():
    # the A^T term in a Hessian (A carries free derivative edges) DOES rotate
    x = Variable("x", i=n)
    A = _A()
    from tensorgrad.extras.book_layout import layout_any

    h = (
        Product([x, A, x.rename(i="j")])
        .grad(x, {"i": "p"})
        .grad(x, {"i": "q"})
        .simplify()
    )
    lay = layout_any(h)
    assert any(nd.rotated for nd in lay.nodes if nd.kind == "var")
