import torch

import tensorgrad.tensor as tg
import tensorgrad.functions as F


def rand_values(variables, **shape):
    return {v: torch.randn([shape[e] for e in v.edges], names=v.edges) for v in variables}


def assert_close(a, b):
    assert set(a.names) == set(b.names)
    a = a.align_to(*b.names)
    torch.testing.assert_close(a.rename(None), b.rename(None))


def test_simple_grad():
    x = tg.Variable("x", ["x"])
    A = tg.Variable("A", ["x", "y"])
    ts = rand_values([x, A], x=2, y=3)
    res = (A @ x).grad(x).simplify()
    assert_close(res.evaluate(ts), ts[A].rename("x_", "y"))


def test_simple_hessian():
    x = tg.Variable("x", ["x"])
    A = tg.Variable("A", ["x", "y"])
    ts = rand_values([x, A], x=2, y=2)
    quad = x @ A @ x.rename({"x": "y"})
    res = quad.grad(x).grad(x).simplify()
    assert_close(res.evaluate(ts), (ts[A].rename(None) + ts[A].rename(None).T).rename("x_", "x__"))

    quad2 = x @ A @ A @ x
    res2 = quad2.grad(x).grad(x).simplify()
    tA = ts[A].rename(None)
    torch.testing.assert_close(res2.evaluate(ts).rename(None), 2 * tA @ tA.T)

    x = tg.Variable("x", ["x"])
    y = tg.Variable("y", ["y"])
    A = tg.Variable("A", ["x", "y"])
    ts = rand_values([x, y, A], x=2, y=3)
    frob = F.frobenius2(A @ x - y)
    hess = frob.grad(x).grad(x).simplify().evaluate(ts)
    # Hessian of ||Ax - y||^2 is 2 * A^T A
    tH = 2 * ts[A].rename(None) @ ts[A].rename(None).T
    assert_close(hess, tH.rename("x_", "x__"))


def test_derivative_of_hadamard_product():
    a = tg.Variable("a", ["i", "j"])
    b = tg.Variable("b", ["i", "j"])
    ts = rand_values([a, b], i=2, j=3)
    res_a = (a * b).grad(a).simplify().evaluate(ts)
    res_b = (a * b).grad(b).simplify().evaluate(ts)
    torch.testing.assert_close(res_a.sum(), ts[b].sum())
    torch.testing.assert_close(res_b.sum(), ts[a].sum())


def test_f_0_0():
    x = tg.Variable("x", [])
    f = F.Function("f", [], (x,))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == set()
    assert expr == tg.Function("D_0f", [], (x,))


def test_f_0_1():
    x = tg.Variable("x", ["i"])
    f = F.Function("f", [], (x, "i"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"i_"}
    fg = F.Function("D_0f", ["i_"], (x, "i"))
    assert expr == fg


# Function("D_0f", ["i_"], (Variable("x", ["i"], ["i"]), "i"), orig_edges_out=["i"])
# == Function("D_0f", ["i_"], (Variable("x", ["i"], ["i"]), "i"))


def test_f_0_2():
    x = tg.Variable("x", ["b", "i"])
    f = F.Function("f", [], (x, "i"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"b", "b_", "i_"}
    x_renamed = x.rename({"b": "b__"})
    fg = F.Function("D_0f", ["i_"], (x_renamed, "i")) @ tg.Copy(["b", "b_", "b__"])
    assert hash(expr) == hash(fg)


def test_f_1_0():
    x = tg.Variable("x", [])
    f = F.Function("f", ["y"], (x,))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"y"}
    assert expr == tg.Function("D_0f", ["y"], (x,))


def test_f_1_1():
    x = tg.Variable("x", ["i"])
    f = F.Function("f", ["y"], (x, "i"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"y", "i_"}
    assert expr == tg.Function("D_0f", ["y", "i_"], (x, "i"))


def test_fxy_1_1_1():
    x = tg.Variable("x", ["i"])
    y = tg.Variable("y", ["i"])
    f = F.Function("f", ["z"], (x, "i"), (y, "i"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"z", "i_"}
    assert expr == tg.Function("D_0f", ["z", "i_"], (x, "i"), (y, "i"))


def test_fxx_1_1_1():
    x = tg.Variable("x", ["i"])
    f = F.Function("f", ["z"], (x, "i"), (x, "i"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"z", "i_"}
    print("hashes")
    # We can't control what connection_edges are used by the chain rule, which makes
    # this test a bit more brittle.
    # fmt: off
    expected = (
        tg.Function("D_0f", ["z", "i_"], (x, "i"), (x, "i"))
      + tg.Function( "D_1f", ["z", "i_"], (x, "i"), (x, "i"))
    )
    # fmt: on
    print(f"{expr=}")
    print(f"{expected=}")
    assert expr == expected


def test_f_2_2_multi_input():
    x = tg.Variable("x", ["i", "j"])
    y = tg.Variable("y", ["k", "l"])
    f = F.Function("f", ["a", "b"], (x, "i"), (y, "k"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"a", "b", "l", "j", "j_", "i_"}
    assert expr == (
        tg.Function("D_0f", ["a", "b", "i_"], (x, "i"), (y, "k")).rename({"j": "j_0"})
        @ tg.Copy(["j", "j_", "j_0"])
    )


def test_f_1_1_nested():
    x = tg.Variable("x", ["i"])
    g = F.Function("g", ["j"], (x, "i"))
    f = F.Function("f", ["k"], (g, "j"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"k", "i_"}
    expected = tg.Function("D_0f", ["k", "j_"], (g, "j")) @ tg.Function(
        "D_0g", ["j_", "i_"], (x, "i"), orig_edges_out=["j", "i_"]
    )
    assert expr == expected


def test_f_2_1_nested():
    x = tg.Variable("x", ["i"])
    g = F.Function("g", ["j", "k"], (x, "i"))
    f = F.Function("f", ["l"], (g, "j", "k"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"l", "i_"}
    assert expr == tg.Function("D_0f", ["l", "j_", "k_"], (g, "j", "k")) @ tg.Function(
        "D_0g", ["j_", "k_", "i_"], (x, "i"), orig_edges_out=["j", "k", "i_"]
    )


def test_f_1_2_nested():
    x = tg.Variable("x", ["i", "j"])
    g = F.Function("g", ["k"], (x, "i", "j"))
    f = F.Function("f", ["l", "m"], (g, "k"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"l", "m", "i_", "j_"}
    assert expr == (
        tg.Function("D_0f", ["l", "m", "k_"], (g, "k"))
        @ tg.Function("D_0g", ["k_", "i_", "j_"], (x, "i", "j"), orig_edges_out=["k", "i_", "j_"])
    )


def test_f_2_2_nested():
    x = tg.Variable("x", ["i", "j"])
    g = F.Function("g", ["k", "l"], (x, "i", "j"))
    f = F.Function("f", ["m", "n"], (g, "k", "l"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"m", "n", "i_", "j_"}
    expected = tg.Function("D_0f", ["m", "n", "k_", "l_"], (g, "k", "l")) @ tg.Function(
        "D_0g", ["k_", "l_", "i_", "j_"], (x, "i", "j"), orig_edges_out=["k", "l", "i_", "j_"]
    )
    assert expr == expected


def test_f_2_2_double_nested():
    x = tg.Variable("x", ["i", "j"])
    h = F.Function("h", ["k", "l"], (x, "i", "j"))
    g = F.Function("g", ["m", "n"], (h, "k", "l"))
    f = F.Function("f", ["o", "p"], (g, "m", "n"))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == {"o", "p", "i_", "j_"}
    print(expr)
    assert expr == tg.Product(
        [
            tg.Function("D_0f", ["o", "p", "m", "n"], (g, "m", "n"), orig_edges_out=["o", "p", "m_", "n_"]),
            tg.Function("D_0g", ["m", "n", "k", "l"], (h, "k", "l"), orig_edges_out=["m", "n", "k_", "l_"]),
            tg.Function("D_0h", ["k", "l", "i_", "j_"], (x, "i", "j")),
        ]
    )
