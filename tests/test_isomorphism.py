from sympy import symbols
from tensorgrad import Variable
from tensorgrad.tensor import Delta, Product, Sum
import tensorgrad.functions as F


def test_simple():
    i = symbols("i")
    A = Variable("A", i, j=i)
    assert A.symmetries == {frozenset({"i"}), frozenset({"j"})}
    # Note: F.symmetrize(A) may not have the right .symmetries before it has been simplified.
    # This is because structure in how it is created break the symmetries.
    ApAt = F.symmetrize(A).simplify()
    assert ApAt.symmetries == {frozenset({"i", "j"})}
    assert ApAt.simplify() != (2 * A).simplify()

    Asym = Variable("A", i, j=i).with_symmetries("i j")
    assert A != Asym
    assert F.symmetrize(Asym).simplify() == (2 * Asym).simplify()


def test_simple2():
    x, y = symbols("x y")
    A = Variable("A", x, y)
    B = A.rename(x="x2", y="y2")
    (mapping,) = A.isomorphisms(B)
    assert mapping == {"x": "x2", "y": "y2"}
    (mapping,) = B.isomorphisms(A)
    assert mapping == {"x2": "x", "y2": "y"}


def test_simple3():
    i = symbols("i")
    A2 = Variable("A", x=i, y=i)
    A1 = A2.rename(x="x__", y="x_")
    A0 = Variable("A", x__=i, x_=i)
    assert A0 != A1, "A0 and A1 are not isomorphic, because their original names are different."
    () = A2.isomorphisms(A0)


def test_hash_counterexample():
    i, j, k = symbols("i j k")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    C = Variable("C", k, i)

    x = (A @ B @ C) @ (A @ B @ C)
    y = A @ B @ C.rename(i="i2") @ A.rename(i="i2") @ B @ C
    x = x.simplify()
    y = y.simplify()

    assert x != y


def test_hash_counterexample2():
    i, j, k, l = symbols("i j k l")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    D = Variable("D", k=k, k2=k, l=l)
    A2 = A.rename(i="i2")
    B2 = B.rename(k="k2")
    half1 = A @ B @ D @ B2 @ A2
    expr1 = half1 @ half1
    half2 = A @ B @ D @ B2 @ A
    expr2 = half2 @ half2

    assert expr1 != expr2

    expr1.structural_graph()


def test_6cycle_vs_two_3cycles():
    i, j, k, i2, j2, k2 = symbols("i j k i2 j2 k2")
    one_2cycle = Product(
        [
            Variable("A", i, j),
            Variable("B", j, k),
            Variable("C", k, i2=i),
            Variable("D", i2=i, j2=j),
            Variable("E", j2=j, k2=k),
            Variable("F", k2=k, i=i),
        ]
    )
    two_3cycles = Product(
        [
            Variable("A", i, j),
            Variable("B", j, k),
            Variable("C", k, i),
            Variable("D", i2=i, j2=j),
            Variable("E", j2=j, k2=k),
            Variable("F", k2=k, i2=i),
        ]
    )

    assert one_2cycle != two_3cycles


def test_4cycle_vs_two_2cycles():
    i, j, k, l = symbols("i j k l")
    A = Variable("A", i, j)
    B = Variable("B", j, k=i)
    C = Variable("C", k=i, l=l)
    D = Variable("D", l, i)
    one_4cycle = Product([A, B, C, D])
    two_2cycles = Product([A, B.rename(k="i"), C, D.rename(i="k")])
    assert one_4cycle != two_2cycles


def test_symmetry():
    i = symbols("i")
    x = Variable("x", i)
    expr = x @ x.rename(i="j")
    assert expr.edges == {"i", "j"}
    assert expr.symmetries == {frozenset({"i", "j"})}

    expr = x @ Delta(i, "i, j, k")
    assert expr.edges == {"j", "k"}
    assert expr.symmetries == {frozenset({"j", "k"})}


def test_example_from_softmax_hessian():
    i = symbols("i")
    A = Variable("A", i)
    B = Variable("B", l=i)

    # The two a are isomorphic, but they can't be added / subtracted, because the edges are different.
    #
    #    A   B       A   B
    #    |   ⅄   vs  |   ⅄
    #    i  j k      j  i k
    #
    graph = A @ B @ Delta(i, "l, j, k")
    graph2 = graph.rename(i="j", j="i")
    assert graph == graph2

    # These two are isomorphic, but they _can_ be added because of the symmetry.
    #
    #    A   B       A   B
    #    |   ⅄   vs  |   ⅄
    #    i  j k      i  k j
    #
    graph3 = graph.rename(j="k", k="j")
    assert graph == graph3

    # The point is that when we include the graphs in bigger context, the (lack of) symmetry matters.
    variables = Product(
        [
            Variable("x", i),
            Variable("y", j=i),
            Variable("z", k=i),
        ]
    )

    assert graph @ variables != graph2 @ variables
    assert graph @ variables == graph3 @ variables


def test_symmetries():
    i = symbols("i")
    A = Variable("A", i)
    B = Variable("B", l=i)

    #    A   B
    #    |   ⅄
    #    i  j k
    graph = A @ B @ Delta(i, "l, j, k")
    assert graph.symmetries == {frozenset("i"), frozenset("jk")}

    #    A   B      A   B
    #    |   ⅄   +  |   ⅄
    #    i  j k     i  k j
    graph2 = graph + graph.rename(j="k", k="j")
    assert graph2.symmetries == {frozenset("i"), frozenset("jk")}

    #    A   B      A   B
    #    |   ⅄   +  |   ⅄
    #    i  j k     k  j i
    graph3 = graph + graph.rename(i="k", k="i")
    assert graph3.symmetries == {frozenset("ik"), frozenset("j")}

    #    A   B      A   B      A   B
    #    |   ⅄   +  |   ⅄   +  |   ⅄
    #    i  j k     j  i k     k  j i
    graph4 = Sum([graph, graph.rename(i="j", j="i"), graph.rename(i="k", k="i")])
    assert graph4.symmetries == {frozenset("ijk")}


def test_symmetries_simplify_sum():
    i = symbols("i")

    #    A   B
    #    |   ⅄
    #    i  j k
    A = Variable("A", i)
    B = Variable("B", l=i)
    graph1 = A @ B @ Delta(i, "l, j, k")

    #    A   B
    #    |   ⅄
    #    i  k j
    graph2 = graph1.rename(j="k", k="j")

    # Flipping j and k we can still combine the two graphs.
    assert len((graph1 + graph2).simplify().tensors) == 1

    #    A   B
    #    |   ⅄
    #    k  j i
    graph3 = graph1.rename(i="k", k="i")

    # Flipping i and k we can't combine the two graphs.
    assert len((graph1 + graph3).simplify().tensors) == 2

    #    A   B
    #    |   ⅄
    #    m  j k
    graph4 = graph1.rename(i="m")

    # Even if graph1 and graph4 are isomorphic, we can't combine them.
    assert len((graph1 + graph4).simplify().tensors) == 2


def test_empty_graphs():
    empty_graph1 = Product([])
    empty_graph2 = Product([])
    assert empty_graph1 == empty_graph2


def test_single_variable_graphs():
    i, j = symbols("i j")
    single_var1 = Product([Variable("A", i, j)])
    single_var2 = Product([Variable("B", i, j)])
    assert single_var1 != single_var2

    A = Variable("A", i, j)
    single_var1 = Product([A])
    single_var2 = Product([A.rename(i="i2")])
    assert single_var1 == single_var2

    single_var3 = Product([A.rename(i="j", j="i")])
    assert single_var1 == single_var3


def test_same_variable_names():
    i, j, k = symbols("i j k")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    C = Variable("C", k, i)
    graph1 = Product([A, B, C])
    graph2 = Product(
        [
            A.rename(i="i2", j="j2"),
            B.rename(k="k2", j="j2"),
            C.rename(i="i2", k="k2"),
        ]
    )
    assert graph1 == graph2


def test_3cycle_with_selfloop():
    i, j, k, l, m = symbols("i j k l m")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    C = Variable("C", k, i)
    D = Variable("D", l, m)
    graph1 = Product([A, B, C, D])
    graph2 = Product([A, B, C, D.rename(l="n")])
    assert graph1 == graph2


def test_disconnected_graphs():
    i, j, k, l, m, n = symbols("i j k l m n")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    C = Variable("C", k, i)
    D = Variable("D", l, m)
    E = Variable("E", m, n)
    F = Variable("F", n, l)
    graph1 = Product([A, B, C, D, E, F])
    graph2 = Product(
        [
            A,
            B,
            C,
            D.rename(l="l2", m="m2"),
            E.rename(m="m2", n="n2"),
            F.rename(n="n2", l="l2"),
        ]
    )
    assert graph1 == graph2


def test_different_number_of_variables():
    i = symbols("i")
    A = Variable("A", i=i, j=i)
    B = Variable("B", j=i, k=i)
    C = Variable("C", k=i, i=i)
    D = Variable("D", l=i, m=i)
    graph1 = Product([A, B, C])
    graph2 = Product([A, B, C, D])
    assert graph1 != graph2


def test_isomorphic_with_rename():
    i, j, k = symbols("i j k")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    C = Variable("C", k, i)
    graph1 = Product([A, B, C])
    graph2 = Product(
        [
            A.rename(i="x", j="y"),
            B.rename(j="y", k="z"),
            C.rename(k="z", i="x"),
        ]
    )
    assert graph1 == graph2


def test_different_variables():
    i = symbols("i")
    e1 = Product([Variable("a", i), Delta(i, "j, j1")])
    e2 = Product([Variable("b", i), Delta(i, "j, j1")])
    assert e1 != e2
    e1 = Product([Variable("a", i), Delta(i, "i, j, j1")])
    e2 = Product([Variable("b", i), Delta(i, "i, j, j1")])
    assert e1 != e2
    a = Variable("a", i, j=i)
    b = Variable("b", i, j=i)
    e1 = a + b
    e2 = b + b
    assert e1 != e2


def test_broadcasted():
    x_, y_ = symbols("x_ y_")
    e1 = Product(
        [
            Delta(y_),
            Sum(
                [
                    Product([Variable("x", x_), Delta(y_)]),
                    Product([Variable("y", y_), Delta(x_)]),
                ],
                (1, 1),
            ),
        ]
    )
    e2 = Product(
        [
            Sum(
                [
                    Product([Variable("x", x_), Delta(y_)]),
                    Product([Variable("y", y_), Delta(x_)]),
                ],
                (1, 1),
            ),
            Delta(y_),
        ]
    )
    assert e1 == e2


def test_transpose_grad():
    i, j = symbols("i j")
    x = Variable("X", i, j)
    xt = x.rename(j="i", i="j")
    res = xt.grad(x, {"i": "a", "j": "b"}).simplify()
    expected = Delta(j, "j, a") @ Delta(i, "i, b")
    assert expected.symmetries == {frozenset({"a", "j"}), frozenset({"b", "i"})}
    assert not res.is_isomorphic(expected, match_edges=True)
    assert res.is_isomorphic(expected, match_edges=False)


def test_symmetries_simplify_sum2():
    C = symbols("C")
    logits = Variable("logits", C)
    expr = Sum(
        [
            Product(
                [
                    logits.rename(C="C__"),
                    logits.rename(C="C_"),
                ]
            ),
            Product(
                [
                    logits.rename(C="C_"),
                    logits.rename(C="C__"),
                ]
            ),
        ],
    )
    assert sorted(expr.tensors[0].edges) == sorted(expr.tensors[1].edges)
    assert len({hash(t) for t in expr.tensors}) == 1
    assert expr.symmetries == {frozenset({"C_", "C__"})}
    for t in expr.tensors:
        assert t.symmetries == {frozenset({"C_", "C__"})}
    assert len(expr.simplify().tensors) == 1


def _test_links():
    i, j = symbols("i j")
    x = Variable("x", i)
    y = Variable("y", j)
    xy = x @ y
    trans = {j: i, i: j}
    xt = x.rename(**trans)
    yt = y.rename(**trans)
    yx = yt @ xt
    I1 = xy.grad(x).grad(y).simplify()
    I2 = yx.grad(xt).grad(yt).simplify()
    assert not I1.is_isomorphic(I2, match_edges=True)


def test_copy():
    # Copy tensors mostly combine, but order 0 copy tensors don't.
    p = symbols("p")
    assert Delta(p, "p") != Delta(p)
    twop = Delta(p, "p") @ Delta(p, "p")
    onep = Delta(p) @ Delta(p)
    assert twop != onep
    assert twop.simplify() != onep.simplify()


def test_copy3():
    a, b, c = symbols("a b c")
    prod = Product([Delta(a, "a"), Delta(b, "b"), Delta(c, "c")])
    assert prod.symmetries == {frozenset({"a"}), frozenset({"b"}), frozenset({"c"})}


def test_copy0():
    i, j = symbols("i j")
    assert Delta(i) != Delta(j)
