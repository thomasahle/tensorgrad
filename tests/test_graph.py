from sympy import symbols

from tensorgrad.tensor import Copy, Variable
import tensorgrad.functions as F


def test_trace():
    i = symbols("i")
    X = Variable("X", i, j=i)
    assert F.trace(X) == F.graph("X -i-j- X", X=X)


def test_self_edges():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    expr = F.sum(X, ["i", "j"])
    assert expr.simplify() == F.graph("X -i- *0; X -j- *1", X=X).simplify()


def test_frobenius():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    assert F.frobenius2(X) == F.graph("X -i- X1 -j- X", X=X, X1=X)


def _test_group():
    i = symbols("i")
    X = Variable("X", i)
    Y = Variable("Y", i)
    # Graphviz also supports grouping, so we could have a graph like this:
    # But let's wait before trying to implement stuff like that.
    F.graph("""
        {X Y} -i- * -i-
    """)


def test_XAXBXCX():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    A = Variable("A", j, j1=j)
    B = Variable("B", i, i1=i)
    C = Variable("C", j, j1=j)
    XAXBXCX = X.rename(i="i0") @ A @ X.rename(j="j1") @ B @ X.rename(i="i1") @ C @ X.rename(j="j1")
    XAXBXCX = XAXBXCX.simplify()

    # DOT like graph syntax, inline
    assert (
        XAXBXCX
        == F.graph(
            "-i0-i- X0 -j- A -j1-j- X1 -i- B -i1-i- X2 -j- C -j1-j- X3", X0=X, X1=X, X2=X, X3=X, A=A, B=B, C=C
        ).simplify()
    )

    # DOT like graph syntax, multi-line
    assert (
        XAXBXCX
        == F.graph(
            """
        -i0-i- X0
        X0 -j- A
        A -j1-j- X1
        X1 -i- B
        B -i1-i- X2
        X2 -j- C
        C -j1-j- X3
    """,
            X0=X,
            X1=X,
            X2=X,
            X3=X,
            A=A,
            B=B,
            C=C,
        ).simplify()
    )


def test_copy():
    # Maybe we can do copy tensors like this, using *i as their name.
    # Note, we don't care about the edge names of copy tensors, so we always have
    # just a single edge name.
    i = symbols("i")
    X = Variable("X", i)
    Y = Variable("Y", i)
    expected = (X * Y).rename(i="i0") @ Copy(i, "i0", "i", "j")
    assert (
        expected.simplify()
        == F.graph(
            """
        X -i- *
        Y -i- *
        * -i-
        * -j-
    """,
            X=X,
            Y=Y,
        ).simplify()
    )


def test_ST2_graph():
    i, j = symbols("i j")
    S = Variable("S", i)
    T = Variable("T", i, j)
    ST = S @ T
    expected = ST * ST
    assert (
        expected.simplify()
        == F.graph(
            """
        S0 -i- T0 -j- *0
        S1 -i- T1 -j- *0
        *0 -j-
    """,
            S0=S,
            S1=S,
            T0=T,
            T1=T,
        ).simplify()
    )


def test_G_graph():
    a, b, c = symbols("a b c")
    S = Variable("S", a)
    U = Variable("U", a)
    G = Variable("G", a, b, c)
    V = Variable("V", c)
    expected = ((S * U) @ F.sum(G * G, ["b"]) @ V).full_simplify()
    expr = F.graph(
        """
        S -a- *0 -a- U
        G0 -a- *0 -a- G1
        G0 -b- G1
        G0 -c- *1 -c- G1
        *1 -c- V
    """,
        S=S,
        U=U,
        V=V,
        G0=G,
        G1=G,
    ).full_simplify()
    assert expected == expr
