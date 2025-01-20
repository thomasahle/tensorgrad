import pytest
from sympy import symbols

from tensorgrad.tensor import Delta, Variable
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


def test_copy_circle():
    i = symbols("i")
    X = Variable("X", i, j=i)
    assert (
        F.trace(X)
        == F.graph(
            """
        *1 -i- *2
        *2 -i- *3
        *3 -i- X
        X -j- *1
        """,
            X=X,
        ).simplify()
    )


def test_aXXb():
    # Based on https://math.stackexchange.com/questions/4948734
    i = symbols("i")
    a = Variable("a", i)
    b = Variable("b", i)
    X = Variable("X", i, j=i)
    with pytest.raises(ValueError):
        graph = F.graph("a -i- X -j-i- X -j-i- b", a=a, X=X, b=b)
    graph = F.graph("a -i- X0 -j-i- X1 -j-i- b", a=a, X0=X, X1=X, b=b)
    assert graph.simplify() == (((a @ X).rename(j="i") @ X).rename(j="i") @ b).simplify()


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
    expected = (X * Y).rename(i="i0") @ Delta(i, "i0", "i", "j")
    assert (
        expected.simplify()
        == F.graph(
            """
            X -i- * -i-
            Y -i- * -j-
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


def test_matrix_multiplication():
    i, j, k = symbols("i j k")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    expected = A @ B
    assert expected.simplify() == F.graph("A -j- B", A=A, B=B).simplify()


def test_tensor_contraction():
    i, j, k = symbols("i j k")
    T = Variable("T", i, j, k, j)
    expected = F.sum(T, ["j"])
    with pytest.raises(ValueError, match="Cannot have a self loop on a single edge"):
        assert expected.simplify() == F.graph("T -j- T", T=T).simplify()


def test_hadamard_product():
    i, j = symbols("i j")
    A = Variable("A", i, j)
    B = Variable("B", i, j)
    expected = A * B
    assert (
        expected.simplify()
        == F.graph(
            """
        A -i- *0
        B -i- *0
        A -j- *1
        B -j- *1
        *0 -i-
        *1 -j-
    """,
            A=A,
            B=B,
        ).simplify()
    )


def test_outer_product():
    i, j = symbols("i j")
    a = Variable("a", i)
    b = Variable("b", j)
    expected = a @ b
    assert (
        expected.simplify()
        == F.graph(
            """
        a -i-
        b -j-
    """,
            a=a,
            b=b,
        ).simplify()
    )


def test_invalid_edge():
    i, j = symbols("i j")
    A = Variable("A", i, j)
    with pytest.raises(ValueError, match="Edge k not found in variable A"):
        F.graph("A -k- B", A=A)


def test_invalid_variable():
    i, j = symbols("i j")
    A = Variable("A", i, j)
    with pytest.raises(ValueError, match="Variable B not found in vars"):
        F.graph("A -i- B", A=A)


def test_self_loop():
    i = symbols("i")
    A = Variable("A", i)
    with pytest.raises(ValueError, match="Cannot have a self loop on a single edge"):
        F.graph("A -i-i- A", A=A)


def test_complex_tensor_network():
    i, j, k, l = symbols("i j k l")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    C = Variable("C", k, l)
    D = Variable("D", l, i)
    expected = A @ B @ C @ D
    assert (
        expected.simplify()
        == F.graph(
            """
        A -i- D
        A -j- B
        B -k- C
        C -l- D
    """,
            A=A,
            B=B,
            C=C,
            D=D,
        ).simplify()
    )


def test_hyperedge_copy():
    i, j = symbols("i j")
    A = Variable("A", i)
    B = Variable("B", j)
    expected = (A @ B) * (A @ B)
    assert (
        expected.simplify()
        == F.graph(
            """
        A0 -i- *0
        B0 -j- *1
        A1 -i- *0
        B1 -j- *1
        *0 -i-
        *1 -j-
    """,
            A0=A,
            A1=A,
            B0=B,
            B1=B,
        ).simplify()
    )


def test_quantum_circuit_simulation():
    #         ┌───┐ ┌─────────┐ ┌───┐ ┌───┐
    # |ψ⟩ ────┤ H ├─┤         ├─┤ R ├─┤ M ├─ (i)
    #         └───┘ │  CNOT   │ └───┘ └───┘
    #               │         │       ┌───┐
    # |ψ⟩ ──────────┤         ├───────┤ M ├─ (j)
    #               └─────────┘       └───┘

    i = symbols("i")
    # Define quantum gates and states
    H = Variable("H", i, j=i)  # Hadamard gate
    CNOT = Variable("CNOT", i1=i, i2=i, o1=i, o2=i)  # Controlled-NOT gate
    R = Variable("R", i, j=i)  # Rotation gate
    psi = Variable("psi", i)  # Initial state
    measure = Variable("M", i, j=i)  # Measurement operator

    expected = (
        (
            (psi @ H).rename(j="i1")  # First input rotated
            @ psi.rename(i="i2")  # Second input not rotated
            @ CNOT  # Apply CNOT to both qubits
        )
        @ (
            R.rename(i="o1", j="i")  # Apply R to first qubit
            @ measure.rename(j="m1")  # Measure first qubit
        )
        @ measure.rename(i="o2", j="m2")  # Measure second qubits
    ).rename(m1="o1", m2="o2")

    assert (
        expected.simplify()
        == F.graph(
            """
        psi0 -i- H -j-i1- CNOT -o1-i- R -j-i- M0 -j-o1-
        psi1    -i-i2-    CNOT -o2-i- M1 -j-o2-
    """,
            psi0=psi,
            psi1=psi,
            H=H,
            CNOT=CNOT,
            R=R,
            M0=measure,
            M1=measure,
        ).simplify()
    )


def test_broadcast1():
    b, i = symbols("b i")
    A = Variable("A", i)
    B = Variable("B", b, i)
    expected = F.dot(A, B, ["i"])
    assert expected.simplify() == F.graph("A -i- B", A=A, B=B).simplify()


def test_broadcast2():
    b, i = symbols("b i")
    A = Variable("A", b, i)
    B = Variable("B", b, i)
    expected = F.dot(A, B, ["i"])
    assert expected.simplify() == F.graph("A -i- B", A=A, B=B).simplify()


def test_broadcast3():
    b, i = symbols("b i")
    A = Variable("A", b, i)
    B = Variable("B", b, i)
    expected = F.dot(A, B, ["i"])
    assert (
        expected.simplify()
        == F.graph(
            """
            A -i- B
            A -b-
            B -b-
            """,
            A=A,
            B=B,
        ).simplify()
    )


def test_broadcast4():
    b, i = symbols("b i")
    A = Variable("A", b, i)
    B = Variable("B", b, i)
    expected = F.dot(A, B, ["i"])
    assert (
        expected.simplify()
        == F.graph(
            """
            A -i- B
            A -b- *0
            B -b- *0
            *0 -b-
            """,
            A=A,
            B=B,
        ).simplify()
    )
