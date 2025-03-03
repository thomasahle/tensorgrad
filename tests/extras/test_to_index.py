from sympy import symbols

from tensorgrad import Variable, Product, Function, Zero, Delta, Derivative, Expectation
from tensorgrad import functions as F
from tensorgrad.extras.to_index import to_index, to_index_free


def test_cycles():
    i = symbols("i")
    M = Variable("M", i, j=i)
    for k in range(2, 5):
        # Using pow function
        Mk = F.pow(M, k)
        assert isinstance(Mk, Function)
        assert to_index_free(Mk) == f"(M)^{{{k}}}"
        assert to_index(Mk) == f"(M_{{i,j}})^{{{k}}}"
        assert to_index_free(F.trace(Mk)) == f"tr((M)^{{{k}}})"

        # Using a product
        Mk2 = F.multi_dot([M] * k, ("i", "j")).simplify()
        assert isinstance(Mk2, Product)

        # Paths are annoying because we don't have a good way to decide transpose.
        # We just assume that the order in Mk2.edges is meaningful.
        if list(Mk2.edges) == ["i", "j"]:
            assert to_index_free(Mk2) == " ".join("M" * k)
        else:
            assert to_index_free(Mk2) == " ".join(("M^T",) * k)

        trMk2 = F.trace(Mk2).simplify()
        assert to_index_free(trMk2) == f"tr({' '.join('M' * k)})"


def test_variable_and_rename():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    Y = X.rename(i="a", j="b")

    assert to_index(X) == "X_{i,j}"
    assert to_index(Y) == "X_{a,b}"
    assert to_index_free(X) == "X"
    assert to_index_free(Y) == "X"


def test_zero_and_delta():
    i, j = symbols("i j")

    Z = Zero(i, j)
    assert to_index(Z) == "0_{i,j}"
    assert to_index_free(Z) == "0"

    delta = Delta(i, "i, j")
    assert to_index(delta) == "δ_{i,j}"
    assert to_index_free(delta) == "I"


def test_sum():
    i, j = symbols("i j")
    X = Variable("X", i)
    Y = Variable("Y", i)

    expr = X + 2 * Y
    assert to_index(expr) == "X_{i} + 2 Y_{i}"
    assert to_index_free(expr) == "X + 2 Y"


def test_derivative():
    i = symbols("i")
    X = Variable("X", i)
    expr = Derivative(X, X)

    assert to_index(expr) == "d(X_{i})/d(X_{i})"
    assert to_index_free(expr) == "d(X)/d(X)"


def test_function():
    i = symbols("i")
    X = Variable("X", i)

    square_X = F.pow(X, 2)
    sin_X = F.exp(X)

    assert to_index(square_X) == "(X_{i})^{2}"
    assert to_index(sin_X) == "exp(X_{i})"

    assert to_index_free(square_X) == "(X)^{2}"
    assert to_index_free(sin_X) == "exp(X)"


def test_expectation():
    i = symbols("i")
    X = Variable("X", i)
    expectation = Expectation(X, wrt=X)

    assert to_index(expectation) == "E_X[X_{i}]"
    assert to_index_free(expectation) == "E_X[X]"


def test_self_product():
    i, j = symbols("i j")
    A = Variable("A", i, j)
    prod = A @ A

    assert to_index(prod) == "A_{i,j} A_{i,j}"
    assert to_index_free(prod) == "tr(A A^T)"


def test_higher_order_trace():
    i, j, k = symbols("i j k")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    C = Variable("C", k, i)

    trace_ABC = F.trace(A @ B @ C).simplify()
    assert to_index_free(trace_ABC) == "tr(A B C)"


def test_complex_expression():
    i, j, k = symbols("i j k")
    X = Variable("X", i)
    Y = Variable("Y", i)
    Z = Zero(i)

    expr = (2 * X + -3 * Y + Z).simplify()
    assert to_index(expr) == "2 X_{i} - 3 Y_{i}"
    assert to_index_free(expr) == "2 X - 3 Y"


def test_variable_no_edges():
    v = Variable("x")
    assert to_index(v) == "x"
    assert to_index_free(v) == "x"


def test_MT_string():
    i = symbols("i")
    M = Variable("M", i, j=i)
    T = M.rename(i="j", j="i")
    A = F.trace(F.multi_dot([M, M, T, M, T, T], ("i", "j"))).simplify()
    B = F.trace(F.multi_dot([M, M, T, T, M, T], ("i", "j"))).simplify()
    assert A != B
    assert to_index_free(A) == "tr(M M M^T M M^T M^T)"
    assert to_index_free(B) == "tr(M M M^T M^T M M^T)"


def test_inner_product():
    # Disabled: We don't have a way to handle this right now
    i = symbols("i")
    v1 = Variable("x", i)
    v2 = Variable("y", i)
    p = v1 @ v2
    assert to_index(p) == "x_{i} y_{i}"
    assert to_index_free(p) == "x^T y"


def _test_outer_product():
    # Disabled: We don't have a way to handle this right now
    i, j = symbols("i j")
    v1 = Variable("x", i)
    v2 = Variable("y", j)
    p = v1 @ v2
    assert to_index(p) == "x_{i} y_{j}"
    assert to_index_free(p) == "x y^T"


def test_mixed_product():
    # Product of tensors that do not share edges.
    i, j = symbols("i j")
    v1 = Variable("x", i)
    v2 = Variable("y", j)
    A = Variable("A", i, j)

    p = (v1 @ A @ v2).simplify()
    assert to_index(p) == "x_{i} A_{i,j} y_{j}"
    assert to_index_free(p) == "x^T A y"

    p = (v1 @ A).simplify()
    assert to_index(p) == "x_{i} A_{i,j}"
    assert to_index_free(p) == "x^T A"

    p = (A @ v2).simplify()
    assert to_index(p) == "A_{i,j} y_{j}"
    assert to_index_free(p) == "A y"


def test_lexicographical_minimal_rotation_bug():
    i, j, y = symbols("i j y")
    # Construct three matrices:
    # A has edges (i, j) -> if used with in/out (j, i) then its marker is T.
    A = Variable("A", i, j)
    # B has edges (i, y) -> used with (i, y) gives marker M.
    B = Variable("B", i, y)
    # C has edges (y, j) -> used with (y, j) gives marker M.
    C = Variable("C", y, j)
    # Manually set the cycle's out_edges so that:
    #   out_edges = [i, y, j] and hence in_edges = [j, i, y]
    # This forces:
    #   For A: (in, out) = (j, i) which is reversed (T).
    #   For B: (in, out) = (i, y) which matches (M).
    #   For C: (in, out) = (y, j) which matches (M).
    # out_edges = [i, y, j]
    # result = _handle_trace([A, B, C], out_edges)
    result = to_index_free(Product([A, B, C]))
    expected = "tr(B C A^T)"  # The optimal rotation rotates the pair list to [B, C, A].
    assert result == expected, f"Expected {expected}, got {result}"
