from tensorgrad import Variable
import tensorgrad.functions as F
from sympy import symbols

# Various tests from matrix calculus


def test_gradient_of_product():
    # Testing: ∇(Bx + b)^T C(Dx + d) = B^T C(Dx + d) + D^T C^T (Bx + b)
    x, b, d = symbols("x b d")
    B = Variable("B", x=x, b=b)
    C = Variable("C", b=b, d=d)
    D = Variable("D", x=x, d=d)
    x = Variable("x", x=x)
    b = Variable("b", b=b)
    d = Variable("d", d=d)

    expr = (B @ x + b) @ C @ (D @ x + d)
    gradient = expr.grad(x)

    expected = B @ C @ (D @ x + d) + D @ C @ (B @ x + b)

    assert gradient.simplify() == expected.simplify()


def test_gradient_of_quadratic_form():
    # Testing: ∂b^T X^T X c / ∂X = X(bc^T + cb^T)
    i, j = symbols("i j")
    X = Variable("X", i, j)
    b = Variable("b", i)
    c = Variable("c", i)

    expr = (b @ X) @ (X @ c)
    gradient = expr.grad(X)
    gradient = gradient.simplify({"expand": True})

    expected = X @ F.symmetrize(b @ c.rename(i="i_"))

    # The issue is: We don't factor things, so we end up with $X b c^T + X c b^T$
    # for the gradient. So we'll just expand the expected value too.
    expected = expected.simplify({"expand": True})

    assert gradient == expected


def test_gradient_of_matrix_trace():
    # Testing: ∂ tr(BA) / ∂A = B^T
    i, j = symbols("i j")
    A = Variable("A", i, j)
    B = Variable("B", i, j)

    # We don't really need F.trace here, since B @ A already connects both edges.
    # But it's a good test of what happens when F.trace gets a scalar.
    expr = F.trace(B @ A)
    gradient = expr.grad(A)
    assert gradient.simplify() == B


def test_gradient_of_quadratic_form_with_middle_matrix():
    # Testing: ∂b^T X^T D X c / ∂X = D^T X bc^T + D X cb^T
    i, j = symbols("i j")
    X = Variable("X", i, j)
    D = Variable("D", j1=j, j2=j)
    b = Variable("b", i)
    c = Variable("c", i)

    expr = b @ X.rename(j="j1") @ D @ X.rename(j="j2") @ c
    gradient = expr.grad(X)

    expected = D.rename(j1="j", j2="j'") @ X @ (b @ c.rename(i="i_"))
    expected += D.rename(j2="j", j1="j'") @ X @ (c @ b.rename(i="i_"))

    assert gradient.full_simplify() == expected.full_simplify()


def test_gradient_of_quadratic_form_with_affine_transform():
    # Testing: ∂(Xb + c)^T D(Xb + c) / ∂X = (D + D^T)(Xb + c)b^T
    i, j = symbols("i j")
    X = Variable("X", i, j)
    D = Variable("D", j=j, j2=j)
    b = Variable("b", i)
    c = Variable("c", j)

    expr = (X @ b + c) @ D @ (X @ b + c).rename(j="j2")
    gradient = expr.grad(X)

    expected = F.symmetrize(D) @ (X @ b + c) @ b

    assert gradient.simplify({"expand": True}) == expected.simplify({"expand": True})
