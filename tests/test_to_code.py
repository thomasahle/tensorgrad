import pytest
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad.tensor import Variable, Zero, Delta, Sum, Derivative, Ones
from tensorgrad.serializers.to_pytorch import compile_to_callable
from tensorgrad.testutils import assert_close, rand_values


@pytest.mark.parametrize("compile", [False, True])
def test_codegen_zero(compile):
    """Test generation of a Zero tensor code."""
    i, j = symbols("i j")
    zero_tensor = Zero(i, j)

    # Compile
    compiled_fn = compile_to_callable(zero_tensor, verbose=False, torch_compile=compile)

    # Evaluate reference
    dims = {i: 2, j: 3}
    ref = zero_tensor.evaluate({}, dims)
    # Evaluate compiled
    out = compiled_fn({}, dims)[zero_tensor]
    assert_close(ref, out)


@pytest.mark.parametrize("compile", [False, True])
def test_codegen_copy(compile):
    """Test generation of a Copy (identity) tensor code."""
    i = symbols("i")
    copy_tensor = Delta(i, "i, j")

    # Compile
    compiled_fn = compile_to_callable(copy_tensor, verbose=False, torch_compile=compile)

    # Evaluate reference
    dims = {i: 3}
    ref = copy_tensor.evaluate({}, dims)
    # Evaluate compiled
    out = compiled_fn({}, dims)[copy_tensor]
    assert_close(ref, out)


@pytest.mark.parametrize("compile", [False, True])
def test_codegen_sum(compile):
    """Test generation of a Sum tensor code."""
    i, j = symbols("i j")
    x = Variable("x", i, j)
    y = Variable("y", i, j)

    expr = Sum([x, y], weights=[2, -1])

    # Compile
    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=compile)

    # Evaluate reference
    dims = {i: 4, j: 5}
    vals = rand_values([x, y], dims)
    ref = expr.evaluate(vals, dims)
    # Evaluate compiled
    out = compiled_fn(vals, dims)[expr]
    assert_close(ref, out)


@pytest.mark.parametrize("compile", [False, True])
def test_codegen_product(compile):
    """Test generation of a Product tensor code."""
    i, j, k = symbols("i j k")
    # a has shape (i, j), b has shape (j, k)
    a = Variable("a", i, j)
    b = Variable("b", j, k)
    expr = a @ b

    # Compile
    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=compile)

    # Evaluate reference
    dims = {i: 2, j: 3, k: 4}
    vals = rand_values([a, b], dims)
    ref = expr.evaluate(vals, dims)
    # Evaluate compiled
    out = compiled_fn(vals, dims)[expr]
    assert_close(ref, out)


def test_codegen_simple_function():
    """Test generation of a simple function like ReLU."""
    i = symbols("i")
    x = Variable("x", i)
    expr = F.relu(x)  # edges = [i]

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 5}
    vals = rand_values([x], dims)
    ref = expr.evaluate(vals, dims)
    out = compiled_fn(vals, dims)[expr]
    assert_close(ref, out)


def test_codegen_argmax():
    """Test generation of the argmax function."""
    i, j = symbols("i j")
    x = Variable("x", i, j)
    # Argmax over j => shape is (i,)
    expr = F.argmax(x, dim="j")

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 4, j: 3}
    vals = rand_values([x], dims)
    ref = expr.evaluate(vals, dims)
    out = compiled_fn(vals, dims)[expr]
    assert (ref == out).all()


def test_codegen_power():
    """Test generation of a power function (x^2)."""
    i = symbols("i")
    x = Variable("x", i)
    expr = F.pow(x, 2)

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 6}
    vals = rand_values([x], dims)
    ref = expr.evaluate(vals, dims)
    out = compiled_fn(vals, dims)[expr]
    assert_close(ref, out)


def test_codegen_derivative_placeholder():
    """
    Test generation of a Derivative object.
    The current code generation stub returns a zero tensor,
    but we verify it compiles and has the correct shape.
    """
    i = symbols("i")
    x = Variable("x", i)
    expr = Derivative(x, x).simplify()  # partial derivative w.r.t. x

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 5}
    vals = rand_values([x], dims)
    ref = expr.evaluate(vals, dims)  # We expect a zero of shape (i, i_) in the current stub
    out = compiled_fn(vals, dims)[expr]
    assert_close(ref, out)


def test_codegen_full_expression():
    """
    Test a more involved expression combining Sum, Product, ReLU, etc.
    """
    i, j, k = symbols("i j k")
    x = Variable("x", i, j)
    w = Variable("w", j, k)
    b = Variable("b", i, k)

    expr = F.relu(x @ w + b)  # shape (i, k)

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 2, j: 3, k: 4}
    vals = rand_values([x, w, b], dims)
    ref = expr.evaluate(vals, dims)
    out = compiled_fn(vals, dims)[expr]
    assert_close(ref, out)


def test_codegen_ones():
    """Test generation of an expression that includes a constant ones tensor."""
    i, j = symbols("i j")
    # We'll create a small expression: x + 2 * (Ones(i,j))
    x = Variable("x", i, j)
    ones_expr = Ones(i, j)  # This is not an explicit class in the posted code, but let's assume
    # we can test something similar or your own internal "Ones" approach
    # If not, use e.g. Zero() or Copy() for demonstration.
    expr = x + 2 * ones_expr

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 3, j: 2}
    vals = rand_values([x], dims)
    ref = expr.evaluate(vals, dims)
    out = compiled_fn(vals, dims)[expr]
    assert_close(ref, out)
