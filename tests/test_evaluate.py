from typing import Iterable
import torch
from tensor import Copy, Derivative, Function, Ones, Product, Sum, Tensor, Variable, Zero
import functions as F
from utils import assert_close, rand_values


def test_copy():
    copy_tensor = Copy(["i", "j"])
    result = copy_tensor.evaluate({}, dims={"i": 3, "j": 3})
    expected = torch.eye(3).rename("i", "j")
    assert_close(result, expected)


def test_zero():
    zero_tensor = Zero(["i", "j"])
    result = zero_tensor.evaluate({}, dims={"i": 2, "j": 3})
    expected = torch.zeros(2, 3).rename("i", "j")
    assert_close(result, expected)


def test_ones():
    ones_tensor = Ones(["i", "j"])
    result = ones_tensor.evaluate({}, dims={"i": 2, "j": 3})
    expected = torch.ones(2, 3).rename("i", "j")
    assert_close(result, expected)


def test_product():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    product_tensor = a @ b
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    result = product_tensor.evaluate({a: t_a, b: t_b})
    expected = t_a @ t_b
    assert_close(result, expected)


def test_sum_tensor():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["i", "j"])
    sum_tensor = Sum([a, b], weights=[2, 3])
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(2, 3, names=("i", "j"))
    result = sum_tensor.evaluate({a: t_a, b: t_b})
    expected = 2 * t_a + 3 * t_b
    assert_close(result, expected)


def test_derivative():
    a = Variable("a", ["i"])
    b = Variable("b", ["i"])
    product_tensor = Product([a, b])
    derivative_tensor = Derivative(product_tensor, a, ["j"])
    t_a = torch.randn(3, names=("i",))
    t_b = torch.randn(3, names=("i",))
    result = derivative_tensor.simplify().evaluate({a: t_a, b: t_b})
    expected = t_b.rename("j")
    assert_close(result, expected)


def test_rename():
    a = Variable("a", ["i", "j"])
    renamed_tensor = a.rename({"i": "k", "j": "l"})
    t_a = torch.randn(2, 3, names=("i", "j"))
    result = renamed_tensor.evaluate({a: t_a})
    expected = t_a.rename("k", "l")
    torch.testing.assert_close(result.rename(None), expected.rename(None))


def test_nested_product_and_sum():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    c = Variable("c", ["i", "k"])
    d = Variable("d", ["i", "m"])

    expr = (a @ b + c) @ d

    # Create random input tensors
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    t_c = torch.randn(2, 4, names=("i", "k"))
    t_d = torch.randn(2, 6, names=("i", "m"))

    # Evaluate the tensor expression
    result = expr.evaluate({a: t_a, b: t_b, c: t_c, d: t_d})
    print((t_a @ t_b + t_c))
    print(t_d)
    # Compare with the expected result
    expected = (t_a @ t_b + t_c).transpose("k", "i") @ t_d
    assert_close(result, expected)


def test_derivative_of_product():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    c = Variable("c", ["k", "l"])

    # Create a tensor expression: a @ b @ c
    expr = Product([a, b, c])

    # Take the derivative with respect to b
    derivative_expr = Derivative(expr, b)

    # Create random input tensors
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    t_c = torch.randn(4, 5, names=("k", "l"))

    # Evaluate the derivative expression
    result = derivative_expr.simplify().evaluate({a: t_a, b: t_b, c: t_c})

    # Compare with the expected result
    expected = torch.einsum("ij,kl->ijkl", t_a.rename(None), t_c.rename(None)).rename("i", "j_", "k_", "l")
    assert_close(result, expected)


def test_function_evaluation():
    a = Variable("a", ["i"])
    b = Variable("b", ["i"])

    # Define a custom function that computes the element-wise product of two tensors
    class ElementWiseProduct(Function):
        def __init__(self, a, b):
            super().__init__("element_wise_product", ["i"], (a, "i"), (b, "i"))
            self.a = a
            self.b = b

        def update_edge_dims(self, shapes: dict[int, dict[str, int]]) -> Iterable[tuple[Tensor, str, int]]:
            # Like everybody else, I don't distinguish between the same channel name from different children
            # But I suppose in principle there could be a function that takes two inputs, which use the same
            # edge name, but the two tensors don't have the same size for that edge...
            # Could I just disallow that, which would make the edge_dim api much simpler?...
            union = shapes.get(id(self), {}) | shapes.get(id(self.a), {}) | shapes.get(id(self.b), {})
            if "i" in union:
                return [(t, "i", union["i"]) for t in (self, self.a, self.b)]
            return []

        def __call__(self, v1, v2):
            return v1 * v2

    # Create a tensor expression using the custom function: f(a, b)
    expr = ElementWiseProduct(a, b)

    # Create random input tensors
    t_a = torch.randn(3, names=("i",))
    t_b = torch.randn(3, names=("i",))

    # Evaluate the function expression
    result = expr.evaluate({a: t_a, b: t_b})

    # Compare with the expected result
    expected = t_a * t_b
    assert_close(result, expected)


def test_simplify_product_with_zero():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])

    # Create a tensor expression: a @ 0
    expr = Product([a, Zero(["j", "k"])])

    # Simplify the expression
    simplified_expr = expr.simplify()

    # Check if the simplified expression is a Zero tensor
    assert isinstance(simplified_expr, Zero)
    assert simplified_expr.edges == ["i", "k"]


def test_simplify_sum_of_products():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    c = Variable("c", ["i", "j"])
    d = Variable("d", ["j", "k"])
    values = rand_values([a, b, c, d], i=2, j=3, k=4)

    expr = (a @ b) + (c @ d)
    simplified_expr = expr.simplify()

    # Check if the simplified expression is a single Product
    assert isinstance(simplified_expr, Sum)
    assert len(simplified_expr.tensors) == 2
    assert all(isinstance(t, Product) for t in simplified_expr.tensors)

    assert_close(expr.evaluate(values), simplified_expr.evaluate(values))
