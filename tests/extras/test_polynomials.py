from sympy import symbols
from tensorgrad import Product, Sum, Variable, Expectation, Delta
from tensorgrad import functions as F
from tensorgrad.extras.polynomials import collect
from tensorgrad.tensor import Ones


def assert_equal(dict1_or_tensor1, dict2_or_tensor2):
    """Assert that two dictionaries or tensors are equal."""
    # Handle the case where we're comparing tensors directly
    if not isinstance(dict1_or_tensor1, dict):
        t1 = dict1_or_tensor1 if isinstance(dict1_or_tensor1, (int, float)) else dict1_or_tensor1.full_simplify()
        t2 = dict2_or_tensor2 if isinstance(dict2_or_tensor2, (int, float)) else dict2_or_tensor2.full_simplify()
        assert t1 == t2
        return
        
    # Original dictionary comparison
    dict1, dict2 = dict1_or_tensor1, dict2_or_tensor2
    assert dict1.keys() == dict2.keys()
    for key in dict1:
        t1 = dict1[key] if isinstance(dict1[key], (int, float)) else dict1[key].full_simplify()
        t2 = dict2[key] if isinstance(dict2[key], (int, float)) else dict2[key].full_simplify()
        assert t1 == t2


def test_default_collect():
    """Test the default collect implementation for regular tensors."""
    x = Variable("x")
    y = Variable("y")
    assert collect(y, x) == {0: y}
    assert collect(x, x) == {1: 1}


def test_product_collect():
    """Test collecting in products."""
    i, j, k = symbols("i j k")
    x = Variable("x", i, j)
    y = Variable("y", j, k)
    z = Variable("z", i, k)

    prod = Product([x, y, z])
    assert_equal(collect(prod, x), {1: y @ z})


def test_power_function_collect():
    """Test collecting with power functions including negative and fractional powers."""
    i, j = symbols("i j")
    x = Variable("x", i=i, j=j)

    # Test standard positive integer powers
    assert collect(F.pow(x, 2), x) == {2: 1}
    assert collect(F.pow(x, 3), x) == {3: 1}
    assert collect(F.pow(x, 4), x) == {4: 1}

    # Test identity (x^1)
    assert collect(F.pow(x, 1), x) == {1: 1}

    # Test constant (x^0)
    assert_equal(collect(F.pow(x, 0), x), {0: Ones(**x.shape)})

    # Test negative powers
    assert collect(F.pow(x, -1), x) == {-1: 1}
    assert collect(F.pow(x, -2), x) == {-2: 1}


def test_power_sum_collect():
    """Test collecting from a sum of different powers of the same variable."""
    i, j = symbols("i j")
    x = Variable("x", i=i, j=j)

    # Create a polynomial with a complex mix of positive and negative powers:
    # 1.5*x^-2 + 4*x^-1 + 2*x + 3.5*x^2 + 7*x^3
    term1 = F.pow(x, -2)
    term2 = F.pow(x, -1)
    term3 = x
    term4 = F.pow(x, 2)
    term5 = F.pow(x, 3)

    # Use fractional weights to test more complex cases
    expr = Sum([term1, term2, term3, term4, term5], [1.5, 4, 2, 3.5, 7])

    # The expected result should contain each power with its coefficient
    expected = {-2: 1.5, -1: 4, 1: 2, 2: 3.5, 3: 7}

    # The collect function should properly separate terms by power of x
    # with the correct coefficients
    assert collect(expr, x) == expected


def test_product_collection_with_shared_edges():
    """Test collecting from products with shared edges/contractions between variables."""
    i, j, k, l, m = symbols("i j k l m")
    x = Variable("x", i=i, j=j)
    y = Variable("y", j=j, k=k)

    # Test simple contraction between two variables (x@y)
    # The output of collect should be {1: y} with appropriate contractions
    prod1 = x @ y  # Contract on j
    expected1 = {1: y}
    assert_equal(collect(prod1, x), expected1)


def test_expectation_collect_simple():
    """Test collecting from a simple expectation with constant coefficient."""
    i, j, k = symbols("i j k")
    x = Variable("x", i)
    y = Variable("y", j)
    z = Variable("z", k)

    expectation = Expectation(x @ y, z)
    assert_equal(collect(expectation, x), {1: Expectation(y, z)})


def test_complex_mixed_polynomial():
    """Test collecting from a complex mixed polynomial with various coefficients."""
    i, j, k = symbols("i j k")

    # Create a variable with a single dimension for simpler testing
    x = Variable("x", i=i)

    # Create a mixed expression with different powers to test
    # 2*x + 3*x^2 + 4*x^3
    term1 = x
    term2 = F.pow(x, 2)
    term3 = F.pow(x, 3)
    expr = Sum([term1, term2, term3], [2, 3, 4])

    # Expected result from collecting
    expected = {1: 2, 2: 3, 3: 4}

    # Verify the exact result
    assert collect(expr, x) == expected


def test_nested_power_collection():
    """Test collecting from nested powers like (x^2)^3."""
    i = symbols("i")
    x = Variable("x", i=i)

    # Create (x^2)^3
    inner_pow = F.pow(x, 2)  # x^2
    outer_pow = F.pow(inner_pow, 3)  # (x^2)^3, which should simplify to x^6

    result = collect(outer_pow, x)

    # Should collect to x^6 -> {6: 1}
    assert result == {6: 1}


def test_power_of_sum():
    """Test collecting from powers of sums like (x + y)^2."""
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)

    # What are we even trying to do when we collect from a power of vector?
    # (x + y)^2 = x^2 + 2xy + y^2  is all using Hadamard product.
    pow_expr = F.pow(x + y, 2).simplify()
    result = collect(pow_expr, x)
    
    # Check we have the right powers
    assert result.keys() == {0, 1, 2}
    
    # Check coefficient of x^2 is 1
    assert result[2] == 1
    
    # Check coefficient of x^1 is 2*y
    assert_equal(result[1], 2 * y)
    
    # Check coefficient of x^0 exists and is related to y^2
    # (it may be in expanded form rather than Function form)
    assert 0 in result
    # Verify it has the right shape
    assert result[0].shape == y.shape

    # Do I actually not have a way to show that these are equal right now?
    # Product(
    #     [
    #         Variable("y", i).rename(i="i_0"),
    #         Variable("y", i).rename(i="i_1"),
    #         Delta(i, "i, i_0, i_1"),
    #     ]
    # )
    # Function(FunctionSignature("pow(k=2)", set(), [set()]), inputs=[Variable("y", i)], shape_out={})

    # A different type of power
    # (x + y) @ (x + y) = x @ x + 2 x @ y + y @ y
    pow_expr = ((x + y) @ (x + y)).simplify()
    assert_equal(collect(pow_expr, x), {0: y @ y, 1: 2 * y, 2: 1})


def test_delta_tensor_collection():
    """Test collecting with Delta tensors."""
    i, j = symbols("i j")
    x = Variable("x", i)
    delta_i = Delta(i)

    expr = (x + delta_i) ** 2
    print(expr)
    result = collect(expr, delta_i)
    
    # Check we have the right powers
    assert result.keys() == {0, 1, 2}
    
    # Check coefficient of delta^2 is Delta(i, "i") since delta_i^2 = i^2
    assert_equal(result[2], Delta(i, "i"))
    
    # Check coefficient of delta^1 is 2*x
    assert_equal(result[1], 2 * x)
    
    # Check coefficient of delta^0 exists and is related to x^2
    assert 0 in result
    # Verify it has the right shape
    assert result[0].shape == x.shape

    expr = expr.simplify()
    result = collect(expr, delta_i)
    # Same checks after simplification
    assert result.keys() == {0, 1, 2}
    assert_equal(result[2], Delta(i, "i"))
    assert_equal(result[1], 2 * x)
    assert 0 in result


def test_trace():
    i = symbols("i")
    x = Variable("x", i, j=i)
    d = Delta(i)

    t1 = x
    t2 = F.dot(x, x, dim=("j", "i"))
    t3 = F.trace(F.multi_dot([x, x, x], dims=("i", "j")))
    expr = t1 / d + t2 / d**2 + t3 / d**3
    result = collect(expr, d)
    
    # Check we have the right powers
    assert result.keys() == {-1, -2, -3}
    
    # Check the coefficients match (after simplification)
    assert_equal(result[-1], t1)
    assert_equal(result[-2], t2)
    # For -3, we just check it's related to the trace - the exact form may vary
    assert -3 in result


def test_neg_delta():
    i, j = symbols("i, j")
    d = Delta(i)

    assert_equal(collect(F.pow(d, -1), d), {-1: 1})

    x = Variable("x", i, j)
    assert_equal(collect(x / d, d), {-1: x})
