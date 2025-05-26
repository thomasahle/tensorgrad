from collections import defaultdict
from functools import singledispatch
from tensorgrad import Tensor, Function, Product, Sum, Variable, Expectation, Zero, Delta
from tensorgrad import functions as F


@singledispatch
def collect(expr: Tensor, x: Tensor) -> dict[int, Tensor]:
    if expr == x:
        return {1: 1}
    return {0: expr}  # This is always the default
    # raise NotImplementedError(f"Collect not implemented for {type(expr)}")


@collect.register
def _(expr: Variable, x: Tensor) -> dict[int, Tensor]:
    if expr == x:
        return {1: 1}
    return {0: expr}


def _prod(c1, c2):
    if isinstance(c1, int) or isinstance(c2, int):
        return c1 * c2
    return c1 @ c2


@collect.register
def _(expr: Product, x: Tensor) -> dict[int, Tensor]:
    # TODO: We may want to support the case where x is partially in the product
    # Using isomorphism stuff

    res = {0: 1}  # The convolutional identity
    for factor in expr.factors:
        new_res = defaultdict(int)
        for k1, c1 in res.items():
            for k2, c2 in collect(factor, x).items():
                new_res[k1 + k2] += _prod(c1, c2)
        res = new_res
    return res


@collect.register
def _(expr: Sum, x: Tensor) -> dict[int, Tensor]:
    res = defaultdict(int)
    for weight, term in zip(expr.weights, expr.terms):
        for k, v in collect(term, x).items():
            res[k] += weight * v
    return res


@collect.register
def _(expr: Function, x: Tensor) -> dict[int, Tensor]:
    if isinstance(expr.signature, F._PowerFunction):
        k = expr.signature.k
        inner = collect(expr.inputs[0], x)

        if k == 0:
            # x^0 = 1 (constant)
            return {0: Ones(**expr.inputs[0].shape)}
        
        if k < 0:
            if len(inner) == 1:
                ((k1, c1),) = inner.items()
                return {k1 * k: c1**k}
            raise NotImplementedError("Cannot collect negative powers of sums")

        # Otherwise, use repeated convolution
        res = {0: 1}
        for _ in range(k):
            new_res = defaultdict(int)
            for k1, c1 in res.items():
                for k2, c2 in inner.items():
                    new_res[k1 + k2] += c1 * c2  # mul instead of prod, since pow is hadamard
            res = new_res
        return res

    raise NotImplementedError(f"Function {expr.signature} not implemented for polynomial collection")


@collect.register
def _(expr: Zero, x: Tensor) -> dict[int, Tensor]:
    """Special handler for Zero tensors."""
    return {0: Zero()}


@collect.register
def _(expr: Delta, x: Tensor) -> dict[int, Tensor]:
    """Special handler for Delta tensors."""
    if expr == x:
        return {1: 1}
    return {0: expr}


@collect.register
def _(expr: Expectation, x: Tensor) -> dict[int, Tensor]:
    if x.depends_on(expr.wrt):
        # TODO: What about depdence on mu and covar?
        raise ValueError("Cannot collect a variable that depends on the variable of the expectation")

    res = collect(expr.tensor, x)

    # Convert scalar values to Tensor to avoid errors when creating Expectation
    from tensorgrad import Tensor as TensorClass  # Local import to avoid circular import

    # Create expectation for each term, wrapping scalars as tensors if needed
    result = {}
    for k, v in res.items():
        if isinstance(v, (int, float)):
            # Convert scalar to Tensor with empty shape
            v = TensorClass(v)
        result[k] = Expectation(v, expr.wrt, expr.mu, expr.covar, expr.covar_names)

    return result


# Helper function to normalize tensor representation
def _normalize_tensor(tensor):
    """
    Normalize tensor representation for comparison purposes.
    Converts Sum/Product representations of scalars to simple values.
    """
    if isinstance(tensor, (int, float)):
        return tensor
    # Add more normalization as needed
    return tensor
