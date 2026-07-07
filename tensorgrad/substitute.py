"""The ``substitute`` operation: replace a Variable with another tensor.

tensorgrad's node-type set is closed (Variable, Delta, Zero, Sum, Product,
Function, Derivative, Rename, plus the Expectation/Affine extras) while its
operations keep growing, so operations are organized as modules of per-type
rules rather than as methods spread across the classes (see
tensorgrad/simplify.py, tensorgrad/grad.py and tensorgrad/dependson.py).
This module is the "substitute" operation: one rule per node type, dispatched
by type, plus the memoized driver that ``Tensor.substitute`` (the public entry
in tensor.py) delegates to.

The driver is sharing-preserving: each DAG node is rewritten at most once
(keyed by object identity), and unchanged subtrees are returned as-is. The
stock recursion rebuilt the whole tree per call, which is exponential on
expressions with shared subexpressions (residual networks). Rules recurse into
children via :func:`_substitute_memo`, not ``child.substitute`` (which would
start a fresh memo per child).

The dispatch table is the extension point for types outside the core set
(register a rule on import). Derivative and the Expectation extra are
deliberately NOT registered: they do not support substitution and raise via
the dispatch default. Constant is registered directly, so every Constant
subclass — Delta, Zero, and the Convolution/Reshape/Affine extras — shares
the "nothing to substitute" rule through the MRO.
"""

from functools import singledispatch

from tensorgrad.tensor import (
    Constant,
    Function,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
)


def _substitute_memo(t: Tensor, x: Variable, y: Tensor, memo: dict) -> Tensor:
    """Memoized (by object identity) driver for Tensor.substitute.

    The memo keys are ids of descendants of the substitution root, which stay
    alive for the duration of the call, so ids cannot be recycled."""
    res = memo.get(id(t))
    if res is None:
        res = _dispatch_substitute(t, x, y, memo)
        memo[id(t)] = res
    return res


@singledispatch
def _dispatch_substitute(t: Tensor, x: Variable, y: Tensor, memo: dict) -> Tensor:
    """Dispatch a node to its per-type substitute rule (registered below).

    Deliberately unregistered: Derivative and the Expectation extra do not
    support substitution, so they (and unknown node types) raise here."""
    raise NotImplementedError(f"substitute not implemented for {type(t)}")


@_dispatch_substitute.register
def substitute_variable(t: Variable, x: Variable, y: Tensor, memo: dict) -> Tensor:
    # Name pre-check: Variables with different names are never isomorphic
    # (the name is part of the structural graph), so the expensive
    # isomorphism __eq__ only runs on same-named variables.
    if t is x or (t.name == x.name and x == t):
        return y
    return t


@_dispatch_substitute.register
def substitute_constant(t: Constant, x: Variable, y: Tensor, memo: dict) -> Tensor:
    """Constants (Delta, Zero, Convolution, Reshape, Affine) contain no Variables."""
    return t


@_dispatch_substitute.register
def substitute_rename(t: Rename, x: Variable, y: Tensor, memo: dict) -> Tensor:
    inner = _substitute_memo(t.tensor, x, y, memo)
    return t if inner is t.tensor else Rename(inner, t.mapping)


@_dispatch_substitute.register
def substitute_function(t: Function, x: Variable, y: Tensor, memo: dict) -> Tensor:
    inputs = [_substitute_memo(inp, x, y, memo) for inp in t.inputs]
    if all(a is b for a, b in zip(inputs, t.inputs)):
        return t
    return Function(t.signature, inputs, t.shape_out)


@_dispatch_substitute.register
def substitute_product(t: Product, x: Variable, y: Tensor, memo: dict) -> Tensor:
    factors = [_substitute_memo(f, x, y, memo) for f in t.factors]
    if all(a is b for a, b in zip(factors, t.factors)):
        return t
    return Product(factors)


@_dispatch_substitute.register
def substitute_sum(t: Sum, x: Variable, y: Tensor, memo: dict) -> Tensor:
    terms = [_substitute_memo(term, x, y, memo) for term in t.terms]
    if all(a is b for a, b in zip(terms, t.terms)):
        return t
    return Sum(terms, t.weights)
