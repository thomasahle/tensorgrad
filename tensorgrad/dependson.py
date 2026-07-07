"""The ``depends_on`` predicate: does a tensor depend on a given Variable?

tensorgrad's node-type set is closed (Variable, Delta, Zero, Sum, Product,
Function, Derivative, Rename, plus the Expectation/Affine extras) while its
operations keep growing, so operations are organized as modules of per-type
rules rather than as methods spread across the classes (see
tensorgrad/simplify.py and tensorgrad/grad.py). This module is the
"depends_on" operation: one rule per node type, dispatched by type.

``Tensor.depends_on`` (the public entry in tensor.py) owns the memoization —
the raw recursion visits every root-to-leaf PATH, which is exponential on
shared-subtree DAGs (residual nets) — and delegates the per-type step here.
Rules recurse via ``child.depends_on(x)``, re-entering that memoized entry.

The dispatch table is the extension point: types outside the core set
register their rule on import (tensorgrad/extras/expectation.py registers
Expectation's), so this module stays in the core and never imports from
tensorgrad.extras. Constant is registered directly (rather than Delta/Zero
individually), so every Constant subclass — Delta, Zero, and the
Convolution/Reshape/Affine extras — shares the "depends on nothing" rule
through the MRO.
"""

from functools import singledispatch

from tensorgrad.tensor import (
    Constant,
    Derivative,
    Function,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
)


@singledispatch
def _dispatch_depends_on(tensor: Tensor, x: Variable) -> bool:
    """Dispatch a node to its per-type depends_on rule (registered below).

    A silently wrong answer here would be dangerous, so unknown node types
    raise; register a rule for them instead."""
    raise NotImplementedError(f"depends_on not implemented for {type(tensor)}")


@_dispatch_depends_on.register
def depends_on_variable(t: Variable, x: Variable) -> bool:
    """A Variable depends only on itself (names are load-bearing identity)."""
    return x == t


@_dispatch_depends_on.register
def depends_on_constant(t: Constant, x: Variable) -> bool:
    """Constants (Delta, Zero, Convolution, Reshape, Affine) depend on nothing."""
    return False


@_dispatch_depends_on.register
def depends_on_rename(t: Rename, x: Variable) -> bool:
    return t.tensor.depends_on(x)


@_dispatch_depends_on.register
def depends_on_derivative(t: Derivative, x: Variable) -> bool:
    return t.tensor.depends_on(x)


@_dispatch_depends_on.register
def depends_on_function(t: Function, x: Variable) -> bool:
    return any(inp.depends_on(x) for inp in t.inputs)


@_dispatch_depends_on.register
def depends_on_product(t: Product, x: Variable) -> bool:
    return any(f.depends_on(x) for f in t.factors)


@_dispatch_depends_on.register
def depends_on_sum(t: Sum, x: Variable) -> bool:
    return any(term.depends_on(x) for term in t.terms)
