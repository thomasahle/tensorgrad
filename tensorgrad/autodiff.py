"""Differentiation rules for tensorgrad's node types.

tensorgrad's node-type set is closed (Variable, Delta, Zero, Sum, Product,
Function, Derivative, Rename, plus the Expectation/Affine extras) while its
operations keep growing, so operations are organized as modules of per-type
rules rather than as methods spread across the classes.  This module is the
"grad" operation: one rule function per node type, plus the dispatch that
``Tensor.grad`` (the public entry in tensor.py, which validates ``new_names``
and asserts the result shape) delegates to.

It is also home to :func:`step_derivative`, the chain-rule *stepping* rule
that resolves ``Derivative`` nodes during simplification (the simplify engine
in tensorgrad/simplify.py dispatches ``Derivative`` here).

Every rule receives the already-validated ``new_names`` mapping
``{x_edge: fresh_name}`` and must return a tensor of shape
``t.shape | {new_names[e]: x.shape[e] for e in x.edges}``.  Rules produce
``Derivative`` nodes for their children rather than recursing themselves:
``grad`` pushes the derivative exactly one step, and ``simplify`` (via
:func:`step_derivative`) pushes it the rest of the way.
"""

from types import ModuleType
from typing import Any, Callable, Optional

from tensorgrad.tensor import (
    Constant,
    Delta,
    Derivative,
    Function,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
    Zero,
    _unused_edge_names,
)

# Lazily imported tensorgrad.simplify (for the provenance hook shared with
# the simplify engine). Lazy because simplify.py imports this module at top
# level; the reverse edge must not be static.
_simplify_mod: Optional[ModuleType] = None


def _get_simplify() -> ModuleType:
    global _simplify_mod
    if _simplify_mod is None:
        import tensorgrad.simplify as simplify

        _simplify_mod = simplify
    return _simplify_mod


################################################################################
# Differentiation rules, one per node type
################################################################################


def grad_variable(t: Variable, x: Variable, new_names: dict[str, str]) -> Tensor:
    """d/dx x = identity (an outer product of order-2 Deltas pairing each edge
    of x with its new derivative edge); d/dx y = Zero for any other variable."""
    # Note: Constraints (see with_eq_constraint) deliberately do NOT affect the
    # gradient. They describe the values the variable will hold, not the space
    # it is optimized over, so we differentiate as if the variable were
    # unconstrained.
    if x == t:
        # TODO: If X has symmetries, the derivative can actually be more complex than this.
        # See 2.8.2 Symmetric in the Cookbook: https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf
        return Product(Delta(s, e, new_names[e]) for e, s in t.shape.items())
    # Note: We don't need to tell Zero the symmetries, since it's automatically
    # symmetric in all dimensions that have compatible sizes.
    return Zero(_symmetries=None, **(t.shape | {new_names[e]: s for e, s in x.shape.items()}))


def grad_constant(t: Constant, x: Variable, new_names: dict[str, str]) -> Tensor:
    """d/dx c = Zero, broadcast over the new derivative edges: constants
    (Delta, Zero, Convolution, Reshape, Affine, ...) do not depend on any
    variable."""
    return Zero(_symmetries=None, **(t.shape | {new_names[e]: s for e, s in x.shape.items()}))


def grad_rename(t: Rename, x: Variable, new_names: dict[str, str]) -> Tensor:
    """d/dx Rename(inner) = Rename(d/dx inner): differentiate through the
    (lazy) rename wrapper. If the requested new names are used inside `inner`,
    they are routed through fresh middle names, which the outer Rename maps
    back to the requested names."""
    middle = _unused_edge_names(new_names.values(), t.edges | t.tensor.edges)
    middle_names = {o: middle[n] for o, n in new_names.items()}
    middle_to_new = {middle[n]: n for o, n in new_names.items()}
    return Rename(t.tensor.grad(x, middle_names), middle_to_new | t.mapping)


def grad_function(t: Function, x: Variable, new_names: dict[str, str]) -> Tensor:
    """The chain rule: d/dx f(g1(x), ..., gk(x)) = sum_i (d/dx gi(x)) Dif(...).

    Each part contracts the signature's i-th derivative function ("outside")
    with the Derivative of the i-th input ("inner") over fresh connection
    edges, broadcasting over their remaining shared edges."""
    # D_i adds a new output edge to the function, which is contracted with
    # the normal output edge of the tensor. So we need to make sure this doesn't
    # clash with an existing output edge of f.
    parts = []
    for i, (inp, input_edges) in enumerate(zip(t.inputs, t.signature.inputs)):
        # Take the derivative of the outer function
        # We need "connection" edges for each edge in input_edges. Mostly we could just use the same name
        # but they need to avoid clashing with "new_names" and the output edges of the tensor.
        connection_names = _unused_edge_names(input_edges, t.edges | new_names.values())
        # Just like simple calculus, we need the derivative of the outside times the derivative of the inside
        outside = Function(
            t.signature.derivative(i, connection_names),
            t.inputs,
            t.shape_out | {connection_names[e]: inp.shape[e] for e in input_edges},
        )
        assert outside.edges == t.edges | connection_names.values()

        # The the derivative of the inner function
        # We rename the (former) input edges to the connection edges, but keep the remaining edges
        # (which are part of t.edges) untouched.
        inner = Derivative(inp.rename(**connection_names), x, new_names)

        # The two parts are then multiplied together on the connection names,
        # while broadcasted on their remaining shared edges.
        import tensorgrad.functions as F  # Import here to avoid circular import

        part = F.sum(outside * inner, connection_names.values())
        parts.append(part)
    return Sum(parts)


def grad_derivative(t: Derivative, x: Variable, new_names: dict[str, str]) -> Tensor:
    """d/dx D_y(inner) = D_y(d/dx inner): pass the new derivative through the
    existing one (rather than creating a doubly-nested Derivative, which would
    loop forever when simplify tries to resolve it)."""
    return Derivative(grad_step(t.tensor, x, new_names), t.x, t.new_names)


def grad_product(t: Product, x: Variable, new_names: dict[str, str]) -> Tensor:
    """The product rule: d/dx (f * g) = f' * g + f * g', one term per factor.
    Inner (contraction) edges are first renamed away from the new derivative
    edges so the fresh edges can be threaded through any factor."""
    if not t.factors:
        # d/dx 1 = 0, broadcast over the new derivative edges. (Reachable
        # since grad became lazy: step_derivative simplifies the base first,
        # and e.g. sum-over-softmax collapses to the empty product.)
        return Zero(_symmetries=None, **(t.shape | {new_names[e]: s for e, s in x.shape.items()}))
    # Since we are adding new edges to an internal tensor in the product, we need to make sure
    # none of the other tensors in the product have edges that clash with these new edges.
    inner_names = {e for f in t.factors for e in f.edges if e not in t.edges}
    new_edges = set(new_names.values()) | t.edges
    rename = _unused_edge_names(inner_names, new_edges)
    new_prod = Product([f.rename(**rename) for f in t.factors])
    assert new_prod.shape == t.shape, "Renaming should not change the product"

    # The classic product rule of Calculus: d/dx (f * g) = f' * g + f * g'
    return Sum(
        [
            Product(new_prod.factors[:i] + [Derivative(f, x, new_names)] + new_prod.factors[i + 1 :])
            for i, f in enumerate(new_prod.factors)
        ]
    )


def grad_sum(t: Sum, x: Variable, new_names: dict[str, str]) -> Tensor:
    """Linearity: d/dx sum_i w_i t_i = sum_i w_i (d/dx t_i)."""
    if not t.terms:
        return Zero(_symmetries=None, **(t.shape | {new_names[e]: s for e, s in x.shape.items()}))
    return Sum([Derivative(term, x, new_names) for term in t.terms], t.weights)


################################################################################
# Dispatch
################################################################################

# (Callable[[Any, ...]]: each rule takes its concrete node type as the first
# parameter, which a dict value type of Callable[[Tensor, ...]] would reject.)
_GRAD_RULES: dict[type, Callable[[Any, Variable, dict[str, str]], Tensor]] = {
    Variable: grad_variable,
    Delta: grad_constant,
    Zero: grad_constant,
    Rename: grad_rename,
    Function: grad_function,
    Derivative: grad_derivative,
    Product: grad_product,
    Sum: grad_sum,
}


def grad_step(tensor: Tensor, x: Variable, new_names: dict[str, str]) -> Tensor:
    """Push the derivative one step into `tensor`, dispatching by node type.

    Types outside the core set (Expectation; Constant subclasses such as
    Convolution/Reshape/Affine, which inherit the zero rule) fall back to the
    ``_grad`` method protocol on the class."""
    rule = _GRAD_RULES.get(type(tensor))
    if rule is not None:
        return rule(tensor, x, new_names)
    return tensor._grad(x, new_names)


################################################################################
# Derivative resolution during simplify (chain-rule stepping)
################################################################################


def step_derivative(t: Derivative, args: dict[str, Any]) -> Tensor:
    """Resolve a Derivative node during simplification.

    Rewrites D_x(inner) to Zero when inner doesn't depend on x; otherwise
    simplifies inner and, if the ``grad_steps`` budget allows, applies one
    ``grad`` step and re-simplifies the result. Terminates because each
    application either eliminates the Derivative node or pushes it strictly
    closer to the leaves, where grad_variable/grad_constant resolve it
    (``grad_steps`` is decremented IN PLACE, so a finite budget is shared
    across the whole tree and reaches zero)."""
    if not t.tensor.depends_on(t.x):
        return Zero(_symmetries=None, **t.shape)
    inner = t.tensor.simplify(args)
    if args["grad_steps"] == 0:
        # If grad_steps is 0, we pass the simplify through the derivative.
        res = Derivative(inner, t.x, t.new_names)
    else:
        args["grad_steps"] -= 1
        # Have to call simplify twice to avoid an infinite loop when stacking multiple derivatives.
        # grad_step, not .grad(): the public API is lazy (returns a Derivative
        # node), and this is the place that performs the actual stepping.
        grad = grad_step(inner, t.x, t.new_names)
        _get_simplify()._record_simplify_provenance(args, inner, grad)
        res = grad.simplify(args)
    return res
