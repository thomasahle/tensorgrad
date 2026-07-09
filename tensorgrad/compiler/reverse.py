"""Joint reverse-mode resolution of a shared gradient family.

compile_to_callable(loss, *[loss.grad(p) for p in params]) hands normalize a
family of Derivative nodes over ONE shared scalar base. Resolving each
independently (grad.py's chain-rule stepping) is correct but builds every
gradient as its own forward-shaped Jacobian chain: the 54 nano-GPT gradients
lower to ~2400 genuinely distinct intermediate values that no downstream
pass can re-unify (measured; four re-alignment mechanisms were tried and
failed — multi-term CSE, canonical operand interning, reuse-seeded
contraction, early stabilization).

This module resolves the family JOINTLY instead: one reverse sweep over the
shared base graph computes a symbolic cotangent u[n] = dL/dn per node, and
each requested gradient is u[p] renamed to its Derivative's declared edges.
Every u is built once and interned, so all gradients literally reference the
same subexpressions — the sharing exists by construction, and the interning /
isomorphism machinery propagates it through lowering.

No new calculus: the per-node vector-Jacobian contributions below are the
transposes of grad.py's one-step rules, built from the same signature-level
leaf derivatives (the no-composite-Jacobians law is untouched — this is the
same chain rule in a different evaluation order). Anything the sweep does
not recognize (non-scalar bases, Expectation, singleton families in flat
programs) falls back to the independent path unchanged.

Nested chains (HVPs: grad of <grad(loss), v>) resolve innermost-first over
several rounds — reverse-over-reverse. Each round sweeps the families whose
base is Derivative-free and rebuilds; the round after that sees the outer
Derivative's base as plain algebra and sweeps it too. In nested programs
singleton families ARE swept: one HVP is exactly the case where grad.py's
chain-rule stepping explodes (measured ~10x compile time per stacked
softmax block), while flat singletons keep the classic path unchanged.
"""

from collections import defaultdict
from typing import Optional

from tensorgrad.tensor import (
    Constant,
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

# Master toggle (tests A/B against the independent path by flipping this).
REVERSE_GRADS = True

# Node types the sweep understands. Constant subclasses (Delta, Zero, Ones,
# Convolution, Affine, ...) are leaves with no cotangent flow.
_BRANCH_TYPES = (Sum, Product, Rename, Function)


def _children(t: Tensor) -> list[Tensor]:
    if isinstance(t, Sum):
        return list(t.terms)
    if isinstance(t, Product):
        return list(t.factors)
    if isinstance(t, Rename):
        return [t.tensor]
    if isinstance(t, Function):
        return list(t.inputs)
    if isinstance(t, Derivative):
        return [t.tensor]
    return []


def _supported(base: Tensor) -> Optional[list[Tensor]]:
    """Postorder of the base DAG (unique by identity), or None if any node
    type is outside the sweep's vocabulary."""
    post: list[Tensor] = []
    seen: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(base, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            post.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        if not isinstance(node, (Variable, Constant) + _BRANCH_TYPES):
            return None
        stack.append((node, True))
        for c in _children(node):
            stack.append((c, False))
    return post


def _function_vjp(t: Function, u: Tensor, inp: Tensor, i: int) -> Tensor:
    """Cotangent contribution to t.inputs[i]: contract u with the signature's
    i-th derivative over t's output-only edges (the transpose of
    grad.py's grad_function part for input i)."""
    import tensorgrad.functions as F
    from tensorgrad.compiler.cells import CELLS, _FusedFunction

    # Fused technology-mapping cells (sdpa/layer_norm/gelu/...): bypass the
    # dense Jacobian entirely. u is the output cotangent; the cell emits a
    # clean VJP primitive that lowers to the fused backward kernel.
    if isinstance(t.signature, _FusedFunction) and i < CELLS[t.signature.cell_name].n_diff:
        return CELLS[t.signature.cell_name].vjp(t.inputs, i, u, t.signature.params)

    input_edges = t.signature.inputs[i]
    connection_names = _unused_edge_names(input_edges, t.edges | inp.edges)
    outside = Function(
        t.signature.derivative(i, connection_names),
        t.inputs,
        t.shape_out | {connection_names[e]: inp.shape[e] for e in input_edges},
    )
    # u ⊙ J over shared (broadcast) edges, summed over t's edges that the
    # input does not carry; connection edges then become the input's edges.
    contrib = F.sum(outside * u, [e for e in t.edges if e not in inp.edges])
    return contrib.rename(**{n: o for o, n in connection_names.items()})


def _sweep(base: Tensor, targets: list[Variable]) -> Optional[dict[int, Tensor]]:
    """One reverse pass: cotangents for every occurrence of each target.
    Returns {id(target_occurrence): u} — summed per Variable by the caller —
    or None when the base is outside the sweep's vocabulary."""
    post = _supported(base)
    if post is None:
        return None

    # needs: nodes whose subtree contains a target (cotangents elsewhere are
    # never read, so they are not built).
    needs: set[int] = set()
    for node in post:  # postorder: children first
        if isinstance(node, Variable) and any(node == x for x in targets):
            needs.add(id(node))
        elif any(id(c) in needs for c in _children(node)):
            needs.add(id(node))
    if id(base) not in needs:
        return {}

    contribs: dict[int, list[Tensor]] = defaultdict(list)
    contribs[id(base)].append(Product([]))  # dL/dL = 1 (scalar base)
    u_of: dict[int, Tensor] = {}

    for node in reversed(post):  # reverse topological: parents first
        if id(node) not in needs or not contribs[id(node)]:
            continue
        cs = contribs[id(node)]
        u = cs[0] if len(cs) == 1 else Sum(cs)
        u_of[id(node)] = u
        if isinstance(node, Sum):
            for term, w in zip(node.terms, node.weights):
                if id(term) in needs:
                    contribs[id(term)].append(u if w == 1 else Sum([u], [w]))
        elif isinstance(node, Product):
            factors = list(node.factors)
            for i, f in enumerate(factors):
                if id(f) in needs:
                    contribs[id(f)].append(Product([u] + factors[:i] + factors[i + 1 :]))
        elif isinstance(node, Rename):
            inner = node.tensor
            if id(inner) in needs:
                inverse = {n: o for o, n in node.mapping.items()}
                contribs[id(inner)].append(u.rename(**inverse))
        elif isinstance(node, Function):
            for i, inp in enumerate(node.inputs):
                if id(inp) in needs:
                    contribs[id(inp)].append(_function_vjp(node, u, inp, i))

    return {id(node): u for node in post if isinstance(node, Variable) and id(node) in u_of for u in [u_of[id(node)]]}


def _collect_derivatives(t: Tensor, found: dict[int, Derivative], seen: set[int]) -> None:
    """Find candidate Derivative nodes anywhere in `t` (they are usually not
    roots: optimizers wrap gradients in update algebra, e.g. AdamW)."""
    if id(t) in seen:
        return
    seen.add(id(t))
    if type(t) is Derivative:
        if isinstance(t.x, Variable) and t.tensor.order == 0:
            found[id(t)] = t
        # fall through: nested chains (HVPs) carry inner Derivatives in the
        # base; resolve_shared_gradients' rounds resolve them innermost-first
    for c in _children(t):
        _collect_derivatives(c, found, seen)


def _rebuild(t: Tensor, repl: dict[int, Tensor], memo: dict[int, Tensor]) -> Tensor:
    """Rebuild `t` with replaced Derivative nodes; returns `t` itself when
    nothing under it changed. Raises on node types it cannot reconstruct
    (the caller falls back to the original output tensor)."""
    if id(t) in memo:
        return memo[id(t)]
    if id(t) in repl:
        memo[id(t)] = repl[id(t)]
        return repl[id(t)]
    kids = _children(t)
    new_kids = [_rebuild(c, repl, memo) for c in kids]
    if all(a is b for a, b in zip(kids, new_kids)):
        memo[id(t)] = t
        return t
    if isinstance(t, Sum):
        out: Tensor = Sum(new_kids, list(t.weights))
    elif isinstance(t, Product):
        out = Product(new_kids)
    elif isinstance(t, Rename):
        out = Rename(new_kids[0], dict(t.mapping))
    elif isinstance(t, Function):
        out = Function(t.signature, new_kids, t.shape_out)
    elif isinstance(t, Derivative):
        out = Derivative(new_kids[0], t.x, dict(t.new_names))
    else:
        raise TypeError(f"cannot rebuild through {type(t).__name__}")
    memo[id(t)] = out
    return out


def _contains_derivative(t: Tensor) -> bool:
    seen: set[int] = set()
    stack = [t]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Derivative):
            return True
        stack.extend(_children(node))
    return False


def resolve_shared_gradients(tensors: tuple) -> tuple:
    """Resolve Derivative families jointly, wherever the Derivative nodes sit
    inside the output expressions (optimizer algebra like AdamW wraps them).
    Flat programs resolve families of >= 2 gradients in one round; programs
    with nested chains (HVPs) run innermost-first rounds with singletons
    allowed (see module docstring). Anything unrecognized is returned
    unchanged; any failure falls back per family."""
    if not REVERSE_GRADS:
        return tensors
    found: dict[int, Derivative] = {}
    seen: set[int] = set()
    for t in tensors:
        _collect_derivatives(t, found, seen)
    nested = any(_contains_derivative(d.tensor) for d in found.values())
    for _ in range(4 if nested else 1):  # rounds = nesting depth (HVP: 2)
        tensors, progress = _resolve_round(tensors, allow_singletons=nested)
        if not progress:
            break
    return tensors


def _resolve_round(tensors: tuple, allow_singletons: bool) -> tuple[tuple, bool]:
    found: dict[int, Derivative] = {}
    seen: set[int] = set()
    for t in tensors:
        _collect_derivatives(t, found, seen)
    groups: dict[int, list[Derivative]] = defaultdict(list)
    for d in found.values():
        groups[id(d.tensor)].append(d)

    repl: dict[int, Tensor] = {}
    for _, family in groups.items():
        if len(family) < (1 if allow_singletons else 2):
            continue
        base = family[0].tensor
        if _contains_derivative(base):
            continue  # inner families first; eligible again next round
        targets = [d.x for d in family]
        try:
            occ = _sweep(base, targets)
            if occ is None:
                continue
            post = _supported(base)
            assert post is not None
            family_repl: dict[int, Tensor] = {}
            for d in family:
                # Sum cotangents over all occurrences of the variable
                # (fan-out: equal-but-distinct Variable objects each carry
                # their own accumulated cotangent).
                pieces = [
                    occ[id(node)]
                    for node in post
                    if isinstance(node, Variable) and node == d.x and id(node) in occ
                ]
                if not pieces:
                    g: Tensor = Zero(_symmetries=None, **d.x.shape)
                else:
                    g = pieces[0] if len(pieces) == 1 else Sum(pieces)
                g = g.rename(**d.new_names)
                if set(g.edges) != set(d.edges):
                    raise ValueError(f"reverse sweep produced edges {set(g.edges)}, expected {set(d.edges)}")
                family_repl[id(d)] = g
        except Exception:
            continue  # keep the independent path for this family
        repl.update(family_repl)

    if not repl:
        return tensors, False
    out = []
    memo: dict[int, Tensor] = {}
    for t in tensors:
        try:
            out.append(_rebuild(t, repl, memo))
        except TypeError:
            out.append(t)
    return tuple(out), True
