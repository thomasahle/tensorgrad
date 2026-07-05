"""The simplification rule catalog and engine.

tensorgrad's node-type set is closed (Variable, Delta, Zero, Sum, Product,
Function, Derivative, Rename, plus the Expectation/Affine extras) while its
operations keep growing, so operations are organized as modules of per-type
rules rather than as methods spread across the classes.  This module is the
"simplify" operation: :func:`simplify` is the engine that ``Tensor.simplify``
(the public entry in tensor.py) delegates to, and everything below it is the
rule catalog, one named rule per rewrite.

The engine owns
- the recursion: rules simplify children via ``child.simplify(args)``, which
  re-enters the engine (and therefore the memo);
- the args threading: ONE mutable dict is passed down the whole tree — it
  carries the knobs below, the cross-subtree memo, and the in-place
  ``grad_steps`` budget;
- the memoization block and the shape-preservation assert.

The args knobs (defaults set by the engine; every branch they gate lives in
this module, tensorgrad/grad.py, or the FunctionSignature.simplify overrides
in tensorgrad/functions.py):

- ``grad_steps`` (inf): how many Derivative nodes to resolve; decremented IN
  PLACE by :func:`tensorgrad.grad.step_derivative`, so a finite budget is
  shared across the tree (and across repeated ``simplify`` calls reusing the
  same dict). Used by imgtools/to_diagram for step-by-step derivations.
- ``sum_combine_terms`` (True): gates a filter in :func:`simplify_sum`. The
  compiler's normalize stage sets it False. NOTE: isomorphic-term merging
  itself happens in the Counter accumulation regardless of this flag, and the
  gated filter is subsumed by the unconditional one right after it, so the
  flag currently has no observable effect; kept (and documented) because the
  compiler preset sets it.
- ``combine_products`` (True): gates the pow-combination pass
  (``_PowerFunction.simplify_outer``) in :func:`simplify_product`, i.e.
  pow(x, a) * pow(x, b) -> pow(x, a + b) and cancellations. The compiler's
  normalize stage sets it False (the IR passes own that algebra).
- ``factor_components`` (True): within the pow pass, also combine bare factors
  with pows across shared hyperedges and disjoint components (read in
  functions.py). Expectation._simplify sets it False (moment expansion
  prefers unfactored products); the compiler's normalize stage sets it False.
- ``expand_functions`` (True): let FunctionSignature.simplify overrides expand
  a function into its defining expression (e.g. softmax -> exp/sum-exp; read
  in functions.py). Tests and the compiler's normalize stage set it False to
  keep functions as opaque kernels.
- ``expand`` (False): distribute products over sums (cartesian expansion in
  :func:`simplify_product`; also read by the pow signature's pointwise split
  in functions.py). Set by full_simplify(expand=True) and by callers wanting
  sum-of-products normal form. Part of the memo key.
- ``memoize`` (False): cross-subtree memoization keyed by canonical
  fingerprint + free-edge names + ``expand``. Without it, a node reachable by
  K distinct DAG paths is re-simplified K times; residual nets double the
  path count per layer, so deep-model simplify is exponential in depth even
  though the tree stays small. Opt-in because with the combine passes enabled
  the inflated intermediate tree's working set can exceed memory at depth;
  safe with the combine passes off (the compiler preset sets it True).
  Only active when ``grad_steps`` is inf, so the in-place decrement can never
  make a cached key stale.
- ``provenance`` (unset): optional recorder with a ``record(before, after)``
  method, called on every rewrite (used by the diagram exporter's
  simplify_trace).

Deleted knobs (flag audit, 2026-07): ``associative_products`` and
``associative_sums`` had no callers anywhere (defaults were True and nothing
ever passed False), so nested-product merging and nested-sum flattening are
now unconditional; ``distributed_products`` was never set and only raised
NotImplementedError.
"""

import math
from collections import Counter
from types import ModuleType
from typing import Any, Callable, Optional, cast

from tensorgrad.tensor import (
    Delta,
    Derivative,
    Function,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
    Zero,
    _get_canon,
    _group_edges,
)
from tensorgrad.utils import _MatchEdgesKey

# The Derivative-resolution rule lives with the other differentiation logic.
from tensorgrad.grad import step_derivative


################################################################################
# The engine
################################################################################


def simplify(tensor: Tensor, args: Optional[dict[str, Any]] = None) -> Tensor:
    """Simplify `tensor`: set knob defaults, consult the memo, dispatch to the
    per-type rule, and check shape preservation. See the module docstring for
    the knobs. May rename inner edges but never changes the free edges."""
    if args is None:
        args = {}
    # args["grad_steps"] allows us to control how far we propagate the derivative.
    args.setdefault("grad_steps", float("inf"))
    args.setdefault("sum_combine_terms", True)
    args.setdefault("combine_products", True)
    args.setdefault("factor_components", True)
    args.setdefault("expand_functions", True)
    args.setdefault("expand", False)

    # Cross-subtree memoization (see the module docstring's `memoize` entry).
    # Gated on grad_steps == inf so the in-place decrement in step_derivative
    # (inf - 1 == inf) can never make a cached key stale. The key includes
    # tuple(tensor.edges) so an isomorphic subtree with different free-edge
    # names never returns a result with the wrong edges (the memo-hit path
    # bypasses the shape assert below).
    memo = key = None
    if args.get("memoize") and args["grad_steps"] == float("inf"):
        memo = args.setdefault("_memo", {})
        key = (_get_canon().refined_sort_key(tensor), tuple(tensor.edges), args["expand"])
        if (hit := memo.get(key)) is not None:
            return hit

    rule = _SIMPLIFY_RULES.get(type(tensor))
    if rule is not None:
        result = rule(tensor, args)
    else:
        # Types outside the core set (Expectation; Constant subclasses such
        # as Convolution/Reshape/Affine) use the `_simplify` method protocol,
        # which defaults to the identity.
        result = tensor._simplify(args)

    # Check that the shape is preserved
    assert result.shape == tensor.shape
    _record_simplify_provenance(args, tensor, result)

    if memo is not None:
        memo[key] = result
    return result


def _record_simplify_provenance(args: dict[str, Any], before: Tensor, after: Tensor) -> None:
    recorder = args.get("provenance")
    if recorder is not None:
        recorder.record(before, after)


################################################################################
# Atoms
################################################################################


def keep_atom(t: Tensor, args: dict[str, Any]) -> Tensor:
    """Variable, Delta and Zero are already simple: the identity rewrite.
    (Delta *pair* rules only apply inside a Product; see the Delta pair rules
    below.)"""
    return t


################################################################################
# Rename pushdown
################################################################################


def push_rename_down(t: Rename, args: dict[str, Any]) -> Tensor:
    """Eliminate the (lazy) Rename wrapper: simplify the inner tensor, merge
    chains of Renames into one mapping, and push the rename into the inner
    tensor via its `rename` plumbing. Terminates because the result contains
    no Rename node above the inner tensor (empty mappings are unwrapped)."""
    inner = t.tensor.simplify(args)
    if isinstance(inner, Rename):
        merged = Rename.merge_renames(inner.mapping, t.mapping)
        res = inner.tensor.rename(**merged)
    else:
        res = inner.rename(**t.mapping)
    while isinstance(res, Rename) and not res.mapping:
        res = res.tensor
    return res


################################################################################
# Delta pair rules. These act on the factor list of a Product: whenever two
# factors share an edge and one of a shortlist of local patterns applies, the
# pair is rewritten in place.
################################################################################


def simplify_delta_products(tensors: list[Tensor]) -> list[Tensor]:
    """Run the Delta pair rules on a product's factor list to a fixed point.
    Terminates because every pair rule strictly shrinks the factor list or
    the total edge count (merge removes a factor; identity-matrix removal
    removes a factor; equation rewriting replaces a pair by a strictly
    smaller rhs)."""
    while True:
        tensors, done = _delta_pair_step(tensors)
        if done:
            break
    return tensors


def _delta_pair_step(tensors: list[Tensor]) -> tuple[list[Tensor], bool]:
    """Performs one step of simplification. Returns a new list if changed, or the original if not."""
    for e, ts in _group_edges(tensors).items():
        if len(ts) == 1:
            continue
        t1, t2 = ts
        assert t1.shape[e] == t2.shape[e], f"{t1.shape[e]=} != {t2.shape[e]}, {e=}"

        # Remove t1 and t2 from tensors.  We use the "is" operator, since equality
        # means isomorphic, so it might remove too many unintended tensors.
        other = [t for t in tensors if t is not t1 and t is not t2]

        for simplification in [
            merge_copy_tensors,
            remove_identity_matrix,
            apply_variable_equation,
        ]:
            if (new := simplification(t1, t2, e)) is not None:
                return other + new, False

    return tensors, True


def merge_copy_tensors(t1: Tensor, t2: Tensor, e: str) -> Optional[list[Tensor]]:
    """Two Deltas sharing an edge fuse into one Delta over the symmetric
    difference of their edges (all shared edges are contracted away at once).
    Shrinks: two factors become one."""
    if not (isinstance(t1, Delta) and isinstance(t2, Delta)):
        return None

    # We can't merge order 0 copy tensors, since we now give them a value equal to their size.
    # In principle we could create a new Delta(size1 * size2, []) tensor, but then we start having
    # arbitrary expressions as sizes, which I'm not sure we want yet.
    # Also, merging an order 0 with an order > 0 was never going to work, unless the size of the
    # order 0 Delta was 1, which it probably never is.
    # In either case, this method should only be called if the tensors share an edge, which they
    # can't if one of them has order 0.
    assert t1.order != 0 and t2.order != 0, "Can't merge order 0 Delta tensors"

    # Since the tensors are connected, we can assume they have the same size
    assert t1.size == t2.size, "Contracted Delta tensors must have same size"
    size = t1.size

    # We don't just remove e, but remove all shared edges
    # The amazing thing is that even in the case where all edges disappear, we
    # still get to keep information on the "size" of the Delta tensor.
    return [Delta(size, *(t1.edges ^ t2.edges))]


def remove_identity_matrix(t1: Tensor, t2: Tensor, e: str) -> Optional[list[Tensor]]:
    """Contracting with an identity matrix (an order-2 Delta) is a rename:
    I_{e,f} * T_{...e...} -> T_{...f...}. Shrinks: removes one factor."""
    # If both are Delta's, we use the previous method
    if isinstance(t1, Delta) and isinstance(t2, Delta):
        return None

    # Make t1 the identity matrix
    if isinstance(t2, Delta) and t2.order == 2:
        t1, t2 = t2, t1

    if isinstance(t1, Delta) and t1.order == 2:
        # Find the edge of t1 that's not e
        other_edge = next(iter(set(t1.edges) - {e}))
        # Don't create self loops. We never connect a tensor to itself.
        # Unless it's another Delta, in which case we already handled it above.
        if other_edge not in t2.edges:
            return [t2.rename(**{e: other_edge})]
    return None


def apply_variable_equation(t1: Tensor, t2: Tensor, e: str) -> Optional[list[Tensor]]:
    """General constraint-equation rewriting (see Variable.with_eq_constraint).

    If the 2-factor subnetwork (t1, t2) is isomorphic — free edges matched
    under some renaming σ — to a declared equation's lhs, replace it by the
    equation's rhs renamed by σ. Simplex sums, unit norms and orthogonality
    all arrive here as pairwise patterns, because tensorgrad canonicalizes
    Hadamard squares to pow-Functions and sums to order-1 Delta
    contractions. Sound by construction: a rewrite only fires on a
    verified isomorphism, and the pair's contracted edges are private to
    the pair (Product edges are binary), so splicing rhs is local.
    Terminates because declared equations have a strictly smaller rhs than
    lhs (enforced at declaration).
    """
    from itertools import permutations

    cvars = [v for t in (t1, t2) for v in Variable._find_variables(t) if v._constraints]
    if not cvars:
        return None
    cand = None
    tried = set()
    for v in cvars:
        for lhs, rhs in v._constraints:
            key = v._equation_key(lhs, rhs)
            if key in tried:
                continue
            tried.add(key)
            if not isinstance(lhs, Product) or len(lhs.factors) != 2:
                continue  # only pairwise patterns are matched (documented)
            if cand is None:
                cand = Variable._strip_constraints(Product([t1, t2]))
            cfree, lfree = list(cand.edges), list(lhs.edges)
            if len(cfree) != len(lfree):
                continue
            if sorted(map(str, cand.shape.values())) != sorted(map(str, lhs.shape.values())):
                continue
            # Try name-preserving assignment first (the common, un-renamed case),
            # then all size-consistent bijections of free edges.
            candidates = []
            if set(cfree) == set(lfree):
                candidates.append({a: a for a in lfree})
            candidates += [dict(zip(lfree, p)) for p in permutations(cfree)]
            for sigma in candidates:
                if any(lhs.shape[a] != cand.shape[b] for a, b in sigma.items()):
                    continue
                if cand.is_isomorphic(lhs.rename(**sigma), match_edges=True):
                    return [rhs.rename(**{a: sigma[a] for a in rhs.edges})]
    return None


################################################################################
# Function resolution
################################################################################


def simplify_function(t: Function, args: dict[str, Any]) -> Tensor:
    """Simplify a Function's inputs, pull broadcast Ones factors out of the
    inputs (they commute with the function), and delegate any signature-
    specific rewrites (expansion, pointwise splits, ...) to
    FunctionSignature.simplify. Terminates because the pulled-out factors
    strictly shrink the inputs and signature rewrites are themselves
    terminating (they shrink or expand to fixed known forms)."""
    new_inputs = [inp.simplify(args=args) for inp in t.inputs]

    # Broadcasting can be pulled out of the function.
    pulled_out = []
    new_inputs2 = []
    for inp, es in zip(new_inputs, t.signature.inputs):
        if isinstance(inp, Product):
            new_prod = []
            for u in inp.factors:
                if (
                    isinstance(u, Delta)
                    and u.order == 1
                    and list(u.edges)[0] in inp.edges
                    and list(u.edges)[0] not in es
                ):
                    pulled_out.append(u)
                else:
                    new_prod.append(u)
            new_inputs2.append(Product(new_prod))
        else:
            new_inputs2.append(inp)
    new_inputs = new_inputs2

    res = Function(t.signature, new_inputs, t.shape_out)

    # This results in an extra simplify call to all the children of the function.
    # Maybe we can avoid this somehow?
    old_shape = res.shape
    res = t.signature.simplify(res, args)
    assert res.shape == old_shape, "Function signature simplify should not change shape"

    if pulled_out:
        res = Product([res] + pulled_out)

    return res


################################################################################
# Product assembly
################################################################################


def simplify_product(t: Product, args: dict[str, Any]) -> Tensor:
    """Normalize a Product: simplify factors, annihilate on Zero, flatten
    nested products, pull single-term-Sum weights up, run the Delta pair
    rules and (gated) the pow-combination pass, sort factors canonically, and
    (gated on ``expand``) distribute over Sum factors. Terminates because
    every sub-pass shrinks or canonicalizes: flattening and weight pulling
    strictly reduce nesting, the pair/pow passes run to their own fixed
    points, and the expand recursion re-enters with expand=False."""
    tensors = [f.simplify(args=args) for f in t.factors]

    # If any tensor in a product is 0, so is the whole product
    if any(isinstance(f, Zero) for f in tensors):
        return Zero(_symmetries=None, **t.shape)

    # Combine nested products, unconditionally (the old associative_products
    # knob had no callers; see the flag audit in the module docstring).
    sub_products = [f if isinstance(f, Product) else Product([f]) for f in tensors]
    tensors = Product.merge(sub_products).factors

    # We can do a "small" kind of distributed products, which is handling children that are single sums
    # Also, if a child is a sum with a single element, we can pull the weight up.
    # In general, we can pull out the least common multiple of the weights of the children.
    if single_sums := [f for f in tensors if isinstance(f, Sum) and len(f.terms) == 1]:
        # (cast: membership in single_sums implies f is a Sum)
        tensors = [f if f not in single_sums else cast(Sum, f).terms[0] for f in tensors]
        res_weight = math.prod(f.weights[0] for f in single_sums)
    else:
        res_weight = 1

    def verify_edges(ts: list[Tensor], msg: str = "") -> None:
        cnt = Counter(e for f in ts for e in f.edges)
        assert not cnt or cnt.most_common()[0][1] <= 2, msg

    # Simplify Delta Tensors
    verify_edges(tensors)
    tensors = simplify_delta_products(tensors)
    verify_edges(tensors)

    # Combine / Cancel Product Functions
    if args["combine_products"]:
        from tensorgrad.functions import _PowerFunction

        verify_edges(tensors)
        before = tensors
        tensors = _PowerFunction.simplify_outer(tensors, args)
        verify_edges(tensors, f"{before} -> {tensors}")

    # Deterministic factor order: sort by the seed-stable structural
    # fingerprint, so commutative construction order (and PYTHONHASHSEED)
    # doesn't leak into the simplified result.
    tensors.sort(key=_get_canon().refined_sort_key)

    # Base cases
    if len(tensors) == 1:
        res = tensors[0]
    elif args["expand"]:
        terms = [[]]
        weights = [1]
        for f in tensors:
            # A (lazy) Rename wrapper can hide a Sum factor from the
            # distribution below; push the rename through eagerly, but
            # only here where expand needs to look through it.
            while isinstance(f, Rename) and isinstance(f.tensor, Sum):
                f = f.tensor._rename(**f.mapping)
            if isinstance(f, Sum):
                # Create cartesian product
                terms = [term + [t0] for term in terms for t0 in f.terms]
                weights = [w * w0 for w in weights for w0 in f.weights]
            else:
                for term in terms:
                    term.append(f)
        # Recurse with expand=False to avoid infinite descent, but keep
        # all other caller flags intact.
        res = Sum([Product(ts) for ts in terms], weights).simplify(args={**args, "expand": False})
    else:
        res = Product(tensors)

    if res_weight != 1:
        res = res_weight * res

    return res


################################################################################
# Sum assembly
################################################################################


def simplify_sum(t: Sum, args: dict[str, Any]) -> Tensor:
    """Normalize a Sum: simplify terms, flatten nested sums (weights
    multiply through), merge isomorphic terms by adding their weights (terms
    are matched with edge names, so isomorphic terms with different free-edge
    labels stay separate), drop zeros, and sort terms canonically. Terminates
    because flattening and merging strictly shrink the term list and sorting
    canonicalizes."""
    terms = [term.simplify(args=args) for term in t.terms]

    term_counter = Counter()
    for w, term in zip(t.weights, terms):
        # Flatten nested sums, unconditionally (the old associative_sums knob
        # had no callers; see the flag audit in the module docstring).
        if isinstance(term, Sum):
            for w1, t1 in zip(term.weights, term.terms):
                term_counter[_MatchEdgesKey(t1)] += w * w1
        else:
            term_counter[_MatchEdgesKey(term)] += w

    if args["sum_combine_terms"]:
        # Identify tensors with multiplicity and combine them. We use tensor.canon_with_edges to identify tensors.
        # It is important that isomorphic tensors with different outer edge labels don't get matched. For example
        # (o-i o-<jk) is isomorphic with (o-j o-<ik), but they shouldn't be combined. This example comes up in the
        # Hessian of softmax.
        ws_tensors = [
            (w, key.value)
            for key, w in term_counter.items()
            if w != 0 and not isinstance(key.value, Zero)
        ]
    else:
        ws_tensors = [(w, key.value) for key, w in term_counter.items()]

    # Remove zero tensors or zero weights.
    # Note: This won't change the shape of the tensor, since all summands have been broadcasted.
    ws_tensors = [(w, u) for w, u in ws_tensors if w != 0 and not isinstance(u, Zero)]
    # Base case. Here we can't just return Zero([]), since that would change the signature of the tensor.
    if not ws_tensors:
        return Zero(_symmetries=None, **t.shape)
    # Deterministic term order: sort by the seed-stable (name-sensitive)
    # structural fingerprint, so commutative construction order (and
    # PYTHONHASHSEED) doesn't leak into the simplified result. Positive
    # weights sort first purely for presentation (avoid a leading minus).
    canon = _get_canon()
    ws_tensors.sort(key=lambda wt: (wt[0] < 0, canon.refined_sort_key(wt[1]), str(wt[0])))
    weights, tensors = zip(*ws_tensors)
    # If there is just one tensor with weight 1, we don't need LinearComb
    if weights == (1,):
        return tensors[0]
    return Sum(tensors, weights)


################################################################################
# Dispatch table
################################################################################

# (Callable[[Any, ...]]: each rule takes its concrete node type as the first
# parameter, which a dict value type of Callable[[Tensor, ...]] would reject.)
_SIMPLIFY_RULES: dict[type, Callable[[Any, dict[str, Any]], Tensor]] = {
    Variable: keep_atom,
    Delta: keep_atom,
    Zero: keep_atom,
    Rename: push_rename_down,
    Function: simplify_function,
    Derivative: step_derivative,  # chain-rule stepping lives in tensorgrad.grad
    Product: simplify_product,
    Sum: simplify_sum,
}
