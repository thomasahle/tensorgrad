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
  PLACE by :func:`tensorgrad.autodiff.step_derivative`, so a finite budget is
  shared across the tree (and across repeated ``simplify`` calls reusing the
  same dict). Used by imgtools/to_diagram for step-by-step derivations.
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
NotImplementedError; ``sum_combine_terms`` gated a filter in simplify_sum
that was subsumed by the unconditional one right after it (isomorphic-term
merging itself always happened in the Counter accumulation), so it had no
observable effect even though the compiler preset set it False.
"""

import math
from collections import Counter
from functools import singledispatch
from typing import Any, Optional, cast

from tensorgrad.structure import _children, refined_sort_key
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
    _group_edges,
    _unused_edge_names,
    peel_rename,
)
from tensorgrad.utils import _MatchEdgesKey, _ValueKey, merge_renames

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
        key = (refined_sort_key(tensor), tuple(tensor.edges), args["expand"])
        if (hit := memo.get(key)) is not None:
            return hit

    result = _dispatch_simplify(tensor, args)

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


@singledispatch
def _dispatch_simplify(tensor: Tensor, args: dict[str, Any]) -> Tensor:
    """Dispatch a node to its per-type simplify rule (the rules below register
    themselves).

    The default is the identity: atoms (Variable and Constant — Delta, Zero,
    Convolution, Reshape, Affine) are already simple and deliberately have no
    registered rule. The table is the extension point: types outside the core
    set register their rule on import (tensorgrad/extras/expectation.py
    registers Expectation's); unknown types are treated as already simple
    (the engine's shape assert still guards them)."""
    return tensor


# The chain-rule stepping rule for Derivative lives with the other
# differentiation logic in tensorgrad.grad.
_dispatch_simplify.register(Derivative, step_derivative)


################################################################################
# Rename pushdown
################################################################################


@_dispatch_simplify.register
def push_rename_down(t: Rename, args: dict[str, Any]) -> Tensor:
    """Normalize the (lazy) Rename wrapper: simplify the inner tensor, merge
    chains of Renames into one mapping, drop trivial ones, and materialize
    the mapping via `_rename` for non-composite inners (leaves rename
    cheaply; Derivative/Expectation re-wrap in O(1)).

    A wrapper over a composite (Sum/Product/Function) deliberately STAYS: the
    inner tensor is typically a subexpression shared under several renamings
    (residual networks), and pushing each renaming into a private copy of the
    tree destroys that sharing — measured 4.6x more compiled ops on the
    3-layer minGPT program. Rename is isomorphism-transparent and the
    compiler lowers it as pure edge relabeling, so the wrapper costs nothing;
    structural matchers peel it locally instead (see peel_rename).
    Terminates because the result contains at most one Rename above the
    simplified inner tensor (empty mappings are unwrapped)."""
    inner = t.tensor.simplify(args)
    mapping = t.mapping
    while isinstance(inner, Rename):
        mapping = merge_renames(inner.mapping, mapping)
        inner = inner.tensor
    mapping = {k: v for k, v in mapping.items() if k in inner.edges and k != v}
    if not mapping:
        return inner
    if isinstance(inner, (Sum, Product, Function)):
        return Rename(inner, mapping)
    return inner._rename(**mapping)


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

        for simplification in PAIR_RULES:
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


def sum_one_hot(t1: Tensor, t2: Tensor, e: str) -> Optional[list[Tensor]]:
    """Partition of unity for the gather primitive (#42): contracting a
    one_hot's class edge with a ones vector sums the indicator over all
    classes — 1 wherever idx holds a valid class id, the precondition the
    gather/scatter lowering already relies on (index_select traps on
    out-of-range data). This is what makes cross-entropy gradients of
    IN-PROGRAM one_hot targets target-free (grad = softmax - y needs
    sum_v y[b, v] = 1), the same elimination Variable.with_eq_constraint
    provides for declared simplex variables. Shrinks: the pair becomes
    order(one_hot) - 1 ones vectors."""
    if not (isinstance(t1, Delta) and t1.order == 1):
        t1, t2 = t2, t1
    if not (isinstance(t1, Delta) and t1.order == 1):
        return None
    inner, outer = t2, {}
    if isinstance(inner, Rename):
        outer, inner = inner.mapping, inner.tensor
    if not (isinstance(inner, Function) and inner.signature.name == "one_hot"):
        return None
    (cls,) = inner.shape_out
    if outer.get(cls, cls) != e:
        return None
    return [Delta(s, f) for f, s in t2.shape.items() if f != e]


_CVARS_ATTR = "_has_cvars_v1"


def _has_constrained_vars(t: Tensor) -> bool:
    """Memoized: does a Variable with declared equations occur anywhere under t?

    Cheap gate for apply_variable_equation, which otherwise re-walks the same
    subtrees once per delta-pair probe per sweep (the single hottest call in
    compiler-normalize profiles — ~7.7k bounded walks on constraint-free
    programs). Tensors are immutable, so the cached bit never invalidates;
    a miss costs one bottom-up DAG sweep shared by all later probes."""
    hit = t.__dict__.get(_CVARS_ATTR)
    if hit is not None:
        return hit
    stack = [t]
    while stack:
        node = stack[-1]
        if _CVARS_ATTR in node.__dict__:
            stack.pop()
            continue
        kids = _children(node)
        pending = [k for k in kids if _CVARS_ATTR not in k.__dict__]
        if pending:
            stack.extend(pending)
            continue
        if isinstance(node, Variable):
            bit = bool(node._constraints)
        else:
            bit = any(k.__dict__[_CVARS_ATTR] for k in kids)
        cast(dict, node.__dict__)[_CVARS_ATTR] = bit
        stack.pop()
    return t.__dict__[_CVARS_ATTR]


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

    if not (_has_constrained_vars(t1) or _has_constrained_vars(t2)):
        return None
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



# The pairwise product-rule catalog, tried in order by _delta_pair_step.
# Types outside the core set append their rules on import (the same
# convention as _dispatch_simplify.register): tensorgrad/compiler/affine.py
# appends contract_affines, the structural-tensor contraction closure.
PAIR_RULES: list = [
    merge_copy_tensors,
    remove_identity_matrix,
    apply_variable_equation,
    sum_one_hot,
]


@_dispatch_simplify.register
def simplify_function(t: Function, args: dict[str, Any]) -> Tensor:
    """Simplify a Function's inputs, pull broadcast Ones factors out of the
    inputs (they commute with the function), and delegate any signature-
    specific rewrites (expansion, pointwise splits, ...) to
    FunctionSignature.simplify. Terminates because the pulled-out factors
    strictly shrink the inputs and signature rewrites are themselves
    terminating (they shrink or expand to fixed known forms)."""
    new_inputs = [inp.simplify(args=args) for inp in t.inputs]

    # Broadcasting can be pulled out of the function.
    pulled_out: list[Tensor] = []
    new_inputs2: list[Tensor] = []
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


def merge_products(products: list[Product]) -> Product:
    """Merge several Product tensors into one, renaming inner edges so they
    are distinct."""
    used_edges = {e for p in products for e in p.edges}
    res = []
    for p in products:
        inner_edges = {e for t in p.factors for e in t.edges if e not in p.edges}
        rename = _unused_edge_names(inner_edges, used_edges)
        for t in p.factors:
            res.append(t.rename(**rename))
        used_edges.update(rename.values())  # Later renames should not clash with this one
    return Product(res)



@_dispatch_simplify.register
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
    tensors = merge_products(sub_products).factors

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
        # The renames above (merge_products inner-edge freshening, identity-
        # matrix removal) are lazy on composites, so factors can be Rename-
        # wrapped; peel one level so the pow pass sees pow-Functions rather
        # than Rename nodes. Deliberately NOT done outside this branch: the
        # compiler preset turns combine_products off and relies on wrappers
        # staying put — peeling rebuilds each factor's shell and measurably
        # breaks subtree sharing on deep-model programs.
        tensors = [peel_rename(f) for f in tensors]
        tensors = _PowerFunction.simplify_outer(tensors, args)
        verify_edges(tensors, f"{before} -> {tensors}")

    # Deterministic factor order: sort by the seed-stable structural
    # fingerprint, so commutative construction order (and PYTHONHASHSEED)
    # doesn't leak into the simplified result.
    tensors.sort(key=refined_sort_key)

    # Base cases
    if len(tensors) == 1:
        res = tensors[0]
    elif args["expand"]:
        terms: list[list[Tensor]] = [[]]
        weights = [1]
        for f in tensors:
            # A (lazy) Rename wrapper can hide a Sum factor from the
            # distribution below; push the rename through eagerly, but
            # only here where expand needs to look through it.
            while isinstance(f, Rename) and isinstance(f.tensor, Sum):
                f = peel_rename(f)
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


@_dispatch_simplify.register
def simplify_sum(t: Sum, args: dict[str, Any]) -> Tensor:
    """Normalize a Sum: simplify terms, flatten nested sums (weights
    multiply through), merge isomorphic terms by adding their weights (terms
    are matched with edge names, so isomorphic terms with different free-edge
    labels stay separate), drop zeros, and sort terms canonically. Terminates
    because flattening and merging strictly shrink the term list and sorting
    canonicalizes."""
    terms = [term.simplify(args=args) for term in t.terms]

    # Merge terms by accumulating their weights. The default merge quotient
    # is name-fixed isomorphism (_MatchEdgesKey): outer edge labels must
    # correspond, so (o-i o-<jk) stays separate from its (o-j o-<ik)
    # isomorph — the softmax-Hessian case. Under the sz_cancel knob (the
    # compiler's normalize preset) the quotient COARSENS to value identity
    # (fingerprint.py, exact evaluation mod P): value-equal terms share a
    # Counter slot regardless of structure, so the ordinary `w != 0` filter
    # below performs semantic cancellation — cancellations that would need
    # "expanding just the right factors", found without expansion. This is
    # sound as a coarsening because name-fixed-isomorphic terms always
    # fingerprint equal (szfp draws declared-symmetric variables from the
    # symmetric subspace), and unlowerable terms (fp None) keep the
    # structural key. Merging value-equal but float-DIFFERENT groupings is
    # the compiler preset's accepted trade — the IR consolidation pass makes
    # the same one — which is why the default preset keeps the purely
    # structural quotient.
    from tensorgrad import fingerprint as _fp

    def _key(u: Tensor) -> Any:
        if args.get("sz_cancel"):
            f = _fp.szfp(u)
            if f is not None:
                return _ValueKey(f, u)
        return _MatchEdgesKey(u)

    term_counter: Counter = Counter()
    reps: dict = {}
    for w, term in zip(t.weights, terms):
        # Flatten nested sums, unconditionally (the old associative_sums knob
        # had no callers; see the flag audit in the module docstring).
        for w1, t1 in zip(term.weights, term.terms) if isinstance(term, Sum) else [(1, term)]:
            key = _key(t1)
            term_counter[key] += w * w1
            # Deterministic representative per merge class: keep the
            # refined_sort_key minimum, not the first-seen (construction
            # order must not leak into the simplified result).
            prev = reps.get(key)
            if prev is None or refined_sort_key(t1) < refined_sort_key(prev):
                reps[key] = t1

    # Remove zero tensors or zero weights.
    # Note: This won't change the shape of the tensor, since all summands have been broadcasted.
    ws_tensors = [(w, reps[key]) for key, w in term_counter.items()]
    ws_tensors = [(w, u) for w, u in ws_tensors if w != 0 and not isinstance(u, Zero)]

    if args.get("sz_cancel") and ws_tensors:
        # Value-zero terms contribute nothing at any weight, and a sum whose
        # TOTAL value is zero (cross-slot relations: A - B - C with
        # A == B + C) collapses whole. Partial subset relations among
        # survivors are subset-sum territory — the e-graph's job, not a
        # filter's. The whole-sum test only runs when every term carries the
        # SAME weight-symbol support: a full cancellation must hold for all
        # dims, so mismatched supports (x/n vs x/5) cannot legitimately
        # cancel — and skipping them is what keeps a small-dim draw
        # coincidence from collapsing them wrongly.
        ws_tensors = [(w, u) for w, u in ws_tensors if not _fp.is_zero(u)]
        supports = {(f := _fp.szfp(u)) and f[0] for _, u in ws_tensors}
        if (
            len(ws_tensors) >= 2
            and len(supports) == 1
            and None not in supports
            and _fp.is_zero(Sum([u for _, u in ws_tensors], [w for w, _ in ws_tensors]))
        ):
            ws_tensors = []
    # Base case. Here we can't just return Zero([]), since that would change the signature of the tensor.
    if not ws_tensors:
        return Zero(_symmetries=None, **t.shape)
    # Deterministic term order: sort by the seed-stable (name-sensitive)
    # structural fingerprint, so commutative construction order (and
    # PYTHONHASHSEED) doesn't leak into the simplified result. Positive
    # weights sort first purely for presentation (avoid a leading minus).
    ws_tensors.sort(key=lambda wt: (wt[0] < 0, refined_sort_key(wt[1]), str(wt[0])))
    weights, tensors = zip(*ws_tensors)
    # If there is just one tensor with weight 1, we don't need LinearComb
    if weights == (1,):
        return tensors[0]
    return Sum(tensors, weights)


