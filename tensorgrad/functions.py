import itertools
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from typing import AbstractSet, Any, Callable, Collection, Iterable, Iterator, Optional, Sequence, Union, cast

import sympy
from sympy import Symbol

from tensorgrad.tensor import (
    Constant,
    Delta,
    Function,
    FunctionSignature,
    Ones,
    Product,
    Rename,
    Scalar,
    Sum,
    Tensor,
    Variable,
    Zero,
    _unused_edge_names,
    peel_rename,
)
from tensorgrad.simplify import merge_products, simplify_delta_products
from tensorgrad.utils import DisjointSets, _MatchEdgesKey

# We include a "sum" function, which overloads the python sum. So we keep a reference
# here to the builtin, so we can use it in this module
_sum = sum
DimType = None | str | Iterable[str]

# We mostly try to follow the behavior of pytorch's named tensors:
# https://pytorch.org/docs/stable/name_inference.html


def taylor(f: Tensor, wrt: Variable, eps: Tensor, n: int) -> Tensor:
    """Return the nth order Taylor approximation of f at x+eps."""
    if eps.shape != wrt.shape:
        raise ValueError("eps must have the same shape as wrt.")
    terms = [f]
    for _ in range(n):
        # Taking the derivative of f with respect to wrt creates new edges that match the shape of eps.
        connection_names = _unused_edge_names(wrt.edges, f.edges)
        # To save time, we reuse the derivatives we've already computed
        f = f.grad(wrt, new_names=connection_names) @ eps.rename(**connection_names)
        terms.append(f)
    return Sum(terms, [Fraction(1, math.factorial(i)) for i in range(n + 1)])


def frobenius2(t: Tensor) -> Tensor:
    return Product([t, t])


def _is_even_permutation(permutation: Sequence[int]) -> bool:
    """
    Checks if given permutation is even.
    >>> is_even_permutation(range(10))
    True
    >>> is_even_permutation(range(10)[::-1])
    False
    """
    if len(permutation) == 1:
        return True
    transitions_count = 0
    for index, element in enumerate(permutation):
        for next_element in permutation[index + 1 :]:
            if element > next_element:
                transitions_count += 1
    return not (transitions_count % 2)


# TODO: Make a SymmetrizeFunctionSignature class. This will be extra nice
# when FunctionSignature supports symmetries, since we can then symmetrize stuff
# without having to pay for the super-exponential cost of all permutations.
def symmetrize(t: Tensor, dims: DimType = None, signed: bool = False) -> Tensor:
    """Sum over all permutations of the edges."""
    dims = parse_dim(t.edges, dims, none_is="all")
    edges = list(dims)
    res: list[Tensor] = []
    weights: list[int] = []
    for i_perm in itertools.permutations(range(len(edges))):
        perm = [edges[i] for i in i_perm]
        res.append(t.rename(**dict(zip(edges, perm))))
        if signed:
            weights.append(1 if _is_even_permutation(i_perm) else -1)
        else:
            weights.append(1)
    return Sum(res, weights)


def graph(dot_graph: str, **vars: Tensor) -> Tensor:
    """
    Create a tensor network using a DOT-like graph syntax.

    This function allows you to define tensor networks using a syntax similar to the DOT graph description language.
    It provides a more intuitive way to specify tensor contractions and operations compared to traditional einsum notation.

    Parameters:
    -----------
    dot_graph : str
        A string describing the tensor network using DOT-like syntax. Each line represents an edge or connection
        between tensors. Tensor names are separated by edge specifications, which are hyphen-delimited lists of
        edge names.

    **vars : dict
        Keyword arguments representing the tensors used in the graph. Each key is the tensor name used in the
        dot_graph, and the corresponding value is the actual tensor object.

    Returns:
    --------
    Product
        A Product object representing the tensor network described by the input graph.

    Syntax:
    -------
    - Tensors are represented by their names (e.g., 'A', 'B', 'X').
    - Edge connections are represented by hyphens: '-'. For example, 'A -i- B' connects edge 'i' of A to edge 'i' of B.
    - You can connect edges with different names. For example, 'A -i-j- B' connects edge 'i' of A to edge 'j' of B.
    - Delta tensors are created automatically when a name starting with '*', like '*3' is used.
    - Lines can be separated by newlines or semicolons.
    - Edges not mentioned in the graph are broadcasted.

    Examples:
    ---------
    1. Matrix multiplication:
    >>> i, j, k = symbols("i j k")
    >>> A = Variable("A", i, j)
    >>> B = Variable("B", j, k)
    >>> result = graph("A -j- B", A=A, B=B)

    2. Trace
    >>> i = symbols("i")
    >>> X = Variable("X", i, j=i)
    >>> result = graph('''
    ...     X -i- X
    ...     X -j- X
    ... ''', X=X)

    3. Element-wise multiplication:
    >>> i, j = symbols("i j")
    >>> X = Variable("X", i, j)
    >>> Y = Variable("Y", i, j)
    >>> result = graph('''
    ...     -i- *0 -i- X -j- *1 -j-
    ...         *0 -i- Y -j- *1
    ... ''', X=X, Y=Y)

    4. Frobenius norm (squared)
    >>> i, j = symbols("i j")
    >>> X = Variable("X", i, j)
    >>> result = graph('''
    ...     X1 -i- X2
    ...     X1 -j- X2
    ... ''', X1=X, X2=X)

    Raises:
    -------
    ValueError
        If the graph specification is invalid, such as using undefined variables,
        invalid edge names, or inconsistent hyperedge sizes.
    """
    vars = vars.copy()

    # Parse the graph, converting long lines into a list of edges:
    # st. "A -i-j- B" -> ("A", "i", "j", "B")
    # and "A -i-" -> ("A", "i", "i", None)
    # and "A -i- B - C" -> [("A", "i", "i", "B"), ("B", "i", "i", "C")]
    # and "*1 - *2" -> ("*1", None, None, "*2")
    lines = re.split(r"[\n;]", dot_graph.strip())
    # (Any: rows are (var_or_None, edge, edge, var_or_None); the *edge_names
    # splat keeps the tuple length opaque to the type checker.)
    edges: list[tuple[Any, ...]] = []
    for line in lines:
        parts = line.split()
        last_var = None
        for i in range(len(parts)):
            if parts[i].startswith("-"):
                _, *edge_names, _ = parts[i].split("-")
                if len(edge_names) == 1:
                    edge_names = [edge_names[0]] * 2
                next_var = parts[i + 1] if i + 1 < len(parts) else None
                edges.append((last_var, *edge_names, next_var))
            else:
                last_var = parts[i]

    # Look for edges, e, not mentioned in the graph, and create "X -e- " lines for them
    for v_name, v in vars.items():
        v_unused_edges = set(v.edges)
        for v0, e0, e1, v1 in edges:
            if v0 == v_name:
                v_unused_edges -= {e0}
            if v1 == v_name:
                v_unused_edges -= {e1}
        for e in v_unused_edges:
            edges.append((v_name, e, e, None))

    # Keep track of the hyperedges and their sizes
    # Keys are *i for some i, values are lists of edge names/symbols
    hyperedges: dict[str | tuple[str, str], list[str]] = defaultdict(list)
    hypersizes: DisjointSets[Any, Symbol] = DisjointSets()

    # Keep track of free edge names we can use
    used_edges: set[str] = {e for v in vars.values() for e in v.edges}
    free_edges: Iterator[str] = (f"e{i}" for i in itertools.count() if f"e{i}" not in used_edges)
    free_hyper_edges: Iterator[str] = (f"h{i}" for i in itertools.count() if f"h{i}" not in used_edges)

    for v0, e0, e1, v1 in edges:
        # Syntax checking
        for v, e in ((v0, e0), (v1, e1)):
            if v and not v.startswith("*"):
                if v not in vars:
                    raise ValueError(f"Variable {v} not found in vars")
                if e not in vars[v].edges:
                    raise ValueError(f"Edge {e} not found in variable {v}({vars[v].edges})")
        if v0 is None and v1 is None:
            raise ValueError("Cannot have two free edges")

        # The case 'X -i-' or '-i- X', where an edge is free
        elif v0 is None or v1 is None:
            v, e, eo = (v1, e1, e0) if v0 is None else (v0, e0, e1)
            assert v is not None  # the both-None case was rejected above
            # In the case "*0 -i-" we add the edge to the hyperedge
            if v.startswith("*"):
                hyperedges[v].append(e)
            # Otherwise we create a new hyperedge and add the edge to it
            # This allows us to keep track of broadcasting
            else:
                he = ("external", eo)
                c = next(free_edges)
                if eo not in hyperedges[he]:
                    hyperedges[he].append(eo)
                hyperedges[he].append(c)
                hypersizes[he] = vars[v].shape[e]
                vars[v] = vars[v].rename(**{e: c})

        # The case *0 - *1, that is, two copy edges
        elif v0.startswith("*") and v1.startswith("*"):
            e = next(free_edges)
            hyperedges[v0].append(e)
            hyperedges[v1].append(e)
            hypersizes.union(v0, v1)  # Make sure they have the same size

        # The case *0 -i- X, where we have a single copy edge
        elif v0.startswith("*") or v1.startswith("*"):
            v, e, he = (v0, e0, v1) if v1.startswith("*") else (v1, e1, v0)
            c = next(free_edges)
            hyperedges[he].append(c)
            hypersizes[he] = vars[v].shape[e]  # Fails if not compatible
            vars[v] = vars[v].rename(**{e: c})

        # The case where a variable is connected to itself, like 'A -i- A'
        elif v0 == v1:
            if e0 == e1:
                raise ValueError("Cannot have a self loop on a single edge")
            c0, c1 = next(free_edges), next(free_edges)
            he = next(free_hyper_edges)
            hyperedges[he].extend([c0, c1])
            hypersizes[he] = vars[v0].shape[e0]
            vars[v0] = vars[v0].rename(**{e0: c0, e1: c1})

        # A standard connection between two variables, like 'A -i- B'
        else:
            e = next(free_edges)
            vars[v0] = vars[v0].rename(**{e0: e})
            vars[v1] = vars[v1].rename(**{e1: e})

    copies = []
    for he, he_edges in hyperedges.items():
        size = hypersizes.get(he)
        if size is None:
            raise ValueError(f"Hyperedge {he} has no size")
        if len(he_edges) != len(set(he_edges)):
            raise ValueError("Hyperedges must be disjoint")
        copies.append(Delta(size, *he_edges))

    return Product(copies + list(vars.values()))


def kronecker(*tensors: Tensor) -> Tensor:
    # Basically just rename everything to be distinct, then contraction
    # Note: This function returns the tensor product, which is different from the
    #       Kronecker product as often described in the literature. To get the
    #       Kronecker product you have to flatten the output tensors.
    dis_tensors, _renames = _make_distinct(*tensors)
    return Product(dis_tensors)


def diag(t: Tensor, new_edges: list[str]) -> Tensor:
    """If `t` is a vector, this creates a diagonal tensor with `t` and creates a diagonal.
    In einsum that means "i->iii".
    If `t` is a higher order tensor, with all dims the same size, this extracts the diagonal as a vector.
    In einsum that means "iii->i".
    """
    # Rename the edges to be distinct from the new edges
    (t,), _renames = _make_distinct(t, used_names=new_edges)

    if not t.shape:
        # We can't just return t @ Delta(new_edges), since we don't know the size of the new edges.
        raise ValueError("Cannot take the diagonal of a scalar.")

    edges, sizes = zip(*t.shape.items())
    if len(set(sizes)) != 1:
        raise ValueError("All dimensions must be the same size for the diagonal.")

    return t @ Delta(sizes[0], *edges, *new_edges)


def trace(tensor: Tensor) -> Tensor:
    if not tensor.edges:
        return tensor
    return diag(tensor, [])


def parse_dim(tensor_edges: AbstractSet[str], dim: DimType = None, none_is: str = "error") -> set[str]:
    if dim is None:
        if none_is == "all":
            dim = frozenset(tensor_edges)
        else:
            raise ValueError("Dimension(s) must be set")
    if isinstance(dim, str):
        dim = {dim}
    if not isinstance(dim, set):
        dim = set(dim)
    if not dim.issubset(tensor_edges):
        raise ValueError(f"{dim=} must be a subset of {tensor_edges}")
    return dim


def sum(tensor: Tensor, dim: DimType = None, keepdims: bool = False) -> Tensor:
    """Sum the tensor over the given dimensions."""
    dim = parse_dim(tensor.edges, dim, none_is="all")
    out = Product([tensor] + [Delta(tensor.shape[e], e) for e in dim])
    # Optionally broadcast back to orignal shape
    if keepdims:
        return out @ Ones(**{e: tensor.shape[e] for e in dim})
    return out


def _make_distinct(
    *tensors: Tensor, used_names: Optional[Iterable[str]] = None
) -> tuple[list[Tensor], list[dict[str, str]]]:
    """
    Ensure that the tensors have distinct edge names.

    Args:
        *tensors: A sequence of tensors.
        used_names: Additional names that must be avoided.

    Returns:
        A tuple containing:
        - List of tensors with renamed edges
        - List of rename mappings used for each tensor
    """
    # Delta the set, so we don't modify the input
    used_names = set() if used_names is None else set(used_names)
    # No reason to rename the names that are unique
    cnt = Counter(e for t in tensors for e in t.edges)
    unique_names = {e for e, c in cnt.items() if c == 1}
    # Unless the names are already used, then we can't save them
    unique_names -= used_names
    used_names |= unique_names

    res, renames = [], []
    for i, t in enumerate(tensors):
        edges = t.edges - unique_names
        rename = _unused_edge_names(edges, used_names, suffix=f"_{i}")
        # Make sure all t.edges are present in rename
        rename.update({e: e for e in unique_names})
        used_names.update(rename.values())
        res.append(t.rename(**rename))
        renames.append(rename)
    return res, renames


def prod(*tensors: Tensor) -> Tensor:
    """Element-wise product of tensors."""
    shared_edges = set.intersection(*[set(t.edges) for t in tensors])
    for e in shared_edges:
        if not all(t.shape[e] == tensors[0].shape[e] for t in tensors):
            raise ValueError(f"Shapes of tensors don't match for edge {e}")

    # Rename all edges to be distinct, and use names that won't be in the final output
    # Also doesn't rename unique names, which is good, since those will also be in the output
    ts, renames = _make_distinct(*tensors, used_names=shared_edges)

    # Element-wise (Hadamard) product is easy to implement using Delta tensors
    ts += [Delta(tensors[0].shape[e], e, *(rn[e] for rn in renames)) for e in shared_edges]

    res = Product(ts)
    assert res.shape == {e: s for t in tensors for e, s in t.shape.items()}
    return res


def mean(tensor: Tensor, dim: DimType = None, keepdims: bool = False) -> Tensor:
    dim = parse_dim(tensor.edges, dim, none_is="all")
    s = sum(tensor, dim, keepdims)
    normalization = 1
    for e in dim:
        normalization @= Delta(tensor.shape[e])
    return s / normalization


def dot(t1: Tensor, t2: Tensor, dim: str | tuple[str, str]) -> Tensor:
    """Contract two tensors along the given dimensions, broadcasting over the remaining shared edges.
    If the dimension is a tuple of two strings, the first string is the dimension of t1
    and the second is the dimension of t2."""
    # Like https://pytorch.org/docs/stable/generated/torch.tensordot.html
    # TODO: Should support a list of strings, or a tuple[list[str], list[str]]
    if isinstance(dim, str):
        dim = (dim, dim)
    if len(dim) != 2 or not all(isinstance(d, str) for d in dim):
        raise ValueError(f"Dot product requires one or two dimensions, got {dim=}")
    if dim[0] not in t1.edges or dim[1] not in t2.edges:
        raise ValueError(f"Edges {dim} must be in the tensors")
    free_name = dim[0] + "_"
    while free_name in t1.edges | t2.edges:
        free_name += "_"
    prod = t1.rename(**{dim[0]: free_name}) * t2.rename(**{dim[1]: free_name})
    return sum(prod, free_name)


def multi_dot(ts: list[Tensor], dims: tuple[str, str]) -> Tensor:
    """
    Compute the dot product of two or more tensors.
    """
    if not ts:
        return Ones()
    if any(v for v in ts if not isinstance(v, Tensor)):
        raise ValueError(f"All arguments must be tensors, got {ts}")
    prod = ts[0]
    assert isinstance(prod, Tensor)
    for t2 in ts[1:]:
        # We use the right edge of t and the left edge of t2
        prod = dot(prod, t2, dim=(dims[1], dims[0]))
        assert isinstance(prod, Tensor)
    return prod


def contract(ts: list[Tensor], inputs: list[dict[str, str]], output: set[str]) -> Tensor:
    # Like einsum, but with tensors
    # CSRz = ctg.array_contract(
    #    arrays=(x, y),
    #    inputs=[{'i': 'i', 'j': 'k'}, {'i': 'k', 'j': 'j'}],
    #    output={'i', 'j'},
    # )
    raise NotImplementedError("Not clear this is useful")


class _ScaleFunction(FunctionSignature):
    def __init__(self, inner: FunctionSignature, alpha: Scalar):
        # Represents alpha * inner(x, ...)
        # This mostly exists to help represent the PowerFunction derivative
        self.name = f"{alpha} * {inner.name}"
        self.edges = inner.edges
        self.inputs = inner.inputs
        self.inner = inner
        self.alpha = alpha

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        return _ScaleFunction(self.inner.derivative(i, new_edges), self.alpha)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        assert f.signature is self
        return self.alpha * Function(self.inner, f.inputs, f.shape_out).simplify(args)


class _PowerFunction(FunctionSignature):
    def __init__(self, k: Union[int, Fraction]):
        self.name = f"pow({k=})"
        self.edges = frozenset()
        self.inputs = (frozenset(),)
        self.k = k

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        assert i == 0 and (new_edges is None or not new_edges)
        return _ScaleFunction(_PowerFunction(self.k - 1), self.k)

    def simplify(self, func: Function, args: dict[str, Any]) -> Tensor:
        assert func.signature is self
        assert len(func.inputs) == 1, "pow should only have one input"
        (inner,) = func.inputs
        # Renaming is lazy on composites, so the input may be Rename-wrapped;
        # every rewrite below dispatches on the input's structure, so peel
        # first (free edge names are preserved; pow has no output edges, so
        # the peel is never irreducible). When no rewrite fires we return the
        # original `func`, keeping the wrapped (shared) input intact.
        inner = peel_rename(inner)

        if self.k == 0:
            return Ones(**func.shape)
        if self.k == 1:
            return inner.simplify(args)

        # The pow function is multiplicative, so we can pull components out of a product apart.
        # Ie. (A * B)^k = A^k * B^k
        if isinstance(inner, Product):
            assert func.shape_out == {}
            new_comps = [Function(self, (comp,), {}) for comp in inner.components()]
            # We only apply this if we actually got more than one component
            if len(new_comps) > 1:
                return Product(new_comps).simplify(args)
            # More generally, factors joined only *diagonally* — through copy tensors that
            # also carry a free (broadcast) edge — multiply pointwise, so pow distributes
            # over them too: pow(Delta(b, "b,b1,b2") A_b1 B_b2w, k)
            #             = Delta(b, "b,b1,b2") pow(A, k)_b1 pow(B, k)_b2w.
            # E.g. softmax = Delta(b,..) exp(x) pow(Z, -1), so pow(softmax, -1) splits
            # into Z * pow(exp(x), -1), whose parts can cancel in the ambient product.
            # Only under expand=True: it rewrites the rational layer aggressively, which
            # pays off when products are being distributed so the split parts meet their
            # cancellation partners in a flat factor list.
            if args["expand"] and (split := self._pointwise_split(inner)) is not None:
                joins, groups = split
                return Product(joins + [Function(self, (g,), {}) for g in groups]).simplify(args)

        # We can pull out the weight of a sum if it's just a single tensor
        if isinstance(inner, Sum) and len(inner.terms) == 1:
            (w,) = inner.weights
            (t,) = inner.terms
            return Function(self, (t,), {}) * (w**self.k)

        # Base cases. We could add Copy(d)**k = Copy(d**k) here, but we're trying to keep the
        # mathematical expressions in tensorgrad space, rather than in sympy space.
        if isinstance(inner, Zero) or isinstance(inner, Product) and not inner.factors:
            return inner

        # Combine nested pows. Re-simplify, since e.g. pow(pow(x, -1), -1) combines
        # to pow(x, 1), which should further unwrap to just x.
        if isinstance(inner, Function) and isinstance(inner.signature, _PowerFunction):
            return Function(_PowerFunction(k=inner.signature.k * self.k), inner.inputs, func.shape_out).simplify(
                args
            )

        return func

    @staticmethod
    def _pointwise_split(inner: Product) -> tuple[list[Tensor], list[Product]] | None:
        """Split a connected product into diagonally-joined groups.

        A Delta factor with at least one free edge never sums: it aligns the indices of
        its neighbors diagonally (and, when it has two or more free edges, masks
        off-diagonal entries with zeros, where we use the convention 0^k = 0, matching
        pow(Zero) = Zero). Removing all such "joining" deltas partitions the remaining
        factors into groups that combine purely elementwise, so an elementwise power of
        the product distributes over the groups. Deltas with no free edge are genuine
        contractions (sums) and stay inside their group.

        Returns (joining_deltas, groups) or None if there is nothing to split.
        """
        free = inner.edges
        joins: list[Tensor] = []
        rest: list[Tensor] = []
        for t in inner.factors:
            (joins if isinstance(t, Delta) and set(t.edges) & set(free) else rest).append(t)
        if not joins or len(rest) < 2:
            return None
        groups = Product(rest).components()
        if len(groups) < 2:
            return None
        return joins, groups

    @classmethod
    def simplify_outer(cls, tensors: list[Tensor], args: Optional[dict[str, Any]] = None) -> list[Tensor]:
        """Simplify a product by combining pow functions.

        This canonicalizes the "rational layer" of the product: the multiset of base
        subexpressions with signed multiplicities (pow(x, k) contributes (x, k), a bare
        factor contributes (x, 1)) is collected, equal bases are identified up to
        isomorphism (also through hyperedge-connected components), and combined:
        pow(x, a) * pow(x, b) -> pow(x, a + b), including cancellations like
        pow(x, -1) * x -> 1.
        """
        # This is not a general FunctionSignature method, but a special case for pow.
        assert args is not None, "simplify args are required"
        original_edges = Product(tensors).edges

        # Combining powers may leave behind Ones caps that must be re-absorbed into the
        # copy tensors before further combinations become visible, so we iterate the
        # passes to a fixed point (with a safety cap on the number of rounds).
        for _ in range(10):
            before = tensors

            # Combine pow(x, k) * pow(x, l) = pow(x, k + l), anchored at existing pows.
            tensors = cls._combine_powers(tensors)
            assert Product(tensors).edges == original_edges

            if args["factor_components"]:
                # Combine  pow(x, k) * x = pow(x, k + 1)  or cancellations  pow(x, -1) * x = 1
                # (a) through shared hyperedges, and (b) between disjoint components.
                # The hyperedge pass must run first, since the component pass may bundle
                # a connected graph into a single Product factor, hiding its copy tensors.
                # It also needs a flat factor list (e.g. Ones caps emitted above are
                # nested Products) to recognize the copy tensors.
                tensors = cls._flatten(tensors)
                tensors = cls._combine_hyperedge_components(tensors)
                assert Product(tensors).edges == original_edges
                tensors = cls._combine_components(tensors)
                assert Product(tensors).edges == original_edges

            # We have to merge products here because we might otherwise have undone part
            # of the simplification.
            tensors = cls._flatten(tensors)
            # Re-contract the copy tensors, so Ones caps left behind by cancellations get
            # absorbed (e.g. Delta(i, "i,a,b") * Ones(a) -> Delta(i, "i,b")).
            tensors = simplify_delta_products(tensors)
            assert Product(tensors).edges == original_edges

            if len(tensors) == len(before) and all(t1 == t2 for t1, t2 in zip(tensors, before)):
                break

        return tensors

    @staticmethod
    def _flatten(tensors: list[Tensor]) -> list[Tensor]:
        """Flatten nested Products into a single factor list. merge_products only
        flattens one level, so repeat until no factor is itself a Product.
        Merge freshens clashing inner edges via `rename`, which is lazy on
        composites — peel the resulting wrappers so the combination passes
        keep seeing pow-Functions (and copy tensors) rather than Rename nodes."""
        while any(isinstance(t, Product) for t in tensors):
            tensors = merge_products([Product([t]) if not isinstance(t, Product) else t for t in tensors]).factors
            tensors = [peel_rename(t) for t in tensors]
        return tensors

    @staticmethod
    def _is_trivial_scalar(t: Tensor) -> bool:
        """True for the scalar 1, represented as an empty Product."""
        return isinstance(t, Product) and not t.factors

    @staticmethod
    def _is_idempotent_power(t: Tensor, power: Union[int, Fraction]) -> bool:
        """True when pow(t, power) == t, so the base can be passed through unchanged.
        Copy tensors of order >= 1 have only 0/1 entries, so any positive elementwise
        power is a no-op; order-1 Deltas (Ones caps) are all-ones, so every power is.
        Note: order-0 Deltas are scalars equal to their size and must NOT be passed
        through (pow(Delta(d), -1) = 1/d != d)."""
        if not isinstance(t, Delta) or t.order < 1:
            return False
        return power >= 1 or t.order == 1

    @staticmethod
    def _combinable(ts: list[tuple[Union[int, Fraction], Tensor]]) -> bool:
        """Whether a class of equal bases should actually be combined. Creating a new
        pow of a Sum base from bare occurrences (e.g. (x+y) @ (x+y) -> pow(x+y, 2)) is
        sum-level polynomial rewriting, which downstream tools like collect() do not
        expect; we only combine Sum bases when a pow is already involved."""
        if len(ts) <= 1:
            return False
        (_, t0) = ts[0]
        if isinstance(t0, Sum) and all(p == 1 for p, _ in ts):
            return False
        return True

    @classmethod
    def _emit_combined(cls, out: list[Tensor], ts: list[tuple[Union[int, Fraction], Tensor]]) -> None:
        """Append pow(base, sum of multiplicities) for a class of equal bases,
        plus Ones caps replacing the absorbed occurrences (so no edges are left
        unattached on adjacent copy tensors)."""
        if not cls._combinable(ts):
            # Re-emit the original occurrences unchanged.
            out.extend(pow(t, p) for p, t in ts)
            return
        k = _sum(k for k, t in ts)
        (_, t0) = ts[0]  # All the tensors in the class are isomorphic
        combined = pow(t0, k)
        # Take care not to add empty Ones/Products, as it could make our method
        # loop forever as it collects powers of nothing.
        if not cls._is_trivial_scalar(combined):
            out.append(combined)
        for _, t in ts[1:]:
            if t.shape:
                out.append(Ones(**t.shape))

    @classmethod
    def _combine_powers(cls, tensors: list[Tensor]) -> list[Tensor]:
        # Find an existing power, like pow(X, k=2), and merge other powers of X,
        # or instances of X, into it.
        seen = set()
        while True:
            # Find the next power function we haven't seen yet
            try:
                i, t = next(
                    (i, t)
                    for i, t in enumerate(tensors)
                    if isinstance(t, Function) and isinstance(t.signature, _PowerFunction) and t not in seen
                )
            except StopIteration:
                break
            seen.add(t)

            # (cast: the generator above selected t with a _PowerFunction signature)
            power = cast(_PowerFunction, t.signature).k
            (inner,) = t.inputs
            if cls._is_trivial_scalar(inner):
                # pow(1, k) = 1; don't anchor on powers of nothing.
                tensors = tensors[:i] + tensors[i + 1 :]
                continue

            hyperedges = {
                e: min(c.edges)
                for c in tensors
                if isinstance(c, Delta) and c.edges & inner.edges
                for e in c.edges
            }
            partition = defaultdict(list)

            # We can't just call t.rename(**hyperedges) here, since some tensors might be adjacent to the
            # same hyperedge multiple times. Instead we give MatchEdgesKey the names, and it will perform
            # the isomorphism check correctly.
            partition[_MatchEdgesKey(inner, **hyperedges)].append((power, inner))

            # We remove the tensor, as well as all Delta that it's connected to,
            # which makes the rest of the graph fall apart.
            others = tensors[:i] + tensors[i + 1 :]
            copys: list[Tensor] = [t for t in others if isinstance(t, Delta) and t.edges & inner.edges]
            others = [t for t in others if not (isinstance(t, Delta) and t.edges & inner.edges)]
            passthrough: list[Tensor] = []
            for comp in Product(others).components():
                power = 1
                # Unwrap (nested) single-factor products, e.g. Ones caps emitted by an
                # earlier anchor of this loop.
                while isinstance(comp, Product) and len(comp.factors) == 1:
                    (comp,) = comp.factors
                if isinstance(comp, Function) and isinstance(comp.signature, _PowerFunction):
                    power = comp.signature.k
                    (comp,) = comp.inputs
                if cls._is_trivial_scalar(comp):
                    continue
                if cls._is_idempotent_power(comp, power):
                    # Ones caps and other bare copy tensors are already canonical.
                    # Rewriting them as pow functions would create fresh pow tensors
                    # every iteration and prevent this loop from terminating.
                    passthrough.append(comp)
                    continue
                partition[_MatchEdgesKey(comp, **hyperedges)].append((power, comp))

            # Now we have a partition of tensors that share the same edges, and we can combine them.
            # Or in some cases they cancel each other.
            tensors = copys + passthrough
            for ts in partition.values():
                cls._emit_combined(tensors, ts)
        return tensors

    @classmethod
    def _combine_components(cls, tensors: list[Tensor]) -> list[Tensor]:
        # Look for disjoint components that are isomorphic, and combine them.
        # Since matching requires equal free edge names, only scalar components
        # (e.g. repeated fractions) can actually match.
        partition = defaultdict(list)
        for p in Product(tensors).components():
            power = 1
            if len(p.factors) == 1:
                (t,) = p.factors
                if isinstance(t, Function) and isinstance(t.signature, _PowerFunction):
                    power = t.signature.k
                    (t,) = t.inputs
            else:
                t = p
            if cls._is_trivial_scalar(t):
                continue
            partition[_MatchEdgesKey(t)].append((power, t))

        if not any(cls._combinable(ts) for ts in partition.values()):
            # Nothing to combine; leave the factor list (and its order) alone.
            return tensors

        tensors = []
        for ts in partition.values():
            cls._emit_combined(tensors, ts)
        return tensors

    @classmethod
    def _combine_hyperedge_components(cls, tensors: list[Tensor]) -> list[Tensor]:
        # Look for occurrences of the same base connected through shared copy (hyperedge)
        # tensors, e.g. the elementwise product x * x = Delta(i, "i,a,b") x_a x_b, or
        # directly through a contracted edge, e.g. sum(x * x) = x_e x_e, and combine
        # them: pow(x, a) * pow(x, b) -> pow(x, a + b). Copy tensors of order >= 2 act
        # as connectors: their edge names are canonicalized before matching. Directly
        # shared edges match by their (identical) name. Since _MatchEdgesKey matches
        # with match_edges=True, two factors only combine when all their edges
        # correspond, which makes the elementwise rewrite sound (e.g. the two factors
        # of Tr(X X) = X_ij X_ji do not match unless X is elementwise squared).
        copys: list[Tensor] = [t for t in tensors if isinstance(t, Delta) and t.order >= 2]
        others = [t for t in tensors if not (isinstance(t, Delta) and t.order >= 2)]
        hyperedges = {e: min(c.edges) for c in copys for e in c.edges}

        partition = defaultdict(list)
        passthrough: list[Tensor] = []
        for t in others:
            power = 1
            # Unwrap (nested) single-factor products, e.g. Ones caps.
            while isinstance(t, Product) and len(t.factors) == 1:
                (t,) = t.factors
            if isinstance(t, Function) and isinstance(t.signature, _PowerFunction):
                power = t.signature.k
                (t,) = t.inputs
            if cls._is_trivial_scalar(t):
                continue
            if cls._is_idempotent_power(t, power):
                # Ones caps (and other bare copy tensors) are already canonical;
                # rewriting them as pow functions would only obscure them.
                passthrough.append(t)
                continue
            partition[_MatchEdgesKey(t, **hyperedges)].append((power, t))

        if not any(cls._combinable(ts) for ts in partition.values()):
            # Nothing to combine; leave the factor list (and its order) alone.
            return tensors

        tensors = copys + passthrough
        for ts in partition.values():
            cls._emit_combined(tensors, ts)
        return tensors


def pow(tensor: Tensor, k: int | Fraction) -> Tensor:
    """Elementwise t^k"""
    if k == 0:
        return Ones(**tensor.shape)
    if k == 1:
        return tensor
    return Function(_PowerFunction(k), [tensor], {})


def sqrt(tensor: Tensor) -> Tensor:
    """Elementwise sqrt"""
    return pow(tensor, Fraction(1, 2))


class _MatrixInverseFunction(FunctionSignature):
    def __init__(self, edges: Collection[str]) -> None:
        if len(edges) != 2:
            raise ValueError(f"Called Matrix Inverse takes exactly 2 edges. Got {edges=}.")
        super().__init__("inv", frozenset(edges), (frozenset(edges),))

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        assert i == 0 and new_edges is not None
        # Derivative gets new_edges : {old_name : new_name}
        return _MatrixInverseDerivative(self.edges, new_edges)

    def simplify(self, func: Function, args: dict[str, Any]) -> Tensor:
        assert func.signature is self
        (inner,) = func.inputs
        # Two inverses over the same edges can be cancelled
        if (
            isinstance(inner, Function)
            and isinstance(inner.signature, _MatrixInverseFunction)
            and inner.signature.edges == self.edges
        ):
            (inner_inner,) = inner.inputs
            return inner_inner

        # Inverse of Identity is Identity
        if isinstance(inner, Delta):
            assert inner.order >= 2, "Both edges should be fed to the inverse"
            return inner

        return func

    @classmethod
    def simplify_outer(cls, tensors: list[Tensor], args: Optional[dict[str, Any]] = None) -> list[Tensor]:
        """Simplify a product by combining pow functions."""
        # TODO: This is not a general FunctionSignature method, but a special case for pow.
        # Maybe we can make it more general, e.g. using FunctionSignature.instances to get
        # all subclasses that may be relevant.
        # 1) We'd like to apply the rule X @ inv(X) = I.
        # 2) We might also apply inv(X Y) = inv(Y) inv(X).
        return tensors


class _MatrixInverseDerivative(FunctionSignature):
    def __init__(self, edges: AbstractSet[str], new_edges: dict[str, str]) -> None:
        # Derivative gets new_edges : {old_name : new_name}
        super().__init__("inv_grad", edges | set(new_edges.values()), (edges,))
        self.new_edges = new_edges.copy()

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        raise NotImplementedError("Please expand with simplify() first")

    def simplify(self, t: Function, args: dict[str, Any]) -> Tensor:
        (edges,) = self.inputs
        (inner,) = t.inputs
        # Inverse: -inv(A)^T dA inv(A). This expansion IS the derivative
        # rule (differentiation is rewriting), not an optional decomposition,
        # so it fires regardless of expand_functions -- an unexpanded
        # inv_grad has no evaluation and no lowering.
        inv = inverse(inner, edges)
        #    -i- A^{-1} -j-j'-
        # -i'-i- A^{-1} -j-
        (o1, e1), (o2, e2) = self.new_edges.items()
        res = -inv.rename(**{o1: e1}) * inv.rename(**{o2: e2})
        assert res.edges == t.edges
        return res.simplify(args)


def inverse(tensor: Tensor, dims: Optional[AbstractSet[str]] = None) -> Tensor:
    """Matrix inverse over two target edges. Broadcasts over the rest.
    Mirrors torch.inverse / torch.linalg.inv.

    In principle we could generalize the inverse to take a number of input
    edges, and the output would be a tensor with the same edges as the
    original, but if you contract it with the original over the selected
    edges, they cancel, and you get a tensor product over identitiy matrices
    over the remaining edges.

    If dims is None, we assume the input tensor is a matrix.
    """
    if dims is None:
        dims = tensor.edges
    if len(dims) != 2:
        raise ValueError(f"Called Matrix Inverse with {dims=}")
    out_shape = {name: size for name, size in tensor.shape.items() if name in dims}
    s1, s2 = out_shape.values()
    if s1 != s2:
        raise ValueError(f"Inverted dimensions must have same size. Got {s1, s2}")
    return Function(_MatrixInverseFunction(dims), [tensor], out_shape)


def _ExpFunction() -> FunctionSignature:
    # Small hack to handle recursive initialization
    exp_function = _SimpleFunction("exp", None)
    exp_function._derivative = exp_function
    return exp_function


def exp(t: Tensor) -> Tensor:
    return Function(_ExpFunction(), [t], {})


class _LogFunction(FunctionSignature):
    def __init__(self) -> None:
        super().__init__("log", frozenset(), (frozenset(),))

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        return _PowerFunction(-1)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        (inner,) = f.inputs
        # log(1) = 0
        if isinstance(inner, Product) and all(isinstance(t, Delta) and t.order != 0 for t in inner.factors):
            return Zero(_symmetries=None, **inner.shape)
        if isinstance(inner, Delta) and inner.order >= 1:
            return Zero(_symmetries=None, **inner.shape)
        # log(exp(x)) = x (exp is a _SimpleFunction named "exp")
        if (
            isinstance(inner, Function)
            and isinstance(inner.signature, _SimpleFunction)
            and inner.signature.name == "exp"
        ):
            return inner.inputs[0]
        # Other simplifications we could do:
        # - log(1/x) = -log(x)
        # - log(x^k) = k log(x)
        # - log(x y) = log(x) + log(y)
        return f


def log(t: Tensor) -> Tensor:
    return Function(_LogFunction(), [t], {})


class _ExprGradFunction(FunctionSignature):
    """Elementwise derivative signature defined by an expression builder.
    simplify() unconditionally rewrites Function(self, [x]) -> expr(x) in terms
    of other (native) functions, so it never reaches a backend. It follows the
    relu -> gt0 pattern for derivatives of true leaves that are themselves
    compositions (e.g. tanh' = 1 - tanh^2)."""

    def __init__(
        self,
        name: str,
        expr: Callable[[Tensor], Tensor],
        derivative_factory: Optional[Callable[[], FunctionSignature]] = None,
    ) -> None:
        super().__init__(name, frozenset(), (frozenset(),))
        self._expr = expr
        self._derivative_factory = derivative_factory

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        assert i == 0 and not new_edges, "Elementwise functions have no edges"
        if self._derivative_factory is None:
            raise NotImplementedError(f"Simplify {self.name} first, then differentiate")
        return self._derivative_factory()

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        (inner,) = f.inputs
        res = self._expr(inner)
        assert res.shape == f.shape
        return res.simplify(args)


# Module-level defs (not lambdas) so _ExprGradFunction instances — which end
# up inside cached/pickled expressions like d/dx tanh(x) — remain picklable.
def _tanh_grad_expr(x: Tensor) -> Tensor:
    # d/dx tanh(x) = 1 - tanh(x)^2
    return 1 - pow(tanh(x), 2)


def _tanh_grad2_expr(x: Tensor) -> Tensor:
    # d^2/dx^2 tanh(x) = -2 tanh(x) (1 - tanh(x)^2)
    return -2 * tanh(x) * (1 - pow(tanh(x), 2))


def _Tanh2GradFunction() -> FunctionSignature:
    return _ExprGradFunction("D2_tanh", _tanh_grad2_expr)


def _TanhGradFunction() -> FunctionSignature:
    return _ExprGradFunction("D_tanh", _tanh_grad_expr, _Tanh2GradFunction)


def tanh(t: Tensor) -> Tensor:
    """Elementwise tanh. A native (fused) function: unlike the old
    (e^t - e^(-t))/(e^t + e^(-t)) composition it cannot overflow for large
    inputs, and backends emit a single torch.tanh call."""
    return Function(_SimpleFunction("tanh", _TanhGradFunction()), (t,), {})


_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


def _erf_grad_expr(x: Tensor) -> Tensor:
    # d/dx erf(x) = 2/sqrt(pi) exp(-x^2)
    return _TWO_OVER_SQRT_PI * exp(-pow(x, 2))


def _erf_grad2_expr(x: Tensor) -> Tensor:
    # d^2/dx^2 erf(x) = -4x/sqrt(pi) exp(-x^2)
    return (-2 * _TWO_OVER_SQRT_PI) * x * exp(-pow(x, 2))


def _Erf2GradFunction() -> FunctionSignature:
    return _ExprGradFunction("D2_erf", _erf_grad2_expr)


def _ErfGradFunction() -> FunctionSignature:
    # Module-level defs (not lambdas) for picklability, like tanh above.
    return _ExprGradFunction("D_erf", _erf_grad_expr, _Erf2GradFunction)


def erf(t: Tensor) -> Tensor:
    """Elementwise Gauss error function, like torch.erf."""
    return Function(_SimpleFunction("erf", _ErfGradFunction()), (t,), {})


def gelu(t: Tensor, approximate: str = "exact", fused: bool = False) -> Tensor:
    """Gaussian Error Linear Unit, like torch.nn.functional.gelu.

    approximate="exact" (torch's "none"): x * Phi(x) = x/2 * (1 + erf(x/sqrt(2)))
    approximate="tanh": the tanh approximation used by GPT-style models.

    fused=True (approximate="tanh" only): compiles to torch's fused gelu /
    gelu_backward kernels instead of the derived tanh chain. Value-identical.
    """
    if fused:
        if approximate != "tanh":
            raise ValueError("fused gelu supports approximate='tanh' only")
        from tensorgrad.compiler.cells import CELLS
        return CELLS["gelu"].build((t,), {"approximate": approximate})
    if approximate in ("exact", "none"):
        return t * (1 + erf(t / math.sqrt(2))) / 2
    if approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        return t * (1 + tanh(c * (t + 0.044715 * pow(t, 3)))) / 2
    raise ValueError(f"Unknown gelu {approximate=}, expected 'exact'/'none'/'tanh'")


def sigmoid(t: Tensor) -> Tensor:
    """Implements the sigmoid function, 1/(1 + e^-t)"""
    return 1 / (1 + exp(-t))


def softmax(t: Tensor, dim: DimType = None) -> Tensor:
    """softmax(t, dim) = exp(t) / sum(exp(t), dim, keepdims=True).

    A plain composition of primitives — derivatives of any order come from
    the language itself, with no fused signature and no hand-written
    Jacobian. Eager evaluation of this expanded form overflows float32 at
    |t| >= ~89; numerical stability at extreme logits is the COMPILER's
    contract (tensorgrad/compiler/stabilize.py re-fuses the exp / sum-exp
    ratio into a native torch.softmax kernel)."""
    dim = parse_dim(t.edges, dim)
    e = exp(t)
    return e / sum(e, dim, keepdims=True)


def log_softmax(t: Tensor, dim: DimType = None) -> Tensor:
    """log_softmax(t, dim) = t - log(sum(exp(t), dim, keepdims=True)).

    A plain composition, like softmax above. The compiler's stabilize pass
    recovers torch.log_softmax / logsumexp for extreme inputs; eager
    evaluation is only exact at moderate |t|."""
    dim = parse_dim(t.edges, dim)
    return t - log(sum(exp(t), dim, keepdims=True))


def pairwise_distance(t1: Tensor, t2: Tensor, dim: DimType = None) -> Tensor:
    # If t1 and t2 have shape (x, i), pairwise_distance(t1, t2)_b is
    # ||t1[b] - t2[b]||_2^2
    dim = parse_dim(t1.edges, dim)
    return sum(pow(t1 - t2, 2), dim)


def cross_entropy(logits: Tensor, targets: Tensor, dim: DimType = None) -> Tensor:
    # We could make a FunctionSignature for cross entropy, if we want
    # the option of not always expanding it.
    dim = parse_dim(logits.edges, dim)
    return -sum(targets * log_softmax(logits, dim), dim)


class _RenameFunction(FunctionSignature):
    """A FunctionSignature whose OUTPUT edges are a renaming of an inner
    signature's outputs.

    This is signature-level plumbing, NOT a duplicate of the (lazy) Rename
    tensor node, and it cannot collapse into one: ``FunctionSignature.
    derivative(i, new_edges)`` must return a FunctionSignature producing the
    requested new edge names, but a signature like ``_MaxGradFunction``
    has its output names structurally tied to its input dims — only a
    signature-level rename can bridge the two inside ``grad_function``
    (a Rename *tensor* is not a signature and can't sit inside a Function).
    Its ``simplify`` immediately lowers it to the tensor-level
    ``Rename(Function(inner, ...))`` form, so it never survives simplify.
    """

    def __init__(self, inner: FunctionSignature, renames: dict[str, str]):
        # Note: The name is important for equality checking, since two functions
        # with the same name, inputs and edges are considered equal.
        self.name = f"rename({inner.name})"
        if inner.edges != renames.keys():
            raise ValueError("Inner function's edges must match the keys of renames")
        self.inputs = (frozenset(renames.keys()),)
        self.edges = frozenset(renames.values())
        self.renames = renames
        self.inner = inner

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        return _RenameFunction(self.inner.derivative(i, new_edges), self.renames)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        assert f.signature is self
        # Our function represents `inner` as well as a renaming. If we want to simplify
        # this into a Rename(Function(...)) tensor, we need to find the shape for the
        # inner fuction, and then rename the output of that function.
        inverse_renames = {v: k for k, v in self.renames.items()}
        inner_shape_out = {inverse_renames.get(e, e): s for e, s in f.shape_out.items()}
        return Rename(
            Function(self.inner, f.inputs, inner_shape_out),
            self.renames,
        ).simplify(args)


class _ZeroFunction(FunctionSignature):
    def __init__(self, new_edges: Optional[dict[str, str]] = None):
        """
        Takes an input with shape (i, j, ...) and outputs a zero tensor with shape
        {e: input_shape[o] for e, o in new_edges.items()}.
        If new_edges is None, the function is just element-wise.
        """
        self.name = "named-zero"
        self.new_edges = new_edges if new_edges else {}
        self.inputs = (frozenset(self.new_edges.values()),)
        self.edges = frozenset(self.new_edges.keys())

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        # Derivative gets new_edges : {old_name : new_name}
        new_edges = new_edges or {}
        inverse_new_edges = {v: k for k, v in new_edges.items()}
        return _ZeroFunction(self.new_edges | inverse_new_edges)

    def simplify(self, t: Function, args: dict[str, Any]) -> Tensor:
        # Instead of trying to calculate the shape ourselves, which is complicated
        # because of broadcasting, and that the function may have been renamed,
        # we can just copy the function's shape.
        return Zero(_symmetries=None, **t.shape)


class _SimpleFunction(FunctionSignature):
    def __init__(
        self,
        name: str,
        derivative: Union["FunctionSignature", None],
    ) -> None:
        super().__init__(name, frozenset(), (frozenset(),))
        self._derivative = derivative

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        assert i == 0 and not new_edges, "Simple functions are element-wise"
        if self._derivative is None:
            raise NotImplementedError(f"Derivative not implemented for {self.name}")
        return self._derivative


def _SignFunction() -> FunctionSignature:
    # We keep these FunctionObject functions around, since they are used both
    # for the function itself (like sign) and in derivatives (like abs)
    return _SimpleFunction("sign", _ZeroFunction())


def sign(t: Tensor) -> Tensor:
    """Returns a tensor that's
    1 where t > 0
    0 where t = 0
    -1 where t < 0
    like torch.sign"""
    return Function(_SignFunction(), (t,), {})


def _Gt0Function() -> FunctionSignature:
    # This could also be implemented as (sign(x) + 1) / 2
    return _SimpleFunction("gt0", _ZeroFunction())


def gt0(t: Tensor) -> Tensor:
    """Returns a tensor that's 1 where t is > 0 else 0 elsewhere"""
    return Function(_Gt0Function(), (t,), {})


def gt(x: Tensor, y: Tensor) -> Tensor:
    """Returns 1 where x > y, 1/2 where x = y, and 0 where x < y."""
    if x.shape != y.shape:
        raise ValueError(f"Inputs must have same shape, got {x.shape=} != {y.shape=}")
    return (sign(x - y) + 1) / 2


class _EqualFunction(FunctionSignature):
    """Function signature for element-wise equality comparison."""

    def __init__(self, edges: frozenset[str]) -> None:
        # Equal takes two inputs with the same edges and outputs the same edges
        # Both inputs consume all edges (element-wise operation)
        super().__init__("equal", edges, (edges, edges))

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        # The derivative of equal(x, y) is zero everywhere (it's a step function)
        return _ZeroFunction()


def equal(x: Tensor, y: Tensor) -> Tensor:
    """Returns 1.0 where x == y, 0.0 elsewhere.

    Direct implementation as a proper function signature for efficient code generation.
    """
    if x.shape != y.shape:
        raise ValueError(f"Inputs must have same shape, got {x.shape=} != {y.shape=}")
    # Create the equal function with the edges from the inputs
    # The function consumes all edges from both inputs and outputs the same edges
    return Function(_EqualFunction(frozenset(x.edges)), (x, y), x.shape)


def maximum(x: Tensor, y: Tensor) -> Tensor:
    """Like torch.maximum"""
    mask = gt(x, y)
    return x * mask + y * (1 - mask)


def relu(t: Tensor) -> Tensor:
    return Function(_SimpleFunction("relu", _Gt0Function()), (t,), {})


def abs(t: Tensor) -> Tensor:
    return Function(_SimpleFunction("abs", _SignFunction()), (t,), {})


def argmax(t: Tensor, dim: str) -> Tensor:
    return Function(_ArgMaxFunction(dim), (t,), {})


def argsort(t: Tensor, dim: str) -> Tensor:
    """Positions that sort `t` ascending along `dim` (integer values carried
    as floats, like argmax). A true leaf: indices are piecewise constant, so
    the derivative is Zero."""
    return Function(_ArgSortFunction(dim), (t,), {dim: t.shape[dim]})


def sort(t: Tensor, dim: str) -> Tensor:
    """`t` sorted ascending along `dim`: the contraction of `t` with the
    one_hot indicator of argsort — sorted[k] = sum_j [argsort[k] == j] t[j].
    Being linear in `t` (argsort is piecewise constant), its gradient
    DERIVES as the transposed permutation gather; no hand-written rule."""
    src_edge = _unused_edge_names([dim], t.edges)[dim]
    picks = one_hot(argsort(t, dim), t.shape[dim], src_edge)
    return dot(picks, t.rename(**{dim: src_edge}), dim=src_edge)


class _ArgMaxFunction(FunctionSignature):
    def __init__(self, dim: str) -> None:
        super().__init__("argmax", frozenset(), (frozenset([dim]),))
        self.dim = dim

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        raise NotImplementedError(f"Derivative not implemented for {self.name}")


class _ArgSortFunction(FunctionSignature):
    """argsort along `dim`; shape-preserving (unlike argmax, the sorted axis
    survives). Zero derivative: indices are piecewise constant in t."""

    def __init__(self, dim: str) -> None:
        super().__init__("argsort", frozenset([dim]), (frozenset([dim]),))
        self.dim = dim

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        new_edges = new_edges or {}
        return _MultiZeroFunction(self.edges | set(new_edges.values()), self.inputs)


# --------------------------------------------------------------------------
# Fused scaled-dot-product attention: a technology-mapping primitive.
#
# F.sdpa(q, k, v, mask=, scale=) IS softmax(scale*<q,k>_hs + mask) @_key v --
# its VALUE is the composite, machine-verified against that definition by the
# szfp fingerprints and the evaluate() oracle. But it is an OPAQUE Function,
# so differentiation does not expand the softmax; the compiler maps both the
# forward AND the derived backward onto torch's fused flash-attention CPU
# kernels. This is the attention analogue of offering torch.nn.functional.
# scaled_dot_product_attention: the algebra stays derivable (write attention
# from primitives if you want the derivation), this is the fast path.
#
# Edge convention: q (batch..., seq, hs); k, v (batch..., key, hs);
# out (batch..., seq, hs). mask (optional) has edges (seq, key).
# --------------------------------------------------------------------------


def sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    seq: str = "seq",
    key: str = "key",
    hs: str = "hs",
    mask: Optional[Tensor] = None,
    scale: float,
) -> Tensor:
    batch = frozenset(q.edges) - {seq, hs}
    for name, t, want in [("q", q, batch | {seq, hs}), ("k", k, batch | {key, hs}), ("v", v, batch | {key, hs})]:
        if set(t.edges) != set(want):
            raise ValueError(f"sdpa {name} edges {set(t.edges)} != expected {set(want)}")
    ins = [q, k, v]
    if mask is not None:
        if set(mask.edges) != {seq, key}:
            raise ValueError(f"sdpa mask edges {set(mask.edges)} != {{{seq}, {key}}}")
        ins.append(mask)
    from tensorgrad.compiler.cells import CELLS
    params = dict(seq=seq, key=key, hs=hs, batch=batch, scale=float(scale), has_mask=mask is not None)
    return CELLS["sdpa"].build(tuple(ins), params)


# --------------------------------------------------------------------------
# Fused layer normalization: a technology-mapping primitive.
#
# F.layer_norm(x, dim=, weight=, bias=, eps=) IS
#   weight * (x - mean(x, dim)) / sqrt(var(x, dim) + eps) + bias
# (biased variance, eps inside the sqrt) -- its VALUE is that composite,
# machine-verified against the definition by the szfp fingerprints and the
# evaluate() oracle. But it is an OPAQUE Function, so differentiation does not
# expand the mean/var/sqrt; the compiler maps both the forward AND the derived
# backward onto torch's fused native_layer_norm CPU kernels. This is the
# layernorm analogue of F.sdpa: write layer norm from primitives if you want
# the derivation, this is the fast path.
#
# Edge convention: x (batch..., dim); weight, bias (dim,); out (batch..., dim).
# The normalized axis is the single `dim` edge; weight/bias broadcast over it.
# --------------------------------------------------------------------------


def layer_norm(x: Tensor, *, dim: str, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Fused layer normalization over the single edge `dim`. weight and bias
    have edges {dim} and broadcast over it; x has (batch..., dim); output
    edges = x's edges. See the module comment above for the definition."""
    if dim not in x.edges:
        raise ValueError(f"layer_norm dim {dim!r} not in x edges {set(x.edges)}")
    batch = frozenset(x.edges) - {dim}
    if set(weight.edges) != {dim}:
        raise ValueError(f"layer_norm weight edges {set(weight.edges)} != {{{dim}}}")
    if set(bias.edges) != {dim}:
        raise ValueError(f"layer_norm bias edges {set(bias.edges)} != {{{dim}}}")
    from tensorgrad.compiler.cells import CELLS
    params = dict(dim=dim, batch=batch, eps=float(eps))
    return CELLS["layer_norm"].build((x, weight, bias), params)


def adamw(w: Tensor, g: Tensor, m: Tensor, v: Tensor, c1: Tensor, c2: Tensor, *,
          beta1: float, beta2: float, lr: float, eps: float,
          weight_decay: float) -> tuple[Tensor, Tensor, Tensor]:
    """Fused AdamW step -> (w', m', v') from one fused kernel.

    A NON-differentiable, MULTI-OUTPUT technology-mapping cell -- the update
    is a leaf in the training step, so it has no VJP. `c1`, `c2` are the
    runtime bias-correction scalars 1/(1-beta^t). Value-identical to the
    update equations written as tensorgrad algebra (see examples/mingpt.py),
    but the gradient `g` is consumed ONCE as a single operand, so the
    reverse pass is never duplicated across m'/v' (unlike the algebra form).
    Weight decay applies only to matrices (order >= 2), matching the demo."""
    from tensorgrad.compiler.cells import CELLS
    decay = 1.0 - lr * weight_decay if w.order >= 2 else 1.0
    params = dict(b1=float(beta1), b2=float(beta2), lr=float(lr),
                  eps=float(eps), decay=float(decay))
    return CELLS["adamw"].build((w, g, m, v, c1, c2), params)


class _MaxGradFunction(FunctionSignature):
    def __init__(self, dims: Iterable[str]):
        self.name = "max-grad"
        self.inputs = (frozenset(dims),)
        self.edges = frozenset(self.inputs[0])

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        # Zero doesn't broadcast edges that were given as input,
        # so we have to include them manually here
        new_edges = new_edges or {}
        inverse = {n: o for o, n in new_edges.items()}
        identity = {o: o for o, e in new_edges.items()}
        return _ZeroFunction(inverse | identity)


def max_grad(t: Tensor, dim: DimType = None) -> Tensor:
    """
    Tie-splitting subgradient for max. If multiple elements tie for max,
    each gets 1/k of the gradient.
    Maps an order n tensor to an order n tensor.

    Examples:
    (1)   j
        [1 4] -> max({i}) -> [2 4] -> grad({i}) -> [0 1] (broadcasted over j)
      i [2 3]                                      [1 0]

    (2)   j
        [1 4] -> max({i,j}) -> [4] -> grad({i,j}) -> [0 1]
      i [2 3]                                        [0 0]

    (3)   j
        [1 4] -> max({j}) -> [4] -> grad({j}) -> [0   1  ] (broadcasted over i)
      i [3 3]                [3]                 [0.5 0.5]
    """
    if isinstance(dim, str):
        dim = {dim}
    # Just as torch.amax, we default to "all dims".
    if dim is None:
        dim = set(t.edges)

    # All the edges that are consumed by the function are also produced by the function
    func = Function(_MaxGradFunction(dim), [t], {e: t.shape[e] for e in dim})
    assert func.edges == t.edges
    return func


class _MaxFunction(FunctionSignature):
    def __init__(self, dims: Iterable[str]):
        self.name = "max"
        self.edges = frozenset()
        self.inputs = (frozenset(dims),)

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        (dims,) = self.inputs
        # We only get new_edges for the self.inputs edges, not the broadcasted ones.
        # So our derivative function should handle this broadcasting by itself.
        assert new_edges is not None and dims == new_edges.keys(), "New edges should be a dict: old -> new names"
        return _RenameFunction(_MaxGradFunction(dims), new_edges)


def max(t: Tensor, dim: DimType = None, keepdims: bool = False) -> Tensor:
    """
    Return the max along 'dim'. If dim=(), it's the global max (0D).
    The derivative is tie-splitting, handled by gt(...).
    """

    if isinstance(dim, str):
        dim = (dim,)
    if not dim:
        dim = tuple(t.edges)

    fn = Function(_MaxFunction(dim), [t], {})
    if keepdims:
        fn = fn @ Ones(**{e: t.shape[e] for e in dim})
    return fn


class _MultiZeroFunction(FunctionSignature):
    """A zero tensor posing as a function of (possibly several) inputs.

    Unlike _ZeroFunction this keeps an arbitrary `inputs` signature, so it can
    stand in as the derivative of any multi-input function (e.g. one_hot) whose
    gradient vanishes. simplify() collapses it to Zero with the Function's
    (broadcast-aware) shape, so it never needs to be evaluated directly.
    """

    def __init__(self, edges: Iterable[str], inputs: Iterable[AbstractSet[str]]):
        super().__init__("multi-zero", frozenset(edges), tuple(frozenset(s) for s in inputs))

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        new_edges = new_edges or {}
        return _MultiZeroFunction(self.edges | set(new_edges.values()), self.inputs)

    def simplify(self, t: Function, args: dict[str, Any]) -> Tensor:
        return Zero(_symmetries=None, **t.shape)


class _OneHotFunction(FunctionSignature):
    """one_hot(idx, size_carrier)[eq_edge, *idx_edges] = [idx == eq_edge].

    The second input is any tensor with a single edge `dim` whose size is the
    number of classes (a Delta/ones vector); only its size is used. All idx
    edges are broadcast.

    This is the gather-family's only primitive: an indicator of integer data.
    Its derivative is Zero (a true leaf rule, like argmax/sign): `idx` holds
    integers carried as floats, so the indicator is piecewise constant.
    """

    def __init__(self, eq_edge: str, dim: str):
        super().__init__("one_hot", frozenset([eq_edge]), (frozenset(), frozenset([dim])))
        self.eq_edge = eq_edge
        self.dim = dim

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        new_edges = new_edges or {}
        return _MultiZeroFunction(self.edges | set(new_edges.values()), self.inputs)


def one_hot(idx: Tensor, size: Symbol, dim: Optional[str] = None) -> Tensor:
    """Indicator tensor one_hot(idx)[dim, *idx.edges] = [idx[...] == dim-index].

    `dim` defaults to the size symbol's name, the same idiom as
    Variable("x", batch, seq): F.one_hot(idx, vocab) has edge "vocab".

    `idx` holds integral values carried as floats (like argmax output). The
    indicator has no differentiable inputs, so anything contracted with it
    differentiates by the ordinary product rule. It is the whole gather /
    scatter / embedding-lookup family: indexing a table is the contraction

        F.one_hot(idx, vocab, dim="vocab") @ wte        # (*idx.edges, d)

    which is linear in the table, so its gradient derives as the transposed
    contraction. The compiler maps the contractions to torch.index_select
    (class axis summed against a table) and zeros().index_add_ (class axis
    free — the scatter); the dense indicator is never materialized."""
    if dim is None:
        dim = size.name
    if dim in idx.edges:
        raise ValueError(f"{dim=} must not be an edge of idx: {idx.edges=}")
    # Delta(size, dim) is a ones-vector that only carries the number of
    # classes, so the one-hot does not depend on (or broadcast over) a table.
    return Function(_OneHotFunction(dim, dim), [idx, Delta(size, dim)], {dim: size})


def concat(*ts: Tensor, dim: str, size: Symbol) -> Tensor:
    """Concatenate tensors along the edge `dim`.

    `size` is the output edge's size symbol (edge sizes are opaque symbols
    in tensorgrad; that it equals the sum of the parts' sizes is bound with
    the other dimension values at evaluation time).

    Concatenation is algebra: each part embeds through the structural shift
    indicator E[i, o] = [i == o - offset] (one Affine row), so the result is
    a Sum of contractions, the compiler eliminates the shifts into strided
    views, and — being linear in every part — its derivative derives.
    """
    from tensorgrad.compiler.affine import Affine  # a symbolic Constant; lazy: the compiler imports this module

    if not ts:
        raise ValueError("concat needs at least one tensor")
    rest = ts[0].edges - {dim}
    for t in ts:
        if dim not in t.edges:
            raise ValueError(f"{dim=} is not an edge of {t.edges=}")
        if t.edges - {dim} != rest:
            raise ValueError(f"concat parts must share the other edges: {t.edges=} vs {rest | {dim}}")
    parts, offset = [], sympy.Integer(0)
    for t in ts:
        s = t.shape[dim]
        tmp = _unused_edge_names([dim], t.edges)[dim]
        # row: tmp - dim = -offset, i.e. out[dim=k] reads this part at k - offset
        shift = Affine([({tmp: 1, dim: -1}, -offset)], **{tmp: s, dim: size})
        parts.append(t.rename(**{dim: tmp}) @ shift)
        offset = offset + s
    return Sum(parts) if len(parts) > 1 else parts[0]


def window(*, start: int = 0, **shape1: Symbol) -> Tensor:
    """The rectangular diagonal W[i, o] = [i == o + start], as a first-class
    structural tensor: contract it to slice.

        tokens  = full @ F.window(length=buf, seq=seq)           # length[0 : seq]
        targets = full @ F.window(length=buf, seq=seq, start=1)  # length[1 : seq+1]

    Reads as: a window into the FIRST edge (matched by name against the
    tensor you contract with), producing the SECOND edge — which the result
    then carries, so no rename is needed. `start` is where the window
    begins, in input coordinates; it may be an int or a sympy expression.
    One Affine equality row: the compiler eliminates it into a strided
    view, and with a larger output size the same tensor zero-pads instead
    of slicing (windowing and embedding are transposes of one another —
    see concat)."""
    from tensorgrad.compiler.affine import Affine

    if len(shape1) != 2:
        raise ValueError(f"window needs exactly two edges (input, output), got {set(shape1)}")
    (i, si), (o, so) = shape1.items()
    return Affine([({i: 1, o: -1}, sympy.sympify(start))], **{i: si, o: so})


def repeat(t: Tensor, *shape0: Symbol, **shape1: Symbol) -> Tensor:
    """
    Repeat the tensor along the given dimensions.
    E.g. to create a batch of identity matrices, you can do
    I = repeat(Delta(d, i, j=i), b)
    """
    # TODO: Maybe this should instead be implemented using smart slicing,
    # like t[:, :, None, :, :, None] to repeat along the 2nd and 5th dimension.
    # Then we might also want to implement slicing in general, like t[b=3], but
    # unfortuantely python doesn't support this syntax.
    raise NotImplementedError("Not implemented yet")


class _DeterminantDerivative(FunctionSignature):
    def __init__(self, dims: AbstractSet[str], new_edges: dict[str, str]):
        self.name = "det_grad"
        self.edges = frozenset(new_edges.values())
        self.inputs = (frozenset(dims),)
        self.new_edges = new_edges

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        raise NotImplementedError("Please expand with simplify() first")

    def simplify(self, t: Function, args: dict[str, Any]) -> Tensor:
        (dims,) = self.inputs
        (inner,) = t.inputs
        # d det(A) = det(A) A^-T (cookbook 46). Like inv_grad, this IS the
        # derivative rule, so it fires regardless of expand_functions.
        return det(inner, dims) * inverse(inner, dims).rename(**self.new_edges)


class _DeterminantFunction(FunctionSignature):
    def __init__(self, dims: AbstractSet[str]):
        self.name = "det"
        self.edges = frozenset()
        assert len(dims) == 2, f"Determinant takes exactly 2 dims, got {dims=}"
        self.inputs = (frozenset(dims),)

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> FunctionSignature:
        assert new_edges is not None
        return _DeterminantDerivative(self.inputs[0], new_edges)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        assert f.signature is self

        # TODO: Let's of properties we can add:
        # det(cA) = c^n det(A), if A ∈ R^n×n (19)
        # det(A^T) = det(A) (20)
        # det(AB) = det(A) det(B) (21)
        # det(A^−1) = 1/det(A) (22)
        # det(A^n) = det(A)^n (23)
        # det(I + uv^T) = 1 + u^Tv (24)

        # Most (all?) of them would also follow from using Penrose's
        # "implementation" of the determinant, using the Levi-Civita tensor.
        # However, that requires making "size" many copies of f.inner, which is
        # still symbolic for us.

        (edges,) = self.inputs
        (inner,) = f.inputs

        # det(I) = 1
        if isinstance(inner, Delta):
            assert inner.order >= 2
            if inner.order == 2:
                return Ones()
            return Delta(inner._size, *[e for e in inner.edges if e not in edges])

        return f


def det(t: Tensor, dims: DimType = None) -> Tensor:
    """
    Compute the determinant of square matrices.
    """
    dims = parse_dim(t.edges, dims, none_is="all")
    if len(dims) != 2:
        raise ValueError(f"Determinant takes exactly 2 dims, got {dims=}")
    dim0, dim1 = dims
    if t.shape[dim0] != t.shape[dim1]:
        raise ValueError(f"Determinant requires square matrices, got {t.shape[dim0]=} != {t.shape[dim1]=}")
    return Function(_DeterminantFunction(dims), [t], {})


class Convolution(Constant):
    def __init__(
        self,
        *shape0: Symbol,
        _symmetries: None | str | set[frozenset[str]] = None,
        _stride: int = 1,
        **shape1: Symbol,
    ):
        """
        A Convolution is a 3-tensor such that C[i,j,k] = 1 if i=j+k and 0 otherwise.
        Typically the first argument (i) is the input dim, and two others are the kernel and output dims.

        Use Convolution(win, kw, wout) to represent a 1-Dconvolution with
        input size win, kernel size kw, and output size wout.
        wout will be win - kw + 1.
        For 2D convolution, use Convolution(win, kw, wout) @ Convolution(hin, kh, hout).

        We don't currently support stride, but if we did the size would be wout = (win - kw) // stride + 1.
        The formula would be C[i,j,k] = 1 if i = j + k * stride.
        """
        shape = self._check_shape(shape0, shape1)
        assert len(shape) == 3, "Convolution must have exactly 3 edges: input, kernel, output"
        (e0, _s0), (e1, s1), (e2, s2) = shape.items()
        symmetries = {frozenset([e1, e2]), frozenset([e0])} if s1 == s2 else None
        super().__init__(_symmetries=symmetries, **shape)
        self.input_name, self.kernel_name, self.output_name = e0, e1, e2
        self.stride = _stride

    def __repr__(self) -> str:
        return f"Convolution({self.input_name}, {self.kernel_name}, {self.output_name})"

    def __hash__(self) -> int:
        return hash((type(self).__name__,) + tuple(self.shape.items()))

    def _rename(self, **kwargs: str) -> "Tensor":
        return Convolution(_symmetries=None, _stride=self.stride, **{kwargs.get(k, k): v for k, v in self.shape.items()})


class Reshape(Constant):
    """Just the identity matrices, but with a different shape."""

    # Mostly Reshape doesn't have any symmetries, except if the input equals the output,
    # in which case it can be flipped (and just ignored completely.)
    # Generally it's recommended to never reshape anything, and just keep plenty of edges around.

    def __repr__(self) -> str:
        return f"Reshape({self.shape})"

    def __hash__(self) -> int:
        return hash((type(self).__name__,) + tuple(self.shape.items()))

    def _rename(self, **kwargs: str) -> "Tensor":
        return Reshape(_symmetries=None, **{kwargs.get(k, k): v for k, v in self.shape.items()})
