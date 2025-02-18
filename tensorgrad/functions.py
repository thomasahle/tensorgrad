import itertools
import math
import re
from collections import defaultdict
from fractions import Fraction
from numbers import Number
from typing import Any, Iterable, Iterator, Sequence, Union

from sympy import Symbol

from tensorgrad.tensor import (
    Constant,
    Delta,
    Function,
    FunctionSignature,
    Ones,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
    Zero,
    _make_distinct,
    _MatchEdgesKey,
    _unused_edge_names,
)
from tensorgrad.utils import DisjointSets

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
        # Taking the derivative of f wrt wrt, creates new edges that match the shape of eps
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
    weights: list[Number] = []
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
    edges = []
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
    for he, edges in hyperedges.items():
        size = hypersizes.get(he)
        if size is None:
            raise ValueError(f"Hyperedge {he} has no size")
        if len(edges) != len(set(edges)):
            raise ValueError("Hyperedges must be disjoint")
        copies.append(Delta(size, *edges))

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


def parse_dim(tensor_edges: set[str], dim: DimType = None, none_is: str = "error") -> set[str]:
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
    def __init__(self, inner: FunctionSignature, alpha: Number):
        # Represents alpha * inner(x, ...)
        # This mostly exists to help represent the PowerFunction derivative
        self.name = f"{alpha} * {inner.name}"
        self.edges = inner.edges
        self.inputs = inner.inputs
        self.inner = inner
        self.alpha = alpha

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        return _ScaleFunction(self.inner.derivative(i, new_edges), self.alpha)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        assert f.signature is self
        return self.alpha * Function(self.inner, f.inputs, f.shape_out).simplify(args)


class _PowerFunction(FunctionSignature):
    def __init__(self, k: int):
        self.name = f"pow({k=})"
        self.edges = frozenset()
        self.inputs = (frozenset(),)
        self.k = k

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        assert i == 0 and (new_edges is None or not new_edges)
        return _ScaleFunction(_PowerFunction(self.k - 1), self.k)

    def simplify(self, func: Function, args: dict[str, Any]) -> Tensor:
        assert func.signature is self
        assert len(func.inputs) == 1, "pow should only have one input"
        (inner,) = func.inputs

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

        # We can pull out the weight of a sum if it's just a single tensor
        if isinstance(inner, Sum) and len(inner.tensors) == 1:
            (w,) = inner.weights
            (t,) = inner.tensors
            return Function(self, (t,), {}) * (w**self.k)

        # Base cases. We could add Copy(d)**k = Copy(d**k) here, but we're trying to keep the
        # mathematical expressions in tensorgrad space, rather than in sympy space.
        if isinstance(inner, Zero) or isinstance(inner, Product) and not inner.tensors:
            return inner

        # Combine nested pows
        if isinstance(inner, Function) and isinstance(inner.signature, _PowerFunction):
            return Function(_PowerFunction(k=inner.signature.k * self.k), inner.inputs, func.shape_out)

        return func

    @classmethod
    def simplify_outer(cls, tensors: list[Tensor], args: dict[str, Any] = None) -> list[Tensor]:
        """Simplify a product by combining pow functions."""
        # This is not a general FunctionSignature method, but a special case for pow.
        original_edges = Product(tensors).edges

        # New plan:
        #  - First try to combine existing pow functions.
        #  - Then try to do cancelations with non-pow subgraphs
        #  - Finally we could try to recreate some pow functions from the remaining subgraphs,
        #    but it's not clear that this is actually useful.

        # Combine pow(x, k) * pow(x, l) = pow(x, k + l)
        tensors = cls._combine_powers(tensors)
        assert Product(tensors).edges == original_edges

        if args["factor_components"]:
            # Combine  pow(x, k) * x = pow(x, k + 1)  or cancellations  pow(x, -1) * x = 1
            tensors = cls._combine_components(tensors)
            assert Product(tensors).edges == original_edges

        # We have to merge products here because we might otherwise have undone part of the simplification
        tensors = Product.merge([Product([t]) if not isinstance(t, Product) else t for t in tensors]).tensors

        assert Product(tensors).edges == original_edges
        return tensors

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
            if t.signature.k > 5:
                break

            power = t.signature.k
            (inner,) = t.inputs

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
            copys = [t for t in others if isinstance(t, Delta) and t.edges & inner.edges]
            others = [t for t in others if not (isinstance(t, Delta) and t.edges & inner.edges)]
            for comp in Product(others).components():
                power = 1
                if len(comp.tensors) == 1:
                    comp = comp.tensors[0]
                    if isinstance(comp, Function) and isinstance(comp.signature, _PowerFunction):
                        power = comp.signature.k
                        (comp,) = comp.inputs
                partition[_MatchEdgesKey(comp, **hyperedges)].append((power, comp))

            # Now we have a partition of tensors that share the same edges, and we can combine them.
            # Or in some cases they cancel each other.
            tensors = copys
            for ts in partition.values():
                k = _sum(k for k, t in ts)
                (_, t0) = ts[0]  # All the tensors should be the same t
                tensors.append(pow(t0, k))
                # The remaining tensors have been reduced to ones. This prevents leaving unattached edges
                # on the copy tensors.
                for _, t in ts[1:]:
                    # Take care to not add a lot of empty Ones/Products, as it could
                    # make our method loop forever as it collects powers of nothing.
                    if t.shape:
                        tensors.append(Ones(**t.shape))
        return tensors

    @classmethod
    def _combine_components(cls, tensors: list[Tensor]) -> list[Tensor]:
        # Look for tensors that share the same edges, and combine them.
        partition = defaultdict(int)
        for p in Product(tensors).components():
            power = 1
            if len(p.tensors) == 1:
                (t,) = p.tensors
                if isinstance(t, Function) and isinstance(t.signature, _PowerFunction):
                    power = t.signature.k
                    (t,) = t.inputs
            else:
                t = p
            partition[_MatchEdgesKey(t)] += power

        tensors = []
        for key, w in partition.items():
            t = key.value
            tensors.append(pow(t, w))
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
    def __init__(self, edges) -> None:
        if len(edges) != 2:
            raise ValueError(f"Called Matrix Inverse takes exactly 2 edges. Got {edges=}.")
        super().__init__("inv", frozenset(edges), (frozenset(edges),))

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        assert i == 0
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
    def simplify_outer(cls, tensors: list[Tensor], args: dict[str, Any] = None) -> list[Tensor]:
        """Simplify a product by combining pow functions."""
        # TODO: This is not a general FunctionSignature method, but a special case for pow.
        # Maybe we can make it more general, e.g. using FunctionSignature.instances to get
        # all subclasses that may be relevant.
        # 1) We'd like to apply the rule X @ inv(X) = I.
        # 2) We might also apply inv(X Y) = inv(Y) inv(X).
        return tensors


class _MatrixInverseDerivative(FunctionSignature):
    def __init__(self, edges: set[str], new_edges: dict[str, str]) -> None:
        # Derivative gets new_edges : {old_name : new_name}
        super().__init__("inv_grad", edges | set(new_edges.values()), (edges,))
        self.new_edges = new_edges.copy()

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        raise NotImplementedError("Please expand with simplify() first")

    def simplify(self, t: Function, args: dict[str, Any]):
        (edges,) = self.inputs
        (inner,) = t.inputs
        # Inverse: -inv(A)^T dA inv(A)
        if args["expand_functions"]:
            inv = inverse(inner, edges)
            #    -i- A^{-1} -j-j'-
            # -i'-i- A^{-1} -j-
            (o1, e1), (o2, e2) = self.new_edges.items()
            res = -inv.rename(**{o1: e1}) * inv.rename(**{o2: e2})
            assert res.edges == t.edges
            return res.simplify(args)
        return t


def inverse(tensor: Tensor, dims: set[str] = None) -> Tensor:
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
    def __init__(self):
        super().__init__("log", frozenset(), (frozenset(),))

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        return _PowerFunction(-1)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        (inner,) = f.inputs
        # log(1) = 0
        if isinstance(inner, Product) and all(isinstance(t, Delta) and t.order != 0 for t in inner.tensors):
            return Zero(**inner.shape)
        if isinstance(inner, Delta) and inner.order >= 1:
            return Zero(**inner.shape)
        # log(exp(x)) = x
        # if isinstance(inner, Function) and isinstance(inner.signature, _ExpFunction):
        # return inner.inputs[0]
        # Other simplifications we could do:
        # - log(1/x) = -log(x)
        # - log(x^k) = k log(x)
        # - log(x y) = log(x) + log(y)
        return f


def log(t: Tensor) -> Tensor:
    return Function(_LogFunction(), [t], {})


def tanh(t: Tensor) -> Tensor:
    """Implements the tanh function, (e^t - e^(-t))/(e^t + e^(-t))"""
    e = exp(t)
    em = exp(-t)
    return (e - em) / (e + em)


def sigmoid(t: Tensor) -> Tensor:
    """Implements the sigmoid function, 1/(1 + e^-t)"""
    return 1 / (1 + exp(-t))


class _SoftmaxFunction(FunctionSignature):
    # We don't really need this function signature, as we could just use the basic
    # functions of exp, sum and pow(-1) to implement it. However it is useful as a
    # proof of concept, and allows us to represent softmax "unexpanded".

    def __init__(self, dims: set[str]):
        super().__init__("softmax", frozenset(dims), (frozenset(dims),))

    def derivative(self, i: int, new_edges: dict[str, str] = None):
        raise NotImplementedError("Simplify and then differentiate")

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        if args["expand_functions"]:
            (dims,) = self.inputs
            (inner,) = f.inputs
            assert dims.issubset(inner.edges)
            e = exp(inner)
            res = e / sum(e, dims, keepdims=True)
            return res.simplify(args)
        return super().simplify(f, args)


def softmax(t: Tensor, dim: DimType = None) -> Tensor:
    dim = parse_dim(t.edges, dim)
    return Function(_SoftmaxFunction(dim), [t], {e: t.shape[e] for e in dim})


def pairwise_distance(t1: Tensor, t2: Tensor, dim: DimType = None) -> Tensor:
    # If t1 and t2 have shape (x, i), pairwise_distance(t1, t2)_b is
    # ||t1[b] - t2[b]||_2^2
    dim = parse_dim(t1.edges, dim)
    return sum(pow(t1 - t2, 2), dim)


def cross_entropy(logits: Tensor, targets: Tensor, dim: DimType = None) -> Tensor:
    # We could make a FunctionSignature for cross entropy, if we want
    # the option of not always expanding it.
    dim = parse_dim(logits.edges, dim)
    return -sum(targets * log(softmax(logits, dim)), dim)


class _RenameFunction(FunctionSignature):
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

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
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
    def __init__(self, new_edges: dict[str, str] = None):
        """
        Takes an input with shape (i, j, ...) and outputs a zero tensor with shape
        {e: input_shape[o] for e, o in new_edges.items()}.
        If new_edges is None, the function is just element-wise.
        """
        self.name = "named-zero"
        self.new_edges = new_edges if new_edges else {}
        self.inputs = (frozenset(self.new_edges.values()),)
        self.edges = frozenset(self.new_edges.keys())

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        # Derivative gets new_edges : {old_name : new_name}
        inverse_new_edges = {v: k for k, v in new_edges.items()}
        return _ZeroFunction(self.new_edges | inverse_new_edges)

    def simplify(self, t: Function, args: dict[str, Any]) -> Tensor:
        # Instead of trying to calculate the shape ourselves, which is complicated
        # because of broadcasting, and that the function may have been renamed,
        # we can just copy the function's shape.
        return Zero(**t.shape)


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
    a)  1 where t > 0
    a)  0 where t = 0
    a) -1 where t < 0
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


class _ArgMaxFunction(FunctionSignature):
    def __init__(self, dim: str) -> None:
        super().__init__("argmax", frozenset(), (frozenset([dim]),))
        self.dim = dim

    def derivative(self, i: int, new_edges: dict[str, str] | None = None) -> FunctionSignature:
        raise NotImplementedError(f"Derivative not implemented for {self.name}")


class _MaxGradFunction(FunctionSignature):
    def __init__(self, dims: set[str]):
        self.name = "max-grad"
        self.inputs: tuple[set[str], ...] = (frozenset(dims),)
        self.edges: set[str] = frozenset(dims)

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        # Zero doesn't broadcast edges that were given as input,
        # so we have to include them manually here
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
    def __init__(self, dims: set[str]):
        self.name = "max"
        self.edges = frozenset()
        self.inputs = (frozenset(dims),)

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        (dims,) = self.inputs
        # We only get new_edges for the self.inputs edges, not the broadcasted ones.
        # So our derivative function should handle this broadcasting by itself.
        assert dims == new_edges.keys(), "New edges should be a dict: old -> new names"
        return _RenameFunction(_MaxGradFunction(dims), new_edges)


def max(t: Tensor, dim: DimType = None, keepdim: bool = False) -> Tensor:
    """
    Return the max along 'dim'. If dim=(), it's the global max (0D).
    The derivative is tie-splitting, handled by gt(...).
    """

    if isinstance(dim, str):
        dim = (dim,)
    if not dim:
        dim = tuple(t.edges)

    fn = Function(_MaxFunction(dim), [t], {})
    if keepdim:
        fn = fn @ Ones(**{e: t.shape[e] for e in dim})
    return fn


def concat(*ts: Tensor, dim: str):
    """
    Concatenate tensors along the given dimension.
    """
    raise NotImplementedError("Not implemented yet")


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
    def __init__(self, dims: set[str], new_edges: dict[str, str] = None):
        self.name = "det_grad"
        self.edges = frozenset(new_edges.values())
        self.inputs = (frozenset(dims),)
        self.new_edges = new_edges

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        raise NotImplementedError("Please expand with simplify() first")

    def simplify(self, t: Function, args: dict[str, Any]):
        (dims,) = self.inputs
        (inner,) = t.inputs
        if args["expand_functions"]:
            return det(inner, dims) * inverse(inner, dims).rename(**self.new_edges)


class _DeterminantFunction(FunctionSignature):
    def __init__(self, dims: set[str]):
        self.name = "det"
        self.edges = frozenset()
        assert len(dims) == 2, f"Determinant takes exactly 2 dims, got {dims=}"
        self.inputs = (frozenset(dims),)

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
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
        self, *shape0: Symbol, _symmetries: set[frozenset[str]] = None, _stride: int = 1, **shape1: Symbol
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
        return Convolution(**{kwargs.get(k, k): v for k, v in self.shape.items()}, _stride=self.stride)


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
        return Reshape(**{kwargs.get(k, k): v for k, v in self.shape.items()})
