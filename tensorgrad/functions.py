from collections import defaultdict
import itertools
import math
from numbers import Number
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
from sympy import Symbol
import torch
from tensorgrad.tensor import (
    Constant,
    Function,
    FunctionSignature,
    MatchEdgesKey,
    Ones,
    Sum,
    Tensor,
    Product,
    Copy,
    Variable,
    Zero,
    make_distinct,
    unused_edge_names,
)
from fractions import Fraction

from tensorgrad.utils import DisjointSets

# We include a "sum" function, which overloads the python sum. So we keep a reference
# here to the builtin, so we can use it in this module
_sum = sum
DimType = None | str | Iterable[str]

# We mostly try to follow the behavior of pytorch's named tensors:
# https://pytorch.org/docs/stable/name_inference.html


def taylor(f: "Tensor", wrt: "Variable", eps: "Tensor", n: int) -> "Tensor":
    """Return the nth order Taylor approximation of f at x+eps."""
    if eps.edges != wrt.edges:
        raise ValueError("eps must have the same edges as wrt.")
    total = f
    for i in range(1, n + 1):
        connection_names = unused_edge_names(wrt.edges, f.edges)
        fg = f.grad(wrt, new_names=connection_names)
        total += fg @ eps.rename(**connection_names) / math.factorial(i)
    return total


def frobenius2(t: Tensor) -> Tensor:
    return Product([t, t])


def _is_even_permutation(permutation: tuple[int]) -> bool:
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
def symmetrize(t: Tensor, signed: bool = False) -> Tensor:
    """Sum over all permutations of the edges."""
    edges = list(t.edges)
    res = []
    weights = []
    for perm in itertools.permutations(edges):
        res.append(t.rename(**dict(zip(edges, perm))))
        if signed:
            weights.append(1 if _is_even_permutation(perm) else -1)
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
    - Copy tensors are created automatically when a name starting with '*', like '*3' is used.
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
    vars: dict[str, Tensor] = vars.copy()

    # Parse the graph, converting long lines into a list of edges:
    # st. "A -i-j- B" -> ("A", "i", "j", "B")
    # and "A -i-" -> ("A", "i", "i", None)
    # and "A -i- B - C" -> [("A", "i", "i", "B"), ("B", "i", "i", "C")]
    # and "*1 - *2" -> ("*1", None, None, "*2")
    lines = re.split(r"[\n;]", dot_graph.strip())
    edges: list[tuple[str, str, str, str]] = []
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
    hyperedges: dict[str, list[str]] = defaultdict(list)
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
        copies.append(Copy(size, *edges))

    return Product(copies + list(vars.values()))


def kronecker(*tensors: Tensor) -> Tensor:
    # Basically just rename everything to be distinct, then contraction
    # Note: This function returns the tensor product, which is different from the
    #       Kronecker product as often described in the literature. To get the
    #       Kronecker product you have to flatten the output tensors.
    dis_tensors, _renames = make_distinct(*tensors)
    return Product(dis_tensors)


def diag(t: Tensor, new_edges: list[str]) -> Tensor:
    """If `t` is a vector, this creates a diagonal tensor with `t` and creates a diagonal.
    In einsum that means "i->iii".
    If `t` is a higher order tensor, with all dims the same size, this extracts the diagonal as a vector.
    In einsum that means "iii->i".
    """
    # Rename the edges to be distinct from the new edges
    (t,), _renames = make_distinct(t, used_names=new_edges)

    if not t.shape:
        # We can't just return t @ Copy(new_edges), since we don't know the size of the new edges.
        raise ValueError("Cannot take the diagonal of a scalar.")

    edges, sizes = zip(*t.shape.items())
    if len(set(sizes)) != 1:
        raise ValueError("All dimensions must be the same size for the diagonal.")

    return t @ Copy(sizes[0], *edges, *new_edges)


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
    out = Product([tensor] + [Copy(tensor.shape[e], e) for e in dim])
    # Optionally broadcast back to orignal shape
    if keepdims:
        return out @ Ones(**{e: tensor.shape[e] for e in dim})
    return out


def mean(tensor: Tensor, dim: DimType = None, keepdims: bool = False) -> Tensor:
    dim = parse_dim(tensor.edges, dim, none_is="all")
    s = sum(tensor, dim, keepdims)
    normalization = 1
    for e in dim:
        normalization @= Copy(tensor.shape[e])
    return s / normalization


def dot(t1: Tensor, t2: Tensor, dim: DimType = None) -> Tensor:
    """Contract two tensors along the given dimensions, broadcasting over the remaining shared edges."""
    dim = parse_dim(t1.edges & t2.edges, dim)
    return sum(t1 * t2, dim)


class _ScaleFunction(FunctionSignature):
    def __init__(self, inner: FunctionSignature, alpha: Number):
        # Represents alpha * inner(x, ...)
        # This mostly exists to help represent the PowerFunction derivative
        self.name = f"{alpha} * {inner.name}"
        self.edges = inner.edges
        self.inputs = inner.inputs
        self.inner = inner
        self.alpha = alpha

    def eval(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.inner.eval(*xs)

    def derivative(self, i, new_edges=None) -> FunctionSignature:
        return _ScaleFunction(self.inner.derivative(i, new_edges), self.alpha)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        assert f.signature is self
        return self.alpha * Function(self.inner, f.inputs, f.shape_out, f.orig_out).simplify(args)


class _PowerFunction(FunctionSignature):
    def __init__(self, k: int):
        self.name = f"pow({k=})"
        self.edges = frozenset()
        self.inputs = (frozenset(),)
        self.k = k

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        if self.k < 0:
            x = x.to(torch.float)
        return torch.pow(x, float(self.k))

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
            return (inner).simplify(args)

        kwargs = dict(orig_out=func.orig_out)

        # The pow function is multiplicative, so we can pull components out of a product apart.
        if isinstance(inner, Product):
            new_comps = []
            for comp in inner.components():
                new_comps.append(Function(self, (comp,), func.shape_out, **kwargs))
            if len(new_comps) > 1:
                return (Product(new_comps)).simplify(args)

        # We can pull out the weight of a sum if it's just a single tensor
        if isinstance(inner, Sum) and len(inner.tensors) == 1:
            (w,) = inner.weights
            (t,) = inner.tensors
            return Function(self, (t,), func.shape_out, **kwargs) * (w**self.k)

        # Base cases
        if (
            # Pow of 1 is just 1.
            isinstance(inner, Copy)
            # Copy of order 0 can have values other than 1
            and inner.order > 0
            # Pow of 0 is just 0
            or isinstance(inner, Zero)
        ):
            return inner

        # Combine nested pows
        if isinstance(inner, Function) and isinstance(inner.signature, _PowerFunction):
            return Function(
                _PowerFunction(k=inner.signature.k * self.k),
                inner.inputs,
                func.shape_out,
                **kwargs,
            )

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

        tensors = cls._combine_powers(tensors)
        assert Product(tensors).edges == original_edges

        if args["factor_components"]:
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
        s = ",\n    ".join(map(repr, tensors))
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
                if isinstance(c, Copy) and c.edges & inner.edges
                for e in c.edges
            }
            partition = defaultdict(list)

            # We can't just call t.rename(**hyperedges) here, since some tensors might be adjacent to the
            # same hyperedge multiple times. Instead we give MatchEdgesKey the names, and it will perform
            # the isomorphism check correctly.
            partition[MatchEdgesKey(inner, **hyperedges)].append((power, inner))

            # We remove the tensor, as well as all Copys that it's connected to,
            # which makes the rest of the graph fall apart.
            others = tensors[:i] + tensors[i + 1 :]
            copys = [t for t in others if isinstance(t, Copy) and t.edges & inner.edges]
            others = [t for t in others if not (isinstance(t, Copy) and t.edges & inner.edges)]
            for comp in Product(others).components():
                power = 1
                if len(comp.tensors) == 1:
                    comp = comp.tensors[0]
                    if isinstance(comp, Function) and isinstance(comp.signature, _PowerFunction):
                        power = comp.signature.k
                        (comp,) = comp.inputs
                partition[MatchEdgesKey(comp, **hyperedges)].append((power, comp))

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
            partition[MatchEdgesKey(t)] += power

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


def _ExpFunction() -> FunctionSignature:
    # Small hack to handle recursive initialization
    exp_function = _SimpleFunction("exp", torch.exp, None)
    exp_function._derivative = exp_function
    return exp_function


def exp(t: Tensor) -> Tensor:
    return Function(_ExpFunction(), [t], {})


def _LogFunction() -> FunctionSignature:
    # If we want to support more complicated simplification rules,
    # like expanding log(a*b) into log(a) + log(b),
    # we should define a custom function signature.
    return _SimpleFunction("log", torch.log, _PowerFunction(-1))


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

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        (dims,) = self.inputs
        sizes = [x.size(d) for d in dims]
        names = [d for d in x.names if d not in dims]
        other_sizes = [x.size(n) for n in names]
        # Softmax doesn't support named dimensions, so we have to rename them to None.
        # We move the affected dimensions to the front, then flatten them.
        y = x.align_to(*dims, *names).rename(None).flatten(start_dim=0, end_dim=len(dims) - 1)
        return y.softmax(dim=0).reshape(sizes + other_sizes).rename(*dims, *names).align_to(*x.names)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        if args["expand_functions"]:
            (dims,) = self.inputs
            (inner,) = f.inputs
            assert dims.issubset(inner.edges)
            e = exp(inner)
            res = e / sum(e, dims, keepdims=True)
            # Since Function includes an implicit rename, we need to apply that
            # when expanding the Function
            return res.rename(**{o: e for e, o in f.orig_out.items()})
        return super().simplify(f, args)


def softmax(t: Tensor, dim: DimType = None) -> Tensor:
    dim = parse_dim(t.edges, dim)
    return Function(_SoftmaxFunction(dim), [t], {e: t.shape[e] for e in dim})


def pairwise_distance(t1: Tensor, t2: Tensor, dim: DimType = None) -> Tensor:
    # If t1 and t2 have shape (x, i), pairwise_distance(t1, t2)_b is
    # ||t1[b] - t2[b]||_2^2
    dim = parse_dim(t1.edges, dim)
    return sum(pow(t1 - t2, 2), dim)


def cross_entropy(t: Tensor, y: Tensor, dim: DimType = None) -> Tensor:
    # We could make a FunctionSignature for cross entropy, if we want
    # the option of not always expanding it.
    dim = parse_dim(t.edges, dim)
    return -sum(y * log(softmax(t, dim)), dim)


class _SimpleFunction(FunctionSignature):
    def __init__(self, name: str, eval_fn: Callable[[torch.Tensor], torch.Tensor], 
                 derivative: "FunctionSignature"):
        self.name = name
        self.inputs: Tuple[Set[str], ...] = (frozenset(),)
        self.edges: Set[str] = frozenset()
        self._eval_fn = eval_fn
        self._derivative = derivative

    def eval(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        return self._eval_fn(input_tensors[0])

    def derivative(self, i, new_edges=None):
        assert i == 0 and not new_edges, "Simple functions are element-wise"
        return self._derivative


class _RenameFunction(FunctionSignature):
    def __init__(self, inner: Function, renames: dict[str, str]):
        # Note: The name is important for equality checking, since two functions
        # with the same name, inputs and edges are considered equal.
        self.name = f"rename({inner.name})"
        if inner.edges != renames.keys():
            raise ValueError("Inner function's edges must match the keys of renames")
        self.inputs = (frozenset(renames.keys()),)
        self.edges = frozenset(renames.values())
        self.renames = renames
        self.inner = inner

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.eval(x).rename(**self.renames)

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        return _RenameFunction(self.inner.derivative(i, new_edges), self.renames)

    def simplify(self, f: Function, args: dict[str, Any]) -> Tensor:
        assert f.signature is self
        # Since the Function object also keeps track of renaming, we have a chain:
        # inner -> rename -> inverse(orig_out) -> shape_out
        # We want to merge our rename map with the orig out, so we need:
        # inverse(rename . inverse(orig_out)) = orig_out . inverse(rename)
        # Example:
        #   Variable(i,j)
        #   self.shape_out: {j___0: j, i___0: i}
        #   renames: {j: j__, i: i__}
        #   orig_out: {i___0: i__, j___0: j__}
        #   new_orig_out = {i___0: i, j___0: j}
        # Note:
        #   You'd think we could just commute rename and orig_out and rename
        #   the inputs, as [input.rename(**self.renames)]. However, this breaks
        #   the expectation of self.inner.inputs on the input names.
        inverse_renames = {v: k for k, v in self.renames.items()}
        return Function(
            self.inner,
            f.inputs,
            f.shape_out,
            {e: inverse_renames[o] for e, o in f.orig_out.items()},
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

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        # Like any function, we have to support broadcasted inputs, so we detect
        # which names are in x, which are not consumed in self.inputs
        broadcasted = [e for e in x.names if e not in self.inputs[0]]
        return torch.zeros(
            size=[x.size(o) for o in broadcasted + list(self.new_edges.values())],
            names=broadcasted + list(self.new_edges.keys()),
        )

    def simplify(self, t: Function, args: dict[str, Any]) -> Tensor:
        # Instead of trying to calculate the shape ourselves, which is complicated
        # because of broadcasting, and that the function may have been renamed,
        # we can just copy the function's shape.
        return Zero(**t.shape)


def _SignFunction() -> FunctionSignature:
    # We keep these FunctionObject functions around, since they are used both
    # for the function itself (like sign) and in derivatives (like abs)
    return _SimpleFunction("sign", torch.sign, _ZeroFunction())


def sign(t: Tensor) -> Tensor:
    """Returns a tensor that's
    a)  1 where t > 0
    a)  0 where t = 0
    a) -1 where t < 0
    like torch.sign"""
    return Function(_SignFunction(), (t,), {})


def _Gt0Function() -> FunctionSignature:
    # This could also be implemented as (sign(x) + 1) / 2
    return _SimpleFunction(
        "gt0",
        lambda x: torch.where(x.rename(None) > 0, 1.0, 0.0).rename(*x.names),
        _ZeroFunction(),
    )


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
    return Function(
        _SimpleFunction("relu", torch.relu, _Gt0Function()),
        (t,),
        {},
    )


def abs(t: Tensor) -> Tensor:
    return Function(
        _SimpleFunction("abs", torch.abs, _SignFunction()),
        (t,),
        {},
    )


class _MaxGradFunction(FunctionSignature):
    def __init__(self, dims: Set[str]):
        self.name = "max-grad"
        self.inputs: Tuple[Set[str], ...] = (frozenset(dims),)
        self.edges: Set[str] = frozenset(dims)

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> FunctionSignature:
        # Zero doesn't broadcast edges that were given as input,
        # so we have to include them manually here
        inverse = {n: o for o, n in new_edges.items()}
        identity = {o: o for o, e in new_edges.items()}
        return _ZeroFunction(inverse | identity)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        (dims,) = self.inputs
        adim = [x.names.index(e) for e in dims]
        x, names = x.rename(None), x.names
        max_vals = x.amax(dim=adim, keepdim=True)
        mask = (x == max_vals).float()
        res = mask / mask.sum(dim=adim, keepdim=True).clamp(min=1.0)
        return res.rename(*names)


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

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        (dims,) = self.inputs
        return torch.amax(
            x.rename(None),
            dim=[x.names.index(e) for e in dims],
            keepdim=False,
        ).rename(*(n for n in x.names if n not in dims))

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


class Convolution(Constant):
    def __init__(self, *shape0: Symbol, _symmetries: set[frozenset[str]] = None, **shape1: Symbol):
        """
        A Convolution is a 3-tensor such that C[i,j,k] = 1 if i=j+k and 0 otherwise.
        Typically the first argument (i) is the input dim, and two others are the kernel and output dims.

        Use Convolution(win, kw, wout) to represent a 1-Dconvolution with
        input size win, kernel size kw, and output size wout.
        wout will be win - kw + 1.
        For 2D convolution, use Convolution(win, kw, wout) @ Convolution(hin, kh, hout).

        Output shape (patches, dim) where dim = channels * kernel_width * kernel_height
        But that's where I'm arguing that we don't need to flatten the channels unto the output
        we can just keep it broadcasted and people can flatten it if they want.
        I don't know why torch.Unfold doesn't do it this way, but presumably there's some performance hit?

        width_in = 6
        [x x x x x x]
        [1 0 0 - - -] [0 1 0 - - -] [0 0 1 - - -]
        [- 1 0 0 - -] [- 0 1 0 - -] [- 0 0 1 - -]
        [- - 1 0 0 -] [- - 0 1 0 -] [- - 0 0 1 -]
        [- - - 1 0 0] [- - - 0 1 0] [- - - 0 0 1]
        width_out = 4
        kw = 3

        (width_out, kw, width_in)
        [1 0 0 - - -]
        [0 1 0 - - -]
        [0 0 1 - - -]

        [- 1 0 0 - -]
        [- 0 1 0 - -]
        [- 0 0 1 - -]

        [- - 1 0 0 -]
        [- - 0 1 0 -]
        [- - 0 0 1 -]

        [- - - 1 0 0]
        [- - - 0 1 0]
        [- - - 0 0 1]

        width_in = 5
        height_in = 3
        [x x x x x]
        [x x x x x]
        [x x x x x]
        [1 0 - - -]
        [0 1 - - -]

        [1 0 0 - - -] [0 1 0 - - -] [0 0 1 - - -]
        [- 1 0 0 - -] [- 0 1 0 - -] [- 0 0 1 - -]
        [- - 1 0 0 -] [- - 0 1 0 -] [- - 0 0 1 -]
        [- - - 1 0 0] [- - - 0 1 0] [- - - 0 0 1]
        width_out = 4
        kw = 3
        """
        shape = self._check_shape(shape0, shape1)
        assert len(shape) == 3, "Convolution must have exactly 3 edges: input, kernel, output"
        (e0, _s0), (e1, s1), (e2, s2) = shape.items()
        if s1 == s2:
            assert len({s1, s2}) == 1, "Kernel and output size must be the same"
        symmetries = {frozenset([e1, e2]), frozenset([e0])} if s1 == s2 else None
        super().__init__(_symmetries=symmetries, **shape)
        self.input_name, self.kernel_name, self.output_name = e0, e1, e2

    def __repr__(self) -> str:
        return f"Convolution({self.input_name}, {self.kernel_name}, {self.output_name})"

    def rename(self, **kwargs: str) -> "Tensor":
        kwargs = self._check_rename(kwargs)
        return Convolution(**{kwargs.get(k, k): v for k, v in self.shape.items()})

    def evaluate(
        self,
        values: dict["Variable", torch.Tensor],
        dims: dict[Symbol, int] | None = None,
    ) -> torch.Tensor:
        if not self.edges:
            return torch.tensor(1.0)
        w_in = dims[self.shape[self.input_name]]
        k_size = dims[self.shape[self.kernel_name]]

        # TODO: Right now we assume w_in and k_size are given, and compute the output size
        # manually. However, we could use any of the two to compute the third.
        w_out = w_in - k_size + 1

        # Check consistency
        if w_out_given := dims.get(self.shape[self.output_name]) is not None:
            assert w_out == w_out_given, (
                f"Convolution expects its output dim to be win - kw + 1, "
                f"but got {w_in} - {k_size} + 1 = {w_out} vs. {w_out_given}."
            )

        # Make a tensor T, such that T[i,j,k] = 1 iff i=j+k
        res = torch.zeros(w_in, k_size, w_out)
        for k in range(w_out):
            for j in range(k_size):
                res[k + j, j, k] = 1
        return res.rename(self.input_name, self.kernel_name, self.output_name)


class Flatten(Constant):
    def __init__(self, *shape0: Symbol, _symmetries: set[frozenset[str]] = None, **shape1: Symbol):
        shape = self._check_shape(shape0, shape1)
        *self.input_edges, self.output_edge = shape.keys()
        assert len(shape) >= 2, "Flatten must have at least 2 edges"
        # We have symmetry between all input edges, as long as they are all the same size
        groups = defaultdict(list)
        for e in self.input_edges:
            groups[shape[e]].append(e)
        groups[self.output_edge].append(self.output_edge)
        super().__init__(_symmetries=set(map(frozenset, groups.values())), **shape)

    def __repr__(self):
        return f"Flatten({self.input_edges}, {self.output_edge})"

    def __hash__(self):
        return hash((type(self).__name__, len(self.edges)))

    def rename(self, **kwargs: str) -> "Tensor":
        kwargs = self._check_rename(kwargs)
        return Flatten(**{kwargs.get(k, k): v for k, v in self.shape.items()})

    def evaluate(
        self,
        values: dict["Variable", torch.Tensor],
        dims: dict[Symbol, int] | None = None,
    ) -> torch.Tensor:
        """
        Produces a tensor of shape [*input_dims, output_dim] where
        output_dim = product of all input_dims.

        For each multi-index (i1, i2, ..., iN),
        we set res[i1, i2, ..., iN, flatten(i1, i2, ..., iN)] = 1
        and zero otherwise.
        """
        # Collect the sizes of the edges
        input_dims = [dims[self.shape[e]] for e in self.input_edges]
        output_dim = dims.get(self.shape[self.output_edge])

        # Check consistency: output_dim should be the product of input_dims
        prod_in = 1
        for d in input_dims:
            prod_in *= d

        if output_dim is not None:
            assert prod_in == output_dim, (
                f"Flatten expects its output dim == product of input dims, "
                f"but got product({input_dims}) = {prod_in} vs. {output_dim}."
            )
        else:
            output_dim = prod_in

        # Build an identity matrix and reshape it so that row indices
        # become the multi-index (i1, i2, ..., iN),
        # and the column index becomes the flattened coordinate.
        eye = torch.eye(output_dim)  # shape = [output_dim, output_dim]
        res = eye.reshape(*input_dims, output_dim)  # shape = [*input_dims, output_dim]
        return res.rename(*self.input_edges, self.output_edge)
