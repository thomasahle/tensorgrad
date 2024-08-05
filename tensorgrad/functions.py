from collections import defaultdict
import itertools
import math
import re
from typing import Any
from sympy import Symbol
import torch
from tensorgrad.tensor import (
    Constant,
    Function,
    FunctionInfo,
    MatchEdgesKey,
    Ones,
    Sum,
    Tensor,
    Product,
    Copy,
    Variable,
    Zero,
    make_distinct,
)
from fractions import Fraction

from tensorgrad.utils import DisjointSets

# We include a "sum" function, which overloads the python sum. So we keep a reference
# here to the builtin, so we can use it in this module
_sum = sum

# We mostly try to follow the behavior of pytorch's named tensors:
# https://pytorch.org/docs/stable/name_inference.html

# TODO:
# - Add all the functions of PyTorch
# - Add general function inverses, implicit functions
# - Taylor approximation should support chanigng the point of expansion


def taylor(f: Tensor, wrt: Variable, eps: Tensor, n: int) -> Tensor:
    """Return the nth order Taylor approximation of f at x+eps."""
    if eps.edges != wrt.edges:
        raise ValueError("eps must have the same edges as wrt.")
    total = f
    for i in range(1, n + 1):
        # TODO: If the wrt edges are not available, this will throw an error.
        # We should rename wrt and eps to be distinct from the edges of f.
        fg = f.grad(wrt, new_names={e: e for e in wrt.edges})
        total += fg @ eps / math.factorial(i)
    return total


def frobenius2(t: Tensor) -> Tensor:
    return Product([t, t])


def symmetrize(t: Tensor) -> Tensor:
    """Sum over all permutations of the edges."""
    edges = list(t.edges)
    res = 0
    for perm in itertools.permutations(edges):
        res += t.rename(**dict(zip(edges, perm)))
    return res


def einsum(tensors, output_edges):
    if len(output_edges) != len(set(output_edges)):
        # We don't support einsums like "i -> ii".
        # We also don't support "ii -> i", but that's more hidden, because the input tensors can't have double edges.
        raise ValueError("Output edges must be unique.")
    # Basically like Product, but will create some Identity's to ensure only the free_edges are free afterwards.
    sizes = {}
    for t in tensors:
        for e, s in t.shape.items():
            if e not in sizes:
                sizes[e] = s
            elif sizes[e] != s:
                raise ValueError("Mismatched sizes")
    # TODO: We only really need to rename the free edges of each tensor, so `make_distinct`` is overkill.
    dis_tensors, renames = make_distinct(*tensors, used_names=sizes.keys())
    joins = []
    for e, s in sizes.items():
        # We create a Copy([...]) with all the entries that have this edge
        edges = [rename[e] for rename in renames if e in rename]
        if e in output_edges:
            edges.append(e)
        joins.append(Copy(s, *edges))
    return Product(dis_tensors + joins)


def graph(dot_graph, **vars):
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

    vars = vars.copy()
    hyperedges = defaultdict(list)
    hypersizes = defaultdict(list)
    used_edges = {e for v in vars.values() for e in v.edges}
    free_edges = (f"e{i}" for i in itertools.count() if f"e{i}" not in used_edges)
    free_hyper_edges = (f"h{i}" for i in itertools.count() if f"h{i}" not in used_edges)
    he_names = defaultdict(lambda: next(free_hyper_edges))
    ds = DisjointSets()

    for v0, e0, e1, v1 in edges:
        for v, e in ((v0, e0), (v1, e1)):
            if v and not v.startswith("*"):
                if v not in vars:
                    raise ValueError(f"Variable {v} not found in vars")
                if e not in vars[v].edges:
                    raise ValueError(f"Edge {e} not found in variable {v}")

        if v0 is None and v1 is None:
            raise ValueError("Cannot have two free edges")
        elif v0 is None or v1 is None:
            v, e = (v1, e1) if v0 is None else (v0, e0)
            if v.startswith("*"):
                hyperedges[he_names[v]].append(e)
            else:
                vars[v] = vars[v].rename(**{e: e0 if v0 is None else e1})
        elif v0.startswith("*") and v1.startswith("*"):
            ds.union(he_names[v0], he_names[v1])
        elif v0.startswith("*") or v1.startswith("*"):
            v, e, he = (v0, e0, v1) if v1.startswith("*") else (v1, e1, v0)
            c = next(free_edges)
            hyperedges[he_names[he]].append(c)
            hypersizes[he_names[he]].append(vars[v].shape[e])
            vars[v] = vars[v].rename(**{e: c})
        elif v0 == v1:
            if e0 == e1:
                raise ValueError("Cannot have a self loop on a single edge")
            c0, c1 = next(free_edges), next(free_edges)
            he = next(free_hyper_edges)
            hyperedges[he].extend([c0, c1])
            hypersizes[he].append(vars[v0].shape[e0])
            vars[v0] = vars[v0].rename(**{e0: c0, e1: c1})
        else:
            e = next(free_edges)
            vars[v0] = vars[v0].rename(**{e0: e})
            vars[v1] = vars[v1].rename(**{e1: e})

    if any(he not in hyperedges for he in he_names.values()):
        raise ValueError("Some hyperedges are not connected to anything.")

    partition = defaultdict(list)
    for name in hyperedges.keys():
        partition[ds.find(name)].append(name)
    copies = []
    for group in partition.values():
        sizes = {s for he in group for s in hypersizes[he]}
        if len(sizes) != 1:
            raise ValueError(f"Hyperedges {sizes} must have the same size")
        (size,) = sizes
        edges = [e for he in group for e in hyperedges[he]]
        if len(edges) != len(set(edges)):
            raise ValueError("Hyperedges must be disjoint")
        copies.append(Copy(size, *edges))

    return Product(copies + list(vars.values()))


def kronecker(*tensors):
    # Basically just rename everything to be distinct, then contraction
    # Note: This function returns the tensor product, which is different from the
    #       Kronecker product as often described in the literature. To get the
    #       Kronecker product you have to flatten the output tensors.
    dis_tensors, _renames = make_distinct(*tensors)
    return Product(dis_tensors)


def diag(t: Tensor, new_edges: list[str]):
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


def sum(tensor: Tensor, edges: list[str] = None, keepdims=False) -> Tensor:
    """Sum the tensor over the given dimensions."""
    edges = tensor.edges if edges is None else edges
    out = Product([tensor] + [Copy(tensor.shape[e], e) for e in edges])
    # Optionally broadcast back to orignal shape
    if keepdims:
        return out @ Ones(**{e: tensor.shape[e] for e in edges})
    return out


def mean(tensor: Tensor, edges: list[str] = None, keepdims=False) -> Tensor:
    s = sum(tensor, edges, keepdims)
    normalization = 1
    for e in edges:
        normalization @= Copy(tensor.shape[e])
    return s / normalization


def dot(t1: Tensor, t2: Tensor, dims: list[str]) -> Tensor:
    """Contract two tensors along the given dimensions, broadcasting over the remaining shared edges."""
    return sum(t1 * t2, dims)


def log(t: Tensor) -> Tensor:
    return Function(
        FunctionInfo(
            "log",
            eval=lambda x: torch.log(x),
            derivative=lambda _i, _new_edges, t: pow(t, -1),
        ),
        [],
        (t,),
    )


def tanh(t: Tensor) -> Tensor:
    e = exp(t)
    em = exp(-t)
    return (e - em) / (e + em)


class PowFunctionInfo(FunctionInfo):
    def __init__(self, k: int):
        super().__init__(f"pow({k})", eval=self.eval, derivative=self.derivative, simplify=self.simplify)
        self.k = k

    def eval(self, x):
        if self.k < 0:
            return torch.pow(x.to(torch.float), self.k)
        return torch.pow(x, self.k)

    def derivative(self, i, new_edges, t):
        return self.k * pow(t, self.k - 1)

    def simplify(self, func, args):
        assert len(func.inputs) == 1, "pow should only have one input"
        inner, *es = func.inputs[0]
        assert not es, "Multiplicative functions should be element-wise"

        if self.k == 0:
            return Ones(**func.shape)
        if self.k == 1:
            return inner

        kwargs = dict(orig_out=func.orig_out)

        # The pow function is multiplicative, so we can pull components out of a product apart.
        if isinstance(inner, Product):
            new_comps = []
            for comp in inner.components():
                new_comps.append(Function(func.fn_info, func.edges_out, (comp,), **kwargs))
            if len(new_comps) > 1:
                return Product(new_comps).simplify(args)

        # We can pull out the weight of a sum if it's just a single tensor
        if isinstance(inner, Sum) and len(inner.tensors) == 1:
            (w,) = inner.weights
            (t,) = inner.tensors
            return Function(func.fn_info, func.edges_out, (t,), **kwargs) * (w**self.k)

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
        if isinstance(inner, Function) and isinstance(inner.fn_info, PowFunctionInfo):
            return Function(
                PowFunctionInfo(inner.fn_info.k * func.fn_info.k),
                func.edges_out,
                *inner.inputs,
                **kwargs,
            )

        return func

    @classmethod
    def simplify_outer(cls, tensors: list[Tensor], args: dict[str, Any] = None) -> list[Tensor]:
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
        seen = set()
        # for _ in range(3):
        while True:
            # Find the next power function we haven't seen yet
            try:
                i, t = next(
                    (i, t)
                    for i, t in enumerate(tensors)
                    if isinstance(t, Function) and isinstance(t.fn_info, PowFunctionInfo) and t not in seen
                )
            except StopIteration:
                break
            seen.add(t)

            power_weight = t.fn_info.k
            ((inner, *_es),) = t.inputs

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
            partition[MatchEdgesKey(inner, **hyperedges)].append((power_weight, inner))

            # We remove the tensor, as well as all Copy's that it's connected to,
            # which makes the rest of the graph fall apart.
            others = tensors[:i] + tensors[i + 1 :]
            copys = [t for t in others if isinstance(t, Copy) and t.edges & inner.edges]
            others = [t for t in others if not (isinstance(t, Copy) and t.edges & inner.edges)]
            for comp in Product(others).components():
                power_weight = 1
                if len(comp.tensors) == 1:
                    comp = comp.tensors[0]
                    if isinstance(comp, Function) and isinstance(comp.fn_info, PowFunctionInfo):
                        power_weight = comp.fn_info.k
                        ((comp, *_es),) = comp.inputs
                partition[MatchEdgesKey(comp, **hyperedges)].append((power_weight, comp))

            # Now we have a partition of tensors that share the same edges, and we can combine them.
            # Or in some cases they cancel each other.
            tensors = copys
            for ts in partition.values():
                w = _sum(w for w, t in ts)
                t0 = ts[0][1]
                if w == 1:
                    tensors.append(t0)
                else:
                    tensors.append(pow(t0, w))
                # The remaining tensors have been reduced to ones. This prevents leaving unattached edges
                # on the copy tensors.
                for _, t in ts[1:]:
                    tensors.append(Ones(**t.shape))
        return tensors

    @classmethod
    def _combine_components(cls, tensors: list[Tensor]) -> list[Tensor]:
        partition = defaultdict(int)
        for p in Product(tensors).components():
            power_weight = 1
            if len(p.tensors) == 1:
                (t,) = p.tensors
                if isinstance(t, Function) and isinstance(t.fn_info, PowFunctionInfo):
                    power_weight = t.fn_info.k
                    t, *_es = t.inputs[0]
            else:
                t = p
            partition[MatchEdgesKey(t)] += power_weight

        tensors = []
        for key, w in partition.items():
            t = key.value
            tensors.append(pow(t, w))
        return tensors


def pow(tensor: Tensor, k: int) -> Tensor:
    """Elementwise t^k"""
    if k == 0:
        return Ones(**tensor.shape)
    if k == 1:
        return tensor
    return Function(PowFunctionInfo(k), [], (tensor,))


def sqrt(tensor: Tensor) -> Tensor:
    """Elementwise sqrt"""
    return Function(PowFunctionInfo(Fraction(1, 2)), [], (tensor,))


def exp(t: Tensor) -> Tensor:
    return Function(
        FunctionInfo(
            "exp",
            eval=lambda x: torch.exp(x),
            derivative=lambda _i, _nn, t: exp(t),
        ),
        [],
        (t,),
    )


def softmax(t: Tensor, dims: list[str]) -> Tensor:
    if set(dims) - set(t.edges):
        raise ValueError("dims must be a subset of t.edges")
    e = exp(t)
    return e * pow(sum(e, dims, keepdims=True), -1)


def pairwise_distance(t1: Tensor, t2: Tensor, dims: list[str]):
    return pow(t1 - t2, 2).sum(dims)


def cross_entropy(t: Tensor, y: Tensor, dims: list[str]) -> Tensor:
    if set(dims) - set(t.edges):
        raise ValueError("dims must be a subset of t.edges")
    return -sum(y * log(softmax(t, dims)), dims)


class ReluFunctionInfo(FunctionInfo):
    def __init__(self):
        super().__init__("relu", eval=self.eval, derivative=self.derivative)

    def eval(self, x):
        return torch.relu(x)

    def derivative(self, i: int, new_edges: dict[str, Symbol], t: Tensor):
        assert not new_edges, "Relu is element-wise, so there shouldn't be any connection edges"
        return gt0(t)


def relu(t: Tensor) -> Tensor:
    return Function(ReluFunctionInfo(), [], (t,))


def abs(t: Tensor) -> Tensor:
    return Function(
        FunctionInfo(
            "abs",
            eval=lambda x: x.abs(),
            derivative=lambda _i, new_edges, t: sign(t),
        ),
        [],
        (t,),
    )


def sign(t: Tensor) -> Tensor:
    """Returns a tensor that's 1 where t is > 0 and -1 elsewhere"""
    return 2 * gt0(t) - 1


def gt0(t: Tensor) -> Tensor:
    """Returns a tensor that's 1 where t is > 0 else 0 elsewhere"""

    return Function(
        FunctionInfo(
            "gt",
            eval=lambda x: torch.where(x.rename(None) > 0, 1.0, 0.0).rename(*x.names),
            derivative=lambda _i, new_edges, t: Zero(**t.shape),
        ),
        [],
        (t,),
    )


def gt(t: Tensor, dim: str | tuple[str] = (), keepdim=False) -> Tensor:
    """Returns a tensor that's 1 for the largest index in the row (along dim), 0 elsewhere."""
    if isinstance(dim, str):
        dim = (dim,)

    def inner(x):
        indices = torch.max(x, dim=dim).indices
        one_hot = torch.zeros_like(x)
        one_hot.scatter_(dim=dim, index=indices.unsqueeze(x.names.index(dim)), value=1)

    return Function(
        FunctionInfo(
            "gt",
            eval=lambda x: inner(x),
            derivative=lambda _i, new_edges, t: Zero(**(t.shape | new_edges)),
        ),
        [],
        (t, dim),
    )


def max(t: Tensor, dim: str | tuple[str] = (), keepdim=False) -> Tensor:
    """Returns the sum of each row of the input tensor in the given dimension dim.
    If dim is a list of dimensions, reduce over all of them.

    If keepdim is True, the output tensor is of the same size as input except in the dimension(s)
    dim where it is of size 1. Otherwise, dim is squeezed (see torch.squeeze()), resulting in the
    output tensor having 1 (or len(dim)) fewer dimension(s).
    """
    if isinstance(dim, str):
        dim = (dim,)
    if dim == ():
        dim = tuple(t.edges)

    def eval_max(x):
        names = x.names if keepdim else tuple(n for n in x.names if n not in dim)
        adim = tuple(x.names.index(d) for d in dim)
        return torch.amax(x.rename(None), adim, keepdim).rename(*names)

    func = Function(
        FunctionInfo(
            "max",
            eval=eval_max,
            derivative=lambda _i, _nn, t: gt(t, dim, keepdim),
        ),
        [],
        (t, *dim),
    )
    if keepdim:
        func @= Ones(**{d: t.shape[d] for d in dim})
    return func


# Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
# Variable("data", ["batch", "channel_in", "width", "height"])
# @ Unfold(["width", "height"], ["kw", "kh"], ["width_out", "height_out"])
# @ Variable("kernel", ["channel_in", "kw", "kh", "channel_out"])
# -> ["batch", "channel_out", "width_out", "heigth_out"] (where width_out = width - kw + 1 and height_out = height - kh + 1)
def Unfold(input_edges: list[str], kernel_edges: list[str], output_edges: list[str]):
    # The full Unfold function is just the product over individual convolutions
    return Product(Convolution(ie, ke, oe) for ie, ke, oe in zip(input_edges, kernel_edges, output_edges))


class Convolution(Constant):
    def __init__(self, input_edge: str, kernel_edge: str, output_edge: str):
        super().__init__([input_edge, kernel_edge, output_edge])
        assert len(self.edges) == len(set(self.edges))
        self.input_edge = input_edge
        self.kernel_edge = kernel_edge
        self.output_edge = output_edge

    def __repr__(self):
        return f"Convolution({self.input_edge}, {self.kernel_edge}, {self.output_edge})"

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        return Convolution(
            kwargs.get(self.input_edge, self.input_edge),
            kwargs.get(self.kernel_edge, self.kernel_edge),
            kwargs.get(self.output_edge, self.output_edge),
        )

    def evaluate(
        self,
        values: dict[Tensor, torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        if not self.edges:
            return torch.tensor(1.0)
        edge_dims = extras["edge_dims"][id(self)]
        w_in = edge_dims[self.input_edge]
        k_size = edge_dims[self.kernel_edge]
        # TODO: How do I communicate w_out to the next convolution kernel?
        w_out = w_in - k_size + 1
        # return[i,j,k] = 1 iff i=j+k
        res = torch.zeros(w_in, k_size, w_out)
        for k in range(w_out):
            for j in range(k_size):
                res[k + j, j, k] = 1
        return res

    # Output shape (patches, dim) where dim = channels * kernel_width * kernel_height
    # But that's where I'm arguing that we don't need to flatten the channels unto the output
    # we can just keep it broadcasted and people can flatten it if they want.
    # I don't know why torch.Unfold doesn't do it this way, but presumably there's some performance hit?

    # width_in = 6
    # [x x x x x x]
    # [1 0 0 - - -] [0 1 0 - - -] [0 0 1 - - -]
    # [- 1 0 0 - -] [- 0 1 0 - -] [- 0 0 1 - -]
    # [- - 1 0 0 -] [- - 0 1 0 -] [- - 0 0 1 -]
    # [- - - 1 0 0] [- - - 0 1 0] [- - - 0 0 1]
    # width_out = 4
    # kw = 3

    #

    # (width_out, kw, width_in)
    # [1 0 0 - - -]
    # [0 1 0 - - -]
    # [0 0 1 - - -]

    # [- 1 0 0 - -]
    # [- 0 1 0 - -]
    # [- 0 0 1 - -]

    # [- - 1 0 0 -]
    # [- - 0 1 0 -]
    # [- - 0 0 1 -]

    # [- - - 1 0 0]
    # [- - - 0 1 0]
    # [- - - 0 0 1]

    # width_in = 5
    # height_in = 3
    # [x x x x x]
    # [x x x x x]
    # [x x x x x]
    # [1 0 - - -]
    # [0 1 - - -]
    #
    # [1 0 0 - - -] [0 1 0 - - -] [0 0 1 - - -]
    # [- 1 0 0 - -] [- 0 1 0 - -] [- 0 0 1 - -]
    # [- - 1 0 0 -] [- - 0 1 0 -] [- - 0 0 1 -]
    # [- - - 1 0 0] [- - - 0 1 0] [- - - 0 0 1]
    # width_out = 4
    # kw = 3


class Flatten(Constant):
    def __init__(self, input_edges: list[str], output_edge: str):
        self.input_edges = input_edges[:]
        self.output_edge = output_edge
        self.edges = input_edges + [output_edge]
        assert len(self.edges) == len(set(self.edges))

    def __repr__(self):
        return f"Flatten({self.input_edges}, {self.output_edge})"

    def __hash__(self):
        return hash((type(self).__name__, len(self.edges)))

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        return Flatten(
            [kwargs.get(e, e) for e in self.input_edges],
            kwargs.get(self.output_edge, self.output_edge),
        )
