from collections import Counter, defaultdict
import itertools
import math
from typing import Any
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
    def simplify_outer(cls, tensors: list[Tensor]) -> list[Tensor]:
        original_edges = Product(tensors).edges
        # TODO: If the content of a pow is not a single tensor, but a product, we can't expect to find a single match
        # but instead need to look for a similar subgraph. This is a bit more complicated.
        # VF2 has an "isomorphic subgraph search" funtion we can probably use.

        # First group tensors by their edges, taking into account existing Pow functions.
        # We want to group tensors with that share edges, which is made harder by Copy tensors
        # that rename the hyper edges to different names. We make this dict to rename everything
        # to a canonical name.
        hyperedges = {e: min(c.edges) for c in tensors if isinstance(c, Copy) for e in c.edges}
        partition = defaultdict(list)
        for t in tensors:
            # We don't include Copy's since they have been reduced to hyper-edges
            if isinstance(t, Copy):
                continue
            if isinstance(t, Function) and isinstance(t.fn_info, PowFunctionInfo):
                power_weight = t.fn_info.k
                ((t, *_es),) = t.inputs
            else:
                power_weight = 1
            # We rename the hyperedges to a canonical name for matching
            key = MatchEdgesKey(t.rename(**hyperedges))
            partition[key].append((power_weight, t))

        # Now we have a partition of tensors that share the same edges, and we can combine them.
        # Or in some cases they cancel each other.
        tensors = [c for c in tensors if isinstance(c, Copy)]
        for ts in partition.values():
            w = _sum(w for w, t in ts)
            t0 = ts[0][1]
            tensors.append(pow(t0, w))
            # The remaining tensors have been reduced to ones. This prevents leaving unattached edges
            # on the copy tensors.
            for _, t in ts[1:]:
                tensors.append(Ones(**t.shape))

        assert Product(tensors).edges == original_edges

        # Next we look for disjoint sets of tensors that we can group together.
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

        # We have to merge products here because we might otherwise have undone part of the simplification
        tensors = Product.merge([Product([t]) if not isinstance(t, Product) else t for t in tensors]).tensors

        assert Product(tensors).edges == original_edges
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

    def derivative(self, i, new_edges, t):
        # TODO: The mask needs to not be symmetric, which is assumed by the current Constant Tensor
        mask = ...
        return mask * t


def relu(t: Tensor) -> Tensor:
    return Function(ReluFunctionInfo(), [], (t,))


def abs(t: Tensor) -> Tensor:
    raise NotImplementedError


class Mask(Constant):
    # TODO
    """Matrix such that Z_{i,j,k} = 0 for all i, j, k"""

    def _inner_evaluate(self, edge_dims: dict[str, int], values: dict, extras: dict) -> torch.tensor:
        if not self.edges:
            return torch.tensor(0.0)
        return torch.zeros([edge_dims[e] for e in self.edges]).rename(*self.edges)


def gt(t: Tensor, dim: str) -> Tensor:
    """Returns a tensor that's 1 for the largest index in the row (along dim), 0 elsewhere."""

    def inner(x):
        indices = torch.max(x, dim=dim).indices
        one_hot = torch.zeros_like(x)
        one_hot.scatter_(dim=dim, index=indices.unsqueeze(x.names.index(dim)), value=1)

    return Function(
        FunctionInfo(
            "gt",
            eval=lambda x: inner(x),
            derivative=lambda _i, new_edges, t: Zero(t.edges + new_edges),
        ),
        [],
        (t, dim),
    )


def max(t: Tensor, dim: str, keepdim=False) -> Tensor:
    func = Function(
        FunctionInfo(
            "max",
            eval=lambda x: torch.max(x, dim=dim),
            derivative=lambda _i, _nn, t: gt(t, dim),
        ),
        [],
        (t, dim),
    )
    if keepdim:
        func @= Ones(**{dim: t.shape[dim]})
    return func


def foldl(f: FunctionInfo, inputs, dim: str) -> FunctionInfo:
    """Fold the function along the given dimension."""
    # TODO: Maybe FunctionInfo should contain more info about the function.
    # Like the output_shape, and even inputs somehow.
    raise NotImplementedError


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
