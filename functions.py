from typing import Any, Callable, Union
import torch
from tensor import Function, Ones, Tensor, Product, Copy, make_distinct
from math import factorial

# We mostly try to follow the behavior of pytorch's named tensors:
# https://pytorch.org/docs/stable/name_inference.html


def frobenius2(t: Tensor) -> Tensor:
    return Product([t, t])


def einsum(tensors, output_edges):
    # Basically like Product, but will create some Identity's to ensure only the free_edges are free afterwards.
    # cnt_edges = Counter(e for t in tensors for e in t.edges)
    all_free_edges = {e for t in tensors for e in t.edges}
    dis_tensors, renames = make_distinct(*tensors, preserve_free=False, used_names=all_free_edges)
    joins = []
    for e in all_free_edges:
        edges = [rename[e] for rename in renames if e in rename]
        if e in output_edges:
            edges.append(e)
        joins.append(Copy(edges))
    return Product(dis_tensors + joins)


def kronecker(*tensors):
    # Basically just rename everything to be distinct, then contraction
    # FIXME: This method returns the tensor product, not the Kronecker product.
    #        To get the Kronecker product you have to flatten the output tensors.
    dis_tensors, _renames = make_distinct(*tensors, preserve_free=False)
    return Product(dis_tensors)


def diag(t: Tensor, new_edges: list[str]):
    """Takes vector `t` and creates a diagonal matrix with `t` on the diagonal."""
    if len(t.edges) != 1:
        raise ValueError("Expected a vector, got a tensor with more than one edge.")
    # If the vector's edge is in new_edges, we need to rename it
    (t,), _renames = make_distinct(t, preserve_free=False, used_names=new_edges)
    return Product([Copy(new_edges + t.edges), t])


def sum(tensor: Tensor, edges: list[str], keepdims=False) -> Tensor:
    """Sum the tensor over the given dimensions."""
    out = Product([tensor] + [Copy([e]) for e in edges])
    if keepdims:
        return out @ Ones(edges)
    return out


# The common type of cuntion in ML is that of the broadcasted function.
# Say we have shape (B, N, M), we'll typically apply a function "along" axis M,
# which means it takes a vector as an input and outputs a vector.
# And we call it on each vector from (B, N).
#
# It would be cool to express functions such as max(dim=...) as a vector, meaning
# it would show up in the product graph as a node with a single edge, which connects
# to the axis "over which" we apply the function. This matches the intuition of an
# inner product with a vector along that axis, which ends up removing it.
#
# But in this framework, how would we express an elementwise function? It wouldn't have
# any edges to connect to.
class Elementwise(Function):
    def __init__(self, name: str, function: Callable, t: Tensor, derivative: Callable = None):
        # An element wise function takes no input edges and output no output edges
        # That makes it kinda difficult to visualize in a graph... Might just have to surround it
        super().__init__(name, [], (t,))
        self.function = function
        self.derivative = derivative

    def inner_grad(self, i, new_edges) -> Tensor:
        # print("inner_grad", i, new_edge, f"{self.derivative()=}")
        print("inner_grad", self.tensors[0].edges, new_edges)
        return self.derivative() @ Copy(new_edges)

    def edge_dims(self, edge_dims: dict[str, int]) -> dict[str, int]:
        return {}  # No output edges

    def __call__(self, value: torch.tensor) -> torch.tensor:
        return self.function(value)

    def __repr__(self):
        return f"{self.name}({self.tensors[0]})"

    def simplify(self, args: dict[str, Any] = {}):
        return Elementwise(self.name, self.function, self.tensors[0].simplify(args=args), self.derivative)

    def rename(self, kwargs: dict[str, str]):
        return Elementwise(self.name, self.function, self.tensors[0].rename(kwargs), self.derivative)


def log(t: Tensor) -> Tensor:
    return Elementwise("log", torch.log, t, lambda: pow(t, -1))


def exp(t: Tensor) -> Tensor:
    return Elementwise("exp", torch.exp, t, lambda: exp(t))


def pow(tensor: Tensor, k: int) -> Tensor:
    """Elementwise t^k"""
    if k == 0:
        return Ones(tensor.edges)
    return Elementwise(
        f"pow({k})",
        lambda x: torch.pow(x, k),
        tensor,
        lambda: k * pow(tensor, k - 1),
    )


def softmax(t: Tensor, dims: list[str]) -> Tensor:
    e = exp(t)
    return e * pow(sum(e, dims, keepdims=True), -1)


def cross_entropy(t: Tensor, y: Tensor, dims: list[str]) -> Tensor:
    return -(y * log(softmax(t, dims))).sum(dims)


# Some questions:
# - Who's responsible for realizing that 1/x and x cancel out?
#   - Maybe tensors can register simplification rules
# - How do we get names for the derivatives?
# - Should functions be forced to output the right edge names?
# - What really is going on with multiple inputs?
