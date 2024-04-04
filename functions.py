from typing import Callable
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


class Elementwise(Function):
    def __init__(self, name: str, function: Callable, t: Tensor, derivative: Callable = None):
        super().__init__(name, [t], edges_in=t.edges, edges_out=t.edges)
        self.function = function
        self.derivative = derivative

    def edge_dims(self, edge_dims: dict[str, int]) -> dict[str, int]:
        return edge_dims  # Element-wise functions don't change the dimensions

    def __call__(self, *values: list[torch.tensor], edges):
        return self.function(*values)


def log(t: Tensor) -> Tensor:
    return Elementwise("log", torch.log, t, lambda n, x: torch.pow(-x, -n) * factorial(n - 1))


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
        lambda x: k * pow(x, k - 1),
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
