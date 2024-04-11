import torch
from tensorgrad.tensor import Function, FunctionInfo, Ones, Tensor, Product, Copy, Zero, make_distinct

# We mostly try to follow the behavior of pytorch's named tensors:
# https://pytorch.org/docs/stable/name_inference.html

# TODO:
# - Add all the functions of PyTorch
# - Add general function inverses, implicit functions
# - Taylor approximation


def frobenius2(t: Tensor) -> Tensor:
    return Product([t, t])


def einsum(tensors, output_edges):
    if len(output_edges) != len(set(output_edges)):
        # We don't support einsums like "i -> ii".
        # We also don't support "ii -> i", but that's more hidden, because the input tensors can't have double edges.
        raise ValueError("Output edges must be unique.")
    # Basically like Product, but will create some Identity's to ensure only the free_edges are free afterwards.
    all_free_edges = {e for t in tensors for e in t.edges}
    # TODO: We only really need to rename the free edges of each tensor, so `make_distinct`` is overkill.
    dis_tensors, renames = make_distinct(*tensors, used_names=all_free_edges)
    joins = []
    for e in all_free_edges:
        # We create a Copy([...]) with all the entries that have this edge
        edges = [rename[e] for rename in renames if e in rename]
        if e in output_edges:
            edges.append(e)
        joins.append(Copy(edges))
    return Product(dis_tensors + joins)


def kronecker(*tensors):
    # Basically just rename everything to be distinct, then contraction
    # Note: This function returns the tensor product, which is different from the
    #       Kronecker product as often described in the literature. To get the
    #       Kronecker product you have to flatten the output tensors.
    dis_tensors, _renames = make_distinct(*tensors)
    return Product(dis_tensors)


def diag(t: Tensor, new_edges: list[str]):
    """Takes vector `t` and creates a diagonal matrix with `t` on the diagonal."""
    if len(t.edges) != 1:
        raise ValueError("Expected a vector, got a tensor with more than one edge.")
    # If the vector's edge is in new_edges, we need to rename it
    (t,), _renames = make_distinct(t, used_names=new_edges)
    return Copy(new_edges + t.edges) @ t


def sum(tensor: Tensor, edges: list[str] = None, keepdims=False) -> Tensor:
    """Sum the tensor over the given dimensions."""
    edges = edges or tensor.edges
    out = Product([tensor] + [Copy([e]) for e in edges])
    if keepdims:
        return out @ Ones(edges)
    return out


def trace(tensor: Tensor) -> Tensor:
    return tensor @ Copy(tensor.edges)


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


def pow(tensor: Tensor, k: int) -> Tensor:
    # Maybe pow should be moved into the tensor.py file, as it has some special properties:
    # - It's necessary to implement __div__
    # - It can result in cancelations in Product.simplify
    # - It can factor its inputs in Function.simplify
    # - pow(1) just vanishes
    """Elementwise t^k"""
    if k == 0:
        return Ones(tensor.edges)
    return Function(
        FunctionInfo(
            f"pow({k})",
            eval=lambda x: torch.pow(x, k),
            derivative=lambda _i, _nn, t: k * pow(t, k - 1),
        ),
        [],
        (tensor,),
    )


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


def cross_entropy(t: Tensor, y: Tensor, dims: list[str]) -> Tensor:
    if set(dims) - set(t.edges):
        raise ValueError("dims must be a subset of t.edges")
    return -sum(y * log(softmax(t, dims)), dims)


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
        func @= Ones([dim])
    return func
