from tensor import Tensor, Product, Copy, make_distinct


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
    dis_tensors, _renames = make_distinct(*tensors, preserve_free=False)
    return Product(dis_tensors)


def diag(t: Tensor, new_edges: list[str]):
    """Takes vector `t` and creates a diagonal matrix with `t` on the diagonal."""
    if len(t.edges) != 1:
        raise ValueError("Expected a vector, got a tensor with more than one edge.")
    # If the vector's edge is in new_edges, we need to rename it
    (t,), _renames = make_distinct(t, preserve_free=False, used_names=new_edges)
    return Product([Copy(new_edges + t.edges), t])


def sum(tensor, edges):
    """Sum the tensor over the given dimensions."""
    return Product([tensor] + [Copy([e]) for e in edges])
