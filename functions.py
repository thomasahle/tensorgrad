from tensor import Tensor, Product, Copy, free_edge_names


def frobenius2(t: Tensor) -> Tensor:
    return Product([t, t.copy()])


def einsum(tensors, free_edges):
    raise NotImplementedError
    # Basically like Contraction, but will create some Identity's to ensure only the free_edges are free afterwards.
    cnt_edges = Counter(sum(((t.edges for t in tensors)), []))
    dis_tensors = make_distinct(tensors)
    joins = []
    for e, cnt in cnt_edges.most_common():
        edges = [f"{e}{i}" for i in range(cnt)]
        if e in free_edges:
            edges.append(e)
        joins.append(Identity(edges))
    return Product(dis_tensors + joins)


def kronecker(tensors):
    # Basically just rename everything to be distinct.
    # Then contraction
    ...


def diag(t: Tensor, new_edges: list[str]):
    """Takes vector `t` and creates a diagonal matrix with `t` on the diagonal."""
    # Make sure the edge of `t` doesn't appear in `new_edges`
    es, rename = free_edge_names(t.edges, new_edges)
    return Product(
        [
            t.rename(**rename),
            Copy(es + new_edges),
        ]
    )


def sum(tensor, dims):
    """Sum the tensor over the given dimensions."""
    return tensor @ Copy(dims)
