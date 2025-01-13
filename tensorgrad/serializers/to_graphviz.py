from collections import defaultdict
import graphviz
from tensorgrad.tensor import Product, Zero, Copy, Variable, Sum


def _edge(graph, id1, id2, label):
    # Just an annoying thing where graphviz distinguishes between nodes and subgraphs
    kwargs = {}

    for name, sub_name, id_ in [
        ("tail_name", "ltail", id1),
        ("head_name", "lhead", id2),
    ]:
        if isinstance(id_, tuple):
            cluster_id, inner_id = id_
            kwargs[name] = inner_id
            kwargs[sub_name] = cluster_id
        else:
            kwargs[name] = id_
    graph.edge(**kwargs, label=label)


def to_graphviz(tensor):
    g = graphviz.Graph()
    g.attr(compound="true")
    _ = _to_graphviz(tensor, g)
    return g


def _to_graphviz(tensor, g):
    if isinstance(tensor, Copy):
        g.node(node_id := str(id(tensor)), shape="point")
        return node_id

    if isinstance(tensor, Variable):
        # Fixme: If variables (or identity/zero) are not wrapped in Contraction,
        # their edges won't be shown...
        g.node(node_id := str(id(tensor)), label=tensor.name, shape="square")
        return node_id

    if isinstance(tensor, Zero):
        g.node(node_id := str(id(tensor)), label="0", shape="square")
        return node_id

    if isinstance(tensor, Product):
        cluster_id = f"cluster_{id(tensor)}"
        with g.subgraph(name=cluster_id) as outer:
            # Add nodes for each tensor
            ids = {}
            for t in tensor.tensors:
                sub_id = _to_graphviz(t, outer)
                ids[id(t)] = sub_id

            # Add edges for uncontracted edges
            for e in tensor.edges:
                (t,) = [t for t in tensor.tensors if e in t.edges]
                node_id = str(id(t))
                outer.node(f"{node_id}_{e}", label="", style="invisible")
                _edge(outer, node_id, f"{node_id}_{e}", label=e)

            # Add edges for contracted edges
            for e in tensor.contractions:
                t1, t2 = [t for t in tensor.tensors if e in t.edges]
                _edge(outer, ids[id(t1)], ids[id(t2)], label=e)

            # We have to add an extra node somewhere inside the subgraph for the subgraph edges to work
            # outer.node(f"{cluster_id}_inner", label="", style="invisible")
        # return cluster_id, f"{cluster_id}_inner"
        while type(sub_id) == tuple:
            _, sub_id = sub_id
        return cluster_id, sub_id

    if isinstance(tensor, Sum):
        cluster_id = f"cluster_{id(tensor)}"
        with g.subgraph(name=cluster_id) as outer:
            outer.attr(style="rounded", color="black", cluster="true")
            for i, (w, t) in enumerate(zip(tensor.weights, tensor.tensors)):
                with outer.subgraph(name=f"{cluster_id}_{i}") as c:
                    c.attr(style="solid", color="white")
                    node_id = _to_graphviz(t, c)
                    c.attr(label=str(w), labelloc="t")
            # outer.node(f"{cluster_id}_inner", label="", style="invisible")
        # return cluster_id, f"{cluster_id}_inner"
        while type(node_id) == tuple:
            _, node_id = node_id
        return cluster_id, node_id
