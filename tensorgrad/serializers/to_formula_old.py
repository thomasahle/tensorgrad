from collections import defaultdict

from tensorgrad import Sum, Product, Variable, Delta


def to_simple_matrix_formula(expr):
    assert isinstance(expr, Sum)
    res = []
    e0 = list(expr.edges)[0]
    for prod in expr.tensors:
        assert isinstance(prod, Product)
        mat = []
        for comp in prod.components():
            print("Comp:", comp)
            # if len(comp.edges) == 0:
            #    t = comp.tensors[0]
            #    in_edge = "" # FIXME: This prevents us from getting the right transpose on the first matrix in the trace
            # else:
            #    t = next(t for t in comp.tensors if e0 in t.edges)
            #    in_edge = e0
            # s = " ".join(dfs(t, adj, visited, in_edge))
            s = component_to_matrix(comp, e0)
            # if comp.rank == 0:
            if len(comp.edges) == 0:
                s = s.replace("1", "")
                s = f"\\mathrm{{tr}}({s})"
            else:
                s = s.replace("1", "\\mathbf{1}")
            mat.append(s)
        res.append(" \\cdot ".join(mat))
    return "\n\\\\&+ ".join(res)


def component_to_matrix(comp, left_edge):
    adj = defaultdict(list)
    for t in comp.tensors:
        for e in t.edges:
            adj[e].append(t)

    # We can handle 4 cases:
    # - A matrix
    # - A trace
    # - A matrix-vector mult
    # - A vector-matrix*-vector mult

    # If there's a vector, we start from that
    # vec = next((t for t in comp.tensors if len(t.edges) == 1), None)
    # if vec is not None:
    #     # This is a vector
    #     pass

    # If there's supposed to be a free edge on the left, we start from that
    start = next((t for t in comp.tensors if left_edge in t.edges), None)
    if start is not None:
        in_edge = left_edge
    else:
        start = comp.tensors[0]
        in_edge = ""
    cur = start

    visited = set()
    res = []
    while id(cur) not in visited:
        visited.add(id(cur))
        e0, e1 = cur.edges
        res.append(cur.name)
        if e0 != in_edge:
            e0, e1 = e1, e0
            res.append("^T")
        t0, t1 = adj[e1]
        in_edge = e1
        cur = t1 if t0 is cur else t0

    return " ".join(res)


def dfs(t, adj, visited, in_edge):
    visited[t] = True
    if isinstance(t, Variable):
        yield t.name + ("^T" if in_edge == t.edges[1] else "")
    elif isinstance(t, Delta):
        yield "1"
    else:
        assert False
    for e in t.edges:
        for u in adj[e]:
            if u not in visited:
                yield from dfs(u, adj, visited, e)
