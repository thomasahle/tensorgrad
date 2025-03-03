"""
to_index.py

Transforms a tensor expression into index notation.
"""

from functools import singledispatch
import string
from typing import Iterable
import networkx as nx

from tensorgrad.extras.expectation import Expectation
from tensorgrad.tensor import Tensor, Variable, Zero, Delta, Sum, Product, Derivative, Function, Rename
from tensorgrad import functions as F  # if needed for function signatures


@singledispatch
def to_index(expr):
    raise NotImplementedError(f"Index notation not implemented for type {type(expr)}")


@singledispatch
def to_index_free(expr):
    raise NotImplementedError(f"Index notation not implemented for type {type(expr)}")


@to_index.register
def _(expr: Variable):
    if expr.edges:
        return f"{expr.name}_{{{','.join(map(str, expr.edges))}}}"
    return expr.name


@to_index_free.register
def _(expr: Variable):
    return expr.name


@to_index.register
def _(expr: Rename):
    inner = expr.tensor
    if not isinstance(inner, Variable):
        raise NotImplementedError("Rename should not be present in the final expression.")

    if not inner.edges:
        return inner.name

    names = [expr.mapping.get(e, e) for e in inner.edges]
    return f"{inner.name}_{{{','.join(names)}}}"


@to_index_free.register
def _(expr: Rename):
    return to_index_free(expr.tensor)
    # raise NotImplementedError("Rename should not be present in the final expression.")


@to_index.register
def _(expr: Zero):
    if expr.edges:
        return f"0_{{{','.join(map(str, expr.edges))}}}"
    return "0"


@to_index_free.register
def _(expr: Zero):
    return "0"


@to_index.register
def _(expr: Delta):
    # For a Kronecker delta:
    # - If there are no edges, simply output the size (as a string).
    # - Otherwise, output δ(s)_{i,j} where s is expr._size.
    size_str = str(expr._size)
    if not expr.edges:
        return size_str
    else:
        return f"δ_{{{','.join(map(str, expr.edges))}}}"


@to_index_free.register
def _(expr: Delta):
    if expr.order == 0:
        return str(expr._size)
    if expr.order == 1:
        return "𝟙"
    if expr.order == 2:
        return "I"
    return to_index(expr)  # Give up


def _general_handle_sum(expr, term_strs):
    # Sum over terms: combine each term with its coefficient.
    parts = []
    for weight, term_str in zip(expr.weights, term_strs):
        sign = ("+ " if parts else "") if weight > 0 else "- "
        w = f"{abs(weight)} " if abs(weight) != 1 else ""
        parts.append(f"{sign}{w}{term_str}")
    return " ".join(parts)


@to_index.register
def _(expr: Sum):
    term_strs = [to_index(t) for t in expr.terms]
    return _general_handle_sum(expr, term_strs)


@to_index_free.register
def _(expr: Sum):
    term_strs = [to_index_free(t) for t in expr.terms]
    return _general_handle_sum(expr, term_strs)


def _unused_edge_names(edges: Iterable[str], reserved_names: Iterable[str]) -> dict[str, str]:
    reserved_names = set(reserved_names)
    good_names = [c for c in string.ascii_lowercase if c not in reserved_names]
    rename = {}
    for e in edges:
        if e in rename:
            continue
        c = e[0]
        if c not in reserved_names:
            rename[e] = c
            if c in good_names:
                good_names.remove(c)
            reserved_names.add(c)
            continue
        # The good names should alsways be available
        c = good_names.pop(0)
        rename[e] = c
        reserved_names.add(c)
    return rename


def _handle_path(tensors: list[Tensor], edges: list[str]):
    res = []
    for t, in_edge, out_edge in zip(tensors, edges[:-1], edges[1:]):
        inner_str = to_index_free(t)
        if t.order == 1:
            (e,) = t.edges
            if in_edge is None:
                inner_str += "^T"
                assert out_edge == e
        elif isinstance(t, Delta) and len(tensors) != 1:
            continue
        else:
            a, b = t.edges  # We assume the order is meaningful
            if (in_edge, out_edge) != (a, b):
                assert (in_edge, out_edge) == (b, a), f"Edges do not match: {in_edge, out_edge} != {a, b}"
                inner_str += "^T"
        res.append(inner_str)
    return " ".join(res)


def _handle_trace(tensors: list[Tensor], out_edges: list[str]):
    assert len(tensors) >= 2

    if all(isinstance(t, Delta) for t in tensors):
        return str(tensors[0]._size)

    in_edges = out_edges[-1:] + out_edges[:-1]
    factors = []
    trans = []
    for t, in_edge, out_edge in zip(tensors, in_edges, out_edges):
        assert t.order == 2
        if isinstance(t, Delta):
            continue
        a, b = t.edges
        if (in_edge, out_edge) == (a, b):
            trans.append("M")
        elif (in_edge, out_edge) == (b, a):
            trans.append("T")
        else:
            raise ValueError(f"Edges do not match: {in_edge, out_edge} != {a, b}")
        factors.append(to_index_free(t))

    # Compute the lexicographically minimal rotation of the (factor, marker) pairs.
    best_pairs = lexicographically_minimal_rotation(factors, trans)
    return "tr(" + " ".join(f if marker == "M" else f"{f}^T" for f, marker in best_pairs) + ")"


def lexicographically_minimal_rotation(factors: list[str], trans: list[str]):
    """
    Given two lists, `factors` and `trans` (each of the same length),
    compute all rotations of the paired list (both the original ordering
    and the reversed ordering with markers swapped) and return the rotated
    list (of pairs) corresponding to the lexicographically smallest trans string.
    """
    pairs = list(zip(factors, trans))
    n = len(pairs)
    candidates = []
    # Option 1: rotations of the original order
    for r in range(n):
        rotated = pairs[r:] + pairs[:r]
        trans_str = "".join(marker for (_, marker) in rotated)
        candidates.append((trans_str, rotated))

    # Option 2: rotations of the reversed order (with markers swapped)
    def swap_marker(m):
        return "M" if m == "T" else "T" if m == "M" else m

    reversed_pairs = [(f, swap_marker(m)) for f, m in reversed(pairs)]
    for r in range(n):
        rotated = reversed_pairs[r:] + reversed_pairs[:r]
        trans_str = "".join(marker for (_, marker) in rotated)
        candidates.append((trans_str, rotated))
    # Choose the rotated pair list with lexicographically smallest trans string.
    _, best_pairs = min(candidates, key=lambda x: x[0])
    return best_pairs


@to_index.register
def _(prod: Product):
    # Rename all the inner edges to nicer names, since we are going to display them.
    inner_edges = {e for t in prod.factors for e in t.edges if e not in prod.edges}
    rename = _unused_edge_names(inner_edges, prod.edges)
    parts = [to_index(t.rename(**rename)) for t in prod.factors]
    return " ".join(parts)


@to_index_free.register
def _(prod: Product):
    G = nx.MultiGraph()
    G.add_nodes_from(range(len(prod.factors)))
    for i, t1 in enumerate(prod.factors):
        for j, t2 in enumerate(prod.factors[i + 1 :], i + 1):
            for e in set(t1.edges) & set(t2.edges):
                G.add_edge(i, j, label=e)

    # Find connected components
    res = []
    for component in nx.connected_components(G):
        tensors = [prod.factors[i] for i in component]
        subgraph = G.subgraph(component)
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        degree_one_nodes = [n for n in subgraph.nodes if subgraph.degree[n] == 1]

        if len(component) == 1:
            res.append(to_index_free(tensors[0]))
        elif any(t.order > 2 for t in tensors):
            res.append(to_index(Product(tensors)))
        elif num_edges == num_nodes and len(degree_one_nodes) == 0:
            cycle = nx.find_cycle(subgraph)
            edges = [G.edges[edge]["label"] for edge in cycle]
            nodes = [prod.factors[i] for i, _, _ in cycle]
            res.append(_handle_trace(nodes, edges))
        elif num_edges == num_nodes - 1 and len(degree_one_nodes) == 2:
            # Determine right direction of path
            start, end = degree_one_nodes
            prod_edges = list(prod.edges)
            e1s, e2s = tensors[start].edges & prod_edges, tensors[end].edges & prod_edges
            # We need to determine the type of path. Is it a (0, 0) path, like v^T M v,
            # or a (1, 0) path, like M v? Or (1, 1) like M N?
            e1 = list(e1s)[0] if e1s else None
            e2 = list(e2s)[0] if e2s else None
            if e1 is not None and e2 is not None:
                if prod_edges.index(e1) > prod_edges.index(e2):
                    start, end = end, start
                    e1, e2 = e2, e1
            # Then find the path
            order = list(nx.shortest_path(subgraph, source=start, target=end))
            edges = [e1] + [G.edges[i, j, 0]["label"] for i, j in nx.utils.pairwise(order)] + [e2]
            nodes = [tensors[i] for i in order]
            res.append(_handle_path(nodes, edges))
        else:
            # Just use the general product representation
            res.append(to_index(Product(tensors)))

    return " ".join(res)


@to_index.register
def _(expr: Derivative):
    return f"d({to_index(expr.x)})/d({to_index(expr.x)})"


@to_index_free.register
def _(expr: Derivative):
    return f"d({to_index_free(expr.x)})/d({to_index_free(expr.x)})"


@to_index.register
def _(expr: Function):
    args_str = ", ".join(to_index(arg) for arg in expr.inputs)
    if isinstance(expr.signature, F._PowerFunction):
        return f"({args_str})^{{{expr.signature.k}}}"
    return f"{expr.signature.name}({args_str})"


@to_index_free.register
def _(expr: Function):
    args_str = ", ".join(to_index_free(arg) for arg in expr.inputs)
    if isinstance(expr.signature, F._PowerFunction):
        return f"({args_str})^{{{expr.signature.k}}}"
    return f"{expr.signature.name}({args_str})"


@to_index.register
def _(expr: Expectation):
    return f"E_{expr.wrt.name}[{to_index(expr.tensor)}]"


@to_index_free.register
def _(expr: Expectation):
    return f"E_{expr.wrt.name}[{to_index_free(expr.tensor)}]"
