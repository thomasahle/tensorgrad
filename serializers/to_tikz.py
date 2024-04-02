from collections import defaultdict
from tensor import Product, Zero, Copy, Variable, Sum


def to_tikz(tensor):
    tikz_code = []
    tikz_code.append("\\documentclass[tikz]{standalone}")
    tikz_code.append("\\usetikzlibrary{graphs, graphdrawing, quotes}")
    tikz_code.append("\\usegdlibrary{layered}")
    tikz_code.append("\\begin{document}")
    tikz_code.append("\\tikz[")
    tikz_code.append("  node distance=1cm,")
    tikz_code.append("  every node/.style={")
    tikz_code.append("    font=\\scriptsize,")
    tikz_code.append("    inner sep=2pt,")
    tikz_code.append("  },")
    tikz_code.append(
        "  identity/.style={circle, draw=black, fill=white, inner sep=0pt, minimum size=4pt},"
    )
    tikz_code.append("  var/.style={rectangle, draw=black, fill=white, inner sep=2pt},")
    tikz_code.append("  zero/.style={rectangle, draw=black, fill=white, inner sep=2pt},")
    tikz_code.append("  subgraph nodes={draw=gray, rounded corners},")
    tikz_code.append("  subgraph text none,")
    tikz_code.append("]")
    tikz_code.append("\\graph [layered layout, fresh nodes, nodes behind edges] {")
    _to_tikz(tensor, tikz_code)
    tikz_code.append("};")
    tikz_code.append("\\end{document}")
    return "\n".join(tikz_code)


def _node(tikz_code, node_id, node_type, label=None):
    node_id = node_id.replace("_", "+")
    if node_type == "identity":
        tikz_code.append(f"  {node_id}[identity,as=\\tiny{{\\textbullet}}];")
    elif node_type == "var":
        tikz_code.append(f"  {node_id}[var,as={label}];")
    elif node_type == "zero":
        tikz_code.append(f"  {node_id}[zero,as=0];")
    elif node_type == "invisible":
        tikz_code.append(f"  {node_id}[style={{}},as=];")
    else:
        tikz_code.append(f"  {node_id}[as={label}];")


def _edge(tikz_code, id1, id2, label=None):
    id1 = id1.replace("_", "+")
    id2 = id2.replace("_", "+")
    if label:
        tikz_code.append(f'    ({id1}) -- ["${label}$"] ({id2}),')
    else:
        tikz_code.append(f"    ({id1}) -- ({id2}),")


def count_components(con: Product):
    # Counts the individual components of a contraction,
    # that is, subgraphs that are not connected by an edge
    edges = defaultdict(list)
    for t in con.tensors:
        for e in t.edges:
            edges[e].append(t)
    colors = {}
    queue = list(con.tensors)
    while queue:
        t = queue.pop()
        if id(t) not in colors:
            colors[id(t)] = len(colors)
        for e in t.edges:
            for v in edges[e]:
                if id(v) not in colors:
                    colors[id(v)] = colors[id(t)]
                    queue.append(v)
    return max(colors.values()) + 1


def _to_tikz(tensor, tikz_code):
    if isinstance(tensor, Copy):
        _node(tikz_code, str(id(tensor)), "identity")

    if isinstance(tensor, Variable):
        _node(tikz_code, str(id(tensor)), "var", label=tensor.name)

    if isinstance(tensor, Zero):
        _node(tikz_code, str(id(tensor)), "zero")

    if isinstance(tensor, Product):
        ids = []
        for t in tensor.tensors:
            sub_id = _to_tikz(t, tikz_code)
            ids.append(sub_id)

        for e in tensor.edges:
            (t,) = [t for t in tensor.tensors if e in t.edges]
            node_id = str(id(t))
            _node(tikz_code, f"{node_id}_{e}", "invisible")
            _edge(tikz_code, node_id, f"{node_id}_{e}", label=e)

        for e in tensor.contractions:
            t1, t2 = [t for t in tensor.tensors if e in t.edges]
            node_id1 = str(id(t1))
            node_id2 = str(id(t2))
            _edge(tikz_code, node_id1, node_id2, label=e)

    if isinstance(tensor, Sum):
        cluster_id = str(id(tensor))
        tikz_code.append(f"  {cluster_id} / [inner sep=10pt] // [layered layout]{{")
        for i, (w, t) in enumerate(zip(tensor.weights, tensor.tensors)):
            style = "" if isinstance(t, Product) and count_components(t) > 1 else ", draw=none"
            tikz_code.append(
                f"{cluster_id}{i}/[label={{[anchor=east]left:${w}$}} {style}] // [layered layout]{{"
            )
            _to_tikz(t, tikz_code)
            tikz_code.append("  },")

        tikz_code.append("  },")
