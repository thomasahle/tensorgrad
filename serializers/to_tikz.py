from collections import defaultdict
from tensor import Product, Zero, Copy, Variable, Sum, Function


class TikzGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.subgraphs = []

    def add_node(self, node_id, node_type, label=None):
        print(f"adding node {node_id} of type {node_type} with label {label}")
        node_id = node_id.replace("_", "+")
        if isinstance(label, str):
            label = label.replace("_", "+")
        if node_type == "identity":
            self.nodes.add(f"  {node_id}[identity,as=\\tiny{{\\textbullet}}];")
        elif node_type == "var":
            self.nodes.add(f"  {node_id}[var,as={label}];")
        elif node_type == "zero":
            self.nodes.add(f"  {node_id}[zero,as=0];")
        elif node_type == "function":
            self.nodes.add(f"  {node_id}[function,as={label}];")
        elif node_type == "invisible":
            self.nodes.add(f"  {node_id}[style={{}},as=];")
        else:
            self.nodes.add(f"  {node_id}[as={label}];")

    def add_edge(self, id1, id2, label=None, directed=False):
        print(f"adding edge {id1} -> {id2} with label {label}")
        id1 = id1.replace("_", "+")
        id2 = id2.replace("_", "+")
        if isinstance(label, str):
            label = label.replace("_", "+")
        assert id1 in [
            node.split("[")[0].strip() for node in self.nodes
        ], f"Node {id1} does not exist"
        assert id2 in [
            node.split("[")[0].strip() for node in self.nodes
        ], f"Node {id2} does not exist"
        edge_type = " -> " if directed else " -- "
        if label:
            self.edges.append(f'    ({id1}){edge_type}["${label}$"] ({id2}),')
        else:
            self.edges.append(f"    ({id1}){edge_type}({id2}),")

    def add_subgraph(self, subgraph):
        self.subgraphs.append(subgraph)

    def to_tikz(self):
        tikz_code = []
        for node in self.nodes:
            tikz_code.append(node)
        for edge in self.edges:
            tikz_code.append(edge)
        for subgraph in self.subgraphs:
            tikz_code.append(subgraph.to_tikz())
        return "\n".join(tikz_code)


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
    tikz_code.append("  function/.style={rectangle, draw=black, fill=white, inner sep=2pt},")
    tikz_code.append("  subgraph nodes={draw=gray, rounded corners},")
    tikz_code.append("  subgraph text none,")
    tikz_code.append("]")
    tikz_code.append("\\graph [layered layout, fresh nodes, nodes behind edges] {")

    graph = TikzGraph()
    _to_tikz(tensor, graph)

    tikz_code.append(graph.to_tikz())
    tikz_code.append("};")
    tikz_code.append("\\end{document}")
    return "\n".join(tikz_code)


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


def _to_tikz(tensor, graph):
    if isinstance(tensor, Copy):
        graph.add_node(node_id := str(id(tensor)), "identity")
        return node_id

    if isinstance(tensor, Variable):
        graph.add_node(node_id := str(id(tensor)), "var", label=tensor.name)
        return node_id

    if isinstance(tensor, Zero):
        graph.add_node(node_id := str(id(tensor)), "zero")
        return node_id

    if isinstance(tensor, Function):
        graph.add_node(node_id := str(id(tensor)), "function", label=tensor.name)
        for i, (t, e) in enumerate(zip(tensor.tensors, tensor.edges_in)):
            sub_id = _to_tikz(t, graph)
            graph.add_edge(sub_id, str(id(tensor)), label=e, directed=True)
        # This is the same issue as with the Variable: If the node is not in a product,
        # who's responsible for adding the invisible edges?
        # for e in tensor.edges:
        #     graph.add_node(f"{id(tensor)}_{e}", "invisible")
        #     graph.add_edge(str(id(tensor)), f"{id(tensor)}_{e}", label=e)
        return node_id

    if isinstance(tensor, Product):
        sub_ids = {}
        for t in tensor.tensors:
            sub_ids[id(t)] = _to_tikz(t, graph)

        for e in tensor.edges:
            (t,) = [t for t in tensor.tensors if e in t.edges]
            node_id = sub_ids[id(t)]
            graph.add_node(f"{node_id}_{e}", "invisible")
            graph.add_edge(node_id, f"{node_id}_{e}", label=e)

        for e in tensor.contractions:
            t1, t2 = [t for t in tensor.tensors if e in t.edges]
            graph.add_edge(sub_ids[id(t1)], sub_ids[id(t2)], label=e)

        # FIXME: What sub-id should we return?
        # well, it depends on which free edge the parent is interested in...
        # Unless we wrap the whole thing in a subgraph, but we aren't doing that here.
        # Actually it's worse, because we are already creating the edges for those free edges.
        return list(sub_ids.values())[0]

    if isinstance(tensor, Sum):
        cluster_id = str(id(tensor))
        subgraph = TikzGraph()
        for i, (w, t) in enumerate(zip(tensor.weights, tensor.tensors)):
            style = "" if isinstance(t, Product) and count_components(t) > 1 else ", draw=none"
            subsubgraph = TikzGraph()
            _to_tikz(t, subsubgraph)
            subgraph.add_subgraph(subsubgraph)
            subgraph.add_node(
                f"{cluster_id}{i}",
                "subgraph",
                label=f"{{[anchor=east]left:${w}$}} {style}",
            )

        graph.add_subgraph(subgraph)
        return cluster_id

    assert False, "Unknown tensor type"
