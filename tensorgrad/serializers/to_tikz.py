from collections import defaultdict
from tensorgrad.tensor import Derivative, Product, Zero, Copy, Variable, Sum, Function
import random
import re
from dataclasses import dataclass


def format_label(label):
    def replacement(match):
        digits = match.group(0)
        return "_" + "{" + ",".join(filter(None, digits.split("_"))) + "}"

    pairs = []
    while m := re.search(r"(\d+)(_*)$", label):
        number, underscores = m.groups()
        pairs.append((len(underscores), int(number)))
        label = label[: -len(underscores) - len(number)]
    if m := re.search(r"(_+)$", label):
        (underscores,) = m.groups()
        u = len(underscores)
        label = label[:-u] + "'" * u
    if pairs:
        pairs.reverse()
        label += "_{" + ",".join(f"{n}" + "'" * (u) for u, n in pairs) + "}"

    style = ""
    if "D_" in label:
        # label = re.sub("D_[\d+]", "", label, count=1)
        style = "double"
    # TODO: Handle more levels of D

    return label, style


@dataclass
class Node:
    label: str
    node_id: int
    type: str
    style: str


@dataclass
class Edge:
    label: str
    id1: int
    id2: int
    style: str


class TikzGraph:
    def __init__(self):
        self.lines = []
        self.node_ids = set()

    def add_node(self, node_id, node_type, label=None):
        # print(f"adding node {node_id} of type {node_type} with label {label}")
        if isinstance(node_id, dict):
            node_id = node_id["node_id"]
        node_id = node_id.replace("_", "+")
        if node_id in self.node_ids:
            print("Warning: Node already exists. Ignoring")
            return
        self.node_ids.add(node_id)
        if label is not None:
            label, extra_style = format_label(label)
        else:
            label, extra_style = "", ""
        nudge = f"nudge=(left:{random.random()-.5:.3f}em)"
        # nudge = ""
        if node_type == "identity":
            self.lines.append(f"  {node_id}[identity,as=\\tiny{{\\textbullet}},{nudge}];")
        elif node_type == "var":
            self.lines.append(f"  {node_id}[var,as=${label}$,{nudge}];")
        elif node_type == "zero":
            self.lines.append(f"  {node_id}[zero,as=0,{nudge}];")
        elif node_type == "function":
            if len(label) == 1 or "_" in label:
                label = f"${label}$"
            self.lines.append(f"  {node_id}[function,as={label},style={{{extra_style}}},{nudge}];")
        elif node_type == "invisible":
            self.lines.append(f"  {node_id}[style={{}},as=,{nudge}];")
        else:
            self.lines.append(f"  {node_id}[as=${label}$,{nudge}];")

    def add_edge(self, id1, id2, label, directed=False):
        # print(f"adding edge ({id1}) -> ({id2}) with label {label}")
        style = ""
        if isinstance(id1, dict):
            style = id1.get("style", "")
            id1 = id1["node_id"]
        if isinstance(id2, dict):
            style = id2.get("style", "")
            id2 = id2["node_id"]
        id1 = id1.replace("_", "+")
        id2 = id2.replace("_", "+")
        if isinstance(label, str):
            label, _style = format_label(label)
        assert id1 in self.node_ids, f"Node {id1} does not exist in {self.node_ids}"
        assert id2 in self.node_ids, f"Node {id2} does not exist in {self.node_ids}"
        if directed:
            edge_type = " -> "
            style = "-latex"
        else:
            edge_type = " -- "
        self.lines.append(f'    ({id1}){edge_type}[{style}, "${label}$"] ({id2});')

    def add_subgraph(self, subgraph, definition: str, cluster_id: str):
        self.lines.append(f"{definition}{{")
        self.lines += subgraph.lines
        self.lines.append("},")
        self.node_ids |= subgraph.node_ids
        self.node_ids.add(cluster_id)

    def to_tikz(self):
        return "\n".join(self.lines)


tree_layout = """\
[
    tree layout,
    % spring layout,
    %grow'=right,
    fresh nodes,
    sibling sep=3em,
    %level sep=3em,
    node sep=3em,
    nodes behind edges,
]"""

spring_layout = """\
    [
    spring electrical layout,
    %spring electrical layout',
    %spring electrical Walshaw 2000 layout,
    fresh nodes,
    %sibling sep=3em,
    nodes behind edges,
    ]"""


def layout(depth):
    if depth % 2 == 1:
        return tree_layout
    return tree_layout


def to_tikz(tensor):
    prefix = (
        """
    \\documentclass[tikz]{standalone}
    \\usetikzlibrary{graphs, graphdrawing, quotes, arrows.meta, decorations.markings}
    \\usegdlibrary{trees, layered, force}
    \\begin{document}
    \\tikz[
        every node/.style={
            font=\\scriptsize,
            inner sep=2pt,
        },
        identity/.style={circle, draw=black, fill=white, inner sep=0pt, minimum size=4pt},
        var/.style={circle, draw=black, fill=white, inner sep=2pt},
        zero/.style={rectangle, draw=black, fill=white, inner sep=2pt},
        function/.style={circle, draw=black, fill=white, inner sep=2pt},
        subgraph nodes={draw=gray, rounded corners},
        subgraph text none,
    ]
    \\graph """
        + layout(depth=0)
        + "{"
    )

    tikz_code = [prefix]
    graph = TikzGraph()
    free_edges = _to_tikz(tensor, graph)
    # print("final free edges", free_edges)
    if not isinstance(tensor, Sum):
        # Sum handles free edges itself
        handle_free_edges(free_edges, graph)

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


def handle_free_edges(free_edges, graph):
    # print("handling free edges", free_edges)
    for e, node_id in free_edges.items():
        if isinstance(node_id, dict):
            new_node_id = node_id["node_id"] + "_" + e
        else:
            new_node_id = node_id + "_" + e
        graph.add_node(new_node_id, "invisible")
        graph.add_edge(node_id, new_node_id, label=e)


def _to_tikz(tensor, graph, depth=1):
    # We don't want the node_id to be tied to id(tensor), since, we often want
    # multiple tikz nodes for the same variable
    node_id = str(random.randrange(2**64))

    if isinstance(tensor, Copy):
        graph.add_node(node_id, "identity")
        return {e: node_id for e in tensor.edges}

    if isinstance(tensor, Variable):
        graph.add_node(node_id, "var", label=tensor.name)
        return {e: node_id for e in tensor.edges}

    if isinstance(tensor, Zero):
        graph.add_node(node_id := str(id(tensor)), "zero")
        return {e: node_id for e in tensor.edges}

    if isinstance(tensor, Function):
        cluster_id = f"cluster+{node_id}"
        subgraph = TikzGraph()

        subgraph.add_node(node_id, "function", label=tensor.name)

        free_edges = {}
        for t, *es in tensor.inputs:
            edges = _to_tikz(t, subgraph, depth + 1)
            for e in es:
                sub_id = edges.pop(e)
                subgraph.add_edge(sub_id, node_id, label=e, directed=True)
            # Add remaining edges to free edges.
            # Note: The Function should have made sure there is no edge overlap here
            assert not (edges.keys() & free_edges.keys())
            free_edges |= edges

        graph.add_subgraph(subgraph, f"{cluster_id} // {tree_layout}", f"{cluster_id}")

        # We propagate the free edges to the parent to handle
        return {e: node_id for e in tensor.edges_out} | free_edges

    if isinstance(tensor, Derivative):
        cluster_id = f"cluster+{node_id}"
        subgraph = TikzGraph()
        edges = _to_tikz(tensor.tensor, subgraph, depth + 1)
        for e in tensor.new_names:
            edges[e] = {"node_id": cluster_id, "style": "Circle-"}
        style = "[draw=black, circle]"
        graph.add_subgraph(subgraph, f"{cluster_id} {style} // {tree_layout}", f"{cluster_id}")
        return edges

    if isinstance(tensor, Product):
        sub_ids = defaultdict(list)  # edge -> [sub_id1, sub_id2]
        for t in tensor.tensors:
            for e, sub_id in _to_tikz(t, graph, depth + 1).items():
                sub_ids[e].append(sub_id)

        # Handle contractions (edges with multiple sub_ids)
        for e, ts in sub_ids.items():
            assert len(ts) <= 2, "Shouldn't happen"
            if len(ts) == 2:
                sub_id1, sub_id2 = ts
                graph.add_edge(sub_id1, sub_id2, label=e)

        free = {e: ids[0] for e, ids in sub_ids.items() if len(ids) == 1}
        return free

    if isinstance(tensor, Sum):
        cluster_id = f"cluster+{node_id}"
        subgraph = TikzGraph()
        free_edges = {}
        for i, (w, t) in enumerate(zip(tensor.weights, tensor.tensors)):
            subsubgraph = TikzGraph()
            subgraph_edges = _to_tikz(t, subsubgraph, depth + 1)
            handle_free_edges(subgraph_edges, subsubgraph)
            free_edges |= subgraph_edges
            if isinstance(t, Product) and count_components(t) > 1:
                style = ""
            else:
                style = ", draw=none"
            subgraph.add_subgraph(
                subsubgraph,
                f"{cluster_id}+{i}/[label={{[anchor=east, scale=2]left:${format_weight(w, i)}$}} {style}] // {tree_layout}",
                f"{cluster_id}+{i}",
            )

        graph.add_subgraph(subgraph, f"{cluster_id} / [inner sep=10pt] // {layout(depth)}", cluster_id)
        return {e: cluster_id for e in free_edges.keys()}

    assert False, "Unknown tensor type"


def format_weight(w, i):
    if w == 1:
        return "+" if i > 0 else ""
    if w == -1:
        return "-"
    if w > 0:
        return f"+{w}"
    return str(w)
