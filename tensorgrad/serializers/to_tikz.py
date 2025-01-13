from collections import Counter, defaultdict
from tensorgrad.functions import Convolution, Flatten
from tensorgrad.tensor import Derivative, Product, Zero, Copy, Variable, Sum, Function
from tensorgrad.extras import Expectation
import random
import re
from dataclasses import dataclass

# Requirements:
# !sudo apt-get install texlive-luatex
# !sudo apt-get install texlive-latex-extra
# !sudo apt-get install texlive-fonts-extra
# !sudo apt-get install poppler-utils

# TODO:
# - If two tensors are contracted over two edges (like hadamard product) the edges are drawn on top of each other.
# - Maybe we don't need a border around functions if they don't have any broadcasted edges?


prefix = """\
\\documentclass[tikz]{standalone}
\\usetikzlibrary{graphs, graphdrawing, quotes, arrows.meta, decorations.markings, shapes.geometric}
\\usegdlibrary{trees, layered, force}
\\usepackage[T1]{fontenc}
\\usepackage{comicneue}
\\begin{document}
\\tikz[
    font=\\sffamily,
    every node/.style={
        inner sep=3pt,
    },
    identity/.style={circle, draw=black, fill=black, inner sep=0pt, minimum size=4pt},
    zero/.style={rectangle, draw=black, fill=white, inner sep=2pt},
    conv/.style={rectangle, draw=black, fill=white, inner sep=2pt},
    flatten/.style={rectangle, draw=black, fill=white, inner sep=2pt},
    function/.style={circle, draw=black, fill=white, inner sep=2pt},
    var/.style={circle, draw=purple!50!black, very thick, fill=purple!20, inner sep=3pt},
    degree0/.style={circle, draw=orange!50!black, very thick, fill=orange!20, inner sep=3pt},
    degree1/.style={circle, draw=blue!50!black, very thick, fill=blue!20, inner sep=4pt},
    degree2/.style={rectangle, draw=red!50!black, very thick, fill=red!20, inner sep=6pt},
    degree3/.style={diamond, draw=green!50!black, very thick, fill=green!20, inner sep=4pt},
    degree4/.style={star, star points=5, draw=purple!50!black, very thick, fill=purple!20, inner sep=4pt},
    label/.style={scale=2, inner sep=0pt},
    subgraph nodes={draw=gray, rounded corners},
    derivative+subgraph/.style={draw=black, very thick, circle},
    expectation+subgraph/.style={draw=black, very thick, sharp corners},
    function+subgraph/.style={draw=black, thick, dashed, rounded corners},
    subgraph text none,
    every edge/.append style={
        very thick,
    },
    every edge quotes/.style={
        font=\\fontsize{5}{5.5}\\selectfont,
        fill=white,
        fill opacity=0.85,
        text opacity=1,
        midway,
        auto,
        inner sep=1pt,
    },
]
"""


def format_label(label):
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


# Global name dict, to avoid bad characters in Tikz names
name_dict: dict[str, str] = defaultdict(lambda: str(len(name_dict)))


class TikzGraph:
    def __init__(self):
        self.lines = []
        self.node_ids = set()

    def add_node(self, node_id, node_type, label=None, extra=None, degree=None):
        # print(f"adding node {node_id} of type {node_type} with label {label}")
        if isinstance(node_id, dict):
            node_id = node_id["node_id"]
        node_id = name_dict[node_id]
        assert node_id not in self.node_ids, f"Node {node_id} already exists."
        self.node_ids.add(node_id)
        if label is not None:
            label, extra_style = format_label(label)
        else:
            label, extra_style = "", ""
        nudge = f"nudge=(left:{random.random()-.5:.3f}em)"

        if node_type == "identity":
            label = f"${label}$" if label else ""
            self.lines.append(f"  {node_id}[identity,as={label},{nudge}];")
        elif node_type == "var":
            style = "var"
            if degree is not None and degree < 5:
                style = f"degree{degree}"
            if label:
                label = f"${label}$"
            self.lines.append(f"  {node_id}[{style},as={label},{nudge}];")
        elif node_type == "zero":
            self.lines.append(f"  {node_id}[zero,as=0,{nudge}];")
        elif node_type == "conv":
            self.lines.append(f"  {node_id}[conv,as=$\\ast$,{nudge}];")
        elif node_type == "flatten":
            self.lines.append(f"  {node_id}[flatten,as=flatten,{nudge}];")
        elif node_type == "function":
            label = label.replace("_", "\\_")
            label = label.replace("k=", "")
            label = label.replace("=", "")
            if label:
                label = f"${label}$"
            style = "function" if degree is None or degree >= 5 else f"degree{degree}"
            self.lines.append(f"  {node_id}[{style},as={label},style={{{extra_style}}},{nudge}];")
        elif node_type == "invisible":
            self.lines.append(f"  {node_id}[style={{}},as=,{nudge}];")
        elif node_type == "label":
            self.lines.append(f"  {node_id}[label, as=${extra}$];")
        else:
            self.lines.append(f"  {node_id}[as=${label}$,{nudge}];")

    def add_edge(self, id1, id2, label, directed=False, multiplicity=1):
        # print(f"adding edge ({id1}) -> ({id2}) with label {label}")
        style = ""
        start_text = ""
        end_text = ""
        if isinstance(id1, dict):
            style = id1.get("style", "")
            start_text = id1.get("text", "")
            id1 = id1["node_id"]
        if isinstance(id2, dict):
            style = id2.get("style", "")
            end_text = id2.get("text", "")
            id2 = id2["node_id"]
        id1 = name_dict[id1]
        id2 = name_dict[id2]
        if isinstance(label, str):
            label, _style = format_label(label)
        assert id1 in self.node_ids, f"Node {id1} does not exist in {self.node_ids}"
        assert id2 in self.node_ids, f"Node {id2} does not exist in {self.node_ids}"
        if directed:
            edge_type = " -> "
            style = "-latex"
        else:
            edge_type = " -- "

        # angle = 0, -10, 10, -20, -20 depending on multiplicity
        angle = (-1) ** multiplicity * 20 * (multiplicity // 2)
        side = "left" if multiplicity % 2 == 0 else "right"

        # Wrap in math mode, but not empty strings
        if label:
            label = f"${label}$"
        if start_text:
            start_text, _style = format_label(start_text)
            start_text = f"${start_text}$"
        else:
            start_text = label
        if end_text:
            end_text, _style = format_label(end_text)
            end_text = f"${end_text}$"
        else:
            end_text = label

        if start_text == end_text:
            # assert label == start_text
            self.lines.append(
                f'    ({id1}){edge_type}[{style}, bend left={angle}, auto={side}, "{start_text}"] ({id2});'
            )
        else:
            self.lines.append(
                f'    ({id1}){edge_type}[{style}, bend left={angle}, auto={side}, "{start_text}"  at start, "{end_text}"  at end] ({id2});'
            )

    def add_subgraph(self, subgraph, style: str, layout: str, cluster_id: str):
        """
        style or layout might be empty. Ensure we don't produce 'None'.
        """
        cluster_id = name_dict[cluster_id]
        style = style or ""  # never let it be None
        layout = layout or ""
        self.lines.append(f"{cluster_id}[{style}] // [{layout}] {{")
        self.lines += subgraph.lines
        self.lines.append("},")
        self.node_ids |= subgraph.node_ids
        self.node_ids.add(cluster_id)

    def to_tikz(self):
        return "\n".join(self.lines)


tree_layout = """\
    tree layout,
    components go down left aligned,
    fresh nodes,
    sibling sep=1em,
    node sep=1em,
    nodes behind edges,
"""

spring_layout = """\
    spring layout,
    %spring electrical layout,
    %spring electrical layout',
    %spring electrical Walshaw 2000 layout,
    fresh nodes,
    %sibling sep=3em,
    nodes behind edges,
"""


def layout(depth):
    if depth % 2 == 0:
        return tree_layout
    return tree_layout.replace("down left aligned", "right top aligned")


def to_tikz(tensor):
    """
    Main entry point: produce the LaTeX (TikZ) code for 'tensor'.
    """
    tikz_code = [prefix + f"\\graph [{layout(depth=0)}] {{"]
    graph = TikzGraph()
    free_edges = _to_tikz(tensor, graph, depth=1)
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
    return len(con.components())


def handle_free_edges(free_edges, graph):
    # print("handling free edges", free_edges)
    for e, node_id in free_edges.items():
        if isinstance(node_id, dict):
            new_node_id = node_id["node_id"] + "_" + e
        else:
            new_node_id = node_id + "_" + e
        graph.add_node(new_node_id, "invisible")
        graph.add_edge(node_id, new_node_id, label=e)


def _to_tikz(tensor, graph, depth=0):
    # We don't want the node_id to be tied to id(tensor), since, we often want
    # multiple tikz nodes for the same variable
    node_id = str(random.randrange(2**64))

    if isinstance(tensor, Copy):
        graph.add_node(node_id, "identity", label=str(tensor._size))
        return {e: node_id for e in tensor.edges}

    if isinstance(tensor, Variable):
        graph.add_node(node_id, "var", label=tensor.name, degree=len(tensor.edges))
        return {e: {"node_id": node_id, "text": orig_name} for e, orig_name in tensor.orig.items()}

    if isinstance(tensor, Zero):
        graph.add_node(node_id := str(id(tensor)), "zero", degree=len(tensor.edges))
        return {e: node_id for e in tensor.edges}

    if isinstance(tensor, Convolution):
        graph.add_node(node_id := str(id(tensor)), "conv", degree=len(tensor.edges))
        return {e: node_id for e in tensor.edges}

    if isinstance(tensor, Flatten):
        graph.add_node(node_id := str(id(tensor)), "flatten", degree=len(tensor.edges))
        return {e: node_id for e in tensor.edges}

    if isinstance(tensor, Function):
        cluster_id = f"cluster+{node_id}"
        subgraph = TikzGraph()

        subgraph.add_node(node_id, "function", label=tensor.signature.name, degree=len(tensor.edges_out))

        free_edges = {}
        for t, es in zip(tensor.inputs, tensor.signature.inputs):
            edges = _to_tikz(t, subgraph, depth + 1)
            for e in es:
                sub_id = edges.pop(e)
                subgraph.add_edge(sub_id, node_id, label=e, directed=True)
            # Add remaining edges to free edges.
            # Note: The Function should have made sure there is no edge overlap here
            assert not (edges.keys() & free_edges.keys())
            free_edges |= edges

        graph.add_subgraph(subgraph, "function+subgraph", layout(depth), cluster_id)

        # We propagate the free edges to the parent to handle
        return {e: node_id for e in tensor.edges_out} | free_edges

    if isinstance(tensor, Derivative):
        cluster_id = f"cluster+{node_id}"
        subgraph = TikzGraph()
        edges = _to_tikz(tensor.tensor, subgraph, depth + 1)
        # Add new free edges with Circle- arrow-start, similar to Penrose
        for e in tensor.new_names.values():
            edges[e] = {"node_id": cluster_id, "style": "Circle-"}
        graph.add_subgraph(subgraph, "derivative+subgraph", layout(depth), cluster_id)
        return edges

    if isinstance(tensor, Expectation):
        cluster_id = f"cluster+{node_id}"
        subgraph = TikzGraph()
        edges = _to_tikz(tensor.tensor, subgraph, depth + 1)
        graph.add_subgraph(subgraph, "expectation+subgraph", layout(depth), cluster_id)
        return edges

    if isinstance(tensor, Product):
        if len(tensor.tensors) == 0:
            graph.add_node(node_id, "identity")
            return {}

        sub_ids = defaultdict(list)  # edge -> [sub_id1, sub_id2]
        for t in tensor.tensors:
            for e, sub_id in _to_tikz(t, graph, depth + 1).items():
                sub_ids[e].append(sub_id)

        # Handle contractions (edges with multiple sub_ids)
        cnt = Counter()
        for e, ts in sub_ids.items():
            assert len(ts) <= 2, f"Too many ({len(ts)}) tensors with edge {e}"
            if len(ts) == 2:
                key = tuple(sorted(eid["node_id"] if isinstance(eid, dict) else eid for eid in ts))
                cnt[key] += 1
                sub_id1, sub_id2 = ts
                graph.add_edge(sub_id1, sub_id2, label=e, multiplicity=cnt[key])

        free = {e: ids[0] for e, ids in sub_ids.items() if len(ids) == 1}
        return free

    if isinstance(tensor, Sum):
        cluster_id = f"cluster+{node_id}"
        subgraph = TikzGraph()
        free_edges = {}
        for i, (w, t) in enumerate(zip(tensor.weights, tensor.tensors)):
            subsubgraph = TikzGraph()
            sub_id = f"{cluster_id}+{i}"
            subgraph_edges = _to_tikz(t, subsubgraph, depth + 1)
            handle_free_edges(subgraph_edges, subsubgraph)
            free_edges |= subgraph_edges
            if isinstance(t, Product) and count_components(t) > 1:
                style = ""
            else:
                style = "draw=none"
            style = "draw=none"
            if prefix := format_weight(w, i):
                label_subgraph = TikzGraph()
                label_subgraph_id = f"{sub_id}+labelsubgraph"
                label_subgraph.add_node(f"{label_subgraph_id}+label", "label", extra=prefix)
                label_subgraph.add_subgraph(subsubgraph, f"inner sep=0, {style}", layout(depth + 1), sub_id)
                subgraph.add_subgraph(
                    label_subgraph,
                    "inner sep=0, draw=none",
                    "tree layout",
                    label_subgraph_id,
                )
            else:
                subgraph.add_subgraph(subsubgraph, style, layout(depth + 1), sub_id)

        style = "draw=none" if depth == 1 else ""
        graph.add_subgraph(subgraph, f"inner sep=1em, {style}", layout(depth), cluster_id)
        return {e: cluster_id for e in free_edges.keys()}

    raise RuntimeError(f"Unknown tensor type: {type(tensor)}")


def format_weight(w, i):
    if w == 1:
        return "+" if i > 0 else ""
    if w == -1:
        return "-"
    if w > 0:
        return f"+{w}" if i > 0 else f"{w}"
    return str(w)
