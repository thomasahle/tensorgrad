from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import singledispatch
import random
import re

from tensorgrad.functions import Convolution, Reshape
from tensorgrad.tensor import Derivative, Product, Rename, Zero, Delta, Variable, Sum, Function
from tensorgrad.extras import Expectation


# Requirements:
# !sudo apt-get install texlive-luatex
# !sudo apt-get install texlive-latex-extra
# !sudo apt-get install texlive-fonts-extra
# !sudo apt-get install poppler-utils

# TODO:
# - If two tensors are contracted over two edges (like hadamard product) the edges are drawn on top of each other.
# - Maybe we don't need a border around functions if they don't have any broadcasted edges?


@dataclass
class NodeRef:
    """
    Holds a reference to a node in the final TikZ.

    - name:        The unique internal name used in TikZ (e.g. "node12").
    - edge_style:  A style string for the edge leading to/from this node (if any).
    - edge_label:  A label for the edge, if we need it to differ from the normal
                   free-edge name.
    """

    name: str
    edge_style: str = ""
    edge_label: str = ""


class Namer:
    def __init__(self):
        self.counter = 0

    def fresh_name(self, prefix="node"):
        name = f"{prefix}{self.counter}"
        self.counter += 1
        return name


###############################################################################
# The main TikzGraph class
###############################################################################
class TikzGraph:
    def __init__(self, namer: Namer):
        # We store lines of TikZ to build the final diagram.
        self.lines = []
        self.namer = namer

        # Keep track of which node names we've actually added to this graph
        # so we don't add them more than once.
        self.added_node_names = set()

    def add_node(self, node_ref: NodeRef, node_type: str, label: str = None, degree: int = None):
        """
        Add a single node to this graph. We rely on node_ref.name for uniqueness.
        """
        assert node_ref.name not in self.added_node_names
        self.added_node_names.add(node_ref.name)

        if label is not None:
            label, extra_style = format_label(label)
        else:
            label, extra_style = "", ""

        # We can nudge nodes randomly if you like:
        nudge = f"nudge=(left:{random.random() - 0.5:.3f}em)"

        if node_type == "identity":
            label_str = f"${label}$" if label else ""
            self.lines.append(f"  {node_ref.name}[identity, as={label_str},{nudge}];")

        elif node_type == "zero":
            self.lines.append(f"  {node_ref.name}[zero, as=0,{nudge}];")

        elif node_type == "conv":
            self.lines.append(f"  {node_ref.name}[conv, as=$\\ast$,{nudge}];")

        elif node_type == "reshape":
            self.lines.append(f"  {node_ref.name}[reshape, as=reshape,{nudge}];")

        elif node_type == "var":
            # We optionally style the node by degree:
            style = "var"
            if degree is not None and degree < 5:
                style = f"degree{degree}"
            if label:
                label_str = f"${label}$"
            else:
                label_str = ""
            self.lines.append(f"  {node_ref.name}[{style}, as={label_str},{nudge}];")

        elif node_type == "function":
            # If it is a named function, we might want a shape.
            style = "function" if (degree is None or degree >= 5) else f"degree{degree}"
            # Clean up underscores
            clean_label = label.replace("_", "\\_")
            # Some custom replacements:
            clean_label = clean_label.replace("k=", "")
            clean_label = clean_label.replace("=", "")
            if clean_label:
                clean_label = f"${clean_label}$"
            self.lines.append(f"  {node_ref.name}[{style},as={clean_label},style={{{extra_style}}},{nudge}];")

        elif node_type == "invisible":
            self.lines.append(f"  {node_ref.name}[style={{}},as=,{nudge}];")

        elif node_type == "label":
            # Additional label node
            self.lines.append(f"  {node_ref.name}[label, as=${label}$];")

        else:
            # Fallback
            self.lines.append(f"  {node_ref.name}[as=${label}$,{nudge}];")

    def add_edge(self, ref1: NodeRef, ref2: NodeRef, label: str, directed=False, multiplicity=1):
        """
        Add an edge between two NodeRefs. We honor any edge_style stored in
        the NodeRef. If both have styles, we prefer the second's or combine them?
        """
        # Extract the internal node names:
        id1 = ref1.name
        id2 = ref2.name

        labels = set()
        for edge_label in [ref1.edge_label, label, ref2.edge_label]:
            if edge_label:
                formatted_label, _extra_style = format_label(edge_label)
                labels.add(f"${formatted_label}$")

        # Combine or choose an edge style:
        style = ref1.edge_style or ref2.edge_style or ""
        if directed:
            edge_type = " -> "
            style = "-latex"
        else:
            edge_type = " -- "

        # For multiple edges between the same nodes, bend them differently
        angle = (-1) ** multiplicity * 20 * (multiplicity // 2)
        side = "left" if multiplicity % 2 == 0 else "right"

        # If there's a style, we put it in brackets after the edge operation
        style_str = style if style else ""

        labels = list(labels)
        if len(labels) == 0:
            label_str = ""
        elif len(labels) == 1:
            label_str = f', "{labels[0]}"'
        elif len(labels) >= 2:
            label_str = f', "{labels[0]}" at start, "{labels[-1]}" at end'

        self.lines.append(
            f"    ({id1}){edge_type}[{style_str}, bend left={angle}, auto={side} {label_str}] ({id2});"
        )

    def add_subgraph(self, subgraph: "TikzGraph", style: str, layout: str, cluster_id: NodeRef):
        """
        Insert a subgraph as a cluster. We rely on subgraph.lines (already constructed),
        and we do not re-add subgraph's nodes individually if they've already been
        added at a higher scope. We do, however, want them to appear in the subgraph block.
        """
        style = style or ""
        layout = layout or ""
        self.lines.append(f"{cluster_id.name}[{style}] // [{layout}] {{")
        # Insert all the subgraph lines
        self.lines += subgraph.lines
        self.lines.append("},")
        # Mark that we've effectively added all of subgraph's nodes
        self.added_node_names.update(subgraph.added_node_names)

    def handle_free_edges(self, free_edges: dict):
        """
        If an edge is only connected once, we create an 'invisible' node
        to represent that free edge in the diagram, so it’s visible as
        a dangling line.
        """
        for e, node_ref in free_edges.items():
            # Create a new node for the free edge:
            dummy = NodeRef(self.namer.fresh_name("free"), edge_style="", edge_label="")
            self.add_node(dummy, "invisible")
            # Now connect from node_ref -> dummy with the label "e":
            self.add_edge(node_ref, dummy, label=e)

    def to_tikz(self) -> str:
        return "\n".join(self.lines)


###############################################################################
# Formatting helpers
###############################################################################
def format_label(label):
    """
    Attempt to parse trailing digits and underscores to produce e.g. x_1 or x_{12}.
    Also handle "D_" for double lines, etc.
    """
    if re.match(r"D_(\d+)", label):
        return label, "double"

    label = label.replace("_", "'")
    return label, ""


def format_weight(w, i):
    """Just a small helper for printing weights in sums."""
    if w == 1:
        return "+" if i > 0 else ""
    if w == -1:
        return "-"
    if w > 0:
        return f"+{w}" if i > 0 else f"{w}"
    return str(w)


###############################################################################
# Global layout strings
###############################################################################
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
    fresh nodes,
    nodes behind edges,
"""


def layout(depth):
    """
    Switch layout or orientation by depth, if desired.
    """
    # For demonstration, we always use tree_layout; but you could vary.
    return tree_layout if depth % 2 == 0 else tree_layout.replace("down left aligned", "right top aligned")


###############################################################################
# The main entry point
###############################################################################
prefix = r"""\documentclass[tikz]{standalone}
\usetikzlibrary{graphs, graphdrawing, quotes, arrows.meta, decorations.markings, shapes.geometric}
\usegdlibrary{trees, layered, force}
\usepackage[T1]{fontenc}
\usepackage{comicneue}
\begin{document}
\tikz[
    font=\sffamily,
    every node/.style={
        inner sep=3pt,
    },
    identity/.style={circle, draw=black, fill=black, inner sep=0pt, minimum size=4pt},
    zero/.style={rectangle, draw=black, fill=white, inner sep=2pt},
    conv/.style={rectangle, draw=black, fill=white, inner sep=2pt},
    reshape/.style={rectangle, draw=black, fill=white, inner sep=2pt},
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
        font=\fontsize{5}{5.5}\selectfont,
        fill=white,
        fill opacity=0.85,
        text opacity=1,
        midway,
        auto,
        inner sep=1pt,
    },
]
"""


def to_tikz(tensor):
    """
    Main entry point: produce the LaTeX (TikZ) code for 'tensor'.
    """
    namer = Namer()
    graph = TikzGraph(namer)

    # We'll wrap everything in the prefix plus a single subgraph so that
    # if we want to do fancy layering, we can:
    code = [prefix + f"\\graph [{layout(depth=0)}] {{"]

    # Convert the root tensor into a set of edges
    free_edges = _to_tikz(tensor, graph, depth=1)

    # If it's not a Sum, handle the leftover free edges now
    if not isinstance(tensor, Sum):
        graph.handle_free_edges(free_edges)

    code.append(graph.to_tikz())
    code.append("};")
    code.append("\\end{document}")

    return "\n".join(code)


###############################################################################
# Singledispatch for each tensor type
###############################################################################
@singledispatch
def _to_tikz(tensor, graph: TikzGraph, depth=0):
    raise RuntimeError(f"Unknown tensor type: {type(tensor)}")


@_to_tikz.register
def _(tensor: Delta, graph: TikzGraph, depth=0):
    # Make one node
    node_ref = NodeRef(name=graph.namer.fresh_name("copy"))
    graph.add_node(node_ref, "identity", label=str(tensor._size))
    # Return that node for every edge
    return {e: node_ref for e in tensor.edges}


@_to_tikz.register
def _(tensor: Variable, graph: TikzGraph, depth=0):
    node_ref = NodeRef(name=graph.namer.fresh_name("var"))
    graph.add_node(node_ref, "var", label=tensor.name, degree=len(tensor.edges))
    return {e: node_ref for e in tensor.edges}


@_to_tikz.register
def _(tensor: Rename, graph: TikzGraph, depth=0):
    # Build the subgraph from the original
    edges_map = _to_tikz(tensor.tensor, graph, depth + 1)
    # Now rename the keys
    renamed = {}
    for old_e, ref in edges_map.items():
        new_e = tensor.mapping.get(old_e, old_e)
        ref.edge_label = old_e
        renamed[new_e] = ref
    return renamed


@_to_tikz.register
def _(tensor: Zero, graph: TikzGraph, depth=0):
    node_ref = NodeRef(name=graph.namer.fresh_name("zero"))
    graph.add_node(node_ref, "zero", degree=len(tensor.edges))
    return {e: node_ref for e in tensor.edges}


@_to_tikz.register
def _(tensor: Convolution, graph: TikzGraph, depth=0):
    node_ref = NodeRef(name=graph.namer.fresh_name("conv"))
    graph.add_node(node_ref, "conv", degree=len(tensor.edges))
    return {e: node_ref for e in tensor.edges}


@_to_tikz.register
def _(tensor: Reshape, graph: TikzGraph, depth=0):
    node_ref = NodeRef(name=graph.namer.fresh_name("reshape"))
    graph.add_node(node_ref, "reshape", degree=len(tensor.edges))
    return {e: node_ref for e in tensor.edges}


@_to_tikz.register
def _(tensor: Function, graph: TikzGraph, depth=0):
    # We'll wrap the function node in a subgraph
    subgraph = TikzGraph(graph.namer)

    func_ref = NodeRef(name=graph.namer.fresh_name("func"))
    subgraph.add_node(func_ref, "function", label=tensor.signature.name, degree=len(tensor.shape_out))

    free_edges = {}
    # For each input t, connect it to func_ref with directed edges
    for t, input_edges in zip(tensor.inputs, tensor.signature.inputs):
        subedges = _to_tikz(t, subgraph, depth + 1)
        # connect these subedges to func_ref
        for e in input_edges:
            sub_ref = subedges.pop(e)
            subgraph.add_edge(sub_ref, func_ref, label=e, directed=True)
        # everything else remains free
        free_edges |= subedges

    # Put the subgraph inside the main graph
    cluster_id = NodeRef(name=graph.namer.fresh_name("cluster_func"))
    graph.add_subgraph(subgraph, "function+subgraph", layout(depth), cluster_id)

    # Return edges for the function's outputs
    # plus the leftover free edges from the inputs
    out_dict = {e: func_ref for e in tensor.shape_out}
    return {**out_dict, **free_edges}


@_to_tikz.register
def _(tensor: Derivative, graph: TikzGraph, depth=0):
    # Subgraph for the main expression
    subgraph = TikzGraph(graph.namer)
    edges = _to_tikz(tensor.tensor, subgraph, depth + 1)

    # Mark the subgraph as derivative
    cluster_id = NodeRef(name=graph.namer.fresh_name("cluster_deriv"))
    # For each new edge name, attach it to the cluster with style "Circle-"
    for e in tensor.new_names.values():
        edges[e] = NodeRef(cluster_id.name, edge_style="Circle-")

    graph.add_subgraph(subgraph, "derivative+subgraph", layout(depth), cluster_id)
    return edges


@_to_tikz.register
def _(tensor: Expectation, graph: TikzGraph, depth=0):
    # Subgraph for the main expression
    subgraph = TikzGraph(graph.namer)
    edges = _to_tikz(tensor.tensor, subgraph, depth + 1)

    cluster_id = NodeRef(name=graph.namer.fresh_name("cluster_expec"))
    graph.add_subgraph(subgraph, "expectation+subgraph", layout(depth), cluster_id)
    return edges


@_to_tikz.register
def _(tensor: Product, graph: TikzGraph, depth=0):
    # If empty product, return an identity node
    if len(tensor.tensors) == 0:
        node_ref = NodeRef(name=graph.namer.fresh_name("id"))
        graph.add_node(node_ref, "identity")
        return {}

    # Gather sub-ids for each edge
    sub_ids = defaultdict(list)
    for t in tensor.tensors:
        t_edges = _to_tikz(t, graph, depth + 1)
        for e, ref in t_edges.items():
            sub_ids[e].append(ref)

    # If an edge has exactly 2 references, we connect them
    cnt = Counter()
    for e, refs in sub_ids.items():
        if len(refs) == 2:
            # We have a contraction
            # We track how many times these two nodes have been connected so far
            pair_key = tuple(sorted((refs[0].name, refs[1].name)))
            cnt[pair_key] += 1
            multiplicity = cnt[pair_key]
            graph.add_edge(refs[0], refs[1], label=e, multiplicity=multiplicity)

    # If an edge has only one reference, it's free
    free = {e: refs[0] for e, refs in sub_ids.items() if len(refs) == 1}
    return free


@_to_tikz.register
def _(tensor: Sum, graph: TikzGraph, depth=0):
    # We'll represent the sum as a subgraph (unlike product, we
    # actually want to group them visually).
    subgraph = TikzGraph(graph.namer)
    free_edges = {}

    for i, (w, t) in enumerate(zip(tensor.weights, tensor.tensors)):
        # Each term is itself a sub-sub-graph
        term_graph = TikzGraph(graph.namer)
        sub_id = NodeRef(name=graph.namer.fresh_name("sumterm"))

        subedges = _to_tikz(t, term_graph, depth + 1)
        # If the term has any free edges, handle them in the term_graph
        term_graph.handle_free_edges(subedges)
        free_edges |= subedges

        # Possibly style the bounding box if t is a Product with multiple factors
        style = ""
        # If it’s a product of 2 or more sub-tensors, we might want a border
        style = "draw=none" if not isinstance(t, Product) or len(t.components()) <= 1 else ""

        # We add an optional label for the weight
        wt_prefix = format_weight(w, i)
        if wt_prefix:
            # Put a label node in front of the sub-sub-graph
            label_graph = TikzGraph(graph.namer)
            label_id = NodeRef(name=graph.namer.fresh_name("label"))
            label_graph.add_node(label_id, "label", label=wt_prefix)

            # Now add the term_graph as a subgraph of label_graph
            label_cluster = NodeRef(name=graph.namer.fresh_name("labelcluster"))
            label_graph.add_subgraph(term_graph, style, layout(depth + 1), label_cluster)

            # Finally add label_graph to subgraph
            label_cluster2 = NodeRef(name=graph.namer.fresh_name("labelcluster2"))
            subgraph.add_subgraph(label_graph, "inner sep=0, draw=none", "tree layout", label_cluster2)
        else:
            # Just put the term_graph in subgraph
            subgraph.add_subgraph(term_graph, style, layout(depth + 1), sub_id)

    # If this sum is a top-level expression, we might not want a visible box
    style = "draw=none" if depth == 1 else "inner sep=1em"
    cluster_id = NodeRef(name=graph.namer.fresh_name("sumcluster"))
    graph.add_subgraph(subgraph, style, layout(depth), cluster_id)

    # Return references for all free edges as if they connect to sum “cluster_id”
    # so that if the sum is nested in another expression, we treat this sum as if
    # it’s a single node from the outside.
    return {e: cluster_id for e in free_edges.keys()}
