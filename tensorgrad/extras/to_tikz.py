from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import singledispatchmethod
from functools import update_wrapper
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


def to_tikz(tensor):
    """
    Main entry point: produce the LaTeX (TikZ) code for 'tensor'.
    """
    graph = TikzGraph(Namer())
    return graph.to_tikz(tensor)


@dataclass
class NodeRef:
    """
    Holds an edge to a node in the final TikZ.

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
        self.edge_mapping = {}

    def fresh_name(self, prefix="node"):
        name = f"{prefix}{self.counter}"
        self.counter += 1
        return name

    def edge(self, edge):
        return self.edge_mapping.get(edge, edge)


class depth_tracking_dispatcher(singledispatchmethod):
    def register(self, cls, method=None):
        # Get the standard singledispatch register
        dispatcher = super().register(cls, method)

        # Create our own register that wraps the method
        def wrapped_register(method):
            original_method = dispatcher(method)

            def wrapper(self, *args, **kwargs):
                self.depth += 1
                try:
                    return original_method(self, *args, **kwargs)
                finally:
                    self.depth -= 1

            return update_wrapper(wrapper, method)

        return wrapped_register


###############################################################################
# The main TikzGraph class
###############################################################################
class TikzGraph:
    def __init__(self, namer: Namer = None, depth: int = 0):
        # We store lines of TikZ to build the final diagram.
        self.lines = []
        self.namer = Namer() if namer is None else namer

        # For outputting nice TeX
        self.depth = depth

        # Keep track of which node names we've actually added to this graph
        # so we don't add them more than once.
        self.added_node_names = set()

    def subgraph(self):
        return TikzGraph(self.namer, self.depth + 1)

    def to_tikz(self, tensor):
        # We'll wrap everything in the prefix plus a single subgraph so that
        # if we want to do fancy layering, we can:
        code = [prefix + f"\\graph [{choose_layout(depth=0)}] {{"]

        # Convert the root tensor into a set of edges
        free_edges = self._to_tikz(tensor)

        # If it's not a Sum, handle the leftover free edges now
        if not isinstance(tensor, Sum):
            self.handle_free_edges(free_edges)

        code.append("\n".join(self.lines))
        code.append("};")
        code.append("\\end{document}")

        return "\n".join(code)

    def add_node(self, name: str, node_type: str, label: str = None, degree: int = None, style=None):
        """
        Add a single node to this graph. We rely on node_ref.name for uniqueness.
        """
        assert name not in self.added_node_names
        self.added_node_names.add(name)

        label = label or ""
        style = style or ""

        # We can nudge nodes randomly if you like:
        nudge = f"nudge=(left:{random.random() - 0.5:.3f}em)"

        if node_type == "identity":
            label = format_label(label)
            label_str = f"${label}$" if label else ""
            self.lines.append(f" {name}[identity, as={{}}, {nudge}, pin=45:{label_str}];")

        elif node_type == "zero":
            self.lines.append(f"  {name}[zero, as=0,{nudge}];")

        elif node_type == "conv":
            self.lines.append(f"  {name}[conv, as=$\\ast$,{nudge}];")

        elif node_type == "reshape":
            self.lines.append(f"  {name}[reshape, as=reshape,{nudge}];")

        elif node_type == "var":
            # We optionally style the node by degree:
            style = "var"
            if degree is not None and degree < 5:
                style = f"degree{degree}"
            if label:
                label_str = f"${label}$"
            else:
                label_str = ""
            self.lines.append(f"  {name}[{style}, as={label_str},{nudge}];")

        elif node_type == "function":
            # If it is a named function, we might want a shape.
            style = "function" if (degree is None or degree >= 5) else f"degree{degree}"
            self.lines.append(f"  {name}[{style},as={label},style={{{style}}},{nudge}];")

        elif node_type == "invisible":
            self.lines.append(f"  {name}[style={{}},as=,{nudge}];")

        elif node_type == "label":
            # Additional label node
            self.lines.append(f"  {name}[label, as=${label}$];")

        else:
            # Fallback
            self.lines.append(f"  {name}[as=${label}$,{nudge}];")

    def add_edge(self, ref1: NodeRef, ref2: NodeRef, directed=False, multiplicity=1):
        """
        Add an edge between two NodeRefs. We honor any edge_style stored in
        the NodeRef. If both have styles, we prefer the second's or combine them?
        """

        # We use a list because python sets don't keep insertion order like dicts
        labels = []
        for edge_label in [ref1.edge_label, ref2.edge_label]:
            if edge_label:
                formatted_label = format_label(edge_label)
                labels.append(f"${formatted_label}$")
        if labels and labels[0] == labels[-1]:
            labels = labels[:1]

        # Combine or choose an edge style:
        style = ref1.edge_style or ref2.edge_style or ""
        if directed:
            edge_type = " -> "
            style = "-latex"
        else:
            edge_type = " -- "

        # For multiple edges between the same nodes, bend them differently
        angle = (-1) ** multiplicity * 20 * (multiplicity // 2)
        # side = "auto=left" if multiplicity % 2 == 0 else "auto=right"
        # side = "sloped, above"
        side = ""

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
            f"    ({ref1.name}){edge_type}[{style_str}, bend left={angle}, {side} {label_str}] ({ref2.name});"
        )

    def add_subgraph(self, subgraph: "TikzGraph", cluster_id: str, *, style: str = None, layout: str = None):
        """
        Insert a subgraph as a cluster. We rely on subgraph.lines (already constructed),
        and we do not re-add subgraph's nodes individually if they've already been
        added at a higher scope. We do, however, want them to appear in the subgraph block.
        """
        style = "" if style is None else style
        layout = choose_layout(self.depth) if layout is None else layout
        self.lines.append(f"{cluster_id}[{style}] // [{layout}] {{")
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
            name = self.namer.fresh_name("free")
            self.add_node(name, "invisible")
            self.add_edge(node_ref, NodeRef(name, edge_label=e))

    ###############################################################################
    # Singledispatch for each tensor type
    ###############################################################################
    @depth_tracking_dispatcher
    def _to_tikz(self, tensor):
        raise RuntimeError(f"Unknown tensor type: {type(tensor)}")

    @_to_tikz.register
    def _(self, tensor: Delta):
        # Make one node
        name = self.namer.fresh_name("copy")
        label = tensor._size.name if tensor.order == 0 else None
        self.add_node(name, "identity", label=label)
        # Return that node for every edge
        return {e: NodeRef(name) for e in tensor.edges}

    @_to_tikz.register
    def _(self, tensor: Variable):
        name = self.namer.fresh_name("var")
        self.add_node(name, "var", label=tensor.name, degree=len(tensor.edges))
        return {e: NodeRef(name, edge_label=e) for e in tensor.edges}

    @_to_tikz.register
    def _(self, tensor: Rename):
        # Build the subgraph from the original
        edges_map = self._to_tikz(tensor.tensor)
        # Now rename the keys
        renamed = {}
        for old_e, ref in edges_map.items():
            new_e = tensor.mapping.get(old_e, old_e)
            renamed[new_e] = ref
        return renamed

    @_to_tikz.register
    def _(self, tensor: Zero):
        name = self.namer.fresh_name("zero")
        self.add_node(name, "zero", degree=len(tensor.edges))
        return {e: NodeRef(name) for e in tensor.edges}

    @_to_tikz.register
    def _(self, tensor: Convolution):
        name = self.namer.fresh_name("conv")
        self.add_node(name, "conv", degree=len(tensor.edges))
        return {e: NodeRef(name) for e in tensor.edges}

    @_to_tikz.register
    def _(self, tensor: Reshape):
        name = self.namer.fresh_name("reshape")
        self.add_node(name, "reshape", degree=len(tensor.edges))
        return {e: NodeRef(name) for e in tensor.edges}

    @_to_tikz.register
    def _(self, tensor: Function):
        # We'll wrap the function node in a subgraph
        subgraph = self.subgraph()
        func_node = self.namer.fresh_name("func")
        label, style = format_function(tensor.signature.name)
        subgraph.add_node(
            func_node,
            "function",
            label=label,
            style=style,
            degree=len(tensor.shape_out),
        )

        free_edges = {}
        # For each input t, connect it to func_ref with directed edges
        for t, input_edges in zip(tensor.inputs, tensor.signature.inputs):
            subedges = subgraph._to_tikz(t)
            # connect these subedges to func_ref
            for e in input_edges:
                sub_ref = subedges.pop(e)
                subgraph.add_edge(sub_ref, NodeRef(func_node, edge_label=e), directed=True)
            # everything else remains free
            free_edges |= subedges

        # Put the subgraph inside the main graph
        cluster_id = self.namer.fresh_name("cluster_func")
        self.add_subgraph(subgraph, cluster_id, style="function+subgraph")

        # Return edges for the function's outputs
        # plus the leftover free edges from the inputs
        out_dict = {e: NodeRef(func_node) for e in tensor.shape_out}
        return {**out_dict, **free_edges}

    @_to_tikz.register
    def _(self, tensor: Derivative):
        # Subgraph for the main expression
        subgraph = self.subgraph()
        edges = subgraph._to_tikz(tensor.tensor)

        # Mark the subgraph as derivative
        cluster_id = self.namer.fresh_name("cluster_deriv")
        # For each new edge name, attach it to the cluster with style "Circle-"
        for e in tensor.new_names.values():
            edges[e] = NodeRef(cluster_id, edge_style="Circle-")

        self.add_subgraph(subgraph, cluster_id, style="derivative+subgraph")
        return edges

    @_to_tikz.register
    def _(self, tensor: Expectation):
        # Subgraph for the main expression
        subgraph = self.subgraph()
        edges = subgraph._to_tikz(tensor.tensor)

        cluster_id = self.namer.fresh_name("cluster_expec")
        self.add_subgraph(subgraph, cluster_id, style="expectation+subgraph")
        return edges

    @_to_tikz.register
    def _(self, tensor: Product):
        # If empty product, return an identity node
        if len(tensor.tensors) == 0:
            self.add_node(self.namer.fresh_name("id"), "identity", label="1")
            return {}

        # Gather sub-ids for each edge
        sub_ids = defaultdict(list)
        for t in tensor.tensors:
            t_edges = self._to_tikz(t)
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
                self.add_edge(
                    refs[0],
                    refs[1],
                    multiplicity=multiplicity,
                )

        # If an edge has only one reference, it's free
        free = {e: refs[0] for e, refs in sub_ids.items() if len(refs) == 1}
        return free

    @_to_tikz.register
    def _(self, tensor: Sum):
        # We'll represent the sum as a subgraph (unlike product, we
        # actually want to group them visually).
        subgraph = self.subgraph()
        free_edges = {}

        for i, (w, t) in enumerate(zip(tensor.weights, tensor.tensors)):
            # Each term is itself a sub-sub-graph
            term_graph = self.subgraph()

            subedges = term_graph._to_tikz(t)
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
                label_graph = self.subgraph()
                label_graph.add_node(
                    self.namer.fresh_name("label"),
                    node_type="label",
                    label=wt_prefix,
                )
                # Now add the term_graph as a subgraph of label_graph
                label_graph.add_subgraph(
                    term_graph,
                    self.namer.fresh_name("labelcluster"),
                    style=style,
                )
                # Finally add label_graph to subgraph
                subgraph.add_subgraph(
                    label_graph,
                    self.namer.fresh_name("labelcluster2"),
                    style="inner sep=0, draw=none",
                    layout="tree layout",
                )
            else:
                # Just put the term_graph in subgraph
                subgraph.add_subgraph(
                    term_graph,
                    self.namer.fresh_name("sumterm"),
                    style=style,
                )

        # If this sum is a top-level expression, we might not want a visible box
        cluster_id = self.namer.fresh_name("sumcluster")
        self.add_subgraph(
            subgraph,
            cluster_id,
            style="draw=none" if self.depth == 0 else "inner sep=1em",
        )

        # Return references for all free edges as if they connect to sum “cluster_id”
        # so that if the sum is nested in another expression, we treat this sum as if
        # it’s a single node from the outside.
        return {e: NodeRef(cluster_id, edge_label=e) for e in free_edges.keys()}


###############################################################################
# Formatting helpers
###############################################################################
def format_label(label):
    """
    Attempt to parse trailing digits and underscores to produce e.g. x_1 or x_{12}.
    Also handle "D_" for double lines, etc.
    """
    label = label.replace("_", "'")
    return label


def format_weight(w, i):
    """Just a small helper for printing weights in sums."""
    if w == 1:
        return "+" if i > 0 else ""
    if w == -1:
        return "-"
    if w > 0:
        return f"+{w}" if i > 0 else f"{w}"
    return str(w)


def format_function(label):
    """Just a small helper for printing weights in sums."""
    """
    Sanitizes a string so that it becomes safe to include directly in a LaTeX math environment.
    """

    special_chars = {
        "\\": r"\backslash",  # Key difference: use \backslash for math mode
        "{": r"\{",
        "}": r"\}",
    }
    for char, escaped in special_chars.items():
        label = label.replace(char, escaped)

    fraction_pattern = r"Fraction\s*\(\s*([\-0-9]+)\s*,\s*([\-0-9]+)\s*\)"

    def fraction_replacer(match):
        numerator = match.group(1)
        denominator = match.group(2)
        # return r"\\frac{%s}{%s}" % (numerator, denominator)
        return f"{numerator}/{denominator}"

    label = re.sub(fraction_pattern, fraction_replacer, label)

    special_chars = {
        # "_": r"\_",
        "#": r"\#",
        "k=": r"",
        "=": r"",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "~": r"\sim",  # or \approx, or \tilde{}; depends on your need
        "^": r"\hat{}",  # or just "^\," if you need an actual caret
        ",": "{,}",  # curcly brackets are another way to escape
    }
    for char, escaped in special_chars.items():
        label = label.replace(char, escaped)

    style = ""
    if re.match(r"D_(\d+)", label):
        style = "double"

    return f"${label}$", style


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


def choose_layout(depth):
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
        inner sep=1pt,
    },
    pin distance=.5ex,
    every pin/.style={font=\small\itshape}
]
"""
