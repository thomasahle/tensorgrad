from manim import *
from tensorgrad.tensor import Product, Zero, Copy, Variable, Sum, Function
from tensorgrad.functions import frobenius2
import random
from collections import defaultdict
import networkx as nx


class TensorNetworkAnimation(Scene):
    def __init__(self, tensor, **kwargs):
        super().__init__(**kwargs)
        self.tensor = tensor

    def construct(self):
        G = nx.Graph()
        self.tensor_to_graph(self.tensor, G)

        # Extract labels from the NetworkX graph
        vertex_labels = {node: data["label"] for node, data in G.nodes(data=True)}
        edge_labels = {(u, v): data.get("edge_label", "") for u, v, data in G.edges(data=True)}

        # Create the Manim graph with labels
        graph = Graph(
            list(G.nodes),
            list(G.edges),
            layout="tree",
            root_vertex="sum",
            labels={node: label for node, label in vertex_labels.items()},
            # edge_config={(u, v): {"label": label} for (u, v), label in edge_labels.items()},
        )
        graph.scale(1.5)  # Adjust the scale as needed
        self.play(Create(graph))
        self.wait()

    def tensor_to_graph(self, tensor, G, parent=None):
        if isinstance(tensor, Copy):
            node_id = str(id(tensor))
            G.add_node(node_id, vertex_type=Dot(fill_color=WHITE), label="Copy")
            if parent is not None:
                G.add_edge(parent, node_id)
            for i, e in enumerate(tensor.edges):
                edge_id = f"{node_id}_{i}"
                G.add_node(edge_id, vertex_type=Dot(fill_color=WHITE), label=str(e))
                G.add_edge(node_id, edge_id, edge_label=str(e))

        elif isinstance(tensor, Variable):
            node_id = str(id(tensor))
            node = Rectangle(height=0.5, width=1, fill_color=WHITE)
            label = Text(tensor.name).scale(0.5).move_to(node)
            G.add_node(node_id, vertex_type=VGroup(node, label), label=tensor.name)
            if parent is not None:
                G.add_edge(parent, node_id)
            for i, e in enumerate(tensor.edges):
                edge_id = f"{node_id}_{i}"
                G.add_node(edge_id, vertex_type=Dot(fill_color=WHITE), label=str(e))
                G.add_edge(node_id, edge_id, edge_label=str(e))

        elif isinstance(tensor, Zero):
            node_id = str(id(tensor))
            node = Rectangle(height=0.5, width=1, fill_color=WHITE)
            label = Text("0").scale(0.5).move_to(node)
            G.add_node(node_id, vertex_type=VGroup(node, label), label="0")
            if parent is not None:
                G.add_edge(parent, node_id)
            for i, e in enumerate(tensor.edges):
                edge_id = f"{node_id}_{i}"
                G.add_node(edge_id, vertex_type=Dot(fill_color=WHITE), label=str(e))
                G.add_edge(node_id, edge_id, edge_label=str(e))

        elif isinstance(tensor, Function):
            node_id = str(id(tensor))
            node = Rectangle(height=0.75, width=1.5, fill_color=WHITE)
            label = Text(tensor.name).scale(0.5).move_to(node)
            G.add_node(node_id, vertex_type=VGroup(node, label), label=tensor.name)
            if parent is not None:
                G.add_edge(parent, node_id)
            for t, e in zip(tensor.tensors, tensor.edges_in):
                self.tensor_to_graph(t, G, parent=node_id)
                edge_id = f"{id(t)}_{t.edges.index(e)}"
                G.add_edge(edge_id, node_id, edge_label=str(e))
            for i, e in enumerate(tensor.edges):
                edge_id = f"{node_id}_{i}"
                G.add_node(edge_id, vertex_type=Dot(fill_color=WHITE), label=str(e))
                G.add_edge(node_id, edge_id, edge_label=str(e))

        elif isinstance(tensor, Product):
            for t in tensor.tensors:
                self.tensor_to_graph(t, G, parent=parent)
            for e in tensor.contractions:
                if len(e) == 2:
                    t1, t2 = e
                    edge_label = str(t1.edges[t1.edges.index(t2)])
                    G.add_edge(
                        f"{id(t1)}_{t1.edges.index(t2)}",
                        f"{id(t2)}_{t2.edges.index(t1)}",
                        edge_label=edge_label,
                    )

        elif isinstance(tensor, Sum):
            G.add_node("sum", vertex_type=Dot(fill_color=WHITE), label="Sum")
            if parent is not None:
                G.add_edge(parent, "sum")
            for i, (w, t) in enumerate(zip(tensor.weights, tensor.tensors)):
                node_id = f"sum_{i}"
                G.add_node(node_id, vertex_type=Dot(fill_color=WHITE), label=str(w))
                G.add_edge("sum", node_id, edge_label=str(w))
                label = Text(self.format_weight(w)).scale(0.5)
                vertex_pos = G.nodes[node_id]["vertex_type"].get_center()
                label.next_to(vertex_pos, LEFT)
                G.nodes[node_id]["vertex_type"].add(label)
                self.tensor_to_graph(t, G, parent=node_id)

    def format_weight(self, w):
        if w == 1:
            return "+"
        if w == -1:
            return "-"
        if w > 0:
            return f"+{w}"
        return str(w)


def create_tensor_network():
    X = Variable("X", ["b", "x"])
    Y = Variable("Y", ["b", "y"])
    W = Variable("W", ["x", "y"])
    F = frobenius2(W @ X - Y)
    grad = F.grad(W).simplify()
    return grad


def main3():
    tensor = create_tensor_network()
    scene = TensorNetworkAnimation(tensor)
    scene.render()
