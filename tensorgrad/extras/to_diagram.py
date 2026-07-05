"""Renderer-neutral tensor diagram export.

This module intentionally does not know about TikZ, Manim, SVG, or Canvas.  It
turns tensorgrad expressions into a small semantic IR that renderers can consume
and that animation code can match across simplification steps.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Iterable, Literal

from tensorgrad.functions import Convolution, Reshape
from tensorgrad.tensor import (
    Delta,
    Derivative,
    Function,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
    Zero,
)
from tensorgrad.extras import Expectation


EndpointOwner = Literal["node", "box"]


@dataclass(frozen=True)
class DiagramEndpoint:
    """One attachment point for a tensor edge/wire."""

    owner: str
    port: str
    owner_kind: EndpointOwner = "node"
    label: str | None = None


@dataclass
class DiagramNode:
    """A drawable tensor node."""

    id: str
    kind: str
    label: str | None
    ports: list[str]
    shape: dict[str, str | None]
    semantic_key: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagramWire:
    """A semantic tensor edge.

    Wires may have one endpoint for free edges, two endpoints for ordinary
    contractions, or more endpoints for future hyperedge renderers.
    """

    id: str
    kind: str
    endpoints: list[DiagramEndpoint]
    label: str | None
    shape: str | None
    semantic_key: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagramBox:
    """A region grouping a subdiagram, for sums, derivatives, expectations, etc."""

    id: str
    kind: str
    label: str
    children: list[str]
    ports: list[str]
    semantic_key: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagramSnapshot:
    """A complete tensor diagram snapshot."""

    schema: str
    root: str
    nodes: list[DiagramNode]
    wires: list[DiagramWire]
    boxes: list[DiagramBox]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class DiagramTransition:
    """Best-effort object-constancy information between two snapshots."""

    schema: str
    matches: dict[str, dict[str, str]]
    events: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagramTrace:
    """A sequence of snapshots and inferred transitions."""

    schema: str
    frames: list[DiagramSnapshot]
    transitions: list[DiagramTransition]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


class SimplificationProvenance:
    """Identity-based lineage tracker used by diagram traces.

    Normal tensor equality is structural, so this tracker deliberately keys by
    object identity.  The simplifier calls ``record(before, after)`` through the
    optional ``args["provenance"]`` hook, and the diagram exporter uses the
    resulting lineage IDs to keep visual objects stable across frames.
    """

    def __init__(self) -> None:
        self._source_by_id: dict[int, str] = {}
        self._parent: dict[str, str] = {}
        self._counter = 0
        self.events: list[dict[str, str]] = []

    def observe(self, tensor: Tensor) -> None:
        self.ensure(tensor)
        for child in _tensor_children(tensor):
            self.observe(child)

    def ensure(self, tensor: Tensor) -> str:
        key = id(tensor)
        if key not in self._source_by_id:
            source = f"src{self._counter}"
            self._counter += 1
            self._source_by_id[key] = source
            self._parent[source] = source
        return self._source_by_id[key]

    def source_id(self, tensor: Tensor) -> str | None:
        return self._source_by_id.get(id(tensor))

    def lineage_id(self, source: str) -> str:
        parent = self._parent.setdefault(source, source)
        if parent != source:
            self._parent[source] = self.lineage_id(parent)
        return self._parent[source]

    def record(self, before: Tensor, after: Tensor) -> None:
        before_source = self.ensure(before)
        after_source = self._source_by_id.get(id(after))
        if after_source is None:
            after_source = before_source
            self._source_by_id[id(after)] = after_source
        else:
            self._union(before_source, after_source)
        self.events.append(
            {
                "from": before_source,
                "to": after_source,
                "from_type": type(before).__name__,
                "to_type": type(after).__name__,
            }
        )

    def _union(self, a: str, b: str) -> None:
        ra = self.lineage_id(a)
        rb = self.lineage_id(b)
        if ra != rb:
            self._parent[rb] = ra


class _DiagramBuilder:
    def __init__(self, provenance: SimplificationProvenance | None = None) -> None:
        self.nodes: list[DiagramNode] = []
        self.wires: list[DiagramWire] = []
        self.boxes: list[DiagramBox] = []
        self._counter: Counter[str] = Counter()
        self._child_stack: list[list[str]] = []
        self.provenance = provenance

    def build(self, tensor: Tensor) -> DiagramSnapshot:
        free_edges = self._build(tensor, "root")
        for edge, endpoint in free_edges.items():
            self._add_wire(
                "free",
                [endpoint],
                label=edge,
                shape=_shape_name(tensor.shape.get(edge)),
                path=f"root.free.{edge}",
                metadata=self._source_metadata(tensor, f"wire:free:{edge}"),
            )
        return DiagramSnapshot(
            schema="tensorgrad.diagram.v1",
            root="root",
            nodes=self.nodes,
            wires=self.wires,
            boxes=self.boxes,
            metadata={
                "tensor_repr": repr(tensor),
                "shape": _shape_dict(tensor),
            },
        )

    def _source_metadata(self, tensor: Tensor, role: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = dict(extra or {})
        if self.provenance is None:
            return metadata
        source = self.provenance.ensure(tensor)
        lineage = self.provenance.lineage_id(source)
        metadata.update(
            {
                "source_id": source,
                "lineage_id": lineage,
                "source_key": f"{role}:{lineage}",
            }
        )
        return metadata

    def _fresh(self, prefix: str) -> str:
        i = self._counter[prefix]
        self._counter[prefix] += 1
        return f"{prefix}{i}"

    def _track_child(self, child_id: str) -> None:
        if self._child_stack:
            self._child_stack[-1].append(child_id)

    def _add_node(
        self,
        *,
        kind: str,
        label: str | None,
        ports: Iterable[str],
        shape: dict[str, Any],
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        node_id = self._fresh("node")
        ports = list(ports)
        node = DiagramNode(
            id=node_id,
            kind=kind,
            label=label,
            ports=ports,
            shape={e: _shape_name(s) for e, s in shape.items()},
            semantic_key=_semantic_key(kind, label, ports, shape),
            path=path,
            metadata=metadata or {},
        )
        self.nodes.append(node)
        self._track_child(node_id)
        return node_id

    def _add_box(
        self,
        *,
        kind: str,
        label: str,
        ports: Iterable[str],
        children: list[str],
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        box_id = self._fresh("box")
        ports = list(ports)
        box = DiagramBox(
            id=box_id,
            kind=kind,
            label=label,
            children=children,
            ports=ports,
            semantic_key=_semantic_key(f"box:{kind}", label, ports, {}),
            path=path,
            metadata=metadata or {},
        )
        self.boxes.append(box)
        self._track_child(box_id)
        return box_id

    def _add_wire(
        self,
        kind: str,
        endpoints: list[DiagramEndpoint],
        *,
        label: str | None,
        shape: str | None,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        wire_id = self._fresh("wire")
        endpoint_key = "|".join(sorted(f"{e.owner}:{e.port}:{e.label or ''}" for e in endpoints))
        wire = DiagramWire(
            id=wire_id,
            kind=kind,
            endpoints=endpoints,
            label=label,
            shape=shape,
            semantic_key=f"{kind}:{label or ''}:{shape or ''}:{endpoint_key}",
            metadata=metadata or {},
        )
        self.wires.append(wire)
        self._track_child(wire_id)
        return wire_id

    def _build(self, tensor: Tensor, path: str) -> dict[str, DiagramEndpoint]:
        if isinstance(tensor, Variable):
            return self._build_variable(tensor, path)
        if isinstance(tensor, Delta):
            return self._build_delta(tensor, path)
        if isinstance(tensor, Zero):
            return self._build_constant_node(tensor, "zero", "0", path)
        if isinstance(tensor, Convolution):
            return self._build_constant_node(tensor, "convolution", "*", path)
        if isinstance(tensor, Reshape):
            return self._build_constant_node(tensor, "reshape", "reshape", path)
        if isinstance(tensor, Rename):
            return self._build_rename(tensor, path)
        if isinstance(tensor, Product):
            return self._build_product(tensor, path)
        if isinstance(tensor, Sum):
            return self._build_sum(tensor, path)
        if isinstance(tensor, Function):
            return self._build_function(tensor, path)
        if isinstance(tensor, Derivative):
            return self._build_derivative(tensor, path)
        if isinstance(tensor, Expectation):
            return self._build_expectation(tensor, path)
        raise RuntimeError(f"Unknown tensor type for diagram export: {type(tensor)}")

    def _build_variable(self, tensor: Variable, path: str) -> dict[str, DiagramEndpoint]:
        node_id = self._add_node(
            kind="variable",
            label=tensor.name,
            ports=tensor.edges,
            shape=tensor.shape,
            path=path,
            metadata=self._source_metadata(tensor, "node:variable"),
        )
        return {
            edge: DiagramEndpoint(node_id, edge, "node", label=edge)
            for edge in tensor.edges
        }

    def _build_delta(self, tensor: Delta, path: str) -> dict[str, DiagramEndpoint]:
        label = _shape_name(tensor.size) if tensor.order == 0 else "Delta"
        node_id = self._add_node(
            kind="delta",
            label=label,
            ports=tensor.edges,
            shape=tensor.shape,
            path=path,
            metadata=self._source_metadata(tensor, "node:delta", {"size": _shape_name(tensor.size), "order": tensor.order}),
        )
        return {
            edge: DiagramEndpoint(node_id, edge, "node", label=edge)
            for edge in tensor.edges
        }

    def _build_constant_node(self, tensor: Tensor, kind: str, label: str, path: str) -> dict[str, DiagramEndpoint]:
        node_id = self._add_node(
            kind=kind,
            label=label,
            ports=tensor.edges,
            shape=tensor.shape,
            path=path,
            metadata=self._source_metadata(tensor, f"node:{kind}"),
        )
        return {
            edge: DiagramEndpoint(node_id, edge, "node", label=edge)
            for edge in tensor.edges
        }

    def _build_rename(self, tensor: Rename, path: str) -> dict[str, DiagramEndpoint]:
        endpoints = self._build(tensor.tensor, f"{path}.inner")
        return {tensor.mapping.get(edge, edge): endpoint for edge, endpoint in endpoints.items()}

    def _build_product(self, tensor: Product, path: str) -> dict[str, DiagramEndpoint]:
        grouped: dict[str, list[DiagramEndpoint]] = defaultdict(list)
        for i, factor in enumerate(tensor.factors):
            for edge, endpoint in self._build(factor, f"{path}.factor{i}").items():
                grouped[edge].append(endpoint)

        free: dict[str, DiagramEndpoint] = {}
        pair_counts: Counter[tuple[str, ...]] = Counter()
        for edge, endpoints in grouped.items():
            if len(endpoints) == 1:
                free[edge] = endpoints[0]
            else:
                key = tuple(sorted(e.owner for e in endpoints))
                pair_counts[key] += 1
                label = _wire_label(edge, endpoints)
                self._add_wire(
                    "contracted",
                    endpoints,
                    label=label,
                    shape=_shape_name(tensor.shape.get(edge)),
                    path=f"{path}.wire.{edge}",
                    metadata=self._source_metadata(
                        tensor,
                        f"wire:contracted:{edge}",
                        {"multiplicity": pair_counts[key]},
                    ),
                )
        return free

    def _build_sum(self, tensor: Sum, path: str) -> dict[str, DiagramEndpoint]:
        box_ports = list(tensor.edges)
        self._child_stack.append([])
        term_endpoints: list[dict[str, DiagramEndpoint]] = []
        for i, (weight, term) in enumerate(zip(tensor.weights, tensor.terms)):
            self._child_stack.append([])
            sub_edges = self._build(term, f"{path}.term{i}")
            term_children = self._child_stack.pop()
            term_box = self._add_box(
                kind="sum_term",
                label=_format_weight(weight, i),
                ports=sub_edges.keys(),
                children=term_children,
                path=f"{path}.term{i}.box",
                metadata=self._source_metadata(term, "box:sum_term", {"weight": str(weight), "term_index": i}),
            )
            term_endpoints.append(
                {
                    edge: DiagramEndpoint(term_box, edge, "box", label=edge)
                    for edge in sub_edges
                }
            )
            for edge, endpoint in sub_edges.items():
                self._add_wire(
                    "term_port",
                    [endpoint, DiagramEndpoint(term_box, edge, "box", label=edge)],
                    label=edge,
                    shape=_shape_name(term.shape.get(edge)),
                    path=f"{path}.term{i}.port.{edge}",
                    metadata=self._source_metadata(term, f"wire:term_port:{edge}"),
                )
        children = self._child_stack.pop()
        box_id = self._add_box(
            kind="sum",
            label="+",
            ports=box_ports,
            children=children,
            path=path,
            metadata=self._source_metadata(tensor, "box:sum", {"weights": [str(w) for w in tensor.weights]}),
        )
        for i, sub_edges in enumerate(term_endpoints):
            for edge, endpoint in sub_edges.items():
                self._add_wire(
                    "sum_port",
                    [endpoint, DiagramEndpoint(box_id, edge, "box", label=edge)],
                    label=edge,
                    shape=_shape_name(tensor.shape.get(edge)),
                    path=f"{path}.term{i}.sum_port.{edge}",
                    metadata=self._source_metadata(tensor, f"wire:sum_port:{edge}"),
                )
        return {
            edge: DiagramEndpoint(box_id, edge, "box", label=edge)
            for edge in tensor.edges
        }

    def _build_function(self, tensor: Function, path: str) -> dict[str, DiagramEndpoint]:
        node_id = self._add_node(
            kind="function",
            label=tensor.signature.name,
            ports=list(tensor.shape_out) + sorted({e for es in tensor.signature.inputs for e in es}),
            shape=tensor.shape,
            path=path,
            metadata=self._source_metadata(tensor, "node:function", {
                "input_edges": [sorted(es) for es in tensor.signature.inputs],
                "output_edges": list(tensor.shape_out),
            }),
        )
        free: dict[str, DiagramEndpoint] = {}
        for i, (inner, input_edges) in enumerate(zip(tensor.inputs, tensor.signature.inputs)):
            sub_edges = self._build(inner, f"{path}.input{i}")
            for edge in input_edges:
                endpoint = sub_edges.pop(edge)
                self._add_wire(
                    "function_input",
                    [endpoint, DiagramEndpoint(node_id, edge, "node", label=edge)],
                    label=_wire_label(edge, [endpoint]),
                    shape=_shape_name(inner.shape.get(edge)),
                    path=f"{path}.input{i}.{edge}",
                    metadata=self._source_metadata(inner, f"wire:function_input:{edge}", {"input_index": i, "directed": True}),
                )
            free.update(sub_edges)
        return {
            **{
                edge: DiagramEndpoint(node_id, edge, "node", label=edge)
                for edge in tensor.shape_out
            },
            **free,
        }

    def _build_derivative(self, tensor: Derivative, path: str) -> dict[str, DiagramEndpoint]:
        if identity_edges := _variable_derivative_identity_edges(tensor):
            return self._build_variable_derivative_identity(tensor, identity_edges, path)

        self._child_stack.append([])
        inner_edges = self._build(tensor.tensor, f"{path}.inner")
        children = self._child_stack.pop()
        box_id = self._add_box(
            kind="derivative",
            label=f"d/d{tensor.x.name}",
            ports=tensor.edges,
            children=children,
            path=path,
            metadata=self._source_metadata(tensor, "box:derivative", {
                "wrt": tensor.x.name,
                "new_names": tensor.new_names,
            }),
        )
        for edge, endpoint in inner_edges.items():
            self._add_wire(
                "derivative_passthrough",
                [endpoint, DiagramEndpoint(box_id, edge, "box", label=edge)],
                label=_wire_label(edge, [endpoint]),
                shape=_shape_name(tensor.tensor.shape.get(edge)),
                path=f"{path}.passthrough.{edge}",
                metadata=self._source_metadata(tensor, f"wire:derivative_passthrough:{edge}"),
            )
        endpoints = {
            edge: DiagramEndpoint(box_id, edge, "box", label=edge)
            for edge in tensor.tensor.edges
        }
        for old_edge, new_edge in tensor.new_names.items():
            endpoints[new_edge] = DiagramEndpoint(
                box_id,
                new_edge,
                "box",
                label=new_edge,
            )
        return endpoints

    def _build_variable_derivative_identity(
        self,
        tensor: Derivative,
        identity_edges: list[tuple[str, str, Any]],
        path: str,
    ) -> dict[str, DiagramEndpoint]:
        endpoints: dict[str, DiagramEndpoint] = {}
        for old_edge, new_edge, size in identity_edges:
            node_id = self._add_node(
                kind="delta",
                label="Delta",
                ports=[old_edge, new_edge],
                shape={old_edge: size, new_edge: size},
                path=f"{path}.identity.{old_edge}",
                metadata=self._source_metadata(
                    tensor,
                    "node:derivative_identity",
                    {"wrt": tensor.x.name, "old_edge": old_edge, "new_edge": new_edge},
                ),
            )
            endpoints[old_edge] = DiagramEndpoint(node_id, old_edge, "node", label=old_edge)
            endpoints[new_edge] = DiagramEndpoint(node_id, new_edge, "node", label=new_edge)
        return endpoints

    def _build_expectation(self, tensor: Expectation, path: str) -> dict[str, DiagramEndpoint]:
        self._child_stack.append([])
        inner_edges = self._build(tensor.tensor, f"{path}.inner")
        children = self._child_stack.pop()
        box_id = self._add_box(
            kind="expectation",
            label=f"E[{tensor.wrt.name}]",
            ports=tensor.edges,
            children=children,
            path=path,
            metadata=self._source_metadata(tensor, "box:expectation", {"wrt": tensor.wrt.name}),
        )
        for edge, endpoint in inner_edges.items():
            self._add_wire(
                "expectation_passthrough",
                [endpoint, DiagramEndpoint(box_id, edge, "box", label=edge)],
                label=_wire_label(edge, [endpoint]),
                shape=_shape_name(tensor.shape.get(edge)),
                path=f"{path}.passthrough.{edge}",
                metadata=self._source_metadata(tensor, f"wire:expectation_passthrough:{edge}"),
            )
        return {
            edge: DiagramEndpoint(box_id, edge, "box", label=edge)
            for edge in tensor.edges
        }


def to_diagram(tensor: Tensor, provenance: SimplificationProvenance | None = None) -> DiagramSnapshot:
    """Export a tensorgrad expression as renderer-neutral diagram IR."""

    return _DiagramBuilder(provenance).build(tensor)


def to_diagram_json(tensor: Tensor, **kwargs: Any) -> str:
    """Export a tensorgrad expression as JSON."""

    return to_diagram(tensor).to_json(**kwargs)


def diagram_transition(before: DiagramSnapshot, after: DiagramSnapshot) -> DiagramTransition:
    """Infer a best-effort transition between two diagram snapshots.

    This is deliberately conservative.  Future trace-aware simplification can
    add exact provenance; this function is the fallback matcher.
    """

    matches = {
        "nodes": _match_items(before.nodes, after.nodes),
        "wires": _match_items(before.wires, after.wires),
        "boxes": _match_items(before.boxes, after.boxes),
    }
    events: list[dict[str, Any]] = []
    for kind, match_key, before_items, after_items in [
        ("node", "nodes", before.nodes, after.nodes),
        ("wire", "wires", before.wires, after.wires),
        ("box", "boxes", before.boxes, after.boxes),
    ]:
        mapping = matches[match_key]
        old_matched = set(mapping)
        new_matched = set(mapping.values())
        events.extend(
            {"kind": "remove", "item_kind": kind, "id": item.id}
            for item in before_items
            if item.id not in old_matched
        )
        events.extend(
            {"kind": "create", "item_kind": kind, "id": item.id}
            for item in after_items
            if item.id not in new_matched
        )
        events.extend(
            {"kind": "match", "item_kind": kind, "from": old_id, "to": new_id}
            for old_id, new_id in mapping.items()
        )
    return DiagramTransition(
        schema="tensorgrad.diagram_transition.v1",
        matches=matches,
        events=events,
    )


def simplify_trace(
    tensor: Tensor,
    *,
    max_steps: int = 50,
    slow_grad: bool = True,
    expand: bool = True,
) -> DiagramTrace:
    """Generate diagram snapshots for a simplification sequence.

    The default uses one gradient propagation step at a time, matching the common
    pedagogical use case of animating derivatives.
    """

    provenance = SimplificationProvenance()
    provenance.observe(tensor)
    frames = [to_diagram(tensor, provenance=provenance)]
    expr = tensor
    expanded = False

    for _ in range(max_steps):
        args: dict[str, Any] = {"grad_steps": 1} if slow_grad else {}
        args["provenance"] = provenance
        if expanded:
            args["expand"] = True
        new = expr.simplify(args).simplify({"grad_steps": 0, "provenance": provenance})
        if new == expr:
            if expand and not expanded:
                expanded = True
                continue
            break
        expr = new
        provenance.observe(expr)
        frames.append(to_diagram(expr, provenance=provenance))

    transitions = [
        diagram_transition(before, after)
        for before, after in zip(frames, frames[1:])
    ]
    return DiagramTrace(
        schema="tensorgrad.diagram_trace.v1",
        frames=frames,
        transitions=transitions,
        metadata={
            "max_steps": max_steps,
            "slow_grad": slow_grad,
            "expand": expand,
            "provenance_events": provenance.events,
        },
    )


def _match_items(before: Iterable[Any], after: Iterable[Any]) -> dict[str, str]:
    before = list(before)
    after = list(after)
    matches = _match_items_by_source(before, after)
    old_matched = set(matches)
    new_matched = set(matches.values())

    before_by_key: dict[str, list[Any]] = defaultdict(list)
    after_by_key: dict[str, list[Any]] = defaultdict(list)
    for item in before:
        if item.id not in old_matched:
            before_by_key[item.semantic_key].append(item)
    for item in after:
        if item.id not in new_matched:
            after_by_key[item.semantic_key].append(item)

    for key, old_items in before_by_key.items():
        new_items = after_by_key.get(key, [])
        if len(old_items) == 1 and len(new_items) == 1:
            matches[old_items[0].id] = new_items[0].id
            continue
        for old_item, new_item in zip(old_items, new_items):
            matches[old_item.id] = new_item.id
    return matches


def _match_items_by_source(before: list[Any], after: list[Any]) -> dict[str, str]:
    before_by_key: dict[str, list[Any]] = defaultdict(list)
    after_by_key: dict[str, list[Any]] = defaultdict(list)
    for item in before:
        if key := item.metadata.get("source_key"):
            before_by_key[key].append(item)
    for item in after:
        if key := item.metadata.get("source_key"):
            after_by_key[key].append(item)

    matches: dict[str, str] = {}
    for key, old_items in before_by_key.items():
        new_items = after_by_key.get(key, [])
        old_items = sorted(old_items, key=lambda item: (item.semantic_key, getattr(item, "path", "")))
        new_items = sorted(new_items, key=lambda item: (item.semantic_key, getattr(item, "path", "")))
        for old_item, new_item in zip(old_items, new_items):
            matches[old_item.id] = new_item.id
    return matches


def _tensor_children(tensor: Tensor) -> list[Tensor]:
    if isinstance(tensor, Rename):
        return [tensor.tensor]
    if isinstance(tensor, Product):
        return list(tensor.factors)
    if isinstance(tensor, Sum):
        return list(tensor.terms)
    if isinstance(tensor, Function):
        return list(tensor.inputs)
    if isinstance(tensor, Derivative):
        return [tensor.tensor, tensor.x]
    if isinstance(tensor, Expectation):
        return [tensor.tensor, tensor.wrt]
    return []


def _variable_derivative_identity_edges(tensor: Derivative) -> list[tuple[str, str, Any]]:
    inner = tensor.tensor
    edge_map = {edge: edge for edge in tensor.x.edges}
    if isinstance(inner, Rename):
        edge_map = {edge: inner.mapping.get(edge, edge) for edge in tensor.x.edges}
        inner = inner.tensor
    if not isinstance(inner, Variable) or inner != tensor.x:
        return []
    return [
        (edge_map[old_edge], new_edge, tensor.x.shape[old_edge])
        for old_edge, new_edge in tensor.new_names.items()
    ]


def _shape_name(size: Any) -> str | None:
    if size is None:
        return None
    return getattr(size, "name", str(size))


def _shape_dict(tensor: Tensor) -> dict[str, str]:
    return {e: _shape_name(s) or "" for e, s in tensor.shape.items()}


def _semantic_key(kind: str, label: str | None, ports: Iterable[str], shape: dict[str, Any]) -> str:
    shape_part = ",".join(f"{e}:{_shape_name(s)}" for e, s in sorted(shape.items()))
    port_part = ",".join(sorted(ports))
    return f"{kind}|{label}|{port_part}|{shape_part}"


def _wire_label(edge: str, endpoints: list[DiagramEndpoint]) -> str | None:
    labels = {endpoint.label for endpoint in endpoints if endpoint.label}
    if len(labels) == 1:
        return next(iter(labels))
    if not labels:
        return edge
    return None


def _format_weight(weight: Any, index: int) -> str:  # weight: numeric scalar (Number has no static operators)
    if weight == 1:
        return "+" if index > 0 else ""
    if weight == -1:
        return "-"
    if weight > 0:
        return f"+{weight}" if index > 0 else str(weight)
    return str(weight)
