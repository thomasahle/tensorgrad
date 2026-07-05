"""Renderer-neutral visual layout for tensor diagram IR.

This module intentionally sits between ``to_diagram`` and concrete renderers.
It turns semantic diagram snapshots into measured positions, boxes, ports, and
basic wire routes.  Canvas, SVG, and Manim renderers can then draw the same
layout without each backend inventing its own geometry.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import cos, hypot, pi, sin, sqrt
from typing import Any, cast

from tensorgrad.extras.to_diagram import DiagramSnapshot


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def to_list(self) -> list[float]:
        return [_round(self.x), _round(self.y)]


@dataclass(frozen=True)
class Size:
    w: float
    h: float

    def to_list(self) -> list[float]:
        return [_round(self.w), _round(self.h)]


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> Point:
        return Point(self.x + self.w / 2, self.y + self.h / 2)

    def to_list(self) -> list[float]:
        return [_round(self.x), _round(self.y), _round(self.w), _round(self.h)]


@dataclass
class MeasuredItem:
    id: str
    w: float
    h: float
    children: list["MeasuredItem"] | None = None
    gap: float = 0
    label_slot: float = 0
    rows: list[list["MeasuredItem"]] | None = None


@dataclass
class DiagramLayout:
    nodes: dict[str, Point]
    boxes: dict[str, Rect]
    node_sizes: dict[str, Size]
    node_ports: dict[str, list[str]]
    free_ports: dict[str, list[str]]
    wire_routes: dict[str, list[Point]]
    wire_label_candidates: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorgrad.diagram_layout.v1",
            "nodes": {k: v.to_list() for k, v in self.nodes.items()},
            "boxes": {k: v.to_list() for k, v in self.boxes.items()},
            "nodeSizes": {k: v.to_list() for k, v in self.node_sizes.items()},
            "nodePorts": self.node_ports,
            "freePorts": self.free_ports,
            "wireRoutes": {k: [p.to_list() for p in v] for k, v in self.wire_routes.items()},
            "wireLabelCandidates": self.wire_label_candidates,
        }


def attach_layout(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied payload with visual layouts attached to frames."""

    payload = deepcopy(payload)
    if payload.get("frames"):
        for frame in payload["frames"]:
            _attach_layout_to_frame(frame)
    else:
        _attach_layout_to_frame(payload)
    return payload


def layout_diagram(diagram: DiagramSnapshot | dict[str, Any]) -> DiagramLayout:
    """Compute a deterministic visual layout for one diagram snapshot."""

    frame = _snapshot_dict(diagram)
    nodes_by_id = {node["id"]: node for node in frame.get("nodes", [])}
    boxes_by_id = {box["id"]: box for box in frame.get("boxes", [])}
    known = set(nodes_by_id) | set(boxes_by_id)
    parent: dict[str, str] = {}
    for box in boxes_by_id.values():
        for child in box.get("children", []) or []:
            if child in known:
                parent[child] = box["id"]

    root_boxes = [box for box in boxes_by_id.values() if box["id"] not in parent]
    roots = [box["id"] for box in root_boxes] or [
        node["id"] for node in nodes_by_id.values() if node["id"] not in parent
    ]

    free_ports: dict[str, list[str]] = {}
    for wire in frame.get("wires", []) or []:
        if wire.get("kind") != "free" or not wire.get("endpoints"):
            continue
        endpoint = wire["endpoints"][0]
        free_ports.setdefault(endpoint["owner"], []).append(endpoint["port"])

    layout = DiagramLayout(
        nodes={},
        boxes={},
        node_sizes={},
        node_ports={},
        free_ports=free_ports,
        wire_routes={},
        wire_label_candidates={},
    )

    measured_roots = [_measure(root, nodes_by_id, boxes_by_id, known) for root in roots]
    total_w = sum(m.w for m in measured_roots) + 46 * max(0, len(measured_roots) - 1)
    x = -total_w / 2
    for measured in measured_roots:
        _place(measured, x + measured.w / 2, 20, layout, nodes_by_id, boxes_by_id)
        x += measured.w + 70

    layout.wire_routes = _route_wires(frame, layout)
    layout.wire_label_candidates = _wire_label_candidates(frame, layout)
    return layout


def _attach_layout_to_frame(frame: dict[str, Any]) -> None:
    metadata = frame.setdefault("metadata", {})
    metadata["layout"] = layout_diagram(frame).to_dict()


def _snapshot_dict(diagram: DiagramSnapshot | dict[str, Any]) -> dict[str, Any]:
    if isinstance(diagram, DiagramSnapshot):
        return diagram.to_dict()
    return diagram


def _measure(
    id: str,
    nodes_by_id: dict[str, dict[str, Any]],
    boxes_by_id: dict[str, dict[str, Any]],
    known: set[str],
) -> MeasuredItem:
    if node := nodes_by_id.get(id):
        if node.get("kind") == "delta":
            return MeasuredItem(id=id, w=9, h=9)
        label = str(node.get("label") or "")
        return MeasuredItem(id=id, w=max(48, 30 + len(label) * 15), h=38)

    box = boxes_by_id.get(id)
    if box is None:
        return MeasuredItem(id=id, w=0, h=0)
    children = [child for child in box.get("children", []) if child in known]
    if not children:
        return MeasuredItem(id=id, w=110, h=74, children=[])

    measured = [_measure(child, nodes_by_id, boxes_by_id, known) for child in children]
    kind = box.get("kind")
    if kind == "sum_term":
        label_slot = _term_label_slot(box.get("label") or "")
        gap = 28 if len(measured) > 1 else 0
        rows = _pack_rows(measured, gap, target=560)
        if rows is not None:
            inner_w = max(_row_width(row, gap) for row in rows)
            inner_h = _rows_height(rows)
        else:
            inner_w = sum(m.w for m in measured) + gap * max(0, len(measured) - 1)
            inner_h = max(m.h for m in measured)
        port_right = 30 if _has_visible_node_port(children, nodes_by_id) else 8
        port_top = 24 if _has_visible_node_port(children, nodes_by_id) else 8
        return MeasuredItem(
            id=id,
            w=max(54, inner_w + label_slot + port_right + (10 if label_slot else 14)),
            h=max(50, inner_h + port_top + 8),
            children=measured,
            gap=gap,
            label_slot=label_slot,
            rows=rows,
        )
    has_delta_child = any(nodes_by_id.get(child, {}).get("kind") == "delta" for child in children)
    has_box_child = any(child in boxes_by_id for child in children)
    if kind == "sum":
        gap = 20
    elif kind == "derivative":
        gap = 26
    elif has_delta_child and has_box_child:
        gap = 42
    else:
        gap = 20
    rows = _pack_rows(measured, gap) if kind == "sum" else None
    if rows is not None:
        row_widths = [_row_width(row, gap) for row in rows]
        inner_w = max(row_widths)
        inner_h = sum(max(item.h for item in row) for row in rows) + 22 * (len(rows) - 1)
    else:
        inner_w = sum(m.w for m in measured) + gap * max(0, len(measured) - 1)
        inner_h = max(m.h for m in measured)
    pad_x = 36 if kind == "derivative" else 22 if kind == "sum" else 18
    pad_y = 38 if kind == "derivative" else 24 if kind == "sum" else 16
    return MeasuredItem(
        id=id,
        w=max(80, inner_w + pad_x),
        h=max(54, inner_h + pad_y),
        children=measured,
        gap=gap,
        rows=rows,
    )


def _place(
    item: MeasuredItem,
    cx: float,
    cy: float,
    layout: DiagramLayout,
    nodes_by_id: dict[str, dict[str, Any]],
    boxes_by_id: dict[str, dict[str, Any]],
) -> None:
    if node := nodes_by_id.get(item.id):
        layout.nodes[item.id] = Point(cx, cy)
        layout.node_sizes[item.id] = Size(item.w, item.h)
        layout.node_ports[item.id] = list(node.get("ports", []) or [])
        return

    layout.nodes[item.id] = Point(cx, cy)
    layout.boxes[item.id] = Rect(cx - item.w / 2, cy - item.h / 2, item.w, item.h)
    if not item.children:
        return
    parent_kind = boxes_by_id.get(item.id, {}).get("kind")
    if item.rows is not None:
        y = cy - _rows_height(item.rows) / 2
        content_cx = cx
        if parent_kind == "sum_term" and item.label_slot:
            content_left = cx - item.w / 2 + item.label_slot + 6
            content_cx = content_left + (item.w - item.label_slot - 12) / 2
        for row in item.rows:
            row_h = max(child.h for child in row)
            x = content_cx - _row_width(row, item.gap) / 2
            for child in row:
                _place(
                    child,
                    x + child.w / 2,
                    y + row_h / 2 + 5,
                    layout,
                    nodes_by_id,
                    boxes_by_id,
                )
                x += child.w + item.gap
            y += row_h + 22
        return
    children_w = sum(child.w for child in item.children) + item.gap * (len(item.children) - 1)
    if parent_kind == "sum_term" and item.label_slot:
        x = cx - item.w / 2 + item.label_slot + 6
    else:
        x = cx - children_w / 2
    for child in item.children:
        _place(
            child,
            x + child.w / 2,
            cy + (8 if parent_kind == "derivative" else 7 if parent_kind == "sum_term" else 5),
            layout,
            nodes_by_id,
            boxes_by_id,
        )
        x += child.w + item.gap


def _route_wires(frame: dict[str, Any], layout: DiagramLayout) -> dict[str, list[Point]]:
    routes: dict[str, list[Point]] = {}
    for wire in frame.get("wires", []) or []:
        endpoints = wire.get("endpoints", []) or []
        if not endpoints:
            continue
        if wire.get("kind") == "term_port":
            node_endpoint = next(
                (e for e in endpoints if e.get("owner_kind") == "node" and e.get("owner") in layout.nodes),
                None,
            )
            if node_endpoint is None:
                continue
            node = layout.nodes[node_endpoint["owner"]]
            start = _endpoint_pos(node_endpoint, layout, Point(node.x + 36, node.y - 28))
            routes[wire["id"]] = [start, Point(start.x + 25, start.y - 25)]
            continue
        if wire.get("kind") == "derivative_passthrough":
            node_endpoint = next(
                (e for e in endpoints if e.get("owner_kind") == "node" and e.get("owner") in layout.nodes),
                None,
            )
            box_endpoint = next(
                (e for e in endpoints if e.get("owner_kind") == "box" and e.get("owner") in layout.boxes),
                None,
            )
            if node_endpoint is None or box_endpoint is None:
                continue
            rect = layout.boxes[box_endpoint["owner"]]
            end = Point(rect.x + rect.w, rect.y + rect.h / 2)
            routes[wire["id"]] = [_endpoint_pos(node_endpoint, layout, end), end]
            continue

        rough = [_owner_center(endpoint, layout) for endpoint in endpoints]
        is_free = len(endpoints) == 1
        points = []
        for i, endpoint in enumerate(endpoints):
            target = _free_target(endpoint, layout) if is_free else rough[1 if i == 0 else i - 1]
            points.append(_endpoint_pos(endpoint, layout, target))
        if len(points) == 1:
            points.append(_free_endpoint(points[0], endpoints[0], layout))
        routes[wire["id"]] = _dogleg_route(points)

    return routes


def _pack_rows(items: list[MeasuredItem], gap: float, *, target: float = 620) -> list[list[MeasuredItem]] | None:
    if len(items) <= 2:
        return None
    total_w = _row_width(items, gap)
    if total_w <= target:
        return None
    rows: list[list[MeasuredItem]] = []
    row: list[MeasuredItem] = []
    row_w = 0.0
    for item in items:
        projected = item.w if not row else row_w + gap + item.w
        if row and projected > target:
            rows.append(row)
            row = [item]
            row_w = item.w
        else:
            row.append(item)
            row_w = projected
    if row:
        rows.append(row)
    return rows if len(rows) > 1 else None


def _row_width(row: list[MeasuredItem], gap: float) -> float:
    return sum(item.w for item in row) + gap * max(0, len(row) - 1)


def _rows_height(rows: list[list[MeasuredItem]], gap_y: float = 22) -> float:
    return sum(max(item.h for item in row) for row in rows) + gap_y * max(0, len(rows) - 1)


def _dogleg_route(points: list[Point]) -> list[Point]:
    if len(points) != 2:
        return points
    start, end = points
    if abs(start.y - end.y) < 34 or abs(start.x - end.x) < 24:
        return points
    mid_x = (start.x + end.x) / 2
    return [start, Point(mid_x, start.y), Point(mid_x, end.y), end]


def _owner_center(endpoint: dict[str, Any], layout: DiagramLayout) -> Point:
    owner = endpoint["owner"]
    if owner in layout.nodes:
        return layout.nodes[owner]
    if owner in layout.boxes:
        return layout.boxes[owner].center
    return Point(0, 0)


def _endpoint_pos(endpoint: dict[str, Any], layout: DiagramLayout, toward: Point | None = None) -> Point:
    owner = endpoint["owner"]
    base = layout.nodes.get(owner, Point(0, 0))
    if owner in layout.node_sizes:
        direction = _node_port_direction(endpoint, layout, toward)
        rx = max(4, layout.node_sizes[owner].w / 2)
        ry = max(4, layout.node_sizes[owner].h / 2)
        denom = sqrt((direction.x * direction.x) / (rx * rx) + (direction.y * direction.y) / (ry * ry)) or 1
        return Point(base.x + direction.x / denom, base.y + direction.y / denom)

    rect = layout.boxes.get(owner)
    if rect is not None:
        if _is_free_port(endpoint, layout):
            return _box_free_port_position(endpoint, rect, layout)
        center = rect.center
        tx = toward.x if toward else center.x + 1
        ty = toward.y if toward else center.y
        dx = tx - center.x
        dy = ty - center.y
        if abs(dx) * rect.h > abs(dy) * rect.w:
            sx = rect.x + rect.w if dx >= 0 else rect.x
            return Point(sx, center.y + dy * ((sx - center.x) / (dx or 1)))
        sy = rect.y + rect.h if dy >= 0 else rect.y
        return Point(center.x + dx * ((sy - center.y) / (dy or 1)), sy)
    return base


def _is_free_port(endpoint: dict[str, Any], layout: DiagramLayout) -> bool:
    return endpoint["port"] in layout.free_ports.get(endpoint["owner"], [])


def _box_free_port_position(endpoint: dict[str, Any], rect: Rect, layout: DiagramLayout) -> Point:
    direction = _free_direction(endpoint, layout)
    ports = [port for port in layout.free_ports.get(endpoint["owner"], [endpoint["port"]]) if port]
    n = max(1, len(ports))
    idx = max(0, ports.index(endpoint["port"]) if endpoint["port"] in ports else 0)
    rank = idx - (n - 1) / 2
    x = rect.x + rect.w if direction.x >= 0 else rect.x
    spread = min(rect.h * 0.32, 24)
    return Point(x, rect.y + rect.h / 2 + rank * spread)


def _node_port_direction(endpoint: dict[str, Any], layout: DiagramLayout, toward: Point | None) -> Point:
    base = layout.nodes.get(endpoint["owner"], Point(0, 0))
    ports = layout.node_ports.get(endpoint["owner"], [])
    size = layout.node_sizes.get(endpoint["owner"])
    if size is not None and size.w <= 18 and len(ports) == 2:
        side = 1 if toward is None or toward.x >= base.x else -1
        port = str(endpoint.get("port") or "")
        vertical = 0.68 if port.endswith("_") else -0.68
        return Point(side, vertical)
    if toward is not None:
        dx = toward.x - base.x
        dy = toward.y - base.y
        if hypot(dx, dy) > 0.001:
            return Point(dx, dy)
    n = max(1, len(ports))
    idx = max(0, ports.index(endpoint["port"]) if endpoint["port"] in ports else 0)
    if n == 1:
        return Point(1, 0)
    angle = -pi * 0.82 + (pi * 1.64 * idx) / (n - 1)
    return Point(cos(angle), sin(angle))


def _free_endpoint(start: Point, endpoint: dict[str, Any], layout: DiagramLayout) -> Point:
    direction = _free_direction(endpoint, layout)
    length = hypot(direction.x, direction.y) or 1
    return Point(start.x + direction.x / length * 34, start.y + direction.y / length * 34)


def _free_target(endpoint: dict[str, Any], layout: DiagramLayout) -> Point:
    base = _owner_center(endpoint, layout)
    direction = _free_direction(endpoint, layout)
    return Point(base.x + direction.x, base.y + direction.y)


def _free_direction(endpoint: dict[str, Any], layout: DiagramLayout) -> Point:
    base = _owner_center(endpoint, layout)
    ports = [port for port in layout.free_ports.get(endpoint["owner"], [endpoint["port"]]) if port]
    n = max(1, len(ports))
    idx = max(0, ports.index(endpoint["port"]) if endpoint["port"] in ports else 0)
    rank = idx - (n - 1) / 2
    outward = 1 if base.x >= 0 else -1
    return Point(outward, rank * 0.72)


def _wire_label_candidates(frame: dict[str, Any], layout: DiagramLayout) -> dict[str, list[dict[str, Any]]]:
    boxes_by_id = {box["id"]: box for box in frame.get("boxes", [])}
    nodes_by_id = {node["id"]: node for node in frame.get("nodes", [])}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for wire in frame.get("wires", []) or []:
        route = layout.wire_routes.get(wire["id"], [])
        if len(route) < 2:
            continue
        if wire.get("kind") == "term_port":
            candidates[wire["id"]] = _term_boundary_label_candidates(route[0], route[-1])
            continue

        endpoints = wire.get("endpoints", []) or []
        endpoint_labels = [
            (index, endpoint.get("label"))
            for index, endpoint in enumerate(endpoints)
            if endpoint.get("label")
        ]
        unique_labels = {label for _, label in endpoint_labels}
        if len(unique_labels) >= 2:
            for index, _label in endpoint_labels:
                point = route[index]
                neighbor = route[1 if index == 0 else index - 1]
                endpoint = endpoints[index]
                key = f"{wire['id']}:{index}"
                if _is_derivative_boundary_endpoint(endpoint, boxes_by_id):
                    candidates[key] = _derivative_boundary_label_candidates(point, neighbor)
                else:
                    candidates[key] = _label_candidates_near_endpoint(point, neighbor)
            continue

        if wire.get("label"):
            derivative_index = next(
                (
                    index
                    for index, endpoint in enumerate(endpoints)
                    if _is_derivative_boundary_endpoint(endpoint, boxes_by_id)
                ),
                None,
            )
            if derivative_index is not None:
                point = route[derivative_index]
                neighbor = route[1 if derivative_index == 0 else derivative_index - 1]
                candidates[wire["id"]] = _derivative_boundary_label_candidates(point, neighbor)
            elif any(_is_delta_endpoint(endpoint, nodes_by_id) for endpoint in endpoints):
                delta_index = next(
                    index for index, endpoint in enumerate(endpoints) if _is_delta_endpoint(endpoint, nodes_by_id)
                )
                point = route[delta_index]
                neighbor = route[1 if delta_index == 0 else delta_index - 1]
                candidates[wire["id"]] = _delta_wire_label_candidates(point, neighbor)
            else:
                candidates[wire["id"]] = _label_candidates_on_wire(route)
    return candidates


def _is_derivative_boundary_endpoint(endpoint: dict[str, Any], boxes_by_id: dict[str, dict[str, Any]]) -> bool:
    # (cast: endpoints with owner_kind == "box" always carry an "owner" id)
    return endpoint.get("owner_kind") == "box" and boxes_by_id.get(cast(str, endpoint.get("owner")), {}).get("kind") == "derivative"


def _is_delta_endpoint(endpoint: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> bool:
    return endpoint.get("owner_kind") == "node" and nodes_by_id.get(cast(str, endpoint.get("owner")), {}).get("kind") == "delta"


def _candidate(point: Point, *, anchor: str = "center", penalty: float = 0) -> dict[str, Any]:
    return {"point": point.to_list(), "anchor": anchor, "penalty": _round(penalty)}


def _label_candidates_on_wire(route: list[Point]) -> list[dict[str, Any]]:
    a = route[0]
    b = route[-1]
    mid = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
    dx = b.x - a.x
    dy = b.y - a.y
    length = hypot(dx, dy) or 1
    normal = Point(-dy / length, dx / length)
    preferred = -1 if normal.y > 0 else 1
    out = []
    for i, side in enumerate([-1, 1, 0]):
        s = side or preferred
        out.append(
            _candidate(
                Point(mid.x + normal.x * 10 * s, mid.y + normal.y * 10 * s - 4),
                penalty=0 if i == 0 else i * 18,
            )
        )
    return out


def _delta_wire_label_candidates(point: Point, neighbor: Point) -> list[dict[str, Any]]:
    dx = neighbor.x - point.x
    dy = neighbor.y - point.y
    length = hypot(dx, dy) or 1
    along = Point(dx / length, dy / length)
    normal = Point(-along.y, along.x)
    base = Point(point.x + along.x * 23, point.y + along.y * 23)
    preferred = -1 if normal.y > 0 else 1
    return [
        _candidate(Point(base.x + normal.x * 18 * preferred, base.y + normal.y * 18 * preferred - 2)),
        _candidate(Point(base.x - normal.x * 18 * preferred, base.y - normal.y * 18 * preferred - 2), penalty=16),
        _candidate(Point(point.x + along.x * 28, point.y + along.y * 28 - 18), penalty=28),
    ]


def _label_candidates_near_endpoint(point: Point, neighbor: Point, along_bias: float = 1) -> list[dict[str, Any]]:
    dx = point.x - neighbor.x
    dy = point.y - neighbor.y
    length = hypot(dx, dy) or 1
    along = Point(dx / length, dy / length)
    normal = Point(-along.y, along.x)
    base = Point(point.x + along.x * 19 * along_bias, point.y + along.y * 19 * along_bias - 4)
    return [
        _candidate(Point(base.x + normal.x * 16, base.y + normal.y * 16)),
        _candidate(Point(base.x - normal.x * 16, base.y - normal.y * 16), penalty=18),
        _candidate(Point(base.x + along.x * 9, base.y + along.y * 9), penalty=28),
    ]


def _derivative_boundary_label_candidates(point: Point, neighbor: Point) -> list[dict[str, Any]]:
    dx = point.x - neighbor.x
    dy = point.y - neighbor.y
    length = hypot(dx, dy) or 1
    along = Point(dx / length, dy / length)
    normal = Point(-along.y, along.x)
    base = Point(point.x + along.x * 18, point.y + along.y * 18)
    side = 1 if point.x <= neighbor.x else -1
    return [
        _candidate(Point(point.x + side * 17, point.y - 21)),
        _candidate(Point(base.x + normal.x * 17, base.y + normal.y * 17 - 7), penalty=12),
        _candidate(Point(base.x - normal.x * 17, base.y - normal.y * 17 - 7), penalty=24),
    ]


def _term_boundary_label_candidates(start: Point, end: Point) -> list[dict[str, Any]]:
    p = _line_point(start, end, 0.88)
    q = _line_point(start, end, 0.7)
    n = _line_normal(start, end)
    return [
        _candidate(Point(end.x + 7, end.y - 15)),
        _candidate(Point(p.x + n.x * 17, p.y + n.y * 17), penalty=12),
        _candidate(Point(q.x - n.x * 15, q.y - n.y * 15), penalty=24),
    ]


def _line_point(start: Point, end: Point, t: float) -> Point:
    return Point(start.x + (end.x - start.x) * t, start.y + (end.y - start.y) * t)


def _line_normal(start: Point, end: Point) -> Point:
    dx = end.x - start.x
    dy = end.y - start.y
    length = hypot(dx, dy) or 1
    return Point(-dy / length, dx / length)


def _has_visible_node_port(children: list[str], nodes_by_id: dict[str, dict[str, Any]]) -> bool:
    return any(nodes_by_id.get(child, {}).get("ports") for child in children)


def _term_label_slot(label: str) -> float:
    if not label:
        return 0
    if label == "+":
        return 18
    if str(label).startswith("+"):
        return 32 + max(0, len(str(label)) - 2) * 8
    return 26 + max(0, len(str(label)) - 1) * 8


def _round(value: float) -> float:
    return round(float(value), 3)
