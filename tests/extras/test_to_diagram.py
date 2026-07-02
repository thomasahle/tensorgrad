import json

from sympy import symbols

import tensorgrad.functions as F
from tensorgrad.tensor import Derivative, Sum, Variable
from tensorgrad.extras.diagram_browser import diagram_html
from tensorgrad.extras.diagram_layout import attach_layout, layout_diagram
from tensorgrad.extras.diagram_manim import diagram_mobject
from tensorgrad.extras.to_diagram import diagram_transition, simplify_trace, to_diagram, to_diagram_json


def test_variable_diagram_json():
    i = symbols("i")
    x = Variable("x", i)

    snapshot = to_diagram(x)

    assert snapshot.schema == "tensorgrad.diagram.v1"
    assert len(snapshot.nodes) == 1
    assert snapshot.nodes[0].kind == "variable"
    assert snapshot.nodes[0].label == "x"
    assert len(snapshot.wires) == 1
    assert snapshot.wires[0].kind == "free"
    assert snapshot.wires[0].label == "i"

    payload = json.loads(to_diagram_json(x))
    assert payload["nodes"][0]["label"] == "x"


def test_product_contraction_and_rename_endpoint_labels():
    i = symbols("i")
    a = Variable("A", i=i, j=i)
    b = Variable("B", j=i, k=i)

    snapshot = to_diagram(a @ b.rename(k="j", j="k"))
    contracted = [wire for wire in snapshot.wires if wire.kind == "contracted"]

    assert contracted
    labels = {endpoint.label for wire in contracted for endpoint in wire.endpoints}
    assert {"j", "k"} <= labels


def test_derivative_trace_has_frames_and_transitions():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    expr = Derivative(F.frobenius2(x + y), x)

    trace = simplify_trace(expr, max_steps=4)

    assert trace.schema == "tensorgrad.diagram_trace.v1"
    assert len(trace.frames) >= 2
    assert len(trace.transitions) == len(trace.frames) - 1
    assert trace.transitions[0].schema == "tensorgrad.diagram_transition.v1"
    assert trace.metadata["provenance_events"]


def test_trace_uses_provenance_for_transition_matches():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    expr = Derivative(F.frobenius2(x + y), x)

    trace = simplify_trace(expr, max_steps=4)
    source_keys = [
        node.metadata.get("source_key")
        for frame in trace.frames
        for node in frame.nodes
        if node.label == "x"
    ]

    assert any(source_keys)
    assert any(transition.matches["nodes"] for transition in trace.transitions)


def test_derivative_variable_exports_as_identity_delta():
    i = symbols("i")
    x = Variable("x", i)

    snapshot = to_diagram(Derivative(x, x, {"i": "i_"}))

    assert [node.kind for node in snapshot.nodes] == ["delta"]
    assert not snapshot.boxes
    assert {edge for wire in snapshot.wires for edge in [wire.label]} == {"i", "i_"}


def test_derivative_renamed_variable_exports_as_identity_delta():
    i = symbols("i")
    x = Variable("x", i)

    snapshot = to_diagram(Derivative(x.rename(i="j"), x, {"i": "i_"}))

    assert [node.kind for node in snapshot.nodes] == ["delta"]
    assert not snapshot.boxes
    assert {wire.label for wire in snapshot.wires} == {"j", "i_"}


def test_sum_term_port_labels_follow_exterior_edge_names():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    expr = Derivative(F.frobenius2(x + y), x)

    trace = simplify_trace(expr, max_steps=4)
    renamed_ports = [
        wire
        for frame in trace.frames
        for wire in frame.wires
        if wire.kind == "term_port" and wire.label == "i_"
    ]

    assert renamed_ports
    assert all(
        wire.label == next(endpoint.label for endpoint in wire.endpoints if endpoint.owner_kind == "box")
        for wire in renamed_ports
    )


def test_transition_matches_stable_variables():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)

    before = to_diagram(x + y)
    after = to_diagram((x + y).simplify())
    transition = diagram_transition(before, after)

    assert transition.matches["nodes"]


def test_browser_html_embeds_payload():
    i = symbols("i")
    x = Variable("x", i)

    html = diagram_html(x)

    assert "<canvas" in html
    assert "tensorgrad.diagram.v1" in html
    assert "Variable" in html or "variable" in html
    assert "tensorgrad.diagram_layout.v1" in html
    assert "wireRoutes" in html
    assert "wireLabelCandidates" in html
    assert "visibleEndpointLabels" in html
    assert '.replaceAll("_", "\'")' in html
    assert "mousedown" not in html
    assert "<button" not in html


def test_renderer_neutral_layout_has_positions_and_routes():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)

    snapshot = to_diagram(2 * x + y)
    layout = layout_diagram(snapshot).to_dict()

    assert layout["schema"] == "tensorgrad.diagram_layout.v1"
    assert set(layout["nodes"]) >= {node.id for node in snapshot.nodes}
    assert layout["boxes"]
    assert layout["wireRoutes"]
    assert layout["wireLabelCandidates"]
    assert all(len(route) >= 2 for route in layout["wireRoutes"].values())


def test_layout_term_boxes_contain_visible_term_stubs():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)

    snapshot = to_diagram(2 * x + y)
    payload = attach_layout(snapshot.to_dict())
    layout = payload["metadata"]["layout"]
    boxes = layout["boxes"]

    for wire in payload["wires"]:
        if wire["kind"] != "term_port":
            continue
        box_endpoint = next(endpoint for endpoint in wire["endpoints"] if endpoint["owner_kind"] == "box")
        rect = boxes[box_endpoint["owner"]]
        left, top, width, height = rect
        right = left + width
        bottom = top + height
        route = layout["wireRoutes"].get(wire["id"])
        if route is None:
            continue
        for x_coord, y_coord in route:
            assert left - 1 <= x_coord <= right + 1
            assert top - 1 <= y_coord <= bottom + 1


def test_delta_wire_endpoints_match_visible_delta_radius():
    i = symbols("i")
    x = Variable("x", i)

    snapshot = to_diagram(Derivative(x, x, {"i": "i_"}))
    payload = attach_layout(snapshot.to_dict())
    layout = payload["metadata"]["layout"]
    delta = snapshot.nodes[0]

    assert layout["nodeSizes"][delta.id] == [9.0, 9.0]
    center_x, center_y = layout["nodes"][delta.id]
    for route in layout["wireRoutes"].values():
        start_x, start_y = route[0]
        distance = ((start_x - center_x) ** 2 + (start_y - center_y) ** 2) ** 0.5
        assert distance <= 4.6


def test_layout_wraps_wide_sums_into_rows():
    i = symbols("i")
    terms = [Variable(f"x{k}", i) for k in range(10)]
    expr = Sum(terms, [1] * len(terms))

    payload = attach_layout(to_diagram(expr).to_dict())
    layout = payload["metadata"]["layout"]
    sum_box = next(box for box in payload["boxes"] if box["kind"] == "sum")
    _x, _y, width, height = layout["boxes"][sum_box["id"]]

    assert width < 700
    assert height > 120


def test_attach_layout_handles_traces_without_mutating_source():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    trace = simplify_trace(Derivative(F.frobenius2(x + y), x), max_steps=3).to_dict()

    payload = attach_layout(trace)

    assert "layout" not in trace["frames"][0]["metadata"]
    assert all(frame["metadata"]["layout"]["wireRoutes"] for frame in payload["frames"])


def test_manim_renderer_is_lazy_optional():
    i = symbols("i")
    x = Variable("x", i)

    try:
        diagram_mobject(to_diagram(x))
    except ImportError as exc:
        assert "requires Manim" in str(exc)
