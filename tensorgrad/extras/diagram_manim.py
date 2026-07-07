# pyright: reportArgumentType=false
# (manim stubs expect Point3DLike/Vector3DLike; the plain lists passed here
#  are accepted at runtime — same stub-gap category as torch._dynamo.)
"""Optional Manim renderer for tensor diagram IR.

Manim is intentionally not a tensorgrad dependency.  Importing this module is
safe without Manim installed; calling the renderer requires Manim.
"""

from __future__ import annotations

from typing import Any

from tensorgrad.tensor import Tensor
from tensorgrad.extras.diagram_layout import attach_layout
from tensorgrad.extras.to_diagram import DiagramSnapshot, to_diagram


def diagram_mobject(diagram: Tensor | DiagramSnapshot | dict[str, Any], *, scale: float = 1.0) -> Any:
    """Create a Manim ``VGroup`` for a tensor diagram snapshot.

    The function accepts a tensorgrad expression, a ``DiagramSnapshot``, or a
    snapshot dictionary.  Positions come from ``metadata.layout`` when present;
    otherwise a deterministic circular fallback layout is used.
    """

    try:
        from manim import (  # pyright: ignore[reportMissingImports]  # optional dependency
            BLUE,
            GREEN,
            ORANGE,
            PURPLE,
            RED,
            WHITE,
            Circle,
            Dot,
            Line,
            MathTex,
            Rectangle,
            RoundedRectangle,
            Square,
            Text,
            VGroup,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("diagram_mobject requires Manim. Install `manim` to use this renderer.") from exc

    snapshot = _snapshot_dict(diagram)
    layout = _layout(snapshot)
    unit = 0.02 * scale
    group = VGroup()
    mobjects: dict[str, Any] = {}

    for node in snapshot["nodes"]:
        point = _manim_point(layout["nodes"].get(node["id"], (0, 0)), unit)
        kind = node["kind"]
        label = node["label"]
        if kind == "variable":
            shape = Circle(radius=0.28, color=PURPLE).set_fill(PURPLE, opacity=0.18)
        elif kind == "delta":
            shape = Dot(radius=0.08, color=WHITE)
        elif kind == "zero":
            shape = Square(side_length=0.38, color=RED)
        elif kind in {"convolution", "reshape"}:
            shape = Square(side_length=0.46, color=ORANGE)
        elif kind == "function":
            shape = Circle(radius=0.26, color=GREEN)
        else:
            shape = Circle(radius=0.24, color=BLUE)
        shape.move_to(point)
        text = MathTex(_tex_escape(label)).scale(0.55).move_to(point) if label else VGroup()
        node_group = VGroup(shape, text)
        mobjects[node["id"]] = node_group
        group.add(node_group)

    for box in snapshot["boxes"]:
        rect_info = layout["boxes"].get(box["id"])
        if rect_info:
            x, y, w, h = rect_info
            center = _manim_point((x + w / 2, y + h / 2), unit)
            width = max(w * unit, 0.3)
            height = max(h * unit, 0.24)
        else:
            center = _manim_point(layout["nodes"].get(box["id"], (0, 0)), unit)
            width = 1.0 * unit
            height = 0.7 * unit
        if box["kind"] in {"derivative", "function"}:
            rect = RoundedRectangle(width=width, height=height, corner_radius=0.12, color=WHITE)
        else:
            rect = Rectangle(width=width, height=height, color=WHITE)
        rect.move_to(center)
        rect.set_stroke(opacity=0.55)
        label = Text(box["label"], font_size=18).next_to(rect, direction=(0, 1, 0), buff=0.05)
        box_group = VGroup(rect, label)
        mobjects[box["id"]] = box_group
        group.add_to_back(box_group)

    for wire in snapshot["wires"]:
        points = [_manim_point(point, unit) for point in layout["wireRoutes"].get(wire["id"], [])]
        if len(points) < 2:
            continue
        for p0, p1 in zip(points, points[1:]):
            group.add_to_back(Line(p0, p1, color=WHITE).set_stroke(width=4))
        if wire.get("label"):
            mx = sum(p[0] for p in points) / len(points)
            my = sum(p[1] for p in points) / len(points)
            group.add(MathTex(_tex_escape(wire["label"])).scale(0.32).move_to((mx, my + 0.08, 0)))

    return group


def play_diagram_transform(
    scene: Any, current: Any, target: Tensor | DiagramSnapshot | dict[str, Any], **play_kwargs: Any
) -> Any:
    """Play a generic Manim transform to another diagram snapshot."""

    try:
        from manim import Transform  # pyright: ignore[reportMissingImports]  # optional dependency
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("play_diagram_transform requires Manim. Install `manim` to use it.") from exc

    target_mobject = diagram_mobject(target)
    scene.play(Transform(current, target_mobject), **play_kwargs)
    return current


def _snapshot_dict(diagram: Tensor | DiagramSnapshot | dict[str, Any]) -> dict[str, Any]:
    if isinstance(diagram, Tensor):
        return attach_layout(to_diagram(diagram).to_dict())
    if isinstance(diagram, DiagramSnapshot):
        return attach_layout(diagram.to_dict())
    return attach_layout(diagram)


def _layout(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stored = snapshot.get("metadata", {}).get("layout")
    return {
        "nodes": {k: tuple(v) for k, v in stored.get("nodes", {}).items()},
        "boxes": {k: tuple(v) for k, v in stored.get("boxes", {}).items()},
        "wireRoutes": {
            k: [tuple(point) for point in route]
            for k, route in stored.get("wireRoutes", {}).items()
        },
    }


def _manim_point(point: tuple[float, float], unit: float) -> tuple[float, float, float]:
    x, y = point
    return (x * unit, -y * unit, 0)


def _tex_escape(text: str) -> str:
    return text.replace("\\", r"\backslash ").replace("_", r"\_")
