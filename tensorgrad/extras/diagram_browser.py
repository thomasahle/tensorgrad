"""Self-contained browser viewer for tensor diagram IR."""

from __future__ import annotations

import html
import json
from typing import Any

from tensorgrad.tensor import Tensor
from tensorgrad.extras.diagram_layout import attach_layout
from tensorgrad.extras.to_diagram import DiagramSnapshot, DiagramTrace, to_diagram


def diagram_html(diagram: Tensor | DiagramSnapshot | DiagramTrace | dict[str, Any], *, title: str = "Tensor Diagram") -> str:
    """Return a self-contained HTML viewer for a diagram or trace."""

    payload = _payload(diagram)
    return _HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__PAYLOAD__",
        json.dumps(payload).replace("</", "<\\/"),
    )


def save_diagram_html(
    diagram: Tensor | DiagramSnapshot | DiagramTrace | dict[str, Any],
    path: str,
    *,
    title: str = "Tensor Diagram",
) -> None:
    """Write a self-contained HTML viewer to ``path``."""

    with open(path, "w", encoding="utf-8") as file:
        file.write(diagram_html(diagram, title=title))


def _payload(diagram: Tensor | DiagramSnapshot | DiagramTrace | dict[str, Any]) -> dict[str, Any]:
    if isinstance(diagram, Tensor):
        return attach_layout(to_diagram(diagram).to_dict())
    if isinstance(diagram, (DiagramSnapshot, DiagramTrace)):
        return attach_layout(diagram.to_dict())
    return attach_layout(diagram)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #fff; color: #111; font: 13px system-ui, sans-serif; }
    canvas { display: block; width: 100vw; height: 100vh; }
    #labelLayer { position: fixed; inset: 0; pointer-events: none; z-index: 1; }
    .math-label { position: absolute; color: #000; transform: translate(-50%, -50%); white-space: nowrap; line-height: 1; }
    .math-label .katex { font-size: 1.08em; }
    .node-label .katex { font-size: 1.24em; }
    .wire-label { background: rgba(255,255,255,.72); border-radius: 2px; padding: 0 1px; }
    .wire-label .katex { font-size: .8em; }
    .box-label .katex { font-size: 1em; }
    .derivative-label .katex { font-size: 1.2em; }
    .term-label .katex { font-size: 1em; }
    .coef-label .katex { font-size: 1.62em; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div id="labelLayer"></div>
  <script>
const payload = __PAYLOAD__;
const frames = payload.frames || [payload];
const transitions = payload.transitions || [];
let frameIndex = initialFrameIndex();
let transition = null;
let autoFitDone = false;
let placedLabelRects = [];

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const labelLayer = document.getElementById("labelLayer");
let view = {x: 0, y: 0, scale: 1};
let layouts = frames.map(makeLayout);
const STYLE = {
  edge: "#090909",
  edgeWidth: 1.75,
  nodeWidth: 1.45,
  boxWidth: 1.75,
  dotRadius: 4.2
};

function initialFrameIndex() {
  const params = new URLSearchParams(window.location.search);
  const raw = params.get("frame") || (window.location.hash || "").replace(/^#frame=?/, "");
  const index = Number.parseInt(raw || "1", 10) - 1;
  return Math.max(0, Math.min((payload.frames || [payload]).length - 1, Number.isFinite(index) ? index : 0));
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!autoFitDone) {
    autoFit();
    autoFitDone = true;
  }
  draw();
}
window.addEventListener("resize", resize);

function makeLayout(frame) {
  const stored = frame.metadata && frame.metadata.layout;
  if (!stored || !stored.nodes) {
    return {nodes: {}, boxes: {}, nodeSizes: {}, nodePorts: {}, freePorts: {}, wireRoutes: {}};
  }
  return {
    nodes: Object.fromEntries(Object.entries(stored.nodes || {}).map(([k, v]) => [k, {x: v[0], y: v[1]}])),
    boxes: Object.fromEntries(Object.entries(stored.boxes || {}).map(([k, v]) => [k, {x: v[0], y: v[1], w: v[2], h: v[3]}])),
    nodeSizes: Object.fromEntries(Object.entries(stored.nodeSizes || {}).map(([k, v]) => [k, {w: v[0], h: v[1]}])),
    nodePorts: Object.fromEntries(Object.entries(stored.nodePorts || {}).map(([k, v]) => [k, [...v]])),
    freePorts: Object.fromEntries(Object.entries(stored.freePorts || {}).map(([k, v]) => [k, [...v]])),
    wireRoutes: Object.fromEntries(Object.entries(stored.wireRoutes || {}).map(([k, route]) => [k, route.map(p => ({x: p[0], y: p[1]}))])),
    wireLabelCandidates: Object.fromEntries(
      Object.entries(stored.wireLabelCandidates || {}).map(([k, candidates]) => [
        k,
        candidates.map(candidate => ({
          ...candidate,
          point: {x: candidate.point[0], y: candidate.point[1]}
        }))
      ])
    )
  };
}

function autoFit() {
  const layout = layouts[frameIndex] || layouts[0];
  if (!layout) return;
  const boxes = Object.values(layout.boxes || {});
  const nodeEntries = Object.entries(layout.nodes || {});
  const rects = boxes.concat(nodeEntries.map(([id, p]) => {
    const s = (layout.nodeSizes && layout.nodeSizes[id]) || {w: 38, h: 30};
    return {x: p.x - s.w / 2, y: p.y - s.h / 2, w: s.w, h: s.h};
  }));
  if (!rects.length) return;
  const minX = Math.min(...rects.map(r => r.x)) - 56;
  const maxX = Math.max(...rects.map(r => r.x + r.w)) + 68;
  const minY = Math.min(...rects.map(r => r.y)) - 44;
  const maxY = Math.max(...rects.map(r => r.y + r.h)) + 38;
  const w = maxX - minX, h = maxY - minY;
  view.scale = Math.min(2.25, Math.max(0.7, Math.min((canvas.clientWidth - 80) / w, (canvas.clientHeight - 80) / h)));
  view.x = -((minX + maxX) / 2) * view.scale;
  view.y = -((minY + maxY) / 2) * view.scale;
}

function screen(p) {
  return [canvas.clientWidth / 2 + view.x + p.x * view.scale, canvas.clientHeight / 2 + view.y + p.y * view.scale];
}

function screenPoint(p) {
  const [x, y] = screen(p);
  return {x, y};
}

function addMathLabel(tex, p, className, alpha = 1, anchor = "center") {
  addMathLabelCandidates(tex, [{point: p, anchor}], className, alpha);
}

function addMathLabelCandidates(tex, candidates, className, alpha = 1) {
  if (alpha <= 0.001 || !tex) return;
  const candidate = chooseLabelCandidate(tex, candidates, className);
  const s = screenPoint(candidate.point);
  const el = document.createElement("div");
  el.className = `math-label ${className || ""}`;
  el.style.left = `${s.x}px`;
  el.style.top = `${s.y}px`;
  el.style.opacity = String(alpha);
  el.style.fontSize = `${Math.max(0.85, view.scale).toFixed(3)}em`;
  const anchor = candidate.anchor || "center";
  if (anchor === "left") el.style.transform = "translate(0, -50%)";
  if (anchor === "right") el.style.transform = "translate(-100%, -50%)";
  if (anchor === "top-left") el.style.transform = "translate(0, 0)";
  if (anchor === "above") el.style.transform = "translate(-50%, -100%)";
  if (anchor === "below") el.style.transform = "translate(-50%, 0)";
  if (window.katex) {
    window.katex.render(tex, el, {throwOnError: false, displayMode: false});
  } else {
    el.textContent = tex.replaceAll("\\", "");
  }
  labelLayer.appendChild(el);
  placedLabelRects.push(expandRect(el.getBoundingClientRect(), 2));
}

function chooseLabelCandidate(tex, candidates, className) {
  let best = candidates[0];
  let bestScore = Infinity;
  for (const candidate of candidates) {
    const rect = estimateLabelRect(tex, className, candidate.point, candidate.anchor || "center");
    const score = (candidate.penalty || 0) + labelCollisionPenalty(rect);
    if (score < bestScore) {
      best = candidate;
      bestScore = score;
    }
  }
  return best;
}

function estimateLabelRect(tex, className, point, anchor) {
  const s = screenPoint(point);
  const clean = String(tex).replace(/\\[a-zA-Z]+|[{}\\]/g, "");
  const scale = Math.max(0.85, view.scale);
  let multiplier = 1.08;
  if ((className || "").includes("wire-label")) multiplier = 0.8;
  if ((className || "").includes("node-label")) multiplier = 1.24;
  if ((className || "").includes("coef-label")) multiplier = 1.85;
  if ((className || "").includes("derivative-label")) multiplier = 1.2;
  const font = 13 * scale * multiplier;
  const w = Math.max(12, clean.length * font * 0.48 + 6);
  const h = font * 1.25 + 4;
  return anchoredRect(s.x, s.y, w, h, anchor || "center");
}

function anchoredRect(x, y, w, h, anchor) {
  if (anchor === "left") return {left: x, top: y - h / 2, right: x + w, bottom: y + h / 2};
  if (anchor === "right") return {left: x - w, top: y - h / 2, right: x, bottom: y + h / 2};
  if (anchor === "top-left") return {left: x, top: y, right: x + w, bottom: y + h};
  if (anchor === "above") return {left: x - w / 2, top: y - h, right: x + w / 2, bottom: y};
  if (anchor === "below") return {left: x - w / 2, top: y, right: x + w / 2, bottom: y + h};
  return {left: x - w / 2, top: y - h / 2, right: x + w / 2, bottom: y + h / 2};
}

function labelCollisionPenalty(rect) {
  let score = 0;
  for (const other of placedLabelRects) score += overlapArea(rect, other) * 6;
  if (rect.left < 6) score += (6 - rect.left) * 3;
  if (rect.top < 46) score += (46 - rect.top) * 3;
  if (rect.right > canvas.clientWidth - 6) score += (rect.right - canvas.clientWidth + 6) * 3;
  if (rect.bottom > canvas.clientHeight - 6) score += (rect.bottom - canvas.clientHeight + 6) * 3;
  return score;
}

function overlapArea(a, b) {
  const x = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
  const y = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
  return x * y;
}

function expandRect(rect, pad) {
  return {left: rect.left - pad, top: rect.top - pad, right: rect.right + pad, bottom: rect.bottom + pad};
}

function draw() {
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  labelLayer.textContent = "";
  placedLabelRects = [];
  ctx.save();
  ctx.translate(canvas.clientWidth / 2 + view.x, canvas.clientHeight / 2 + view.y);
  ctx.scale(view.scale, view.scale);

  if (transition) {
    const raw = (performance.now() - transition.start) / transition.duration;
    const t = Math.min(1, Math.max(0, ease(raw)));
    drawTransition(transition.from, transition.to, t);
    if (raw >= 1) {
      frameIndex = transition.to;
      transition = null;
    } else {
      requestAnimationFrame(draw);
    }
  } else {
    const frame = frames[frameIndex];
    const layout = layouts[frameIndex];
    drawScene(frame, layout, () => 1);
  }
  ctx.restore();
}

function drawScene(frame, layout, alphaFor) {
  reserveDiagramGeometry(layout);
  for (const box of frame.boxes) drawBox(box, layout, alphaFor("box", box.id));
  for (const wire of frame.wires) drawWire(wire, layout, alphaFor("wire", wire.id), frame);
  for (const node of frame.nodes) drawNode(node, layout, alphaFor("node", node.id));
}

function reserveDiagramGeometry(layout) {
  for (const [id, p] of Object.entries(layout.nodes || {})) {
    const size = (layout.nodeSizes && layout.nodeSizes[id]) || {w: 34, h: 30};
    const a = screenPoint({x: p.x - size.w / 2 - 4, y: p.y - size.h / 2 - 4});
    const b = screenPoint({x: p.x + size.w / 2 + 4, y: p.y + size.h / 2 + 4});
    placedLabelRects.push({
      left: Math.min(a.x, b.x),
      top: Math.min(a.y, b.y),
      right: Math.max(a.x, b.x),
      bottom: Math.max(a.y, b.y)
    });
  }
}

function drawTransition(fromIndex, toIndex, t) {
  const baseTransition = transitions[Math.min(fromIndex, toIndex)] || {matches: {nodes: {}, wires: {}, boxes: {}}};
  const matches = fromIndex < toIndex ? (baseTransition.matches || {}) : invertMatches(baseTransition.matches || {});
  const tr = {matches};
  const fromFrame = frames[fromIndex], toFrame = frames[toIndex];
  const fromLayout = layouts[fromIndex], toLayout = layouts[toIndex];
  const interp = interpolateLayout(fromLayout, toLayout, tr.matches || {}, t);
  const matchedOld = {
    node: new Set(Object.keys((tr.matches && tr.matches.nodes) || {})),
    wire: new Set(Object.keys((tr.matches && tr.matches.wires) || {})),
    box: new Set(Object.keys((tr.matches && tr.matches.boxes) || {}))
  };
  const matchedNew = {
    node: new Set(Object.values((tr.matches && tr.matches.nodes) || {})),
    wire: new Set(Object.values((tr.matches && tr.matches.wires) || {})),
    box: new Set(Object.values((tr.matches && tr.matches.boxes) || {}))
  };
  drawScene(fromFrame, fromLayout, (kind, id) => matchedOld[kind].has(id) ? 0 : 1 - t);
  drawScene(toFrame, interp, (kind, id) => matchedNew[kind].has(id) ? 1 : t);
}

function invertMatches(matches) {
  const invert = obj => Object.fromEntries(Object.entries(obj || {}).map(([k, v]) => [v, k]));
  return {
    nodes: invert(matches.nodes),
    wires: invert(matches.wires),
    boxes: invert(matches.boxes)
  };
}

function interpolateLayout(fromLayout, toLayout, matches, t) {
  const out = {
    nodes: Object.fromEntries(Object.entries(toLayout.nodes).map(([id, p]) => [id, {...p}])),
    boxes: Object.fromEntries(Object.entries(toLayout.boxes).map(([id, r]) => [id, {...r}])),
    nodeSizes: Object.fromEntries(Object.entries(toLayout.nodeSizes || {}).map(([id, r]) => [id, {...r}])),
    nodePorts: Object.fromEntries(Object.entries(toLayout.nodePorts || {}).map(([id, ports]) => [id, [...ports]])),
    freePorts: Object.fromEntries(Object.entries(toLayout.freePorts || {}).map(([id, ports]) => [id, [...ports]])),
    wireRoutes: Object.fromEntries(Object.entries(toLayout.wireRoutes || {}).map(([id, route]) => [id, route.map(p => ({...p}))])),
    wireLabelCandidates: Object.fromEntries(Object.entries(toLayout.wireLabelCandidates || {}).map(([id, candidates]) => [id, candidates.map(c => ({...c, point: {...c.point}}))]))
  };
  for (const [oldId, newId] of Object.entries(matches.nodes || {})) {
    const a = fromLayout.nodes[oldId], b = toLayout.nodes[newId];
    if (a && b) out.nodes[newId] = {x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t)};
  }
  for (const [oldId, newId] of Object.entries(matches.boxes || {})) {
    const a = fromLayout.boxes[oldId], b = toLayout.boxes[newId];
    if (a && b) out.boxes[newId] = {
      x: lerp(a.x, b.x, t),
      y: lerp(a.y, b.y, t),
      w: lerp(a.w, b.w, t),
      h: lerp(a.h, b.h, t)
    };
  }
  for (const [oldId, newId] of Object.entries(matches.wires || {})) {
    const a = fromLayout.wireRoutes && fromLayout.wireRoutes[oldId];
    const b = toLayout.wireRoutes && toLayout.wireRoutes[newId];
    if (a && b && a.length === b.length) {
      out.wireRoutes[newId] = b.map((p, i) => ({x: lerp(a[i].x, p.x, t), y: lerp(a[i].y, p.y, t)}));
    }
  }
  for (const [oldId, newId] of Object.entries(matches.wires || {})) {
    const a = fromLayout.wireLabelCandidates && fromLayout.wireLabelCandidates[oldId];
    const b = toLayout.wireLabelCandidates && toLayout.wireLabelCandidates[newId];
    if (a && b && a.length === b.length) {
      out.wireLabelCandidates[newId] = b.map((candidate, i) => ({
        ...candidate,
        point: {
          x: lerp(a[i].point.x, candidate.point.x, t),
          y: lerp(a[i].point.y, candidate.point.y, t)
        }
      }));
    }
  }
  return out;
}

function lerp(a, b, t) { return a + (b - a) * t; }
function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

function animateTo(nextIndex) {
  nextIndex = Math.max(0, Math.min(frames.length - 1, nextIndex));
  if (nextIndex === frameIndex) return;
  transition = {
    from: frameIndex,
    to: nextIndex,
    start: performance.now(),
    duration: 850
  };
  requestAnimationFrame(draw);
}

function withAlpha(alpha, fn) {
  if (alpha <= 0.001) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  fn();
  ctx.restore();
}

function applyEdgeStyle(width = STYLE.edgeWidth) {
  ctx.strokeStyle = STYLE.edge;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function drawWire(wire, layout, alpha = 1, frame = null) {
  if (wire.kind === "term_port") return drawTermStub(wire, layout, alpha);
  if (wire.kind === "derivative_passthrough") return drawDerivativePassthrough(wire, layout, alpha);
  if (wire.kind === "sum_port" || wire.kind.endsWith("_passthrough")) return;
  withAlpha(alpha, () => {
  const pts = routeForWire(wire, layout);
  if (pts.length < 2) return;
  applyEdgeStyle();
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) drawCurveSegment(pts[i - 1], pts[i], wire);
  ctx.stroke();
  drawBoundaryDots(wire, pts, frame);
  const endpointLabels = visibleEndpointLabels(wire);
  if (endpointLabels.length >= 2) {
    for (const item of endpointLabels) {
      const p = pts[item.index];
      const neighbor = pts[item.index === 0 ? 1 : item.index - 1];
      const candidates = isDerivativeBoundaryEndpoint(wire.endpoints[item.index], frame)
        ? labelCandidatesForWire(layout, `${wire.id}:${item.index}`, derivativeBoundaryLabelCandidates(p, neighbor))
        : labelCandidatesForWire(layout, `${wire.id}:${item.index}`, labelCandidatesNearEndpoint(p, neighbor));
      addMathLabelCandidates(texForLabel(item.label), candidates, "wire-label", alpha);
    }
  } else if (wire.label) {
    const derivativeIndex = wire.endpoints.findIndex(e => isDerivativeBoundaryEndpoint(e, frame));
    if (derivativeIndex >= 0) {
      const p = pts[derivativeIndex];
      const neighbor = pts[derivativeIndex === 0 ? 1 : derivativeIndex - 1];
      addMathLabelCandidates(texForLabel(wire.label), labelCandidatesForWire(layout, wire.id, derivativeBoundaryLabelCandidates(p, neighbor)), "wire-label", alpha);
    } else {
      addMathLabelCandidates(texForLabel(wire.label), labelCandidatesForWire(layout, wire.id, labelCandidatesOnWire(pts)), "wire-label", alpha);
    }
  }
  });
}

function routeForWire(wire, layout) {
  if (layout.wireRoutes && layout.wireRoutes[wire.id]) return layout.wireRoutes[wire.id];
  return [];
}

function labelCandidatesForWire(layout, key, fallback) {
  return (layout.wireLabelCandidates && layout.wireLabelCandidates[key]) || fallback;
}

function isDerivativeBoundaryEndpoint(endpoint, frame) {
  if (!frame || endpoint.owner_kind !== "box") return false;
  const box = frame.boxes.find(b => b.id === endpoint.owner);
  return !!box && box.kind === "derivative";
}

function derivativeBoundaryLabelCandidates(point, neighbor) {
  const dx = point.x - neighbor.x, dy = point.y - neighbor.y;
  const len = Math.hypot(dx, dy) || 1;
  const along = {x: dx / len, y: dy / len};
  const normal = {x: -along.y, y: along.x};
  const base = {x: point.x + along.x * 18, y: point.y + along.y * 18};
  const side = point.x <= neighbor.x ? 1 : -1;
  return [
    {point: {x: point.x + side * 17, y: point.y - 21}, anchor: "center", penalty: 0},
    {point: {x: base.x + normal.x * 17, y: base.y + normal.y * 17 - 7}, anchor: "center", penalty: 12},
    {point: {x: base.x - normal.x * 17, y: base.y - normal.y * 17 - 7}, anchor: "center", penalty: 24}
  ];
}

function drawDerivativePassthrough(wire, layout, alpha = 1) {
  const nodeEndpoint = wire.endpoints.find(e => e.owner_kind === "node" && layout.nodes[e.owner]);
  const boxEndpoint = wire.endpoints.find(e => e.owner_kind === "box" && layout.boxes && layout.boxes[e.owner]);
  if (!nodeEndpoint || !boxEndpoint) return;
  withAlpha(alpha, () => {
    const route = routeForWire(wire, layout);
    if (route.length < 2) return;
    const start = route[0];
    const end = route[route.length - 1];
    applyEdgeStyle();
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    if (nodeEndpoint.label) {
      addMathLabelCandidates(texForLabel(nodeEndpoint.label), labelCandidatesNearEndpoint(start, end, 0.65), "wire-label", alpha);
    }
  });
}

function drawTermStub(wire, layout, alpha = 1) {
  const nodeEndpoint = wire.endpoints.find(e => e.owner_kind === "node" && layout.nodes[e.owner]);
  const boxEndpoint = wire.endpoints.find(e => e.owner_kind === "box");
  if (!nodeEndpoint) return;
  withAlpha(alpha, () => {
    const route = routeForWire(wire, layout);
    if (route.length < 2) return;
    const start = route[0];
    const end = route[route.length - 1];
    applyEdgeStyle();
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    const localLabel = nodeEndpoint.label;
    const boundaryLabel = (boxEndpoint && boxEndpoint.label) || wire.label;
    if (localLabel && boundaryLabel && localLabel !== boundaryLabel) {
      addMathLabelCandidates(texForLabel(boundaryLabel), labelCandidatesForWire(layout, wire.id, termBoundaryLabelCandidates(start, end)), "wire-label", alpha);
    } else if (wire.label) {
      addMathLabelCandidates(texForLabel(wire.label), labelCandidatesForWire(layout, wire.id, termBoundaryLabelCandidates(start, end)), "wire-label", alpha);
    }
  });
}

function termBoundaryLabelCandidates(start, end) {
  const p = termStubPoint(start, end, 0.88);
  const q = termStubPoint(start, end, 0.7);
  const n = termStubNormal(start, end, 0.82);
  return [
    {point: {x: end.x + 7, y: end.y - 15}, anchor: "center", penalty: 0},
    {point: {x: p.x + n.x * 17, y: p.y + n.y * 17}, anchor: "center", penalty: 12},
    {point: {x: q.x - n.x * 15, y: q.y - n.y * 15}, anchor: "center", penalty: 24}
  ];
}

function termStubPoint(start, end, t) {
  return {
    x: start.x + (end.x - start.x) * t,
    y: start.y + (end.y - start.y) * t
  };
}

function termStubNormal(start, end, t) {
  const a = termStubPoint(start, end, Math.max(0, t - 0.03));
  const b = termStubPoint(start, end, Math.min(1, t + 0.03));
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  return {x: -dy / len, y: dx / len};
}

function labelCandidatesOnWire(pts) {
  const a = pts[0], b = pts[pts.length - 1];
  const mid = {x: (a.x + b.x) / 2, y: (a.y + b.y) / 2};
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const normal = {x: -dy / len, y: dx / len};
  const preferred = normal.y > 0 ? -1 : 1;
  return [-1, 1, 0].map((side, i) => {
    const s = side || preferred;
    return {
      point: {x: mid.x + normal.x * 10 * s, y: mid.y + normal.y * 10 * s - 4},
      anchor: "center",
      penalty: i === 0 ? 0 : i * 18
    };
  });
}

function visibleEndpointLabels(wire) {
  const labels = (wire.endpoints || []).map((endpoint, index) => ({index, label: endpoint.label})).filter(x => x.label);
  const unique = new Set(labels.map(x => x.label));
  if (unique.size <= 1) return [];
  return labels;
}

function labelOffset(point, neighbor) {
  const dx = point.x - neighbor.x, dy = point.y - neighbor.y;
  const len = Math.hypot(dx, dy) || 1;
  const along = {x: dx / len, y: dy / len};
  const normal = {x: -along.y, y: along.x};
  return {x: along.x * 11 + normal.x * 9, y: along.y * 11 + normal.y * 9 - 3};
}

function labelCandidatesNearEndpoint(point, neighbor, alongBias = 1) {
  const dx = point.x - neighbor.x, dy = point.y - neighbor.y;
  const len = Math.hypot(dx, dy) || 1;
  const along = {x: dx / len, y: dy / len};
  const normal = {x: -along.y, y: along.x};
  const base = {x: point.x + along.x * 19 * alongBias, y: point.y + along.y * 19 * alongBias - 4};
  return [
    {point: {x: base.x + normal.x * 16, y: base.y + normal.y * 16}, anchor: "center", penalty: 0},
    {point: {x: base.x - normal.x * 16, y: base.y - normal.y * 16}, anchor: "center", penalty: 18},
    {point: {x: base.x + along.x * 9, y: base.y + along.y * 9}, anchor: "center", penalty: 28}
  ];
}

function drawBoundaryDots(wire, pts, frame) {
  if (!frame) return;
  const boxes = Object.fromEntries(frame.boxes.map(b => [b.id, b]));
  wire.endpoints.forEach((endpoint, i) => {
    const box = boxes[endpoint.owner];
    if (!box || box.kind !== "derivative") return;
    const p = pts[i];
    ctx.save();
    ctx.fillStyle = "#000";
    ctx.beginPath();
    ctx.arc(p.x, p.y, STYLE.dotRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });
}

function drawCurveSegment(a, b, wire) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const dist = Math.hypot(dx, dy) || 1;
  const multiplicity = (wire.metadata && wire.metadata.multiplicity) || 1;
  if (wire.kind === "free" || multiplicity <= 1) {
    ctx.lineTo(b.x, b.y);
    return;
  }
  const nx = -dy / dist, ny = dx / dist;
  const bend = Math.min(14, dist * 0.1) * (multiplicity % 2 ? 1 : -1);
  const c1 = {x: a.x + dx * 0.42 + nx * bend, y: a.y + dy * 0.22 + ny * bend};
  const c2 = {x: b.x - dx * 0.42 + nx * bend, y: b.y - dy * 0.22 + ny * bend};
  ctx.bezierCurveTo(c1.x, c1.y, c2.x, c2.y, b.x, b.y);
}

function drawNode(node, layout, alpha = 1) {
  withAlpha(alpha, () => {
  const p = layout.nodes[node.id] || {x: 0, y: 0};
  const size = (layout.nodeSizes && layout.nodeSizes[node.id]) || {w: 34, h: 30};
  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.lineWidth = STYLE.nodeWidth;
  const style = nodeStyle(node.kind);
  ctx.strokeStyle = style.stroke;
  ctx.fillStyle = style.fill;
  if (node.kind === "variable") {
    ctx.beginPath();
    ctx.ellipse(0, 0, size.w / 2, size.h / 2, 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  } else if (node.kind === "function") {
    ctx.beginPath();
    ctx.ellipse(0, 0, Math.max(17, size.w / 2), Math.max(15, size.h / 2), 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  } else if (node.kind === "delta") {
    ctx.beginPath(); ctx.arc(0, 0, 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  } else {
    ctx.fillRect(-18, -13, 36, 26); ctx.strokeRect(-18, -13, 36, 26);
  }
  if (node.kind !== "delta" || node.label !== "Delta") {
    addMathLabel(texForLabel(node.label), p, "node-label", alpha);
  }
  ctx.restore();
  });
}

function drawBox(box, layout, alpha = 1) {
  withAlpha(alpha, () => {
  const rect = layout.boxes && layout.boxes[box.id];
  if (!rect) return;
  let {x, y, w, h} = rect;
  if (box.kind === "sum_term") {
    if (box.label) {
      const isCoeff = box.label !== "+";
      addMathLabel(
        texForLabel(box.label),
        {x: x + termLabelSlot(box.label) - (isCoeff ? 8 : 7), y: y + h / 2},
        isCoeff ? "coef-label" : "term-label",
        alpha,
        "right"
      );
    }
    return;
  }
  if (box.kind === "sum") {
    ctx.save();
    ctx.strokeStyle = "#8f8f8f";
    ctx.lineWidth = 1.05;
    ctx.beginPath();
    drawParenPath(x + 13, y + 7, h - 14, -1);
    drawParenPath(x + w - 6, y + 7, h - 14, 1);
    ctx.stroke();
    ctx.restore();
    return;
  }
  ctx.save();
  ctx.strokeStyle = box.kind === "derivative" ? "#050505" : "#aaa";
  ctx.setLineDash([]);
  ctx.lineWidth = box.kind === "derivative" ? STYLE.boxWidth : 1.1;
  roundedRectPath(ctx, x, y, w, h, box.kind === "derivative" ? 6 : 0);
  ctx.stroke();
  if (box.label) {
    addMathLabel(
      texForLabel(box.label),
      {x: x + 12, y: y + 12},
      box.kind === "derivative" ? "box-label derivative-label" : "box-label",
      alpha,
      "top-left"
    );
  }
  ctx.restore();
  });
}

function termLabelSlot(label) {
  if (!label) return 0;
  if (label === "+") return 18;
  if (String(label).startsWith("+")) return 32 + Math.max(0, String(label).length - 2) * 8;
  return 26 + Math.max(0, String(label).length - 1) * 8;
}

function drawParenPath(x, y, h, side) {
  const bow = 13 * side;
  ctx.moveTo(x, y);
  ctx.bezierCurveTo(x + bow, y + h * 0.2, x + bow, y + h * 0.8, x, y + h);
}

function roundedRectPath(ctx, x, y, w, h, r) {
  if (!r) {
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    return;
  }
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
}

function texForLabel(label) {
  if (label == null) return "";
  const text = String(label);
  if (text === "+") return "+";
  if (text.startsWith("+")) return "+\\," + texEscape(text.slice(1));
  if (text === "-") return "-";
  if (text.startsWith("d/d")) return "\\frac{d}{d" + texEscape(text.slice(3)) + "}";
  if (text === "Delta") return "\\Delta";
  return texEscape(text);
}

function texEscape(text) {
  return String(text)
    .replaceAll("\\", "\\backslash ")
    .replaceAll("_", "'")
    .replaceAll("{", "\\{")
    .replaceAll("}", "\\}");
}

function nodeStyle(kind) {
  if (kind === "variable") return {stroke: "#000", fill: "#fff"};
  if (kind === "function") return {stroke: "#000", fill: "#fff"};
  if (kind === "zero") return {stroke: "#000", fill: "#fff"};
  return {stroke: "#000", fill: "#fff"};
}

resize();
window.addEventListener("load", () => draw());
  </script>
</body>
</html>
"""
