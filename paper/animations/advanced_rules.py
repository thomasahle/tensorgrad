"""Three Matrix-Cookbook rules as smooth tensor-diagram motions.

1. DeleteTheNode:  d(a^T X b)/dX = a b^T
   The derivative loop localizes to X (linearity), X vanishes,
   its edges are left dangling: the outer product.

2. InverseSplit:   d(X^{-1}) = -X^{-1} dX X^{-1}
   The inverse node undergoes mitosis: it splits into two copies
   with the new derivative edges rising in the gap.

3. KroneckerTrace: Tr(A (x) B) = Tr(A) Tr(B)
   The two wires of the bundle trace are separate loops all along;
   pulling them apart proves the identity.

Book style: black ink on white, plain math labels, thin edges.
"""

import numpy as np
from manim import (
    Scene, MathTex, VGroup, Dot, Circle, Ellipse, Line, ArcBetweenPoints,
    CubicBezier,
    ValueTracker, always_redraw, BackgroundRectangle,
    FadeIn, FadeOut, Write, Create, Transform, ReplacementTransform,
    WHITE, BLACK, UP, DOWN, LEFT, RIGHT, ORIGIN, config, rate_functions,
)

config.background_color = WHITE
INK = BLACK
EASE = rate_functions.ease_in_out_sine


def node(tex, pos, scale=1.1):
    lbl = MathTex(tex, color=INK).scale(scale).move_to(pos)
    mask = BackgroundRectangle(lbl, color=WHITE, fill_opacity=1.0, buff=0.09)
    return VGroup(mask, lbl)


def dloop(center, width, height, dot_angle_deg=55, long_whiskers=False,
          wscale=1.0):
    """Penrose derivative loop: ellipse + dot on it + two bent whiskers
    (row whisker bending left, column whisker bending right).  With
    long_whiskers the two whiskers extend out to point horizontally left
    and right -- where the new edges will end up."""
    ell = Ellipse(width=width, height=height, color=INK, stroke_width=2.2)
    ell.move_to(center)
    a = np.deg2rad(dot_angle_deg)
    p = center + np.array([width / 2 * np.cos(a), height / 2 * np.sin(a), 0])
    dot = Dot(p, radius=0.05, color=INK)
    if long_whiskers:
        w = wscale
        wl = CubicBezier(p, p + w * np.array([-0.05, 0.28, 0]),
                         p + w * np.array([-0.60, 0.62, 0]),
                         p + w * np.array([-1.40, 0.60, 0]),
                         color=INK, stroke_width=2.2)
        wr = CubicBezier(p, p + w * np.array([0.10, 0.24, 0]),
                         p + w * np.array([0.60, 0.52, 0]),
                         p + w * np.array([1.35, 0.49, 0]),
                         color=INK, stroke_width=2.2)
    else:
        wl = ArcBetweenPoints(p, p + np.array([-0.28, 0.38, 0]), angle=-0.5,
                              color=INK, stroke_width=2.2)
        wr = ArcBetweenPoints(p, p + np.array([0.40, 0.26, 0]), angle=0.5,
                              color=INK, stroke_width=2.2)
    return VGroup(ell, dot, wl, wr)


class DeleteTheNode(Scene):
    def construct(self):
        # title
        lhs = MathTex(r"\frac{\partial\, (a^T X b)}{\partial X}", color=INK)
        eq = MathTex(r"=", color=INK)
        rhs = MathTex(r"a\, b^T", color=INK)
        title = VGroup(lhs, eq, rhs).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.7)

        # chain  -a--X--b-
        ax, xx, bx = np.array([-1.6, 0, 0]), np.array([0.0, 0, 0]), np.array([1.6, 0, 0])
        na, nX, nb = node("a", ax), node("X", xx), node("b", bx)
        # a and b are VECTORS: one edge each (into X); the whole thing
        # is a scalar, so there are no free edges anywhere.
        eaX = Line(ax + 0.25 * RIGHT, xx + 0.28 * LEFT, color=INK, stroke_width=2.2)
        eXb = Line(xx + 0.28 * RIGHT, bx + 0.25 * LEFT, color=INK, stroke_width=2.2)

        self.play(Write(lhs), run_time=1.0)
        self.play(FadeIn(na, nX, nb), Create(eaX), Create(eXb), run_time=1.0)
        self.wait(0.4)

        # derivative loop around the WHOLE expression...
        big = dloop(ORIGIN, 5.4, 1.7)
        self.play(Create(big[0]), FadeIn(big[1], big[2], big[3]), run_time=1.1)
        self.wait(0.5)

        # ...localizes to the only X (linearity)
        small = dloop(xx + 0.05 * UP, 1.1, 1.05)
        self.play(Transform(big, small), run_time=1.6, rate_func=EASE)
        self.wait(0.6)

        # delete the node: X and the loop vanish; its edges dangle
        da = Line(ax + 0.25 * RIGHT, ax + 0.80 * RIGHT + 0.16 * DOWN,
                  color=INK, stroke_width=2.2)
        db = Line(bx + 0.25 * LEFT, bx + 0.80 * LEFT + 0.16 * DOWN,
                  color=INK, stroke_width=2.2)
        self.play(FadeOut(nX, big),
                  Transform(eaX, da), Transform(eXb, db),
                  run_time=1.4, rate_func=EASE)
        self.wait(0.5)

        # the dangling edges bend around to face outwards (row edge left,
        # column edge right) while the two vectors close ranks:  -a b-
        phiA = ValueTracker(-16.2)
        phiB = ValueTracker(196.2)

        def dirv(deg):
            r = np.deg2rad(deg)
            return np.array([np.cos(r), np.sin(r), 0])

        stubA = always_redraw(lambda: Line(
            na.get_center() + 0.25 * dirv(phiA.get_value()),
            na.get_center() + 0.82 * dirv(phiA.get_value()),
            color=INK, stroke_width=2.2))
        stubB = always_redraw(lambda: Line(
            nb.get_center() + 0.25 * dirv(phiB.get_value()),
            nb.get_center() + 0.82 * dirv(phiB.get_value()),
            color=INK, stroke_width=2.2))
        self.remove(eaX, eXb)
        self.add(stubA, stubB)
        self.play(phiA.animate.set_value(180),
                  phiB.animate.set_value(0),
                  na.animate.shift(1.15 * RIGHT),
                  nb.animate.shift(1.15 * LEFT),
                  run_time=2.2, rate_func=EASE)
        self.wait(0.3)
        self.play(Write(eq), Write(rhs), run_time=0.9)
        self.wait(1.4)



def tri(pos, filled, size=0.30):
    """Penrose triangle glyph pointing right: outline = X, filled = X^{-1}."""
    pts = [pos + np.array([-0.7 * size, 0.62 * size, 0]),
           pos + np.array([-0.7 * size, -0.62 * size, 0]),
           pos + np.array([0.9 * size, 0, 0])]
    from manim import Polygon
    return Polygon(*pts, color=INK, stroke_width=2.2,
                   fill_color=(BLACK if filled else WHITE), fill_opacity=1.0)


def wire(x0, x1, y=0.0, sw=2.2):
    return Line(np.array([x0, y, 0]), np.array([x1, y, 0]),
                color=INK, stroke_width=sw)


def dangles(cx, y=0.0, gap=0.5):
    """Deleted-node stubs: two short edges pointing into the gap at cx."""
    l = Line(np.array([cx - gap, y, 0]),
             np.array([cx - gap + 0.30, y + 0.26, 0]), color=INK, stroke_width=2.2)
    r = Line(np.array([cx + gap, y, 0]),
             np.array([cx + gap - 0.30, y + 0.26, 0]), color=INK, stroke_width=2.2)
    return VGroup(l, r)



from manim import CubicBezier


def pendant(pos, s=0.55, gap=True):
    """Penrose's inverse, chain-embeddable: two vertical antisymmetrizer
    bars holding n-1 = 2 copies of X and one empty slot at the bottom,
    whose two legs exit downward and CROSS (the crossing is the transpose
    that turns the adjugate into the inverse) before landing on the chain
    wire at pos+-(.95s, 0).  With gap=False the slot is filled by a third
    X and there are no legs: the closed pendant is det(X)."""
    P = lambda x, y: pos + np.array([x * s, y * s, 0])
    lsc = max(0.8 * s, 0.45)
    g = VGroup()
    g.add(Line(P(-.52, .25), P(-.52, 1.75), color=INK, stroke_width=3.4))
    g.add(Line(P(.52, .25), P(.52, 1.75), color=INK, stroke_width=3.4))
    ys = [1.45, 1.0] + ([] if gap else [.55])
    for yy in ys:
        lbl = MathTex("X", color=INK).scale(lsc).move_to(P(0, yy))
        g.add(Line(P(-.52, yy), P(-.18, yy), color=INK, stroke_width=2.2))
        g.add(Line(P(.18, yy), P(.52, yy), color=INK, stroke_width=2.2))
        g.add(lbl)
    if gap:
        g.add(Line(P(-.52, .55), P(-.24, .55), color=INK, stroke_width=2.2))
        g.add(Line(P(.52, .55), P(.24, .55), color=INK, stroke_width=2.2))
        g.add(CubicBezier(P(-.24, .55), P(-.06, .32), P(.5, .08),
                          P(.95, 0), color=INK, stroke_width=2.2))
        g.add(CubicBezier(P(.24, .55), P(.06, .32), P(-.5, .08),
                          P(-.95, 0), color=INK, stroke_width=2.2))
    return g


def prefactor(pos, s=0.62):
    """The n/det(X) normalization: 3 over a closed det pendant."""
    det = pendant(pos + np.array([0, -1.05, 0]), s=s, gap=False)
    bar = Line(pos + np.array([-0.55, 0.28, 0]), pos + np.array([0.55, 0.28, 0]),
               color=INK, stroke_width=1.6)
    three = MathTex("3", color=INK).move_to(pos + np.array([0, 0.72, 0]))
    return VGroup(three, bar, det)


class InverseSplit(Scene):
    """d(X^-1) = -X^-1 dX X^-1, in Penrose's representation of the
    inverse: the adjugate gap-stack over the determinant."""

    def eqrow(self, y, lpend, loopon, minus=True, rpend=True):
        """Build one equation state:  [lpend?] X|- (loop inv) = - [rpend?] dX inv.
        Returns dict of mobject groups, all positioned in fixed slots."""
        S = 0.55
        out = {}
        # LHS: wire, optional composed pendant, X (or nothing), looped pendant
        xslots = {'pend1': -5.1, 'X': -3.85, 'pend2': -2.35}
        segs = [Line([-6.1, y, 0], [-1.35, y, 0], color=INK, stroke_width=2.2)]
        out['lwire'] = VGroup(*segs)
        if lpend:
            out['pend1'] = pendant(np.array([xslots['pend1'], y, 0]), s=S)
        out['Xn'] = node("X", np.array([xslots['X'], y, 0]))
        out['pend2'] = pendant(np.array([xslots['pend2'], y, 0]), s=S)
        if loopon:
            out['loop'] = dloop(np.array([xslots['pend2'], y + 0.5, 0]),
                                2.0, 2.15, dot_angle_deg=38)
        out['eq'] = MathTex(r"=", color=INK).move_to([-0.72, y, 0])
        if minus:
            out['minus'] = MathTex(r"-", color=INK
                                   ).scale(1.1).move_to([-0.28, y, 0])
        # RHS: wire with a GAP at the dangles, optional pendant, pendant
        xr = {'pend3': 1.15, 'dang': 2.9, 'pend4': 4.55}
        gap = 0.38
        out['rwire'] = VGroup(
            Line([0.15, y, 0], [xr['dang'] - gap, y, 0], color=INK, stroke_width=2.2),
            Line([xr['dang'] + gap, y, 0], [5.9, y, 0], color=INK, stroke_width=2.2))
        if rpend:
            out['pend3'] = pendant(np.array([xr['pend3'], y, 0]), s=S)
        out['dang'] = dangles(xr['dang'], y, gap=gap)
        out['pend4'] = pendant(np.array([xr['pend4'], y, 0]), s=S)
        return out

    def construct(self):
        lhs = MathTex(r"\mathrm{d}(X^{-1})", color=INK)
        eq0 = MathTex(r"=", color=INK)
        rhs = MathTex(r"-\,X^{-1}\,\mathrm{d}X\,X^{-1}", color=INK)
        title = VGroup(lhs, eq0, rhs).arrange(RIGHT, buff=0.22).to_edge(UP, buff=0.55)

        # ---- Act 1: Penrose's inverse ----
        c1 = np.array([-1.6, -1.35, 0])
        inv_lhs = MathTex(r"X^{-1}", color=INK).move_to(c1 + np.array([-1.7, 0.75, 0]))
        inv_eq = MathTex(r"=", color=INK).move_to(c1 + np.array([-0.95, 0.75, 0]))
        pre = prefactor(c1 + np.array([0, 0.75, 0]))
        cdot = MathTex(r"\cdot", color=INK).move_to(c1 + np.array([0.9, 0.75, 0]))
        p1 = c1 + np.array([2.6, -0.1, 0])
        big = pendant(p1, s=0.9)
        wdef = Line(p1 + np.array([-1.5, 0, 0]), p1 + np.array([1.5, 0, 0]),
                    color=INK, stroke_width=2.2)
        self.play(Write(lhs), run_time=0.8)
        self.play(Write(inv_lhs), Write(inv_eq), FadeIn(pre), Write(cdot),
                  Create(wdef), FadeIn(big), run_time=1.4)
        self.wait(1.4)

        # ---- Act 2: the key fact  X X^{-1} = I : X docks into the slot ----
        sub = MathTex(r"X\,X^{-1} \;=\; I:", color=INK
                      ).scale(0.8).move_to([-4.9, 1.7, 0])
        p0 = np.array([0.6, -0.6, 0])
        wire = Line(p0 + np.array([-4.0, 0, 0]), p0 + np.array([2.6, 0, 0]),
                    color=INK, stroke_width=2.2)
        big2 = pendant(p0, s=0.9)
        pre2 = prefactor(p0 + np.array([-2.9, 0.9, 0]))
        Xn = node("X", p0 + np.array([1.8, 0, 0]))
        self.play(FadeOut(inv_lhs, inv_eq, cdot),
                  ReplacementTransform(big, big2),
                  ReplacementTransform(wdef, wire),
                  ReplacementTransform(pre, pre2),
                  FadeIn(sub, Xn), run_time=1.2)
        self.wait(0.6)
        # X slides along the wire to sit under the empty slot...
        self.play(Xn.animate.move_to(p0 + np.array([0, 0.05, 0])),
                  run_time=1.5, rate_func=EASE)
        self.wait(0.3)
        # ...and docks: the slot closes into det(X)
        closed = pendant(p0, s=0.9, gap=False)
        self.play(ReplacementTransform(VGroup(big2, Xn), closed), run_time=1.0)
        self.wait(0.4)
        # det/det = 1: the normalization cancels, a plain wire remains
        self.play(FadeOut(closed, pre2), run_time=1.0)
        self.wait(0.9)

        # ---- Act 3: differentiate the identity ----
        self.play(FadeOut(sub, wire), run_time=0.7)
        yA, yB = 1.0, -1.7
        S = 0.55

        def chain(P, looped):
            Xn = node("X", P + np.array([-1.55, 0, 0]))
            g = VGroup(Line(P + np.array([-2.75, 0, 0]), P + np.array([-1.8, 0, 0]),
                            color=INK, stroke_width=2.2),
                       Xn,
                       Line(P + np.array([-1.3, 0, 0]), P + np.array([1.45, 0, 0]),
                            color=INK, stroke_width=2.2),
                       pendant(P, s=S))
            if looped == "X":
                lp = dloop(P + np.array([-1.55, 0.05, 0]), 1.0, 0.9)
            else:
                lp = dloop(P + np.array([0, 0.5, 0]), 2.0, 2.15, dot_angle_deg=38)
            return g, lp

        PA, PB = np.array([0.5, yA, 0]), np.array([0.5, yB, 0])
        gA, loopA = chain(PA, "X")
        gB, loopB = chain(PB, "inv")
        plusA = MathTex(r"+", color=INK).move_to(PA + np.array([2.1, 0, 0]))
        eqB = MathTex(r"=\,0", color=INK).move_to(PB + np.array([2.25, 0, 0]))
        self.play(FadeIn(gA, gB), Create(loopA[0]), Create(loopB[0]),
                  FadeIn(loopA[1], loopA[2], loopA[3],
                         loopB[1], loopB[2], loopB[3]),
                  Write(plusA), Write(eqB), run_time=1.4)
        self.wait(1.0)

        # delete the node in the first term: X -> dangling dX edges
        dA = dangles(PA[0] - 1.55, yA, gap=0.38)
        brA = VGroup(Line(PA + np.array([-2.75, 0, 0]), PA + np.array([-1.93, 0, 0]),
                          color=INK, stroke_width=2.2),
                     Line(PA + np.array([-1.17, 0, 0]), PA + np.array([-1.3, 0, 0]),
                          color=INK, stroke_width=2.2))
        self.play(FadeOut(gA[1], loopA), FadeIn(dA), run_time=1.0)
        self.wait(0.8)

        # solve: move the dX term across the equals sign
        ym = -0.6
        st = self.eqrow(ym, lpend=False, loopon=True)
        self.play(
            ReplacementTransform(VGroup(gB[0], gB[2]), st['lwire']),
            ReplacementTransform(gB[1], st['Xn']),
            ReplacementTransform(gB[3], st['pend2']),
            ReplacementTransform(loopB, st['loop']),
            ReplacementTransform(VGroup(gA[0], gA[2]), st['rwire']),
            ReplacementTransform(dA, st['dang']),
            ReplacementTransform(gA[3], st['pend4']),
            FadeOut(plusA, eqB), Write(st['eq']), Write(st['minus']),
            run_time=1.7, rate_func=EASE)
        self.wait(1.0)

        # compose both sides with X^{-1} on the left
        pend1 = pendant(np.array([-5.1, ym, 0]), s=S)
        pend3 = pendant(np.array([1.15, ym, 0]), s=S)
        self.play(FadeIn(pend1, pend3), run_time=0.9)
        self.wait(0.7)

        # the new X^{-1} swallows X: dock, close into det, cancel
        self.play(st['Xn'].animate.move_to([-5.1, ym + 0.03, 0]),
                  run_time=1.3, rate_func=EASE)
        closedL = pendant(np.array([-5.1, ym, 0]), s=S, gap=False)
        self.play(ReplacementTransform(VGroup(pend1, st['Xn']), closedL),
                  run_time=0.9)
        self.play(FadeOut(closedL), run_time=0.8)
        self.wait(0.8)

        # ---- read off the identity ----
        self.play(Write(eq0), Write(rhs), run_time=1.0)
        self.wait(1.8)


class KroneckerTrace(Scene):
    """Tr(A (x) B) = Tr(A) Tr(B), following the storyboard: the closed
    bundle trace with flatten triangles; the triangles cancel and the two
    strands of the double wire peel apart (crossing at the sides); A's
    loop slips inside B's (nested); the loops separate, stacked A over B.

    Each loop is ONE smooth closed parametric curve throughout, whose
    shape parameters interpolate between the four keyframes, so every
    frame is a valid pair of closed loops.
    """

    def construct(self):
        from manim import ParametricFunction, Polygon
        t1 = MathTex(r"\mathrm{Tr}(A \otimes B)", color=INK).to_edge(UP, buff=0.6)
        t3l = MathTex(r"\mathrm{Tr}(A \otimes B)", color=INK)
        t3e = MathTex(r"=", color=INK)
        t3r = MathTex(r"\mathrm{Tr}(A)\,\mathrm{Tr}(B)", color=INK)
        t3 = VGroup(t3l, t3e, t3r).arrange(RIGHT, buff=0.22).to_edge(UP, buff=0.6)

        # keyframes:   panel1/2 (closed trace),  panel3 (nested),  panel4 (split)
        # params per loop: cx, cy, rtx, rty, rbx, rby  (top/bottom x/y radii)
        KA = np.array([
            [0, -0.30, 2.42, 2.30, 1.25, 1.05],   # 0: double top, A inner-bottom
            [0, -0.30, 2.42, 2.30, 1.28, 1.08],   # 1: triangles gone (peel)
            [0,  0.20, 1.00, 0.72, 0.95, 0.62],   # 2: small loop inside B
            [0,  0.85, 0.85, 0.52, 0.85, 0.48],   # 3: its own trace, on top
        ])
        KB = np.array([
            [0, -0.30, 2.28, 2.16, 2.10, 1.85],   # 0
            [0, -0.30, 2.28, 2.16, 2.08, 1.83],   # 1
            [0, -0.20, 2.15, 1.95, 1.95, 1.80],   # 2: still the big loop
            [0, -1.35, 0.85, 0.52, 0.85, 0.48],   # 3: its own trace, below
        ])
        tt = ValueTracker(0.0)

        def P(K):
            u = tt.get_value()
            return np.array([np.interp(u, [0, 1, 2, 3], K[:, k])
                             for k in range(6)])

        def looppt(K, th):
            cx, cy, rtx, rty, rbx, rby = P(K)
            # keep the ribbon parallel along the top; fan only at the sides
            w = np.clip((np.sin(th) + 0.45) / 0.9, 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            rx = w * rtx + (1 - w) * rbx
            ry = w * rty + (1 - w) * rby
            return np.array([cx + rx * np.cos(th), cy + ry * np.sin(th), 0])

        def mkloop(K):
            return always_redraw(lambda: ParametricFunction(
                lambda th: looppt(K, th), t_range=[0, 2 * np.pi + 1e-3],
                color=INK, stroke_width=2.2))

        loopA, loopB = mkloop(KA), mkloop(KB)
        labA = always_redraw(lambda: node("A", looppt(KA, 3 * np.pi / 2)))
        labB = always_redraw(lambda: node("B", looppt(KB, 3 * np.pi / 2)))

        # flatten triangles, sitting over the side fan regions as masks
        def tri_mask(x, flip):
            sz = 0.50
            yc = 0.08
            pts = [np.array([x + flip * sz, yc + 0.5 * sz, 0]),
                   np.array([x + flip * sz, yc - 0.5 * sz, 0]),
                   np.array([x - flip * 0.65 * sz, yc, 0])]
            return Polygon(*pts, color=INK, stroke_width=2.2,
                           fill_color=WHITE, fill_opacity=1.0)

        triL, triR = tri_mask(-2.12, +1), tri_mask(2.12, -1)

        # ---- panel 1: the closed bundle trace ----
        self.play(Write(t1), run_time=0.8)
        self.play(Create(loopA), Create(loopB), run_time=1.6)
        self.play(FadeIn(triL, triR, labA, labB), run_time=0.8)
        self.wait(1.0)

        # ---- panel 2: the triangles cancel; the strands peel apart ----
        self.play(FadeOut(triL, triR),
                  tt.animate.set_value(1.0), run_time=1.4, rate_func=EASE)
        self.wait(0.8)

        # ---- panel 3: A's loop slips inside ----
        self.play(tt.animate.set_value(2.0), run_time=2.2, rate_func=EASE)
        self.wait(0.7)

        # ---- panel 4: two separate traces, A over B ----
        self.play(tt.animate.set_value(3.0), run_time=2.2, rate_func=EASE)
        self.play(ReplacementTransform(t1, t3), run_time=0.9)
        self.wait(1.8)


class TraceDelete(Scene):
    """d Tr(AXB)/dX = A^T B^T, following the storyboard: derivative loop
    around the closed trace; the loop localizes to X; X is deleted leaving
    two curled hooks; the closure arc pulls straight while both nodes turn
    around (their glyphs flipping is the transpose!), and the flipped
    nodes are relabeled A^T, B^T."""

    def construct(self):
        from manim import ArcBetweenPoints as ABP
        lhs = MathTex(r"\frac{\partial\, \mathrm{Tr}(AXB)}{\partial X}", color=INK)
        eq = MathTex(r"=", color=INK)
        rhs = MathTex(r"A^T B^T", color=INK)
        title = VGroup(lhs, eq, rhs).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.55)

        ca, cb, cX = np.array([-1.15, -0.6, 0]), np.array([1.15, -0.6, 0]), np.array([0, -0.6, 0])
        phi = ValueTracker(0.0)     # how far each node has turned (deg)
        kap = ValueTracker(1.0)     # hook curl: 1 = curled, 0 = straight

        def dirv(deg):
            r = np.deg2rad(deg)
            return np.array([np.cos(r), np.sin(r), 0])

        # the closure: a dome from A's arc-edge to B's arc-edge; the
        # attachment angles rotate over the top as the nodes turn, and
        # the dome flattens into the straight middle wire.
        def dome():
            u = phi.get_value() / 180.0
            aA, aB = 180 - phi.get_value(), 0 + phi.get_value()
            p1 = ca + 0.30 * dirv(aA)
            p2 = cb + 0.30 * dirv(aB)
            lift = (1.55 * (1 - u) + 0.06) * np.array([0, 1, 0])
            c1 = p1 + (1.25 * (1 - u) + 0.12) * dirv(aA) + lift
            c2 = p2 + (1.25 * (1 - u) + 0.12) * dirv(aB) + lift
            return CubicBezier(p1, c1, c2, p2, color=INK, stroke_width=2.2)

        # the dangling derivative edges: curled hooks that unroll outward
        # around the bottom as the nodes turn.
        def hook(c, base):
            th = base - phi.get_value() if base == 0 else base + phi.get_value()
            p = c + 0.30 * dirv(th)
            k = kap.get_value()
            return ABP(p, p + (0.42 + 0.13 * (1 - k)) * dirv(th + 55 * k),
                       angle=2.2 * k, color=INK, stroke_width=2.2)

        # labels rotate rigidly with their nodes
        def lab(tex, c, sgn):
            m = MathTex(tex, color=INK).scale(1.1)
            m.rotate(sgn * np.deg2rad(phi.get_value())).move_to(c)
            mask = BackgroundRectangle(m, color=WHITE, fill_opacity=1.0, buff=0.08)
            return VGroup(mask, m)

        domeM = always_redraw(dome)
        nA = always_redraw(lambda: lab("A", ca, -1))
        nB = always_redraw(lambda: lab("B", cb, +1))
        nX = node("X", cX)
        wAX = Line(ca + 0.30 * RIGHT, cX + 0.30 * LEFT, color=INK, stroke_width=2.2)
        wXB = Line(cX + 0.30 * RIGHT, cb + 0.30 * LEFT, color=INK, stroke_width=2.2)

        # ---- build the formula and the diagram together:
        #      X,  AXB,  Tr(AXB),  d Tr(AXB)/dX ----
        tpos = lhs.get_center()
        tX = MathTex("X", color=INK).move_to(tpos)
        tAXB = MathTex("A", "X", "B", color=INK).move_to(tpos)
        tTr = MathTex(r"\mathrm{Tr}(", "A", "X", "B", ")", color=INK).move_to(tpos)
        stubL = Line(cX + 0.30 * LEFT, cX + 1.05 * LEFT, color=INK, stroke_width=2.2)
        stubR = Line(cX + 0.30 * RIGHT, cX + 1.05 * RIGHT, color=INK, stroke_width=2.2)
        self.play(Write(tX), FadeIn(nX), Create(stubL), Create(stubR), run_time=0.9)
        self.wait(0.6)

        soA = Line(ca + 0.30 * LEFT, ca + 1.0 * LEFT, color=INK, stroke_width=2.2)
        soB = Line(cb + 0.30 * RIGHT, cb + 1.0 * RIGHT, color=INK, stroke_width=2.2)
        self.add(nA, nB)
        self.play(ReplacementTransform(tX, tAXB[1]), FadeIn(tAXB[0], tAXB[2]),
                  FadeIn(nA, nB),
                  ReplacementTransform(stubL, wAX),
                  ReplacementTransform(stubR, wXB),
                  Create(soA), Create(soB), run_time=1.1)
        self.wait(0.6)

        # close the trace: each stub sweeps up into its half of the dome
        # (splitting the dome bezier at the apex avoids a ghost double)
        p1 = ca + 0.30 * dirv(180)
        p2 = cb + 0.30 * dirv(0)
        lift = 1.61 * np.array([0, 1, 0])
        c1 = p1 + 1.37 * dirv(180) + lift
        c2 = p2 + 1.37 * dirv(0) + lift
        apex = (p1 + 3 * c1 + 3 * c2 + p2) / 8
        domeL = CubicBezier(p1, (p1 + c1) / 2, (p1 + 2 * c1 + c2) / 4, apex,
                            color=INK, stroke_width=2.2)
        domeR = CubicBezier(apex, (c1 + 2 * c2 + p2) / 4, (c2 + p2) / 2, p2,
                            color=INK, stroke_width=2.2)
        self.play(ReplacementTransform(tAXB[0], tTr[1]),
                  ReplacementTransform(tAXB[1], tTr[2]),
                  ReplacementTransform(tAXB[2], tTr[3]),
                  FadeIn(tTr[0], tTr[4]),
                  ReplacementTransform(soA, domeL),
                  ReplacementTransform(soB, domeR),
                  run_time=1.2, rate_func=EASE)
        self.remove(domeL, domeR)
        self.add(domeM)
        self.wait(0.6)

        # d/dX just appears around the existing Tr(AXB), which glides
        # into its numerator slot -- no glyph morphing.
        g = lhs[0]
        ymid = lhs.get_center()[1]
        num = [m for m in g if m.get_center()[1] > ymid + 0.05]
        num.sort(key=lambda m: m.get_center()[0])
        tr_target = VGroup(*num[1:])          # numerator minus the partial
        rest = VGroup(*[m for m in g if m not in num[1:]])
        big = dloop(np.array([0, -0.25, 0]), 5.6, 2.9, dot_angle_deg=72,
                    long_whiskers=True)
        self.play(tTr.animate.move_to(tr_target.get_center()
                  ).scale(tr_target.height / tTr.height),
                  FadeIn(rest),
                  Create(big[0]), FadeIn(big[1], big[2], big[3]), run_time=1.2)
        self.remove(tTr)
        self.add(lhs)
        self.wait(0.7)

        # ---- panel 2: the loop localizes to the only X ----
        small = dloop(cX + np.array([0, 0.05, 0]), 1.05, 1.0, dot_angle_deg=72,
                      long_whiskers=True, wscale=0.6)
        self.play(Transform(big, small), run_time=1.5, rate_func=EASE)
        self.wait(0.6)

        # ---- panel 3: delete the node; its edges are left as curled hooks ----
        hookA = always_redraw(lambda: hook(ca, 0))
        hookB = always_redraw(lambda: hook(cb, 180))
        hA0, hB0 = hook(ca, 0), hook(cb, 180)
        self.play(FadeOut(nX, big),
                  ReplacementTransform(wAX, hA0),
                  ReplacementTransform(wXB, hB0), run_time=1.2, rate_func=EASE)
        self.remove(hA0, hB0)
        self.add(hookA, hookB)
        self.wait(0.7)

        # ---- panels 4-5: the arc pulls straight; the nodes turn around;
        #      the hooks unroll outward.  The flip IS the transpose. ----
        self.play(phi.animate.set_value(180), kap.animate.set_value(0.02),
                  run_time=3.2, rate_func=EASE)
        self.wait(0.5)

        # ---- panel 6: the glyphs spin upright as the ^T appears ----
        from manim import Rotate, PI
        finA = MathTex("A", "^{T}", color=INK).scale(1.1)
        finA.shift(ca - finA[0].get_center())
        finB = MathTex("B", "^{T}", color=INK).scale(1.1)
        finB.shift(cb - finB[0].get_center())
        rotA = MathTex("A", color=INK).scale(1.1).move_to(ca).rotate(-PI)
        rotB = MathTex("B", color=INK).scale(1.1).move_to(cb).rotate(PI)
        self.remove(nA, nB)
        self.add(rotA, rotB)
        self.play(Rotate(rotA, angle=-PI, about_point=ca),
                  Rotate(rotB, angle=PI, about_point=cb),
                  FadeIn(finA[1], finB[1]),
                  run_time=1.1, rate_func=EASE)
        self.remove(rotA, rotB)
        self.add(finA[0], finB[0])
        self.play(Write(eq), Write(rhs), run_time=1.0)
        self.wait(1.8)
