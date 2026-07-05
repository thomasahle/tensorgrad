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


def dloop(center, width, height, dot_angle_deg=55):
    """Penrose derivative loop: ellipse + dot on it + two bent whiskers
    (row whisker bending left, column whisker bending right)."""
    ell = Ellipse(width=width, height=height, color=INK, stroke_width=2.2)
    ell.move_to(center)
    a = np.deg2rad(dot_angle_deg)
    p = center + np.array([width / 2 * np.cos(a), height / 2 * np.sin(a), 0])
    dot = Dot(p, radius=0.05, color=INK)
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
        self.wait(0.4)
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



class InverseSplit(Scene):
    """d(X^-1) = -X^-1 dX X^-1, derived Penrose-style from X X^-1 = I."""

    def T(self, x, y, filled):
        return tri(np.array([x, y, 0]), filled)

    def construct(self):
        lhs = MathTex(r"\mathrm{d}(X^{-1})", color=INK)
        eq0 = MathTex(r"=", color=INK)
        rhs = MathTex(r"-\,X^{-1}\,\mathrm{d}X\,X^{-1}", color=INK)
        title = VGroup(lhs, eq0, rhs).arrange(RIGHT, buff=0.22).to_edge(UP, buff=0.6)
        legend = VGroup(
            tri(np.array([0, 0, 0]), False), MathTex(r"= X", color=INK).scale(0.8),
            tri(np.array([0, 0, 0]), True), MathTex(r"= X^{-1}", color=INK).scale(0.8),
        ).arrange(RIGHT, buff=0.3).to_edge(DOWN, buff=0.6)
        self.play(Write(lhs), FadeIn(legend), run_time=1.0)

        # ---- S1: the identity  -X-X^{-1}-  =  --- ----
        y = 0.3
        ident = VGroup(wire(-2.1, -1.08, y), self.T(-0.85, y, False),
                       wire(-0.60, 0.12, y), self.T(0.35, y, True),
                       wire(0.62, 1.85, y))
        eqI = MathTex(r"=", color=INK).move_to([2.5, y, 0])
        plain = wire(3.0, 4.3, y)
        self.play(FadeIn(ident), run_time=0.8)
        self.play(Write(eqI), Create(plain), run_time=0.8)
        self.wait(0.6)

        # ---- S2: differentiate both sides ----
        loop = dloop(np.array([-0.25, y, 0]), 2.6, 1.05)
        zero = MathTex(r"0", color=INK).move_to(plain.get_center())
        self.play(Create(loop[0]), FadeIn(loop[1], loop[2], loop[3]),
                  ReplacementTransform(plain, zero), run_time=1.1)
        self.wait(0.7)

        # ---- S3: product rule -- the loop distributes over the two nodes ----
        yA, yB = 0.55, -0.75
        rowA = VGroup(wire(-2.6, -1.58, yA), self.T(-1.35, yA, False),
                      wire(-1.10, 0.12, yA), self.T(0.35, yA, True),
                      wire(0.62, 1.6, yA))
        loopA = dloop(np.array([-1.35, yA, 0]), 1.15, 0.95)
        plusA = MathTex(r"+", color=INK).move_to([2.1, yA, 0])
        rowB = VGroup(wire(-2.6, -1.58, yB), self.T(-1.35, yB, False),
                      wire(-1.10, 0.12, yB), self.T(0.35, yB, True),
                      wire(0.62, 1.6, yB))
        loopB = dloop(np.array([0.35, yB, 0]), 1.15, 0.95)
        eqB = MathTex(r"=\,0", color=INK).move_to([2.35, yB, 0])
        self.play(ReplacementTransform(ident, rowA),
                  ReplacementTransform(loop, loopA),
                  FadeOut(eqI, zero),
                  FadeIn(rowB, loopB), Write(plusA), Write(eqB),
                  run_time=1.5, rate_func=EASE)
        self.wait(0.8)

        # ---- S4: delete the node in the first term:  loop X -> dangling dX ----
        dA = dangles(-1.35, yA, gap=0.42)
        self.play(FadeOut(rowA[1], loopA), FadeIn(dA), run_time=1.0)
        self.wait(0.7)

        # ---- S5: solve for the looped term (move the dX term across) ----
        ym = -0.1
        L = VGroup(wire(-4.3, -3.60, ym), self.T(-3.35, ym, False),
                   wire(-3.10, -2.40, ym), self.T(-2.15, ym, True),
                   wire(-1.90, -1.20, ym))
        loopL = dloop(np.array([-2.15, ym, 0]), 1.15, 0.95)
        eqM = MathTex(r"=", color=INK).move_to([-0.72, ym, 0])
        minus = MathTex(r"-", color=INK).scale(1.1).move_to([-0.30, ym, 0])
        R = VGroup(wire(0.05, 0.75, ym),
                   VGroup(Line([0.75, ym, 0], [1.05, ym + 0.26, 0],
                               color=INK, stroke_width=2.2),
                          Line([1.65, ym, 0], [1.35, ym + 0.26, 0],
                               color=INK, stroke_width=2.2)),
                   wire(1.65, 2.35, ym), self.T(2.60, ym, True),
                   wire(2.85, 3.55, ym))
        self.play(ReplacementTransform(VGroup(rowB), L),
                  ReplacementTransform(loopB, loopL),
                  ReplacementTransform(VGroup(rowA[0], rowA[2], rowA[3],
                                              rowA[4], dA), R),
                  FadeOut(plusA, eqB),
                  Write(eqM), Write(minus),
                  run_time=1.6, rate_func=EASE)
        self.wait(0.8)

        # ---- S6: compose with X^{-1} on the left of both sides ----
        newL = VGroup(wire(-5.35, -4.85, ym), self.T(-4.60, ym, True))
        newR = self.T(-0.05 + 0.45, ym, True)   # placed just after the minus
        newRw = wire(0.70, 1.05, ym)
        # shift the old right side to make room
        self.play(FadeIn(newL),
                  R.animate.shift(np.array([1.0, 0, 0])),
                  FadeIn(newR.shift(np.array([-0.25, 0, 0])), newRw),
                  run_time=1.2, rate_func=EASE)
        self.wait(0.6)

        # ---- S7: X^{-1} X annihilates into a plain wire ----
        pair = VGroup(newL[1], L[1])   # black then white triangle
        mid = np.array([-3.95, ym, 0])
        self.play(pair[0].animate.move_to(mid + np.array([-0.18, 0, 0])),
                  pair[1].animate.move_to(mid + np.array([0.18, 0, 0])),
                  run_time=0.9, rate_func=EASE)
        bridge = wire(-5.35, -2.40, ym)
        self.play(FadeOut(pair, L[0], L[2], newL[0]), FadeIn(bridge),
                  run_time=0.8)
        self.wait(0.6)

        # ---- S8: read off the identity ----
        self.play(Write(eq0), Write(rhs), run_time=1.0)
        self.wait(1.6)


class KroneckerTrace(Scene):
    """Tr(A (x) B) = Tr(A) Tr(B), from the definition of (x)."""

    def construct(self):
        from manim import Arc
        t1 = MathTex(r"A \otimes B", color=INK).to_edge(UP, buff=0.7)
        t2 = MathTex(r"\mathrm{Tr}(A \otimes B)", color=INK).to_edge(UP, buff=0.7)
        t3l = MathTex(r"\mathrm{Tr}(A \otimes B)", color=INK)
        t3e = MathTex(r"=", color=INK)
        t3r = MathTex(r"\mathrm{Tr}(A)\,\mathrm{Tr}(B)", color=INK)
        t3 = VGroup(t3l, t3e, t3r).arrange(RIGHT, buff=0.22).to_edge(UP, buff=0.7)

        # ---- Stage 1: the definition ----
        y0 = -0.55
        A0, B0 = np.array([0, y0 + 0.55, 0]), np.array([0, y0 - 0.55, 0])
        nA, nB = node("A", A0), node("B", B0)
        LT = tri(np.array([-1.6, y0, 0]), False).rotate(np.pi)
        RT = tri(np.array([1.6, y0, 0]), False)
        # wires attach at the triangles' fan corners (upper / lower)
        lu, ll = np.array([-1.39, y0 + 0.19, 0]), np.array([-1.39, y0 - 0.19, 0])
        ru, rl = np.array([1.39, y0 + 0.19, 0]), np.array([1.39, y0 - 0.19, 0])
        wAl = Line(lu, A0 + 0.28 * LEFT, color=INK, stroke_width=2.2)
        wAr = Line(A0 + 0.28 * RIGHT, ru, color=INK, stroke_width=2.2)
        wBl = Line(ll, B0 + 0.28 * LEFT, color=INK, stroke_width=2.2)
        wBr = Line(B0 + 0.28 * RIGHT, rl, color=INK, stroke_width=2.2)
        apexL, apexR = np.array([-1.87, y0, 0]), np.array([1.87, y0, 0])
        dy = 0.055
        stubs = VGroup(
            Line(apexL, apexL + np.array([-0.75, dy, 0]), color=INK, stroke_width=2.2),
            Line(apexL, apexL + np.array([-0.75, -dy, 0]), color=INK, stroke_width=2.2),
            Line(apexR, apexR + np.array([0.75, dy, 0]), color=INK, stroke_width=2.2),
            Line(apexR, apexR + np.array([0.75, -dy, 0]), color=INK, stroke_width=2.2))

        self.play(Write(t1), run_time=0.8)
        self.play(FadeIn(nA, nB, LT, RT),
                  Create(VGroup(wAl, wAr, wBl, wBr)), Create(stubs), run_time=1.2)
        self.wait(0.8)

        # ---- Stage 2: close the bundle over the top ----
        rO, rI = 1.98, 1.87
        C = np.array([0, y0, 0])
        arcO_l = Arc(radius=rO, start_angle=np.pi, angle=-np.pi / 2, color=INK,
                     stroke_width=2.2, arc_center=C)
        arcO_r = Arc(radius=rO, start_angle=np.pi / 2, angle=-np.pi / 2, color=INK,
                     stroke_width=2.2, arc_center=C)
        arcI_l = Arc(radius=rI, start_angle=np.pi, angle=-np.pi / 2, color=INK,
                     stroke_width=2.2, arc_center=C)
        arcI_r = Arc(radius=rI, start_angle=np.pi / 2, angle=-np.pi / 2, color=INK,
                     stroke_width=2.2, arc_center=C)
        self.play(ReplacementTransform(t1, t2), run_time=0.7)
        self.play(ReplacementTransform(stubs[0], arcO_l),
                  ReplacementTransform(stubs[2], arcO_r),
                  ReplacementTransform(stubs[1], arcI_l),
                  ReplacementTransform(stubs[3], arcI_r),
                  run_time=1.5, rate_func=EASE)
        self.wait(0.8)

        # ---- Stage 3: the flatten triangles cancel; wires join up ----
        pO_l, pI_l = C + np.array([-rO, 0, 0]), C + np.array([-rI, 0, 0])
        pO_r, pI_r = C + np.array([rO, 0, 0]), C + np.array([rI, 0, 0])
        brO_l = ArcBetweenPoints(pO_l, lu, angle=0.7, color=INK, stroke_width=2.2)
        brI_l = ArcBetweenPoints(pI_l, ll, angle=-0.7, color=INK, stroke_width=2.2)
        brO_r = ArcBetweenPoints(ru, pO_r, angle=0.7, color=INK, stroke_width=2.2)
        brI_r = ArcBetweenPoints(rl, pI_r, angle=-0.7, color=INK, stroke_width=2.2)
        self.play(FadeOut(LT, RT), run_time=0.6)
        self.play(Create(brO_l), Create(brI_l), Create(brO_r), Create(brI_r),
                  run_time=0.9)
        self.wait(0.6)

        # ---- Stage 4: two closed curves that were never linked ----
        outer = VGroup(arcO_l, arcO_r, brO_l, brO_r, wAl, wAr, nA)
        inner = VGroup(arcI_l, arcI_r, brI_l, brI_r, wBl, wBr, nB)
        self.play(outer.animate.shift(np.array([-1.45, 0, 0])),
                  inner.animate.shift(np.array([1.45, 0, 0])),
                  run_time=2.4, rate_func=EASE)
        self.play(ReplacementTransform(t2, t3), run_time=0.9)
        self.wait(1.6)
