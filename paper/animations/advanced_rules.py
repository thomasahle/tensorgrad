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
        eaX = Line(ax + 0.25 * RIGHT, xx + 0.28 * LEFT, color=INK, stroke_width=2.2)
        eXb = Line(xx + 0.28 * RIGHT, bx + 0.25 * LEFT, color=INK, stroke_width=2.2)
        sa = Line(ax + 0.25 * LEFT, ax + 0.85 * LEFT, color=INK, stroke_width=2.2)
        sb = Line(bx + 0.25 * RIGHT, bx + 0.85 * RIGHT, color=INK, stroke_width=2.2)

        self.play(Write(lhs), run_time=1.0)
        self.play(FadeIn(na, nX, nb), Create(eaX), Create(eXb),
                  Create(sa), Create(sb), run_time=1.0)
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


class InverseSplit(Scene):
    def construct(self):
        lhs = MathTex(r"\mathrm{d}(X^{-1})", color=INK)
        eq = MathTex(r"=", color=INK)
        rhs = MathTex(r"-\,X^{-1}\, \mathrm{d}X\, X^{-1}", color=INK)
        title = VGroup(lhs, eq, rhs).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.7)

        # chain  --X^{-1}--  with derivative loop
        s = ValueTracker(0.0)          # half-separation of the two copies
        L, Rr = np.array([-2.4, 0, 0]), np.array([2.4, 0, 0])

        nl = always_redraw(lambda: node("X^{-1}",
                np.array([-s.get_value(), 0, 0])))
        nr = always_redraw(lambda: node("X^{-1}",
                np.array([+s.get_value(), 0, 0])))
        el = always_redraw(lambda: Line(
            L, np.array([-s.get_value() - 0.45, 0, 0]),
            color=INK, stroke_width=2.2))
        er = always_redraw(lambda: Line(
            np.array([+s.get_value() + 0.45, 0, 0]), Rr,
            color=INK, stroke_width=2.2))
        # the new derivative edges, growing in the gap
        def gap_edges():
            g = s.get_value()
            # delayed growth: tips never cross while the gap opens
            k = min(1.0, max(0.0, (g - 0.60) / 0.45))
            p1 = np.array([-g + 0.42, 0.10, 0])
            p2 = np.array([+g - 0.42, 0.10, 0])
            d1 = Line(p1, p1 + k * np.array([0.30, 0.34, 0]),
                      color=INK, stroke_width=2.2)
            d2 = Line(p2, p2 + k * np.array([-0.30, 0.34, 0]),
                      color=INK, stroke_width=2.2)
            return VGroup(d1, d2) if k > 0.03 else VGroup()
        dang = always_redraw(gap_edges)

        loop = dloop(ORIGIN, 1.6, 1.05)
        minus = MathTex(r"-", color=INK).scale(1.3).move_to(L + 0.7 * LEFT)

        self.play(Write(lhs), run_time=0.9)
        self.add(el, er, nl, nr)
        self.play(FadeIn(nl, nr), Create(el), Create(er), run_time=0.8)
        self.play(Create(loop[0]), FadeIn(loop[1], loop[2], loop[3]), run_time=1.0)
        self.wait(0.5)

        # mitosis: the loop is consumed, then the node splits and the
        # derivative edges rise in the gap
        self.play(FadeOut(loop), run_time=0.6)
        self.add(dang)
        self.play(s.animate.set_value(1.05),
                  FadeIn(minus, run_time=1.0),
                  run_time=2.2, rate_func=EASE)
        self.wait(0.4)
        self.play(Write(eq), Write(rhs), run_time=1.0)
        self.wait(1.4)


class KroneckerTrace(Scene):
    def construct(self):
        lhs = MathTex(r"\mathrm{Tr}(A \otimes B)", color=INK)
        eq = MathTex(r"=", color=INK)
        rhs = MathTex(r"\mathrm{Tr}(A)\,\mathrm{Tr}(B)", color=INK)
        title = VGroup(lhs, eq, rhs).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.7)

        # two wires of the bundle trace: concentric loops, A on the outer,
        # B on the inner, both at the top -- reading like the stacked pair.
        d = ValueTracker(0.0)  # horizontal separation
        def circA():
            t = d.get_value()
            r = 1.45 - 0.25 * min(1, t / 1.5)
            return Circle(radius=r, color=INK, stroke_width=2.2
                          ).move_to(np.array([-t, -0.2, 0]))
        def circB():
            t = d.get_value()
            r = 1.00 + 0.20 * min(1, t / 1.5)
            return Circle(radius=r, color=INK, stroke_width=2.2
                          ).move_to(np.array([+t, -0.2, 0]))
        cA = always_redraw(circA)
        cB = always_redraw(circB)
        def labA():
            t = d.get_value()
            r = 1.45 - 0.25 * min(1, t / 1.5)
            return node("A", np.array([-t, -0.2 + r, 0]))
        def labB():
            t = d.get_value()
            r = 1.00 + 0.20 * min(1, t / 1.5)
            return node("B", np.array([+t, -0.2 + r, 0]))
        nA = always_redraw(labA)
        nB = always_redraw(labB)

        self.play(Write(lhs), run_time=0.9)
        self.add(cA, cB, nA, nB)
        self.play(FadeIn(cA, cB, nA, nB), run_time=1.0)
        self.wait(0.6)

        # the two wires were never linked: pull them apart
        self.play(d.animate.set_value(1.75), run_time=2.6, rate_func=EASE)
        self.wait(0.3)
        self.play(Write(eq), Write(rhs), run_time=1.0)
        self.wait(1.4)
