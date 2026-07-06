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


def node(tex, pos, scale=1.1, color=INK):
    lbl = MathTex(tex, color=color).scale(scale).move_to(pos)
    mask = BackgroundRectangle(lbl, color=WHITE, fill_opacity=1.0, buff=0.09)
    return VGroup(mask, lbl)


def dloop(center, width, height, dot_angle_deg=55, long_whiskers=False,
          wscale=1.0, color=INK):
    """Penrose derivative loop: ellipse + dot on it + two bent whiskers
    (row whisker bending left, column whisker bending right).  With
    long_whiskers the two whiskers extend out to point horizontally left
    and right -- where the new edges will end up."""
    ell = Ellipse(width=width, height=height, color=color, stroke_width=2.2)
    ell.move_to(center)
    a = np.deg2rad(dot_angle_deg)
    p = center + np.array([width / 2 * np.cos(a), height / 2 * np.sin(a), 0])
    dot = Dot(p, radius=0.05, color=color)
    if long_whiskers:
        w = wscale
        wl = CubicBezier(p, p + w * np.array([-0.05, 0.28, 0]),
                         p + w * np.array([-0.60, 0.62, 0]),
                         p + w * np.array([-1.40, 0.60, 0]),
                         color=color, stroke_width=2.2)
        wr = CubicBezier(p, p + w * np.array([0.10, 0.24, 0]),
                         p + w * np.array([0.60, 0.52, 0]),
                         p + w * np.array([1.35, 0.49, 0]),
                         color=color, stroke_width=2.2)
    else:
        wl = ArcBetweenPoints(p, p + np.array([-0.28, 0.38, 0]), angle=-0.5,
                              color=color, stroke_width=2.2)
        wr = ArcBetweenPoints(p, p + np.array([0.40, 0.26, 0]), angle=0.5,
                              color=color, stroke_width=2.2)
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
    """Tr(A (x) B) = Tr(A) Tr(B), 3b1b-style.  The bundle is drawn as a
    blue strand (A) and a green strand (B) traveling together, so the
    whole proof reads as untangling: the flatten triangles cancel, the
    two colored loops were never linked, and they come apart into two
    traces.  Formula, captions, and emphasis in sync throughout."""

    def __init__(self, **kwargs):
        from manim import config as _cfg
        _cfg.frame_height = 5.6
        _cfg.frame_width = 5.6 * 16 / 9
        super().__init__(**kwargs)

    def construct(self):
        from manim import ParametricFunction, Polygon, Tex, Indicate, Circumscribe

        CA, CB = "#1F6FB2", "#188A54"
        GREY = "#888888"
        y0 = -0.55
        C = np.array([0, y0, 0])

        # ---- title templates (final layout, centered stages) ----
        tTrT = MathTex(r"\mathrm{Tr}(", "A", r"\otimes", "B", ")", color=INK)
        tTrT[1].set_color(CA)
        tTrT[3].set_color(CB)
        m2 = MathTex("=", r"\;\mathrm{Tr}(", "A", ")", r"\,\mathrm{Tr}(", "B", ")",
                     color=INK)
        m2[2].set_color(CA)
        m2[5].set_color(CB)
        toprow = VGroup(tTrT, m2).arrange(RIGHT, buff=0.28).to_edge(UP, buff=0.35)
        v0 = -tTrT.get_center()[0]
        tTrT.shift(RIGHT * v0)          # centered until the equation grows
        tpos = tTrT.get_center()

        tAoB = MathTex("A", r"\otimes", "B", color=INK).move_to(tpos)
        tAoB[0].set_color(CA)
        tAoB[2].set_color(CB)
        tTr = MathTex(r"\mathrm{Tr}(", "A", r"\otimes", "B", ")", color=INK
                      ).move_to(tpos)
        tTr[1].set_color(CA)
        tTr[3].set_color(CB)

        # ---- caption line ----
        self.cap = None

        def caption(text):
            new = Tex(text, color=GREY).scale(0.62).to_edge(DOWN, buff=0.18)
            anims = [FadeIn(new, run_time=0.6)]
            if self.cap is not None:
                anims.append(FadeOut(self.cap, run_time=0.4))
            self.cap = new
            return anims

        # ---- the two strands as parametric closed loops ----
        # A is the TOP strand of the bundle, so through every turn it
        # takes the INSIDE track: A = inner loop, B = outer loop, and
        # the strands never cross.
        KA = np.array([
            [0, y0, 1.88, 1.82, 1.20, -0.38],
            [0, y0 + 0.50, 1.00, 0.62, 0.95, 0.50],
            [0, 0.75, 0.80, 0.45, 0.80, 0.42],
        ])
        KB = np.array([
            [0, y0, 2.00, 1.97, 1.60, 0.55],
            [0, y0, 1.98, 1.94, 1.60, 1.30],
            [0, -0.75, 0.80, 0.45, 0.80, 0.42],
        ])
        tt = ValueTracker(1.0)

        def P(K):
            u = tt.get_value()
            return np.array([np.interp(u, [1, 2, 3], K[:, k]) for k in range(6)])

        def looppt(K, th):
            cx, cy, rtx, rty, rbx, rby = P(K)
            w = np.clip((np.sin(th) + 0.60) / 1.20, 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            rx = w * rtx + (1 - w) * rbx
            ry = w * rty + (1 - w) * rby
            return np.array([cx + rx * np.cos(th), cy + ry * np.sin(th), 0])

        def quad(K, th0, th1, col):
            return ParametricFunction(
                lambda u: looppt(K, th0 + u * (th1 - th0)), t_range=[0, 1],
                color=col, stroke_width=2.2)

        def gapd(K):
            # parameter half-gap so the wire clears its riding label
            rbx = P(K)[4]
            return np.clip(0.33 / max(rbx, 0.4), 0.18, 0.75)

        def mkloop(K, col):
            def build():
                d = gapd(K)
                return ParametricFunction(
                    lambda th: looppt(K, th),
                    t_range=[3 * np.pi / 2 + d, 3 * np.pi / 2 + 2 * np.pi - d],
                    color=col, stroke_width=2.2)
            return always_redraw(build)

        def glyph(tex, pos, color):
            m = MathTex(tex, color=color).scale(1.1).move_to(pos)
            m.set_background_stroke(color=WHITE, width=6)
            return m

        # ---- the closed-bundle ribbon: one center path, two offset
        #      strands (constant separation, so they stay parallel) ----
        gap = 0.09                       # node clearance
        ax_gap = 1.85 + gap              # bundle attaches here
        r0, RX, RY = 0.35, ax_gap + r0 if False else 0.35, 0.0  # placeholders
        r0 = 0.35
        RX, RY = ax_gap + r0, 1.55
        cT = np.array([0, y0 + r0, 0])

        def ribbon_path(u):
            # u in [0,1]: left foot curl, elliptical dome, right foot curl
            if u < 0.15:
                a = -np.pi / 2 - (u / 0.15) * (np.pi / 2)
                c = np.array([-ax_gap, y0 + r0, 0])
                return c + r0 * np.array([np.cos(a), np.sin(a), 0])
            if u > 0.85:
                a = -np.pi / 2 + ((1 - u) / 0.15) * (np.pi / 2)
                c = np.array([ax_gap, y0 + r0, 0])
                return c + r0 * np.array([np.cos(a), np.sin(a), 0])
            a = np.pi - ((u - 0.15) / 0.70) * np.pi
            return cT + np.array([RX * np.cos(a), RY * np.sin(a), 0])

        def strand(u0, u1, off, col):
            eps = 1e-4

            def f(v):
                u = u0 + v * (u1 - u0)
                p1 = ribbon_path(np.clip(u - eps, 0, 1))
                p2 = ribbon_path(np.clip(u + eps, 0, 1))
                t = p2 - p1
                t = t / (np.linalg.norm(t) + 1e-12)
                n = np.array([-t[1], t[0], 0])
                return ribbon_path(u) + off * n

            return ParametricFunction(f, t_range=[0, 1], color=col,
                                      stroke_width=2.2)

        # ---- regime 1: the flat definition ----
        pA0 = np.array([0, y0 + 0.55, 0])
        pB0 = np.array([0, y0 - 0.55, 0])
        nA = glyph("A", pA0, CA)
        nB = glyph("B", pB0, CB)
        apxL, apxR = np.array([-1.85, y0, 0]), np.array([1.85, y0, 0])
        cUL, cLL = np.array([-1.35, y0 + 0.20, 0]), np.array([-1.35, y0 - 0.20, 0])
        cUR, cLR = np.array([1.35, y0 + 0.20, 0]), np.array([1.35, y0 - 0.20, 0])
        triL = Polygon(apxL, cUL, cLL, color=INK, stroke_width=2.2,
                       fill_color=WHITE, fill_opacity=1.0)
        triR = Polygon(apxR, cUR, cLR, color=INK, stroke_width=2.2,
                       fill_color=WHITE, fill_opacity=1.0)

        def cleared(p, q, d=0.09):
            v = q - p
            v = v / np.linalg.norm(v)
            return p + d * v

        bA = np.array([0.31, 0, 0])
        wAl = Line(cleared(cUL, pA0), pA0 - bA, color=CA, stroke_width=2.2)
        wAr = Line(pA0 + bA, cleared(cUR, pA0), color=CA, stroke_width=2.2)
        wBl = Line(cleared(cLL, pB0), pB0 - bA, color=CB, stroke_width=2.2)
        wBr = Line(pB0 + bA, cleared(cLR, pB0), color=CB, stroke_width=2.2)
        dy = 0.006   # zero-gap double wire: the strands touch until resolved
        sAl = Line(np.array([-ax_gap, y0 + dy, 0]),
                   np.array([-2.62, y0 + dy, 0]), color=CA, stroke_width=2.2)
        sBl = Line(np.array([-ax_gap, y0 - dy, 0]),
                   np.array([-2.62, y0 - dy, 0]), color=CB, stroke_width=2.2)
        sAr = Line(np.array([ax_gap, y0 + dy, 0]),
                   np.array([2.62, y0 + dy, 0]), color=CA, stroke_width=2.2)
        sBr = Line(np.array([ax_gap, y0 - dy, 0]),
                   np.array([2.62, y0 - dy, 0]), color=CB, stroke_width=2.2)

        # z-order: wires and stubs, then triangles, then labels, then text
        self.play(Create(VGroup(wAl, wAr, wBl, wBr)),
                  Create(VGroup(sAl, sBl, sAr, sBr)),
                  FadeIn(triL, triR),
                  FadeIn(nA, nB),
                  Write(tAoB),
                  *caption(r"the Kronecker product bundles two matrices"),
                  run_time=1.3)
        self.wait(0.9)

        # ---- close the bundle over the top (anchored end first);
        #      A rides the inside of the turn, B the outside ----
        arcAl = strand(0.0, 0.5, -dy, CA)
        arcAr = strand(1.0, 0.5, -dy, CA)
        arcBl = strand(0.0, 0.5, +dy, CB)
        arcBr = strand(1.0, 0.5, +dy, CB)
        self.play(ReplacementTransform(tAoB[0], tTr[1]),
                  ReplacementTransform(tAoB[1], tTr[2]),
                  ReplacementTransform(tAoB[2], tTr[3]),
                  FadeIn(tTr[0], tTr[4]),
                  Transform(sAl, arcAl),
                  Transform(sAr, arcAr),
                  Transform(sBl, arcBl),
                  Transform(sBr, arcBr),
                  *caption(r"closing the bundle takes the trace"),
                  run_time=1.4, rate_func=EASE)
        self.wait(0.8)

        # ---- the flatten triangles cancel; wires relax into two loops ----
        dA0, dB0 = gapd(KA), gapd(KB)
        qAl = quad(KA, np.pi, 3 * np.pi / 2 - dA0, CA)
        qAr = quad(KA, 3 * np.pi / 2 + dA0, 2 * np.pi, CA)
        qBl = quad(KB, np.pi, 3 * np.pi / 2 - dB0, CB)
        qBr = quad(KB, 3 * np.pi / 2 + dB0, 2 * np.pi, CB)
        tAl = quad(KA, np.pi, np.pi / 2, CA)
        tAr = quad(KA, 0, np.pi / 2, CA)
        tBl = quad(KB, np.pi, np.pi / 2, CB)
        tBr = quad(KB, 0, np.pi / 2, CB)
        self.play(FadeOut(triL, scale=0.4), FadeOut(triR, scale=0.4),
                  Transform(wAl, qAl), Transform(wAr, qAr),
                  Transform(wBl, qBl), Transform(wBr, qBr),
                  Transform(sAl, tAl), Transform(sAr, tAr),
                  Transform(sBl, tBl), Transform(sBr, tBr),
                  nA.animate.move_to(looppt(KA, 3 * np.pi / 2)),
                  nB.animate.move_to(looppt(KB, 3 * np.pi / 2)),
                  *caption(r"the flatten triangles cancel"),
                  run_time=1.2, rate_func=EASE)
        loopA, loopB = mkloop(KA, CA), mkloop(KB, CB)
        labA = always_redraw(lambda: glyph("A", looppt(KA, 3 * np.pi / 2), CA))
        labB = always_redraw(lambda: glyph("B", looppt(KB, 3 * np.pi / 2), CB))
        self.remove(sAl, sAr, sBl, sBr, wAl, wAr, wBl, wBr, nA, nB)
        self.add(loopA, loopB, labA, labB)
        self.wait(0.5)

        # ---- the two loops were never linked ----
        self.play(tt.animate.set_value(2.0),
                  *caption(r"the two loops were never linked"),
                  run_time=2.0, rate_func=EASE)
        self.wait(0.5)

        # ---- pull them apart: two separate traces ----
        self.play(tt.animate.set_value(3.0),
                  VGroup(tTr).animate.shift(RIGHT * (-v0)),
                  *caption(r"each loop is its own trace"),
                  run_time=2.0, rate_func=EASE)
        self.play(Write(m2[0]), FadeIn(*m2[1:]), run_time=0.7)
        self.play(Indicate(VGroup(*m2[1:]), color=INK, scale_factor=1.1),
                  run_time=0.7)
        self.play(Circumscribe(VGroup(*m2[1:]), color=INK, buff=0.12),
                  FadeOut(self.cap), run_time=1.0)
        self.wait(2.6)


class TraceDelete(Scene):
    """d Tr(AXB)/dX = A^T B^T, 3b1b-style: semantic color (A blue, B
    green, X red, derivative apparatus amber -- the whiskers' promise
    becomes the answer's outer edges), a quiet caption line narrating
    each phase, and emphasis beats on the results."""

    def __init__(self, **kwargs):
        from manim import config as _cfg
        _cfg.frame_height = 5.6
        _cfg.frame_width = 5.6 * 16 / 9
        super().__init__(**kwargs)

    def construct(self):
        from manim import ArcBetweenPoints as ABP
        from manim import Rotate, PI, Tex, Indicate, Circumscribe

        CA, CB, CX, CD = "#1F6FB2", "#188A54", "#C03B2B", "#B07300"
        GREY = "#888888"

        # ---- hand-built title fraction so the partial can glide ----
        num0 = MathTex(r"\partial", r"\,\mathrm{Tr}(", "A", "X", "B", ")",
                       color=INK)
        for i, c in [(0, CD), (2, CA), (3, CX), (4, CB)]:
            num0[i].set_color(c)
        den = MathTex(r"\partial", "X", color=INK)
        den[0].set_color(CD)
        den[1].set_color(CX)
        bar0 = Line(ORIGIN, RIGHT * (num0.width + 0.2), color=INK,
                    stroke_width=1.6)
        frac = VGroup(num0, bar0, den).arrange(DOWN, buff=0.14)
        eq1 = MathTex("=", color=INK)
        m1 = MathTex("(", "B", "A", ")^T", color=INK)
        m1[1].set_color(CB)
        m1[2].set_color(CA)
        eq2 = MathTex("=", color=INK)
        m2 = MathTex(r"A^T\!", "B^T", color=INK)
        m2[0].set_color(CA)
        m2[1].set_color(CB)
        title = VGroup(frac, eq1, m1, eq2, m2).arrange(RIGHT, buff=0.28)
        title.to_edge(UP, buff=0.35)
        v0 = -frac.get_center()[0]
        v1 = -VGroup(frac, eq1, m1).get_center()[0]
        frac.shift(RIGHT * v0)
        eq1.shift(RIGHT * v1)
        m1.shift(RIGHT * v1)
        tpos = frac.get_center()

        # ---- caption line ----
        self.cap = None

        def caption(text):
            new = Tex(text, color=GREY).scale(0.62).to_edge(DOWN, buff=0.18)
            anims = [FadeIn(new, run_time=0.6)]
            if self.cap is not None:
                anims.append(FadeOut(self.cap, run_time=0.4))
            self.cap = new
            return anims

        ca, cb, cX = np.array([-1.15, -1.3, 0]), np.array([1.15, -1.3, 0]), np.array([0, -1.3, 0])
        phi = ValueTracker(0.0)
        kap = ValueTracker(1.0)

        def dirv(deg):
            r = np.deg2rad(deg)
            return np.array([np.cos(r), np.sin(r), 0])

        def dome():
            u = phi.get_value() / 180.0
            aA, aB = 180 - phi.get_value(), 0 + phi.get_value()
            p1 = ca + 0.30 * dirv(aA)
            p2 = cb + 0.30 * dirv(aB)
            lift = (1.55 * (1 - u) + 0.06) * np.array([0, 1, 0])
            c1 = p1 + (1.25 * (1 - u) + 0.12) * dirv(aA) + lift
            c2 = p2 + (1.25 * (1 - u) + 0.12) * dirv(aB) + lift
            return CubicBezier(p1, c1, c2, p2, color=INK, stroke_width=2.2)

        def hook(c, base):
            th = base - phi.get_value() if base == 0 else base + phi.get_value()
            p = c + 0.30 * dirv(th)
            k = kap.get_value()
            return ABP(p, p + (0.42 + 0.13 * (1 - k)) * dirv(th + 55 * k),
                       angle=2.2 * k, color=CD, stroke_width=2.2)

        def lab(tex, c, sgn, col):
            m = MathTex(tex, color=col).scale(1.1)
            m.rotate(sgn * np.deg2rad(phi.get_value())).move_to(c)
            mask = BackgroundRectangle(m, color=WHITE, fill_opacity=1.0, buff=0.08)
            return VGroup(mask, m)

        domeM = always_redraw(dome)
        nA = always_redraw(lambda: lab("A", ca, -1, CA))
        nB = always_redraw(lambda: lab("B", cb, +1, CB))
        nX = node("X", cX, color=CX)
        wAX = Line(ca + 0.30 * RIGHT, cX + 0.30 * LEFT, color=INK, stroke_width=2.2)
        wXB = Line(cX + 0.30 * RIGHT, cb + 0.30 * LEFT, color=INK, stroke_width=2.2)

        # ---- build the formula and the diagram together ----
        tX = MathTex("X", color=CX).move_to(tpos)
        tAXB = MathTex("A", "X", "B", color=INK).move_to(tpos)
        tAXB[0].set_color(CA)
        tAXB[1].set_color(CX)
        tAXB[2].set_color(CB)
        tTr = MathTex(r"\mathrm{Tr}(", "A", "X", "B", ")", color=INK).move_to(tpos)
        tTr[1].set_color(CA)
        tTr[2].set_color(CX)
        tTr[3].set_color(CB)
        stubL = Line(cX + 0.30 * LEFT, cX + 1.05 * LEFT, color=INK, stroke_width=2.2)
        stubR = Line(cX + 0.30 * RIGHT, cX + 1.05 * RIGHT, color=INK, stroke_width=2.2)
        self.play(Write(tX), FadeIn(nX), Create(stubL), Create(stubR),
                  *caption(r"a matrix is a node with two edges"), run_time=0.9)
        self.wait(0.7)

        soA = Line(ca + 0.30 * LEFT, ca + 1.0 * LEFT, color=INK, stroke_width=2.2)
        soB = Line(cb + 0.30 * RIGHT, cb + 1.0 * RIGHT, color=INK, stroke_width=2.2)
        self.add(nA, nB)
        wAXr = Line(cX + 0.30 * LEFT, ca + 0.30 * RIGHT, color=INK,
                    stroke_width=2.2)
        self.play(ReplacementTransform(tX, tAXB[1]), FadeIn(tAXB[0], tAXB[2]),
                  FadeIn(nA, nB),
                  ReplacementTransform(stubL, wAXr),
                  ReplacementTransform(stubR, wXB),
                  Create(soA), Create(soB),
                  *caption(r"a shared edge is a matrix product"), run_time=1.1)
        self.remove(wAXr)
        self.add(wAX)
        self.wait(0.7)

        p1 = ca + 0.30 * dirv(180)
        p2 = cb + 0.30 * dirv(0)
        lift = 1.61 * np.array([0, 1, 0])
        c1 = p1 + 1.37 * dirv(180) + lift
        c2 = p2 + 1.37 * dirv(0) + lift
        apex = (p1 + 3 * c1 + 3 * c2 + p2) / 8
        domeL = CubicBezier(p1, (p1 + c1) / 2, (p1 + 2 * c1 + c2) / 4, apex,
                            color=INK, stroke_width=2.2)
        domeR = CubicBezier(p2, (c2 + p2) / 2, (c1 + 2 * c2 + p2) / 4, apex,
                            color=INK, stroke_width=2.2)
        self.play(ReplacementTransform(tAXB[0], tTr[1]),
                  ReplacementTransform(tAXB[1], tTr[2]),
                  ReplacementTransform(tAXB[2], tTr[3]),
                  FadeIn(tTr[0], tTr[4]),
                  ReplacementTransform(soA, domeL),
                  ReplacementTransform(soB, domeR),
                  *caption(r"closing the loop takes the trace"),
                  run_time=1.2, rate_func=EASE)
        self.remove(domeL, domeR)
        self.add(domeM)
        self.wait(0.7)

        tr_target = VGroup(*num0[1:])
        big = dloop(np.array([0, -0.95, 0]), 5.6, 2.9, dot_angle_deg=72,
                    long_whiskers=True, color=CD)
        self.play(tTr.animate.move_to(tr_target.get_center()
                  ).scale(tr_target.height / tTr.height),
                  FadeIn(num0[0], bar0, den),
                  Create(big[0]), FadeIn(big[1], big[2], big[3]),
                  *caption(r"the derivative promises two new edges"),
                  run_time=1.2)
        self.remove(tTr)
        self.add(num0)
        self.wait(0.8)

        # ---- the loop localizes to X; the partial slides inside too ----
        num1 = MathTex(r"\mathrm{Tr}(", "A", r"\,\partial", "X", r"\,B", ")",
                       color=INK).move_to(num0.get_center())
        num1[1].set_color(CA)
        num1[2].set_color(CD)
        num1[3].set_color(CX)
        num1[4].set_color(CB)
        bar1 = Line(ORIGIN, RIGHT * (num1.width + 0.2), color=INK,
                    stroke_width=1.6).move_to(bar0.get_center())
        small = dloop(cX + np.array([0, 0.05, 0]), 1.05, 1.0, dot_angle_deg=72,
                      long_whiskers=True, wscale=0.6, color=CD)
        self.play(Transform(big, small),
                  ReplacementTransform(num0[0], num1[2], path_arc=-1.4),
                  ReplacementTransform(num0[1], num1[0]),
                  ReplacementTransform(num0[2], num1[1]),
                  ReplacementTransform(num0[3], num1[3]),
                  ReplacementTransform(num0[4], num1[4]),
                  ReplacementTransform(num0[5], num1[5]),
                  Transform(bar0, bar1),
                  *caption(r"linearity: it tightens around the only $X$"),
                  run_time=1.6, rate_func=EASE)
        self.wait(0.7)

        # ---- delete the node; the remaining diagram IS (BA)^T ----
        hookA = always_redraw(lambda: hook(ca, 0))
        hookB = always_redraw(lambda: hook(cb, 180))
        hA0, hB0 = hook(ca, 0), hook(cb, 180)
        self.play(FadeOut(nX, scale=0.4), FadeOut(big),
                  ReplacementTransform(wAX, hA0),
                  ReplacementTransform(wXB, hB0),
                  VGroup(num1, bar0, den).animate.shift(RIGHT * (v1 - v0)),
                  Write(eq1), FadeIn(m1),
                  *caption(r"delete the node; its edges dangle"),
                  run_time=1.2, rate_func=EASE)
        self.remove(hA0, hB0)
        self.add(hookA, hookB)
        self.play(Indicate(m1, color=CD, scale_factor=1.12), run_time=0.7)
        self.wait(0.5)

        # ---- the arc pulls straight; the nodes turn around ----
        self.play(phi.animate.set_value(180), kap.animate.set_value(0.02),
                  *caption(r"straightening the loop turns $A$ and $B$ around"),
                  run_time=3.2, rate_func=EASE)
        self.wait(0.5)

        # ---- the glyphs spin upright as the ^T (and A^T B^T) appear ----
        finA = MathTex("A", "^{T}", color=CA).scale(1.1)
        finA.shift(ca - finA[0].get_center())
        finB = MathTex("B", "^{T}", color=CB).scale(1.1)
        finB.shift(cb - finB[0].get_center())
        rotA = MathTex("A", color=CA).scale(1.1).move_to(ca).rotate(-PI)
        rotB = MathTex("B", color=CB).scale(1.1).move_to(cb).rotate(PI)
        self.remove(nA, nB)
        self.add(rotA, rotB)
        self.play(Rotate(rotA, angle=-PI, about_point=ca),
                  Rotate(rotB, angle=PI, about_point=cb),
                  FadeIn(finA[1], finB[1]),
                  VGroup(num1, bar0, den, eq1, m1).animate.shift(RIGHT * (-v1)),
                  Write(eq2), FadeIn(m2),
                  *caption(r"a turned matrix is its transpose"),
                  run_time=1.1, rate_func=EASE)
        self.remove(rotA, rotB)
        maskA = BackgroundRectangle(finA, color=WHITE, fill_opacity=1.0, buff=0.04)
        maskB = BackgroundRectangle(finB, color=WHITE, fill_opacity=1.0, buff=0.04)
        self.add(maskA, maskB, finA, finB)
        self.wait(0.4)

        # settle: the finished diagram floats up under its equation
        domeS, hA_s, hB_s = dome(), hook(ca, 0), hook(cb, 180)
        self.remove(domeM, hookA, hookB)
        self.add(domeS, hA_s, hB_s, maskA, maskB, finA, finB)
        self.play(VGroup(domeS, hA_s, hB_s, maskA, maskB, finA, finB
                         ).animate.shift(1.55 * UP),
                  FadeOut(self.cap),
                  run_time=0.9, rate_func=EASE)
        self.play(Circumscribe(m2, color=CD, buff=0.12), run_time=1.0)
        self.wait(1.8)


class CrossEntropyHessian(Scene):
    """The Hessian of cross-entropy = the softmax Jacobian, in the
    book's function notation: function names are plain nodes, input
    edges carry arrows (dotted = elementwise, with the data wire
    staying on the argument), sums are copydot-terminated wires, and
    the chain rule circles the differentiated node and chains d-edges
    through copies -- 'the chain rule is actually a chain'."""

    def __init__(self, **kwargs):
        from manim import config as _cfg
        _cfg.frame_height = 5.6
        _cfg.frame_width = 5.6 * 16 / 9
        super().__init__(**kwargs)

    def construct(self):
        from manim import Circle as MCircle
        from manim import DashedLine, Tex, Indicate, Circumscribe

        CA, CB, CX, CD = "#1F6FB2", "#188A54", "#C03B2B", "#B07000"
        GREY = "#888888"

        def glyph(tex, pos, color, sc=1.0):
            m = MathTex(tex, color=color).scale(sc).move_to(pos)
            m.set_background_stroke(color=WHITE, width=6)
            return m

        def wire(p, q, col=INK):
            return Line(p, q, color=col, stroke_width=2.2)

        def farrow(p, q, dotted=True):
            """Book-style function-application edge: arrow INTO the
            function; densely dotted when elementwise."""
            if dotted:
                ln = DashedLine(p, q, dash_length=0.055, color=INK,
                                stroke_width=1.9)
            else:
                ln = Line(p, q, color=INK, stroke_width=1.9)
            ln.add_tip(tip_length=0.13, tip_width=0.10)
            ln.get_tip().set_fill(INK, 1).set_stroke(INK, 0)
            return ln

        def cdot(pos):
            return Dot(pos, radius=0.055, color=INK)

        def dcircle(mob, extra=0.10):
            return MCircle(radius=max(mob.width, mob.height) / 2 + extra,
                           color=CD, stroke_width=2.2).move_to(mob)

        def dedge(p, q):
            """Derivative chain edge, book d0 style: open circle tip
            at the departure end."""
            tip = MCircle(radius=0.045, color=CD, stroke_width=2.0).move_to(p)
            ln = Line(p + 0.045 * RIGHT, q, color=CD, stroke_width=2.0)
            return VGroup(tip, ln)

        # factor builders (origin = position of z)
        def F1(zp, sc=1.0):
            """exp <... z --- (free data wire on z)"""
            d = {}
            d['z'] = glyph("z", zp, CX, 0.95 * sc)
            d['e'] = glyph(r"\exp", zp + np.array([-1.0 * sc, 0, 0]), CA, 0.8 * sc)
            d['a'] = farrow(zp + np.array([-0.22 * sc, 0, 0]),
                            zp + np.array([-0.62 * sc, 0, 0]))
            d['w'] = wire(zp + np.array([0.20 * sc, 0, 0]),
                          zp + np.array([0.78 * sc, 0, 0]))
            return d

        def F2(pp, sc=1.0):
            """pow_{-1} <... exp <... z ---.   (sum-dot on z's wire)"""
            d = {}
            d['pw'] = glyph(r"\mathrm{pow}_{-1}", pp, CA, 0.8 * sc)
            d['e'] = glyph(r"\exp", pp + np.array([1.4 * sc, 0, 0]), CA, 0.8 * sc)
            d['z'] = glyph("z", pp + np.array([2.25 * sc, 0, 0]), CX, 0.95 * sc)
            d['a1'] = farrow(pp + np.array([2.03 * sc, 0, 0]),
                             pp + np.array([1.77 * sc, 0, 0]))
            d['w'] = wire(pp + np.array([2.45 * sc, 0, 0]),
                          pp + np.array([3.0 * sc, 0, 0]))
            d['s'] = cdot(pp + np.array([3.05 * sc, 0, 0]))
            d['a2'] = farrow(pp + np.array([1.02 * sc, 0.0, 0]),
                             pp + np.array([0.68 * sc, 0, 0]))
            return d

        def fgroup(d):
            return VGroup(*[d[k] for k in d])

        # ---- title: softmax(z) throughout, no abbreviation ----
        num_t = MathTex(r"\partial\,", r"\mathrm{softmax}(", "z", ")",
                        color=INK).scale(0.8)
        num_t[0].set_color(CD)
        num_t[1].set_color(CB)
        num_t[2].set_color(CX)
        num_t[3].set_color(CB)
        den_t = MathTex(r"\partial", "z", color=INK).scale(0.8)
        den_t[0].set_color(CD)
        den_t[1].set_color(CX)
        tbar = Line(ORIGIN, RIGHT * (num_t.width + 0.2), color=INK,
                    stroke_width=1.6)
        tfrac = VGroup(num_t, tbar, den_t).arrange(DOWN, buff=0.13)
        m2 = MathTex("=", r"\;\mathrm{diag}(\mathrm{softmax}(", "z", "))",
                     "-", r"\mathrm{softmax}(", "z", ")",
                     r"\,\mathrm{softmax}(", "z", r")^T",
                     color=INK).scale(0.66)
        for i in (1, 3, 5, 7, 8, 10):
            m2[i].set_color(CB)
        for i in (2, 6, 9):
            m2[i].set_color(CX)
        VGroup(tfrac, m2).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.4)
        v0 = -tfrac.get_center()[0]
        tfrac.shift(RIGHT * v0)
        tpos = tfrac.get_center()
        t0 = MathTex(r"\mathrm{softmax}(", "z", ")", color=INK
                     ).move_to(tpos)
        t0[0].set_color(CB)
        t0[1].set_color(CX)
        t0[2].set_color(CB)

        self.cap = None

        def caption(text):
            new = Tex(text, color=GREY).scale(0.62).to_edge(DOWN, buff=0.18)
            anims = [FadeIn(new, run_time=0.6)]
            if self.cap is not None:
                anims.append(FadeOut(self.cap, run_time=0.4))
            self.cap = new
            return anims

        # ============ stage 1: softmax from elementary functions ============
        yC = -0.55
        A0 = F1(np.array([-2.9, yC, 0]))
        B0 = F2(np.array([-0.2, yC, 0]))
        # -- the definition as a fraction, first --
        FN = F1(np.array([0.0, 0.35, 0]))
        FD = F1(np.array([0.0, -1.4, 0]), 0.85)
        FDs = cdot(np.array([0.71, -1.4, 0]))
        bar0 = Line([-1.6, yC, 0], [1.6, yC, 0], color=INK, stroke_width=1.9)
        self.play(Write(t0), FadeIn(fgroup(FN), fgroup(FD), FDs),
                  Create(bar0),
                  *caption(r"$\mathrm{softmax}$: exponentiate, "
                           r"then normalize"),
                  run_time=1.3)
        self.wait(0.8)

        # -- dividing by the sum is a function too: pow_{-1} --
        self.play(*[ReplacementTransform(FN[k], A0[k]) for k in FN],
                  ReplacementTransform(FD['e'], B0['e']),
                  ReplacementTransform(FD['z'], B0['z']),
                  ReplacementTransform(FD['a'], B0['a1']),
                  ReplacementTransform(FD['w'], B0['w']),
                  ReplacementTransform(FDs, B0['s']),
                  FadeOut(bar0, target_position=[-0.65, yC, 0]),
                  FadeIn(B0['pw'], B0['a2'],
                         target_position=bar0.get_center()),
                  *caption(r"dividing by the sum is a function: "
                           r"$\mathrm{pow}_{-1}$"),
                  run_time=1.3, rate_func=EASE)
        self.wait(0.8)

        # ============ stage 2: differentiate ============
        self.play(ReplacementTransform(t0[0], num_t[1]),
                  ReplacementTransform(t0[1], num_t[2]),
                  ReplacementTransform(t0[2], num_t[3]),
                  FadeIn(num_t[0], tbar, den_t),
                  run_time=0.9)
        loop = Ellipse(width=8.2, height=1.9, color=CD, stroke_width=2.2
                       ).move_to([-0.2, yC, 0])
        ldot = Dot([-0.2, yC + 0.95, 0], radius=0.05, color=CD)
        w0 = ldot.get_center()
        whis = CubicBezier(w0, w0 + np.array([-0.4, 0.22, 0]),
                           w0 + np.array([-1.0, 0.3, 0]),
                           w0 + np.array([-1.6, 0.28, 0]),
                           color=CD, stroke_width=2.2)
        self.play(Create(loop), FadeIn(ldot), Create(whis),
                  *caption(r"differentiate: one new edge $j$"),
                  run_time=1.1)
        self.wait(0.8)

        # ============ stage 3: product rule -- two rows ============
        yA, yB = 0.62, -1.55
        A1 = F1(np.array([-2.7, yA, 0]))
        B1 = F2(np.array([-0.3, yA, 0]))
        loopA = Ellipse(width=2.5, height=1.15, color=CD, stroke_width=2.2
                        ).move_to([-3.05, yA, 0])
        ldotA = Dot([-3.05, yA + 0.575, 0], radius=0.05, color=CD)
        wA0 = ldotA.get_center()
        whisA = CubicBezier(wA0, wA0 + np.array([-0.3, 0.18, 0]),
                            wA0 + np.array([-0.7, 0.24, 0]),
                            wA0 + np.array([-1.05, 0.22, 0]),
                            color=CD, stroke_width=2.2)
        plusA = MathTex("+", color=INK).scale(1.15).move_to([0, (yA + yB) / 2, 0])
        A2 = F1(np.array([-3.5, yB, 0]))
        B2 = F2(np.array([-1.1, yB, 0]))
        loopB = Ellipse(width=4.0, height=1.15, color=CD, stroke_width=2.2
                        ).move_to([0.1, yB, 0])
        ldotB = Dot([0.1, yB + 0.575, 0], radius=0.05, color=CD)
        wB0 = ldotB.get_center()
        whisB = CubicBezier(wB0, wB0 + np.array([-0.3, 0.18, 0]),
                            wB0 + np.array([-0.7, 0.24, 0]),
                            wB0 + np.array([-1.05, 0.22, 0]),
                            color=CD, stroke_width=2.2)
        A0c = {k: A0[k].copy() for k in A0}
        B0c = {k: B0[k].copy() for k in B0}
        loopc, ldotc, whisc = loop.copy(), ldot.copy(), whis.copy()
        self.add(*A0c.values(), *B0c.values(), loopc, ldotc, whisc)
        self.play(
            *[ReplacementTransform(A0[k], A1[k]) for k in A0],
            *[ReplacementTransform(B0[k], B1[k]) for k in B0],
            Transform(loop, loopA), Transform(ldot, ldotA),
            Transform(whis, whisA),
            Write(plusA),
            *[ReplacementTransform(A0c[k], A2[k]) for k in A0c],
            *[ReplacementTransform(B0c[k], B2[k]) for k in B0c],
            ReplacementTransform(loopc, loopB),
            ReplacementTransform(ldotc, ldotB),
            ReplacementTransform(whisc, whisB),
            *caption(r"product rule: $z$ appears in both factors"),
            run_time=1.8, rate_func=EASE)
        self.wait(1.0)

        # ===== stage 4: row A -- elementwise derivative, copy-dot =====
        circA = dcircle(A1['e'])
        cdA = cdot(np.array([-2.2, yA, 0]))
        jA = wire(np.array([-2.24, yA + 0.04, 0]),
                  np.array([-2.72, yA + 0.62, 0]), CD)
        self.play(FadeOut(loop, ldot),
                  Create(circA),
                  Transform(whis, jA), FadeIn(cdA),
                  *caption(r"elementwise derivative: circle the function, "
                           r"copy-dot on the wire"),
                  run_time=1.5, rate_func=EASE)
        self.wait(0.6)
        self.play(FadeOut(circA),
                  *caption(r"evaluate: $\exp' = \exp$ --- "
                           r"the circle is spent"),
                  run_time=0.8)
        self.wait(0.5)

        # ===== stage 5: row B -- the chain rule is actually a chain =====
        circB = dcircle(B2['pw'])
        G3z = np.array([3.85, yB, 0])
        G3 = {}
        G3['z'] = glyph("z", G3z, CX, 0.85)
        G3['e'] = glyph(r"\exp", G3z + np.array([0.78, 0, 0]), CA, 0.65)
        G3['a'] = farrow(G3z + np.array([0.2, 0, 0]),
                         G3z + np.array([0.49, 0, 0]))
        G3['w'] = wire(G3z + np.array([-0.16, 0, 0]),
                       G3z + np.array([-0.62, 0, 0]))
        G3['s'] = cdot(G3z + np.array([-0.67, 0, 0]))
        loopG = Ellipse(width=2.15, height=1.0, color=CD, stroke_width=2.2
                        ).move_to(G3z + np.array([0.12, 0, 0]))
        ldotG = Dot(loopG.get_center() + np.array([-0.55, 0.42, 0]),
                    radius=0.05, color=CD)
        wG0 = ldotG.get_center()
        whisG = CubicBezier(wG0, wG0 + np.array([-0.25, 0.16, 0]),
                            wG0 + np.array([-0.55, 0.2, 0]),
                            wG0 + np.array([-0.85, 0.19, 0]),
                            color=CD, stroke_width=2.2)
        G3src = {k: B2[k if k != 'a' else 'a1'].copy()
                 for k in ('z', 'e', 'a', 'w', 's')}
        self.add(*G3src.values())
        self.play(Create(circB),
                  Transform(loopB, loopG),
                  Transform(ldotB, ldotG),
                  Transform(whisB, whisG),
                  *[ReplacementTransform(G3src[k], G3[k]) for k in G3src],
                  *caption(r"the chain rule is actually a chain"),
                  run_time=1.6, rate_func=EASE)
        self.wait(0.9)

        # pow_{-1}' = -pow_{-2}: the minus is born
        pw2 = glyph(r"\mathrm{pow}_{-2}", B2['pw'].get_center(), CA, 0.8)
        minusT = MathTex("-", color=INK).scale(1.15).move_to(plusA.get_center())
        self.play(ReplacementTransform(B2['pw'], pw2),
                  FadeOut(circB),
                  Transform(plusA, minusT),
                  *caption(r"evaluate: $\mathrm{pow}_{-1}' = "
                           r"-\,\mathrm{pow}_{-2}$ --- the minus is born"),
                  run_time=1.1, rate_func=EASE)
        self.wait(0.8)

        # the inner derivative: copy-dot appears, sum-dot absorbs it
        circG = dcircle(G3['e'])
        cdG = cdot(G3z + np.array([-0.4, 0, 0]))
        jG0 = wire(G3z + np.array([-0.44, 0.04, 0]),
                   G3z + np.array([-0.92, 0.62, 0]), CD)
        self.play(FadeOut(loopB, ldotB), Create(circG), FadeIn(cdG),
                  Transform(whisB, jG0),
                  *caption(r"and again for the inner chain"),
                  run_time=1.1)
        self.wait(0.5)
        self.play(FadeOut(circG),
                  G3['s'].animate.move_to(cdG.get_center()),
                  Transform(G3['w'], wire(G3z + np.array([-0.16, 0, 0]),
                                          G3z + np.array([-0.37, 0, 0]))),
                  *caption(r"$\exp' = \exp$; the sum slides into "
                           r"the copy-dot"),
                  run_time=1.0, rate_func=EASE)
        self.wait(0.4)
        jG1 = wire(G3z + np.array([-0.16, 0, 0]),
                   G3z + np.array([-0.8, 0, 0]), CD)
        self.play(FadeOut(G3['s'], scale=0.3), FadeOut(cdG, scale=0.3),
                  ReplacementTransform(VGroup(G3['w'], whisB), jG1),
                  *caption(r"$\textstyle\sum_k \delta_{jk}\exp(z)_k"
                           r" = \exp(z)_j$"),
                  run_time=0.9, rate_func=EASE)
        self.wait(0.7)

        # ==== stage 6a: the pow's are fractions ====
        gB1 = VGroup(B1['e'], B1['a1'], B1['z'], B1['w'], B1['s'])
        bar1 = Line([-4.0, yA - 0.5, 0], [-1.4, yA - 0.5, 0],
                    color=INK, stroke_width=1.9)
        gB2 = VGroup(B2['e'], B2['a1'], B2['z'], B2['w'], B2['s'])
        bar2 = Line([-2.1, yB - 0.3, 0], [2.4, yB - 0.3, 0],
                    color=INK, stroke_width=1.9)
        pow2g = glyph(r"\mathrm{pow}_{2}", np.array([-1.05, yB - 0.62, 0]),
                      CA, 0.55)
        a2new = farrow(np.array([-0.47, yB - 0.62, 0]),
                       np.array([-0.72, yB - 0.62, 0]))
        numL = VGroup(*[A2[k] for k in A2])
        numR = VGroup(G3['z'], G3['e'], G3['a'], jG1)
        self.play(FadeOut(B1['pw'], target_position=[-3.6, yA - 0.88, 0]),
                  FadeOut(B1['a2'], target_position=[-3.0, yA - 0.88, 0]),
                  Create(bar1), Create(bar2),
                  gB1.animate.scale(0.7).move_to([-2.7, yA - 0.88, 0]),
                  ReplacementTransform(pw2, pow2g),
                  Transform(B2['a2'], a2new),
                  gB2.animate.scale(0.6).move_to([0.6, yB - 0.62, 0]),
                  numL.animate.shift(2.7 * RIGHT),
                  numR.animate.shift(2.7 * LEFT),
                  *caption(r"the $\mathrm{pow}$'s are fractions"),
                  run_time=1.5, rate_func=EASE)
        self.wait(0.9)

        # ==== stage 6a2: both fractions onto one line; factors swap so
        #      the amber j-edge ends up leftmost (covariance) ====
        fr1 = VGroup(A1['z'], A1['e'], A1['a'], A1['w'], cdA, whis, gB1, bar1)
        bar2t = Line([0.2, -0.55, 0], [4.9, -0.55, 0],
                     color=INK, stroke_width=1.9)
        self.play(fr1.animate.shift(np.array([-0.1, -0.72, 0])),
                  plusA.animate.shift(np.array([-0.55, 0, 0])),
                  numR.animate(path_arc=0.45).shift(np.array([0.0, 1.45, 0])),
                  numL.animate(path_arc=-0.85).shift(np.array([4.85, 1.45, 0])),
                  Transform(bar2, bar2t),
                  VGroup(pow2g, B2['a2'], gB2
                         ).animate.shift(np.array([2.7, 1.12, 0])),
                  *caption(r"one line --- and the factors swap, so the"
                           r" new edge $j$ stays left"),
                  run_time=1.5, rate_func=EASE)
        self.wait(0.8)

        # ==== stage 6b: each fraction is a softmax ====
        yF = -0.45
        SC = 0.8

        def smgroup(sx, zx_off, y, adir):
            g = {}
            g['sm'] = glyph(r"\mathrm{softmax}", np.array([sx, y, 0]), CB, SC)
            zx = sx + zx_off
            g['z'] = glyph("z", np.array([zx, y, 0]), CX, SC)
            if adir > 0:
                g['a'] = farrow(np.array([zx - 0.22, y, 0]),
                                np.array([sx + 0.62, y, 0]), dotted=False)
            else:
                g['a'] = farrow(np.array([zx + 0.22, y, 0]),
                                np.array([sx - 0.62, y, 0]), dotted=False)
            return g

        d1 = Dot([-2.9, yF, 0], radius=0.06, color=INK)
        j1f = wire(np.array([-3.7, yF, 0]), np.array([-2.96, yF, 0]), CD)
        i1f = wire(np.array([-2.84, yF, 0]), np.array([-2.1, yF, 0]))
        pv1 = wire(np.array([-2.9, yF - 0.06, 0]), np.array([-2.9, yF - 0.6, 0]))
        sm1 = smgroup(-2.9, 1.15, yF - 0.92, 1)
        minus2 = MathTex("-", color=INK).scale(1.1).move_to([-1.55, yF, 0])
        jw2 = wire(np.array([-1.05, yF, 0]), np.array([-0.4, yF, 0]), CD)
        sm2b = smgroup(0.3, 1.15, yF, 1)         # (j)-- softmax <- z
        sm2a = smgroup(3.4, -1.15, yF, -1)       # z -> softmax --(i)
        iw2 = wire(np.array([4.02, yF, 0]), np.array([4.67, yF, 0]))
        self.play(
            ReplacementTransform(VGroup(*[A1[k] for k in A1 if k != 'w'],
                                        gB1, bar1),
                                 VGroup(pv1, sm1['sm'], sm1['z'], sm1['a'])),
            ReplacementTransform(cdA, d1),
            Transform(whis, j1f),
            ReplacementTransform(A1['w'], i1f),
            Transform(plusA, minus2),
            ReplacementTransform(VGroup(*[A2[k] for k in A2 if k != 'w'],
                                        gB2, pow2g, B2['a2'], bar2),
                                 VGroup(sm2a['sm'], sm2a['z'], sm2a['a'])),
            ReplacementTransform(A2['w'], iw2),
            ReplacementTransform(VGroup(G3['z'], G3['e'], G3['a']),
                                 VGroup(sm2b['sm'], sm2b['z'], sm2b['a'])),
            ReplacementTransform(jG1, jw2),
            *caption(r"each fraction is a $\mathrm{softmax}$"),
            run_time=1.8, rate_func=EASE)
        self.wait(0.9)

        # ============ stage 7: read off the identity ============
        final_grp = VGroup(d1, i1f, pv1, plusA, iw2, jw2, whis,
                           sm1['sm'], sm1['z'], sm1['a'],
                           sm2a['sm'], sm2a['z'], sm2a['a'],
                           sm2b['sm'], sm2b['z'], sm2b['a'])
        self.play(VGroup(num_t, tbar, den_t).animate.shift(RIGHT * (-v0)),
                  final_grp.animate.shift(0.75 * UP),
                  run_time=0.8, rate_func=EASE)
        self.play(Write(m2[0]), FadeIn(*m2[1:]),
                  *caption(r"the Hessian of cross-entropy"),
                  run_time=0.9)
        self.play(Indicate(VGroup(*m2[1:]), color=CD, scale_factor=1.1),
                  run_time=0.7)
        self.play(Circumscribe(VGroup(*m2[1:]), color=CD, buff=0.12),
                  run_time=1.0)
        self.wait(2.6)
