"""Tr(AB) = Tr(BA): the cyclic property as a smooth tensor-diagram motion.

The trace of AB is a closed loop passing through A and B. Sliding the
tensors around the loop changes nothing about the diagram's value -- and
half a turn later the diagram reads Tr(BA). The animation IS the proof.

Style follows the book: black ink on white, plain math labels, thin edges.
"""

from manim import (
    Scene, MathTex, VGroup, Dot, Circle, ValueTracker, always_redraw,
    FadeIn, FadeOut, Write, TransformMatchingTex, Create,
    WHITE, BLACK, UP, DOWN, LEFT, RIGHT, ORIGIN,
    config, rate_functions, Arc, BackgroundRectangle, SurroundingRectangle,
)
import numpy as np

config.background_color = WHITE

INK = BLACK
R = 1.4  # loop radius


class TraceCyclic(Scene):
    def make_node(self, tex: str, angle_tracker: ValueTracker, offset: float):
        """A math label riding on the loop at (angle+offset), masking the
        loop line behind it (like a tikz node with white background)."""
        label = MathTex(tex, color=INK).scale(1.1)

        def at_angle():
            a = angle_tracker.get_value() + offset
            pos = R * np.array([np.cos(a), np.sin(a), 0.0])
            lbl = MathTex(tex, color=INK).scale(1.1).move_to(pos)
            mask = BackgroundRectangle(lbl, color=WHITE, fill_opacity=1.0, buff=0.09)
            return VGroup(mask, lbl)

        return always_redraw(at_angle)

    def construct(self):
        # ---- title: the classical identity, revealed in stages ----
        lhs = MathTex(r"\mathrm{Tr}(AB)", color=INK).scale(1.1)
        eq = MathTex(r"=", color=INK).scale(1.1)
        rhs = MathTex(r"\mathrm{Tr}(BA)", color=INK).scale(1.1)
        title = VGroup(lhs, eq, rhs).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.9)

        # ---- the diagram: a closed loop through A and B ----
        theta = ValueTracker(0.0)
        loop = Circle(radius=R, color=INK, stroke_width=2.5).move_to(ORIGIN)
        node_A = self.make_node("A", theta, offset=np.pi)   # starts on the left
        node_B = self.make_node("B", theta, offset=0.0)     # starts on the right

        self.play(Write(lhs), run_time=0.9)
        self.play(Create(loop), run_time=1.0)
        self.play(FadeIn(node_A), FadeIn(node_B), run_time=0.7)
        self.wait(0.6)

        # ---- the proof: slide the tensors half-way around the loop ----
        # (nothing detaches, nothing changes -- the value is constant)
        self.play(
            theta.animate.set_value(np.pi),
            run_time=3.0,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait(0.4)

        # ---- read off the result ----
        self.play(Write(eq), Write(rhs), run_time=0.9)
        self.wait(0.5)

        # a second lap, because it is satisfying (and shows full cyclicity)
        self.play(
            theta.animate.set_value(3 * np.pi),
            run_time=4.0,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait(1.2)
