"""szfp refusal gates around the pure-algebra compiler passes (task #60).

The paper's claim: a wrong rewrite in a gated pass becomes a MISSED
OPTIMIZATION, never a miscompile -- the pass's outputs must evaluate
exactly equal (mod P, seeded points) to its inputs or the whole pass is
refused. The money test corrupts a pass and checks the compiled result is
still correct; the irrational-weight test pins the false-refusal class
that sank the first attempt (symbolic weights folded to floats).
"""

import math

import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.compiler.codegen_torch as cg
import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler import szfp
from tensorgrad.testutils import assert_close


def _mini_program():
    i, j = symbols("i j")
    x = Variable("x", i=i, j=j)
    w = Variable("w", j=j)
    # 1/sqrt(j)-style irrational scaling: with concrete dims the factoring
    # pass may fold it into float weights -- the first gate attempt falsely
    # refused exactly this (symbolic sqrt vs its float64 image).
    y = F.sum((x @ w) * (x @ w)) / math.sqrt(2)
    return y, x, w, {i: 3, j: 4}


def test_gates_verify_clean_on_a_real_program():
    y, x, w, dims = _mini_program()
    g = y.grad(w)
    prog = compile_to_callable(y, g, y.grad(x))
    torch.manual_seed(0)
    vals = {x: torch.randn(3, 4).rename("i", "j"), w: torch.randn(4).rename("j")}
    prog(vals, dims)
    stats = prog.codegen.gate_stats
    refused = {k: v for k, v in stats.items() if k.endswith(":refused")}
    assert not refused, f"false refusals on a clean program: {stats}"
    assert any(k.endswith(":verified") for k in stats), (
        f"no gate actually ran (silent skip is indistinguishable from verify): {stats}"
    )


def test_gate_refuses_a_corrupted_pass(monkeypatch):
    """Corrupt factor_outputs to drop every output to its first term twice
    (double it): the gate must refuse, and the compiled program must produce
    the CORRECT (pre-pass) values -- a wrong rewrite becomes a no-op."""
    from tensorgrad.compiler import factor as factor_mod

    real = factor_mod.factor_outputs

    def corrupted(builder, outputs, dims, **kw):
        outs = real(builder, outputs, dims, **kw)
        # double every output: y -> y + y via a Linear node
        return [(builder.linear([n, n], [tuple(range(len(o))), tuple(range(len(o)))], [1, 1]), o)
                for n, o in outs]

    monkeypatch.setattr(cg, "factor_outputs", corrupted)

    y, x, w, dims = _mini_program()
    prog = compile_to_callable(y)
    torch.manual_seed(1)
    vals = {x: torch.randn(3, 4).rename("i", "j"), w: torch.randn(4).rename("j")}
    out = prog(vals, dims)
    out = out[0] if isinstance(out, tuple) else out
    from tensorgrad.extras.evaluate import evaluate

    expected = evaluate(y.full_simplify(), vals, dict(dims))
    assert_close(out, expected, rtol=1e-4, atol=1e-6)
    stats = prog.codegen.gate_stats
    assert any(k.endswith(":refused") and v > 0 for k, v in stats.items()), (
        f"corrupted pass was not refused: {stats}"
    )


def test_outputs_equal_primitive():
    """Direct check of the gate primitive: equal programs agree; a doubled
    output does not."""
    from tensorgrad.compiler.ir import Builder
    from tensorgrad.compiler.lower import lower_program

    y, x, w, dims = _mini_program()
    b, outs = lower_program([y.full_simplify()])
    assert szfp.outputs_equal(outs, list(outs))
    (n0, o0) = outs[0]
    doubled = [(b.linear([n0, n0], [tuple(range(len(o0))), tuple(range(len(o0)))], [1, 1]), o0)]
    assert not szfp.outputs_equal(outs, doubled)


def test_atoms_commute_with_broadcast():
    """The recorded false-refusal class behind factor:refused on layernorm
    gradients: factor sinks broadcasts through elementwise maps, so
    pow(broadcast(x)) and broadcast(pow(x)) must fingerprint EQUAL (atoms
    are pointwise random functions, not whole-tensor hashes) -- while a
    different exponent must still fingerprint unequal."""
    from fractions import Fraction

    from sympy import symbols as _syms

    from tensorgrad.compiler.ir import Builder

    i, j = _syms("i j")
    b = Builder()
    x = Variable("x", i=i)
    xin = b.input(x)

    def broadcast(n):
        return b.einsum([n], [(0,)], (0, 1), {0: i, 1: j})

    pow_of_broadcast = b.map("pow", (Fraction(-1, 2),), [broadcast(xin)])
    broadcast_of_pow = broadcast(b.map("pow", (Fraction(-1, 2),), [xin]))
    order = ("i", "j")
    assert szfp.outputs_equal([(pow_of_broadcast, order)], [(broadcast_of_pow, order)])
    other_exponent = broadcast(b.map("pow", (Fraction(-3, 2),), [xin]))
    assert not szfp.outputs_equal([(pow_of_broadcast, order)], [(other_exponent, order)])


def test_irrational_weights_do_not_false_refuse():
    """sqrt-scaled programs (the recorded blocker): symbolic irrational
    weights and their float64-folded images must fingerprint equal."""
    a = szfp._scalar_residue(math.sqrt(2), {}, ("t",))
    import sympy

    b = szfp._scalar_residue(sympy.sqrt(2), {}, ("t",))
    assert a == b, "sympy sqrt(2) and float64 sqrt(2) must share a residue"
    c = szfp._scalar_residue(sympy.sqrt(2) * sympy.sqrt(2), {}, ("t",))
    assert c == szfp._scalar_residue(2, {}, ("t",)), "sqrt(2)^2 must equal 2"
