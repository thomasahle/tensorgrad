from sympy import symbols

from tensorgrad.tensor import Delta, Variable
from tensorgrad.structure import graph_to_string
import tensorgrad.functions as F


def test_interaction_with_copy():
    i = symbols("i")
    x0 = Variable("x", i)
    x1 = x0.rename(i="i_")
    c0 = Delta(i, "i")
    c1 = Delta(i, "i_")
    assert (x0 @ c0).simplify() == (x1 @ c1).simplify()


def test_interaction_with_copy2():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i_=i)
    assert (x.rename(i="i_") @ y).simplify() == (x @ Delta(i, "i", "i_") @ y).simplify()
    assert (x @ y).simplify() != (x @ Delta(i, "i", "i_") @ y).simplify()


def test_rename_with_function():
    i = symbols("i")
    x = Variable("x", i)
    a = (F.relu(x) @ Delta(i, "i")).simplify()
    b = (F.relu(x.rename(i="i_")) @ Delta(i, "i_")).simplify()
    c = (F.relu(x).rename(i="i_") @ Delta(i, "i_")).simplify()
    print(graph_to_string(a))
    print(graph_to_string(b))
    print(graph_to_string(c))
    assert b == c
    assert a == b
    assert a == c


def test_function_broadcast_rename_collision():
    # Regression test: Function._rename used to crash when a broadcast edge was
    # renamed to a name that collides with an edge consumed by the function.
    # E.g. softmax(x[s, t], dim='t') has broadcast edge 's' and output edge 't';
    # renaming s->t (with t going elsewhere) passed rename={'s': 't'} to the
    # input tensor x, which still has its own edge 't'.
    import torch
    from tensorgrad.testutils import rand_values
    from tensorgrad.extras.evaluate import evaluate

    s, t, u = symbols("s t u")
    x = Variable("x", s, t)
    f = F.softmax(x, dim="t")
    assert f.edges == {"s", "t"}

    ts = rand_values([x], {s: 3, t: 4})
    ref = evaluate(f, ts)

    # Swap rename: s->t, t->s (previously ValueError at construction)
    g_swap = f.rename(s="t", t="s")
    assert g_swap.shape == {"t": s, "s": t}
    out = evaluate(g_swap, ts)
    expected = ref.rename(s="t", t="s")
    torch.testing.assert_close(out.align_to(*expected.names).rename(None), expected.rename(None))

    # Chain rename: s->t, t->u
    g_chain = f.rename(s="t", t="u")
    assert g_chain.shape == {"t": s, "u": t}
    out = evaluate(g_chain, ts)
    expected = ref.rename(s="t", t="u")
    torch.testing.assert_close(out.align_to(*expected.names).rename(None), expected.rename(None))

    # Simplify with unexpanded functions must not crash and stay correct
    for expr, exp in [(g_swap, ref.rename(s="t", t="s")), (g_chain, ref.rename(s="t", t="u"))]:
        for args in ({"expand_functions": False}, None):
            simplified = expr.simplify(args)
            out = evaluate(simplified, ts)
            torch.testing.assert_close(out.align_to(*exp.names).rename(None), exp.rename(None))


def test_function_broadcast_rename_collision_two_blocks():
    # Two chained softmax "attention" blocks with a transpose rename in between,
    # like composing two attention blocks with fused (unexpanded) softmax.
    import torch
    from tensorgrad.testutils import rand_values
    from tensorgrad.extras.evaluate import evaluate

    s, t = symbols("s t")
    x = Variable("x", s, t)
    block1 = F.softmax(x, dim="t")  # edges {s, t}
    # Feed the transposed output into a second fused softmax block
    block2 = F.softmax(block1.rename(s="t", t="s"), dim="t")  # previously crashed
    assert block2.edges == {"s", "t"}

    ts = rand_values([x], {s: 3, t: 4})
    # After the transpose rename, block2's 't' edge is the original 's' axis of
    # block1's output, so the second softmax runs over dim=0 of the first result.
    ref = torch.softmax(torch.softmax(ts[x].rename(None), dim=1), dim=0)

    for args in ({"expand_functions": False}, None):
        out = evaluate(block2.simplify(args), ts)
        torch.testing.assert_close(out.align_to("t", "s").rename(None), ref)
