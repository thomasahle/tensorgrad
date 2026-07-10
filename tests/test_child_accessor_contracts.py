"""Contract guards for the tensor-level child accessors.

Three modules walk tensor children with their own accessor, each with a
deliberately different vocabulary: structure._children (the one-source-of-
truth, driven by structure() declarations), reverse._children (the sweep's
vocabulary + Derivative bases), and fold._kids (the fold engine's five
REBUILDABLE types — Expectation etc. are deliberately leaves there).

The hazard this file exists for: a NEW Tensor type gets a structure()
declaration but is forgotten in reverse/fold, which then silently treat it
as a leaf — gradients stop flowing through it / folds stop rebuilding under
it, with no error anywhere (the szfp gates would demote the damage to
missed optimizations, but silently). These tests make the divergence LOUD:
extending the type system means either updating the accessor or updating
the pinned contract here, consciously.
"""

from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler.fold import _kids
from tensorgrad.compiler.reverse import _children as reverse_children
from tensorgrad.structure import _children as structure_children
from tensorgrad.tensor import Derivative, Product, Rename, Sum


def _cases():
    n = symbols("n")
    x = Variable("x", i=n)
    y = Variable("y", i=n)
    return {
        "Sum": Sum([x, y]),
        "Product": Product([x, y.rename(i="j")]),
        "Rename": Rename(x, {"i": "k"}),
        "Function": F.exp(x),
        "Derivative": Derivative(F.sum(x * x), x),
    }


def test_accessors_agree_with_structure_declarations():
    """On every core type, each accessor's child SET must match the pinned
    contract relative to structure()'s declaration."""
    for name, t in _cases().items():
        s_kids = {id(c) for c in structure_children(t)}
        r_kids = {id(c) for c in reverse_children(t)}
        f_kids = {id(c) for c in _kids(t)}
        if name == "Derivative":
            # reverse descends into bases (reverse-over-reverse); fold and
            # structure treat Derivative according to their own contracts.
            assert r_kids == {id(t.tensor)}
            continue
        assert r_kids == s_kids, f"{name}: reverse._children diverges from structure()"
        assert f_kids == s_kids, f"{name}: fold._kids diverges from structure()"


def test_child_order_matches_semantic_fields():
    """Order is LOAD-BEARING for the rebuild paths (Function inputs index
    VJPs; Sum children align with weights): pin it explicitly."""
    n = symbols("n")
    x, y = Variable("x", i=n), Variable("y", i=n)
    s = Sum([x, y], [2, 3])
    assert [id(c) for c in reverse_children(s)] == [id(t) for t in s.terms]
    f = F.pow(x, -1)
    assert [id(c) for c in reverse_children(f)] == [id(i) for i in f.inputs]
    assert [id(c) for c in _kids(f)] == [id(i) for i in f.inputs]
