"""Gather formation as a pass (task #43): lowering emits the plain
one_hot-contraction einsum; compiler/gather.py's form_gathers — the first
pass of specialize() — makes the index_select cost decision. Verified
during the move: emitted source for gather+scatter programs is byte-for-
byte identical to the in-lowering version (same pipeline point)."""

import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler.gather import form_gathers
from tensorgrad.compiler.ir import EinsumNode, GatherNode
from tensorgrad.compiler.lower import lower_program


def _embed_program():
    b, v, d = symbols("b v d")
    idx = Variable("idx", b=b)
    wte = Variable("wte", v=v, d=d)
    return F.one_hot(idx, v) @ wte


def test_lowering_leaves_the_dense_form():
    builder_outs = lower_program([_embed_program().full_simplify()])
    b_, outs = builder_outs
    (root, _) = outs[0]
    assert isinstance(root, EinsumNode)
    assert any(isinstance(op, GatherNode) and op.op == "one_hot" for op in root.ops), (
        "lowering must emit the plain one_hot einsum (no cost decision)"
    )


def test_form_gathers_makes_the_lookup():
    b_, outs = lower_program([_embed_program().full_simplify()])
    (root, order) = form_gathers(b_, outs)[0]
    def has_gather(n, seen=None):
        seen = set() if seen is None else seen
        if id(n) in seen:
            return False
        seen.add(id(n))
        if isinstance(n, GatherNode) and n.op == "gather":
            return True
        return any(has_gather(o, seen) for o in n.operands())
    assert has_gather(root), "the pass must form the index_select GatherNode"
