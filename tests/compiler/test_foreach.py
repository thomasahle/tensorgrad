"""Foreach grouping (codegen_torch.FOREACH_GROUPING): families of
shape-isomorphic per-parameter elementwise chains (MapNode/LinearNode)
collapse into multi-tensor torch._foreach_* calls at emission time.

The A/B contract is *bitwise*: only out-of-place _foreach variants are
emitted, and each one runs the exact same ATen kernel as its singleton twin
in the exact same dataflow — so flag on/off outputs must be torch.equal,
not merely allclose.
"""

import re

import torch
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
import tensorgrad.compiler.codegen_torch as cg

torch.set_num_threads(2)


def _run(outs, vals, dims, flag, monkeypatch):
    """Compile with FOREACH_GROUPING set to `flag`; return (outputs, source)."""
    monkeypatch.setattr(cg, "FOREACH_GROUPING", flag)
    fn = compile_to_callable(*outs)
    res = fn(dict(vals), dict(dims))
    if len(outs) == 1:
        res = (res,)
    (spec,) = fn._specializations.values()
    return [t.rename(None) for t in res], spec._source


def _adamw_like_chains(k=8):
    """k isomorphic AdamW-style updates over k different shapes:
    w' = 0.9*w - 0.01 * m / (sqrt(v) + 1e-8), all elementwise."""
    outs, vals, dims = [], {}, {}
    for i in range(k):
        di, dj = symbols(f"i{i} j{i}")
        w = Variable(f"w{i}", di, dj)
        m = Variable(f"m{i}", di, dj)
        v = Variable(f"v{i}", di, dj)
        outs.append(0.9 * w - 0.01 * m / (F.sqrt(v) + 1e-8))
        si, sj = 3 + i % 3, 4 + i % 4
        dims[di], dims[dj] = si, sj
        gen = torch.Generator().manual_seed(i)
        names = tuple(w.shape.keys())
        vals[w] = torch.randn(si, sj, generator=gen).rename(*names)
        vals[m] = torch.randn(si, sj, generator=gen).rename(*names)
        vals[v] = (torch.rand(si, sj, generator=gen) + 0.5).rename(*names)
    return outs, vals, dims


def test_foreach_bitwise_ab_and_source(monkeypatch):
    """8 isomorphic chains: flag on groups them into _foreach calls, and the
    two emissions are bit-identical."""
    outs, vals, dims = _adamw_like_chains(8)
    on, src_on = _run(outs, vals, dims, True, monkeypatch)
    off, src_off = _run(outs, vals, dims, False, monkeypatch)
    assert "_foreach_" in src_on
    assert "_foreach_" not in src_off
    for a, b in zip(on, off):
        assert torch.equal(a, b)


def test_foreach_grouped_lines_and_liveness(monkeypatch):
    """Grouped emission shape: tuple-unpack onto the pre-assigned t-names
    (>= 4 members per call), and EMIT_DEL still frees grouped intermediates
    (the tuple-LHS extension of the liveness regex)."""
    outs, vals, dims = _adamw_like_chains(8)
    _, src = _run(outs, vals, dims, True, monkeypatch)
    unpacks = re.findall(r"^\s*((?:t\d+, )+t\d+) = torch\._foreach_\w+\(", src, re.M)
    assert unpacks, f"no tuple-unpacked _foreach lines in:\n{src}"
    assert all(len(u.split(", ")) >= cg.FOREACH_MIN for u in unpacks)
    # At least one grouped (non-output) member must be freed by EMIT_DEL.
    deleted = {nm for line in re.findall(r"^\s*del (.+)$", src, re.M) for nm in line.split(", ")}
    grouped = {nm for u in unpacks for nm in u.split(", ")}
    assert grouped & deleted, f"no grouped name is ever del'd:\n{src}"
    # Intermediate list temps are freed too.
    if "_f0" in src:
        assert "_f0" in deleted


def test_foreach_mixes_tensor_orders(monkeypatch):
    """Shape erasure includes the tensor order: 1-D and 2-D members with
    identity alignment share one family."""
    outs, vals, dims = [], {}, {}
    for i in range(6):
        if i % 2:
            di, dj = symbols(f"mi{i} mj{i}")
            x = Variable(f"x{i}", di, dj)
            y = Variable(f"y{i}", di, dj)
            sz = (3 + i, 2 + i)
            dims[di], dims[dj] = sz
        else:
            di = symbols(f"mi{i}")
            x = Variable(f"x{i}", di)
            y = Variable(f"y{i}", di)
            sz = (4 + i,)
            dims[di] = sz[0]
        outs.append(2.0 * x - 3.0 * F.sqrt(y) + 0.25)
        gen = torch.Generator().manual_seed(100 + i)
        names = tuple(x.shape.keys())
        vals[x] = torch.randn(*sz, generator=gen).rename(*names)
        vals[y] = (torch.rand(*sz, generator=gen) + 0.5).rename(*names)
    on, src_on = _run(outs, vals, dims, True, monkeypatch)
    off, _ = _run(outs, vals, dims, False, monkeypatch)
    assert "torch._foreach_sqrt" in src_on
    # All six chains land in ONE family per step (orders mixed).
    m = re.search(r"(?m)^\s*((?:t\d+, ){5}t\d+) = torch\._foreach_sqrt\(", src_on)
    assert m, f"expected a 6-member _foreach_sqrt in:\n{src_on}"
    for a, b in zip(on, off):
        assert torch.equal(a, b)


def test_foreach_demotes_member_with_early_consumer(monkeypatch):
    """A family member whose consumer sits before the family anchor must be
    demoted to individual emission (verify-and-demote fixpoint): here
    sum(sqrt(v0)) is toposorted between sqrt(v0) and the later sqrt members,
    so sqrt(v0) cannot wait for the anchor. The rest still group, and the
    program stays valid and bit-identical."""
    outs, vals, dims = [], {}, {}
    sqrts = []
    for i in range(5):
        di, dj = symbols(f"di{i} dj{i}")
        v = Variable(f"v{i}", di, dj)
        sqrts.append(F.sqrt(v))
        si, sj = 2 + i, 3 + i
        dims[di], dims[dj] = si, sj
        gen = torch.Generator().manual_seed(200 + i)
        vals[v] = (torch.rand(si, sj, generator=gen) + 0.5).rename(*v.shape.keys())
    outs = [sqrts[0], F.sum(sqrts[0])] + sqrts[1:]
    on, src_on = _run(outs, vals, dims, True, monkeypatch)
    off, _ = _run(outs, vals, dims, False, monkeypatch)
    assert "torch._foreach_sqrt" in src_on  # the surviving 4-member family
    assert "torch.sqrt(" in src_on  # the demoted member emits individually
    for a, b in zip(on, off):
        assert torch.equal(a, b)
