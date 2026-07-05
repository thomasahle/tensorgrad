"""Torch code generation: IR DAG -> straight-line Python source.

Code is generated once per *shape signature* (a concrete assignment of the
symbolic dims). At that point every planning decision is taken:
  - contraction paths are computed with opt_einsum on concrete shapes and
    unrolled into explicit pairwise torch.einsum calls,
  - all dim-dependent constants (zeros, rare diagonal fallbacks, convolution
    structure) are built once and closed over as default arguments,
  - scalar weights (which may be symbolic, e.g. 1/batch) become float literals.

The hot path is then nothing but tensor-op calls on positional tensors:
no eval(), no named tensors, no dict lookups, no path planning.
"""

import re
import string
import textwrap

import opt_einsum as oe
import sympy
import torch

from tensorgrad.compiler.ir import (
    Builder,
    ConstNode,
    EinsumNode,
    GatherNode,
    InputNode,
    LinearNode,
    MapNode,
    Node,
    ReduceNode,
    to_float,
    toposort,
)
from tensorgrad.compiler.affine import indicator_tensor
from tensorgrad.compiler.factor import factor_outputs
from tensorgrad.compiler.layout import assign_layouts, matmul_groups
from tensorgrad.compiler.stabilize import stabilize_outputs

_LETTERS = string.ascii_letters

# When False, affine rows are never eliminated into views — every structured
# tensor is materialized as a dense indicator. Exists so tests can check the
# fast path against the always-correct fallback.
AFFINE_FAST = True

# When True, the attention pattern
#   EinsumNode(Q,K) -> [scale/mask LinearNodes] -> ReduceNode(softmax) -> EinsumNode(.,V)
# is emitted as a single torch.nn.functional.scaled_dot_product_attention call
# (when the intermediates have no other consumers and the axis layout permits
# a clean permute). Toggleable so tests can compare against the unfused path.
SDPA_FUSION = True

# When True (default), affine view rewrites emit spec-time INTEGER strides:
# the layout of every tensor an as_strided view reads from is either tracked
# exactly (hoisted constants, pad outputs, previous as_strided views we
# emitted ourselves) or normalized with .contiguous() (a no-op for operands
# that are already row-major). The generated code then contains no
# .stride()/.storage_offset() calls — torch ops returning python ints, which
# torch.compile(fullgraph=True) refuses to trace. When False, the old
# runtime-stride form is emitted (pure views on arbitrary layouts, but graph
# breaks / fullgraph failures under torch.compile). Kept as a flag so tests
# can compare the two emissions and exercise the runtime fallback.
STATIC_STRIDES = True

# Global layout assignment (compiler/layout.py): each node's physical axis
# order is chosen so producers emit for free and consumers read without
# permutes. False pins every node to its logical axis order (the old
# behavior); kept so tests can compare the two emissions.
LAYOUT_ASSIGN = True

# Map matmul-shaped einsum steps onto torch.mm/bmm/matmul (measured ~2x
# cheaper dispatch than torch.einsum: 7.1us vs 14.8us for a small mm).
# False falls back to einsum everywhere.
MATMUL_CELLS = True

# Fuse `x @ W + bias` LinearNodes into a single torch.addmm call (the mm
# term is suppressed and computed inside the Linear emission).
ADDMM_FUSION = True

# Emit `del name` after each intermediate's last use, so peak memory tracks
# the live set instead of the sum of all intermediates. False keeps every
# intermediate alive to function end (the old behavior); kept as a flag so
# the memory effect can be measured and tested against the plain emission.
EMIT_DEL = True


def _fmt_weight(w: float) -> str:
    return repr(w)


def _prod(xs) -> int:
    r = 1
    for x in xs:
        r *= x
    return r


def _tup(items) -> str:
    """Format a valid python tuple literal: (), (a,), (a, b)."""
    items = list(items)
    if len(items) == 1:
        return f"({items[0]},)"
    return f"({', '.join(items)})"


def _row_major(sizes) -> tuple[int, ...]:
    """Standard contiguous (row-major) strides for concrete sizes."""
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return tuple(strides)


def _normalize_rows(rows, wire_dims, dim_of):
    """Drop always-true rows, detect unsatisfiable systems.
    Returns (rows, unsat)."""
    out = []
    for r in rows:
        if r[0] == "eq":
            _, coeffs, const = r
            if not coeffs:
                if const != 0:
                    return [], True
                continue
            out.append(r)
        else:
            _, coeffs, k, X = r
            lo = k + sum(min(0, c * (dim_of(wire_dims[w]) - 1)) for w, c in coeffs.items())
            hi = k + sum(max(0, c * (dim_of(wire_dims[w]) - 1)) for w, c in coeffs.items())
            if lo >= 0 and hi <= X - 1:
                continue  # always satisfied: the summed wire hits exactly once
            if hi < 0 or lo > X - 1:
                return [], True
            out.append(r)
    return out, False


def _canon_eq(eq: str) -> str:
    """Relabel einsum letters by first occurrence, for step-CSE keys."""
    remap: dict[str, str] = {}
    out = []
    for ch in eq:
        if ch in ",->":
            out.append(ch)
        else:
            if ch not in remap:
                remap[ch] = _LETTERS[len(remap)]
            out.append(remap[ch])
    return "".join(out)


class TorchCodegen:
    def __init__(self, builder: Builder, outputs: list[tuple[Node, tuple[str, ...]]]):
        self.builder = builder
        self.outputs = outputs
        # Deterministic input order (sorted by variable name).
        self.input_names = sorted(builder.input_vars)

    def specialize(self, dims: dict[sympy.Symbol, int], verbose: bool = False, dtype: torch.dtype = None):
        """Generate and exec source for concrete dims.
        Returns a function f(*input_tensors) -> tuple[torch.Tensor, ...].

        `dtype` is the floating dtype of the call's inputs (threaded into
        hoisted constants and torch.full literals, so bf16/fp64 inputs never
        silently meet fp32 constants). Defaults to torch.get_default_dtype()."""
        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        self._dtype_str = str(self._dtype)  # e.g. "torch.float64"

        def dim_of(expr) -> int:
            e = sympy.sympify(expr)
            v = e.subs(dims) if e.free_symbols else e
            if not v.is_Integer:
                raise ValueError(f"Dimension {expr} did not resolve to an integer (got {v})")
            return int(v)

        # Factoring pass: per-shape rewrite of the DAG (un-distribution /
        # distribution / delta absorption) before any emission planning.
        outputs = factor_outputs(self.builder, self.outputs, dims)
        # Stability pass: re-fuse expanded exp/sum-exp ratios, log-sum-exp and
        # exp-ratio tanh into stable softmax/log_softmax/tanh/logsumexp forms.
        outputs = stabilize_outputs(self.builder, outputs)
        # Second round: stabilization introduces new Linear structure (the
        # ±tanh splits, softmax/sum-exp nodes) whose like terms only the
        # factoring pass can merge/cancel, and re-factoring in turn co-locates
        # numerator/denominator pairs for one more stabilization sweep.
        outputs = factor_outputs(self.builder, outputs, dims)
        outputs = stabilize_outputs(self.builder, outputs)

        order = toposort([node for node, _ in outputs])
        output_ids = {id(node) for node, _ in outputs}
        # Global layout assignment: physical axis order per node ({} = logical
        # order everywhere). Emitters consult _phys_of() for operands and MUST
        # produce their own node in its assigned order.
        var_orders = [tuple(self.builder.input_vars[n].edges) for n in self.input_names]
        self._phys = assign_layouts(order, outputs, var_orders) if LAYOUT_ASSIGN else {}
        sdpa_plans, sdpa_suppressed = self._plan_sdpa(order, output_ids, dims)
        addmm_plans, addmm_suppressed = self._plan_addmm(
            order, output_ids, dims, dim_of, sdpa_plans, sdpa_suppressed
        )
        self._addmm_plans = addmm_plans
        # Spec-time layout knowledge: emitted name -> (sizes, strides, offset).
        # offset None = "unknown but inherited correctly" (see _try_eliminate_row).
        # Only names whose layout we construct ourselves are entered; everything
        # else is normalized with .contiguous() before being viewed.
        self._layouts: dict[str, tuple[tuple[int, ...], tuple[int, ...], int | None]] = {}
        names: dict[int, str] = {}
        lines: list[str] = []
        consts: dict[str, torch.Tensor] = {}
        emitted_inputs: dict[str, str] = {}
        # Step-level CSE: identical pairwise contractions (same equation on the
        # same operands) across different einsum nodes are computed once. This
        # is what lets a loss and its gradients share partial contractions
        # (e.g. dlogits) even when the symbolic Sum-of-Products form doesn't.
        step_cache: dict[tuple, str] = {}

        def const_name(tensor: torch.Tensor, base: str) -> str:
            name = f"_c{len(consts)}_{base}"
            consts[name] = tensor
            return name

        deferred_consts: list[tuple[int, "ConstNode", str]] = []
        # Deferred code lines (dense one-hot materializations): only kept if
        # some consumer did not turn them into an index_add_ scatter.
        deferred_lines: list[tuple[int, str, str]] = []
        for i, node in enumerate(order):
            name = f"t{i}"
            names[id(node)] = name
            if id(node) in sdpa_suppressed:
                continue  # consumed only by a fused SDPA call; never emitted
            if id(node) in addmm_suppressed:
                continue  # computed inside its consuming Linear's addmm call
            if isinstance(node, InputNode):
                # Inputs arrive as positional args in self.input_names order.
                emitted_inputs[node.var_name] = name
            elif id(node) in sdpa_plans:
                lines.append(self._emit_sdpa(sdpa_plans[id(node)], name, names, node))
            elif isinstance(node, ConstNode):
                if node.kind == "scalar":
                    lines.append(f"{name} = {_fmt_weight(to_float(node.params[0].subs(dims)))}")
                else:
                    # Deferred: only built if still referenced after passes like
                    # convolution elimination have run (dead-const elimination).
                    deferred_consts.append((len(lines), node, name))
                    lines.append(None)
                    if node.kind in ("zero", "delta", "reshape"):
                        # _build_const makes these row-major contiguous
                        # (zeros / zeros+diag-fill / eye().reshape), so their
                        # layout is known before the tensor even exists.
                        sizes = tuple(dim_of(d) for d in node.dims)
                        self._layouts[name] = (sizes, _row_major(sizes), 0)
            elif isinstance(node, EinsumNode):
                lines.extend(self._emit_einsum(node, name, names, dim_of, dims, step_cache, const_name))
            elif isinstance(node, LinearNode):
                lines.append(self._emit_linear(node, name, names, dims, dim_of))
            elif isinstance(node, MapNode):
                lines.append(self._emit_map(node, name, names))
            elif isinstance(node, GatherNode):
                if node.op == "gather":
                    lines.append(self._emit_gather(node, name, names, dim_of))
                else:  # one_hot: deferred, usually replaced by a scatter
                    deferred_lines.append((len(lines), name, self._emit_one_hot(node, name, names, dim_of)))
                    lines.append(None)
            elif isinstance(node, ReduceNode):
                lines.append(self._emit_reduce(node, name, names))
            else:
                raise NotImplementedError(f"codegen for {type(node).__name__}")

        # Scalars must come back as tensors for a consistent API. Outputs are
        # returned in logical (edge) order: un-permute assigned layouts.
        rets = []
        for node, _ in outputs:
            n = names[id(node)]
            if node.order == 0:
                rets.append(f"torch.as_tensor({n}, dtype={self._dtype_str})")
            else:
                phys = self._phys_of(node)
                inv = tuple(phys.index(j) for j in range(node.order))
                rets.append(self._perm_str(n, inv))

        # Dead-const elimination: build only constants whose alias name is
        # still referenced somewhere (conv elimination removes most uses).
        used_text = "\n".join(line for line in lines if line is not None) + "\n" + " ".join(rets)
        for line_idx, node, name in deferred_consts:
            if re.search(rf"\b{name}\b", used_text):
                tensor = self._build_const(node, dim_of)
                lines[line_idx] = f"{name} = {const_name(tensor, node.kind)}"
        for line_idx, name, text in deferred_lines:
            if re.search(rf"\b{name}\b", used_text):
                lines[line_idx] = text
        lines = [line for line in lines if line is not None]

        # Dead-line elimination: every emitted line is a pure assignment
        # (the only in-place op, index_add_, is not of `name = ...` form), so
        # straight-line SSA applies: an assignment whose name appears nowhere
        # else is dead. Iterate to a fixpoint — removing a dead broadcast
        # view or torch.full orphans its own operands. (Consumers routinely
        # fold such terms away: Linear scalar offsets, einsum weight
        # absorption, addmm bias fusion.)
        while True:
            counts: dict[str, int] = {}
            for tok in re.findall(r"\w+", "\n".join(lines) + "\n" + " ".join(rets)):
                counts[tok] = counts.get(tok, 0) + 1
            keep = []
            for line in lines:
                m = re.match(r"([A-Za-z_]\w*) = ", line)
                if m and "\n" not in line and counts.get(m.group(1)) == 1:
                    continue
                keep.append(line)
            if len(keep) == len(lines):
                break
            lines = keep

        # Liveness: free every intermediate at its last textual use. The
        # program is straight-line SSA, so a name's final occurrence ends its
        # live range exactly. For deep-gradient programs the SUM of all
        # intermediates can be 10-100x the peak LIVE set (measured 2.25 GB
        # total vs ~0.1 GB live on a GPT-2 block loss+grads program); without
        # the dels the peak RSS is the sum. Views keep their base storage
        # alive through torch's refcounting, so deleting a base name early is
        # safe. Return values and closed-over constants are never deleted
        # (constants arrive as default parameters, inputs as positional
        # parameters — neither is ever in assigned_names).
        if EMIT_DEL:
            ret_names = set(re.findall(r"\w+", " ".join(rets)))
            assigned_names = {
                m.group(1) for line in lines if (m := re.match(r"([A-Za-z_]\w*) = ", line))
            }
            last_use: dict[str, int] = {}
            for i, line in enumerate(lines):
                for tok in set(re.findall(r"\w+", line)):
                    if tok in assigned_names:
                        last_use[tok] = i
            by_line: dict[int, list[str]] = {}
            for nm, i in last_use.items():
                if nm not in ret_names:
                    by_line.setdefault(i, []).append(nm)
            lines = [
                text
                for i, line in enumerate(lines)
                for text in ([line] + ([f"del {', '.join(sorted(by_line[i]))}"] if i in by_line else []))
            ]

        params = [emitted_inputs.get(v, f"_unused_{v}") for v in self.input_names]
        params += [f"{k}={k}" for k in consts]
        body = "\n".join(lines) or "pass"
        src = (
            f"def _compiled({', '.join(params)}):\n"
            + textwrap.indent(body, "    ")
            + f"\n    return ({', '.join(rets)},)\n"
        )
        if verbose:
            print(src)

        ns: dict = dict(consts)
        ns["torch"] = torch
        exec(src, ns)
        fn = ns["_compiled"]
        fn._source = src
        return fn

    # ---- layout helpers ---------------------------------------------------

    def _phys_of(self, node: Node) -> tuple[int, ...]:
        """Assigned physical layout of a node: phys[p] = logical axis stored
        at physical position p. Identity when no layout was assigned."""
        p = self._phys.get(id(node))
        return p if p is not None else tuple(range(node.order))

    @staticmethod
    def _perm_str(name: str, perm) -> str:
        perm = tuple(perm)
        if perm == tuple(range(len(perm))):
            return name
        return f"{name}.permute({', '.join(map(str, perm))})"

    def _entry_subs(self, op: Node, subs) -> list:
        """Wire subscripts of an operand in its PHYSICAL axis order (physical
        axis q of the emitted tensor carries wire subs[phys[q]])."""
        phys = self._phys_of(op)
        return [subs[q] for q in phys]

    # ---- matmul cell selection --------------------------------------------

    @staticmethod
    def _matmul_fit(a_axes, b_axes, out_axes, size_of):
        """Try to express `contract(a, b) -> out` (axis labels of the two
        operands' PHYSICAL layouts, and the desired physical output order) as
        a (batched) matrix multiply with only free reshapes/transposes:

          out = matmul(a viewed (*batch, M, K), b viewed (*batch, K, N))

        The batch block must lead all three, identically ordered (BLAS batch
        dims cannot be transposed for free); within each operand the trailing
        two blocks may appear in either order (op(A) transpose flags are
        free); intra-block orders must agree pairwise so the merged M/K/N
        dims flatten consistently.

        Returns a spec dict for _matmul_render, or None. `size_of` maps an
        axis label to its concrete size."""
        for a, b, x1, x2 in ((a_axes, b_axes, 0, 1), (b_axes, a_axes, 1, 0)):
            A, B, O = set(a), set(b), set(out_axes)
            if len(A) != len(a) or len(B) != len(b) or len(O) != len(out_axes):
                return None  # repeated labels: diagonals, einsum territory
            if not O <= (A | B) or (A - B - O) or (B - A - O):
                return None  # broadcast-only outputs / single-operand sums
            k = [c for c in a if c in B and c not in O]
            if not k:
                return None  # no contraction: TensorIterator territory
            m = [c for c in out_axes if c in A and c not in B]
            n = [c for c in out_axes if c in B and c not in A]
            if not m and not n:
                return None  # batched dot: (A*B).sum() beats a degenerate bmm
            batch = [c for c in out_axes if c in A and c in B]
            nb = len(batch)
            # batch block must lead everywhere, in the output's order
            if list(a[:nb]) != batch or list(b[:nb]) != batch:
                continue
            o_rest = list(out_axes[nb:])
            if set(o_rest[: len(m)]) != set(m):
                continue  # out must be (batch, m-block, n-block) for THIS roles split
            mo, no = o_rest[: len(m)], o_rest[len(m):]

            def blocks(rest, first, second):
                for swap in (False, True):
                    f, s = (second, first) if swap else (first, second)
                    if list(rest[: len(f)]) == list(f) and list(rest[len(f):]) == list(s):
                        return swap
                return None

            a_swap = blocks(list(a[nb:]), mo, k)
            if a_swap is None:
                continue
            b_swap = blocks(list(b[nb:]), k, no)
            if b_swap is None:
                continue
            return {
                "x1": x1,  # which arg is the (batch, M, K) operand
                "x2": x2,
                "batch": [size_of(c) for c in batch],
                "msz": _prod([size_of(c) for c in mo]),
                "ksz": _prod([size_of(c) for c in k]),
                "nsz": _prod([size_of(c) for c in no]),
                "a_swap": a_swap,  # operand stored (batch, K, M): transpose
                "b_swap": b_swap,  # operand stored (batch, N, K): transpose
                "a_merge": len(mo) != 1 or len(k) != 1,
                "b_merge": len(k) != 1 or len(no) != 1,
                "out_sizes": [size_of(c) for c in out_axes],
            }
        return None

    @staticmethod
    def _matmul_operands(fit, args) -> tuple[str, str, str]:
        """The two operand expressions (viewed to (*batch, M, K)/(*batch, K, N))
        and the trailing .view suffix ('' if the raw matmul shape is final)."""
        bsz, msz, ksz, nsz = fit["batch"], fit["msz"], fit["ksz"], fit["nsz"]
        va, vb = args[fit["x1"]], args[fit["x2"]]
        if fit["a_merge"]:
            shape = bsz + ([ksz, msz] if fit["a_swap"] else [msz, ksz])
            va = f"{va}.reshape({_tup(map(str, shape))})"
        if fit["a_swap"]:
            va = f"{va}.transpose(-2, -1)"
        if fit["b_merge"]:
            shape = bsz + ([nsz, ksz] if fit["b_swap"] else [ksz, nsz])
            vb = f"{vb}.reshape({_tup(map(str, shape))})"
        if fit["b_swap"]:
            vb = f"{vb}.transpose(-2, -1)"
        view = ""
        if bsz + [msz, nsz] != fit["out_sizes"]:
            view = f".view({_tup(map(str, fit['out_sizes']))})"
        return va, vb, view

    @classmethod
    def _matmul_render(cls, fit, args) -> str:
        """Emit the torch.mm/bmm/matmul expression for a _matmul_fit spec."""
        va, vb, view = cls._matmul_operands(fit, args)
        bsz = fit["batch"]
        fn = "torch.mm" if not bsz else ("torch.bmm" if len(bsz) == 1 else "torch.matmul")
        return f"{fn}({va}, {vb}){view}"

    def _matmul_expr(self, step_eq: str, args, size_of):
        """Cell selection for a 2-operand einsum step. Returns an expression
        string using torch.mm/bmm/matmul, or None (einsum fallback)."""
        if not MATMUL_CELLS:
            return None
        ins, out = step_eq.split("->")
        in_eqs = ins.split(",")
        if len(in_eqs) != 2:
            return None
        fit = self._matmul_fit(tuple(in_eqs[0]), tuple(in_eqs[1]), tuple(out), size_of)
        if fit is None:
            return None
        return self._matmul_render(fit, args)

    # ---- addmm fusion -----------------------------------------------------

    def _plan_addmm(self, order, output_ids, dims, dim_of, sdpa_plans, sdpa_suppressed):
        """Find LinearNodes computable as torch.addmm(bias, A, B) (+ leftover
        terms): one term is a single-consumer un-batched matmul-shaped einsum,
        another term broadcasts a bias over exactly the matmul's N-block.
        Returns (plans: id(Linear) -> plan, suppressed einsum ids)."""
        if not (ADDMM_FUSION and MATMUL_CELLS and LAYOUT_ASSIGN):
            return {}, set()
        consumers: dict[int, int] = {}
        for nd in order:
            for op in nd.operands():
                consumers[id(op)] = consumers.get(id(op), 0) + 1
        plans, suppressed = {}, set()
        for node in order:
            if not isinstance(node, LinearNode) or id(node) in sdpa_suppressed:
                continue
            if len(node.terms) < 2:
                continue
            plan = self._addmm_fit(node, consumers, output_ids, dims, dim_of,
                                   sdpa_plans, sdpa_suppressed, suppressed)
            if plan is not None:
                plans[id(node)] = plan
                suppressed.add(plan["t_id"])
        return plans, suppressed

    def _addmm_fit(self, node, consumers, output_ids, dims, dim_of,
                   sdpa_plans, sdpa_suppressed, already_suppressed):
        physN = self._phys_of(node)
        for i, (term, perm) in enumerate(zip(node.terms, node.perms)):
            if not isinstance(term, EinsumNode):
                continue
            tid = id(term)
            if (
                consumers.get(tid, 0) != 1
                or tid in output_ids
                or tid in sdpa_plans
                or tid in sdpa_suppressed
                or tid in already_suppressed
            ):
                continue
            groups = matmul_groups(term)
            if groups is None or groups[0]:  # addmm is strictly un-batched 2D
                continue
            # Desired physical output of the term, seen through the Linear's
            # layout: physical position p of the Linear holds term logical
            # axis perm[physN[p]], i.e. wire term.out_subs[perm[physN[p]]].
            target = tuple(term.out_subs[perm[physN[p]]] for p in range(node.order))
            size_of = lambda w: dim_of(term.wire_dims[w])  # noqa: E731
            a_axes = tuple(self._entry_subs(term.ops[0], term.in_subs[0]))
            b_axes = tuple(self._entry_subs(term.ops[1], term.in_subs[1]))
            fit = self._matmul_fit(a_axes, b_axes, target, size_of)
            if fit is None:
                continue
            # Find a bias term: a pure-broadcast einsum whose real wires are
            # exactly the trailing N-block (a per-column bias, input shape
            # (Nsz,)) or the leading M-block (a per-row bias, input shape
            # (Msz, 1)) of `target`, in physical order.
            n_len = len([w for w in target if w in set(term.in_subs[fit["x2"]])
                         and w not in set(term.in_subs[fit["x1"]])])
            m_len = node.order - n_len
            for j, (bt, bperm) in enumerate(zip(node.terms, node.perms)):
                if j == i or not isinstance(bt, EinsumNode):
                    continue
                if bt.constraints or len(bt.ops) != 1:
                    continue
                (bop,) = bt.ops
                (bsubs,) = bt.in_subs
                if not bsubs or len(set(bsubs)) != len(bsubs) or not set(bsubs) <= set(bt.out_subs):
                    continue
                btarget = tuple(bt.out_subs[bperm[physN[p]]] for p in range(node.order))
                op_phys_wires = tuple(self._entry_subs(bop, list(bsubs)))
                if btarget[m_len:] == op_phys_wires and not any(
                    w in set(bsubs) for w in btarget[:m_len]
                ):
                    side = "n"
                elif btarget[:len(op_phys_wires)] == op_phys_wires and len(
                    op_phys_wires
                ) == m_len and not any(w in set(bsubs) for w in btarget[m_len:]):
                    side = "m"
                else:
                    continue
                return {"t_id": id(term), "term_idx": i, "bias_idx": j,
                        "fit": fit, "bias_op": bop, "bias_side": side,
                        "alpha_extra": to_float(term.weight.subs(dims)),
                        "beta_extra": to_float(bt.weight.subs(dims))}
        return None

    # ---- affine constraint elimination -----------------------------------

    def _eliminate_affine(self, entries, rows, out_subs, wire_dims, dim_of, emit_cached, const_name):
        """Eliminate affine constraint rows into strided views of dense operands.

        Rows come in two kinds:
          ("eq", coeffs, const)     [sum c_i*w_i == const]
          ("range", coeffs, k, X)   [0 <= sum c_i*w_i + k <= X-1]
        Range rows arise from summing an eq row over a free wire:
        sum_x [x == e] = [0 <= e < X]. When e's range provably fits, the row
        drops entirely — this is the algebraic telescoping that makes
        convolution-derivative compositions cheap.

        Elimination ladder, in order:
          1. entry rewrite: a ±1-coeff wire living on exactly one dense
             operand turns that operand's axis into a torch.as_strided view
             (unfold/diag/reshape/shift are all instances; zero-padding for
             out-of-range, flip for negative coefficients);
          2. free-wire summation with Gaussian elimination of the wire from
             all other rows (this is how conv∘conv compositions resolve);
          3. leftover rows materialize as hoisted dense indicators —
             structure is a fast path, never a correctness requirement.

        Returns (entries, orphan_factor): wires left with no entry, no output
        position and no row are free summations of 1 and contribute their
        size as a multiplicative factor.
        """
        all_wires = {w for _, subs, _ in entries for w in subs} | set(out_subs)
        for coeffs, _const in rows:
            all_wires |= set(coeffs)
        consumed: set = set()

        rows = [("eq", dict(coeffs), const) for coeffs, const in rows]
        rows, unsat = _normalize_rows(rows, wire_dims, dim_of)

        if AFFINE_FAST and not unsat:
            progress = True
            while progress and rows:
                progress = False
                # 1. rewrite a dense operand axis through an eq row
                for ri in range(len(rows)):
                    if rows[ri][0] != "eq":
                        continue
                    x = self._try_eliminate_row(entries, rows, ri, out_subs, wire_dims, dim_of, emit_cached)
                    if x is not None:
                        consumed.add(x)
                        rows.pop(ri)
                        progress = True
                        break
                if progress:
                    continue
                # 2. free-wire summation: sum_x [x == e] -> [0 <= e < X],
                #    Gaussian-eliminating x from every other row first.
                for ri in range(len(rows)):
                    kind, coeffs, const = rows[ri]
                    if kind != "eq":
                        continue
                    x = next(
                        (
                            w
                            for w, c in coeffs.items()
                            if abs(c) == 1
                            and w not in out_subs
                            and not any(w in s for _, s, _ in entries)
                        ),
                        None,
                    )
                    if x is None:
                        continue
                    ax, b = coeffs[x], const
                    for rj in range(len(rows)):
                        if rj == ri or x not in rows[rj][1]:
                            continue
                        r = rows[rj]
                        c2 = dict(r[1])
                        f = c2.pop(x) * ax
                        for w, aw in coeffs.items():
                            if w != x:
                                c2[w] = c2.get(w, 0) - f * aw
                        c2 = {w: c for w, c in c2.items() if c != 0}
                        if r[0] == "eq":
                            rows[rj] = ("eq", c2, r[2] - f * b)
                        else:
                            rows[rj] = ("range", c2, r[2] + f * b, r[3])
                    rcoeffs = {w: -ax * aw for w, aw in coeffs.items() if w != x}
                    rows[ri] = ("range", rcoeffs, ax * b, dim_of(wire_dims[x]))
                    consumed.add(x)
                    rows, unsat = _normalize_rows(rows, wire_dims, dim_of)
                    progress = True
                    break
                if unsat:
                    break

        if unsat:
            # The indicator system is unsatisfiable: the whole term is zero.
            entries.append([const_name(torch.zeros((), dtype=self._dtype), "zero"), []])
            return entries, 1

        if rows:
            # Group leftover rows by shared wires; one dense indicator each.
            groups: list[list] = []
            for row in rows:
                ws = set(row[1])
                hit = [g for g in groups if any(ws & set(r[1]) for r in g)]
                newg = [row]
                for g in hit:
                    newg.extend(g)
                    groups.remove(g)
                groups.append(newg)
            for g in groups:
                wires = sorted({w for r in g for w in r[1]})
                sizes = [dim_of(wire_dims[w]) for w in wires]
                pos_rows = [
                    (r[0], {wires.index(w): c for w, c in r[1].items()}, *r[2:]) for r in g
                ]
                tensor = indicator_tensor(sizes, pos_rows).to(self._dtype)
                cname = const_name(tensor, "affine")
                entries.append([cname, list(wires), None])

        # Orphaned wires: summed over with no remaining constraint or operand
        # — each contributes a factor of its size (sum of the constant 1).
        live = {w for _, subs, _ in entries for w in subs} | set(out_subs) | consumed
        for r in rows:
            live |= set(r[1])
        orphan_factor = 1
        for w in all_wires - live:
            orphan_factor *= dim_of(wire_dims[w])
        return entries, orphan_factor

    def _try_eliminate_row(self, entries, rows, ri, out_subs, wire_dims, dim_of, emit_cached):
        _, coeffs, const = rows[ri]
        # Collect every eligible wire, then eliminate the CHEAPEST one. The
        # choice matters enormously: solving a conv row for the kernel wire
        # instead of the output wire creates an O(w_in*w_out) band view plus
        # a flip copy where the handwritten transpose-conv is O(w_in*k) —
        # measured 15x on 1D conv input-gradients at w=1024.
        candidates = []
        for x, cx in coeffs.items():
            if abs(cx) != 1 or x in out_subs:
                continue
            if any(x in rows[k][1] for k in range(len(rows)) if k != ri):
                continue
            holders = [j for j, (_, s2, _) in enumerate(entries) if x in s2]
            if not holders or any(entries[j][1].count(x) != 1 for j in holders):
                continue

            # Solve the row for x:  x = k0 + sum a_w * w. Every operand
            # carrying x gets the same affine view axes (shared einsum
            # labels), so multi-holder elimination is sound:
            # sum_x A[x] B[x] [x == e] = A[e] * B[e].
            k0 = const * cx
            terms = [(w, -cw * cx) for w, cw in coeffs.items() if w != x]
            X = dim_of(wire_dims[x])
            m = k0 + sum(min(0, a * (dim_of(wire_dims[w]) - 1)) for w, a in terms)
            M = k0 + sum(max(0, a * (dim_of(wire_dims[w]) - 1)) for w, a in terms)
            pf, pb = max(0, -m), max(0, M - (X - 1))

            cost = 0
            for j in holders:
                subs = entries[j][1]
                lead_sizes = [dim_of(wire_dims[subs[p]]) for p in range(len(subs)) if subs[p] != x]
                c = _prod(lead_sizes) * _prod([dim_of(wire_dims[w]) for w, _ in terms])
                if pf or pb:
                    c *= 4  # the pad (and any flip) copies the view
                cost += c
            candidates.append((cost, x, holders, k0, terms, X, m, M, pf, pb))

        if not candidates:
            return None
        _, x, holders, k0, terms, X, m, M, pf, pb = min(candidates, key=lambda c: c[0])

        for j in holders:
            nm, subs, _ = entries[j]
            d = subs.index(x)
            lead = [p for p in range(len(subs)) if p != d]
            sizes = [dim_of(wire_dims[subs[p]]) for p in lead] + [dim_of(wire_dims[w]) for w, _ in terms]

            # All planning below is spec-time when STATIC_STRIDES: strides and
            # offsets become INTEGER LITERALS in the generated code, so dynamo
            # never sees a .stride()/.storage_offset() call (torch ops
            # returning python ints break torch.compile(fullgraph=True)).
            stride_ints = off_int = None
            if pf or pb:
                # Out-of-range accesses must read 0: move the axis last, pad,
                # then view the (contiguous, spec-time-known) padded tensor.
                mv = nm if d == len(subs) - 1 else f"{nm}.movedim({d}, -1)"
                base = emit_cached(f"torch.nn.functional.pad({mv}, ({pf}, {pb}))", "ap")
                padded = tuple([dim_of(wire_dims[subs[p]]) for p in lead] + [X + pf + pb])
                rm = _row_major(padded)
                self._layouts[base] = (padded, rm, 0)
                stride_ints = [rm[i] for i in range(len(lead))] + [abs(a) for _, a in terms]
                off_int = m + pf
                stride_exprs, offset = map(str, stride_ints), str(off_int)
            elif STATIC_STRIDES:
                base = nm
                layout = self._layouts.get(base)
                if layout is None:
                    # Unknown layout (inputs, einsum/map outputs — pointwise
                    # kernels follow their inputs' layout, so "fresh" does NOT
                    # imply row-major): normalize once. A no-op for already-
                    # contiguous operands; where it copies, the copy is the
                    # price of a literal-stride, fullgraph-traceable view.
                    base = emit_cached(f"{nm}.contiguous()", "ac")
                    csizes = tuple(dim_of(wire_dims[s]) for s in subs)
                    # contiguous() preserves a nonzero storage offset when it
                    # no-ops, so the offset stays unknown (None) — handled by
                    # offset inheritance below.
                    layout = (csizes, _row_major(csizes), None)
                    self._layouts[base] = layout
                _, lstr, loff = layout
                sd_int = lstr[d]
                stride_ints = [lstr[p] for p in lead] + [abs(a) * sd_int for _, a in terms]
                stride_exprs = map(str, stride_ints)
                if loff is not None:
                    off_int = loff + m * sd_int
                    offset = str(off_int)
                else:
                    # Offset unknown: shift the base pointer with narrow (a
                    # view; offset += m * stride(d)) and let as_strided inherit
                    # the base's storage offset (storage_offset arg omitted).
                    if m:
                        base = emit_cached(f"{base}.narrow({d}, {m}, {X - m})", "an")
                    offset = None
            else:
                # Escape hatch (STATIC_STRIDES=False): a pure view on the
                # operand as it is, whatever its layout — strides are queried
                # at runtime, nothing is copied, but torch.compile graph-breaks.
                base = nm
                lead_strides = [f"{base}.stride({p})" for p in lead]
                sd = f"{base}.stride({d})"
                offset = f"{base}.storage_offset()" + (f" + {m} * {sd}" if m else "")
                stride_exprs = lead_strides + [f"{abs(a)} * {sd}" for _, a in terms]

            view = f"torch.as_strided({base}, {_tup(map(str, sizes))}, {_tup(stride_exprs)}"
            view += f", {offset})" if offset is not None else ")"
            vn = emit_cached(view, "av")
            if stride_ints is not None:
                self._layouts[vn] = (tuple(sizes), tuple(stride_ints), off_int)
            flips = [len(lead) + i for i, (_, a) in enumerate(terms) if a < 0]
            if flips:
                arg = str(flips[0]) if len(flips) == 1 else str(tuple(flips))
                # flip copies, but into a layout-FOLLOWING fresh tensor (not
                # necessarily row-major): no layout entry for the result.
                vn = emit_cached(f"{vn}.flip({arg})", "af")

            entries[j] = [vn, [subs[p] for p in lead] + [w for w, _ in terms], None]
        return x

    # ---- node emitters --------------------------------------------------

    def _build_const(self, node: ConstNode, dim_of):
        kind = node.kind
        if kind == "scalar":
            return None
        if kind == "zero":
            return torch.zeros([dim_of(d) for d in node.dims], dtype=self._dtype)
        if kind == "delta":
            size, k = node.params
            n = dim_of(size)
            out = torch.zeros([n] * k, dtype=self._dtype)
            idx = torch.arange(n)
            out[(idx,) * k] = 1.0
            return out
        if kind == "reshape":
            sizes = [dim_of(d) for d in node.dims]
            total = 1
            for s in sizes:
                total *= s
            half = int(total**0.5)
            if half * half != total:
                raise ValueError(f"Reshape constant is not a perfect square: {total}")
            return torch.eye(half, dtype=self._dtype).reshape(sizes)
        raise NotImplementedError(f"const kind {kind}")

    def _emit_einsum(self, node: EinsumNode, name, names, dim_of, dims, step_cache, const_name) -> list[str]:
        lines = []
        weight = node.weight if node.weight != 1 else None
        wf = to_float(node.weight.subs(dims)) if weight is not None else 1.0
        # Physical output order: the wire of each physical output position.
        pout = [node.out_subs[j] for j in self._phys_of(node)]

        if not node.ops and not node.constraints:
            # Pure constant: weight * ones(broadcast dims), as an expanded view.
            sizes = ", ".join(str(dim_of(node.wire_dims[w])) for w in pout)
            w = wf if weight is not None else 1.0
            if pout:
                lines.append(f"{name} = torch.full(({sizes},), {_fmt_weight(w)}, dtype={self._dtype_str})")
            else:
                lines.append(f"{name} = {_fmt_weight(w)}")
            return lines

        def emit_cached(expr: str, base: str) -> str:
            """Emit an assignment, deduplicated across the whole program."""
            key = ("raw", expr)
            if (hit := step_cache.get(key)) is not None:
                return hit
            tmp = f"_{base}{len(step_cache)}"
            lines.append(f"{tmp} = {expr}")
            step_cache[key] = tmp
            return tmp

        def do_step(step_eq: str, args: list[str], sizes=None) -> str:
            ins, out = step_eq.split("->")
            if len(args) == 1 and ins == out and len(set(out)) == len(out):
                return args[0]  # identity einsum: nothing to do (avoids a copy)
            key = (_canon_eq(step_eq), tuple(args))
            if (hit := step_cache.get(key)) is not None:
                return hit
            tmp = f"_e{len(step_cache)}"
            expr = self._matmul_expr(step_eq, args, sizes.__getitem__) if sizes else None
            if expr is None:
                expr = self._step_expr(step_eq, args)
            if expr is None:
                expr = f"torch.einsum('{step_eq}', {', '.join(args)})"
            lines.append(f"{tmp} = {expr}")
            step_cache[key] = tmp
            return tmp

        # (operand name, wire subscripts) pairs, subscripts in each operand's
        # PHYSICAL axis order; affine constraint rows are statically
        # eliminated into strided views of their neighbours.
        rows = [
            (
                {w: int(sympy.sympify(c).subs(dims)) for w, c in coeffs},
                int(sympy.sympify(const).subs(dims)),
            )
            for coeffs, const in node.constraints
        ]
        entries = [
            [names[id(op)], self._entry_subs(op, subs), op]
            for op, subs in zip(node.ops, node.in_subs)
        ]
        entries, orphan_factor = self._eliminate_affine(
            entries, rows, node.out_subs, node.wire_dims, dim_of, emit_cached, const_name
        )
        if orphan_factor != 1:
            wf *= orphan_factor
            weight = True  # force the scalar multiply below

        if not entries:
            # Everything eliminated algebraically: the result is a constant
            # (e.g. sum over a structure tensor = its solution count).
            w = wf if weight is not None else 1.0
            if pout:
                sizes = ", ".join(str(dim_of(node.wire_dims[w2])) for w2 in pout)
                lines.append(f"{name} = torch.full(({sizes},), {_fmt_weight(w)}, dtype={self._dtype_str})")
            else:
                lines.append(f"{name} = {_fmt_weight(w)}")
            return lines

        # Scatter peephole: an einsum that keeps a one-hot's class wire and
        # contracts away all of its idx wires is the embedding-gradient
        # pattern; emit it as zeros().index_add_ instead of a dense one-hot
        # contraction (O(N*D) writes instead of O(N*V*D) flops).
        self._try_scatter(node, entries, dim_of, lines, step_cache, names, do_step, pout)

        op_wires = {w for _, subs, _ in entries for w in subs}
        core_out = tuple(w for w in pout if w in op_wires)
        broadcast = [w for w in pout if w not in op_wires]

        expr = self._contract(
            [(nm, subs) for nm, subs, _ in entries], core_out, node.wire_dims, dim_of, do_step
        )

        if weight is not None:
            # Apply the weight before any broadcast-expand, so the expand
            # below stays a zero-copy view.
            expr = f"({_fmt_weight(wf)} * {expr})"

        if broadcast:
            # Result axes: core_out order; unsqueeze+expand to the full
            # physical order (an expand is a view — no memory is written).
            if not core_out:
                # A fully-scalar core may be a python float (scalar-const
                # emission); as_tensor is a no-op on real 0-dim tensors.
                expr = f"torch.as_tensor({expr}, dtype={self._dtype_str})"
            unsq = expr
            positions = [i for i, w in enumerate(pout) if w not in op_wires]
            for p in positions:
                unsq = f"{unsq}.unsqueeze({p})"
            sizes = ", ".join(
                str(dim_of(node.wire_dims[w])) if i in positions else "-1"
                for i, w in enumerate(pout)
            )
            expr = f"{unsq}.expand({sizes})"

        lines.append(f"{name} = {expr}")
        return lines

    @staticmethod
    def _step_expr(step_eq: str, args: list[str]):
        """Cheaper-dispatch emission for einsum steps that need no contraction
        machinery. Returns an expression string, or None (einsum fallback).

        - single operand: plain axis reductions become .sum(dim=...), pure
          permutations become .permute(...) views (einsum would copy);
        - multi operand, no summed index: a broadcast `*` chain over
          permute/unsqueeze views (one TensorIterator kernel, no einsum
          parse/plan dispatch per call).
        """
        ins, out = step_eq.split("->")
        in_eqs = ins.split(",")
        if len(set(out)) != len(out) or any(len(set(e)) != len(e) for e in in_eqs):
            return None  # diagonals: einsum territory
        if len(in_eqs) == 1:
            e = in_eqs[0]
            if not set(out) <= set(e):
                return None  # broadcast is handled at the node level
            red = [i for i, ch in enumerate(e) if ch not in out]
            kept = [ch for ch in e if ch in out]
            expr = args[0]
            if red:
                expr = f"{expr}.sum(dim={_tup(map(str, red))})"
            if kept != list(out):
                perm = [kept.index(ch) for ch in out]
                expr = f"{expr}.permute({', '.join(map(str, perm))})"
            return expr
        if set().union(*map(set, in_eqs)) != set(out):
            # Hadamard-then-reduce: both operands span the SAME index set and
            # some indices are summed. torch.einsum lowers this to a batched
            # matmul of degenerate (n,1)x(1,m) blocks — measured 323us vs 33us
            # for (A*B).sum(1) at (256,1024). Scalar output is excluded: the
            # flattened-dot path einsum picks there is the fast one.
            if len(in_eqs) == 2 and out:
                e1, e2 = in_eqs
                if set(e1) == set(e2) and set(out) < set(e1):
                    perm2 = [e2.index(ch) for ch in e1]
                    v2 = args[1]
                    if perm2 != list(range(len(e2))):
                        v2 = f"{v2}.permute({', '.join(map(str, perm2))})"
                    red = [i for i, ch in enumerate(e1) if ch not in out]
                    expr = f"({args[0]} * {v2}).sum(dim={_tup(map(str, red))})"
                    kept = [ch for ch in e1 if ch in out]
                    if kept != list(out):
                        perm = [kept.index(ch) for ch in out]
                        expr = f"({expr}).permute({', '.join(map(str, perm))})"
                    return expr
            return None  # a contracted index: real einsum work
        views = []
        for e, a in zip(in_eqs, args):
            order = sorted(range(len(e)), key=lambda i: out.index(e[i]))
            v = a
            if order != list(range(len(e))):
                v = f"{a}.permute({', '.join(map(str, order))})"
            if e:  # 0-dim (possibly python-float) operands broadcast as-is
                for j, ch in enumerate(out):
                    if ch not in e:
                        v = f"{v}.unsqueeze({j})"
            views.append(v)
        return " * ".join(views)

    def _contract(self, entries, out_ws, wire_dims, dim_of, do_step) -> str:
        """Contract (name, subs) operand pairs down to the wires `out_ws`
        (in order), emitting explicit pairwise torch.einsum steps.

        The contraction order is planned ONCE with concrete shapes. The path
        search is memory-bounded: opt_einsum otherwise happily picks an
        equal-FLOP order with intermediates orders of magnitude larger than
        any input (measured: 268MB intermediates in the MLP softmax grad).
        """
        all_wires = sorted({w for _, subs in entries for w in subs} | set(out_ws))
        if len(all_wires) > len(_LETTERS):
            raise ValueError(f"Einsum with {len(all_wires)} distinct indices exceeds the {len(_LETTERS)} limit")
        letters = {w: _LETTERS[i] for i, w in enumerate(all_wires)}
        lsizes = {letters[w]: dim_of(wire_dims[w]) for w in all_wires}
        in_eqs = ["".join(letters[w] for w in subs) for _, subs in entries]
        out_eq = "".join(letters[w] for w in out_ws)
        op_names = [nm for nm, _ in entries]

        if len(entries) <= 2:
            return do_step(f"{','.join(in_eqs)}->{out_eq}", op_names, lsizes)

        shapes = [tuple(dim_of(wire_dims[w]) for w in subs) for _, subs in entries]
        eq = f"{','.join(in_eqs)}->{out_eq}"
        sizes = [1] + [s for shape in shapes for s in [_prod(shape)]]
        out_size = _prod([dim_of(wire_dims[w]) for w in out_ws])
        limit = 4 * max(sizes + [out_size])
        path_info = None
        if len(shapes) <= 16:
            # DP with combined flops+memory objective finds equal-FLOP
            # orders with orders-of-magnitude smaller intermediates
            # (e.g. 268MB -> 4MB on the MLP softmax gradient). The cap is 16
            # (not 12) because factor.py's adjoint chain collapse emits
            # 11-16 operand einsums for deep-parameter gradients, and only
            # the DP objective reliably finds their cotangent-first path.
            try:
                _, path_info = oe.contract_path(
                    eq, *shapes, shapes=True, optimize=oe.DynamicProgramming(minimize="combo")
                )
            except Exception:
                path_info = None
        if path_info is None:
            try:
                _, path_info = oe.contract_path(
                    eq, *shapes, shapes=True, optimize="auto", memory_limit=limit
                )
            except Exception:
                _, path_info = oe.contract_path(eq, *shapes, shapes=True, optimize="auto")
        # opt_einsum's contraction_list positions come reverse-sorted, and
        # its step equations expect operands in exactly that popped order.
        stack = list(op_names)
        for (pos, _, step_eq, _, _) in path_info.contraction_list:
            args = [stack.pop(p) for p in pos]
            stack.append(do_step(step_eq, args, lsizes))
        return stack[0]

    def _try_scatter(self, node: EinsumNode, entries, dim_of, lines, step_cache, names, do_step, pout=None) -> bool:
        """Rewrite `entries` in place when the einsum matches the scatter
        pattern: one operand is a one-hot whose class wire is a free output
        and whose idx wires are all contracted with the other operands.
        The other operands are contracted down to (idx wires, kept wires) and
        scattered with index_add_; the dense one-hot is never materialized
        (its deferred emission line is dropped as dead code)."""
        out_subs = node.out_subs
        if pout is None:
            pout = list(out_subs)
        for j, (_, subs, op) in enumerate(entries):
            if not (isinstance(op, GatherNode) and op.op == "one_hot"):
                continue
            # `subs` is in the one-hot's physical order = logical (pinned):
            # (class wire, idx wires in idx-LOGICAL axis order). index_add_
            # flattens the idx tensor in its PHYSICAL layout, so reorder.
            idx_phys = self._phys_of(op.ops[0])
            v, idx_ws = subs[0], [subs[1 + q] for q in idx_phys]
            if len(set(subs)) != len(subs):
                continue  # diagonal one-hot: keep the dense fallback
            if out_subs.count(v) != 1 or any(w in out_subs for w in idx_ws):
                continue
            remaining = [e for k, e in enumerate(entries) if k != j]
            if not remaining:
                continue
            rem_wires = {w for _, s, _ in remaining for w in s}
            if v in rem_wires or not set(idx_ws) <= rem_wires:
                continue
            tail_ws = [w for w in pout if w != v and w in rem_wires]
            # Contract everything else down to (idx wires..., kept out wires...).
            rest_out = idx_ws + tail_ws
            rest = self._contract(
                [(nm, s) for nm, s, _ in remaining], rest_out, node.wire_dims, dim_of, do_step
            )
            idx_name = names[id(op.ops[0])]
            num_classes = dim_of(op.dims[0])
            n = _prod([dim_of(node.wire_dims[w]) for w in idx_ws])
            tail_sizes = [dim_of(node.wire_dims[w]) for w in tail_ws]
            tail = _prod(tail_sizes)
            key = ("scatter", idx_name, rest, num_classes)
            if (zn := step_cache.get(key)) is None:
                zn = f"_s{len(step_cache)}"
                lines.append(f"{zn} = torch.zeros(({num_classes}, {tail}), dtype={rest}.dtype)")
                lines.append(
                    f"{zn}.index_add_(0, {idx_name}.reshape(-1).long(), {rest}.reshape({n}, {tail}))"
                )
                step_cache[key] = zn
            view = f"{zn}.view({_tup(map(str, [num_classes] + tail_sizes))})"
            entries[:] = [[view, [v] + tail_ws, None]]
            return True
        return False

    # ---- SDPA peephole ---------------------------------------------------

    def _plan_sdpa(self, order, output_ids, dims):
        """Find fusable attention patterns. Returns (plans, suppressed):
        plans maps id(final einsum) -> plan dict for _emit_sdpa; suppressed is
        the set of node ids consumed exclusively by a fused SDPA call."""
        if not SDPA_FUSION:
            return {}, set()
        consumers: dict[int, int] = {}
        for n in order:
            for op in n.operands():
                consumers[id(op)] = consumers.get(id(op), 0) + 1
        plans, suppressed = {}, set()
        for node in order:
            plan = self._match_sdpa(node, consumers, output_ids, dims)
            if plan is None:
                continue
            # Patterns must not overlap: an already-fused node cannot be an
            # intermediate of a second pattern, and a plan must not reference
            # (as Q/K/V/mask) a node another plan suppressed.
            refs = [plan["q"], plan["k"], plan["v"]] + ([plan["mask"]] if "mask" in plan else [])
            if plan["suppress"] & (plans.keys() | suppressed):
                continue
            if any(id(r) in suppressed for r in refs):
                continue
            plans[id(node)] = plan
            suppressed |= plan["suppress"]
        return plans, suppressed

    def _match_sdpa(self, e2, consumers, output_ids, dims):
        """Match e2 = einsum(softmax(scale * einsum(Q,K) [+ mask]), V) with all
        intermediates single-consumer, one contracted feature wire, one query
        wire, one key wire, one value-feature wire and shared batch wires.
        Returns a plan dict, or None (silent fallback to the generic path)."""
        if not isinstance(e2, EinsumNode) or e2.constraints or len(e2.ops) != 2:
            return None
        if len(set(e2.out_subs)) != len(e2.out_subs):
            return None
        for i, S in enumerate(e2.ops):
            if not (isinstance(S, ReduceNode) and S.op == "softmax" and len(S.axes) == 1):
                continue
            V, v_subs = e2.ops[1 - i], list(e2.in_subs[1 - i])
            att_subs = list(e2.in_subs[i])
            if V is S or len(set(att_subs)) != len(att_subs) or len(set(v_subs)) != len(v_subs):
                continue
            if consumers.get(id(S)) != 1 or id(S) in output_ids:
                continue

            # Walk down the scale/mask chain between softmax and the QK einsum.
            # amap[j] = axis of `cur` feeding att axis j of the softmax operand.
            cur = S.ops[0]
            amap = list(range(cur.order))
            chain_ids: list[int] = []
            scale = sympy.Integer(1)
            mask = None  # (mask_node, mask_axis: att axis j -> mask axis, weight)
            ok = True
            for _ in range(4):
                if not isinstance(cur, LinearNode):
                    break
                if consumers.get(id(cur)) != 1 or id(cur) in output_ids:
                    ok = False
                    break
                if len(cur.terms) == 1:
                    chain_ids.append(id(cur))
                    scale *= sympy.sympify(cur.weights[0])
                    amap = [cur.perms[0][a] for a in amap]
                    cur = cur.terms[0]
                elif len(cur.terms) == 2 and mask is None and scale == 1:
                    # mask add directly under the softmax (before any outer
                    # scale, so the mask is not rescaled behind our back).
                    cont = next(
                        (
                            t
                            for t in range(2)
                            if isinstance(cur.terms[t], EinsumNode)
                            and len(cur.terms[t].ops) == 2
                            and not cur.terms[t].constraints
                        ),
                        None,
                    )
                    if cont is None:
                        ok = False
                        break
                    m = 1 - cont
                    mask = (
                        cur.terms[m],
                        [cur.perms[m][a] for a in amap],
                        sympy.sympify(cur.weights[m]),
                    )
                    chain_ids.append(id(cur))
                    scale *= sympy.sympify(cur.weights[cont])
                    amap = [cur.perms[cont][a] for a in amap]
                    cur = cur.terms[cont]
                else:
                    ok = False
                    break
            e1 = cur
            if (
                not ok
                or not isinstance(e1, EinsumNode)
                or e1.constraints
                or len(e1.ops) != 2
                or consumers.get(id(e1)) != 1
                or id(e1) in output_ids
                or len(set(e1.out_subs)) != len(e1.out_subs)
            ):
                continue
            scale *= e1.weight

            # ---- validate the QK einsum (E1 wire space) ----
            s1a, s1b = (list(s) for s in e1.in_subs)
            if len(set(s1a)) != len(s1a) or len(set(s1b)) != len(s1b):
                continue
            outw = set(e1.out_subs)
            if not (set(s1a) | set(s1b)) >= outw:
                continue  # broadcast-only out wires: not plain attention
            contracted = (set(s1a) | set(s1b)) - outw
            if len(contracted) != 1:
                continue
            (cw,) = contracted
            if cw not in s1a or cw not in s1b:
                continue
            ax = S.axes[0]  # softmax axis == key axis of the scores
            skw = e1.out_subs[amap[ax]]
            if skw in s1a and skw in s1b:
                continue
            qi = 0 if skw in s1b else 1  # Q = operand NOT carrying the key wire
            q_subs, k_subs = (s1a, s1b) if qi == 0 else (s1b, s1a)
            sq_cands = [w for w in e1.out_subs if w in q_subs and w not in k_subs]
            if len(sq_cands) != 1:
                continue
            (sqw,) = sq_cands
            batch1 = [w for w in e1.out_subs if w not in (sqw, skw)]
            if any(w not in q_subs or w not in k_subs for w in batch1):
                continue

            # ---- validate the AV einsum (E2 wire space) ----
            # att axis j carries E2 wire att_subs[j] and E1 wire e1.out_subs[amap[j]].
            e1w_of_att = [e1.out_subs[amap[j]] for j in range(len(att_subs))]
            sk2 = att_subs[ax]
            j_sq = e1w_of_att.index(sqw)
            sq2 = att_subs[j_sq]
            batch2 = {att_subs[j] for j in range(len(att_subs)) if j not in (ax, j_sq)}
            dv = [w for w in v_subs if w not in att_subs]
            if len(dv) != 1:
                continue  # SDPA needs exactly one value-feature axis
            if sk2 not in v_subs or sk2 in e2.out_subs or sq2 in v_subs:
                continue
            if set(v_subs) != batch2 | {sk2, dv[0]}:
                continue  # V must carry exactly the batch wires + key + feature
            if set(e2.out_subs) != batch2 | {sq2, dv[0]}:
                continue

            # ---- permutations to (B..., S, E) layouts ----
            b2_order = [w for w in e2.out_subs if w in batch2]
            att_axis_of_e2w = {w: j for j, w in enumerate(att_subs)}
            b1_order = [e1w_of_att[att_axis_of_e2w[w]] for w in b2_order]
            plan = {
                "q": e1.ops[qi],
                "k": e1.ops[1 - qi],
                "v": V,
                "q_perm": tuple(q_subs.index(w) for w in b1_order + [sqw, cw]),
                "k_perm": tuple(k_subs.index(w) for w in b1_order + [skw, cw]),
                "v_perm": tuple(v_subs.index(w) for w in b2_order + [sk2, dv[0]]),
                "scale": to_float(scale.subs(dims)),
                "out_perm": tuple(
                    (b2_order + [sq2, dv[0]]).index(w) for w in e2.out_subs
                ),
                "weight": to_float(e2.weight.subs(dims)),
                "suppress": {id(S), id(e1)} | set(chain_ids),
            }
            if mask is not None:
                m_node, m_axis, m_w = mask
                if m_node.order != len(att_subs):
                    continue
                target_att_axes = [att_axis_of_e2w[w] for w in b2_order] + [j_sq, ax]
                plan["mask"] = m_node
                plan["mask_perm"] = tuple(m_axis[j] for j in target_att_axes)
                plan["mask_weight"] = to_float(sympy.sympify(m_w).subs(dims))
            return plan
        return None

    def _emit_sdpa(self, plan, name, names, node) -> str:
        def permuted(op_node, perm):
            # `perm` is in the operand's LOGICAL axes; compose with its
            # assigned physical layout to index the emitted tensor.
            phys = self._phys_of(op_node)
            actual = tuple(phys.index(a) for a in perm)
            return self._perm_str(names[id(op_node)], actual)

        args = [permuted(plan[t], plan[f"{t}_perm"]) for t in ("q", "k", "v")]
        if "mask" in plan:
            m = permuted(plan["mask"], plan["mask_perm"])
            if plan["mask_weight"] != 1.0:
                m = f"({_fmt_weight(plan['mask_weight'])} * {m})"
            args.append(f"attn_mask={m}")
        args.append(f"scale={_fmt_weight(plan['scale'])}")
        expr = f"torch.nn.functional.scaled_dot_product_attention({', '.join(args)})"
        if plan["weight"] != 1.0:
            expr = f"{_fmt_weight(plan['weight'])} * {expr}"
        # SDPA produces (batch..., seq_q, dv); out_perm[i] = produced position
        # of logical axis i. Land it directly in OUR assigned layout.
        out_perm = plan["out_perm"]
        physN = self._phys_of(node)
        final = tuple(out_perm[j] for j in physN)
        if final != tuple(range(len(final))):
            expr = f"({expr}).permute({', '.join(map(str, final))})"
        return f"{name} = {expr}"

    def _to_logical(self, node: Node, names) -> str:
        """Reference a node's tensor in LOGICAL axis order (a permute view
        when its assigned layout differs; gather-family consumers index
        logically)."""
        phys = self._phys_of(node)
        inv = tuple(phys.index(j) for j in range(node.order))
        return self._perm_str(names[id(node)], inv)

    def _emit_gather(self, node: GatherNode, name, names, dim_of) -> str:
        table = self._to_logical(node.ops[0], names)
        idx = self._to_logical(node.ops[1], names)
        sizes = [str(dim_of(d)) for d in node.dims]
        return (
            f"{name} = torch.index_select({table}, {node.axis}, "
            f"{idx}.reshape(-1).long()).view({_tup(sizes)})"
        )

    def _emit_one_hot(self, node: GatherNode, name, names, dim_of) -> str:
        idx = self._to_logical(node.ops[0], names)
        num_classes = dim_of(node.dims[0])
        sizes = [str(num_classes)] + [str(dim_of(d)) for d in node.dims[1:]]
        return (
            f"{name} = ({idx}.reshape(1, -1) == torch.arange({num_classes}, "
            f"dtype={idx}.dtype).unsqueeze(1)).to({idx}.dtype).view({_tup(sizes)})"
        )

    def _emit_linear(self, node: LinearNode, name, names, dims, dim_of) -> str:
        physN = self._phys_of(node)
        plan = getattr(self, "_addmm_plans", {}).get(id(node))
        parts = []
        const_acc = 0.0

        def weight_of(w):
            return to_float(sympy.sympify(w).subs(dims) if getattr(w, "free_symbols", None) else w)

        if plan is not None:
            # Fused x@W + bias: torch.addmm(bias, A, B, beta, alpha). The
            # matmul term was suppressed; the bias enters RAW (pre-broadcast:
            # addmm broadcasts it over the flattened M rows).
            term = node.terms[plan["term_idx"]]
            alpha = weight_of(node.weights[plan["term_idx"]]) * plan["alpha_extra"]
            beta = weight_of(node.weights[plan["bias_idx"]]) * plan["beta_extra"]
            fit = plan["fit"]
            bop = plan["bias_op"]
            bias = names[id(bop)]
            if plan["bias_side"] == "m":  # per-row bias: (Msz, 1)
                bias = f"{bias}.reshape({_tup([str(fit['msz']), '1'])})"
            elif bop.order != 1:  # per-column bias: flatten to (Nsz,)
                bias = f"{bias}.reshape({fit['nsz']})"
            margs = [names[id(op)] for op in term.ops]
            va, vb, view = self._matmul_operands(fit, margs)
            kw = ""
            if beta != 1.0:
                kw += f", beta={_fmt_weight(beta)}"
            if alpha != 1.0:
                kw += f", alpha={_fmt_weight(alpha)}"
            parts.append(f"torch.addmm({bias}, {va}, {vb}{kw}){view}")

        for i, (term, perm, w) in enumerate(zip(node.terms, node.perms, node.weights)):
            if plan is not None and i in (plan["term_idx"], plan["bias_idx"]):
                continue
            wf = weight_of(w)
            if isinstance(term, EinsumNode) and not term.ops and not term.constraints:
                # Pure-ones term: a scalar offset (TensorIterator broadcasts it
                # in the same kernel — no torch.full materialization).
                const_acc += wf * to_float(term.weight.subs(dims))
                continue
            # Align the term to the node's physical layout: physical position
            # p holds node-logical axis physN[p] = term-logical perm[physN[p]].
            want = tuple(perm[j] for j in physN)
            physT = self._phys_of(term)
            tname = self._perm_str(names[id(term)], tuple(physT.index(a) for a in want))
            if wf == 1.0:
                parts.append(f"+ {tname}" if parts else tname)
            elif wf == -1.0:
                parts.append(f"- {tname}" if parts else f"-{tname}")
            else:
                parts.append(f"{'+ ' if parts else ''}{_fmt_weight(wf)} * {tname}")
        if not parts:
            sizes = _tup(str(dim_of(node.dims[j])) for j in physN)
            return f"{name} = torch.full({sizes}, {_fmt_weight(const_acc)}, dtype={self._dtype_str})"
        if const_acc > 0:
            parts.append(f"+ {_fmt_weight(const_acc)}")
        elif const_acc < 0:
            parts.append(f"- {_fmt_weight(-const_acc)}")
        return f"{name} = {' '.join(parts)}"

    def _emit_map(self, node: MapNode, name, names) -> str:
        physN = self._phys_of(node)
        args = []
        for opnd, perm in zip(node.ops, node.perms):
            want = tuple(perm[j] for j in physN)
            physT = self._phys_of(opnd)
            args.append(self._perm_str(names[id(opnd)], tuple(physT.index(a) for a in want)))
        op = node.op
        if op == "exp":
            expr = f"torch.exp({args[0]})"
        elif op == "log":
            expr = f"torch.log({args[0]})"
        elif op == "relu":
            expr = f"torch.relu({args[0]})"
        elif op == "sign":
            expr = f"torch.sign({args[0]})"
        elif op == "tanh":
            expr = f"torch.tanh({args[0]})"
        elif op == "erf":
            expr = f"torch.erf({args[0]})"
        elif op == "abs":
            expr = f"torch.abs({args[0]})"
        elif op == "gt0":
            expr = f"({args[0]} >= 0).to({args[0]}.dtype)"
        elif op == "pow":
            (k,) = node.params
            kf = to_float(k)
            if kf == -1.0:
                expr = f"torch.reciprocal({args[0]})"
            elif kf == 2.0:
                expr = f"torch.square({args[0]})"
            elif kf == 0.5:
                expr = f"torch.sqrt({args[0]})"
            elif kf == -0.5:
                expr = f"torch.rsqrt({args[0]})"
            else:
                expr = f"torch.pow({args[0]}, {kf})"
        elif op == "equal":
            expr = f"({args[0]} == {args[1]}).to({args[0]}.dtype)"
        else:
            raise NotImplementedError(f"map op {op}")
        return f"{name} = {expr}"

    def _emit_reduce(self, node: ReduceNode, name, names) -> str:
        (opnd,) = node.ops
        a = names[id(opnd)]
        physO = self._phys_of(opnd)
        physN = self._phys_of(node)
        if node.op in ("argmax", "max"):
            # Reduce the operand as it lies; permute-view the result to the
            # assigned layout if needed (the layout pass votes the operand
            # into "kept-in-output-order + reduced-last", making this identity).
            dims_phys = sorted(physO.index(x) for x in node.axes)
            kept_log = [x for x in range(opnd.order) if x not in node.axes]
            renum = {x: j for j, x in enumerate(kept_log)}  # operand ax -> node ax
            produced = tuple(renum[x] for x in physO if x not in node.axes)
            view = tuple(produced.index(a2) for a2 in physN)
            if node.op == "argmax":
                (axis,) = dims_phys
                return f"{name} = {self._perm_str(f'{a}.argmax(dim={axis})', view)}.to({a}.dtype)"
            axes = ", ".join(map(str, dims_phys))
            return f"{name} = {self._perm_str(f'torch.amax({a}, dim=({axes},))', view)}"
        if node.op in ("softmax", "log_softmax"):
            # node axes == operand axes; align the operand to OUR layout with
            # a view, then reduce over the physical positions of node.axes.
            a = self._perm_str(a, tuple(physO.index(physN[p]) for p in range(node.order)))
            dims_phys = sorted(physN.index(x) for x in node.axes)
            fn = "torch.softmax" if node.op == "softmax" else "torch.log_softmax"
            if len(node.axes) == 1:
                return f"{name} = {fn}({a}, dim={dims_phys[0]})"
            # Multi-axis: no native torch op; emit the numerically stable
            # max-subtracted form reducing jointly over all axes.
            dims = "(" + ", ".join(map(str, dims_phys)) + ",)"
            if node.op == "softmax":
                return (
                    f"{name} = torch.exp({a} - torch.amax({a}, dim={dims}, keepdim=True))\n"
                    f"{name} = {name} / {name}.sum(dim={dims}, keepdim=True)"
                )
            return (
                f"{name} = {a} - torch.amax({a}, dim={dims}, keepdim=True)\n"
                f"{name} = {name} - {name}.exp().sum(dim={dims}, keepdim=True).log()"
            )
        raise NotImplementedError(f"reduce op {node.op}")
