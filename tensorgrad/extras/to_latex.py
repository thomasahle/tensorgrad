# tensorgrad/serializers/to_latex.py

from functools import singledispatch

from tensorgrad.tensor import (
    Tensor,
    Variable,
    Zero,
    Delta,
    Sum,
    Product,
    Derivative,
    Rename,
)
import tensorgrad.functions as F


###############################################################################
# 1) Index Notation
###############################################################################
@singledispatch
def to_latex_indexed(expr: Tensor) -> str:
    """
    Convert Tensor to LaTeX in Einstein-style index notation.
    By default, raise if unhandled.
    """
    raise NotImplementedError(f"No index-based LaTeX for {type(expr).__name__}")


@to_latex_indexed.register
def _(expr: Variable) -> str:
    """If edges=(i,j) => X_{i,j}, else X if no edges."""
    es = list(expr.edges)
    if es:
        return f"{expr.name}_{{{','.join(map(str,es))}}}"
    return expr.name


@to_latex_indexed.register
def _(expr: Zero) -> str:
    es = list(expr.edges)
    if es:
        return rf"\mathbf{{0}}_{{{','.join(map(str,es))}}}"
    else:
        return r"\mathbf{0}"


@to_latex_indexed.register
def _(expr: Delta) -> str:
    es = list(expr.edges)
    if len(es) == 2 and es[0] != es[1]:
        return rf"\delta_{{{es[0]},{es[1]}}}"
    return rf"\text{{Delta}}_{{{','.join(map(str,es))}}}"


@to_latex_indexed.register
def _(expr: Rename) -> str:
    """Rename => X_{k,l} if base was X_{i,j} rename i->k, j->l."""
    base = expr.tensor
    es = list(expr.edges)
    if isinstance(base, Variable):
        name = base.name
    else:
        name = "Expr"
    if es:
        return f"{name}_{{{','.join(map(str,es))}}}"
    else:
        return name


@to_latex_indexed.register
def _(expr: Sum) -> str:
    parts = []
    for w, t in zip(expr.weights, expr.tensors):
        sign = "+" if w >= 0 else "-"
        factor = abs(w)
        sub = to_latex_indexed(t)
        if factor == 1:
            parts.append(f"{sign} {sub}")
        else:
            parts.append(f"{sign} {factor} {sub}")
    out = " ".join(parts).strip()
    if out.startswith("+"):
        out = out[1:].strip()
    return out


@to_latex_indexed.register
def _(expr: Product) -> str:
    """
    Just multiply => A_{i,j} B_{j,k} ...
    If your library expands trace(X) => X_{i} Delta_{i}, you see that directly.
    """
    subs = [to_latex_indexed(t) for t in expr.tensors]
    return " ".join(subs)


@to_latex_indexed.register
def _(expr: Derivative) -> str:
    return rf"\frac{{d\bigl({to_latex_indexed(expr.of)})\bigr}}{{d\bigl({to_latex_indexed(expr.x)})\bigr}}"


@to_latex_indexed.register
def _(expr: F.Function) -> str:
    """
    If trace => produce the diagonal form, e.g. X_{i,i}.
    (Or just show X_{i} if 1D.)
    But we do *not* produce \mathrm{tr}(\dots) here, per your statement that
    index-based trace is simply 'X_{i,i}' or 'X_{i,j} Y_{j,i}' etc.
    """
    fn = expr.signature.name
    inputs = expr.inputs
    if fn == "trace":
        # We just show the input as i->i. If the library expands, so be it.
        # For a single input X, we produce e.g. "X_{i,i}" if X is 2D square?
        # We'll do a naive approach: if there's exactly 1 input, transform edges => i,i
        if len(inputs) == 1 and isinstance(inputs[0], Variable):
            v = inputs[0]
            es = list(v.edges)
            if len(es) == 2 and es[0] == es[1]:
                # Already X_{i,i}
                return f"{v.name}_{{{es[0]},{es[0]}}}"
            elif len(es) == 2:
                # e.g. (i,j). We'll unify them => i,i
                return f"{v.name}_{{{es[0]},{es[0]}}}"
            elif len(es) == 1:
                # (i) => i,i
                return f"{v.name}_{{{es[0]},{es[0]}}}"
            else:
                # fallback if 0D or >2D
                return to_latex_indexed(v)
        else:
            # multiple inputs or not a variable => fallback
            inside = ", ".join(to_latex_indexed(a) for a in inputs)
            return f"{inside}_{{i,i}}"  # a hack
    else:
        subs = ", ".join(to_latex_indexed(a) for a in inputs)
        return rf"\mathrm{{{fn}}}\bigl({subs}\bigr)"


###############################################################################
# 2) Index-Free
###############################################################################
@singledispatch
def to_latex_index_free(expr: Tensor) -> str:
    """
    Convert to classical notation (e.g. 'X', 'X^T', 'AB', 'tr(X)')
    """
    raise NotImplementedError(f"No index-free for {type(expr).__name__}")


@to_latex_index_free.register
def _(expr: Variable) -> str:
    """
    For 2D if edges are reversed in alphabetical order => X^T, else 'X'.
    """
    name = expr.name
    es = list(expr.edges)
    if len(es) == 2:
        # sort them as strings:
        sorted_es = sorted(es, key=str)
        if es == sorted_es[::-1] and es != sorted_es:
            return f"{name}^T"
    return name


@to_latex_index_free.register
def _(expr: Zero) -> str:
    return r"\mathbf{0}"


@to_latex_index_free.register
def _(expr: Delta) -> str:
    es = list(expr.edges)
    if len(es) == 2 and es[0] == es[1]:
        return "I"
    return r"\mathrm{Delta}"


@to_latex_index_free.register
def _(expr: Rename) -> str:
    """
    We'll just do 'Rename(...)' since we can't guess if it's X^T w/o a custom field.
    If the base is 2D with reversed edges, the base's to_latex_index_free might produce X^T already.
    """
    base_str = to_latex_index_free(expr.tensor)
    return f"Rename({base_str})"


@to_latex_index_free.register
def _(expr: Sum) -> str:
    parts = []
    for w, t in zip(expr.weights, expr.tensors):
        sign = "+" if w >= 0 else "-"
        factor = abs(w)
        sub = to_latex_index_free(t)
        if factor == 1:
            parts.append(f"{sign} {sub}")
        else:
            parts.append(f"{sign} {factor}{sub}")
    out = " ".join(parts).strip()
    if out.startswith("+"):
        out = out[1:].strip()
    return out


@to_latex_index_free.register
def _(expr: Product) -> str:
    """
    e.g. "A B"
    """
    parts = [to_latex_index_free(t) for t in expr.tensors]
    return " ".join(parts)


@to_latex_index_free.register
def _(expr: Derivative) -> str:
    return r"\frac{d(\dots)}{d(\dots)}"


@to_latex_index_free.register
def _(expr: F.Function) -> str:
    """
    If trace => \mathrm{tr}(X).
    """
    fn = expr.signature.name
    inputs = expr.inputs
    if fn == "trace":
        inside = ", ".join(to_latex_index_free(a) for a in inputs)
        return rf"\mathrm{{tr}}({inside})"
    else:
        subs = ", ".join(to_latex_index_free(a) for a in inputs)
        return rf"{fn}\bigl({subs}\bigr)"


###############################################################################
# 3) Public
###############################################################################
def to_latex(expr: Tensor, index_free: bool = False) -> str:
    """
    If index_free=False => index-based notation.
    If index_free=True => matrix-style notation.
    """
    if index_free:
        return to_latex_index_free(expr)
    else:
        return to_latex_indexed(expr)
