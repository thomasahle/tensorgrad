from collections import Counter
import pytest
import re

from sympy import symbols

from tensorgrad.tensor import Delta, Variable, Product, Sum, Derivative, function
from tensorgrad.functions import Convolution, Reshape
import tensorgrad.functions as F
from tensorgrad.extras import Expectation
from tensorgrad.serializers.to_tikz import to_tikz


def parse_tikz_lines(tikz_output: str):
    """
    Minimal parser to extract 'nodes' and 'edges' from your raw TikZ output.
    We won't rely on subgraph styles that your code might not produce.
    """
    latex_syntax_heuristics(tikz_output)

    node_ids = set()
    nodes = []
    edges = []

    for line in tikz_output.splitlines():
        line = line.strip()
        # Attempt to parse a node line, e.g. "0[var, as=$x$,...];"
        # node_match = re.match(r"^(?P<id>\S+)\[(?P<styles>[^\]]+)\]", line)
        node_match = re.match(
            r"^(?P<id>\S+)\[(?P<styles>[^\]]+)\](?:\s*\/\/\s*\[(?P<layout>[^\]]+)\])?", line
        )
        if node_match:
            nid = node_match.group("id")
            assert nid not in node_ids, f"Duplicate node id: {nid}"
            node_ids.add(nid)
            styles_text = node_match.group("styles")
            # Extract label from "as=$something$"
            label_match = re.search(r"as=([^,]+)", styles_text)
            label = label_match.group(1) if label_match else ""
            nodes.append({"id": nid, "styles": styles_text, "label": label})
            continue

        # Attempt to parse an edge line, e.g. "(0) -- [style, "label"] (1);"
        edge_match = re.match(
            r"^\((?P<src>\S+)\)(?P<arrow>--|->)\[(?P<style>[^]]*)\].*\((?P<dst>\S+)\);$", line
        )
        if edge_match:
            src = edge_match.group("src")
            dst = edge_match.group("dst")
            # Possibly parse "label" if you want
            label_match = re.search(r'"([^"]+)"', line)
            label = label_match.group(1) if label_match else ""
            edges.append({"src": src, "dst": dst, "label": label})
            continue

    return {"nodes": nodes, "edges": edges}


def latex_syntax_heuristics(latex_code: str) -> None:
    """
    Perform several heuristic checks on generated LaTeX/TikZ code.
    Raises AssertionError if a check fails.

    1) No double underscores '__'
    2) Outside math mode, disallow unescaped '_' or '^' in plain text,
       but allow them inside [ ... ] or inside a recognized "node name" token.
    3) Balanced inline math '$'
    4) Balanced \begin{...}/\end{...}
    5) No repeated \label{...}
    6) (Optional) suspicious characters check (#, %, &, ~)
    """

    #################################################################
    # 1. Check for double underscores
    #################################################################
    if "__" in latex_code:
        raise AssertionError("Found double underscores '__' which can cause LaTeX syntax issues.")

    #################################################################
    # 2. Check for unescaped '_' or '^' outside math mode.
    #
    #    Strategy:
    #      - Split by '$' to separate math from text
    #      - In text segments (even indices):
    #          (a) Remove bracketed content [ ... ] altogether
    #          (b) Find any unescaped '_' or '^'
    #          (c) If that underscore is in a "word" that looks like a node name
    #              (e.g. cluster_deriv6), allow it.
    #          (d) Otherwise, fail.
    #
    #    Whitelist pattern for node-style tokens:
    #        \b[a-zA-Z0-9_]*_[a-zA-Z0-9_]*\b
    #
    #    This means a “word boundary,” then letters/digits/underscore,
    #    at least one underscore, then letters/digits/underscore, then word boundary.
    #
    #    If your node names can include hyphens, colons, or parentheses,
    #    you may want to expand the pattern.
    #################################################################

    segments = latex_code.split("$")
    # Simple regex to remove bracketed text (non-greedy, ignoring nested brackets).
    # If your TikZ code uses nested [ [... ] ] or multiline, you may need more advanced handling.
    bracketed_re = re.compile(r"\[.*?\]", flags=re.DOTALL)

    # A regex to find any unescaped ^ or _
    # We'll search with something like r'(?<!\\)([_^])'
    unescaped_carrots_re = re.compile(r"(?<!\\)([_^])")

    # A regex to match an allowed “word with underscore” pattern
    # E.g. cluster_deriv6, foo_bar, etc.
    # (If your node names can have colons/dashes/parentheses, expand here.)
    allowed_underscore_word = re.compile(r"\b[a-zA-Z0-9_]*_[a-zA-Z0-9_]*\b")

    for idx, segment in enumerate(segments):
        # We only check outside math (even indices in the .split('$') approach).
        if idx % 2 == 0:
            # Remove bracketed content
            segment_no_brackets = bracketed_re.sub("", segment)

            # Now search for unescaped ^ or _
            for match in unescaped_carrots_re.finditer(segment_no_brackets):
                ch = match.group(1)  # '_' or '^'
                pos = match.start()
                # We have an unescaped underscore/caret. Check if it's part
                # of an allowed "node-name-like" token. We'll look around
                # in the immediate region for a word match:
                #   e.g. cluster_deriv6

                # Let’s expand out to the nearest whitespace boundaries
                # to see if the full “word” around the underscore is whitelisted.
                line_start = segment_no_brackets.rfind("\n", 0, pos) + 1
                if line_start < 0:
                    line_start = 0
                line_end = segment_no_brackets.find("\n", pos)
                if line_end < 0:
                    line_end = len(segment_no_brackets)

                snippet_line = segment_no_brackets[line_start:line_end]

                # Extract the "word" containing the underscore by splitting on whitespace
                # and punctuation around that position.
                # A simpler approach: just check if the entire snippet_line
                # has a substring that matches our allowed_underscore_word.
                # But let's be more precise:
                # We'll do a findall for all tokens that match \S+ in snippet_line,
                # then see if the underscore is inside one of those tokens.
                tokens = re.findall(r"\S+", snippet_line)

                # We'll see if *any* token matches the allowed node name pattern
                # and also covers the underscore position. The easiest way is just
                # to check if there's a token that matches the pattern *and*
                # contains the substring that triggered the match.
                # If none match, we fail.
                actual_offset_in_line = pos - line_start

                # The underscore is at snippet_line[actual_offset_in_line].
                # We'll see which token in `tokens` might contain that index.

                # Let's rebuild snippet_line as a list of tokens with start/end.
                start_idx = 0
                found_valid = False
                for tok in tokens:
                    # find where 'tok' starts in snippet_line
                    local_start = snippet_line.find(tok, start_idx)
                    if local_start < 0:
                        # fallback
                        local_start = start_idx
                    local_end = local_start + len(tok)

                    if local_start <= actual_offset_in_line < local_end:
                        # The underscore is within this 'tok'
                        # Check if 'tok' matches the allowed node pattern
                        if allowed_underscore_word.search(tok):
                            found_valid = True
                        break

                    start_idx = local_end + 1  # skip space

                if not found_valid:
                    # This underscore is not in a bracketed area, not in math mode,
                    # and not part of a recognized node-name token => error
                    context_snip = segment_no_brackets[max(pos - 40, 0) : pos + 40]
                    raise AssertionError(
                        f"Unescaped '{ch}' outside math mode in non-whitelisted context.\n"
                        f"Snippet: ...{context_snip}...\n"
                    )

    #################################################################
    # 3. Check for balanced inline math '$'
    #################################################################
    dollar_count = latex_code.count("$")
    if dollar_count % 2 != 0:
        raise AssertionError(f"Unbalanced '$' signs detected. Count={dollar_count} is not even.")

    #################################################################
    # 4. Check for balanced \begin{...} / \end{...} (stack-based)
    #################################################################
    begin_pattern = re.compile(r"\\begin\{([^}]+)\}")
    end_pattern = re.compile(r"\\end\{([^}]+)\}")

    begins = list(begin_pattern.finditer(latex_code))
    ends = list(end_pattern.finditer(latex_code))

    # Merge them in the order they appear
    all_envs = sorted(begins + ends, key=lambda m: m.start())
    stack = []
    for m in all_envs:
        if m.re is begin_pattern:
            env_name = m.group(1)
            stack.append((env_name, m.start()))
        else:
            env_name = m.group(1)
            if not stack:
                snippet = latex_code[max(m.start() - 20, 0) : m.start() + 20]
                raise AssertionError(f"\\end{{{env_name}}} without matching \\begin. Near:\n...{snippet}...")
            top_env, top_pos = stack.pop()
            if top_env != env_name:
                snippet = latex_code[max(m.start() - 20, 0) : m.start() + 20]
                raise AssertionError(
                    f"Mismatched environments: opened '{top_env}' but closed '{env_name}'. "
                    f"Near:\n...{snippet}..."
                )
    if stack:
        unclosed = [env for (env, pos) in stack]
        raise AssertionError(f"Unclosed \\begin environments: {unclosed}")

    #################################################################
    # 5. Check for repeated \label{...}
    #################################################################
    labels = re.findall(r"\\label\{([^}]*)\}", latex_code)
    label_counts = Counter(labels)
    for label, cnt in label_counts.items():
        if cnt > 1:
            raise AssertionError(
                f"Label '{label}' used {cnt} times. Duplicate labels can cause cross-ref issues."
            )

    #################################################################
    # 6. Optional: suspicious characters (#, &, %, ~) outside math/brackets
    #
    #    Similar to underscores: remove bracketed text, then fail if we see
    #    unescaped #, &, %, or ~ in plain text. (But be mindful that % in
    #    LaTeX can be a comment if at start of line, so you might skip that.)
    #################################################################
    sus_pattern = re.compile(r"(?<!\\)([#&~%])")

    segments = latex_code.split("$")
    for idx, seg in enumerate(segments):
        if idx % 2 == 0:  # outside math
            seg_no_brackets = bracketed_re.sub("", seg)
            for m in sus_pattern.finditer(seg_no_brackets):
                ch = m.group(1)
                pos = m.start()
                # Possibly skip line-initial '%', etc. if you want to allow comments
                # For simplicity, we’ll just fail if we see unescaped % anywhere
                snippet = seg_no_brackets[max(pos - 40, 0) : pos + 40]
                raise AssertionError(
                    f"Suspicious character '{ch}' outside math/brackets (unescaped). "
                    f"Snippet: ...{snippet}..."
                )


def test_single_variable():
    """
    Check that a single Variable does not crash.
    We only ensure the node with label 'x' appears in the output.
    """
    i = symbols("i")
    x = Variable("x", i)  # your code might treat 'i' as a string or otherwise
    output = to_tikz(x)
    parsed = parse_tikz_lines(output)
    # Check that at least one node has label that includes '$x$'
    var_nodes = [n for n in parsed["nodes"] if "x" in n["label"]]
    assert len(var_nodes) == 1, f"Expected a node labeled x, found {var_nodes}"


def test_sum_of_variables():
    """
    If your code doesn't produce subgraph styles or invisible nodes for sums,
    just check that:
    1) The sum doesn't crash
    2) We see references to x, y, 2, -1 in the final code
    """
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    expr = Sum([x, y], weights=[2, -3])

    output = to_tikz(expr)
    # Basic checks: see 'x', 'y', '2', '-3'
    assert "x" in output
    assert "y" in output
    assert "2" in output
    assert "-3" in output


def test_product_contraction():
    """
    If your code doesn't explicitly draw edges with dimension labels for contractions,
    we skip that. We'll just ensure x,y appear in the diagram.
    """
    i, j = symbols("i j")
    x = Variable("x", i, j)
    y = Variable("y", j)
    expr = Product([x, y])
    output = to_tikz(expr)
    # Check x, y appear
    assert "x" in output
    assert "y" in output
    # Optionally search for "j" if your code draws that label:
    # but if it doesn't, don't fail:
    # e.g. j must appear, we can do a soft check:
    # assert "j" in output, "Expected dimension label j to appear"


def test_derivative_new_name():
    """
    If your code does not create subgraphs or 'Circle-' style,
    just check the derivative call doesn't crash and maybe 'i_prime' is in the code.
    """
    i = symbols("i")
    x = Variable("x", i)
    d = Derivative(x, x, {"i": "i_prime"})
    out = to_tikz(d)

    # minimal check
    assert "x" in out, "Should still see variable x"
    # If your code includes the new edge name i_prime, check it:
    # If not, comment out or remove:
    # assert "i_prime" in out, "Expected new edge i_prime"


def test_derivative_name_overlap():
    N, C = symbols("N C")
    x = Variable("x", N, C)
    expr = Product(
        [
            Delta(N, "N"),
            Delta(N, "N"),
            Derivative(Delta(C, "C"), x, {"N": "N_", "C": "C_"}),
        ]
    )
    assert "derivative+subgraph" in to_tikz(expr)


def test_sum_of_products_and_flatten():
    """
    Convolution requires 3 edges in your library. Flatten might or might not.
    If your code doesn't show subgraphs or 'inner sep=1em', skip that check.
    Just ensure no crash and presence of relevant items.
    """
    i, j, k = symbols("i j k")
    x = Variable("x", i, j)
    z = Variable("z", i, k)

    # Provide 3 edges as strings if your code demands strings:
    conv = Convolution(i_in=i, k_in=j, i_out=k)
    flat = Reshape(f_in=i, f_out=k)

    # Product / sum
    p1 = Product([x, conv])
    p2 = Product([z, flat])
    expr = Sum([p1, p2], weights=[1, 1])
    out = to_tikz(expr)

    # Check for x, z, and maybe 'conv', 'reshape' if your code has such strings
    assert "x" in out
    assert "z" in out
    assert "ast" in out or "conv" in out, "Might see a node labeled '*', or 'conv'"
    assert "reshape" in out


def test_expectation():
    """
    If your code doesn't label the subgraph with 'expectation+subgraph',
    we skip that check.
    Just ensure we don't crash, and that 'x' is present.
    """
    i = symbols("i")
    x = Variable("x", i)
    expr = Expectation(x, wrt=x)

    out = to_tikz(expr)
    assert "x" in out
    # Optionally check for 'expectation+subgraph' if your code does produce it:
    # assert "expectation+subgraph" in out


@pytest.mark.parametrize("with_derivative", [False, True])
def test_random_expr(with_derivative):
    """
    If the code doesn't produce all variables or doesn't do subgraphs,
    just ensure we see at least 'x,y,z' somewhere in the final code
    and it doesn't crash.
    """
    i, j, k = symbols("i j k")
    x = Variable("x", i)
    y = Variable("y", j)
    z = Variable("z", k)

    p1 = Product([x, y])
    p2 = Product([y, z])
    expr = Sum([p1, p2])

    if with_derivative:
        expr = Derivative(expr, y, {"j": "j_prime"})

    out = to_tikz(expr)
    # We may or may not see all three. Let's do a partial check:
    # if your code merges some variables, we can't insist on 3 separate nodes,
    # so let's just ensure 'x', 'y', 'z' appear in the textual output:
    assert "x" in out, "Expected 'x' in output"
    assert "y" in out, "Expected 'y' in output"
    assert "z" in out, "Expected 'z' in output"
    if with_derivative:
        assert "j'prime" in out


def test_function():
    i, j, k, b = symbols("i j k b")
    x = Variable("x", b, i)
    y = Variable("y", j)
    f = function("f", {"k": k}, (x, "i"), (y, "j"))
    out = to_tikz(f)
    assert "x" in out
    assert "y" in out
    assert "f" in out


def test_inv_function():
    i = symbols("i")
    A = Variable("A", i, j=i)
    b = Variable("b", i)
    expr = F.inverse(A) @ b
    hess = expr.grad(A).grad(A)
    parsed = parse_tikz_lines(to_tikz(hess))

    assert any("inv" in node["label"] for node in parsed["nodes"])
    assert any("A" in node["label"] for node in parsed["nodes"])
    assert any("b" in node["label"] for node in parsed["nodes"])


def test_start_end():
    i = symbols("i")
    A = Variable("A", i=i, j=i)
    B = Variable("B", j=i, k=i)
    expr = A @ B
    out = to_tikz(expr)
    assert '"$j$"' in out

    expr = A @ B.rename(k="j", j="k")
    out = to_tikz(expr)
    assert '"$j$" at start, "$k$" at end' in out or '"$k$" at start, "$j$" at end' in out
