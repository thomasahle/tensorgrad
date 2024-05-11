import graphviz
from pdf2image import convert_from_path

import os, subprocess
from PIL import Image, ImageDraw

from tensorgrad import Variable, Product, Function, Derivative, Sum, Copy, Zero, Ones, Unfold
from tensorgrad.tensor import TensorDict
from collections import defaultdict
import tensorgrad.functions as F
from tensorgrad.serializers.to_manim import TensorNetworkAnimation
from tensorgrad.serializers.to_graphviz import to_graphviz
from tensorgrad.serializers.to_tikz import to_tikz
from tensorgrad.serializers.to_d3 import to_d3
from tensorgrad.serializers.to_pytorch import to_pytorch

from tensorgrad.utils import generate_random_tensor_expression
from tensorgrad.extras.expectation import Expectation


def compile_latex(expr, suffix=""):
    latex_code = to_tikz(expr)
    print(latex_code)

    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    # Save the LaTeX code to a file
    tex_file_path = os.path.join(output_dir, f"output_{suffix}.tex")
    with open(tex_file_path, "w") as file:
        file.write(latex_code)

    # Compile the LaTeX file to PDF
    subprocess.run(
            ["lualatex", "-output-directory", output_dir, tex_file_path],
            check=True,
            )

    # Convert the PDF to an image
    pdf_path = tex_file_path.replace(".tex", ".pdf")
    images = convert_from_path(pdf_path, dpi=300)

    # Save the first page as an image
    image_path = pdf_path.replace(".pdf", ".png")
    images[0].save(image_path, "PNG")

    return image_path


def combine_images_vertically(image_paths, padding=10, line_padding=5, background_color="white", line_color="black", line_width=2):
    images = [Image.open(x) for x in image_paths]

    # Calculate total height considering padding, maximum width, separating lines, and line padding
    total_height = sum(image.height for image in images) + padding * (len(images) + 1) + line_width * (len(images) - 1) + line_padding * 2 * (len(images) - 1)
    max_width = max(image.width for image in images) + padding * 2

    # Create a new image with a white background
    combined_image = Image.new("RGB", (max_width, total_height), color=background_color)

    # Create a drawing context
    draw = ImageDraw.Draw(combined_image)

    # Paste images into the new image with padding, separating lines, and line padding
    y_offset = padding
    for i, image in enumerate(images):
        # Calculate horizontal padding to center images
        x_padding = (max_width - image.width) // 2

        combined_image.paste(image, (x_padding, y_offset))
        y_offset += image.height + padding

        # Add line padding below the current image
        y_offset += line_padding

        # Draw a black separating line below the current image (except for the last image)
        if i < len(images) - 1:
            line_y = y_offset + line_width // 2
            draw.line([(0, line_y), (max_width, line_y)], fill=line_color, width=line_width)
            y_offset += line_width + line_padding

    # Save the combined image
    combined_image_path = "combined_image.png"
    combined_image.save(combined_image_path)

    return combined_image_path


def l2_grad_x():
    # ||Ax - y||_2^2
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    A = Variable("A", ["x", "y"])
    Axmy = A @ x - y
    frob = F.frobenius2(Axmy)
    grad = Derivative(frob, x)
    assert grad.edges == ["x'"]
    return grad


def l2_hess_x():
    # ||Ax - y||_2^2
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    A = Variable("A", ["x", "y"])
    Axmy = A @ x - y
    frob = F.frobenius2(Axmy)
    grad = Derivative(Derivative(frob, x), x)
    assert grad.edges == ["x'"]
    return grad


def l2_grad_W():
    # ||Ax - y||_2^2
    X = Variable("X", ["b", "x"])
    Y = Variable("Y", ["b", "y"])
    W = Variable("W", ["x", "y"])
    frob = F.frobenius2(W @ X - Y)
    grad = Derivative(frob, W)
    assert set(grad.edges) == {"x_", "y_"}
    return grad


def l2_grad_b(mode):
    # ||Ax - y||_2^2
    X = Variable("X", ["b", "x"])
    Y = Variable("Y", ["b", "y"])
    W = Variable("W", ["x", "y"])
    b = Variable("b", ["y"])
    frob = F.frobenius2(W @ X + b - Y)
    return Derivative(frob, b)

def trace_grad():
    x = Variable("X", ["i", "j"])
    y = F.trace(x)
    return Derivative(y, x)

def trace_function_grad():
    x = Variable("X", ["i"])
    y = Function("f", ["j", "k"], (x, "i"))
    z = F.trace(y)
    return Derivative(z, x)

def chain_rule_hess():
    # f(v(x))
    x = Variable("x", ["x"])
    v = Function("v", ["y"], (x, "x"))
    f = Function("f", [], (v, "y"))

    hess = Derivative(Derivative(f, x), x)
    assert hess.edges == ["x_", "x__"]
    return hess


def Hvp(mode, depth=2):
    # f(v(x))
    x = Variable("x", ["y0"])
    fs = [x]
    for i in range(depth - 1):
        fs.append(Function(f"f{i}", [fs[-1]], [f"y{i}"], [f"y{i+1}"]))
    fs.append(Function(f"f{depth-1}", [fs[-1]], [f"y{depth-1}"], []))

    v = Variable("v", ["b", "y0'"])

    H = fs[-1].grad(x).grad(x)

    hvp = H @ v
    return hvp


def softmax_grad():
    x = Variable("x", ["i"])
    y = F.softmax(x, ["i"])
    return Derivative(y, x)

def softmax_func_grad():
    x = Variable("x", ["i"])
    y = Function("f", ["i"], (x, "i"))
    z = F.softmax(y, ["i"])
    return Derivative(z, x)


def ce():
    logits = Variable("logits", ["C"])
    target = Variable("target", ["C"])
    expr = F.cross_entropy(logits, target, ["C"]).simplify()
    return expr


def ce_grad():
    logits = Variable("logits", ["C"])
    target = Variable("target", ["C"])
    expr = F.cross_entropy(logits, target, ["C"]).simplify()
    return expr.grad(logits)


def ce_hess():
    logits = Variable("logits", ["C"])
    target = Variable("target", ["C"])
    expr = F.cross_entropy(logits, target, ["C"]).simplify()
    return expr.grad(logits).grad(logits)


def ce():
    logits = Variable("logits", ["C"])
    target = Variable("target", ["C"])
    expr = F.cross_entropy(logits, target, ["C"]).simplify()
    return expr


def func_max():
    x = Variable("x", ["b", "i"])
    f = Function("max", [], (x, "i"))
    expr = Derivative(f, x)
    return expr


def milanfar():
    # Derivation of Peyman Milanfar’s gradient, d[A(x)x]
    x = Variable("x", ["i"])
    A = Function("A", ["i", "j"], (x, "i"))
    return Derivative(A @ x, x)

def taylor(k):
    # Derivation of Peyman Milanfar’s gradient, d[A(x)x]
    x = Variable("x", ["i"])
    # f = Function("f", ["o"], (x, "i"))
    z = Zero(["i"])
    f = Function("f", ["o"], (z, "i"))
    res = f
    for i in range(1, k):
        f = Derivative(f, x, [f"i_{i}"]) @ x.rename({"i": f"i_{i}"})
        res += f
    return res

def division():
    a = Variable("a", [])
    x = Variable("x", ["i"])
    return (a / x).grad(x).simplify()


def save_steps_old(expr, min_steps=None):
    images = []
    images.append(compile_latex(expr, suffix=f"0"))
    old = expr
    while True:
        new = expr.simplify({"grad_steps": len(images)})
        if new == old and (min_steps is None or len(images) >= min_steps):
            print(f"{new=} = {old=}")
            break
        old = new
        images.append(compile_latex(new, suffix=f"{len(images)}"))

    output_path = combine_images_vertically(images)
    print(f"Combined image saved to {output_path}")


def save_steps(expr, slow_grad=False):
    images = []
    images.append(compile_latex(expr, suffix=f"0"))

    # TODO: This is still not good enough, since there may be a double derivative somewhere inside the tensor.
    cnt_derivatives = 0
    d = expr
    while isinstance(d, Derivative):
        cnt_derivatives += 1
        d = d.tensor
    if cnt_derivatives > 1:
        expr = expr.simplify({"grad_steps": cnt_derivatives})
        images.append(compile_latex(expr, suffix=f"{len(images)}"))

    expand = False
    while True:
        try:
            args = {"grad_steps": 1} if slow_grad else {}
            if expand:
                args["expand"] = True
            new = expr.simplify(args).simplify()
        except Exception as e:
            print(e)
            break
        if new == expr:
            if not expand:
                expand = True
                continue
            break
        print(new)
        images.append(compile_latex(new, suffix=f"{len(images)}"))
        expr = new

    output_path = combine_images_vertically(images)
    print(f"Combined image saved to {output_path}")

def main():

    # A = Variable("A", ["i", "j"])
    # x = F.softmax(A, ["i"]).grad(A).simplify()

    # X = Variable("X", ["i"])
    # S = F.softmax(X, ["i"])
    # x = Derivative(S, X).simplify()

    logits = Variable("logits", ["C"])
    target = Variable("target", ["C"])
    ce = F.cross_entropy(logits, target, ["C"])
    expr = ce.grad(logits)
    expr = expr.grad(logits)

    # for _ in range(4):
    #     expr = expr.simplify()
    # expr = expr.simplify({"expand": True})
    # for _ in range(4):
    #     expr = expr.simplify()

    expr = expr.full_simplify()


    #a = Variable("a", [])
    #x = Variable("x", ["i"])
    #expr = Derivative(a / x, x)

    #X = Variable("X", "i, j")
    #expr = Derivative(F.sum(X)**2, X)

    #b = Variable("b", ["i"])
    #c = Variable("c", ["i"])
    #X = Variable("X", "j, i")
    #expr = Derivative(b @ X @ X @ c, X)

    # x = Variable("x", ["i"])
    # eps = Variable("eps", ["j"])
    # f = F.softmax(x, dims=["i"])
    # expr = F.taylor(f, x, eps, 2).simplify()

    #expr, _, _ = generate_random_tensor_expression(20)
    #expr = expr.simplify()

    #x = Variable("x", ["i"])
    #expr = Derivative(F.softmax(x, ["i"]), x)
    #expr = Derivative(expr, x)

    # x = Variable("x", ["i"])
    # y = Variable("y", ["i"])
    # expr = F.sum(F.pow(x - y, 3))
    # expr = expr.grad(x).grad(y).simplify()

    #expr = taylor(2)

    #A = Variable("A", ["i", "j"])
    #B = Variable("B", ["j", "k"])
    #C = Variable("C", ["k", "i"])
    #x = (A @ B @ C) @ (A @ B @ C)
    #y = A @ B @ C.rename({"i": "i'"}) @ A.rename({"i": "i'"}) @ B @ C
    #expr = x - y

    #  B B      B     B
    # A D A    A \   / A
    # A D A vs A  D-D  A
    #  B B      B/   \B

    #D = Variable("D", ["k", "k'", "l"])
    #A2 = A.rename({"i": "i'"})
    #B2 = B.rename({"k": "k'"})
    #half1 = A @ B @ D @ B2 @ A2
    #expr1 = half1 @ half1
    #half2 = A @ B @ D @ B2 @ A
    #expr2 = half2 @ half2
    #expr  = expr1 - expr2


    # data = Variable("data", ["b", "cin", "win", "hin"])
    # unfold = F.Convolution("win", "kw", "wout") @ F.Convolution("hin", "kh", "hout")
    # kernel = Variable("kernel", ["cin", "kw", "kh", "cout"])
    # expr = data @ unfold @ kernel
    #expr = Derivative(expr, kernel)

    # data = Variable("data", ["b", "c1", "w1"])
    # kernel = Variable("kernel", ["c1", "kw", "c2"])
    # expr = data @ Unfold(["w1"], ["kw"], ["w2"]) @ kernel
    # expr = F.relu(expr)
    # kernel2 = Variable("kernel2", ["c2", "kw", "c3"])
    # expr = expr @ Unfold(["w2"], ["kw"], ["w3"]) @ kernel2
    # expr = expr @ F.Flatten(["c3", "w3"], "out")

    # data = Variable("X", ["b", "c", "w", "h"])
    # unfold = F.Convolution("w", "j", "w2") @ F.Convolution("h", "i", "h2")
    # kernel = Variable("kernel", ["c", "i", "j", "c2"])
    # expr = data @ unfold @ kernel
    # expr = Derivative(expr, kernel).simplify()

    #x = Variable("x", "i, j")
    #expr = x + x.rename({"i": "j", "j": "i"})
    #print(expr)
    #return

    # expr = Sum([Variable('y', ['a', 'b'], ['a', 'b']), Sum([Sum([Variable('y', ['a', 'b'], ['a', 'b']), Product([Product([Variable('y', ['a', 'b'], ['a', 'b']), Variable('z', ['a'], ['a'])]), Product([Copy(['a'])])])], [1, 1]), Product([Variable('z', ['a'], ['a']), Product([Copy(['b'])])])], [1, 1])], [1, 1])
    # expr = Sum([Variable('y', ['a'], ['a']), Product([Product([Variable('z', ['a'], ['a']), Sum([Product([Product([Variable('x', ['a'], ['a']), Variable('z', ['a'], ['a'])]), Product([Copy(['a'])])]), Variable('x', ['a'], ['a'])], [1, 1])]), Product([Copy(['a'])])])], [1, 1])
    # expr = Variable("x", ["a"]) + Variable("y", ["a", "b", "c"])

    # expr = Ones(["a", "b", "c"]) + Ones(["a", "b", "c"])

    if False:
        X = Variable("X", "i, j")
        A = Variable("A", "j, j1")
        B = Variable("B", "i, i1")
        C = Variable("C", "j, j1")
        expr = (
                X.rename({"i":"i0"})
                @ A
                @ X.rename({"j":"j1"})
                @ B
                @ X.rename({"i":"i1"})
                @ C
                @ X.rename({"j":"j1"})
            )
        mu = Copy(["i"]) @ Variable("m", "j")
        covar = Copy("i, k") @ Variable("S", "j, l")
        #mu = Zero(["i", "j"])
        #covar = Copy(["i", "k"]) @ Copy(["j", "l"])
        assert covar.edges == ["i", "k", "j", "l"]
        expr = Expectation(expr, X, mu, covar).full_simplify()


    if False:
        X = Variable("X", "i, j")
        A = Variable("A", "j, j1")
        expr = (
                X.rename({"j":"j0"}) # (i, j0)
                @ X # (j0, j)
                @ A # (j0, j1)
                @ X.rename({"j":"j1"}) # (j0, i)
                @ X # (j0, j)
            )
        mu = Copy(["i"]) @ Variable("m", ["j"])
        covar = Copy("i, k") @ Variable("S", "j, l")
        #mu = Zero(["i", "j"])
        #covar = Copy(["i", "k"]) @ Copy(["j", "l"])
        expr = Expectation(expr, X, mu, covar).full_simplify()

    #mu = Variable("m", ["i", "j"])
    ##covar = Variable("M", "i, j, k, l")
    #expr = (X.rename({"i":"i0"}) # (i0, j)
    #        @ A # (j, j1)
    #        @ X.rename({"j":"j1"})) # (j1, i)
    #expr = Expectation(expr, X, mu, covar)

    # expr = (Xt.rename({"i":"i0"})
    #         @ X.rename({"i":"i1"}))
    # expr = Expectation(expr, X, mu, covar)


    # mu = Zero(["i", "j"])
    # covar = Copy(["i", "k"]) @ Copy(["j", "l"])
    # assert covar.edges == ["i", "k", "j", "l"]
    # expr = Expectation(expr, X, mu, covar)

    #A = Variable("A", ["i", "j"])
    #x = Variable("x", ["i"])
    #expr = x @ A @ (A @ x)
    #mu = Variable("m", ["i"])
    #covar = Variable("M", ["i", "j"])
    #expr = Expectation(expr, x, mu, covar)
    #expr = expr.simplify()

    #expr = chain_rule_hess()
    #expr = l2_grad_W().simplify()
    #expr = l2_grad_W()
    #expr = trace_grad()
    #expr = trace_function_grad()
    #expr = ce()
    #expr = ce_grad().simplify()
    #expr = ce_hess().simplify()
    #expr = milanfar()
    #expr = division()
    #expr = func_max()
    #expr = softmax_func_grad()
    save_steps(expr)
    #save_steps_old(expr, min_steps=7)
    #print(to_pytorch(expr))

    #print(to_simple_matrix_formula(expr))

    # save_steps(Hvp().simplify())
    # save_steps(rand0())
    # save_steps(func_max())
    # save_steps(ce_hess().simplify())

def to_simple_matrix_formula(expr):
    assert isinstance(expr, Sum)
    res = []
    e0 = expr.edges[0]
    for prod in expr.tensors:
        assert isinstance(prod, Product)
        mat = []
        for comp in prod.components():
            print("Comp:", comp)
            #if len(comp.edges) == 0:
            #    t = comp.tensors[0]
            #    in_edge = "" # FIXME: This prevents us from getting the right transpose on the first matrix in the trace
            #else:
            #    t = next(t for t in comp.tensors if e0 in t.edges)
            #    in_edge = e0
            #s = " ".join(dfs(t, adj, visited, in_edge))
            s = component_to_matrix(comp, e0)
            #if comp.rank == 0:
            if len(comp.edges) == 0:
                s = s.replace("1", "")
                s = f"\\mathrm{{tr}}({s})"
            else:
                s = s.replace("1", "\\mathbf{1}")
            mat.append(s)
        res.append(" \\cdot ".join(mat))
    return "\n\\\\&+ ".join(res)

def component_to_matrix(comp, left_edge):
    adj = defaultdict(list)
    for t in comp.tensors:
        for e in t.edges:
            adj[e].append(t)

    # We can handle 4 cases:
    # - A matrix
    # - A trace
    # - A matrix-vector mult
    # - A vector-matrix*-vector mult

    # If there's a vector, we start from that
    # vec = next((t for t in comp.tensors if len(t.edges) == 1), None)
    # if vec is not None:
    #     # This is a vector
    #     pass

    # If there's supposed to be a free edge on the left, we start from that
    start = next((t for t in comp.tensors if left_edge in t.edges), None)
    if start is not None:
        in_edge = left_edge
    else:
        start = comp.tensors[0]
        in_edge = ""
    cur = start

    visited = TensorDict(key_fn=id)
    res = []
    while cur not in visited:
        visited[cur] = True
        e0, e1 = cur.edges
        res.append(cur.name)
        if e0 != in_edge:
            e0, e1 = e1, e0
            res.append("^T")
        t0, t1 = adj[e1]
        in_edge = e1
        cur = t1 if t0 is cur else t0

    return " ".join(res)

def dfs(t, adj, visited, in_edge):
    visited[t] = True
    if isinstance(t, Variable):
        yield t.name + ("^T" if in_edge == t.edges[1] else "")
    elif isinstance(t, Copy):
        yield "1"
    else:
        assert False
    for e in t.edges:
        for u in adj[e]:
            if u not in visited:
                yield from dfs(u, adj, visited, e)





if __name__ == "__main__":
    main()
