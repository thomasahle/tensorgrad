import graphviz
from tensorgrad.tensor import Variable, Product, Function
import tensorgrad.functions as F
from tensorgrad.serializers.to_manim import TensorNetworkAnimation
from tensorgrad.serializers.to_graphviz import to_graphviz
from tensorgrad.serializers.to_tikz import to_tikz
from tensorgrad.serializers.to_d3 import to_d3
from tensorgrad.serializers.to_pytorch import to_pytorch


def main0():
    x = Variable("x", ["i"])
    y = Variable("y", ["i"])
    prod = x @ y

    prod.to_graphviz(g := graphviz.Graph())
    g.view()


def main1():
    x = Variable("x", ["i"])
    A = Variable("A", ["i", "o"])
    prod = A @ x

    prod.to_graphviz(g := graphviz.Graph())
    g.view()


def main(mode):
    # ||Ax - y||_2^2
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    A = Variable("A", ["x", "y"])
    Axmy = A @ x - y
    frob = F.frobenius2(Axmy)
    grad = frob.grad(x)
    assert grad.edges == ["x'"]

    out = grad.simplify()
    print(f"{out=}")

    if mode == "graphviz":
        to_graphviz(out).view()
        print(to_graphviz(out).source)
    if mode == "tikz":
        latex_code = to_tikz(out)
        print(latex_code)
        compile_latex(latex_code)
    if mode == "d3":
        html_code = to_d3(out)
        with open("output.html", "w") as file:
            file.write(html_code)


def compile_latex(latex_code):
    import os, subprocess

    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    # Save the LaTeX code to a file in the output directory
    with open(os.path.join(output_dir, "output.tex"), "w") as file:
        file.write(latex_code)

    # Compile the LaTeX file using pdflatex with the -output-directory option
    subprocess.run(
        [
            "lualatex",
            "-output-directory",
            output_dir,
            os.path.join(output_dir, "output.tex"),
        ],
        check=True,
    )


def main2(mode):
    # ||Ax - y||_2^2
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    A = Variable("A", ["x", "y"])
    frob = F.frobenius2(A @ x - y)
    grad = frob.grad(x).grad(x).simplify()
    assert grad.edges == ["x'", "x''"]

    if mode == "tikz":
        latex_code = to_tikz(grad)
        compile_latex(latex_code)


def main3(mode):
    # ||Ax - y||_2^2
    X = Variable("X", ["b", "x"])
    Y = Variable("Y", ["b", "y"])
    W = Variable("W", ["x", "y"])
    frob = F.frobenius2(W @ X - Y)
    grad = frob.grad(W).simplify()
    assert set(grad.edges) == {"x'", "y'"}

    if mode == "tikz":
        latex_code = to_tikz(grad)
        compile_latex(latex_code)

    print(to_pytorch(grad))


def main4(mode):
    # ||Ax - y||_2^2
    X = Variable("X", ["b", "x"])
    Y = Variable("Y", ["b", "y"])
    W = Variable("W", ["x", "y"])
    b = Variable("b", ["y"])
    frob = F.frobenius2(W @ X + b - Y)
    grad = frob.grad(b).simplify()

    if mode == "tikz":
        latex_code = to_tikz(grad)
        compile_latex(latex_code)

    print(to_pytorch(grad))


def main5(mode):
    # f(v(x))
    x = Variable("x", ["x"])
    v = Function("v", [x], ["x"], ["y"])
    # g = Function("g", [v], ["y"], ["z"])
    # f = Function("f", [g], ["z"], [])
    f = Function("f", [v], ["y"], [])

    # grad = f.grad(x).simplify()
    grad = f.grad(x).grad(x).simplify()

    if mode == "tikz":
        print(grad)
        latex_code = to_tikz(grad)
        print(latex_code)
        for i, line in enumerate(latex_code.split("\n")):
            print(f"{i+1:2d} {line}")
        compile_latex(latex_code)

    print(to_pytorch(grad))


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
    out = hvp.simplify()
    # out = H.simplify()
    # out = out @ v

    if mode == "tikz":
        print(out)
        latex_code = to_tikz(out)
        for i, line in enumerate(latex_code.split("\n")):
            print(f"{i+1:2d} {line}")
        compile_latex(latex_code)

    print(to_pytorch(out))



def manim():
    tensor = create_tensor_network()
    scene = TensorNetworkAnimation(tensor)
    scene.render()


def milanfar():
    # Derivation of Peyman Milanfar’s gradient, d[A(x)x]
    x = Variable("x", ["i"])
    A = Function("A", ["i", "j"], (x, "i"))
    # expr = (A @ x).grad(x).simplify({"grad_steps": 0})
    expr = (A @ x).grad(x).simplify()

    compile_latex(to_tikz(expr))
    print(to_tikz(expr))


def simple():
    x = Variable("x", ["i"])
    y = Variable("y", ["i"])
    expr = (x * y)
    compile_latex(to_tikz(expr))
    print(expr)
    to_tikz(expr)



def softmax():
    x = Variable("x", ["i"])
    y = F.softmax(x, ["i"])
    expr = y.simplify()
    compile_latex(to_tikz(expr))


def softmax_jac():
    x = Variable("x", ["i"])
    y = F.softmax(x, ["i"])
    expr = y.grad(x).simplify()
    compile_latex(to_tikz(expr))

if __name__ == "__main__":
    import sys

    # mode = sys.argv[1]
    # Hvp(mode)
    #milanfar()
    softmax_jac()
    #simple()
