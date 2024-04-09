import graphviz
from pdf2image import convert_from_path
import os, subprocess
from PIL import Image

from tensorgrad.tensor import Variable, Product, Function, Derivative, Sum, Copy
import tensorgrad.functions as F
from tensorgrad.serializers.to_manim import TensorNetworkAnimation
from tensorgrad.serializers.to_graphviz import to_graphviz
from tensorgrad.serializers.to_tikz import to_tikz
from tensorgrad.serializers.to_d3 import to_d3
from tensorgrad.serializers.to_pytorch import to_pytorch


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
    images[0].save(image_path, 'PNG')

    return image_path

def combine_images_vertically(image_paths, padding=10, background_color='white'):
    images = [Image.open(x) for x in image_paths]

    # Calculate total height considering padding and maximum width
    total_height = sum(image.height for image in images) + padding * (len(images) + 1)
    max_width = max(image.width for image in images) + padding * 2

    # Create a new image with a white background
    combined_image = Image.new('RGB', (max_width, total_height), color=background_color)

    # Paste images into the new image with padding
    y_offset = padding
    for image in images:
        # Calculate horizontal padding to center images
        x_padding = (max_width - image.width) // 2
        combined_image.paste(image, (x_padding, y_offset))
        y_offset += image.height + padding

    # Save the combined image
    combined_image_path = "combined_image.png"
    combined_image.save(combined_image_path)
    return combined_image_path


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


def chain_rule_hess():
    # f(v(x))
    x = Variable("x", ["x"])
    v = Function("v", ["y"], (x, "x"))
    f = Function("f", [], (v, "y"))

    hess = Derivative(Derivative(f, x), x)
    print(hess.edges)
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



def manim():
    tensor = create_tensor_network()
    scene = TensorNetworkAnimation(tensor)
    scene.render()



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

def softmax_mat():
    A = Variable("A", ["i", "j"])
    expr = F.softmax(A, ["i"]).grad(A).simplify()
    compile_latex(to_tikz(expr))


def softmax_grad():
    x = Variable("x", ["i"])
    y = F.softmax(x, ["i"])
    return Derivative(y, x).simplify()


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

def save_steps(expr, min_steps=None):
    images = []
    images.append(compile_latex(expr, suffix=f"0"))
    old = expr
    while True:
        new = expr.simplify({"grad_steps": len(images)})
        if new == old and (min_steps is None or len(images) >= min_steps):
            break
        old = new
        images.append(compile_latex(new, suffix=f"{len(images)}"))

    output_path = combine_images_vertically(images)
    print(f"Combined image saved to {output_path}")


if __name__ == "__main__":
    import sys

    # mode = sys.argv[1]
    # Hvp(mode)
    #milanfar()
    #ce()
    #simple()
    #softmax_jac()
    #softmax()
    #main3()
    #main5()
    save_steps(chain_rule_hess(), min_steps=3)
    #save_steps(Hvp().simplify())
    #save_steps(rand0())
    #save_steps(func_max())
    #save_steps(milanfar())
    #save_steps(ce())
    #save_steps(softmax_grad())
    #save_steps(ce_hess().simplify())
    #save_steps(ce_grad().simplify())
