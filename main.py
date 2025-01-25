from tensorgrad import Variable, Product, Function, Derivative, Sum, Delta, Zero, Ones, function
from collections import defaultdict
import tensorgrad.functions as F
from tensorgrad.extras import Expectation
from tensorgrad.extras.to_tikz import to_tikz
from tensorgrad.testutils import generate_random_tensor_expression, make_random_tree
from tensorgrad.imgtools import save_steps, save_as_image
from tensorgrad.extras.to_pytorch import compile_to_callable
from tensorgrad.testutils import assert_close, rand_values
from sympy import symbols
import torch
import tqdm

# Examples from the notebook

def notebook0():
    b, x, y = symbols("b x y")
    X = Variable("X", b, x)
    Y = Variable("Y", b, y)
    W = Variable("W", x, y)
    frob = F.frobenius2(W @ X - Y)
    grad = frob.grad(W).simplify()
    save_steps(grad)
    print(to_tikz(grad))

def notebook1():
    i, j = symbols("i j")
    x = Variable("x", x=i)
    v = function("v", {"y": j}, (x, "x"))
    f = function("f", {}, (v, "y"))
    grad = f.grad(x).simplify()
    save_steps(grad)

def notebook2():
    b, i = symbols("b i")
    x = Variable("x", b, i)
    f = function("max", {}, (x, "i"))
    expr = f.grad(x).simplify()
    print(to_tikz(expr))
    save_steps(expr)

def notebook3():
    C = symbols("C")
    logits = Variable("logits", C)
    target = Variable("target", C)
    ce = F.cross_entropy(logits, target, ["C"])
    expr = ce.grad(logits)
    expr = expr.grad(logits)
    expr = expr.simplify()
    save_steps(expr)


def main():
    i = symbols("i")
    a = Variable("a", i)
    b = Variable("b", i)
    X = Variable("X", i, j=i)
    expr = F.graph("a -i- X1 -j-i- X2 -j-i- b", a=a, X1=X, X2=X, b=b)

    # cross entropy softmax hessian
    C = symbols("C")
    logits = Variable("logits", C)
    target = Variable("target", C)
    e = F.exp(logits)
    softmax = e / (1 + F.sum(e))  # Altnernative softmax
    ce = -F.sum(target * F.log(softmax))
    expr = ce.grad(logits).grad(logits)
    print(expr)

    # Just softmax grad
    C = symbols("C")
    logits = Variable("logits", C)
    target = Variable("target", C)
    e = F.exp(logits)
    softmax = e / F.sum(e)
    expr = Derivative(softmax @ target, logits)
    print(expr)

    save_steps(expr)


def main2():
    i, j = symbols("i j")
    x = Variable("x", i)
    A = Variable("A", i, j=i)
    xAx = x.rename(i="j") @ A @ x
    xAxxAx = xAx @ xAx
    expr = Derivative(Derivative(xAxxAx, x), x)
    expr = expr.simplify()
    save_steps(expr)


def main3():
    i, j = symbols("i j")
    R = Variable("R", i, j)
    p = R / F.sum(R, edges=["i"])
    U = Variable("U", i, j)
    z = U @ p
    expr = Derivative(z, R)
    save_steps(expr)


def main4():
    # Showcase Isserlis theorem
    K = 4
    i = symbols("i")
    eps = symbols("e")
    u = Variable(f"u", i)
    C = Variable(f"C", i, j=i).with_symmetries("i j")
    prod = Product([u.rename(i=f"i{k}") for k in range(K)])
    expr = Expectation(prod, u, mu=Zero(i), covar=C, covar_names={"i": "j"})
    expr = expr.full_simplify()
    save_steps(expr)


def main5():
    K = 3
    i = symbols("i")
    eps = symbols("e")
    u = Variable(f"u", i)
    X = Delta(i, "i", "j") + u @ u.rename(i="j")  # * Copy(eps)
    # X = u @ u.rename(i="j") @ Copy(eps)
    M = Variable("M", i, j=i).with_symmetries("i j")
    if K == 3:
        prod = F.graph(
            """*1 -i- X1 -j- M1
                       M1 -i- X2 -j- M2
                       M2 -i- X3 -j- M3
                       M3 -i- *1""",
            X1=X,
            M1=M,
            X2=X,
            M2=M,
            X3=X,
            M3=M,
        )
    if K == 2:
        prod = F.graph(
            """*1 -i- X1 -j- M1
                       M1 -i- X2 -j- M2
                       M2 -i- *1""",
            X1=X,
            M1=M,
            X2=X,
            M2=M,
        )
    expr = Expectation(prod, u)
    expr = expr.full_simplify()
    print(expr)
    print(len(expr.tensors))
    print("expanding")
    expr = expr.simplify({"expand": True})
    print(expr)
    print(len(expr.tensors))
    save_as_image(expr, "m3.png")


def main6():
    i, j, k = symbols("i j k")
    Y = Variable("Y", i, k)
    A = Variable("A", i, j)
    X = Variable("X", j, k)
    expr = F.frobenius2(Y - A @ X) / 2
    hess = Derivative(Derivative(expr, X), X)
    hess = hess.full_simplify()
    print(hess)
    save_steps(hess)


def main7():
    i = symbols("i")
    A = Variable("A", i, j=i)
    b = Variable("b", i)
    expr = F.inverse(A) @ b
    frob = F.frobenius2(expr.grad(A).grad(A))
    frob += 2 * F.frobenius2(expr.grad(A).grad(b))
    frob += F.frobenius2(expr.grad(b).grad(b))
    frob = frob#.full_simplify()
    #print(to_latex_indexed(frob))
    save_steps(frob)


def main9():
    i = symbols("i")
    A = Variable("A", i=i, j=i)
    B = Variable("B", j=i, k=i)
    expr = A @ B.rename(k="j", j="k")
    save_steps(expr)
    print(to_tikz(expr))


def main10():
    b, x, y = symbols("b x y")
    X = Variable("X", b, x)
    Y = Variable("Y", b, y)
    W = Variable("W", x, y)
    frob = F.frobenius2(W @ X - Y)
    grad = Derivative(frob, W)
    save_steps(grad)
    print(to_tikz(grad))

def main11():
    i, j, k = symbols("i j k")
    x = Variable("x", i, j)
    w = Variable("w", j, k)
    b = Variable("b", i, k)

    expr = F.relu(x @ w + b)  # shape (i, k)

    compiled_fn = compile_to_callable(expr, verbose=True, torch_compile=False)

    dims = {i: 2, j: 3, k: 4}
    vals = rand_values([x, w, b], dims)
    ref = expr.evaluate(vals, dims)
    out = compiled_fn(vals, dims)[expr]
    assert_close(out, ref)

def main12():
    n, i = symbols("n i")
    X = Variable("X", i, n1=n)
    Y = Variable("Y", i, n2=n)
    ips = X @ Y  # shape (n1, n2)
    norm2s = F.dot(X, X, dim='i') @ F.dot(Y, Y, dim='i')
    cosine = ips / norm2s
    cosine = ips / F.sqrt(norm2s)
    grad = Derivative(cosine, X)
    save_steps(grad)
    #print(to_tikz(grad))


def main13():
    n, i = symbols("n i")
    X = Variable("X", i, n)
    ips = F.dot(X, X, dim='i')
    save_steps(ips)
    print(to_tikz(ips.full_simplify()))

def main14():
    i = symbols("i")
    X = Variable("X", i)
    expr = X @ Delta(i, "i", "j", "k", "l")
    save_steps(expr)
    print(to_tikz(expr.full_simplify()))

def main15():
    d = symbols("d")
    gs = [Variable(f"g{k}", i=d) for k in range(1)]
    Ms = [Delta(d, "i", "j") - g @ g.rename(i='j') / Delta(d) for g in gs]
    A = F.multi_dot(Ms * 2, dims=("i", "j"))
    assert A.edges == {"i", "j"}, A.edges
    B = A @ A
    for g in gs:
        B = Expectation(B, g).full_simplify()
    save_steps(B)

def main16():
    d = symbols("d")
    g = Variable(f"g", i=d)
    A = Delta(d, "i", "j") - g @ g.rename(i='j') / Delta(d)
    M = Variable("M", i=d, j=d).with_symmetries("i j")

    assert M == M.rename(i="j", j="i")

    C = F.multi_dot([A,M] * 2, dims=("i", "j"))
    assert C.edges == {"i", "j"}, A.edges
    frob = C @ C
    B = Expectation(frob, g).full_simplify()
    save_steps(B)

if __name__ == "__main__":
    main16()
