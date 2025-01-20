from tensorgrad import Variable, Product, Function, Derivative, Sum, Copy, Zero, Ones, function
from collections import defaultdict
import tensorgrad.functions as F
from tensorgrad.extras import Expectation
from tensorgrad.serializers.to_tikz import to_tikz
# from tensorgrad.serializers.to_pytorch import to_pytorch
from tensorgrad.testutils import generate_random_tensor_expression, make_random_tree
from tensorgrad.imgtools import save_steps, save_as_image
from sympy import symbols
import torch
import tqdm


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
    X = Copy(i, "i", "j") + u @ u.rename(i="j")  # * Copy(eps)
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
    save_as_image(expr, 'm3.png')

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
    hess = expr.grad(A).grad(A)
    print(hess)
    print(to_tikz(hess))
    save_steps(hess)

def main9():
    i = symbols("i")
    A = Variable("A", i=i, j=i)
    B = Variable("B", j=i, k=i)
    expr = A @ B.rename(k="j", j="k")
    save_steps(expr)
    print(to_tikz(expr))

if __name__ == "__main__":
    main9()
