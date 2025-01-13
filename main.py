from tensorgrad import Variable, Product, Function, Derivative, Sum, Copy, Zero, Ones, simple_function
from collections import defaultdict
import tensorgrad.functions as F
from tensorgrad.extras import Expectation
from tensorgrad.serializers.to_tikz import to_tikz
from tensorgrad.serializers.to_pytorch import to_pytorch
from tensorgrad.testutils import generate_random_tensor_expression, make_random_tree
from tensorgrad.imgtools import save_steps, save_as_image
from sympy import symbols




def main():
    i = symbols("i")
    a = Variable("a", i)
    b = Variable("b", i)
    X = Variable("X", i, j=i)
    expr = F.graph('a -i- X1 -j-i- X2 -j-i- b', a=a, X1=X, X2=X, b=b)
    # expr = Derivative(Derivative(graph, X), X)
    print(to_pytorch(expr))

    # cross entropy softmax hessian
    C = symbols("C")
    logits = Variable("logits", C)
    target = Variable("target", C)
    e = F.exp(logits)
    softmax = e / (1 + F.sum(e)) # Altnernative softmax
    ce = -F.sum(target * F.log(softmax))
    expr = ce.grad(logits).grad(logits)
    expr = expr.full_simplify()
    print(expr)

    # Just softmax grad
    C = symbols("C")
    logits = Variable("logits", C)
    target = Variable("target", C)
    e = F.exp(logits)
    softmax = e / F.sum(e)
    expr = (softmax @ target).grad(logits)
    print(expr)

    # expr = 1 + F.sum(logits)
    # expr = expr.full_simplify()
    # print(expr)

    # print(to_tikz(expr))
    save_steps(expr)


def main2():
    i, j = symbols("i j")
    x = Variable("x", i)
    A = Variable("A", i, j=i)
    xAx = x.rename(i='j') @ A @ x
    xAxxAx = xAx @ xAx
    expr = Derivative(Derivative(xAxxAx, x), x)
    expr = expr.simplify()
    save_steps(expr)


def main3():
    i, j = symbols("i j")
    R = Variable("R", i, j)
    p = R / F.sum(R, edges=['i'])
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
    X = Copy(i, "i", "j") + u @ u.rename(i="j")# * Copy(eps)
    # X = u @ u.rename(i="j") @ Copy(eps)
    M = Variable("M", i, j=i).with_symmetries("i j")
    if K == 3:
        prod = F.graph('''*1 -i- X1 -j- M1
                       M1 -i- X2 -j- M2
                       M2 -i- X3 -j- M3
                       M3 -i- *1''',
                       X1=X, M1=M, X2=X, M2=M, X3=X, M3=M)
    if K == 2:
        prod = F.graph('''*1 -i- X1 -j- M1
                       M1 -i- X2 -j- M2
                       M2 -i- *1''',
                       X1=X, M1=M, X2=X, M2=M)
    expr = Expectation(prod, u)
    expr = expr.full_simplify()
    print(expr)
    print(len(expr.tensors))
    print("expanding")
    expr = expr.simplify({'expand': True})
    print(expr)
    print(len(expr.tensors))
    # save_as_image(expr, 'm3.png')

def main6():
    K = 2
    i = symbols("i")
    eps = symbols("e")
    u = Variable(f"u", i)
    X = Copy(i, "i", "j") - u @ u.rename(i="j") * Copy(eps)
    M = Variable("M", i, j=i)
    XM = F.graph("X -j-i- M", X=X, M=M)
    XMk = Copy(i, "i", "j")
    for k in range(K):
        XMk = F.graph("XMk -j-i- XM", XMk=XMk, XM=XM)
    trXMkXMkt = XMk @ XMk
    expr = Expectation(trXMkXMkt, u)
    # expr = Expectation(XMk, u) @ Copy(i, "i", "j")
    expr = expr.full_simplify()
    expr = expr.simplify({'extract_constants_from_expectation': True})
    print(expr)
    save_steps(expr)



def main7():
    i, j = symbols("i j")
    X = Variable("x", i, j)
    expr = Derivative(F.max(X, "i"), X)
    print(expr)
    save_steps(expr)

def main8():
    i, j = symbols("i j")
    X = Variable("X", i, j)

    # Test gradients using autograd
    res_max = F.max(X, ("i", "j"), keepdim=True)
    res = F.sum(res_max).grad(X)

    save_steps(res_max)


def main10():
    N, C = symbols("N C")
    x = Variable("logits", N=N, C=C)
    y = Variable("target", N=N, C=C)

    expr = x
    expr = F.sum(expr).grad(x)#.grad(x)
    print(expr)

    save_steps(expr)

def main11():
    N, C = symbols("N C")
    x = Variable("logits", N=N, C=C)
    expr = Product(
                [
                    #x,
                    Copy(N, "N"),
                    Copy(N, "N"),
                    Derivative(Copy(C, "C"), x, {"N": "N_", "C": "C_"}),
                    #Zero({'C': C, 'N_': N, 'C_': C})
                ]
            )
    print(Derivative(Copy(C, "C"), x, {"N": "N_", "C": "C_"}).simplify())
    print(Derivative(Copy(C, "C"), x, {"N": "N_", "C": "C_"}).shape)
    print(Derivative(Copy(C, "C"), x, {"N": "N_", "C": "C_"}).simplify().shape)
    print(to_tikz(expr))

if __name__ == "__main__":
    main11()

