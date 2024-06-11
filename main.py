from tensorgrad import Variable, Product, Function, Derivative, Sum, Copy, Zero, Ones, Unfold
from tensorgrad.tensor import TensorDict
from collections import defaultdict
import tensorgrad.functions as F
from tensorgrad.extras import Expectation
from tensorgrad.utils import generate_random_tensor_expression
from tensorgrad.extras.expectation import Expectation
from tensorgrad.imgtools import save_steps
from tensorgrad.random_tree import make_random_tree


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



def main():

    # A = Variable("A", ["i", "j"])
    # x = F.softmax(A, ["i"]).grad(A).simplify()

    # X = Variable("X", ["i"])
    # S = F.softmax(X, ["i"])
    # x = Derivative(S, X).simplify()

    # logits = Variable("logits", ["C"])
    # target = Variable("target", ["C"])

    # e = F.exp(logits)
    # softmax = e / F.sum(e)
    # ce = -F.sum(target * F.log(softmax))

    # expr = ce.grad(logits)
    # expr = expr.grad(logits)
    # expr = expr.full_simplify()

    # Derivation of Peyman Milanfar’s gradient, d[A(x)x]
    x = Variable("x", ["i"])
    A = Function("A", ["i", "j"], (x, "i"))
    expr = Derivative(A @ x, x)

    # Randomized Strassen
    SA = Variable("SA", "i, j, p1")
    SB = Variable("SB", "m, n, p2")
    W = Variable("W", "r, s, p3")
    expr = Product([SA, SB, W, Copy("p1, p2, p3")])
    expr = Expectation(expr, SA, Variable("ma", "i, j, p1"), Variable("CA", "i, j, p1, i2, j2, q1"))
    expr = Expectation(expr, SB, Variable("mb", "m, n, p2"), Variable("CB", "m, n, p2, m2, n2, q2"))
    expr = Expectation(expr, W, Variable("mw", "r, s, p3"), Variable("CW", "r, s, p3, r2, s2, q3"))

    # Same tensor
    S = Variable("S", "i, j, p")
    SA = S.rename({"p":"p1"})
    SB = S.rename({"p":"p2", "i":"m", "j":"n"})
    W = S.rename({"p":"p3", "i":"r", "j":"s"})
    expr = Product([SA, SB, W, Copy("p1, p2, p3")])
    # expr = Expectation(expr, S, Variable("ma", "i, j, p"), Variable("CA", "i, j, p, i2, j2, p2"))
    cov = Product([Copy("i, i2"), Copy("j, j2"), Copy("p, p2")])
    expr = Expectation(expr, S, Copy("i, j, p"), cov)

    # X = Variable("X", "b, x")
    # Y = Variable("Y", "b, y")
    # W = Variable("W", "x, y")
    # XWmY = X @ W - Y
    # l2 = F.sum(XWmY * XWmY)
    # expr = Derivative(l2, W)
    # expr = expr.simplify()

    # X = Variable("X", "b, x")
    # Y = Variable("Y", "b, y")
    # W = Variable("W", "x, y")
    # mu = Variable("mu", "x, y")
    # covar = Variable("C", "x, y, x2, y2")
    # XWmY = X @ W - Y
    # l2 = F.sum(XWmY * XWmY)
    # expr = Expectation(l2, W, mu, covar)

    #a = Variable("a", [])
    #x = Variable("x", ["i"])
    #expr = Derivative(a / x, x)

    #X = Variable("X", "i, j")
    #expr = Derivative(F.sum(X)**2, X)

    #b = Variable("b", ["i"])
    #c = Variable("c", ["i"])
    #X = Variable("X", "j, i")
    #expr = Derivative(b @ X @ X @ c, X)

    #x = Variable("x", "i")
    #f = F.softmax(x, dims=["i"])
    #expr = Derivative(f, x)
    #expr = Derivative(expr, x)

    x = Variable("x", "batch, din")
    W1 = Variable("W", "din1, dout1, k")
    #act = F.relu(x @ Copy("din, din1, din2") @ W1)
    act = x @ Copy("din, din1, din2") @ W1
    W2 = Variable("W2", "din2, dout2, k")
    expr = act @ Copy("dout1, dout2, dout") @ W2


    vectors, variables = make_random_tree(2)
    for vec in vectors:
        print(vec)
    for var in variables:
        print(var)
    expr = Product(vectors + variables)
    expr = F.frobenius2(expr)
    expected = 1
    for v in vectors:
        expected = expected * v @ v
    expr = expr - expected
    # expr = expr @ expr
    for v in variables:
        expr = Expectation(expr, v)
    # expr = expr.full_simplify()
    # expr = expr.full_simplify()
    # expr = expr.full_simplify()


    # S = Variable("S", "i, j, p")
    # SA = S.rename({"p":"p1"})
    # SB = S.rename({"p":"p2", "i":"m", "j":"n"})
    # W = S.rename({"p":"p3", "i":"r", "j":"s"})
    # expr = Product([SA, SB, W, Copy("p1, p2, p3")])
    # cov = Product([Copy("i, i2"), Copy("j, j2"), Copy("p, p2")])
    # expr = Expectation(expr, S, Copy("i, j, p"), cov)

    # A = Variable('A', 'i')
    # B = Variable('V', 'i')
    # expr = A @ B
    # expr = expr @ expr - B @ B
    # expr = Expectation(expr, A)

    # KAN network:
    # x = Variable("x", "batch, din")
    # W1 = Variable("W", "din1, dout1, k")
    # act = x @ Copy("din, din1, din2") @ W1
    # act = Function("f", [], (act,))
    # W2 = Variable("W2", "din2, dout2, k")
    # expr = act @ Copy("dout1, dout2, dout") @ W2

    # S = Variable("S", "i, j, p")
    # SA = S.rename({"p":"p1"})
    # SB = S.rename({"p":"p2", "i":"m", "j":"n"})
    # W = S.rename({"p":"p3", "i":"r", "j":"s"})
    # expr = Product([SA, SB, W, Copy("p1, p2, p3")])
    # # expr = Expectation(expr, S, Copy("i, j, p"))
    # # expr.full_simplify()

    # S = Variable("S", "p")
    # S1 = S.rename({"p":"p1"})
    # expr = Product([S, S1, Copy("p, p1, p2")])
    # expr = Expectation(expr, S)
    # # expr.full_simplify()


    # x = Variable("x", ["i"])
    # eps = Variable("eps", ["j"])
    # f = F.softmax(x, dims=["i"])
    # expr = F.taylor(f, x, eps, 2).simplify()

    #expr, _, _ = generate_random_tensor_expression(20)
    #expr = expr.simplify()

    # x = Variable("x", ["i"])
    # expr = Derivative(F.softmax(x, ["i"]), x)
    # expr = Derivative(expr, x)

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




if __name__ == "__main__":
    main()
