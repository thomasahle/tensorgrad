from tensorgrad import Variable, Product, Derivative, Delta, Zero, function
import tensorgrad.functions as F
from tensorgrad import Expectation
from tensorgrad.extras.to_tikz import to_tikz
from tensorgrad.imgtools import save_steps, save_as_image
from tensorgrad.extras.to_pytorch import compile_to_callable
from tensorgrad.extras.to_index import to_index, to_index_free
from tensorgrad.testutils import assert_close, rand_values
from sympy import symbols

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
    #print(to_index_free(expr.full_simplify()))

    # Just softmax grad
    C = symbols("C")
    logits = Variable("logits", C)
    target = Variable("target", C)
    e = F.exp(logits)
    softmax = e / F.sum(e)
    expr = Derivative(softmax @ target, logits)
    print(expr)
    #print(to_index_free(expr.full_simplify()))

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
    u = Variable("u", i)
    C = Variable("C", i, j=i).with_symmetries("i j")
    prod = Product([u.rename(i=f"i{k}") for k in range(K)])
    expr = Expectation(prod, u, mu=Zero(i), covar=C, covar_names={"i": "j"})
    expr = expr.full_simplify()
    save_steps(expr)


def main5():
    K = 3
    i = symbols("i")
    u = Variable("u", i)
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
    out = compiled_fn(vals, dims)
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
    g = Variable("g", i=d)
    A = Delta(d, "i", "j") - g @ g.rename(i='j') / Delta(d)
    M = Variable("M", i=d, j=d).with_symmetries("i j")

    assert M == M.rename(i="j", j="i")

    C = F.multi_dot([A,M] * 2, dims=("i", "j"))
    assert C.edges == {"i", "j"}, A.edges
    frob = C @ C
    B = Expectation(frob, g).full_simplify()
    save_steps(B)

def main17():
    i = symbols("i")
    X = Variable("X", i, j=i)
    det = F.det(X)
    inv = F.inverse(X)
    adj = det * inv
    expr = det.grad(X).grad(X).grad(X)

    expected = (
        det
        * F.symmetrize(
            inv @ inv.rename(i="i_", j="j_") @ inv.rename(i="i__", j="j__"),
            dims={"j", "j_", "j__"},
            signed=True,
        ).full_simplify()
    )

    save_steps(expr)
    print(to_tikz(expr))

def main18():
    b, x, y = symbols("b x y")
    X = Variable("X", x)
    eps = Variable("eps", x)
    Y = F.softmax(X, dim='x').simplify({'expand_functions': True})
    expr = F.taylor(Y, X, eps, n=2)
    expr = expr / Y
    save_steps(expr)

def main19(n=2, symmetric=True):
    i = symbols("i")
    M = Variable("M", i, j=i)
    if symmetric:
        M = M.with_symmetries("i j")
    u = Variable("u", i)
    U = Delta(i, "i", "j") - u @ u.rename(i="j") / Delta(i)
    #TrUMs = F.graph("*1 -i- U1 -j-i- M1 -j-i- U2 -j-i- M2 -j- *1",
                    #U1=U, U2=U, M1=M, M2=M)
    UMs = F.multi_dot([U, M]*n, dims=("i", "j"))
    TrUMs = F.trace(UMs)
    expr = Expectation(TrUMs, u)
    save_steps(expr.full_simplify())

def main20():
    # Define sizes for the tensor edges and variables
    d = symbols("d")
    T = Variable("T", i=d, j=d)
    eps = Variable("eps", i=d, j=d)

    # This is the cumulant generating function of (I - gg^t/d).
    # which is Tr(T) - log(det(I + 2T/d))/2
    I = Delta(d, 'i', 'j')
    expr = F.trace(T) - F.log(F.det(I+2*T/Delta(d)))/2

    tayl = F.taylor(expr, T, eps, n=2)
    tayl = tayl.full_simplify()  # Need this, since we can't sub through grad
    tayl = tayl.substitute(T, Zero(i=d, j=d))

    # Simplify the expression and save the steps
    save_steps(tayl)

    # Press "Run" to see the result!

def main21():
    # Define sizes for the tensor edges and variables
    d = symbols("d")
    g = Variable("g", i=d)
    A = Variable("A", i=d, j=d)
    B = function("B", {"i": d, "j": d}, (g, "i"))

    U = Delta(d, "i", "j") - g @ g.rename(i="j") / Delta(d)

    expr = F.multi_dot([A, U, B], ("i", "j"))
    expr = F.trace(expr)

    e = Expectation(expr, g)
    save_steps(e)


def main22(signature:str):
    i = symbols("i")
    M = Variable("M", i, j=i)
    T = M.rename(i="j", j="i")
    u = Variable("u", i)
    U = Delta(i, "i", "j") - u @ u.rename(i="j") / Delta(i)
    cycles = [[]]
    for s in signature:
        assert s in 'MT|'
        if s == 'M':
            cycles[-1] += [U, M]
        elif s == 'T':
            cycles[-1] += [T, U]
        elif s == '|':
            cycles.append([])
    print(cycles)
    traces = [F.trace(F.multi_dot(cycle, dims=("i", "j"))) for cycle in cycles]
    prod = Product(traces)
    expr = Expectation(prod, u)
    save_steps(expr.full_simplify())

def main23(n:int, m:int):
    # n is the number of cycles, m is the number of indep. variables
    # eg.. (3, 2) is Tr(MMM) where each M is U1 U2
    i = symbols("i")
    us = [Variable(f"u{k}", i) for k in range(m)]
    Us = [Delta(i, "i", "j") - u @ u.rename(i="j") / Delta(i) for u in us]
    tr = F.trace(F.multi_dot(Us * n, dims=("i", "j"))) / Delta(i)
    for u in us:
        tr = Expectation(tr, u)
    save_steps(tr.full_simplify())

def main24(signature:tuple[str], m:int=2):
    i = symbols("i")
    us = [Variable(f"u{k}", i) for k in range(m)]
    Us = [Delta(i, "i", "j") - u @ u.rename(i="j") / Delta(i) for u in us]
    M = F.multi_dot(Us, dims=("i", "j"))
    cycles = []
    for part in signature:
        cycle = []
        for s in part:
            if s == 'M':
                cycle.append(M)
            elif s == 'T':
                cycle.append(M.rename(i="j", j="i"))
        cycles.append(F.trace(F.multi_dot(cycle, dims=("i", "j"))))
    prod = Product(cycles)
    for u in us:
        prod = Expectation(prod, u)
    save_steps(prod.full_simplify())


def tg_to_signature(tensor):
    signature = []
    for w, term in zip(tensor.weights, tensor.tensors):
        part = []
        for f in term.factors:
            if f.name == "M":
                part.append("M")
            elif f.name == "T":
                part.append("T")
        signature.append(part)
    return tuple(signature)

def main25(signature:tuple[str]):
    i = symbols("i")
    u = Variable("u", i)
    U = Delta(i, "i", "j") - u @ u.rename(i="j") / Delta(i)
    M = Variable("M", i, j=i)
    UM = F.dot(U, M, dim=('j', 'i'))
    UM2 = F.graph("U -j-i- M", U=U, M=M)
    assert UM.simplify() == UM2.simplify()
    cycles = []
    for part in signature:
        cycle = []
        for s in part:
            if s == 'M':
                cycle.append(UM)
            elif s == 'T':
                cycle.append(UM.rename(i="j", j="i"))
        cycles.append(F.trace(F.multi_dot(cycle, dims=("i", "j"))))
    prod = Product(cycles)
    prod = Expectation(prod, u).full_simplify()
    print(to_index(prod))
    print(to_index_free(prod))
    save_steps(prod)

if __name__ == "__main__":
    #main24(("MMM",), m=2)
    #main24(("MMTT",), m=2)
    #main23(n=3, m=2)
    #main25(("MMTT",))
    main()
