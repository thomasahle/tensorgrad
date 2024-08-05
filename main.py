from tensorgrad import Variable, Product, Function, Derivative, Sum, Copy, Zero, Ones, Unfold
from collections import defaultdict
import tensorgrad.functions as F
from tensorgrad.extras import Expectation
from tensorgrad.serializers.to_tikz import to_tikz
from tensorgrad.serializers.to_pytorch import to_pytorch
from tensorgrad.testutils import generate_random_tensor_expression, make_random_tree
from tensorgrad.imgtools import save_steps
from sympy import symbols




def main():
    i = symbols("i")
    a = Variable("a", i)
    b = Variable("b", i)
    X = Variable("X", i, j=i)
    expr = F.graph('a -i- X1 -j-i- X2 -j-i- b', a=a, X1=X, X2=X, b=b)
    # expr = Derivative(Derivative(graph, X), X)
    print(to_pytorch(expr))


    C = symbols("C")
    logits = Variable("logits", C)
    target = Variable("target", C)
    e = F.exp(logits)
    softmax = e / (1 + F.sum(e))
    ce = -F.sum(target * F.log(softmax))
    expr = ce.grad(logits).grad(logits)
    expr = expr.full_simplify()
    print(expr)

    # expr = 1 + F.sum(logits)
    # expr = expr.full_simplify()
    # print(expr)

    # print(to_tikz(expr))
    save_steps(expr)



if __name__ == "__main__":
    main()
