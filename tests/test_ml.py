from sympy import symbols

from tensorgrad import Variable, Function
from tensorgrad import functions as F


def test_attention():
    seq, dim, inner, head = symbols("seq, dim, inner, head")
    X = Variable("X", seq, dim)
    W_q = Variable("W_q", dim, inner, head)
    W_k = Variable("W_k", dim, inner, head)
    W_v = Variable("W_v", dim, inner, head)
    W_o = Variable("W_o", dim, inner, head)
    query = (W_q @ X).rename(seq="seq_q")
    key = (W_k @ X).rename(seq="seq_k")
    value = (W_v @ X).rename(seq="seq_k")
    logits = F.dot(query, key, ["inner"])
    attention_scores = Function("softmax", {"seq_k": seq}, (logits, "seq_k"))
    expr = F.dot(value, attention_scores, ["seq_k"])
    expr = F.dot(W_o, expr, ["inner", "head"])
