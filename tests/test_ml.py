from sympy import symbols

from tensorgrad import Variable, Function
from tensorgrad import functions as F
from tensorgrad.tensor import function


def test_attention():
    batch, seq, dim, inner, head = symbols("batch, seq, dim, inner, head")
    X = Variable("X", batch, seq, dim)
    W_q = Variable("W_q", dim, inner, head)
    W_k = Variable("W_k", dim, inner, head)
    W_v = Variable("W_v", dim, inner, head)
    W_o = Variable("W_o", dim, inner, head)

    query = (W_q @ X).rename(seq="seq_q")
    key = (W_k @ X).rename(seq="seq_k")
    value = (W_v @ X).rename(seq="seq_k")
    logits = F.dot(query, key, ["inner"])
    attention_scores = function("softmax", {"seq_k": seq}, (logits, "seq_k"))
    expr = F.dot(value, attention_scores, ["seq_k"])
    expr = F.dot(W_o, expr, ["inner", "head"])


def test_attention_2():
    batch, seq, dim, inner, head = symbols("batch, seq, dim, inner, head")
    X = Variable("X", batch, seq, dim)
    W_q = Variable("W_q", dim, inner, head)
    W_k = Variable("W_k", dim, inner, head)
    W_v = Variable("W_v", dim, inner, head)
    W_o = Variable("W_o", dim, inner, head)

    logits = F.graph(
        """
        X1 -dim- Wq
        X2 -dim- Wk
        X2 -seq-sk-
        Wq -inner- Wk
        """,
        X1=X,
        X2=X,
        Wq=W_q,
        Wk=W_k,
    )
    assert logits.edges == {"batch", "seq", "sk", "head"}

    attention_scores = F.softmax(logits, ["sk"])

    expr = F.graph(
        """
        X -seq-sk- attention_scores -head- *0
        X -dim- Wv -head- *0
        *0 -head- Wo
        Wv -inner- Wo
        """,
        X=X,
        Wv=W_v,
        Wo=W_o,
        attention_scores=attention_scores,
    )
    assert expr.edges == {"batch", "seq", "dim"}
