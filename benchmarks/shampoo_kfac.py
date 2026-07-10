"""Shampoo and K-FAC as tensor diagrams: correctness + the factorization win.

Both optimizers precondition the gradient by (an approximation of) the
full-matrix Adagrad statistic

    H_t = sum_s  vec(G_s) vec(G_s)^T          (Shampoo paper, arXiv:1802.09568)

which for a parameter with N entries is an N x N matrix -- inverting it costs
O(N^3) and storing it O(N^2). The whole trick of both methods is to never
form H: factor it as a *tensor (Kronecker) product* of small per-axis
matrices and invert those. The algebra is heavy in index notation but is one
picture in tensor-diagram notation, so it is a natural tensorgrad example.

Three facts, each checked numerically below against a dense torch reference:

1. SHAMPOO preconditioner statistic = a contraction.  For an order-k gradient
   tensor G_t, the mode-i preconditioner is the "Gram matrix" of the mode-i
   unfolding: contract two copies of G_t over EVERY axis except i.

       H^(i)_{a a'}  =  sum_{rest}  G_{a, rest} G_{a', rest}          (= G G^T)

   In a diagram that is two copies of the G_t box with all legs joined except
   leg i (the user's top-left picture: legs 1,2,3,5 looped, leg 4 free gives
   the mode-4 factor). Eigendecomposing these H^(i) is exactly one step of a
   Higher-Order SVD, which is why Shampoo "is a single HOSVD step".

2. K-FAC factorization = the natural outer-product gradient of a linear layer.
   For y = W a, the gradient of y wrt W is delta_{i i'} (x) a_{j'} -- an
   OUTER PRODUCT of an identity on the output leg and the input activation a
   (derived symbolically below with .grad, no hand algebra). Backpropagating a
   loss cotangent g gives the weight gradient G = g a^T, so the Fisher block

       E[vec G vec G^T] = E[a a^T (x) g g^T]  ~=  E[a a^T] (x) E[g g^T] = A (x) S

   is a Kronecker product of the input covariance A and the output-gradient
   covariance S -- the user's top-right picture (a_t a_t on top, g_t g_t
   below, split into the two boxes).

3. The shared punchline -- inverse of a tensor product is the tensor product
   of the inverses:

       (H^(1) (x) H^(2) (x) ... (x) H^(k))^{-1}
                       =  H^(1)^{-1} (x) H^(2)^{-1} (x) ... (x) H^(k)^{-1}

   so preconditioning contracts G_t with each small inverse along its own leg
   instead of applying one N x N inverse (the user's bottom-right picture: the
   big bracketed inverse equals the product of the small bracketed inverses).
   The identity holds for ANY scalar power p, (A (x) B)^p = A^p (x) B^p, so
   Shampoo's actual exponent -1/(2k) (= -1/4 for a matrix, k=2) slots into the
   identical contraction -- only the scalar matrix-function changes, which is
   orthogonal to the diagram. We validate the exact-inverse (p = -1) form,
   which tensorgrad expresses fully symbolically via F.inverse.

Verdict on the user's diagram: it is correct. (One label typo: the bottom-left
stack reads G^(1),G^(2),G^(3),G^(3),G^(4); for a 5-mode tensor it should run
G^(1)..G^(5).)

Run `python benchmarks/shampoo_kfac.py` for the identity confirmations and the
factored-vs-dense timing. No timing-campaign markers, so run.py skips it; it is
a self-verifying example, gated at import.
"""

import time

import torch
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.tensor import Delta
from tensorgrad.extras.evaluate import evaluate

torch.set_grad_enabled(False)
torch.set_num_threads(2)

EPS = 1e-1  # Tikhonov ridge on each factor (Shampoo's epsilon I), keeps SPD


# --------------------------------------------------------------------------
# 1 + 3.  Shampoo: mode Gram matrices, and the factored preconditioner.
# --------------------------------------------------------------------------
def shampoo_symbolic(axes: list[str]):
    """Build, for an order-k gradient tensor G over `axes`, the symbolic
    mode-i Gram matrices H^(i) (each ridged with EPS*I) and the factored
    preconditioned gradient  U = G contracted with every H^(i)^{-1}.

    Returns (G, [H_i...], U). Every H^(i) is one contraction of two G copies;
    U applies the factored inverse -- no N x N object is ever built."""
    dims = symbols(" ".join(axes))
    dims = [dims] if len(axes) == 1 else list(dims)
    G = Variable("G", **{e: d for e, d in zip(axes, dims)})

    His = []
    for e, d in zip(axes, dims):
        e2 = e + "2"
        gram = G @ G.rename(**{e: e2})  # contract over all-but-e -> (e, e2)
        His.append(gram + EPS * Delta(d, e, e2))  # ridge: + EPS I

    # apply each factor's inverse along its own leg (the tensor-product inverse)
    U = G
    for e, H in zip(axes, His):
        e2 = e + "2"
        inv = F.inverse(H, dims={e, e2})  # small d x d inverse
        U = inv @ U.rename(**{e: e2})  # contract e2 -> restores leg e
    return G, His, U


def dense_preconditioned(Gv: torch.Tensor) -> torch.Tensor:
    """Reference: build the FULL Kronecker preconditioner P = (x)_i H^(i)
    (an N x N matrix, N = prod of dims), and apply P^{-1} to vec(G_t). This is
    exactly what the factored form avoids."""
    dims = list(Gv.shape)
    facts = []
    for i, d in enumerate(dims):
        Mi = Gv.movedim(i, 0).reshape(d, -1)  # mode-i unfolding
        facts.append(Mi @ Mi.T + EPS * torch.eye(d))
    P = facts[0]
    for f in facts[1:]:
        P = torch.kron(P, f)  # row-major, matching axis order
    return torch.linalg.solve(P, Gv.reshape(-1)).reshape(*dims)


# --------------------------------------------------------------------------
# 2.  K-FAC: the gradient of a linear layer IS an outer product.
# --------------------------------------------------------------------------
def kfac_linear_gradient():
    """Return the symbolic dy/dW for y = W a (a linear layer). tensorgrad
    derives the outer-product structure delta (x) a with no hand algebra."""
    i, j = symbols("i j")
    W = Variable("W", i, j)
    a = Variable("a", j)
    y = W @ a  # output leg i
    return W, a, y.grad(W).simplify()


# --------------------------------------------------------------------------
# Correctness gate (runs at import).
# --------------------------------------------------------------------------
def _check() -> None:
    torch.manual_seed(0)

    # -- Shampoo, order-3 tensor gradient (the k > 2 case: a genuine tensor) --
    axes = ["a", "b", "c"]
    shp = (3, 4, 5)
    G, His, U = shampoo_symbolic(axes)
    dims = {G.shape[e]: n for e, n in zip(axes, shp)}
    Gv = torch.randn(*shp)
    env = {G: Gv.refine_names(*axes)}

    # (1) each H^(i) equals the mode-i unfolding Gram G G^T (+ ridge)
    for i, (e, H) in enumerate(zip(axes, His)):
        Hv = evaluate(H, env, dims).align_to(e, e + "2").rename(None)
        Mi = Gv.movedim(i, 0).reshape(shp[i], -1)
        assert torch.allclose(Hv, Mi @ Mi.T + EPS * torch.eye(shp[i]), atol=1e-4)

    # (3) factored preconditioning == dense full-Kronecker inverse
    Uv = evaluate(U, env, dims).align_to(*axes).rename(None)
    assert torch.allclose(Uv, dense_preconditioned(Gv), atol=1e-4), \
        (Uv - dense_preconditioned(Gv)).abs().max().item()

    # sanity: the matrix (k=2) case too -- this is matrix-Shampoo / K-FAC shape
    G2, _, U2 = shampoo_symbolic(["m", "n"])
    d2 = {G2.shape["m"]: 6, G2.shape["n"]: 7}
    Gv2 = torch.randn(6, 7)
    Uv2 = evaluate(U2, {G2: Gv2.refine_names("m", "n")}, d2).align_to("m", "n").rename(None)
    assert torch.allclose(Uv2, dense_preconditioned(Gv2), atol=1e-4)

    # (2) K-FAC: dy/dW is the outer product delta_{i i'} (x) a_{j'}
    W, a, g = kfac_linear_gradient()
    # numeric check: contract the local Jacobian with a random cotangent u_i,
    # weight-gradient must equal outer(u, a)
    i, j = W.shape["i"], W.shape["j"]
    di, dj = 4, 5
    uv = torch.randn(di)
    av = torch.randn(dj)
    u = Variable("u", i=i)
    wg = (u @ g).simplify()  # cotangent (x) local Jacobian -> weight gradient
    wgv = evaluate(wg, {a: av.refine_names("j"), u: uv.refine_names("i")},
                   {i: di, j: dj})
    assert torch.allclose(
        wgv.align_to("i_", "j").rename(None), torch.outer(uv, av), atol=1e-4
    ), "K-FAC weight gradient is not the outer product g a^T"


_check()  # correctness gate


# --------------------------------------------------------------------------
# Timing: factored (k small inverses) vs dense (one N x N inverse).
# --------------------------------------------------------------------------
def _time_factored_vs_dense(shp=(16, 16, 16), reps=50) -> None:
    axes = ["a", "b", "c"][: len(shp)]
    G, His, U = shampoo_symbolic(axes)
    dims = {G.shape[e]: n for e, n in zip(axes, shp)}
    Gv = torch.randn(*shp)
    env = {G: Gv.refine_names(*axes)}

    t0 = time.perf_counter()
    for _ in range(reps):
        evaluate(U, env, dims)
    t_fac = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        dense_preconditioned(Gv)
    t_dense = (time.perf_counter() - t0) / reps

    N = 1
    for d in shp:
        N *= d
    print(f"  shape {shp}: full preconditioner is {N} x {N}")
    print(f"  factored (tensorgrad, {len(shp)} small inverses): {1e3 * t_fac:7.2f} ms")
    print(f"  dense    (one {N}x{N} inverse):                  {1e3 * t_dense:7.2f} ms")
    print(f"  speedup: {t_dense / t_fac:5.1f}x")


if __name__ == "__main__":
    print("Shampoo / K-FAC as tensor diagrams -- identities confirmed at import.\n")

    _, _, g = kfac_linear_gradient()
    print("K-FAC: grad(W @ a, W) simplifies to the outer product")
    print(f"       {g}\n")

    G, His, U = shampoo_symbolic(["a", "b", "c"])
    print("Shampoo order-3: each mode factor is one contraction of two G copies,")
    print("       e.g. H^(a) =", His[0].simplify(), "\n")

    print("Factored vs dense preconditioning (same result, different cost):")
    _time_factored_vs_dense((16, 16, 16))
