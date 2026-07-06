"""JAX baseline for the second-order benches: uv run --with jax python benchmarks/second_order_jax.py"""
import os, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
os.environ["OMP_NUM_THREADS"] = "2"
import jax, jax.numpy as jnp
import numpy as np

def bench(f, budget_s=2.0):
    f()
    t0 = time.perf_counter(); reps = 0
    while time.perf_counter() - t0 < budget_s:
        f(); reps += 1
    return (time.perf_counter() - t0) / reps * 1e3

print(f"{'case':8s} {'N':>5s} {'jax ms':>9s}")
for N in (64, 256, 1024):
    key = jax.random.PRNGKey(0)
    xv = jax.random.normal(key, (N,)); Av = jax.random.normal(key, (N, N))
    hq = jax.jit(jax.hessian(lambda xx: xx @ Av @ xx))
    r = hq(xv); r.block_until_ready()
    t = bench(lambda: hq(xv).block_until_ready())
    print(f"{'quad':8s} {N:5d} {t:9.3f}")
for N in (64, 256, 1024):
    xv = jax.random.normal(jax.random.PRNGKey(1), (N,))
    hl = jax.jit(jax.hessian(lambda xx: jax.scipy.special.logsumexp(xx)))
    hl(xv).block_until_ready()
    t = bench(lambda: hl(xv).block_until_ready())
    print(f"{'lse':8s} {N:5d} {t:9.3f}")
