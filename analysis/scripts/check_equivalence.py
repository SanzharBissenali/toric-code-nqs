"""
analysis/scripts/check_equivalence.py
────────────────────────────────────────────────────────────────────────────
Identical-samples equivalence gate for the speed levers at the PRODUCTION
config (default L=4 OBC dual gridinv, hx=0.2 hz=0.26 hy=0.4, 8192 samples /
1024 chains / chunk 2048).

Builds the baseline state (inv_impl=conv, compute_dtype=float64), optionally
trains it --warm_steps so the noninv block has left the identity, draws ONE
sample set and freezes it; then, ON THOSE SAMPLES and the SAME parameters,
evaluates for each lever:
  * log psi                       (max relative deviation from the baseline),
  * the energy estimate, its variance and the energy gradient,
  * the SR update dp = (S + lam)^-1 g on the dense QGT,
  * the dense S matrix itself (unless --no_S).
Levers: qgt_solver kernel vs cholesky (same model); inv_impl dense (same
params, GEMM instead of conv); compute_dtype float32 (strict) and tf32 with the
double QGT twin (and, for contrast, with their own single-precision QGT).
Exact levers must sit at float64 roundoff (~1e-12); float32 at ~1e-6, tf32 ~1e-3.

    python analysis/scripts/check_equivalence.py --L 4 --warm_steps 5 --out check_L4.json
"""
import argparse
import json
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import netket as nk

from tc3d.builders import (build_state, run_loop, with_defaults, exact_qgt_apply_fun,
                           _qgt_dense_with_apply, _resolve_dense_solver)


def _flat(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])


def _rel(a, b):
    a, b = _flat(a), _flat(b)
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-300))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--bc", default="OBC")
    p.add_argument("--hx", type=float, default=0.2)
    p.add_argument("--hz", type=float, default=0.26)
    p.add_argument("--hy", type=float, default=0.4)
    p.add_argument("--n_samples", type=int, default=8192)
    p.add_argument("--n_chains", type=int, default=1024)
    p.add_argument("--chunk_size", type=int, default=2048)
    p.add_argument("--diag_shift", type=float, default=1e-3)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--warm_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_S", action="store_true", help="skip the dense S comparison")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    base = dict(L=a.L, bc=a.bc, dual_basis=True, hx=a.hx, hz=a.hz, hy=a.hy,
                arch="ToricCNN_gridinv", noninv_hidden=[4, 8], inv_hidden=[8, 8],
                kernel_size=a.L - 1, n_samples=a.n_samples, n_chains=a.n_chains,
                n_sweeps=48, chunk_size=a.chunk_size, seed=a.seed)
    geo, hi, Ham, vs, _ = build_state(with_defaults(base))
    print(f"[equiv] N={geo.N} n_params={vs.n_parameters} n_conn={Ham.max_conn_size}", flush=True)
    if a.warm_steps:
        run_loop(vs, Ham, n_iter=a.warm_steps, dt=a.dt, diag_shift=a.diag_shift,
                 qgt="dense", qgt_solver="cholesky")
    vs.reset()
    sig = vs.samples                                   # frozen sample set
    sig2 = sig.reshape((-1, geo.N))
    params = vs.parameters

    def state(**over):
        _, _, _, v, _ = build_state(with_defaults({**base, **over}), build_ham=False)
        v.parameters = params                          # identical tree by construction
        v._samples = sig                               # identical samples
        return v

    def sr_dp(v, grad, solver, apply_fun=None):
        qgt = (nk.optimizer.qgt.QGTJacobianDense if apply_fun is None
               else partial(_qgt_dense_with_apply, apply_fun=apply_fun))
        sr = nk.optimizer.SR(qgt=qgt, diag_shift=a.diag_shift, holomorphic=False,
                             solver=_resolve_dense_solver(solver))
        return sr(v, grad, 0)

    def S_of(v, apply_fun=None):
        if a.no_S:
            return None
        qgt = (nk.optimizer.qgt.QGTJacobianDense(v, diag_shift=a.diag_shift, holomorphic=False)
               if apply_fun is None else
               _qgt_dense_with_apply(v, apply_fun=apply_fun, diag_shift=a.diag_shift,
                                     holomorphic=False))
        return qgt.to_dense()

    lp0 = vs.log_value(sig2)
    E0, g0 = vs.expect_and_grad(Ham)
    dp0 = sr_dp(vs, g0, "cholesky")
    S0 = S_of(vs)
    out = {"config": {**base, "diag_shift": a.diag_shift, "warm_steps": a.warm_steps},
           "n_params": int(vs.n_parameters),
           "baseline": {"E": float(np.real(E0.mean)), "E_err": float(E0.error_of_mean),
                        "E_var": float(np.real(E0.variance))},
           "kernel_vs_cholesky": {"dp_rel": _rel(sr_dp(vs, g0, "kernel"), dp0)}}
    print(f"[equiv] E={out['baseline']['E']:.5f} ± {out['baseline']['E_err']:.5f}; "
          f"qgt_solver kernel vs cholesky: dp rel {out['kernel_vs_cholesky']['dp_rel']:.2e}",
          flush=True)

    levers = {"dense_f64": dict(inv_impl="dense"),
              "conv_f32": dict(compute_dtype="float32"),
              "dense_f32": dict(inv_impl="dense", compute_dtype="float32"),
              "dense_tf32": dict(inv_impl="dense", compute_dtype="tf32")}
    for name, over in levers.items():
        v = state(**over)
        lp = v.log_value(sig2)
        E, g = v.expect_and_grad(Ham)
        twin = exact_qgt_apply_fun(v)                  # None for the double model
        dp = sr_dp(v, g, "cholesky", apply_fun=twin)
        r = {"logpsi_rel": _rel(lp, lp0),
             "E_diff": float(np.real(E.mean - E0.mean)),
             "E_diff_over_err": float(np.real(E.mean - E0.mean) / E0.error_of_mean),
             "E_var_rel": float(abs(E.variance - E0.variance) / abs(E0.variance)),
             "grad_rel": _rel(g, g0), "dp_rel": _rel(dp, dp0)}
        if S0 is not None:
            r["S_rel"] = _rel(S_of(v, twin), S0)
        if twin is not None:                            # contrast: QGT from the float32 model
            r["dp_rel_float32_qgt"] = _rel(sr_dp(v, g, "cholesky"), dp0)
            if S0 is not None:
                r["S_rel_float32_qgt"] = _rel(S_of(v), S0)
        out[name] = r
        print(f"[equiv] {name:10s} " + "  ".join(f"{k}={val:.2e}" for k, val in r.items()),
              flush=True)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[equiv] wrote {a.out}")


if __name__ == "__main__":
    main()
