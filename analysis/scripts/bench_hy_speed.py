"""
analysis/scripts/bench_hy_speed.py
────────────────────────────────────────────────────────────────────────────
PROTOTYPE benchmark driver for the hy!=0 complex-lane speed investigation
(NOT production). Builds the exact production dual_basis/OBC/ToricCNN_gridinv
config (matching results/hy_axis campaign JSONs: kernel_size=L-1,
noninv_hidden=[4,8], inv_hidden=[8,8], n_samples=8192, n_chains=1024,
chunk_size=2048, n_sweeps=48), optionally warm-starts from an existing
checkpoint (--init_from BASE, path without .mpack) for a REPRESENTATIVE
state at a given hy (avoids re-training just to get a realistic polarized
state), then runs `tc3d.builders.run_loop(time_phases=True)` for a handful
of steps under a chosen qgt/qgt_solver variant and dumps the per-step [t]
breakdown + median summary to JSON.

Usage:
    python analysis/scripts/bench_hy_speed.py \
        --L 4 --hy 1.4 --qgt dense --qgt_solver cholesky \
        --init_from $OUT/gridinv_dual_L4_OBC_hx0.0_hz0.0_hy1.4_..._k3 \
        --n_iter 8 --out results_bench/L4_hy1.4_dense_cholesky.json
"""
import argparse
import json
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from tc3d.builders import build_state, run_loop, with_defaults
from tc3d.io import load_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--bc", default="OBC")
    p.add_argument("--hy", type=float, required=True)
    p.add_argument("--hx", type=float, default=0.0)
    p.add_argument("--hz", type=float, default=0.0)
    p.add_argument("--kernel_size", type=int, default=None,
                   help="default L-1 (the production/tune-rect winner)")
    p.add_argument("--n_samples", type=int, default=8192)
    p.add_argument("--n_chains", type=int, default=1024)
    p.add_argument("--n_sweeps", type=int, default=48)
    p.add_argument("--chunk_size", type=int, default=2048)
    p.add_argument("--diag_shift", type=float, default=0.001)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--qgt", default="dense", choices=["dense", "onthefly", "srt", "minsr", "auto"])
    p.add_argument("--qgt_solver", default=None,
                   help="dense-only: cg (default)/cgN/cholesky/solve")
    p.add_argument("--n_iter", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_from", default=None, metavar="WEIGHTS_BASE",
                   help="warm-start from {base}.mpack for a representative "
                        "(e.g. deep-polarized) state instead of cold init")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ks = args.kernel_size if args.kernel_size is not None else args.L - 1
    cfg = with_defaults(dict(
        L=args.L, bc=args.bc, dual_basis=True, hx=args.hx, hy=args.hy, hz=args.hz,
        arch="ToricCNN_gridinv", noninv_hidden=[4, 8], inv_hidden=[8, 8],
        kernel_size=ks, n_samples=args.n_samples, n_chains=args.n_chains,
        n_sweeps=args.n_sweeps, chunk_size=args.chunk_size, seed=args.seed,
    ))

    t0 = time.time()
    geo, hi, Ham, vs, xz = build_state(cfg)
    t_build = time.time() - t0
    print(f"[bench] build_state: {t_build:.1f}s  n_params={vs.n_parameters}  "
          f"N={geo.N}", flush=True)

    warm = False
    if args.init_from:
        try:
            vs = load_weights(vs, args.init_from)
            warm = True
            print(f"[bench] warm-started from {args.init_from}.mpack", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[bench] init_from failed ({type(e).__name__}: {e}); "
                  f"cold start.", flush=True)

    timing_log = []

    def on_timing(step, td):
        timing_log.append({"step": step, **{k: float(v) for k, v in td.items()}})

    t0 = time.time()
    out = run_loop(vs, Ham, n_iter=args.n_iter, dt=args.dt,
                   diag_shift=args.diag_shift, qgt=args.qgt,
                   qgt_solver=args.qgt_solver, time_phases=True,
                   on_timing=on_timing)
    wall = time.time() - t0
    # run_loop's return convention differs by branch: bare vs, or (vs, n_rollbacks)
    vs_final = out[0] if isinstance(out, tuple) else out

    rows = timing_log[1:] if len(timing_log) > 1 else timing_log  # drop compile step
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]} if rows else {}

    # GPU peak-memory headroom check (dense-QGT S matrix is the usual worry at
    # large L): jax's per-device memory_stats() is CUDA-only and best-effort
    # (returns None on CPU or if the backend doesn't implement it).
    mem_stats = None
    try:
        dev = jax.devices()[0]
        if "cuda" in str(dev).lower() or "gpu" in str(dev).lower():
            ms = dev.memory_stats()
            if ms:
                mem_stats = {"peak_bytes_in_use": ms.get("peak_bytes_in_use"),
                            "bytes_in_use": ms.get("bytes_in_use"),
                            "bytes_limit": ms.get("bytes_limit")}
    except Exception as e:                                            # noqa: BLE001
        print(f"[bench] memory_stats unavailable: {type(e).__name__}: {e}", flush=True)

    result = {
        "config": {**cfg, "qgt": args.qgt, "qgt_solver": args.qgt_solver,
                   "n_iter": args.n_iter, "init_from": args.init_from,
                   "warm_started": warm},
        "n_params": int(vs_final.n_parameters),
        "build_time_s": t_build,
        "wall_total_s": wall,
        "timing_per_step": timing_log,
        "median_excl_compile": med,
        "gpu_memory_stats": mem_stats,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench] median (excl. compile step): {med}", flush=True)
    if mem_stats and mem_stats.get("peak_bytes_in_use") is not None:
        print(f"[bench] GPU peak_bytes_in_use: "
              f"{mem_stats['peak_bytes_in_use'] / 2**30:.2f} GiB "
              f"(bytes_limit: {mem_stats.get('bytes_limit', 0) / 2**30:.1f} GiB)",
              flush=True)
    print(f"[bench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
