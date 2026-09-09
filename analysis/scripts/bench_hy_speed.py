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

At L>=5 cold-start, build_state's Hamiltonian assembly alone can take
15-25+ min (measured: L=5 up to 1228s, L=6 exceeded the entire 30-min
gpu_debug cap TWICE outright) -- each is its own fresh process, so the cost
is paid once per invocation regardless of --qgt_solver. --qgt_solvers
(plural, space-separated) amortizes that ONE-TIME cost across several
solver variants in a SINGLE process/build_state call: params+sampler_state
are snapshotted right after build (+ optional warm start) and restored
before EACH variant's run_loop call, so every variant times the identical
starting state (a fair A/B, and step-0 JIT-compile cost is still excluded
from each variant's own median, same convention as the single-solver path).

Usage (single solver, unchanged):
    python analysis/scripts/bench_hy_speed.py \
        --L 4 --hy 1.4 --qgt dense --qgt_solver cholesky \
        --init_from $OUT/gridinv_dual_L4_OBC_hx0.0_hz0.0_hy1.4_..._k3 \
        --n_iter 8 --out results_bench/L4_hy1.4_dense_cholesky.json

Usage (multi-solver A/B sharing one build_state, e.g. for the 30-min cap at
large L where a second process's H-assembly wouldn't fit):
    python analysis/scripts/bench_hy_speed.py \
        --L 6 --hy 0.3 --qgt dense --qgt_solvers cg cholesky \
        --n_iter 4 --out results_bench/L6_hy0.3_dense_AB.json
"""
import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from tc3d.builders import build_state, run_loop, with_defaults, exact_qgt_apply_fun
from tc3d.io import load_weights


def _gpu_memory_stats():
    """GPU peak-memory headroom check (dense-QGT S matrix is the usual worry
    at large L): jax's per-device memory_stats() is CUDA-only and best-effort
    (returns None on CPU or if the backend doesn't implement it)."""
    try:
        dev = jax.devices()[0]
        if "cuda" in str(dev).lower() or "gpu" in str(dev).lower():
            ms = dev.memory_stats()
            if ms:
                return {"peak_bytes_in_use": ms.get("peak_bytes_in_use"),
                        "bytes_in_use": ms.get("bytes_in_use"),
                        "bytes_limit": ms.get("bytes_limit")}
    except Exception as e:                                            # noqa: BLE001
        print(f"[bench] memory_stats unavailable: {type(e).__name__}: {e}", flush=True)
    return None


def _model_flops(cfg, geo, n_v):
    """Real multiply-adds of ONE forward evaluation of the production dual gridinv
    (noninv GeoConv3D stack + invariant convs), x4 for a complex ansatz -- so
    the microbench can quote achieved TFLOP/s next to us/config."""
    S, Pe = 15, geo.N // 3
    widths = [1] + list(cfg["noninv_hidden"] or [cfg["noninv_channels"]] * cfg["n_noninv"])
    macs = sum(3 * Pe * S * a * b for a, b in zip(widths[:-1], widths[1:]))
    k3 = (cfg["kernel_size"] or cfg["L"]) ** 3
    inv = [widths[-1]] + list(cfg["inv_hidden"]) + [1]
    macs += sum(n_v * k3 * a * b for a, b in zip(inv[:-1], inv[1:]))
    return 2 * macs * (4 if cfg["dtype"] == "complex" else 1)


def _microbench(vs, Ham, cfg, geo, n_rep=5):
    """Explain the `grad` stage from first principles: NetKet's E_loc kernel feeds the
    network `chunk_size // n_conn` samples' worth of connected states per call
    (chunk >= n_conn), i.e. n_samples*n_conn forward evaluations per step, plus
    get_conn_padded per chunk. Times ONE such forward call of THIS model (us/config,
    achieved TFLOP/s) and one get_conn_padded call, extrapolates both to a step, and
    takes an HLO census of the compiled forward (how the conv is lowered: cuDNN
    custom-call vs. XLA decomposition; how many dot/gather kernels)."""
    N, n_conn = vs.hilbert.size, int(Ham.max_conn_size)
    chunk = cfg["chunk_size"] or cfg["n_samples"]
    ns = max(1, chunk // n_conn)
    per_call = ns * n_conn if chunk >= n_conn else chunk
    sig = jnp.asarray(np.random.default_rng(0).choice([-1, 1], size=(per_call, N)),
                      dtype=jnp.int8)
    fwd = jax.jit(vs._apply_fun)
    txt = fwd.lower(vs.variables, sig).compile().as_text()
    census = {k: txt.count(k) for k in ("convolution(", "custom-call(", "cudnn",
                                        "cublas", " dot(", "gather(", "scatter(")}
    fwd(vs.variables, sig).block_until_ready()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fwd(vs.variables, sig).block_until_ready()
        ts.append(time.perf_counter() - t0)
    t_fwd = min(ts)
    Ham.get_conn_padded(sig[:ns])
    tc = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        jax.block_until_ready(Ham.get_conn_padded(sig[:ns]))
        tc.append(time.perf_counter() - t0)
    t_conn = min(tc)
    flops = _model_flops(cfg, geo, len(geo.vertex_all))
    res = {
        "n_conn": n_conn, "N": N, "per_call_configs": per_call,
        "fwd_us_per_config": 1e6 * t_fwd / per_call,
        "fwd_TFLOPs": flops * per_call / t_fwd / 1e12,
        "flops_per_config": flops,
        "evals_per_step": cfg["n_samples"] * n_conn,
        "eloc_forward_s_per_step_est": cfg["n_samples"] * n_conn * t_fwd / per_call,
        "get_conn_padded_us_per_sample": 1e6 * t_conn / ns,
        "get_conn_padded_s_per_step_est": cfg["n_samples"] * t_conn / ns,
        "hlo_census": census,
    }
    print(f"[micro] n_conn={n_conn} N={N} batch/call={per_call}: forward "
          f"{res['fwd_us_per_config']:.2f} us/config ({res['fwd_TFLOPs']:.2f} TFLOP/s); "
          f"E_loc forward est {res['eloc_forward_s_per_step_est']:.1f} s/step over "
          f"{res['evals_per_step']} evals; get_conn_padded est "
          f"{res['get_conn_padded_s_per_step_est']:.2f} s/step; HLO {census}", flush=True)
    return res


def _run_one(vs, Ham, snapshot, args, qgt_solver):
    """Restore `snapshot` (fair A/B: every variant starts identically), run
    run_loop for one solver variant, return (timing_log, median, vs_final)."""
    params, sstate = snapshot
    vs.parameters = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)
    vs.sampler_state = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), sstate)

    timing_log = []

    def on_timing(step, td):
        timing_log.append({"step": step, **{k: float(v) for k, v in td.items()}})

    out = run_loop(vs, Ham, n_iter=args.n_iter, dt=args.dt,
                   diag_shift=args.diag_shift, qgt=args.qgt,
                   qgt_solver=qgt_solver, time_phases=True, on_timing=on_timing,
                   qgt_apply_fun=exact_qgt_apply_fun(vs))
    vs_final = out[0] if isinstance(out, tuple) else out   # (vs) or (vs, n_rollbacks)

    rows = timing_log[1:] if len(timing_log) > 1 else timing_log  # drop compile step
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]} if rows else {}
    return timing_log, med, vs_final


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
    p.add_argument("--qgt_solvers", nargs="+", default=None,
                   help="A/B several solvers sharing ONE build_state call "
                        "(e.g. --qgt_solvers cg cholesky); overrides --qgt_solver")
    p.add_argument("--compute_dtype", default=None, choices=[None, "float64", "float32", "tf32"],
                   help="ansatz arithmetic precision (train.py flag); QGT stays double")
    p.add_argument("--inv_impl", default="conv", choices=["conv", "dense"],
                   help="invariant block: nn.Conv or unfolded GEMM twin (train.py flag)")
    p.add_argument("--microbench", action="store_true",
                   help="also time one forward call at the E_loc batch size + one "
                        "get_conn_padded call and take an HLO census (explains 'grad')")
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
        compute_dtype=args.compute_dtype, inv_impl=args.inv_impl,
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

    snapshot = (jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), vs.parameters),
               jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), vs.sampler_state))

    micro = _microbench(vs, Ham, cfg, geo) if args.microbench else None

    solvers = args.qgt_solvers if args.qgt_solvers else [args.qgt_solver]
    variants = {}
    n_params_final = None
    for solver in solvers:
        label = solver if solver else "cg"     # None -> NetKet's true default
        print(f"[bench] === variant qgt_solver={label} ===", flush=True)
        t0 = time.time()
        timing_log, med, vs_final = _run_one(vs, Ham, snapshot, args, solver)
        wall = time.time() - t0
        n_params_final = int(vs_final.n_parameters)
        variants[label] = {"timing_per_step": timing_log, "median_excl_compile": med,
                           "wall_total_s": wall}
        print(f"[bench] {label} median (excl. compile step): {med}", flush=True)

    mem_stats = _gpu_memory_stats()
    result = {
        "config": {**cfg, "qgt": args.qgt, "n_iter": args.n_iter,
                   "init_from": args.init_from, "warm_started": warm},
        "n_params": n_params_final,
        "build_time_s": t_build,
        "variants": variants,
        "gpu_memory_stats": mem_stats,
        "microbench": micro,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    if mem_stats and mem_stats.get("peak_bytes_in_use") is not None:
        print(f"[bench] GPU peak_bytes_in_use: "
              f"{mem_stats['peak_bytes_in_use'] / 2**30:.2f} GiB "
              f"(bytes_limit: {mem_stats.get('bytes_limit', 0) / 2**30:.1f} GiB)",
              flush=True)
    print(f"[bench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
