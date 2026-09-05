"""analysis/scripts/bench_sign_heads.py

Benchmark/diagnostic tool for the fermionic per-configuration sign heads
(`tc3d.sign_decoders`: cup | linear | vote | pt2) plus the host analytic head
(`tc3d.sign_frame.anaC_sign`). Two independent subcommands:

  micro  -- throughput (us/row) on synthetic configurations, for each (L, head,
            config-type) cell: uniformly random +-1 rows, and "all-up plus k
            random single flips" rows for k in --ks. Safe/fast on the dev
            machine at small L; L up to 6 is allowed to construct (geometry
            only -- CLAUDE.md's "never run 3D TC ED/sweeps locally" is about
            training/ED, not building a geometry + O(N) GF(2) decoder object).

  stats  -- structural statistics (lit-class k histogram, first-order-tie
            fraction, second-order fraction, cap-fallback fraction) on
            REALISTIC configurations drawn from an actual trained checkpoint
            (or, for a local smoke test, a freshly initialized random network
            via --random_state). Intended to run on the cluster against real
            checkpoints; never trains.

Only public, documented `tc3d.sign_decoders` surface is load-bearing here:
`make_decoder_sign(kind, geo, stabs=None, k_cap=8, J=1.0) -> sign_fn`, with
`sign_fn(configs)` and `sign_fn.pop_stats()`. A couple of diagnostics ask for
quantities that module does NOT expose publicly (a full k-histogram instead of
pop_stats' k_sum mean; the per-row recovery-enumeration size). Those are
attempted via the decoder's internal `.support` structure on a best-effort
basis (guarded by try/except AttributeError) and degrade to a documented
fallback or an explicit "not accessible" note if that internal shape changes
-- `tc3d/sign_decoders.py` is being actively edited by another agent while
this script is written, so nothing here assumes its private layout is stable.

Thread control
--------------
--threads sets NUMBA_NUM_THREADS / OMP_NUM_THREADS / OPENBLAS_NUM_THREADS /
MKL_NUM_THREADS BEFORE numpy (and therefore its BLAS backend) is imported --
these env vars are read once at library init, so setting them after `import
numpy` is a no-op. Default 1 (single-threaded, reproducible us/row). The sign
heads here are pure numpy/GF(2) linear algebra with no numba in the hot path
today, but if a future accelerated head adds one, its numba thread pool and
numpy's BLAS thread pool both claim cores independently -- pin them to the
SAME value (this flag does both) or the two pools oversubscribe the machine
together and every timing in this file stops meaning anything.

Usage
-----
    cd toric-code-nqs-fsign
    PYTHONPATH=. .venv/bin/python analysis/scripts/bench_sign_heads.py micro \\
        --L 2 3 4 --heads cup linear vote pt2 anaC
    PYTHONPATH=. .venv/bin/python analysis/scripts/bench_sign_heads.py stats \\
        --random_state --L 2
    PYTHONPATH=. .venv/bin/python analysis/scripts/bench_sign_heads.py stats \\
        --ckpt results/some_run/some_name.mpack --L 4 --n_samples 4096

Writes JSON to --out ONLY (default results/fermionic_speed/bench_sign_heads.json,
created if missing); never touches any other path.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# ---------------------------------------------------------------------------
# Thread env vars MUST be set before numpy (BLAS) is imported -- pull --threads
# out of raw argv with a tiny hand parser so nothing here imports numpy first.
# ---------------------------------------------------------------------------

def _early_threads(argv):
    val = "1"
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith("--threads="):
            val = a.split("=", 1)[1]
    return val


_THREADS = _early_threads(sys.argv[1:])
for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS"):
    os.environ[_v] = _THREADS

import json          # noqa: E402
import math           # noqa: E402
import platform       # noqa: E402
import subprocess     # noqa: E402
import time           # noqa: E402

import numpy as np    # noqa: E402  (import order matters -- see above)

from tc3d.geometry import ThreeD_ToricCodeGeometry as Geo          # noqa: E402
from tc3d.fermionic_decoration import fermionic_plaquettes         # noqa: E402
from tc3d.sign_decoders import make_decoder_sign, KINDS             # noqa: E402
from tc3d.sign_frame import anaC_sign                                # noqa: E402
from tc3d.sign_geometry import CupSign                                # noqa: E402

DEFAULT_OUT = "results/fermionic_speed/bench_sign_heads.json"
D = argparse.SUPPRESS


# =============================================================================
# shared helpers
# =============================================================================

def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _sys_info():
    return {"python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "machine": platform.machine(),
            "numpy": np.__version__}


def _meta(args):
    return {"git_rev": _git_rev(), "sys": _sys_info(), "threads": _THREADS,
            "args": vars(args), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_json(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"[bench_sign_heads] wrote {path}")


def _make_head(kind, geo, stabs, k_cap, J):
    if kind == "anaC":
        return anaC_sign(geo, J=J)
    return make_decoder_sign(kind, geo, stabs=stabs, k_cap=k_cap, J=J)


# =============================================================================
# micro: throughput
# =============================================================================

def _random_configs(rng, n, N):
    return 1.0 - 2.0 * rng.integers(0, 2, size=(n, N)).astype(np.float64)


def _flip_k_configs(rng, n, N, k):
    """All-up (+1) rows, each with k DISTINCT random edges flipped to -1
    (a flip of edge e = set spin e to -1). Vectorized via an argsort trick."""
    x = np.ones((n, N), dtype=np.float64)
    if k > 0:
        idx = np.argsort(rng.random((n, N)), axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        x[rows, idx] = -1.0
    return x


def _chunked_call(sign_fn, configs, chunk):
    t0 = time.perf_counter()
    for a in range(0, configs.shape[0], chunk):
        sign_fn(configs[a:a + chunk])
    return time.perf_counter() - t0


def _run_cell(sign_fn, configs, repeats, chunk, max_seconds, label):
    """One untimed warm-up (primes the decoder's lazy per-coset cache; its
    pop_stats counters are discarded) + up to `repeats` timed chunked passes,
    bounded by `max_seconds` total wall-clock for the whole cell.

    Returns (status, warmup_seconds, [repeat_seconds...], pop_stats_or_None).
    status: "ok" (all repeats ran), "skipped" (warm-up alone blew the budget,
    zero repeats run), "skipped_partial" (budget hit mid-way, >=0 repeats kept).
    """
    has_pop = hasattr(sign_fn, "pop_stats")
    t_cell0 = time.perf_counter()

    tw0 = time.perf_counter()
    sign_fn(configs)
    warmup_s = time.perf_counter() - tw0
    if has_pop:
        sign_fn.pop_stats()
    print(f"    [{label}] warmup {warmup_s:.3f}s", flush=True)
    if warmup_s > max_seconds:
        return "skipped", warmup_s, [], None

    repeat_times = []
    status = "ok"
    for r in range(repeats):
        if time.perf_counter() - t_cell0 > max_seconds:
            status = "skipped_partial"
            break
        dt = _chunked_call(sign_fn, configs, chunk)
        repeat_times.append(dt)
        us_row = dt / configs.shape[0] * 1e6
        print(f"    [{label}] repeat {r + 1}/{repeats}: {dt:.3f}s ({us_row:.3f} us/row)",
              flush=True)
    stats = sign_fn.pop_stats() if has_pop else None
    return status, warmup_s, repeat_times, stats


def _cell_summary(configs, repeat_times, stats):
    n = configs.shape[0]
    us_per_row = [t / n * 1e6 for t in repeat_times]
    out = {
        "n_configs": int(n),
        "repeat_seconds": repeat_times,
        "us_per_row_min": min(us_per_row) if us_per_row else None,
        "us_per_row_median": float(np.median(us_per_row)) if us_per_row else None,
    }
    if stats is not None:
        nr = max(int(stats.get("n_rows", 0)), 1)
        out["pop_stats"] = {
            "n_rows": int(stats.get("n_rows", 0)),
            "k_mean": stats.get("k_sum", 0) / nr,
            "k_max": int(stats.get("k_max", 0)),
            "fallback_frac": stats.get("n_fallback", 0) / nr,
            "tie_frac": stats.get("n_tie", 0) / nr,
            "pt2_frac": stats.get("n_pt2", 0) / nr,
        }
    else:
        out["pop_stats"] = None
    return out


def _print_table(rows):
    hdr = f"\n{'L':>3} {'N':>5} {'head':<8} {'cfg':<12} {'us/row':>10} {'k_mean':>8} {'fallback%':>9}"
    print(hdr)
    print("-" * len(hdr.strip()))
    for L, N, head, cfg, us, stats in rows:
        us_s = f"{us:.3f}" if us is not None else "n/a"
        k_s = f"{stats['k_mean']:.2f}" if stats else "n/a"
        fb_s = f"{stats['fallback_frac'] * 100:.1f}" if stats else "n/a"
        print(f"{L:>3} {N:>5} {head:<8} {cfg:<12} {us_s:>10} {k_s:>8} {fb_s:>9}")


def cmd_micro(args):
    cells, table_rows = [], []
    for L in args.L:
        geo = Geo(L, L, L, bc="OBC")
        stabs = fermionic_plaquettes(geo)
        N, NP = geo.N, len(stabs)
        print(f"\n[micro] L={L} N={N} NP={NP}", flush=True)

        rng_random = np.random.default_rng(args.seed + 1000 * L)
        random_configs = _random_configs(rng_random, args.n, N)
        flip_configs = {}
        for k in args.ks:
            rng_k = np.random.default_rng(args.seed + 2000 * L + k)
            flip_configs[k] = _flip_k_configs(rng_k, 2000, N, k)

        for head in args.heads:
            t0 = time.perf_counter()
            err, sign_fn = None, None
            try:
                sign_fn = _make_head(head, geo, stabs, args.k_cap, args.J)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            t_constr = time.perf_counter() - t0
            base = {"L": L, "N": N, "NP": NP, "head": head,
                    "construction_seconds": t_constr, "construction_error": err}

            if sign_fn is None:
                print(f"  [{head}] L={L}: construction FAILED ({err})")
                cells.append({**base, "config_type": None, "status": "construction_error"})
                continue
            print(f"  [{head}] L={L}: constructed in {t_constr:.4f}s")

            plan = [("random", random_configs)] + [(f"flip_k{k}", flip_configs[k])
                                                    for k in args.ks]
            for cfg_type, configs in plan:
                label = f"L{L}/{head}/{cfg_type}"
                status, warm, reps, stats = _run_cell(
                    sign_fn, configs, args.repeats, args.chunk,
                    args.max_seconds_per_cell, label)
                summary = _cell_summary(configs, reps, stats)
                cells.append({**base, "config_type": cfg_type, "status": status,
                              "warmup_seconds": warm, **summary})
                table_rows.append((L, N, head, cfg_type, summary["us_per_row_median"],
                                   summary["pop_stats"]))

    _print_table(table_rows)
    out = {"meta": _meta(args), "cells": cells}
    _write_json(out, args.out)
    return out


def add_micro_args(p):
    p.add_argument("--L", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--heads", type=str, nargs="+", choices=list(KINDS) + ["anaC"],
                   default=list(KINDS) + ["anaC"])
    p.add_argument("--n", type=int, default=20000, help="random-config cell size")
    p.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                   help="flip counts for the all-up+k-flips cells (2000 configs each)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--chunk", type=int, default=4096,
                   help="rows per sign_fn() call within a timed pass (mimics the "
                        "production get_conn_padded call shape)")
    p.add_argument("--k_cap", type=int, default=8)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_seconds_per_cell", type=float, default=60.0,
                   help="wall-clock budget per (L, head, config-type) cell; a cell "
                        "already over budget after warm-up records status="
                        "'skipped' with zero repeats instead of hanging")
    p.add_argument("--out", default=DEFAULT_OUT)


# =============================================================================
# stats: structural diagnostics on realistic samples
# =============================================================================

_CLI_OVERRIDE_KEYS = (
    "bc", "model", "arch", "kernel_size", "hidden", "vanilla_depth",
    "noninv_channels", "n_noninv", "noninv_hidden", "inv_hidden", "cnn_hidden",
    "dual_basis", "sign_frame", "sign_table", "sign_k_cap", "dtype",
    "flux_penalty", "phase_head", "phase_head_frozen",
    "n_chains", "n_discard", "chunk_size", "seed", "hx", "hy", "hz", "J",
)


def _find_sidecar_json(ckpt_path):
    """ckpt_path (.../name.mpack or .../name.stepNNN.mpack) -> that run's
    train.py metadata JSON (.../name.json), stripping any snapshot suffix --
    same convention as analysis/scripts/eval_snapshots.py's SNAPSHOT_RE. Returns
    (json_path_or_None, weights_base_for_load_weights)."""
    base = ckpt_path[:-len(".mpack")] if ckpt_path.endswith(".mpack") else ckpt_path
    m = re.match(r"^(.*)\.step\d+$", base)
    root = m.group(1) if m else base
    json_path = root + ".json"
    return (json_path if os.path.exists(json_path) else None), base


def _build_cfg(args):
    """cfg for build_state: sidecar-JSON config as the base when found (the
    EXACT config the checkpoint was trained with), else this script's own
    OBC/fermionic defaults; CLI flags the user actually passed (argparse.
    SUPPRESS keeps omitted ones out of vars(args)) always win on top."""
    ns = vars(args)
    overrides = {k: ns[k] for k in _CLI_OVERRIDE_KEYS if k in ns}

    if args.random_state:
        cfg = {"L": args.L, "bc": "OBC", "model": "fermionic"}
        cfg.update(overrides)
        return cfg, "random_state", None

    json_path, ckpt_base = _find_sidecar_json(args.ckpt)
    if json_path:
        with open(json_path) as f:
            meta = json.load(f)
        cfg = dict(meta["config"])
        if int(cfg.get("L", args.L)) != args.L:
            raise SystemExit(f"--L {args.L} conflicts with sidecar JSON's "
                             f"L={cfg.get('L')} ({json_path})")
        cfg["L"] = args.L
        cfg.update(overrides)
        return cfg, f"sidecar_json:{json_path}", ckpt_base

    cfg = {"L": args.L, "bc": "OBC", "model": "fermionic"}
    cfg.update(overrides)
    return cfg, "cli_defaults (no sidecar json found)", ckpt_base


def _k_histogram(geo, stabs, configs):
    """Per-row lit-class count k (see tc3d/sign_decoders.py's module docstring:
    "A lit line class ..."). Prefers the decoder's internal labelling
    (`DecoderSign.support.label`) when exposed -- the exact quantity
    pop_stats' k_sum/k_max are a mean/max OF, but as a full histogram here --
    and falls back to `CupSign.syndrome(configs)` nonzero-count (a coarser,
    stable-API-only "how far off support" proxy: the residual line-parity
    defect after gauge-fixing, NOT the same detector set as the lit-class
    labelling -- see sign_geometry.py's CupSign docstring) if that internal
    attribute is not there."""
    if configs.shape[0] == 0:
        return {"source": None, "counts": {}, "mean": None, "max": None, "n": 0}
    try:
        dec = make_decoder_sign("linear", geo, stabs=stabs)
        b = (configs < 0).astype(np.float64)
        _, coef = dec.support.label(b)
        k = coef.sum(axis=1).astype(np.int64)
        source = "decoder_support_label (exact lit-class count k)"
    except AttributeError as e:
        cs = CupSign(geo, stabs)
        k = (cs.syndrome(configs) != 0).sum(axis=-1).astype(np.int64)
        source = ("cupsign_syndrome_nonzero_count (PROXY: off-support residual "
                  f"count, NOT the lit-class count -- support.label unavailable: {e})")
    counts = np.bincount(k)
    return {"source": source,
            "counts": {str(i): int(c) for i, c in enumerate(counts) if c},
            "mean": float(k.mean()), "max": int(k.max()), "n": int(k.size)}


def _enum_size(geo, stabs, configs, k_cap):
    """Per-row first-order recovery enumeration size Pi_i |class_i| over a
    row's lit classes (the combinatorial fan-out `vote`/`pt2` face before the
    k_cap fallback bails to `linear`) -- an internal `DecoderSign`/`_Support`
    quantity, not part of the declared stable API. Best-effort; explicitly
    skipped (with a note) if those hooks are not there."""
    if configs.shape[0] == 0:
        return {"available": False, "note": "no rows"}
    try:
        dec = make_decoder_sign("vote", geo, stabs=stabs, k_cap=k_cap)
        b = (configs < 0).astype(np.float64)
        u, coef = dec.support.label(b)
        sizes = np.empty(configs.shape[0], dtype=np.float64)
        for uv in np.unique(u):
            idx = np.nonzero(u == uv)[0]
            cls = [dec.support.classes[dec.support.lit[i]]
                  for i in range(len(dec.support.lit)) if coef[idx[0], i]]
            sizes[idx] = float(math.prod(len(c) for c in cls)) if cls else 1.0
        return {"available": True, "mean": float(sizes.mean()),
                "median": float(np.median(sizes)), "max": float(sizes.max())}
    except AttributeError as e:
        return {"available": False,
                "note": f"class-size internals not accessible ({e}); the stable "
                        "tc3d.sign_decoders API is sign_fn/pop_stats only -- skipped"}


def cmd_stats(args):
    from tc3d.builders import build_state
    from tc3d.io import load_weights

    if not args.random_state and not args.ckpt:
        raise SystemExit("stats needs --ckpt PATH.mpack or --random_state")

    cfg, source, ckpt_base = _build_cfg(args)
    cfg["n_samples"] = args.n_samples          # this diagnostic's own draw size
    print(f"[stats] config source: {source}")
    print(f"[stats] cfg (minus 'name'): "
          f"{ {k: v for k, v in cfg.items() if k != 'name'} }")

    geo, hi, Ham, vs, xz_stabs = build_state(cfg, build_ham=True)
    if args.random_state:
        print("[stats] --random_state: freshly initialized network, no "
              "checkpoint loaded, no training run")
    else:
        vs = load_weights(vs, ckpt_base)
        print(f"[stats] loaded weights from {ckpt_base}.mpack")

    vs.sample()
    samples = np.asarray(vs.samples).reshape(-1, geo.N).astype(np.float64)
    print(f"[stats] drew {samples.shape[0]} samples (requested {args.n_samples})")

    base_op = getattr(Ham, "base", Ham)          # bypass any sign-framing: we
    xp, mels = base_op.get_conn_padded(samples)   # want the RAW connected set
    xp, mels = np.asarray(xp), np.asarray(mels)
    conn_mask = mels != 0
    conn_configs = xp[conn_mask].astype(np.float64)
    n_padded_total = int(xp.shape[0] * xp.shape[1])
    print(f"[stats] connected rows: {conn_configs.shape[0]} real connections "
          f"(of {n_padded_total} padded slots, max_conn_size={base_op.max_conn_size})")

    stabs = fermionic_plaquettes(geo)
    head_J = getattr(args, "J", 1.0)         # --J is SUPPRESS (a cfg/Hamiltonian
                                              # override); default to 1.0 for our
                                              # own decoder construction when omitted
    result = {
        "meta": _meta(args),
        "config_source": source,
        "config": {k: v for k, v in cfg.items() if k != "name"},
        "geometry": {"L": geo.Lx, "bc": geo.bc, "N": geo.N, "NP": len(stabs)},
        "samples_n": int(samples.shape[0]),
        "connected": {"n_padded_total": n_padded_total,
                      "n_connected_total": int(conn_configs.shape[0]),
                      "max_conn_size": int(base_op.max_conn_size)},
    }

    for name, configs in (("samples", samples), ("connected", conn_configs)):
        result.setdefault("k_histogram", {})[name] = _k_histogram(geo, stabs, configs)
        for kind in ("linear", "vote", "pt2", "cup"):
            head = make_decoder_sign(kind, geo, stabs=stabs, k_cap=args.k_cap, J=head_J)
            head(configs)
            st = head.pop_stats()
            nr = max(int(st["n_rows"]), 1)
            result.setdefault(kind, {})[name] = {
                "n_rows": int(st["n_rows"]),
                "n_fallback": int(st["n_fallback"]), "fallback_frac": st["n_fallback"] / nr,
                "n_tie": int(st["n_tie"]), "tie_frac": st["n_tie"] / nr,
                "n_pt2": int(st["n_pt2"]), "pt2_frac": st["n_pt2"] / nr,
                "k_mean": st["k_sum"] / nr, "k_max": int(st["k_max"]),
            }
        result.setdefault("enumeration_size", {})[name] = _enum_size(
            geo, stabs, configs, args.k_cap)
        print(f"[stats] {name}: n={configs.shape[0]}  "
              f"k_mean={result['k_histogram'][name]['mean']}  "
              f"vote_tie_frac={result['vote'][name]['tie_frac']:.4f}  "
              f"pt2_frac={result['pt2'][name]['pt2_frac']:.4f}")

    _write_json(result, args.out)
    return result


def add_stats_args(p):
    p.add_argument("--ckpt", default=None,
                   help="checkpoint PATH.mpack (base name or with .mpack); a "
                        "sibling <base>.json (train.py's run metadata, snapshot "
                        "suffix stripped) is used to rebuild the EXACT training "
                        "config when present")
    p.add_argument("--random_state", action="store_true",
                   help="skip --ckpt: use a freshly initialized (random) network "
                        "-- no training -- for a quick local sanity check")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n_samples", type=int, default=4096)
    p.add_argument("--k_cap", type=int, default=8)
    p.add_argument("--out", default=DEFAULT_OUT)
    # Model-defining overrides: SUPPRESS so an omitted flag falls through to the
    # sidecar JSON's config (or this command's OBC/fermionic defaults) instead
    # of clobbering it with an argparse default.
    p.add_argument("--hx", type=float, default=D)
    p.add_argument("--hy", type=float, default=D)
    p.add_argument("--hz", type=float, default=D)
    p.add_argument("--J", type=float, default=D)
    p.add_argument("--bc", choices=["PBC", "OBC"], default=D)
    p.add_argument("--model", choices=["bosonic", "fermionic"], default=D)
    p.add_argument("--arch", default=D)
    p.add_argument("--kernel_size", type=int, default=D)
    p.add_argument("--hidden", type=int, default=D)
    p.add_argument("--vanilla_depth", type=int, default=D)
    p.add_argument("--noninv_channels", type=int, default=D)
    p.add_argument("--n_noninv", type=int, default=D)
    p.add_argument("--noninv_hidden", type=str, nargs="*", default=D)
    p.add_argument("--inv_hidden", type=int, nargs="*", default=D)
    p.add_argument("--cnn_hidden", type=int, nargs="*", default=D)
    p.add_argument("--dual_basis", action="store_true", default=D)
    p.add_argument("--sign_frame", default=D)
    p.add_argument("--sign_table", default=D)
    p.add_argument("--sign_k_cap", type=int, default=D)
    p.add_argument("--dtype", choices=["float64", "complex"], default=D)
    p.add_argument("--flux_penalty", type=float, default=D)
    p.add_argument("--phase_head", action="store_true", default=D)
    p.add_argument("--phase_head_frozen", action="store_true", default=D)
    p.add_argument("--n_chains", type=int, default=D)
    p.add_argument("--n_discard", type=int, default=D)
    p.add_argument("--chunk_size", type=int, default=D)
    p.add_argument("--seed", type=int, default=D)


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--threads", type=int, default=1,
                        help="NUMBA_NUM_THREADS/OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/"
                            "MKL_NUM_THREADS -- set BEFORE numpy import from raw "
                            "argv (see module docstring); this flag definition "
                            "exists so --help/JSON show it, the env vars are "
                            "already set by the time argparse runs")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    add_micro_args(sub.add_parser("micro", parents=[common],
                                  help="us/row throughput on synthetic configs"))
    add_stats_args(sub.add_parser("stats", parents=[common],
                                  help="structural stats on realistic NQS samples"))
    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "micro":
        cmd_micro(args)
    elif args.cmd == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
