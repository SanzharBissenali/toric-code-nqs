"""
Three_TC/validate_hy_l2.py
─────────────────────────────────────────────────────────────────────────────
L=2 sign-full (h_y) validation campaign: for each (h_x, h_y, h_z) point, train the
complex NQS, exact-diagonalise, and score BOTH

    • ΔE/|E|  = |E_var^NQS − E_0^ED| / |E_0^ED|      (energy — forgiving of a wrong sign)
    • F       = ‖P_manifold |ψ_NQS⟩‖²                (fidelity — the real sign-sector test)

E_var^NQS here is the EXACT full-sum variational energy ⟨ψ_NQS|H|ψ_NQS⟩ over the 2^N
basis (not the MC estimate), so the energy score carries no MC noise. The MC energy from
training is used only for a basis-alignment cross-check (they must agree within MC error;
a mismatch means the ED and NQS qubit orderings/conventions diverged — verified equal at
L=2, but re-checked at runtime).

RUN ON A NODE (2^N = 2^24 exact diag needs ~6–8 GB RAM + a few GB for k eigenvectors).
NEVER on the 8 GB dev box — see CLAUDE.md. The NQS training wants a GPU; the ED is scipy
(CPU) and can run alongside.

Workflow:
    # 1) pick diag_shift (4 fast runs at pure h_y=0.5, best by fidelity):
    python -m Three_TC.validate_hy_l2 --mini-sweep
    # 2) run the Balanced-7 grid at the chosen shift:
    python -m Three_TC.validate_hy_l2 --diag-shift 5e-3 --out validate_hy_l2.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from Three_TC.builders import build_state, run_loop
from Three_TC.tests.colab_exact_diag import (
    ThreeD_ToricCodeGeometry_PBC, make_hamiltonian_sparse)
from Three_TC.fidelity import (
    nqs_amplitudes, degenerate_manifold, subspace_fidelity)


# --- Balanced-7 grid (L=2 PBC, bosonic). Continuity anchor + control (h_y→0), pure-h_y,
#     mixed x+y, full x+y+z. h_y=0 uses the real path; every h_y!=0 point is complex. -----
GRID: List[Dict[str, float]] = [
    {"tag": "anchor_real",   "hx": 0.1, "hy": 0.00, "hz": 0.1},   # 1 continuity anchor (real)
    {"tag": "cont_hy0.01",   "hx": 0.1, "hy": 0.01, "hz": 0.1},   # 2 continuity control
    {"tag": "cont_hy0.02",   "hx": 0.1, "hy": 0.02, "hz": 0.1},   # 3 continuity control
    {"tag": "pure_hy0.5",    "hx": 0.0, "hy": 0.50, "hz": 0.0},   # 4 pure h_y, intermediate
    {"tag": "pure_hy1.0",    "hx": 0.0, "hy": 1.00, "hz": 0.0},   # 5 pure h_y, self-dual (hard)
    {"tag": "mixed_xy",      "hx": 0.3, "hy": 0.30, "hz": 0.0},   # 6 mixed x+y (non-rotatable)
    {"tag": "full_xyz",      "hx": 0.2, "hy": 0.20, "hz": 0.2},   # 7 full x+y+z (target regime)
]

DIAG_SHIFT_SWEEP = [2e-4, 1e-3, 5e-3, 1e-2]
SWEEP_POINT = {"tag": "pure_hy0.5", "hx": 0.0, "hy": 0.5, "hz": 0.0}

TRAIN = dict(L=2, bc="PBC", arch="ToricCNN_full",
             n_iter=500, dt=0.02, qgt="dense",
             n_samples=16384, n_chains=16, n_discard=8, seed=0)
ED_K = 12                # eigenpairs: spans the ≤8-fold topological manifold + gap
ED_TOL = 1e-8            # eigsh residual tol; fidelity needs ~1e-6, so this is ample and
                        #   converges far faster than the tol=0 (machine-eps) default
ED_NCV = 60             # Krylov/Lanczos basis size — enlarged for the near-degenerate
                        #   ground cluster (clustered eigenvalues starve the default ncv~25)
FID_CHUNK = 1 << 18      # logψ eval chunk over the 2^N basis


def _cfg(point: Dict[str, float], diag_shift: float) -> Dict[str, Any]:
    cfg = {**TRAIN, **{k: point[k] for k in ("hx", "hy", "hz")}}
    cfg["diag_shift"] = diag_shift
    return cfg


def _exact_diag(point: Dict[str, float], L: int, k: int = ED_K
                ) -> Tuple[np.ndarray, np.ndarray, Any, np.ndarray]:
    """(evals, evecs, H_ed, basis) via Lanczos on a STORED sparse CSR matrix.

    The sparse matvec is C-optimized and cache-friendly — orders of magnitude faster
    inside eigsh than the Python matrix-free operator (whose per-σ^y random gather over
    the 2^N vector is what stalls the h_y≠0 runs). Needs a big-RAM node (~15–35 GB build
    at L=2), which is why the driver is node-only. tol/ncv are tuned for the clustered
    ground manifold; without them Lanczos thrashes on the ≤8 near-degenerate states."""
    geo = ThreeD_ToricCodeGeometry_PBC(L, L, L)
    t0 = time.time()
    H, basis = make_hamiltonian_sparse(geo, hx=point["hx"], hy=point["hy"],
                                       hz=point["hz"], J=1.0)
    print(f"  [ED] sparse build {H.shape[0]}x{H.shape[0]}, {H.nnz:.3g} nnz, "
          f"{H.data.nbytes/1e9:.1f} GB, dtype={H.dtype}  ({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    evals, evecs = spla.eigsh(H, k=k, which="SA", tol=ED_TOL,
                              ncv=min(H.shape[0] - 1, ED_NCV))
    print(f"  [ED] eigsh k={k} done ({time.time()-t0:.1f}s)  E0={np.min(evals.real):.6f}",
          flush=True)
    order = np.argsort(np.real(evals))
    return evals[order], evecs[:, order], H, basis


def score_point(point: Dict[str, float], diag_shift: float, *,
                ed_cache: Optional[Tuple] = None, verbose: bool = True) -> Dict[str, Any]:
    """Train + ED + score one grid point. `ed_cache` reuses a prior (evals,evecs,H,basis)
    for the same point (used by the diag_shift mini-sweep)."""
    cfg = _cfg(point, diag_shift)
    t0 = time.time()

    # --- train the (complex when h_y!=0) NQS ---------------------------------
    # coarse train-vs-ED timing (flush) so a wall-clock timeout is diagnosable —
    # run_loop itself is silent here (on_step=None) to avoid the per-step expect cost.
    print(f"  [{point['tag']}] arch={cfg['arch']} training {cfg['n_iter']} iters "
          f"(n_samples={cfg['n_samples']}, n_chains={cfg['n_chains']}) ...", flush=True)
    geo, hi, Ham, vs, _ = build_state(cfg)
    run_loop(vs, Ham, n_iter=cfg["n_iter"], dt=cfg["dt"],
             diag_shift=cfg["diag_shift"], qgt=cfg["qgt"], on_step=None)
    E_mc = complex(vs.expect(Ham).mean)                 # MC estimate (for cross-check only)
    t_train = time.time() - t0
    print(f"  [{point['tag']}] training done in {t_train:.0f}s; starting 2^{3*cfg['L']**3} "
          f"exact diag ...", flush=True)

    # --- exact diag (cached per point) ---------------------------------------
    if ed_cache is None:
        evals, evecs, H_ed, basis = _exact_diag(point, cfg["L"])
    else:
        evals, evecs, H_ed, basis = ed_cache
    E0 = float(np.real(evals[0]))

    # --- NQS state vector in the ED basis, then exact variational energy + fidelity ---
    psi = nqs_amplitudes(vs, basis, geo.N, chunk=FID_CHUNK)   # aligned to ED basis
    E_var = complex(np.vdot(psi, H_ed @ psi))                 # exact ⟨ψ|H|ψ⟩ (no MC noise)
    manifold = degenerate_manifold(evals)
    F = subspace_fidelity(psi, evecs[:, manifold])
    F_gs = subspace_fidelity(psi, evecs[:, :1])

    dE = abs(np.real(E_var) - E0) / abs(E0)
    # basis-alignment guard: exact NQS energy must match the training MC energy within noise
    xcheck = abs(np.real(E_var) - np.real(E_mc)) / (abs(E0) + 1e-12)

    rec = {
        "tag": point["tag"], "hx": point["hx"], "hy": point["hy"], "hz": point["hz"],
        "dtype": "complex" if point["hy"] != 0.0 else "float64",
        "diag_shift": diag_shift,
        "E0_ed": E0, "gap": float(np.real(evals[1] - evals[0])),
        "E_var_nqs": float(np.real(E_var)), "E_var_im": float(np.imag(E_var)),
        "E_mc_nqs": float(np.real(E_mc)),
        "dE_rel": dE,
        "fidelity": F, "single_vec_fidelity": F_gs, "manifold_dim": int(len(manifold)),
        "xcheck_rel": float(xcheck),   # should be ≲ MC error; large ⇒ basis/convention bug
        "t_train_s": t_train,
    }
    if verbose:
        warn = "  <-- XCHECK FAIL (basis/convention?)" if xcheck > 5e-2 else ""
        print(f"[{point['tag']:14s}] hy={point['hy']:.2f} ds={diag_shift:.0e}  "
              f"dE={dE:.2e}  F={F:.4f} (gs {F_gs:.4f}, dim {len(manifold)})  "
              f"Im(E)={rec['E_var_im']:+.1e}  xchk={xcheck:.1e}{warn}", flush=True)
    return rec


def mini_sweep(shifts=DIAG_SHIFT_SWEEP) -> Tuple[float, List[Dict[str, Any]]]:
    """Score `SWEEP_POINT` at each diag_shift (ED computed once, reused) → best shift by
    fidelity (tie-break lower ΔE)."""
    print(f"=== diag_shift mini-sweep at {SWEEP_POINT['tag']} "
          f"(hy={SWEEP_POINT['hy']}) over {shifts} ===", flush=True)
    evals, evecs, H_ed, basis = _exact_diag(SWEEP_POINT, TRAIN["L"])
    cache = (evals, evecs, H_ed, basis)
    rows = [score_point(SWEEP_POINT, ds, ed_cache=cache) for ds in shifts]
    best = max(rows, key=lambda r: (round(r["fidelity"], 4), -r["dE_rel"]))
    print(f"--> best diag_shift = {best['diag_shift']:.0e}  "
          f"(F={best['fidelity']:.4f}, dE={best['dE_rel']:.2e})", flush=True)
    return best["diag_shift"], rows


def run_grid(diag_shift: float, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    pts = GRID if not tags else [p for p in GRID if p["tag"] in set(tags)]
    if tags and len(pts) != len(set(tags)):
        raise SystemExit(f"unknown tag(s) in {tags}; valid: {[p['tag'] for p in GRID]}")
    print(f"=== grid @ diag_shift={diag_shift:.0e}  ({len(pts)} pts: "
          f"{[p['tag'] for p in pts]}) ===", flush=True)
    rows = [score_point(p, diag_shift) for p in pts]
    # continuity check: complex tiny-h_y points should track the real anchor
    anchor = next((r for r in rows if r["tag"] == "anchor_real"), None)
    if anchor is not None:
        for r in rows:
            if r["tag"].startswith("cont_"):
                r["dF_vs_anchor"] = r["fidelity"] - anchor["fidelity"]
                r["dEvar_vs_anchor"] = r["E_var_nqs"] - anchor["E_var_nqs"]
    return rows


def _mem_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mini-sweep", action="store_true",
                    help="sweep diag_shift at pure h_y=0.5, then STOP (pick the shift, re-run --diag-shift)")
    ap.add_argument("--diag-shift", type=float, default=None,
                    help="run the Balanced-7 grid at this diag_shift")
    ap.add_argument("--n-iter", type=int, default=None, help="override training iterations")
    ap.add_argument("--arch", type=str, default=None,
                    choices=["ToricCNN", "ToricCNN_full", "ToricCNN_gridinv"],
                    help="override the ansatz (default ToricCNN_full); use ToricCNN_gridinv "
                         "to validate the large-L workhorse in the sign-full regime")
    ap.add_argument("--tags", type=str, default=None,
                    help="comma-separated grid tags to run (split the grid across nodes); "
                         "default = all 7. Tags: " + ",".join(p["tag"] for p in GRID))
    ap.add_argument("--out", type=str, default="validate_hy_l2.json")
    ap.add_argument("--force", action="store_true", help="run even if RAM looks too small")
    args = ap.parse_args()

    ram = _mem_gb()
    print(f"host RAM ≈ {ram:.0f} GB;  2^{3*TRAIN['L']**3} ED needs ~6-8 GB + eigenvectors.")
    if np.isfinite(ram) and ram < 12 and not args.force:
        raise SystemExit("Refusing to run: <12 GB RAM (L=2 3D ED will OOM). Use a node, or --force.")
    if args.n_iter:
        TRAIN["n_iter"] = args.n_iter
    if args.arch:
        TRAIN["arch"] = args.arch
    print(f"arch = {TRAIN['arch']}")

    out: Dict[str, Any] = {"train": TRAIN, "grid": GRID}
    if args.mini_sweep:
        best, rows = mini_sweep()
        out["mini_sweep"] = rows
        out["best_diag_shift"] = best
    elif args.diag_shift is not None:
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        out["diag_shift"] = args.diag_shift
        out["rows"] = run_grid(args.diag_shift, tags=tags)
    else:
        raise SystemExit("Pass --mini-sweep (choose diag_shift) or --diag-shift <v> (run grid).")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
