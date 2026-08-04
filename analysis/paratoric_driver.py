"""Local ParaToric driver: validation ladder + single-point production runs.

Runs the same chain/block scheme as colab/qmc_benchmarks_colab.ipynb, but as a
plain module so multiprocessing works under macOS spawn. Requires the pybind
extension built by external/build_paratoric_local.sh.

  .venv/bin/python analysis/paratoric_driver.py --validate            # ladder (SMOKE-sized)
  .venv/bin/python analysis/paratoric_driver.py --L 4 --hx 0.2 --hz 0.1 --out results/...json
"""

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "external", "ParaToric", "python"),
          os.path.join(_ROOT, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

# Sampler hygiene scaled from the ParaToric paper's own practice (~110 update
# steps per edge between samples; here ~120/edge) — with 16x less decorrelation
# the ladder showed a uniform NEGATIVE bias (impossible for unbiased <H> >= E0)
# and tau_int warnings. Thermalization additionally scales with beta (kink
# density, hence relaxation time, grows with beta).
N_BETWEEN = {2: 2_000, 4: 16_000, 5: 36_000, 6: 64_000, 7: 104_000}
N_THERM = {2: 30_000, 4: 250_000, 5: 500_000, 6: 800_000, 7: 1_300_000}
OBS = ["energy", "star_x", "plaquette_z", "sigma_x", "sigma_z"]
N_RESAMPLES = 1_000


def run_chain(job):
    """One Markov chain at one (L, hx, hz, beta). Returns per-observable stats."""
    import paratoric
    t0 = time.time()
    series, mean, std, _, _, tau = paratoric.extended_toric_code.get_sample(
        N_samples=job["ns"], N_thermalization=job["nth"], N_between_samples=job["nbs"],
        beta=job["beta"], mu=1.0, h=job["hx"], h_therm=job["hx"], J=1.0,
        lmbda=job["hz"], lmbda_therm=job["hz"], N_resamples=N_RESAMPLES,
        custom_therm=False, observables=OBS, seed=job["seed"], basis=job["basis"],
        lattice_type="cubic", system_size=job["L"], boundaries="open", default_spin=1)
    out = {o: (float(mean[k]), float(std[k]), float(tau[k])) for k, o in enumerate(OBS)}
    out["runtime_s"] = time.time() - t0
    return out


def run_point(L, hx, hz, beta, n_chains, ns, n_blocks=4, basis="x", seed0=1000,
              workers=None, quiet=False, nbs_mult=1.0):
    """n_chains x n_blocks independent blocks -> inverse-variance-weighted E.

    nbs_mult: scale decorrelation steps up near criticality (chi2_red >> 1 is the tell).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # decorrelation must scale with beta (kink density ~ beta), not just with nbs_mult
    scale = nbs_mult * max(1.0, beta / 12.0)
    nth = int(N_THERM[L] * scale)
    nbs = int(N_BETWEEN[L] * scale)
    jobs = [dict(L=L, hx=hx, hz=hz, beta=beta, ns=max(1000, ns // n_blocks),
                 nbs=nbs, nth=nth, seed=seed0 + 7 * i, basis=basis)
            for i in range(n_chains * n_blocks)]
    chains, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        futs = [ex.submit(run_chain, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            chains.append(f.result())
            e = np.array([c["energy"][0] for c in chains])
            if not quiet:
                se = np.std(e, ddof=1) / np.sqrt(len(e)) if len(e) > 1 else float("nan")
                print(f"  [L={L} {k}/{len(jobs)} blocks, {(time.time()-t0)/60:5.1f} min] "
                      f"running E = {np.mean(e):.6f} +- {se:.6f}", flush=True)
    E = np.array([c["energy"][0] for c in chains])
    sE = np.array([c["energy"][1] for c in chains])
    # blocks are exchangeable (same L/beta/ns/nbs): equal-weight mean. Inverse-variance
    # weights estimated from the blocks themselves correlate with the block means and
    # carry a ~1e-4..1e-3 bias floor — exactly at the precision targets we care about.
    Ebar = float(np.mean(E))
    err = float(np.std(E, ddof=1) / np.sqrt(len(E)))
    chi2r = float(np.sum((E - Ebar) ** 2 / sE**2) / max(1, len(E) - 1))  # diagnostic only
    return dict(engine="paratoric", L=L, hx=hx, hz=hz, beta=beta, E=Ebar, E_err=err,
                chi2_red=chi2r, n_chains=n_chains, n_blocks=n_blocks,
                ns_per_block=jobs[0]["ns"], nbs=nbs, nth=nth, nbs_mult=nbs_mult,
                seed0=seed0, basis=basis, chains=chains)


def validate(beta, ns, n_chains):
    """Exact-target ladder: h=0 anchors, L=2 vs ED, pure-hz vs series4, beta-doubling."""
    from exact_benchmarks import counts, E_lowfield
    from scipy.sparse.linalg import eigsh
    from tc3d.exact_diag import hamiltonian_linop
    from tc3d.geometry import ThreeD_ToricCodeGeometry

    hx, hz = 0.2, 0.1
    H, _ = hamiltonian_linop(ThreeD_ToricCodeGeometry(2, 2, 2, bc="OBC"), hx=hx, hz=hz)
    e2_ed = float(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])

    rows = []
    for L in (4, 5):                                        # anchors == geometry check
        r = run_point(L, 0.0, 0.0, beta, n_chains, ns, quiet=True)
        rows.append((f"anchor L={L}", r["E"], r["E_err"], counts(L, "OBC").E0))
    r = run_point(2, hx, hz, beta, 2 * n_chains, ns, quiet=True)
    rows.append(("L=2 vs ED", r["E"], r["E_err"], e2_ed))
    r = run_point(4, 0.0, 0.1, beta, 2 * n_chains, ns, basis="z", quiet=True)
    rows.append(("L=4 hz=0.1 vs series4", r["E"], r["E_err"],
                 E_lowfield(4, "OBC", hz=0.1, order=4)))
    r1 = run_point(4, hx, hz, beta, n_chains, ns, quiet=True)
    r2 = run_point(4, hx, hz, 2 * beta, n_chains, ns, quiet=True)
    rows.append(("L=4 beta-doubling", r1["E"] - r2["E"],
                 float(np.hypot(r1["E_err"], r2["E_err"])), 0.0))

    print(f"\n{'check':24s} {'E_QMC':>14s} {'err':>9s} {'target':>14s} {'z':>6s}")
    ok = True
    for name, E, err, tgt in rows:
        z = (E - tgt) / err
        ok &= abs(z) < 3
        print(f"{name:24s} {E:14.6f} {err:9.6f} {tgt:14.6f} {z:6.2f} "
              f"{'PASS' if abs(z) < 3 else 'FAIL'}")
    print("\nladder", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true", help="run the exact-target ladder")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--hx", type=float, default=0.2)
    ap.add_argument("--hz", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=12.0)
    ap.add_argument("--chains", type=int, default=max(2, (os.cpu_count() or 2) // 2))
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--samples", type=int, default=20_000, help="stored samples per chain")
    ap.add_argument("--basis", default="x")
    ap.add_argument("--nbs_mult", type=float, default=1.0,
                    help="scale decorrelation+thermalization (use >1 near criticality)")
    ap.add_argument("--seed0", type=int, default=1000,
                    help="base RNG seed; use a fresh value per run (never reuse across runs)")
    ap.add_argument("--out", default=None, help="write result JSON here")
    args = ap.parse_args()

    if args.validate:
        ok = validate(args.beta, max(200, args.samples // 10), args.chains)
        sys.exit(0 if ok else 1)

    r = run_point(args.L, args.hx, args.hz, args.beta, args.chains, args.samples,
                  n_blocks=args.blocks, basis=args.basis, nbs_mult=args.nbs_mult,
                  seed0=args.seed0)
    print(f"\nL={args.L} ({args.hx},{args.hz}) beta={args.beta}: "
          f"E = {r['E']:.6f} +- {r['E_err']:.6f}  chi2_red={r['chi2_red']:.2f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=1)
        print("wrote", args.out)
