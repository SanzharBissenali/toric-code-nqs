"""Local ParaToric driver: validation ladder + single-point production runs.

Runs the same chain/block scheme as colab/qmc_benchmarks_colab.ipynb, but as a
plain module so multiprocessing works under macOS spawn. Requires the pybind
extension built by external/build_paratoric_local.sh.

  .venv/bin/python analysis/paratoric_driver.py --validate            # ladder (SMOKE-sized)
  .venv/bin/python analysis/paratoric_driver.py --L 4 --hx 0.2 --hz 0.1 --out results/...json
  .venv/bin/python analysis/paratoric_driver.py --validate_fm         # FM anchor rungs
  .venv/bin/python analysis/paratoric_driver.py --L 4 --hx 0.2 --hz 0.1 \
      --basis z --fm --out results/...json                            # + Z-string O_FM
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
# L>=8: ~120 updates/edge (3L^2(L-1) edges) and ~1500*edges thermalization,
# extrapolated from the audited L<=7 ladder. The L=8 FM noise pilot gates any
# L=10/12 use — do not launch those without it.
N_BETWEEN = {2: 2_000, 4: 16_000, 5: 36_000, 6: 64_000, 7: 104_000,
             8: 160_000, 10: 324_000, 12: 570_000}
N_THERM = {2: 30_000, 4: 250_000, 5: 500_000, 6: 800_000, 7: 1_300_000,
           8: 2_000_000, 10: 4_000_000, 12: 7_000_000}
OBS = ["energy", "star_x", "plaquette_z", "sigma_x", "sigma_z"]
# Fredenhagen-Marcu string ratio <open half-string>/sqrt(<closed loop>). ParaToric
# builds the loops on the cubic lattice ONLY in the z basis (the x-basis branch has
# no cubic case and silently returns 1), and the hard-coded rectangle degenerates
# below L=4: plane z=(L-1)//2, corners x,y in [ (L-1)//4, 3*(L-1)//4 ]. In the z
# basis the string is the ELECTRIC Z-string (matches tc3d.fm --sector electric).
FM_OBS = "fredenhagen_marcu"
# Magnetic ('t Hooft) FM membrane ratios — our patch to ParaToric (see
# external/paratoric_membrane.patch): basis "x" only (sigma^x products are
# diagonal there). TWO families, each its own FSS curve (never mixed in a fit):
#   FM_MEM_OBS    — the GROWING corner-rule cube: vertices [s,e]^3 with the same
#                   corners as the Z-string (s=(L-1)//4, e=3(L-1)//4, R=e-s),
#                   L >= 5. Matches fm.paratoric_membrane_kwargs(geo) edge-for-edge.
#   FM_MEM_R1_OBS — the fixed-size anchor: centered R=1 cube, L >= 4 — the
#                   cross-method (NQS-vs-QMC) anchor. Matches
#                   fm.paratoric_membrane_kwargs(geo, R=1) edge-for-edge.
FM_MEM_OBS = "fredenhagen_marcu_membrane"
FM_MEM_R1_OBS = "fredenhagen_marcu_membrane_r1"
N_RESAMPLES = 1_000


def run_chain(job):
    """One Markov chain at one (L, hx, hz, beta). Returns per-observable stats."""
    import paratoric
    obs = job.get("obs", OBS)   # must travel in the job: spawn workers re-import OBS
    t0 = time.time()
    series, mean, std, _, _, tau = paratoric.extended_toric_code.get_sample(
        N_samples=job["ns"], N_thermalization=job["nth"], N_between_samples=job["nbs"],
        beta=job["beta"], mu=1.0, h=job["hx"], h_therm=job["hx"], J=1.0,
        lmbda=job["hz"], lmbda_therm=job["hz"], N_resamples=N_RESAMPLES,
        custom_therm=False, observables=obs, seed=job["seed"], basis=job["basis"],
        lattice_type="cubic", system_size=job["L"], boundaries="open", default_spin=1)
    out = {o: (float(mean[k]), float(std[k]), float(tau[k])) for k, o in enumerate(obs)}
    for name, raw_key in ((FM_OBS, "fm_num_den"), (FM_MEM_OBS, "fm_mem_num_den"),
                          (FM_MEM_R1_OBS, "fm_mem_r1_num_den")):
        if name in obs:
            # raw ratio ingredients (series packs them as half + i*full) so the pooled
            # ratio mean(num)/sqrt(|mean(den)|) can be recombined offline; the per-chain
            # (mean, std) above is ParaToric's bias-corrected paired-bootstrap ratio.
            s = np.asarray(series[obs.index(name)])
            out[raw_key] = (float(np.mean(s.real)), float(np.mean(s.imag)))
    out["runtime_s"] = time.time() - t0
    return out


def pooled_fm(chains, raw_key):
    """Pooled FM ratio from the per-chain raw ingredients: mean_c(num̄)/√|mean_c(den̄)|
    with a chain-level jackknife error through the assembled ratio (equal ns per
    chain ⇒ equal weights, so per-chain means suffice).

    This is the trustworthy readout when ⟨closed⟩ is small: the per-chain
    bias-corrected ratio underlying ``combined[obs]["mean"]`` carries an
    O(Var(den̄_c)/den²) systematic that does NOT average away over chains
    (simulated +0.07 on a true 0.5 at ⟨closed⟩=0.05, ns=5000; 2026-08-10 audit),
    while the pooled estimator stays unbiased ~16× deeper. ``den_z`` = pooled
    ⟨closed⟩ over its chain-scatter sem is the conditioning number: below ~5 the
    per-chain mean is not trustworthy; if den_z < 5 for the POOLED denominator
    the point is unmeasurable at this budget (the L>=10/12 no-go criterion).
    A non-positive pooled ⟨closed⟩ returns NaN with a warning — never the C++'s
    silent √|·| sign-fold. Returns None when <2 chains carry `raw_key`."""
    raw = np.array([c[raw_key] for c in chains if raw_key in c], dtype=float)
    if len(raw) < 2:
        return None
    num, den = raw[:, 0], raw[:, 1]

    def ratio(n_, d_):
        d = float(np.mean(d_))
        return float(np.mean(n_)) / np.sqrt(d) if d > 0 else float("nan")

    full = ratio(num, den)
    n = len(raw)
    jk = np.array([ratio(np.delete(num, i), np.delete(den, i)) for i in range(n)])
    err = float(np.sqrt((n - 1) / n * np.nansum((jk - np.nanmean(jk)) ** 2)))
    den_mean = float(np.mean(den))
    den_sem = float(np.std(den, ddof=1) / np.sqrt(n))
    if den_mean <= 0:
        print(f"[fm] WARNING {raw_key}: pooled <closed> = {den_mean:.4g} <= 0 — "
              f"ratio undefined at this budget (NaN, not |.|-folded)", flush=True)
    return dict(pooled=full, pooled_err=err, den=den_mean, den_err=den_sem,
                den_z=(den_mean / den_sem if den_sem > 0 else float("inf")))


def run_point(L, hx, hz, beta, n_chains, ns, n_blocks=4, basis="x", seed0=1000,
              workers=None, quiet=False, nbs_mult=1.0, fm=False, fm_membrane=False,
              fm_membrane_r1=False):
    """n_chains x n_blocks independent blocks -> equal-weight E (see combine note below).

    nbs_mult: scale decorrelation steps up near criticality (chi2_red >> 1 is the tell).
    fm: append the Fredenhagen-Marcu Z-string ratio (requires basis="z", L >= 4).
    fm_membrane: append the corner-rule X-membrane ratio (requires basis="x", L >= 5).
    fm_membrane_r1: append the R=1 anchor X-membrane ratio (requires basis="x", L >= 4).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if fm and (basis != "z" or L < 4):
        raise ValueError("fm requires basis='z' and L>=4 (cubic FM loops exist only "
                         "in the z basis; the rectangle degenerates below L=4)")
    if fm_membrane and (basis != "x" or L < 5):
        raise ValueError("fm_membrane requires basis='x' and L>=5 (sigma^x products "
                         "are diagonal only in the x basis; the corner-rule cube "
                         "[s,e]^3 needs s=(L-1)//4 >= 1)")
    if fm_membrane_r1 and (basis != "x" or L < 4):
        raise ValueError("fm_membrane_r1 requires basis='x' and L>=4 (the centered "
                         "R=1 anchor cube needs R <= L-3)")
    obs = (OBS + ([FM_OBS] if fm else []) + ([FM_MEM_OBS] if fm_membrane else [])
           + ([FM_MEM_R1_OBS] if fm_membrane_r1 else []))
    # decorrelation must scale with beta (kink density ~ beta), not just with nbs_mult
    scale = nbs_mult * max(1.0, beta / 12.0)
    nth = int(N_THERM[L] * scale)
    nbs = int(N_BETWEEN[L] * scale)
    jobs = [dict(L=L, hx=hx, hz=hz, beta=beta, ns=max(1000, ns // n_blocks),
                 nbs=nbs, nth=nth, seed=seed0 + 7 * i, basis=basis, obs=obs)
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
    # same equal-weight combine for every observable (nan-safe: e.g. the sigma_x
    # kink estimator divides by hx and is NaN at hx=0)
    combined = {}
    for o in obs:
        v = np.array([c[o][0] for c in chains], dtype=float)
        s = np.array([c[o][1] for c in chains], dtype=float)
        keep = np.isfinite(v)
        v, s = v[keep], s[keep]
        if len(v) >= 2:
            m = float(np.mean(v))
            chi2 = (float(np.sum((v - m) ** 2 / s**2) / (len(v) - 1))
                    if np.all(s > 0) else None)   # exact observables have std 0
            combined[o] = dict(
                mean=m, sem=float(np.std(v, ddof=1) / np.sqrt(len(v))),
                chi2_red=chi2)
    # FM observables additionally get the pooled readout (see `pooled_fm`):
    # `mean` (chain-averaged bias-corrected ratio) is fine at <closed>~0.7+,
    # `pooled` is the one to trust when <closed> is small (L>=8 program).
    for obs_name, raw_key in ((FM_OBS, "fm_num_den"), (FM_MEM_OBS, "fm_mem_num_den"),
                              (FM_MEM_R1_OBS, "fm_mem_r1_num_den")):
        if obs_name in combined:
            p = pooled_fm(chains, raw_key)
            if p:
                combined[obs_name].update(p)
    return dict(engine="paratoric", L=L, hx=hx, hz=hz, beta=beta, E=Ebar, E_err=err,
                chi2_red=chi2r, n_chains=n_chains, n_blocks=n_blocks,
                ns_per_block=jobs[0]["ns"], nbs=nbs, nth=nth, nbs_mult=nbs_mult,
                seed0=seed0, basis=basis, combined=combined, chains=chains)


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


def validate_fm(beta, ns, n_chains):
    """FM anchor rungs (basis z, L=4). No ED-sized loop exists (degenerate at L=2),
    so the certificates are symmetry-exact instead:
      - hz=0: the open half-string anticommutes with its two endpoint A_v, which
        commute with H -> <open> = 0 EXACTLY at any hx, beta. Two-sided z-test.
      - deep polarized (hz >> 1): half ~ m^2R, full ~ m^4R -> ratio -> 1. One-sided.
    Also reports <closed loop> (fm_den ~ 1 near h=0 checks normalization)."""
    rows = []
    r = run_point(4, 0.2, 0.0, beta, n_chains, ns, basis="z", fm=True, quiet=True)
    c = r["combined"][FM_OBS]
    den = float(np.mean([ch["fm_num_den"][1] for ch in r["chains"]]))
    rows.append(("FM zero (hx=.2,hz=0)", c["mean"], c["sem"], 0.0,
                 abs(c["mean"]) < 3 * c["sem"], f"<full>={den:+.4f}"))
    r = run_point(4, 0.0, 2.5, beta, n_chains, ns, basis="z", fm=True, quiet=True)
    c = r["combined"][FM_OBS]
    rows.append(("FM trivial (hx=0,hz=2.5)", c["mean"], c["sem"], 1.0,
                 c["mean"] > 0.9, ""))

    print(f"\n{'check':26s} {'O_FM':>10s} {'err':>9s} {'target':>7s}")
    ok = True
    for name, v, err, tgt, passed, note in rows:
        ok &= passed
        print(f"{name:26s} {v:10.6f} {err:9.6f} {tgt:7.2f} "
              f"{'PASS' if passed else 'FAIL'}  {note}")
    print("\nFM ladder", "PASSED" if ok else "FAILED")
    return ok


def _membrane_rungs(obs_name, raw_key, Ls, beta, ns, n_chains, **flag):
    """Anchor rungs for ONE membrane family at each L in `Ls` (dual to validate_fm):
      - hx=0: the open half-membrane anticommutes with the B_p ring along its
        equatorial boundary loop, and [H, B_p] = 0 at hx=0 -> <open> = 0 EXACTLY
        at any hz, beta. Two-sided z-test.
      - deep polarized (hx >> 1): both surfaces ~ m^Area, areas halve -> ratio -> 1.
    Also reports <closed cube> (~ 1 near h=0: it equals prod of enclosed A_v).
    NB the anchors are symmetry-exact for ANY valid halved cube, so they certify
    sampling + construction health (asserts/throws run), not the specific R —
    geometry identity vs the NQS operators is tests/test_fm_paratoric.py's job."""
    rows = []
    for L in Ls:
        r = run_point(L, 0.0, 0.2, beta, n_chains, ns, basis="x", quiet=True, **flag)
        c = r["combined"][obs_name]
        rows.append((f"FMm zero L={L} (hx=0,hz=.2)", c["mean"], c["sem"], 0.0,
                     abs(c["mean"]) < 3 * c["sem"],
                     f"<full>={c.get('den', float('nan')):+.4f} "
                     f"pooled={c.get('pooled', float('nan')):+.4f} "
                     f"den_z={c.get('den_z', float('nan')):.1f}"))
        r = run_point(L, 2.5, 0.0, beta, n_chains, ns, basis="x", quiet=True, **flag)
        c = r["combined"][obs_name]
        rows.append((f"FMm trivial L={L} (hx=2.5,hz=0)", c["mean"], c["sem"], 1.0,
                     c["mean"] > 0.9, ""))
    return rows


def _print_rungs(rows, title):
    print(f"\n{'check':30s} {'value':>10s} {'err':>9s} {'target':>7s}")
    ok = True
    for name, v, err, tgt, passed, note in rows:
        ok &= passed
        print(f"{name:30s} {v:10.6f} {err:9.6f} {tgt:7.2f} "
              f"{'PASS' if passed else 'FAIL'}  {note}")
    print(f"\n{title}", "PASSED" if ok else "FAILED")
    return ok


def validate_fm_membrane(beta, ns, n_chains):
    """Corner-rule membrane rungs at L=5 AND L=6. L=5 doubles as a continuity
    check (corner-rule cube == the pre-2026-08-10 centered R=2 cube there); L=6
    is the first size where the convention changed (R: 3 -> 2) — running it
    exercises the new construction path end-to-end (count asserts + edge checks)
    on a size the old geometry never had."""
    rows = _membrane_rungs(FM_MEM_OBS, "fm_mem_num_den", (5, 6), beta, ns,
                           n_chains, fm_membrane=True)
    return _print_rungs(rows, "FM membrane ladder")


def validate_fm_membrane_r1(beta, ns, n_chains):
    """R=1 anchor-family rungs at L=4 — the smallest lattice the anchor exists on
    (and the one the corner-rule family cannot reach)."""
    rows = _membrane_rungs(FM_MEM_R1_OBS, "fm_mem_r1_num_den", (4,), beta, ns,
                           n_chains, fm_membrane_r1=True)
    return _print_rungs(rows, "FM membrane R1 ladder")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true", help="run the exact-target ladder")
    ap.add_argument("--validate_fm", action="store_true",
                    help="run the Fredenhagen-Marcu anchor rungs (basis z, L=4)")
    ap.add_argument("--validate_fm_membrane", action="store_true",
                    help="run the corner-rule X-membrane anchor rungs (basis x, L=5+6)")
    ap.add_argument("--validate_fm_membrane_r1", action="store_true",
                    help="run the R=1 anchor X-membrane rungs (basis x, L=4)")
    ap.add_argument("--fm", action="store_true",
                    help="append the FM Z-string ratio (requires --basis z, L>=4)")
    ap.add_argument("--fm_membrane", action="store_true",
                    help="append the corner-rule FM X-membrane ratio "
                         "(requires --basis x, L>=5)")
    ap.add_argument("--fm_membrane_r1", action="store_true",
                    help="append the R=1 anchor FM X-membrane ratio "
                         "(requires --basis x, L>=4)")
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
    if args.validate_fm:
        ok = validate_fm(args.beta, max(200, args.samples // 10), args.chains)
        sys.exit(0 if ok else 1)
    if args.validate_fm_membrane:
        ok = validate_fm_membrane(args.beta, max(200, args.samples // 10), args.chains)
        sys.exit(0 if ok else 1)
    if args.validate_fm_membrane_r1:
        ok = validate_fm_membrane_r1(args.beta, max(200, args.samples // 10), args.chains)
        sys.exit(0 if ok else 1)

    r = run_point(args.L, args.hx, args.hz, args.beta, args.chains, args.samples,
                  n_blocks=args.blocks, basis=args.basis, nbs_mult=args.nbs_mult,
                  seed0=args.seed0, fm=args.fm, fm_membrane=args.fm_membrane,
                  fm_membrane_r1=args.fm_membrane_r1)
    def _fm_str(name, key):
        c = r["combined"][key]
        s = f"  {name} = {c['mean']:.6f} +- {c['sem']:.6f}"
        if "pooled" in c:      # trust `pooled` when <closed> is small (den_z gates)
            s += f" [pooled {c['pooled']:.6f} +- {c['pooled_err']:.6f}, den_z={c['den_z']:.1f}]"
        return s
    fm_str = (_fm_str("O_FM", FM_OBS) if args.fm else "")
    fm_str += (_fm_str("O_FM_mem", FM_MEM_OBS) if args.fm_membrane else "")
    fm_str += (_fm_str("O_FM_memR1", FM_MEM_R1_OBS) if args.fm_membrane_r1 else "")
    print(f"\nL={args.L} ({args.hx},{args.hz}) beta={args.beta}: "
          f"E = {r['E']:.6f} +- {r['E_err']:.6f}  chi2_red={r['chi2_red']:.2f}{fm_str}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=1)
        print("wrote", args.out)
