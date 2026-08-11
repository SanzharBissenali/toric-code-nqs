"""Multi-L phase-diagram analysis from the cluster-extracted FM curves.

Reads the per-L JSONs produced by `python -m tc3d.fm` (pulled local) and,
for every L present, fits a logistic (sigmoid) to both the Fredenhagen-Marcu
order parameter O_FM(hz) and the magnetization <sigma_z>(hz). Panels:

  1. O_FM(hz)         — data + sigmoid fit, L overlaid; h_c = sigmoid inflection
  2. <sigma_z>(hz)    — data + sigmoid fit, L overlaid
  3. dO_FM/dhz        — finite-difference (np.gradient of data) AND the analytic
                        sigmoid derivative; its maximum locates the transition
  (--fss)             — h_c(L) vs 1/L with a linear L->inf extrapolation

Pure numpy / scipy / matplotlib — NO NetKet. Run after pulling the curves:
    rsync -avz <host>:$PSCRATCH/tc_nqs/phase_hx0.2/fm_L*.json ./results/phase_hx0.2/
    python analysis/plot_phase_diagram.py --dir results/phase_hx0.2 --fss --out phase.png
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logistic(h, a, b, h0, w):
    return a + b / (1.0 + np.exp(-(h - h0) / w))


def dlogistic(h, a, b, h0, w):
    """Analytic derivative of the logistic; peaks at h = h0 (the inflection)."""
    z = np.exp(-(h - h0) / w)
    return (b / w) * z / (1.0 + z) ** 2


def fit_sigmoid(x, y, ye=None):
    """Logistic fit -> (popt, h0_err). popt is None if the fit fails."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ye = None if ye is None else np.asarray(ye, float)
    # Drop non-finite VALUES (den<=0 -> NaN convention makes them reachable);
    # mirrors fm.fit_transition (2026-08-11 re-verification, finding 2).
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.all():
        print(f"  [fit] dropped {int((~keep).sum())} non-finite point(s)")
        x, y = x[keep], y[keep]
        ye = ye if ye is None else ye[keep]
    if len(y) < 4:
        print(f"  [fit] only {len(y)} finite points — skipping fit")
        return None, float("nan")
    p0 = [y[0], y[-1] - y[0], float(np.median(x)), 0.1 * (x[-1] - x[0]) or 0.1]
    kw = {}
    if ye is not None:
        # Saturated Oe=0.0 points floor at the smallest positive error; a
        # NON-FINITE error means "least trustworthy" and CEILINGS at the largest
        # (the old min-floor gave those points maximum weight) — mirrors
        # fm.fit_transition.
        s = ye
        pos = s[np.isfinite(s) & (s > 0)]
        if len(pos):
            s = np.where(np.isfinite(s), np.where(s > 0, s, pos.min()), pos.max())
            kw = dict(sigma=s, absolute_sigma=True)
    try:
        popt, pcov = curve_fit(logistic, x, y, p0=p0, maxfev=20000, **kw)
        return popt, float(np.sqrt(abs(pcov[2, 2])))
    except Exception as exc:
        print(f"  [fit] logistic fit failed: {exc}")
        return None, float("nan")


def load_curves(directory):
    recs = [json.load(open(jp))
            for jp in sorted(glob.glob(os.path.join(directory, "fm_L*.json")))]
    if not recs:
        raise SystemExit(f"no fm_L*.json in {directory} — pull the extracted curves first")
    # One operator family per dir (the TAG discipline): FSS over mixed families is
    # not meaningful. Warn on the family markers the JSONs carry; R is not compared
    # here because it legitimately varies with L for the growing families.
    fams = {(r.get("placement"), r.get("convention"), str(r.get("aspect")))
            for r in recs}
    if len(fams) > 1:
        print(f"[plot] WARNING: {directory} mixes operator families {sorted(fams)}"
              f" — split them into separate dirs before fitting")
    # Family markers can't separate two BULK dirs (fixed-R vs largest both look
    # ('bulk', None, None)), but their symptom can: duplicate L entries, which
    # double-count that L in every fit downstream.
    Ls = [r["L"] for r in recs]
    if len(set(Ls)) < len(Ls):
        dups = sorted({x for x in Ls if Ls.count(x) > 1})
        print(f"[plot] WARNING: {directory} has multiple curves at L={dups} — "
              f"mixed families or stale files; FSS would double-count these L")
    return sorted(recs, key=lambda r: r["L"])


def _parse_exclusions(items):
    """['6:0.24', '6:0.36'] -> {6: [0.24, 0.36]}: (L, hz) points to drop before fitting."""
    excl = {}
    for it in items:
        Ls, hzs = it.split(":")
        excl.setdefault(int(Ls), []).append(float(hzs))
    return excl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="dir of pulled fm_L*.json")
    p.add_argument("--out", default=None, help="output image (default: show)")
    p.add_argument("--fss", action="store_true", help="add h_c(L) vs 1/L panel")
    p.add_argument("--no-fits", dest="no_fits", action="store_true",
                   help="show data points only — drop the sigmoid overlays (h_c "
                        "still fitted for the labels + FSS)")
    p.add_argument("--exclude", nargs="*", default=[], metavar="L:hz",
                   help="drop specific (L, hz) points before fitting, e.g. "
                        "--exclude 6:0.24 6:0.36 (collapsed/untrusted runs)")
    a = p.parse_args(argv)

    excl = _parse_exclusions(a.exclude)
    recs = load_curves(a.dir)
    hx = recs[0]["hx"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax = axes.flatten()      # [0]=O_FM  [1]=<sigma_z>  [2]=derivative  [3]=FSS
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(recs)))

    Ls, hcs, hc_errs = [], [], []
    print(f"hx = {hx}")
    for rec, c in zip(recs, colors):
        L = rec["L"]
        hz = np.array(rec["field"])
        O, Oe = np.array(rec["O"]), np.array(rec["Oe"])
        mz = np.array(rec["mz"])
        mze = np.array(rec.get("mz_e", np.zeros_like(mz)))
        if L in excl:                    # drop named bad points (e.g. collapsed runs)
            keep = ~np.any([np.isclose(hz, h, atol=1e-6) for h in excl[L]], axis=0)
            print(f"  L={L}: dropped {int((~keep).sum())} point(s) at "
                  f"hz={list(np.round(hz[~keep], 3))}")
            hz, O, Oe, mz, mze = hz[keep], O[keep], Oe[keep], mz[keep], mze[keep]
        hh = np.linspace(hz.min(), hz.max(), 400)

        pO, eO = fit_sigmoid(hz, O, Oe)          # sigmoid on O_FM
        pM, _ = fit_sigmoid(hz, mz, mze)         # sigmoid on <sigma_z>
        h_c = pO[2] if pO is not None else np.nan

        # --no-fits: join neighbouring data points with straight segments instead
        # of overlaying the smooth sigmoid; "-o"/"-s" = connect-the-dots.
        dfmt = ("-o", "-s") if a.no_fits else ("o", "s")

        # panel 0: O_FM(hz) (+ sigmoid unless --no-fits). Label on the markers so
        # the legend survives with the curve suppressed.
        oL = f"L={L}: $h_c$={h_c:.3f}±{eO:.3f}" if pO is not None else f"L={L}"
        ax[0].errorbar(hz, O, yerr=Oe, fmt=dfmt[0], ms=4, lw=1.0, capsize=2,
                       color=c, label=oL)
        if pO is not None:
            if not a.no_fits:
                ax[0].plot(hh, logistic(hh, *pO), "-", color=c)
            ax[0].axvline(h_c, ls="--", color=c, lw=0.8, alpha=0.6)

        # panel 1: <sigma_z>(hz) (+ sigmoid unless --no-fits)
        mL = f"L={L}: infl={pM[2]:.3f}" if pM is not None else f"L={L}"
        ax[1].errorbar(hz, mz, yerr=mze, fmt=dfmt[1], ms=4, lw=1.0, capsize=2,
                       color=c, label=mL)
        if pM is not None and not a.no_fits:
            ax[1].plot(hh, logistic(hh, *pM), "-", color=c)

        # panel 2: derivative — finite-difference (data) (+ analytic sigmoid unless --no-fits)
        ax[2].plot(hz, np.gradient(O, hz), dfmt[0], ms=4, lw=1.0, color=c, alpha=0.55,
                   label=f"L={L} finite-diff")
        if pO is not None:
            if not a.no_fits:
                ax[2].plot(hh, dlogistic(hh, *pO), "-", color=c,
                           label=f"L={L} sigmoid, max={h_c:.3f}")
            ax[2].axvline(h_c, ls="--", color=c, lw=0.8, alpha=0.6)

        if pO is not None:
            Ls.append(L); hcs.append(h_c); hc_errs.append(eO)
        fam = rec.get("convention") or rec.get("placement", "?")
        print(f"  L={L} [{fam}, R={rec.get('R')}]: h_c(sigmoid)={h_c:.4f}  "
              f"h_c_fd(stored)={rec.get('h_c_fd')}  npts={len(hz)}")

    ax[0].set(xlabel="$h_z$", ylabel="$O_{FM}$", title=f"FM order parameter (hx={hx})")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="$h_z$", ylabel=r"$\langle\sigma_z\rangle$", title="magnetization")
    ax[1].legend(fontsize=8)
    ax[2].set(xlabel="$h_z$", ylabel=r"$dO_{FM}/dh_z$",
              title="derivative — peak = transition")
    ax[2].legend(fontsize=8)

    if a.fss and len(Ls) >= 2:
        Ls, hcs, hc_errs = np.array(Ls, float), np.array(hcs), np.array(hc_errs)
        x = 1.0 / Ls
        m, b = np.polyfit(x, hcs, 1)
        xs = np.linspace(0, x.max() * 1.05, 50)
        ax[3].errorbar(x, hcs, yerr=hc_errs, fmt="o", capsize=3, color="C3")
        ax[3].plot(xs, m * xs + b, "-", color="C3", lw=1.2)
        ax[3].axhline(b, ls=":", color="k", label=f"$h_c(\\infty)\\approx{b:.3f}$")
        ax[3].set(xlabel="$1/L$", ylabel="$h_c(L)$", title="finite-size scaling")
        ax[3].legend(fontsize=8)
    else:
        ax[3].axis("off")    # no FSS panel -> leave the 4th cell blank

    fig.suptitle("3D bosonic toric code — topological→trivial transition")
    fig.tight_layout()
    if a.out:
        fig.savefig(a.out, dpi=150, bbox_inches="tight")
        print(f"wrote {a.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
