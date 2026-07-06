"""Multi-L phase-diagram analysis from the cluster-extracted FM curves.

Reads the per-L JSONs produced by `python -m Three_TC.fm` (pulled local) and,
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
    p0 = [y[0], y[-1] - y[0], float(np.median(x)), 0.1 * (x[-1] - x[0]) or 0.1]
    kw = {}
    if ye is not None and np.all(np.asarray(ye) > 0):
        kw = dict(sigma=np.asarray(ye, float), absolute_sigma=True)
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
    return sorted(recs, key=lambda r: r["L"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="dir of pulled fm_L*.json")
    p.add_argument("--out", default=None, help="output image (default: show)")
    p.add_argument("--fss", action="store_true", help="add h_c(L) vs 1/L panel")
    a = p.parse_args(argv)

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
        hh = np.linspace(hz.min(), hz.max(), 400)

        pO, eO = fit_sigmoid(hz, O, Oe)          # sigmoid on O_FM
        pM, _ = fit_sigmoid(hz, mz, mze)         # sigmoid on <sigma_z>
        h_c = pO[2] if pO is not None else np.nan

        # panel 0: O_FM(hz) + sigmoid
        ax[0].errorbar(hz, O, yerr=Oe, fmt="o", ms=4, capsize=2, color=c)
        if pO is not None:
            ax[0].plot(hh, logistic(hh, *pO), "-", color=c,
                       label=f"L={L}: $h_c$={h_c:.3f}±{eO:.3f}")
            ax[0].axvline(h_c, ls="--", color=c, lw=0.8, alpha=0.6)

        # panel 1: <sigma_z>(hz) + sigmoid
        ax[1].errorbar(hz, mz, yerr=mze, fmt="s", ms=4, capsize=2, color=c)
        if pM is not None:
            ax[1].plot(hh, logistic(hh, *pM), "-", color=c,
                       label=f"L={L}: infl={pM[2]:.3f}")

        # panel 2: derivative — finite-difference (data) + analytic sigmoid
        ax[2].plot(hz, np.gradient(O, hz), "o", ms=4, color=c, alpha=0.55,
                   label=f"L={L} finite-diff")
        if pO is not None:
            ax[2].plot(hh, dlogistic(hh, *pO), "-", color=c,
                       label=f"L={L} sigmoid, max={h_c:.3f}")
            ax[2].axvline(h_c, ls="--", color=c, lw=0.8, alpha=0.6)

        if pO is not None:
            Ls.append(L); hcs.append(h_c); hc_errs.append(eO)
        print(f"  L={L}: h_c(sigmoid)={h_c:.4f}  "
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
