"""Multi-L phase-diagram plot from the cluster-extracted FM curves.

Reads the tiny per-L JSONs produced by `python -m Three_TC.fm` (pulled local),
overlays the Fredenhagen-Marcu order parameter O_FM(hz) and its derivative
dO_FM/dhz for every L, and marks the per-L pseudo-critical field h_c(L). The
derivative PEAK is the transition. Optionally extrapolates h_c(L) vs 1/L.

Pure numpy / matplotlib — NO NetKet (the heavy O_FM eval already ran on the
cluster). Run locally after:
    rsync -avz <host>:$PSCRATCH/tc_nqs/phase_hx0.2/fm_L*.json ./results/phase_hx0.2/

    python analysis/plot_phase_diagram.py --dir results/phase_hx0.2 --out phase_hx0.2.png
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def load_curves(directory):
    """Load every fm_L*.json in `directory`, sorted by L."""
    recs = []
    for jp in sorted(glob.glob(os.path.join(directory, "fm_L*.json"))):
        with open(jp) as f:
            recs.append(json.load(f))
    if not recs:
        raise SystemExit(f"no fm_L*.json in {directory} — pull the extracted curves first")
    return sorted(recs, key=lambda r: r["L"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="dir of pulled fm_L*.json")
    p.add_argument("--out", default=None, help="output image (default: show)")
    p.add_argument("--fss", action="store_true", help="add h_c(L) vs 1/L extrapolation panel")
    a = p.parse_args(argv)

    recs = load_curves(a.dir)
    hx = recs[0]["hx"]
    ncol = 3 if a.fss else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.4))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(recs)))

    Ls, hcs, hc_errs = [], [], []
    for rec, c in zip(recs, colors):
        L = rec["L"]
        field = np.array(rec["field"])
        O, Oe = np.array(rec["O"]), np.array(rec["Oe"])
        h_c = rec.get("h_c")
        h_c = h_c if h_c is not None else rec.get("h_c_fd")

        # panel 0: O_FM(hz) with fit overlay
        ax[0].errorbar(field, O, yerr=Oe, fmt="o", ms=4, capsize=2, color=c, label=f"L={L}")
        if rec.get("fit_curve"):
            ax[0].plot(rec["fit_curve"]["h"], rec["fit_curve"]["O_fit"], "-", color=c, lw=1.2)
        if h_c is not None:
            ax[0].axvline(h_c, ls="--", color=c, lw=0.9, alpha=0.7)

        # panel 1: numerical derivative of the DATA (np.gradient -> defined at
        # every hz incl. endpoints); its extremum is the transition.
        dOdh = np.gradient(O, field)
        ax[1].plot(field, dOdh, "o-", ms=4, color=c, label=f"L={L}")
        if h_c is not None:
            ax[1].axvline(h_c, ls="--", color=c, lw=0.9, alpha=0.7)

        if h_c is not None:
            Ls.append(L); hcs.append(h_c)
            hc_errs.append(rec.get("h_c_err") or 0.0)

    ax[0].set(xlabel="$h_z$", ylabel="$O_{FM}$", title=f"FM order parameter (hx={hx})")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="$h_z$", ylabel="$dO_{FM}/dh_z$", title="derivative — peak = transition")
    ax[1].legend(fontsize=8)

    # optional FSS: h_c(L) vs 1/L, linear extrapolation to L->inf
    if a.fss and len(Ls) >= 2:
        Ls = np.array(Ls, float); hcs = np.array(hcs); hc_errs = np.array(hc_errs)
        x = 1.0 / Ls
        m, b = np.polyfit(x, hcs, 1)
        xs = np.linspace(0, x.max() * 1.05, 50)
        ax[2].errorbar(x, hcs, yerr=hc_errs, fmt="o", capsize=3, color="C3")
        ax[2].plot(xs, m * xs + b, "-", color="C3", lw=1.2)
        ax[2].axhline(b, ls=":", color="k", label=f"$h_c(\\infty)\\approx{b:.3f}$")
        ax[2].set(xlabel="$1/L$", ylabel="$h_c(L)$", title="finite-size scaling")
        ax[2].legend(fontsize=8)

    fig.suptitle("3D bosonic toric code — topological→trivial transition")
    fig.tight_layout()
    if a.out:
        fig.savefig(a.out, dpi=150, bbox_inches="tight")
        print(f"wrote {a.out}")
        for L, hc in zip(Ls, hcs):
            print(f"  L={int(L)}: h_c = {hc:.4f}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
