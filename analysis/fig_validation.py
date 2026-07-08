"""Figure 4 — L=2 (OBC) validation of the 3D combo net against exact diagonalization.

Reads the two wandb exports in the repo root (one holds `<model> - energy`, the
other `<model> - delta`, both indexed by `Step`, with `__MIN`/`__MAX` seed bands).

Panel (a): log10 relative error delta vs training step — symmetric combo vs the
symmetry-free GeoCNN baseline, each with its min-max band.
Panel (b): energy vs step near convergence with the ED reference E_exact (backed
out per-run as E/(1-delta)); the combo lands on it, the baseline plateaus above.
"""
import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from report_style import apply_style, BLUE, GREY, ORANGE, INK

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "figures", "validation_L2.png")

SYMM = "cnn-4-4_8-8-1"    # combo: non-inv (4-4) -> Wilson -> inv (8-8-1)
BASE = "geocnn-8-8-8"     # geometry CNN, no Wilson product (not A_v-invariant)


def load_csv(path):
    with open(path) as fh:
        r = csv.reader(fh)
        cols = next(r)
        rows = list(r)
    data = {c: [] for c in cols}
    for row in rows:
        for c, v in zip(cols, row):
            data[c].append(float(v) if v not in ("", "NaN") else np.nan)
    return {c: np.asarray(v) for c, v in data.items()}


def find_exports():
    """Return (energy_csv, delta_csv) by sniffing which columns each holds."""
    e = d = None
    for p in sorted(glob.glob(os.path.join(HERE, "wandb_export_*.csv"))):
        cols = load_csv(p).keys()
        if any(c.endswith(" - energy") for c in cols):
            e = p
        if any(c.endswith(" - delta") for c in cols):
            d = p
    if not (e and d):
        raise SystemExit("could not locate energy/delta wandb exports in repo root")
    return e, d


def series(data, model, metric):
    step = data["Step"]
    y = data[f"{model} - {metric}"]
    lo = data.get(f"{model} - {metric}__MIN")
    hi = data.get(f"{model} - {metric}__MAX")
    m = np.isfinite(y)
    return step[m], y[m], (lo[m] if lo is not None else None), \
        (hi[m] if hi is not None else None)


def main():
    apply_style()
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                         "legend.fontsize": 11, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "lines.linewidth": 1.7})
    ecsv, dcsv = find_exports()
    E, D = load_csv(ecsv), load_csv(dcsv)

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))

    # ---- panel (a): log10 delta -------------------------------------------
    for model, c, lab in [(SYMM, BLUE, "symmetric (combo)"),
                          (BASE, GREY, "non-symmetric (GeoCNN)")]:
        s, y, lo, hi = series(D, model, "delta")
        ax[0].semilogy(s, y, color=c, label=lab)
        if lo is not None and hi is not None:
            ax[0].fill_between(s, lo, hi, color=c, alpha=0.15, lw=0)
        print(f"{lab:26s} best delta = {np.nanmin(y):.2e}  final = {y[-1]:.2e}")
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel(r"relative error  $\delta = |E-E_0|/|E_0|$")
    ax[0].legend(loc="upper right")
    ax[0].text(0.02, 0.03, "(a)", transform=ax[0].transAxes, fontsize=13)

    # ---- panel (b): zoomed energy vs step ---------------------------------
    # back out E_exact per run from the converged tail (E = E0 (1 - delta))
    exacts = []
    for model in (SYMM, BASE):
        s, e, *_ = series(E, model, "energy")
        _, dd, *_ = series(D, model, "delta")
        n = min(len(e), len(dd))
        tail = (e[:n] / (1.0 - dd[:n]))[-25:]
        exacts.append(np.nanmedian(tail))
    E0 = float(np.mean(exacts))
    print(f"E_exact (L=2 OBC) = {E0:.5f}")

    for model, c, lab in [(SYMM, BLUE, "symmetric (combo)"),
                          (BASE, GREY, "non-symmetric (GeoCNN)")]:
        s, e, lo, hi = series(E, model, "energy")
        ax[1].plot(s, e, color=c, label=lab)
        if lo is not None and hi is not None:
            ax[1].fill_between(s, lo, hi, color=c, alpha=0.15, lw=0)
    ax[1].axhline(E0, ls="--", lw=1.0, color=ORANGE,
                  label=fr"$E_0$ (ED) $= {E0:.3f}$")
    # zoom to the last-third of training where the gap is the story; tight
    # energy window so the baseline plateau sitting above E_0 is legible
    s_all = E["Step"][np.isfinite(E[f"{SYMM} - energy"])]
    ax[1].set_xlim(s_all.max() * 0.45, s_all.max())
    ax[1].set_ylim(E0 - 0.003, E0 + 0.016)
    ax[1].set_xlabel("training step")
    ax[1].set_ylabel(r"energy  $E$")
    ax[1].legend(loc="upper right")
    ax[1].text(0.02, 0.03, "(b)", transform=ax[1].transAxes, fontsize=13)

    fig.tight_layout()
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
