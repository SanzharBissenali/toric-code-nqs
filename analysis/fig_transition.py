"""Figure 5 — topological->trivial transition at fixed hx=0.2, sweeping hz.

Features the BULK-placement Fredenhagen-Marcu order parameter (L=4,5,6). Three
slim panels: O_FM(hz) + sigmoid fit; the analytic derivative whose peak locates
h_c(L); and the 1/L finite-size extrapolation. The edge placement (L=3-6) is
overlaid on the FSS panel as a lighter dashed line to show placement dependence.

Reuses logistic / dlogistic / fit_sigmoid / load_curves from plot_phase_diagram.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from report_style import apply_style, l_colors, ORANGE, GREY, INK
from plot_phase_diagram import logistic, dlogistic, fit_sigmoid, load_curves

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK = os.path.join(HERE, "results", "phase_hx0.2_bulk")
EDGE = os.path.join(HERE, "results", "phase_hx0.2")
OUT = os.path.join(HERE, "figures", "transition.png")


def hc_by_L(recs):
    """[(L, h_c, err), ...] from a sigmoid fit of O_FM(hz), sorted by L."""
    out = []
    for r in recs:
        hz, O, Oe = map(np.asarray, (r["field"], r["O"], r["Oe"]))
        p, e = fit_sigmoid(hz, O, Oe)
        if p is not None:
            out.append((r["L"], p[2], e))
    return sorted(out)


def fss_line(ax, LhE, color, ls, label):
    L = np.array([t[0] for t in LhE], float)
    hc = np.array([t[1] for t in LhE])
    err = np.array([t[2] for t in LhE])
    x = 1.0 / L
    m, b = np.polyfit(x, hc, 1)
    xs = np.linspace(0, x.max() * 1.08, 50)
    ax.errorbar(x, hc, yerr=err, fmt="o", ms=4, capsize=2.5, color=color, lw=1.0)
    ax.plot(xs, m * xs + b, ls=ls, lw=1.2, color=color,
            label=f"{label}: $h_c(\\infty)\\approx{b:.2f}$")
    return b


def main():
    apply_style()
    # wide 3-panel figure shrinks a lot at \linewidth; enlarge type so it stays legible
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 14, "axes.labelsize": 15,
                         "legend.fontsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "lines.linewidth": 1.8})
    bulk = load_curves(BULK)
    edge = load_curves(EDGE)
    colors = l_colors(len(bulk))

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))

    hc_bulk = []
    for r, c in zip(bulk, colors):
        hz, O, Oe = map(np.asarray, (r["field"], r["O"], r["Oe"]))
        p, e = fit_sigmoid(hz, O, Oe)
        hh = np.linspace(hz.min(), hz.max(), 400)
        # (a) O_FM(hz)
        ax[0].errorbar(hz, O, yerr=Oe, fmt="o", ms=5, capsize=2.5, lw=0,
                       elinewidth=0.8, color=c, label=f"$L={r['L']}$")
        if p is not None:
            ax[0].plot(hh, logistic(hh, *p), "-", color=c, lw=1.3)
            ax[0].axvline(p[2], ls=":", lw=0.8, color=c, alpha=0.6)
            # (b) derivative
            ax[1].plot(hh, dlogistic(hh, *p), "-", color=c, lw=1.4,
                       label=f"$L={r['L']}$: $h_c={p[2]:.2f}$")
            ax[1].axvline(p[2], ls=":", lw=0.8, color=c, alpha=0.6)
            hc_bulk.append((r["L"], p[2], e))

    ax[0].set_xlabel("$h_z$"); ax[0].set_ylabel("$O_{\\mathrm{FM}}$")
    ax[0].legend(loc="upper left")
    ax[0].text(0.97, 0.06, "(a)", transform=ax[0].transAxes, fontsize=13,
               ha="right")

    ax[1].set_xlabel("$h_z$"); ax[1].set_ylabel("$dO_{\\mathrm{FM}}/dh_z$")
    ax[1].legend(loc="upper right")
    ax[1].text(0.03, 0.92, "(b)", transform=ax[1].transAxes, fontsize=13)

    # (c) FSS: bulk (solid) + edge (dashed, lighter)
    b_bulk = fss_line(ax[2], sorted(hc_bulk), ORANGE, "-", "bulk")
    b_edge = fss_line(ax[2], hc_by_L(edge), GREY, "--", "edge")
    ax[2].axhline(0.2, ls=":", lw=0.8, color=INK, alpha=0.5)
    ax[2].set_xlabel("$1/L$"); ax[2].set_ylabel("$h_c(L)$")
    ax[2].set_xlim(left=0)
    ax[2].legend(loc="lower right")
    ax[2].text(0.03, 0.92, "(c)", transform=ax[2].transAxes, fontsize=13)

    print(f"bulk h_c(inf) = {b_bulk:.3f} | edge h_c(inf) = {b_edge:.3f}")
    print("bulk per-L:", [(L, round(h, 3)) for L, h, _ in sorted(hc_bulk)])

    fig.tight_layout()
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
