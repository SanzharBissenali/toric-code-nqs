"""Figure 3 — schematic of the 3D combo architecture.

Left-to-right block diagram mirroring the 2D paper's Fig 1, adapted to the cubic
lattice: raw edge spins -> chi (non-invariant conv, the approx->exact symmetry
map) -> Wilson 4-product B_p (invariant nonlinearity) -> Omega (invariant conv)
-> sum -> log psi. Pure matplotlib patches; writes figures/arch_3d.png.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from report_style import apply_style, BLUE, GREY, ORANGE, INK

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "figures", "arch_3d.png")


def cube_glyph(ax, cx, cy, s=0.42):
    """Tiny wireframe cube with a couple of highlighted edge-spins."""
    d = s * 0.42  # depth offset
    # front and back squares
    fr = np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]])
    for sq, off in [(fr, (0, 0)), (fr, (d, d))]:
        ax.plot(cx + sq[:, 0] + off[0], cy + sq[:, 1] + off[1],
                color=INK, lw=0.8)
    for corner in fr[:4]:
        ax.plot([cx + corner[0], cx + corner[0] + d],
                [cy + corner[1], cy + corner[1] + d], color=INK, lw=0.8)
    # highlighted edges (the qubits live on edges)
    ax.plot([cx - s, cx + s], [cy - s, cy - s], color=ORANGE, lw=2.2)
    ax.plot([cx + s, cx + s], [cy - s, cy + s], color=BLUE, lw=2.2)


def block(ax, x, y, w, h, face, edge, title, sub):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1, edgecolor=edge, facecolor=face, alpha=0.92))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=11, color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=8.2, color=INK, style="italic")


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=12, lw=1.1, color=INK))


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    ax.set_xlim(0, 20); ax.set_ylim(0, 5); ax.axis("off")

    y, h = 1.3, 2.4
    # tint of the muted palette for block faces
    tint = lambda c: c  # keep edge colour; face is a light wash below
    faces = {"chi": "#e8eef5", "wil": "#f3ece4", "om": "#e8eef5"}

    # input cube glyph
    cube_glyph(ax, 1.3, y + h / 2)
    ax.text(1.3, y - 0.35, r"$\sigma$  (edge spins)", ha="center",
            va="center", fontsize=8.6, color=INK)

    xs = [3.0, 7.3, 11.6, 15.9]   # block left edges
    w = 3.4
    arrow(ax, 2.0, xs[0], y + h / 2)
    block(ax, xs[0], y, w, h, faces["chi"], BLUE,
          r"$\chi$", "non-invariant conv")
    ax.text(xs[0] + w / 2, y - 0.45, "approx. $\\to$ exact symmetry",
            ha="center", va="center", fontsize=7.8, color=BLUE)

    arrow(ax, xs[0] + w, xs[1], y + h / 2)
    block(ax, xs[1], y, w, h, faces["wil"], ORANGE,
          r"$B_p=\prod_{e\in p}\sigma^z_e$", "Wilson 4-product")
    ax.text(xs[1] + w / 2, y - 0.45, "invariant nonlinearity",
            ha="center", va="center", fontsize=7.8, color=ORANGE)

    arrow(ax, xs[1] + w, xs[2], y + h / 2)
    block(ax, xs[2], y, w, h, faces["om"], BLUE,
          r"$\Omega$", r"invariant conv $\times\,d$")

    arrow(ax, xs[2] + w, xs[3], y + h / 2)
    block(ax, xs[3], y, w, h, "#eeeeee", GREY,
          r"$\sum \;\to\; \log\psi$", "log-amplitude")

    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
