#!/usr/bin/env python
"""3D toric-code NQS architecture figure (paper / slides).

A left-to-right banner analogue of the 2D-paper `summary_figure.png` panel (a),
drawn from the *real* `ThreeD_ToricCodeGeometry`.  Mirrors
`tc3d/networks.py::ToricCNN_full` block-for-block:

    edge spins (o)  -> CNN chi (pre-Wilson conv)        [ Non-symmetric ]
      -> sigma : Wilson product  B_p = prod_{i in p} s_i     (edges o -> plaq [])
    plaquette fluxes ([])  -> CNN Omega (post-Wilson conv)  [   Symmetric   ]
      -> Sum -> psi_s

Pure geometry construction + matplotlib (no NetKet training, no ED) -> safe to
run locally.  Exports an editable vector master (SVG + PDF) and a slide PNG.

    .venv/bin/python analysis/scripts/arch_figure.py
"""
import importlib.util
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

# ----------------------------------------------------------------------------- config
L, BC = 3, "OBC"                 # 3x3x3 vertex cube (a 2x2x2 block of cells)
OUT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "figures")
FIGSTEM = "arch_3d"

# semantic palette matched to the 2D reference figure (NOT the plasma data-plot map)
COL = dict(
    nonsym="#3B7DD8",            # CNN chi  (blue)
    wilson="#2E9E5B",            # Wilson product (green)
    sym="#E1802B",               # CNN Omega (orange)
    summ="#CB4335",              # Sum -> psi (red)
    brack="#7D4FA3",             # under-brackets (purple)
    node_ec="0.30",              # node outline
)
# graded face tints (top lightest -> right darkest) give the solid-box read
FACE = dict(top="#e6ecf3", front="#d3dce6", right="#bcc8d6")

# cabinet projection: x -> right, z -> up, y -> receding up-right (foreshortened)
AZ = np.radians(41.0)
KDEPTH = 0.52


# ----------------------------------------------------------------------------- geometry
def load_geo(L, bc):
    """Real 3D geometry; return edge/plaquette coordinates + connectivity."""
    spec = importlib.util.spec_from_file_location(
        "geo3d", os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                              "tc3d", "geometry.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    geo = m.ThreeD_ToricCodeGeometry(L, L, L, bc)

    inv = {v: k for k, v in geo._coord_to_idx.items()}          # 2x-int coord -> qubit
    ec = np.array([np.asarray(inv[i]) / 2.0 for i in range(geo.N)])   # edge midpoints
    pc = np.array([np.asarray(c) for c in geo.plaq_centers])    # plaquette centres
    plaq_edges = [list(map(int, p)) for p in geo.plaq_all]      # 4 edges / plaquette
    return dict(N=geo.N, ec=ec, pc=pc, plaq_edges=plaq_edges, L=L, bc=bc)


# ----------------------------------------------------------------------------- projection
def iso(p, ox=0.0, oy=0.0):
    """Cabinet projection. Returns X, Y and a front-ness weight in [0,1]."""
    p = np.atleast_2d(np.asarray(p, dtype=float))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    X = x + KDEPTH * np.cos(AZ) * y + ox
    Y = z + KDEPTH * np.sin(AZ) * y + oy
    front = 1.0 - y / max(L - 1, 1)          # 1 at the front face, 0 at the back
    return X, Y, front


def _seg(ax, a, b, ox, oy, **kw):
    (xa, xb), (ya, yb) = iso(np.vstack([a, b]), ox, oy)[:2]
    ax.plot([xa, xb], [ya, yb], solid_capstyle="round", **kw)


# ----------------------------------------------------------------------------- box + faces
def draw_box(ax, L, ox, oy, alpha=1.0, z0=0.0):
    """Shaded top/front/right faces + solid visible / dashed hidden edges."""
    m = L - 1
    def C(x, y, z):
        X, Y, _ = iso([x, y, z], ox, oy)
        return (X[0], Y[0])

    faces = [                                                   # (corners, tint)
        ([C(0, 0, m), C(m, 0, m), C(m, m, m), C(0, m, m)], FACE["top"]),    # z=max
        ([C(0, 0, 0), C(m, 0, 0), C(m, 0, m), C(0, 0, m)], FACE["front"]),  # y=0
        ([C(m, 0, 0), C(m, m, 0), C(m, m, m), C(m, 0, m)], FACE["right"]),  # x=max
    ]
    for corners, tint in faces:
        ax.add_patch(Polygon(corners, closed=True, facecolor=tint, edgecolor="none",
                             alpha=0.85 * alpha, zorder=z0 + 1))

    hidden = (0, m, 0)                                          # back-left-bottom vertex
    verts = [(x, y, z) for x in (0, m) for y in (0, m) for z in (0, m)]
    for i, a in enumerate(verts):
        for b in verts[i + 1:]:
            if sum(np.not_equal(a, b)) != 1:                   # not a cube edge
                continue
            hid = (a == hidden or b == hidden)
            _seg(ax, a, b, ox, oy, color="0.5" if hid else "0.35",
                 lw=1.0 if hid else 1.4, ls=(0, (3, 3)) if hid else "-",
                 alpha=(0.55 if hid else 0.9) * alpha, zorder=z0 + 2)


# ----------------------------------------------------------------------------- lattice
def draw_cube(ax, g, ox, oy, kind, highlight=None, hl_color=None, z0=0.0):
    """Shaded box + o (edge midpoints) or [] (plaquette centres), depth-cued."""
    L = g["L"]
    draw_box(ax, L, ox, oy, z0=z0)

    # faint interior lattice lines (the grid the conv walks over)
    for b in range(L):
        for c in range(L):
            _seg(ax, [0, b, c], [L - 1, b, c], ox, oy, color="0.72", lw=0.5,
                 alpha=0.45, zorder=z0 + 1.5)
            _seg(ax, [b, 0, c], [b, L - 1, c], ox, oy, color="0.72", lw=0.5,
                 alpha=0.45, zorder=z0 + 1.5)
            _seg(ax, [b, c, 0], [b, c, L - 1], ox, oy, color="0.72", lw=0.5,
                 alpha=0.45, zorder=z0 + 1.5)

    hl = set() if highlight is None else set(int(i) for i in highlight)
    hl_color = hl_color or COL["nonsym"]
    pts = g["ec"] if kind == "edge" else g["pc"]
    X, Y, fr = iso(pts, ox, oy)

    for i in np.argsort(fr):                     # back -> front (painter's algorithm)
        on = i in hl
        s = 0.70 + 0.30 * fr[i]                  # depth scale: front bigger
        a = 0.60 + 0.40 * fr[i]                  #              front more opaque
        z = z0 + 10 + fr[i]
        if kind == "edge":
            ax.add_patch(Circle((X[i], Y[i]), 0.108 * s,
                                 fc=(hl_color if on else "white"),
                                 ec=(hl_color if on else COL["node_ec"]),
                                 lw=0.8, alpha=a, zorder=z))
        else:
            ax.scatter([X[i]], [Y[i]], marker="s", s=64 * s,
                       c=(hl_color if on else "#5d5d5d"),
                       edgecolors="k", linewidths=0.4, alpha=a, zorder=z)


# ----------------------------------------------------------------------------- transitions
def wilson_transition(ax, g, ox_e, oy_e, ox_p, oy_p, p):
    """Green lines: the 4 o edges of plaquette p -> its [] flux node."""
    Xp, Yp, _ = iso(g["pc"][p], ox_p, oy_p)
    for e in g["plaq_edges"][p]:
        Xe, Ye, _ = iso(g["ec"][e], ox_e, oy_e)
        ax.add_patch(FancyArrowPatch((Xe[0], Ye[0]), (Xp[0], Yp[0]),
                                     arrowstyle="-", color=COL["wilson"], lw=1.2,
                                     linestyle=(0, (4, 2)), alpha=0.85, zorder=25,
                                     shrinkA=3, shrinkB=5))


def sum_transition(ax, g, ox_p, oy_p, xpsi, ypsi):
    """Red converging lines: every [] flux node -> single psi_s dot."""
    X, Y, _ = iso(g["pc"], ox_p, oy_p)
    for xi, yi in zip(X, Y):
        ax.plot([xi, xpsi], [yi, ypsi], color=COL["summ"], lw=0.45, alpha=0.30,
                zorder=3)
    ax.add_patch(Circle((xpsi, ypsi), 0.16, fc=COL["summ"], ec="k", lw=0.9, zorder=30))
    ax.text(xpsi + 0.34, ypsi, r"$\psi_s$", color=COL["summ"], fontsize=19,
            va="center", ha="left", fontweight="bold")


def block_arrow(ax, x0, x1, y, color):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=18, color=color, lw=2.4, zorder=9))


def under_bracket(ax, x0, x1, y, label, color):
    ax.plot([x0, x0, x1, x1], [y + 0.22, y, y, y + 0.22], color=color, lw=1.6,
            solid_capstyle="round", zorder=9)
    ax.text((x0 + x1) / 2, y - 0.28, label, color=color, fontsize=14,
            ha="center", va="top", fontweight="bold")


# ----------------------------------------------------------------------------- assemble
def main():
    g = load_geo(L, BC)
    assert g["N"] == 54 and len(g["pc"]) == 36, "expected L=3 OBC counts"

    # a small, generic local patch (conveys "local convolution" without a real kernel)
    def near(pts, target, k):
        d = np.linalg.norm(pts - target, axis=1)
        return np.argsort(d)[:k]
    p_ctr = int(np.argmin(np.abs(g["pc"] - 1.0).sum(1)))          # a central plaquette
    e_patch = near(g["ec"], np.array([1.0, 1.0, 1.0]), 4)         # 4 edges, blue
    p_patch = near(g["pc"], np.array([1.0, 1.0, 1.0]), 4)         # 4 plaquettes, orange

    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.set_aspect("equal")
    ax.axis("off")

    xA, xB, xC = 0.0, 5.6, 11.2          # input-edge, conv-edge, conv-plaq
    x_psi = 15.6
    yrow = 1.0                           # transition baseline (~cube centre height)

    draw_cube(ax, g, xA, 0.0, "edge")                                    # input spins
    block_arrow(ax, xA + 2.9, xB - 0.7, yrow, COL["nonsym"])             # CNN chi
    draw_cube(ax, g, xB, 0.0, "edge", highlight=e_patch, hl_color=COL["nonsym"])
    wilson_transition(ax, g, xB, 0.0, xC, 0.0, p_ctr)                    # sigma Wilson
    draw_cube(ax, g, xC, 0.0, "plaq", highlight=p_patch, hl_color=COL["sym"])
    sum_transition(ax, g, xC, 0.0, x_psi, yrow)                         # Sum -> psi

    yb = -0.55
    under_bracket(ax, xA - 0.5, xB + 2.9, yb, "Non-symmetric", COL["brack"])
    under_bracket(ax, xC - 0.5, x_psi + 0.7, yb, "Symmetric", COL["brack"])

    ax.set_xlim(xA - 0.9, x_psi + 1.4)
    ax.set_ylim(yb - 1.1, 3.3)
    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    for ext, dpi in (("svg", None), ("pdf", None), ("png", 300)):
        fig.savefig(os.path.join(OUT, f"{FIGSTEM}.{ext}"), dpi=dpi,
                    bbox_inches="tight", transparent=False)
    print("wrote", ", ".join(f"{FIGSTEM}.{e}" for e in ("svg", "pdf", "png")),
          "->", os.path.normpath(OUT))
    plt.close(fig)


if __name__ == "__main__":
    main()
