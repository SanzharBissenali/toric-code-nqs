"""Shared minimalist plot style for the interim-report figures.

One `apply_style()` call gives every report figure the same clean look — serif
type to match the LaTeX body, no top/right spines, a muted palette, tight save.
Import and call at the top of each `fig_*.py` so the three figures read as one
system. Also exposes the muted palette + per-L viridis helper.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# muted, print-friendly palette (symmetric / baseline / accent)
BLUE   = "#2f5c8f"   # symmetric / primary
GREY   = "#9aa0a6"   # non-symmetric baseline
ORANGE = "#c2703d"   # accent / reference lines
INK    = "#222222"


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def l_colors(n):
    """n evenly spaced viridis colours (for the per-L overlays)."""
    import numpy as np
    return plt.cm.viridis(np.linspace(0.05, 0.75, n))
