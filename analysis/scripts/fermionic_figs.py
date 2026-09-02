"""Headless PNG rendering for the fermionic L=2 OBC plane + hx-ladder
campaigns. Repo rule: notebook savefig lines stay commented out (the user
saves manually); scripts may auto-save (cf. analysis/scripts/phaseB_figs.py).
This script mirrors the layout/style of analysis/notebooks/fermionic_plane_L2.ipynb
and fermionic_hx_ladder.ipynb (same loaders: analysis/scripts/plane_summary.py),
but always writes PNGs and is data-driven over whatever tiers/arms/points are
present -- missing points render as blank/hatched cells, never crash.

Usage (from the repo root):
    PYTHONPATH=<worktree> .venv/bin/python analysis/scripts/fermionic_figs.py \
        [--plane_dir results/fermionic_plane_L2] \
        [--ladder_dir results/fermionic_hx_ladder] \
        [--hy 0.0] [--out analysis/figs]

Writes:
    fermionic_plane_L2_relerr[_hy{hy}].png       1x4 arms, shared LogNorm, rel. energy error
    fermionic_plane_L2_infidelity[_hy{hy}].png   same layout, trained infidelity 1-F
    fermionic_plane_L2_vs_ceiling[_hy{hy}].png   achieved 1-F (top) vs gate-0 ceiling 1-F_s (bottom)
    fermionic_hx_ladder_curves.png               2x3 learning curves, all tiers found
    fermionic_hx_ladder_vs_hx.png                rel-err | 1-fidelity vs hx, all tiers found
Also prints a compact markdown table per figure to stdout.
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "scripts"))
from plane_summary import build as build_plane_summary  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

DPI = 150

# ============================================================ plane ==== #
HX_VALUES = [0.0, 0.2, 0.5, 1.0]
HZ_VALUES = [0.0, 0.2, 0.5, 1.0]
ARMS = ["asymm", "anaC_k0", "pt2sf", "pt2sfc"]
ARM_LABEL = {
    "asymm": "asymm (no head)",
    "anaC_k0": r"anaC, $\kappa$=0",
    "pt2sf": "pt2 head (sf)",
    "pt2sfc": "pt2 head (sfc)",
}
# Okabe-Ito palette (plot-style-spec), matching fermionic_plane_L2.ipynb.
ARM_COLOR = {"asymm": "#0072B2", "anaC_k0": "#E69F00", "pt2sf": "#009E73", "pt2sfc": "#CC79A7"}
# arm -> matching gate-0 head name in the sign_fidelity_ftc.py F_s / one_minus_F_s dicts
ARM_HEAD = {"asymm": "plus", "anaC_k0": "anaC", "pt2sf": "pt2", "pt2sfc": "pt2"}

GATE0_PATH = ROOT / "results" / "fermionic_gate0" / "2x2x2_OBC_plane_gate0.json"


def hy_suffix(hy):
    return f"_hy{hy}" if hy else ""


def load_plane(plane_dir, hy):
    """(hx, hz, arm) -> row dict; prints how many of the 16xlen(ARMS) runs
    were found. Tolerant of a missing/empty/partial directory."""
    root = Path(plane_dir)
    n_expected = len(HX_VALUES) * len(HZ_VALUES) * len(ARMS)
    if not root.is_dir():
        print(f"[plane] {root} does not exist -- 0/{n_expected} runs found "
              "(figures render with blank/hatched cells)")
        return {}
    rows = build_plane_summary(root, arms=ARMS, hy=(hy if hy else 0.0))
    by_key = {(r["hx"], r["hz"], r["arm"]): r for r in rows}
    print(f"[plane] found {len(by_key)}/{n_expected} runs in {root} (hy={hy})")
    return by_key


def build_grid(field_fn):
    W = np.full((len(HZ_VALUES), len(HX_VALUES)), np.nan)
    for i, hz in enumerate(HZ_VALUES):
        for j, hx in enumerate(HX_VALUES):
            W[i, j] = field_fn(hx, hz)
    return W


def _positive_finite(mats):
    parts = [m[np.isfinite(m) & (m > 0)] for m in mats.values()]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.array([])


def _annotate_missing(ax):
    """Blank/hatched styling for a fully-nan or partially-nan panel."""
    ax.set_facecolor("0.85")


def plot_arms_plane(by_key, field_fn, title, cbar_label, out_path, cmap="Blues"):
    """P1-style 1x4 arms panel. field_fn(row) -> float (nan if row is None)."""
    def cell(hx, hz, arm):
        r = by_key.get((hx, hz, arm))
        v = field_fn(r)
        return np.nan if v is None else v

    mats = {arm: build_grid(lambda hx, hz, a=arm: cell(hx, hz, a)) for arm in ARMS}
    pos = _positive_finite(mats)
    fig, axes = plt.subplots(1, len(ARMS), figsize=(3.5 * len(ARMS) + 1.3, 4.3),
                              sharey=True, constrained_layout=True)
    if pos.size == 0:
        for ax, arm in zip(axes, ARMS):
            _annotate_missing(ax)
            ax.set_xticks(range(len(HX_VALUES))); ax.set_xticklabels(HX_VALUES)
            ax.set_yticks(range(len(HZ_VALUES))); ax.set_yticklabels(HZ_VALUES)
            ax.set(xlabel="$h_x$", title=ARM_LABEL[arm])
        axes[0].set(ylabel="$h_z$")
        fig.suptitle(title + "  [no data yet]", y=1.05)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[plane] wrote {out_path} (no finite data -- all cells blank)")
        return
    vmin, vmax = pos.min(), pos.max()
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * (1 + 1e-9)))

    im = None
    for ax, arm in zip(axes, ARMS):
        ax.grid(False)
        M = mats[arm]
        Mm = np.ma.masked_invalid(M)
        im = ax.imshow(Mm, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        _annotate_missing(ax)
        for i, hz in enumerate(HZ_VALUES):
            for j, hx in enumerate(HX_VALUES):
                v = M[i, j]
                if not np.isfinite(v):
                    continue
                dark = norm(max(v, vmin)) > 0.6
                ax.text(j, i, f"{v:.1e}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "#1a1a1a")
        ax.set_xticks(range(len(HX_VALUES))); ax.set_xticklabels(HX_VALUES)
        ax.set_yticks(range(len(HZ_VALUES))); ax.set_yticklabels(HZ_VALUES)
        ax.set(xlabel="$h_x$", title=ARM_LABEL[arm])
    axes[0].set(ylabel="$h_z$")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(cbar_label)
    fig.suptitle(title, y=1.05)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plane] wrote {out_path}")


def print_plane_table(by_key, field_fn, title):
    print(f"\n### {title}\n")
    for arm in ARMS:
        print(f"**{arm}**\n")
        header = "| hz\\hx | " + " | ".join(str(hx) for hx in HX_VALUES) + " |"
        sep = "|---" * (len(HX_VALUES) + 1) + "|"
        print(header); print(sep)
        for hz in HZ_VALUES:
            cells = []
            for hx in HX_VALUES:
                r = by_key.get((hx, hz, arm))
                v = field_fn(r)
                cells.append(f"{v:.2e}" if v is not None else "--")
            print(f"| {hz} | " + " | ".join(cells) + " |")
        print()


def load_gate0(path):
    if not path.exists():
        warnings.warn(f"{path} not found -- section skipped.")
        return None
    d = json.load(open(path))
    return {(round(p["hx"], 6), round(p["hz"], 6)): p for p in d["points"]}


def plot_vs_ceiling(by_key, gate0, out_path):
    if gate0 is None:
        print(f"[plane] skip {out_path.name}: no gate-0 plane ceiling file")
        return

    def achieved(hx, hz, arm):
        r = by_key.get((hx, hz, arm))
        if r is None or r.get("fidelity") is None:
            return np.nan
        return 1.0 - r["fidelity"]

    def ceiling(hx, hz, arm):
        p = gate0.get((round(hx, 6), round(hz, 6)))
        if p is None:
            return np.nan
        return p["one_minus_F_s"].get(ARM_HEAD[arm], np.nan)

    W_ach = {arm: build_grid(lambda hx, hz, a=arm: achieved(hx, hz, a)) for arm in ARMS}
    W_ceil = {arm: build_grid(lambda hx, hz, a=arm: ceiling(hx, hz, a)) for arm in ARMS}
    pos_a, pos_c = _positive_finite(W_ach), _positive_finite(W_ceil)
    norm_a = LogNorm(vmin=pos_a.min(), vmax=pos_a.max()) if pos_a.size else None
    norm_c = LogNorm(vmin=pos_c.min(), vmax=pos_c.max()) if pos_c.size else None

    fig, axes = plt.subplots(2, len(ARMS), figsize=(3.5 * len(ARMS) + 1.3, 8.2),
                              sharex=True, sharey=True, constrained_layout=True)
    rows = [(W_ach, norm_a, "Blues", "achieved\n1-F"),
            (W_ceil, norm_c, "Purples", "gate-0 ceiling\n$1-F_s$")]
    for r, (W, norm_r, cmap_r, row_lab) in enumerate(rows):
        im = None
        for ax, arm in zip(axes[r], ARMS):
            ax.grid(False)
            M = W[arm]
            Mm = np.ma.masked_invalid(M)
            Mm = np.ma.masked_less_equal(Mm, 0.0)
            im = (ax.imshow(Mm, origin="lower", cmap=cmap_r, norm=norm_r, interpolation="nearest")
                  if norm_r is not None else
                  ax.imshow(Mm, origin="lower", cmap=cmap_r, interpolation="nearest"))
            ax.set_facecolor("0.9")
            for i, hz in enumerate(HZ_VALUES):
                for j, hx in enumerate(HX_VALUES):
                    v = M[i, j]
                    if not np.isfinite(v):
                        continue
                    if v <= 0:
                        ax.text(j, i, "exact", ha="center", va="center", fontsize=7.5,
                                color="#1a1a1a", style="italic")
                    else:
                        dark = (norm_r(v) > 0.6) if norm_r is not None else False
                        ax.text(j, i, f"{v:.1e}", ha="center", va="center", fontsize=8,
                                color="white" if dark else "#1a1a1a")
            ax.set_xticks(range(len(HX_VALUES))); ax.set_xticklabels(HX_VALUES)
            ax.set_yticks(range(len(HZ_VALUES))); ax.set_yticklabels(HZ_VALUES)
            if r == 0:
                ax.set(title=ARM_LABEL[arm])
            else:
                ax.set(xlabel="$h_x$")
        axes[r][0].set(ylabel=f"{row_lab}\n$h_z$")
        if norm_r is not None and im is not None:
            cb = fig.colorbar(im, ax=list(axes[r]), shrink=0.85, pad=0.02)
            cb.set_label("achieved infidelity $1-F$" if r == 0 else "exact ceiling $1-F_s$")
    fig.suptitle("Achieved trained infidelity (top) vs gate-0 sign-fidelity ceiling "
                 "(bottom), per arm", y=1.02)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plane] wrote {out_path}")


def print_ceiling_table(by_key, gate0, title):
    print(f"\n### {title}\n")
    if gate0 is None:
        print("(no gate-0 file -- skipped)\n")
        return
    for arm in ARMS:
        print(f"**{arm}** (achieved 1-F / gate-0 ceiling 1-F_s for head `{ARM_HEAD[arm]}`)\n")
        header = "| hz\\hx | " + " | ".join(str(hx) for hx in HX_VALUES) + " |"
        sep = "|---" * (len(HX_VALUES) + 1) + "|"
        print(header); print(sep)
        for hz in HZ_VALUES:
            cells = []
            for hx in HX_VALUES:
                r = by_key.get((hx, hz, arm))
                ach = (1.0 - r["fidelity"]) if (r and r.get("fidelity") is not None) else None
                p = gate0.get((round(hx, 6), round(hz, 6)))
                ceil_ = p["one_minus_F_s"].get(ARM_HEAD[arm]) if p else None
                ach_s = f"{ach:.2e}" if ach is not None else "--"
                ceil_s = f"{ceil_:.2e}" if ceil_ is not None else "--"
                cells.append(f"{ach_s} / {ceil_s}")
            print(f"| {hz} | " + " | ".join(cells) + " |")
        print()


# ======================================================== hx ladder ==== #
LADDER_RUN_RE = re.compile(
    r"^(geocnn|gridinv)_fermionic_L2_OBC_hx([\d.]+)_hz([\d.]+)_(?:k\d+_)?(.+)$")

# Known tier -> (color, label). New tiers not listed here (e.g. still-landing
# pt2sf/votesf/anaCsf jobs at some point in the future beyond this list) fall
# back to FALLBACK_COLORS + the bare tier name, so the figures never crash.
TIER_COLOR = {
    "plain": "#808080", "asymm": "#0072B2", "anaC_k0": "#E69F00", "anaC_k6": "#009E73",
    "pt2sf": "#CC79A7", "votesf": "#D55E00", "anaCsf": "#8B4513",
}
TIER_LABEL = {
    "plain": "sign-blind GeoCNN (plain)",
    "asymm": "approx-symm gridinv (asymm, no head)",
    "anaC_k0": r"frozen analytic head, $\kappa$=0",
    "anaC_k6": r"frozen analytic head, $\kappa$=6",
    "pt2sf": "pt2 sign-framed head (sf)",
    "votesf": "vote sign-framed head (sf)",
    "anaCsf": "anaC sign-framed head (sf)",
}
TIER_ORDER = list(TIER_COLOR)  # display order for known tiers; unknowns append after
FALLBACK_COLORS = ["#F0E442", "#56B4E9", "#000000"]  # remaining Okabe-Ito, cycled

# anaC frozen analytic head's intrinsic full-space sign match vs ED (fixed by
# hx alone, independent of training) -- copied from fermionic_hx_ladder.ipynb
# section 1 (source: leader-task log, slurm tc-fhxladder-57836502_0.out).
HEAD_INTRINSIC_SIGN_MATCH = {0.1: 0.9975, 0.2: 0.9890, 0.3: 0.9710,
                             0.5: 0.9125, 0.7: 0.8560, 1.0: 0.7839}


def discover_ladder_runs(ladder_dir):
    """tier -> {hx: run_stem}, magnetic line (hz=0) only, sorted by first
    appearance for known tiers then discovery order for new ones."""
    root = Path(ladder_dir)
    found = {}
    if not root.is_dir():
        return found, root
    for f in sorted(root.glob("*.json")):
        stem = f.stem
        if stem.endswith((".curve", ".snapshots")) or stem == "summary":
            continue
        if stem.startswith(("exact_diag", "ed_L")):
            continue
        m = LADDER_RUN_RE.match(stem)
        if not m:
            continue
        _eng, hx_s, hz_s, tier = m.groups()
        if float(hz_s) != 0.0:
            continue
        found.setdefault(tier, {})[float(hx_s)] = stem
    return found, root


def tier_color(tier, extra_idx):
    if tier in TIER_COLOR:
        return TIER_COLOR[tier]
    return FALLBACK_COLORS[extra_idx % len(FALLBACK_COLORS)]


def load_ladder_row(root, stem, ed_cache):
    d = json.load(open(root / f"{stem}.json"))
    cfg = d["config"]
    hx, hz = cfg["hx"], cfg["hz"]
    E = d["observables"]["E0"]
    Vscore = d["observables"]["Vscore"]
    diverged = d["diverged"]
    last_step = d["curve"]["step"][-1] if d["curve"]["step"] else None

    key = (hx, hz)
    if key not in ed_cache:
        fn = root / f"exact_diag_fermionic_L2_OBC_hx{hx}_hz{hz}.json"
        ed_cache[key] = json.load(open(fn)) if fn.exists() else None
    ed = ed_cache[key]
    E_exact = ed["E0"] if ed else None
    rel = abs(E - E_exact) / abs(E_exact) if E_exact else None

    fidelity = sign_match = None
    snap_f = root / f"{stem}.snapshots.json"
    if snap_f.exists():
        series = json.load(open(snap_f))["series"]
        if series:
            last_snap = series[-1]
            fidelity = last_snap.get("exact", {}).get("fidelity")
            sign_match = last_snap.get("exact", {}).get("sign_match_weighted")

    return dict(name=stem, hx=hx, hz=hz, E=E, E_exact=E_exact, rel=rel,
                fidelity=fidelity, sign_match=sign_match, Vscore=Vscore,
                diverged=diverged, last_step=last_step)


def load_ladder(ladder_dir):
    tiers_map, root = discover_ladder_runs(ladder_dir)
    if not tiers_map:
        print(f"[ladder] {root} has no matching runs")
        return [], {}, {}, root
    tiers = [t for t in TIER_ORDER if t in tiers_map] + \
            sorted(t for t in tiers_map if t not in TIER_COLOR)
    hx_values = sorted({hx for hxmap in tiers_map.values() for hx in hxmap})
    ed_cache = {}
    by_key = {}
    for tier, hxmap in tiers_map.items():
        for hx, stem in hxmap.items():
            try:
                by_key[(hx, tier)] = load_ladder_row(root, stem, ed_cache)
            except (FileNotFoundError, KeyError) as e:
                warnings.warn(f"[ladder] skipping {stem}: {e}")
    n_found = len(by_key)
    n_expected = len(hx_values) * len(tiers)
    known = [t for t in tiers if t in TIER_COLOR]
    new = [t for t in tiers if t not in TIER_COLOR]
    print(f"[ladder] found {n_found}/{n_expected} runs in {root}: "
          f"tiers={tiers} (new since spec: {new or 'none'}), hx={hx_values}")
    return hx_values, tiers, by_key, root


def plot_ladder_curves(hx_values, tiers, by_key, root, out_path):
    if not hx_values:
        print(f"[ladder] skip {out_path.name}: no data")
        return
    n = len(hx_values)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.1 * nrows), squeeze=False)
    axes_flat = axes.flat
    ed_cache = {}
    for ax, hx in zip(axes_flat, hx_values):
        for idx, tier in enumerate(tiers):
            r = by_key.get((hx, tier))
            if r is None:
                continue
            color = tier_color(tier, idx)
            curve_f = root / f"{r['name']}.curve.json"
            if not curve_f.exists():
                continue
            c = json.load(open(curve_f))["curve"]
            step, E = np.asarray(c["step"], float), np.asarray(c["energy"], float)
            label = TIER_LABEL.get(tier, tier)
            ax.plot(step, E, lw=1.4, alpha=0.9, color=color, label=label)
            if r["diverged"] and step.size:
                ax.plot(step[-1], E[-1], marker="x", color=color, ms=9, mew=2, zorder=5)
            key = (hx, 0.0)
            if key not in ed_cache:
                fn = root / f"exact_diag_fermionic_L2_OBC_hx{hx}_hz0.0.json"
                ed_cache[key] = json.load(open(fn))["E0"] if fn.exists() else None
        if ed_cache.get((hx, 0.0)) is not None:
            ax.axhline(ed_cache[(hx, 0.0)], color="k", lw=1.2, zorder=1, label="$E_0$ (ED)")
        ax.set_title(f"$h_x={hx}$")
        ax.set_xlabel("SR step")
        ax.set_ylabel(r"$\langle H \rangle$")
    for ax in list(axes_flat)[n:]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center",
               ncol=min(5, len(by_label)), frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(
        "Fermionic L=2 OBC arch ladder -- learning curves per $h_x$ "
        r"($\times$ = last state before the divergence guard stopped the run)",
        y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ladder] wrote {out_path}")


def plot_ladder_vs_hx(hx_values, tiers, by_key, out_path):
    if not hx_values:
        print(f"[ladder] skip {out_path.name}: no data")
        return
    fig, (ax_rel, ax_fid) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for idx, tier in enumerate(tiers):
        color = tier_color(tier, idx)
        label = TIER_LABEL.get(tier, tier)
        rel = [by_key[(hx, tier)]["rel"] if (hx, tier) in by_key else np.nan for hx in hx_values]
        fid = [by_key[(hx, tier)]["fidelity"] if (hx, tier) in by_key else np.nan for hx in hx_values]
        infid = [1.0 - f if f is not None and np.isfinite(f) else np.nan for f in fid]
        ax_rel.plot(hx_values, rel, marker="o", ms=5.5, lw=1.7, color=color, label=label)
        ax_fid.plot(hx_values, infid, marker="o", ms=5.5, lw=1.7, color=color, label=label)

    intrinsic = [1.0 - HEAD_INTRINSIC_SIGN_MATCH[hx] if hx in HEAD_INTRINSIC_SIGN_MATCH
                 else np.nan for hx in hx_values]
    if not all(np.isnan(intrinsic)):
        ax_rel.plot(hx_values, intrinsic, marker="s", ms=5.5, lw=1.3, ls="--", color="k",
                    label=r"anaC head intrinsic $1-$sign match")
        ax_fid.plot(hx_values, intrinsic, marker="s", ms=5.5, lw=1.3, ls="--", color="k",
                    label=r"anaC head intrinsic $1-$sign match")

    ax_rel.set_yscale("log"); ax_fid.set_yscale("log")
    ax_rel.set_xlabel("$h_x$"); ax_rel.set_ylabel(r"$|E_{final}-E_0|/|E_0|$")
    ax_rel.set_title("Relative energy error vs $h_x$")
    ax_rel.legend(frameon=False, fontsize=7.5)
    ax_fid.set_xlabel("$h_x$"); ax_fid.set_ylabel(r"trained infidelity $1-F$")
    ax_fid.set_title("Infidelity vs $h_x$")
    ax_fid.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Fermionic L=2 OBC arch ladder, magnetic line ($h_z=0$)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ladder] wrote {out_path}")


def print_ladder_table(hx_values, tiers, by_key, title):
    print(f"\n### {title}\n")
    header = "| hx | " + " | ".join(tiers) + " |"
    sep = "|---" * (len(tiers) + 1) + "|"
    print(header); print(sep)
    for hx in hx_values:
        cells = []
        for tier in tiers:
            r = by_key.get((hx, tier))
            if r is None:
                cells.append("--")
                continue
            rel_s = f"{r['rel']:.2e}" if r.get("rel") is not None else "n/a"
            fid_s = f"{1 - r['fidelity']:.2e}" if r.get("fidelity") is not None else "n/a"
            div_s = "DIV" if r["diverged"] else ""
            cells.append(f"rel={rel_s}, 1-F={fid_s} {div_s}".strip())
        print(f"| {hx} | " + " | ".join(cells) + " |")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plane_dir", default=str(ROOT / "results" / "fermionic_plane_L2"))
    ap.add_argument("--ladder_dir", default=str(ROOT / "results" / "fermionic_hx_ladder"))
    ap.add_argument("--hy", type=float, default=0.0)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "figs"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suf = hy_suffix(args.hy)

    # ---- plane ----
    by_key = load_plane(args.plane_dir, args.hy)

    def rel_of(r):
        return None if r is None else r.get("rel")

    def infid_of(r):
        return None if r is None or r.get("fidelity") is None else 1.0 - r["fidelity"]

    plot_arms_plane(by_key, rel_of, "Fermionic L=2 OBC plane: relative energy error per arm",
                     r"rel. energy error $|E-E_0|/|E_0|$",
                     out / f"fermionic_plane_L2_relerr{suf}.png", cmap="Blues")
    plot_arms_plane(by_key, infid_of, "Fermionic L=2 OBC plane: trained infidelity per arm",
                     r"trained infidelity $1-F$",
                     out / f"fermionic_plane_L2_infidelity{suf}.png", cmap="Purples")
    print_plane_table(by_key, rel_of, "fermionic_plane_L2_relerr: rel. energy error per arm")
    print_plane_table(by_key, infid_of, "fermionic_plane_L2_infidelity: 1-F per arm")

    gate0 = load_gate0(GATE0_PATH)
    plot_vs_ceiling(by_key, gate0, out / f"fermionic_plane_L2_vs_ceiling{suf}.png")
    print_ceiling_table(by_key, gate0, "fermionic_plane_L2_vs_ceiling: achieved 1-F / gate-0 ceiling 1-F_s")

    # ---- hx ladder ----
    hx_values, tiers, ladder_by_key, ladder_root = load_ladder(args.ladder_dir)
    plot_ladder_curves(hx_values, tiers, ladder_by_key, ladder_root,
                        out / "fermionic_hx_ladder_curves.png")
    plot_ladder_vs_hx(hx_values, tiers, ladder_by_key, out / "fermionic_hx_ladder_vs_hx.png")
    print_ladder_table(hx_values, tiers, ladder_by_key,
                        "fermionic_hx_ladder_curves / vs_hx: rel-err, 1-F per hx per tier")


if __name__ == "__main__":
    main()
