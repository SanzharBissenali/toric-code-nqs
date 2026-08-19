"""Generate the eight committed phaseB_* benchmark figures in analysis/figs/.

This script IS the provenance of those PNGs (ported 2026-08-19 from the
Phase-B reconciliation session; the plotting cells in phaseB_summary.ipynb
predate the reconciliation and must not be used to regenerate them). It
encodes the three campaign-final conventions:

  1. QMC references use the HIGHEST-beta subset per (point, L, basis) —
     never mix betas (beta=12 x-basis refs are thermally biased near the
     first-order window; see notes/transition_mapping_recipes.md SSB.7).
  2. NQS values come from the per-point BEST STATE: the 500-step rerun
     (phaseB2_* core protocol files) where it landed, overridden by the
     warm-chain / extension winners in WARM_RIGHT (the substitution table
     referenced by BLOG.md's campaign-close entry). Metastable-branch states
     are never substituted — lowest-energy branch only.
  3. Order-parameter panels draw NQS bars x3, labelled "(bars x3)" (NQS
     errors are underestimated ~x3; raw pulls must still be scanned for
     sign-coherent runs — the x3 is a display convention, not a fix).

Usage (from anywhere; paths are repo-relative to this file):
    python analysis/phaseB_figs.py                 # both cuts -> analysis/figs/
    python analysis/phaseB_figs.py --cut right     # h_x sweep only
    python analysis/phaseB_figs.py --no-save       # dry run, prints landings

Known trap baked in: QMC dir names are not {:g}-uniform (qmc_hx1.0_hz0.1);
find_qmc_dir falls back to glob+parse-by-value only when the primary
:g-constructed path is absent.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from exact_benchmarks import Counts

FIGS = os.path.join(ROOT, "analysis", "figs")
FIG_DPI = 300

LS = [4, 5, 6]
PLASMA = {4: plt.cm.plasma(0.15), 5: plt.cm.plasma(0.5), 6: plt.cm.plasma(0.8)}
QMC_COLOR = "0.25"

UP_HZ = [0.10, 0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
         0.32, 0.34, 0.36, 0.40, 0.45, 0.50]
HX_UP = 0.2
RIGHT_HX = [0.20, 0.35, 0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95,
            1.00, 1.05, 1.10, 1.175, 1.25]
HZ_RIGHT = 0.1

SAVE_FIGS = True


def openax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s.replace("$", "").replace(",", "")).strip("_")


def maybe_save(fig, title):
    if not SAVE_FIGS:
        return
    path = os.path.join(FIGS, f"phaseB_{slug(title)}.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print("saved:", path)


# ---------------- QMC loader (highest-beta subset, dir-name fallback) -------

_QMC_DIR_RE = re.compile(r"^qmc_hx([0-9.]+)_hz([0-9.]+)$")


def find_qmc_dir(hx, hz):
    primary = os.path.join(ROOT, "results", f"qmc_hx{hx:g}_hz{hz:g}")
    if os.path.isdir(primary):
        return primary
    for d in glob.glob(os.path.join(ROOT, "results", "qmc_hx*_hz*")):
        m = _QMC_DIR_RE.match(os.path.basename(d))
        if m and abs(float(m.group(1)) - hx) < 1e-9 and abs(float(m.group(2)) - hz) < 1e-9:
            return d
    return primary


def load_qmc_point(L, hx, hz, basis):
    qdir = find_qmc_dir(hx, hz)
    files = []
    for f in sorted(glob.glob(os.path.join(qdir, f"paratoric_L{L}_*.json"))):
        d = json.load(open(f))
        if d.get("basis") != basis or "combined" not in d:
            continue
        files.append((d.get("beta", 0), d))
    if files:
        bmax = max(b for b, _ in files)
        files = [(b, d) for b, d in files if b == bmax]
    means, sems, n = {}, {}, 0
    for _, d in files:
        n += 1
        for k, v in d["combined"].items():
            if isinstance(v, dict) and "mean" in v:
                means.setdefault(k, []).append(v["mean"])
                sems.setdefault(k, []).append(v["sem"])
    if n == 0:
        return None
    return {k: (float(np.mean(means[k])), float(np.sqrt(np.sum(np.square(sems[k]))) / n))
            for k in means}


# ---------------- NQS loaders: original campaign + rerun/warm substitution --

def load_nqs_point(cut, L, hx, hz):
    path = os.path.join(ROOT, "results", "phaseB", cut, f"L{L}",
                        f"phaseB_dual_L{L}_hx{hx:g}_hz{hz:g}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))["observables"]


# up cut: every 500-step rerun point supersedes its old 150-step value.
RERUN_UP_RE = re.compile(r"phaseB2_n500_L(\d+)_hx([0-9.]+)_hz([0-9.]+)\.json$")
# right cut: only the core dt01n500 protocol files (anchored — excludes _s1/_s2/
# _warm*/ds5 probes), L4/L5 only (the L6 dt=0.01 window runs diverged).
RERUN_RIGHT_RE = re.compile(r"^phaseB2_dt01n500_L(\d+)_hx([0-9.]+)_hz0\.1\.json$")


def build_rerun_up():
    idx = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "phaseB_rerun",
                                           "up", "L*", "phaseB2_*.json"))):
        base = os.path.basename(f)
        if base.endswith((".curve.json", ".snapshots.json")):
            continue
        m = RERUN_UP_RE.match(base)
        if m:
            idx[(int(m.group(1)), round(float(m.group(2)), 6),
                 round(float(m.group(3)), 6))] = f
    return idx


def build_rerun_right():
    idx = {}
    for L in (4, 5):
        for f in sorted(glob.glob(os.path.join(ROOT, "results", "phaseB_rerun",
                                               "right", f"L{L}", "phaseB2_*.json"))):
            m = RERUN_RIGHT_RE.match(os.path.basename(f))
            if m:
                idx[(int(m.group(1)), round(float(m.group(2)), 6), 0.1)] = f
    return idx


# Per-point best states from the warm-chain / extension campaign (2026-08-17/19).
# Policy: lowest-energy branch only — metastable-branch states go to the
# hysteresis figure, never here. L6 0.9-1.05 come from the down-chain (wc11);
# L6 0.85 is the prodext topological branch (GS at the resonance is unreachable
# by any single branch — labelled resonance window in the campaign report).
_R = os.path.join(ROOT, "results", "phaseB_rerun", "right")
WARM_RIGHT = {
    (4, 0.75, 0.1): f"{_R}/L4/phaseB2_dt005n200_L4_hx0.75_hz0.1_wc08.json",
    (4, 0.8, 0.1):  f"{_R}/L4/phaseB2_dt005n200_L4_hx0.8_hz0.1_wc09.json",
    (5, 0.8, 0.1):  f"{_R}/L5/phaseB2_dt005n400_L5_hx0.8_hz0.1_coldext.json",
    (5, 0.85, 0.1): f"{_R}/L5/phaseB2_dt005n200_L5_hx0.85_hz0.1_wc090.json",
    (6, 0.75, 0.1): f"{_R}/L6/phaseB2_dt01n500_L6_hx0.75_hz0.1.json",
    (6, 0.8, 0.1):  f"{_R}/L6/phaseB2_dt005n200_L6_hx0.8_hz0.1_wcup075.json",
    (6, 0.85, 0.1): f"{_R}/L6/phaseB2_dt005n400_L6_hx0.85_hz0.1_prodext.json",
    (6, 0.9, 0.1):  f"{_R}/L6/phaseB2_dt005n200_L6_hx0.9_hz0.1_wc11.json",
    (6, 0.95, 0.1): f"{_R}/L6/phaseB2_dt005n200_L6_hx0.95_hz0.1_wc11.json",
    (6, 1.0, 0.1):  f"{_R}/L6/phaseB2_dt005n200_L6_hx1.0_hz0.1_wc11.json",
    (6, 1.05, 0.1): f"{_R}/L6/phaseB2_dt005n200_L6_hx1.05_hz0.1_wc11.json",
    (6, 1.25, 0.1): f"{_R}/L6/phaseB2_dt01n500_L6_hx1.25_hz0.1.json",
}


def build_substitutions(cut):
    if cut == "up":
        return build_rerun_up()
    idx = build_rerun_right()
    for k, p in WARM_RIGHT.items():
        assert os.path.exists(p), f"missing best-state file: {p}"
        idx[k] = p
    return idx


def make_loader(cut):
    subs = build_substitutions(cut)
    print(f"[{cut}] substitutions active: {len(subs)}")

    def load(L, hx, hz):
        key = (L, round(hx, 6), round(hz, 6))
        if key in subs:
            return json.load(open(subs[key]))["observables"]
        return load_nqs_point(cut, L, hx, hz)

    return load


# ---------------- assemble + plot (verbatim campaign helpers) ----------------

def assemble(cut, fields, basis, fixed):
    load = make_loader(cut)
    out = {}
    for L in LS:
        bound = Counts(L, "OBC").E0
        rows = []
        for field in fields:
            hx, hz = (fixed, field) if cut == "up" else (field, fixed)
            qmc = load_qmc_point(L, hx, hz, basis)
            nqs = load(L, hx, hz)
            flags = []
            e0 = nqs["E0"] if nqs is not None else None
            if e0 is not None and e0 >= bound - 1e-6:
                flags.append(f"E0={e0:.3f} >= h=0 bound {bound:.0f}")
            rows.append(dict(field=field, qmc=qmc, nqs=nqs, e0=e0, flags=flags))
        out[L] = rows
    n_landed = sum(1 for L in LS for r in out[L] if r["nqs"] is not None)
    print(f"=== {cut} cut: {n_landed}/{len(LS) * len(fields)} NQS points landed ===")
    for L in LS:
        missing = [r["field"] for r in out[L] if r["nqs"] is None]
        if missing:
            print(f"  L={L}: MISSING {missing}")
        for r in out[L]:
            if r["flags"]:
                print(f"  L={L} field={r['field']}: FLAGGED -> {r['flags']}")
    return out


def series(rows, nqs_key, nqs_err_key, qmc_key):
    out = []
    for r in rows:
        if r["nqs"] is None or r["qmc"] is None or qmc_key not in r["qmc"]:
            continue
        nm, ne = r["nqs"][nqs_key], r["nqs"].get(nqs_err_key, 0.0)
        qm, qe = r["qmc"][qmc_key]
        out.append((r["field"], nm, ne, qm, qe))
    return out


def mark_flags(ax, rows):
    for r in rows:
        if r["flags"]:
            ax.axvline(r["field"], color="red", alpha=0.15, lw=8, zorder=0)


def plot_absolute(data, key_pairs, xlabel, suptitle, norm_by_N=False,
                  nqs_err_scale=1.0):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True, sharey=False)
    for ax, L in zip(axes, LS):
        rows = data[L]
        mark_flags(ax, rows)
        N = Counts(L, "OBC").N if norm_by_N else 1.0
        for nk, nek, qk, lab, mk in key_pairs:
            s = series(rows, nk, nek, qk)
            if not s:
                continue
            f, nm, ne, qm, qe = (np.array(x) for x in zip(*s))
            if norm_by_N:
                nm, ne, qm, qe = nm / N, ne / N, qm / N, qe / N
            lab_q = f"QMC {lab}" if lab else "QMC"
            lab_n = f"NQS {lab}" if lab else "NQS"
            if nqs_err_scale != 1.0:
                ne = ne * nqs_err_scale
                lab_n += rf" (bars $\times${nqs_err_scale:g})"
            ax.errorbar(f, nm, yerr=ne, fmt=mk, color=PLASMA[L], ms=5,
                        capsize=2, ls="none",
                        label=lab_n if ax is axes[0] else None, zorder=2)
            ax.errorbar(f, qm, yerr=qe, fmt=mk, mfc="none", color=QMC_COLOR, ms=5,
                        capsize=2, ls="none",
                        label=lab_q if ax is axes[0] else None, zorder=3)
        ax.set_title(f"L={L}")
        ax.set_xlabel(xlabel)
        if norm_by_N:
            ax.set_ylabel("$E/N$")
        openax(ax)
    axes[0].legend(fontsize=8)
    fig.suptitle(suptitle, y=1.03)
    fig.tight_layout()
    maybe_save(fig, suptitle)
    plt.close(fig)


def figs_up():
    up = assemble("up", UP_HZ, "z", HX_UP)
    plot_absolute(up, [("E0", "E_err", "energy", None, "o")],
                  r"$h_z$", "$h_z$ sweep — energy per spin", norm_by_N=True)
    plot_absolute(up, [("A_v_mean", "A_v_err", "star_x", "$A_v$", "o"),
                       ("B_p_mean", "B_p_err", "plaquette_z", "$B_p$", "s")],
                  r"$h_z$", "$h_z$ sweep — stabilizers")
    plot_absolute(up, [("sz_mean", "sz_err", "sigma_z", None, "o")],
                  r"$h_z$", r"$h_z$ sweep — $\langle\sigma_z\rangle$")
    plot_absolute(up, [("O_FM_paratoric", "O_FM_paratoric_err",
                        "fredenhagen_marcu", None, "o")],
                  r"$h_z$", "$h_z$ sweep — Z-string order parameter",
                  nqs_err_scale=3.0)


def figs_right():
    right = assemble("right", RIGHT_HX, "x", HZ_RIGHT)
    plot_absolute(right, [("E0", "E_err", "energy", None, "o")],
                  r"$h_x$", "$h_x$ sweep — energy per spin", norm_by_N=True)
    plot_absolute(right, [("A_v_mean", "A_v_err", "star_x", "$A_v$", "o"),
                          ("B_p_mean", "B_p_err", "plaquette_z", "$B_p$", "s")],
                  r"$h_x$", "$h_x$ sweep — stabilizers")
    plot_absolute(right, [("sx_mean", "sx_err", "sigma_x", None, "o")],
                  r"$h_x$", r"$h_x$ sweep — $\langle\sigma_x\rangle$")
    plot_absolute(right, [("O_FM_membrane_R1", "O_FM_membrane_R1_err",
                           "fredenhagen_marcu_membrane_r1", None, "o")],
                  r"$h_x$", "$h_x$ sweep — X-membrane R1 order parameter",
                  nqs_err_scale=3.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cut", choices=["up", "right", "both"], default="both")
    ap.add_argument("--no-save", action="store_true",
                    help="assemble and report landings without writing PNGs")
    args = ap.parse_args()
    if args.no_save:
        SAVE_FIGS = False
    if args.cut in ("up", "both"):
        figs_up()
    if args.cut in ("right", "both"):
        figs_right()
