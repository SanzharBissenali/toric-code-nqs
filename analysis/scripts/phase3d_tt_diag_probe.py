"""One-off probe: trivial<->trivial transitions off the h_z=0 and h_x=0 plane
corners. User request 2026-09-22 (see notes/phase3d_L4_plan.md RESUME block +
this session's log):
  - h_x=0 plane: fix h_y (a new value above the plane's roof), sweep h_z --
    the h_x=0 tip sits at (h_z=0.15, h_y=1.245, the landed y-cut (0,0.15)).
  - h_z=0 plane: the user looked at the first cut of this design (fix h_y,
    sweep h_x) and asked for the ORTHODOX y-cut direction instead -- fix h_x,
    sweep h_y over an explicit range (not the roof-formula/override system,
    which only has fixed windows for the pre-existing y-cut points).
Both are above/beyond every known roof height on their plane, so the whole
swept range is trivial-vs-trivial (no topological order anywhere in it).

No existing helper in phase3d_grid.py does either directly: the magnetic/
electric chain tables are keyed by h_z/h_x only (shared across all of
HY_VALUES), and the y-cut machinery's per-point window override
(YCUT_CENTER_OVERRIDE) is skipped entirely for any (h_x,h_z) in the original
15 -- (0.8, 0.0) is one of them, so it cannot be re-centred there without
touching that protected set. This stays a standalone script per the
established PLAN_FILE precedent (see the 2026-09-21/22 log entries in
notes/phase3d_L4_plan.md) rather than bending the state-driven planner or the
y-cut window/anchor tables. Emits nersc/launch_phase3d.sh PLAN_FILE lines;
nothing here touches phase3d_grid.py's tested cut registry/self-tests.

    python analysis/scripts/phase3d_tt_diag_probe.py --dry
    python analysis/scripts/phase3d_tt_diag_probe.py --emit /tmp/tt_probe
    # then, per file, on the cluster. HY= MUST match the rows: the launcher
    # writes it into every manifest row's hy column (the planner dedupes on it)
    # -- the plane value for hy*.tsv, `y` for the *_ysweep.tsv files:
    PLAN_FILE=/tmp/tt_probe/hy1.4.tsv HY=1.4 bash nersc/launch_phase3d.sh
    PLAN_FILE=/tmp/tt_probe/hx0.8_hz0_ysweep.tsv HY=y bash nersc/launch_phase3d.sh
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from phase3d_grid import (arch_env, speed_env, diag_shift_for, kernel_for,
                           SNAP_ARGS, WANDB_PROJECT_VAL, _bash_line, ycut_id)

FINE = 0.05
COARSE = 0.1
WALLTIME = "05:00:00"          # capped at the campaign's 5h max; AUTO_RESUBMIT finishes the rest
ANCHOR_OV = '{"dt":0.01,"lr_min":0.002,"n_iter":1000,"diag_shift":1e-2}'   # gentle: every anchor here
    # starts deep in an unexplored, possibly strongly-polarized corner -- treat it like the campaign's
    # existing h_y>=0.6 dn/y-polarized anchors, not a cold-topological one.

# ---- h_x=0 plane: fix h_y (new value, above the plane's roof), sweep h_z ---
# (hy, center, half_window, up_anchor, dn_anchor)
HZ_SWEEP_POINTS = [
    (1.4, 0.40, 0.30, 0.05, 1.2),   # above the tip (h_z=0.15, h_y=1.245)
    (1.5, 0.55, 0.30, 0.05, 1.3),   # shifted diagonally up in h_z
]

# ---- h_z=0 plane: fix h_x, sweep h_y over an EXPLICIT range (user's numbers,
# 2026-09-22 revision -- replaces the fix-h_y/sweep-h_x design for this plane).
# (hx, hy_lo, hy_hi, up_anchor, dn_anchor)
HY_SWEEP_POINTS = [
    (0.80, 0.60, 1.40, 0.50, 1.50),
    (0.90, 0.70, 1.50, 0.60, 1.60),
]


def broad_links(anchor, lo, hi, branch):
    """Same two-tier rule as phase3d_grid._links (0.05 fine inside [lo,hi],
    <=0.1 coarse from the anchor to that window's edge, anchor excluded)."""
    n_fine = int(round((hi - lo) / FINE)) + 1
    fine = [round(lo + k * FINE, 4) for k in range(n_fine)]
    coarse = []
    if branch == "up":
        h = anchor + COARSE
        while h < lo - 1e-9:
            coarse.append(round(h, 4)); h += COARSE
        return coarse + fine
    h = anchor - COARSE
    while h > hi + 1e-9:
        coarse.append(round(h, 4)); h -= COARSE
    return coarse + fine[::-1]


def _common_env(L, hy_for_speed):
    return {**arch_env(L), **speed_env(L, hy_for_speed), "L": str(L),
            "WARM_START": "1", "ANCHOR_OVERRIDES": ANCHOR_OV,
            "DT": "0.005", "LR_MIN": "0.0005", "DIAG_SHIFT": "3e-3", "N_ITER": "300",
            "CKPT_EVERY": "10", "EXTRA_ARGS": SNAP_ARGS,
            "WANDB_PROJECT": WANDB_PROJECT_VAL, "AUTO_RESUBMIT": "1",
            "CHUNK": "2048", "TOPO_POOLED": "1"}


def hz_sweep_spec(hy, center, half_window, up_anchor, dn_anchor, branch):
    """h_x=0 plane: fixed h_x=0, sweep h_z at fixed h_y (mirrors _zchain_l4_job_spec)."""
    L = 4
    anchor = up_anchor if branch == "up" else dn_anchor
    lo, hi = round(center - half_window, 4), round(center + half_window, 4)
    field_values = [anchor] + broad_links(anchor, lo, hi, branch)
    cut = "electric_hx0.0"
    jobname = f"p3d_hy{hy:g}_e0_L{L}_{branch}"
    name_tpl = (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")
    env = {**_common_env(L, hy), "SWEEP": "hz", "HX": "0.0", "HY": str(hy),
           "HZ": str(field_values[0]), "FIELD_VALUES": " ".join(str(h) for h in field_values),
           "CHUNK_POINTS": str(len(field_values)), "NAME_TEMPLATE": name_tpl,
           "WANDB_GROUP": jobname}
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": field_values, "env": env,
            "dependency": None, "walltime": WALLTIME, "array": "0",
            "out_dir_rel": f"hy{hy:g}/{cut}/L{L}"}


def hy_sweep_spec(hx, hy_lo, hy_hi, up_anchor, dn_anchor, branch):
    """h_z=0 plane: fixed (h_x, h_z=0), sweep h_y over an explicit range
    (mirrors _ycut_l4_job_spec, but a caller-given range instead of the
    roof-formula/override centre -- (0.8,0.0) is one of the ORIGINAL 15
    y-cuts and its window is protected/frozen in phase3d_grid.py). Lands in
    the SAME ycuts/{ycut_id}/L4/ dir as any prior y-cut at this (h_x,h_z),
    so the viewer merges the points into one cut; jobname is prefixed
    p3d_yhc (not p3d_y) so it never collides with that prior chain's own
    (already-completed) job name."""
    L = 4
    hz = 0.0
    anchor = up_anchor if branch == "up" else dn_anchor
    field_values = [anchor] + broad_links(anchor, hy_lo, hy_hi, branch)
    cut = ycut_id(hx, hz)
    jobname = f"p3d_yhc_hx{hx:g}_hz{hz:g}_L{L}_{branch}"
    name_tpl = (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")
    env = {**_common_env(L, field_values[0]), "SWEEP": "hy", "HX": str(hx), "HZ": str(hz),
           "HY": str(field_values[0]), "FIELD_VALUES": " ".join(str(h) for h in field_values),
           "CHUNK_POINTS": str(len(field_values)), "NAME_TEMPLATE": name_tpl,
           "WANDB_GROUP": jobname}
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": field_values, "env": env,
            "dependency": None, "walltime": WALLTIME, "array": "0",
            "out_dir_rel": f"ycuts/{cut}/L{L}"}


def build():
    """{label: [spec_up, spec_dn]}"""
    out = {}
    for hy, center, half_window, up_a, dn_a in HZ_SWEEP_POINTS:
        out[f"hy{hy:g}"] = [hz_sweep_spec(hy, center, half_window, up_a, dn_a, br)
                             for br in ("up", "dn")]
    for hx, hy_lo, hy_hi, up_a, dn_a in HY_SWEEP_POINTS:
        out[f"hx{hx:g}_hz0_ysweep"] = [hy_sweep_spec(hx, hy_lo, hy_hi, up_a, dn_a, br)
                                        for br in ("up", "dn")]
    return out


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true", help="human-readable summary to stdout")
    p.add_argument("--emit", metavar="DIR", help="write one PLAN_FILE per point into DIR")
    a = p.parse_args(argv)
    grid = build()
    if a.emit:
        os.makedirs(a.emit, exist_ok=True)
        for label, specs in grid.items():
            fp = os.path.join(a.emit, f"{label}.tsv")
            with open(fp, "w") as f:
                for s in specs:
                    f.write(_bash_line(s) + "\n")
            print(f"wrote {fp} ({len(specs)} job rows)")
    if a.dry or not a.emit:
        for label, specs in grid.items():
            for s in specs:
                n = len(s["h_list"])
                lo, hi = min(s["h_list"]), max(s["h_list"])
                print(f"{label:<20} {s['jobname']:<26} cut={s['cut']:<16} "
                      f"{n} field points, range [{lo}, {hi}], walltime={s['walltime']}")


if __name__ == "__main__":
    main(sys.argv[1:])
