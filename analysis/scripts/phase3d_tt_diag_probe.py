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
    python analysis/scripts/phase3d_tt_diag_probe.py --emit /tmp/tt_probe [--only hy1.6 ...]
    # A y-sweep at an (h_x, h_z) that already has a y-cut shares its run names:
    # sweep.py SKIPS points whose final JSON exists and warm-starts the next one
    # from them, so only the new (interleaved) h_y values train.
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
    # 2026-09-23 (user): continue the z-pol<->y-pol line outward. Centres from the located points (0.325 at 1.4,
    # ~0.375 at 1.5; slope dh_z/dh_y falling 1.7 -> 1.0 -> 0.5); dn anchor just above the window (at 1.5 it started
    # at 1.3 and spent 5 links deep in the z-polarized phase), up anchor unchanged (0.05, deep y-polarized).
    (1.6, 0.45, 0.25, 0.05, 0.80),
    (1.7, 0.50, 0.25, 0.05, 0.85),
    (1.8, 0.55, 0.25, 0.05, 0.90),
]

# ---- fixed-h_x planes h_x = 0.2 / 0.5 (user, 2026-09-23 evening): the same z-pol <-> y-pol h_z sweeps at h_y = 1.4/1.5
# off the h_x = 0 plane. h_x = 0 located 0.325 / 0.375; shift estimates: mean field +0.01 (0.2) / +0.07 (0.5), the
# h_x = 0.5 pocket tip (1.075, ~0.2) continued with the h_x = 0 slope +0.13-0.17 (user: ~+0.15). Windows cover both.
# (hx, hy, center, half_window, up_anchor, dn_anchor)
OFFAXIS_HZ_SWEEP_POINTS = [
    (0.2, 1.4, 0.35, 0.25, 0.05, 0.70),
    (0.2, 1.5, 0.40, 0.25, 0.05, 0.75),
    (0.5, 1.4, 0.50, 0.25, 0.05, 0.85),
    (0.5, 1.5, 0.55, 0.25, 0.05, 0.90),
]

# ---- h_z=0 plane: fix h_y (above the roof, max 1.185 at h_z=0), sweep h_x (user, 2026-09-23 afternoon): h_y-sweeps
# equilibrate slowly (the state must rotate x -> y), so trace the x-pol <-> y-pol line with h_x-sweeps whose trains
# start deep in their own polarized phase and never cross the topological region. Windows cover both estimates
# (mean-field h_x,c ~ h_y - 0.44; extrapolating the h_x=0.8/0.9 y-cut points gives ~1.1-1.16).
# (hy, center, half_window, up_anchor, dn_anchor)
HX_SWEEP_POINTS = [
    (1.3, 1.00, 0.30, 0.40, 1.60),
    (1.4, 1.10, 0.30, 0.50, 1.70),
]

# ---- h_z=0 plane, x<->y line out of the pocket TIP (user, 2026-09-24): h_y=1.3 is already a crossover and the h_y-cut
# points (h_x=0.8/0.9) are not trusted, so fixed-h_y h_x sweeps just above the tip (h_y~0.97, h_x~0.64). Up trains start
# just outside the roof (y-pol side), dn trains deep x-pol at h_x=1.4; windows on the 0.05 grid around the tip-slope-1
# estimate h_x ~ 0.64 + (h_y - 0.97). (hy, center, half_window, up_anchor, dn_anchor)
TIP_HX_SWEEP_POINTS = [
    (1.05, 0.800, 0.150, 0.60, 1.40),
    (1.10, 0.775, 0.175, 0.55, 1.40),
    (1.15, 0.800, 0.200, 0.50, 1.40),
    (1.20, 0.850, 0.200, 0.45, 1.40),
]

# ---- h_z=0 plane: fix h_x, sweep h_y over an EXPLICIT range (user's numbers,
# 2026-09-22 revision -- replaces the fix-h_y/sweep-h_x design for this plane).
# (hx, hy_lo, hy_hi, up_anchor, dn_anchor)
HY_SWEEP_POINTS = [
    (0.80, 0.60, 1.40, 0.50, 1.50),
    (0.90, 0.70, 1.50, 0.60, 1.60),
]

# ---- h_x = 0.2 plane roof rungs (user, 2026-09-23 evening), mirroring the h_x = 0 rungs at h_z = 0.1/0.15/0.2
# (located 1.18 / 1.245 / 1.28-remnant there; h_x = 0.2 at h_z = 0: 1.185, same as h_x = 0). Standard y-cut anchors
# (up 0.6 inside the pocket, dn 1.5 y-polarized), +-0.15 fine window on the 0.05 grid around the h_x = 0 values.
# (hx, hz, hy_lo, hy_hi, up_anchor, dn_anchor)
ROOF_YSWEEP_POINTS = [
    (0.2, 0.10, 1.05, 1.35, 0.6, 1.5),
    (0.2, 0.15, 1.05, 1.35, 0.6, 1.5),
    (0.2, 0.20, 1.10, 1.40, 0.6, 1.5),
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


def hz_sweep_spec(hy, center, half_window, up_anchor, dn_anchor, branch, hx=0.0):
    """Fixed (h_x, h_y), sweep h_z (mirrors _zchain_l4_job_spec); h_x = 0 unless given."""
    L = 4
    anchor = up_anchor if branch == "up" else dn_anchor
    lo, hi = round(center - half_window, 4), round(center + half_window, 4)
    field_values = [anchor] + broad_links(anchor, lo, hi, branch)
    cut = f"electric_hx{float(hx)}"
    jobname = f"p3d_hy{hy:g}_e{hx:g}_L{L}_{branch}"
    name_tpl = (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")
    env = {**_common_env(L, hy), "SWEEP": "hz", "HX": str(float(hx)), "HY": str(hy),
           "HZ": str(field_values[0]), "FIELD_VALUES": " ".join(str(h) for h in field_values),
           "CHUNK_POINTS": str(len(field_values)), "NAME_TEMPLATE": name_tpl,
           "WANDB_GROUP": jobname}
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": field_values, "env": env,
            "dependency": None, "walltime": WALLTIME, "array": "0",
            "out_dir_rel": f"hy{hy:g}/{cut}/L{L}"}


def hx_sweep_spec(hy, center, half_window, up_anchor, dn_anchor, branch):
    """h_z=0 plane: fixed h_z=0, sweep h_x at fixed h_y (mirrors _chain_l4_job_spec)."""
    L = 4
    anchor = up_anchor if branch == "up" else dn_anchor
    lo, hi = round(center - half_window, 4), round(center + half_window, 4)
    field_values = [anchor] + broad_links(anchor, lo, hi, branch)
    cut = "magnetic_hz0.0"
    jobname = f"p3d_hy{hy:g}_m0_L{L}_{branch}"
    name_tpl = (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")
    env = {**_common_env(L, hy), "SWEEP": "hx", "HZ": "0.0", "HY": str(hy),
           "HX": str(field_values[0]), "FIELD_VALUES": " ".join(str(h) for h in field_values),
           "CHUNK_POINTS": str(len(field_values)), "NAME_TEMPLATE": name_tpl,
           "WANDB_GROUP": jobname}
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": field_values, "env": env,
            "dependency": None, "walltime": WALLTIME, "array": "0",
            "out_dir_rel": f"hy{hy:g}/{cut}/L{L}"}


def hy_sweep_spec(hx, hy_lo, hy_hi, up_anchor, dn_anchor, branch, hz=0.0):
    """Fixed (h_x, h_z) (h_z = 0 unless given), sweep h_y over an explicit range
    (mirrors _ycut_l4_job_spec, but a caller-given range instead of the
    roof-formula/override centre -- (0.8,0.0) is one of the ORIGINAL 15
    y-cuts and its window is protected/frozen in phase3d_grid.py). Lands in
    the SAME ycuts/{ycut_id}/L4/ dir as any prior y-cut at this (h_x,h_z),
    so the viewer merges the points into one cut; jobname is prefixed
    p3d_yhc (not p3d_y) so it never collides with that prior chain's own
    (already-completed) job name."""
    L = 4
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
    for hx, hy, center, half_window, up_a, dn_a in OFFAXIS_HZ_SWEEP_POINTS:
        out[f"hy{hy:g}_hx{hx:g}"] = [hz_sweep_spec(hy, center, half_window, up_a, dn_a, br, hx=hx)
                                     for br in ("up", "dn")]
    for hy, center, half_window, up_a, dn_a in TIP_HX_SWEEP_POINTS:
        out[f"hy{hy:g}_tipx"] = [hx_sweep_spec(hy, center, half_window, up_a, dn_a, br) for br in ("up", "dn")]
    for hy, center, half_window, up_a, dn_a in HX_SWEEP_POINTS:
        out[f"hy{hy:g}_hxsweep"] = [hx_sweep_spec(hy, center, half_window, up_a, dn_a, br)
                                    for br in ("up", "dn")]
    for hx, hy_lo, hy_hi, up_a, dn_a in HY_SWEEP_POINTS:
        out[f"hx{hx:g}_hz0_ysweep"] = [hy_sweep_spec(hx, hy_lo, hy_hi, up_a, dn_a, br)
                                        for br in ("up", "dn")]
    for hx, hz, hy_lo, hy_hi, up_a, dn_a in ROOF_YSWEEP_POINTS:
        out[f"hx{hx:g}_hz{hz:g}_ysweep"] = [hy_sweep_spec(hx, hy_lo, hy_hi, up_a, dn_a, br, hz=hz)
                                           for br in ("up", "dn")]
    return out


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true", help="human-readable summary to stdout")
    p.add_argument("--emit", metavar="DIR", help="write one PLAN_FILE per point into DIR")
    p.add_argument("--only", nargs="+", metavar="LABEL",
                   help="restrict to these labels (e.g. hy1.6 hx1_hz0_ysweep) -- new diagonal steps")
    a = p.parse_args(argv)
    grid = build()
    if a.only:
        missing = set(a.only) - set(grid)
        if missing:
            raise SystemExit(f"unknown label(s) {sorted(missing)}; have {sorted(grid)}")
        grid = {k: v for k, v in grid.items() if k in a.only}
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
