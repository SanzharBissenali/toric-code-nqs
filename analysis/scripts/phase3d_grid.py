"""Deterministic grid for the phase3d campaign: h_y planes HY_VALUES = {0.0..1.0}
(+ the y-cut pseudo-plane "y": sweep h_y at fixed (h_x, h_z)) x electric cuts (fixed
h_x, sweep h_z), magnetic cuts (fixed h_z <= TOPO_HZ_MAX, sweep h_x) and
trivial-trivial tail cuts (fixed h_z > TOPO_HZ_MAX). The grid math still supports
L in L_VALUES = {4,5,6}, but only the first h_y=0 electric lines ran at L=4/5/6:
since 2026-09-17 the campaign is L=4-only multi-plane mapping
(notes/phase3d_L4_plan.md). Pure python, NetKet-free -- consumed by
nersc/launch_phase3d.sh and inspectable standalone. Fixed campaign decisions (do
not change without re-deriving the physics): see
notes/transition_mapping_recipes.md and notes/phase3d_campaign.md.

    python analysis/scripts/phase3d_grid.py --dry                     # human dump
    python analysis/scripts/phase3d_grid.py --json                    # same, JSON
    python analysis/scripts/phase3d_grid.py --dry --hy 0.0 --L 4       # filtered
    python analysis/scripts/phase3d_grid.py --refs                    # hy=0 QMC ref table
    python analysis/scripts/phase3d_grid.py --ref_lookup --hx 0.2 --hz 0.26 --L 4

Self-tests (grid math only, no filesystem) run automatically before any action.
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import shlex
import shutil
import sys

HY_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
L_VALUES = [4, 5, 6]
ELECTRIC_HX = [0.0, 0.2, 0.25, 0.5, 0.65, 0.8]   # 0.65 (2026-09-17): pins the corner with magnetic_hz0.25; 0.25 (2026-09-19, hy=0.8 only)
MAGNETIC_HZ = [0.0, 0.1, 0.2, 0.25]           # topological -> trivial (membrane O_FM primary), hz <= TOPO_HZ_MAX
TAIL_HZ = [0.4, 0.7, 0.85, 1.0]               # trivial -> trivial (M_x jump primary); 0.85 brackets the line's endpoint
TOPO_HZ_MAX = 0.3                             # magnetic cuts at hz <= this are topological -> trivial
DEFAULT_SKIP_CUTS = {"electric_hx0.2", "magnetic_hz0.1", "electric_hx0.25"}   # 0.2/hz0.1: older campaign at hy=0/0.2/0.4; 0.25: hy=0.8 only, explicit CUTS

# ---- electric (2nd-order) grid: 7 points, center + fixed offsets, rounded 0.01 ---
ELECTRIC_OFFSETS = [-0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12]
_BASE_L = {4: 0.30, 5: 0.27, 6: 0.26}
_DHX = {0.0: 0.00, 0.2: 0.00, 0.25: 0.005, 0.5: 0.02, 0.65: 0.04, 0.8: 0.08}
_DHY = {0.0: 0.0, 0.2: -0.006, 0.4: -0.023,     # empirical (L4 fits), ~ -0.14*hy^2 ...
        0.6: -0.05, 0.8: -0.09, 1.0: -0.14}     # ... extrapolated for the new planes; the refine round corrects


def electric_center(hx, L, hy):
    return _BASE_L[L] + _DHX[hx] + _DHY[hy]


def electric_grid(hx, L, hy):
    c = electric_center(hx, L, hy)
    return [round(c + d, 2) for d in ELECTRIC_OFFSETS]


# ---- magnetic/tail (1st-order) warm-chain links -----------------------------
# Rule (user, 2026-09-17): 0.05 spacing across a +-CHAIN_HALF_WINDOW window around
# the cut's seeded crossing, <= 0.1 spacing everywhere else (anchor -> first link
# included), and BOTH branches cover the whole window so they overlap there. The
# up branch walks from the low anchor to the window's top, the dn branch from the
# high anchor to the window's bottom. Links train 300 steps (was 200).
_ANCHORS = {0.0: (0.6, 1.25), 0.1: (0.6, 1.25), 0.2: (0.6, 1.25), 0.25: (0.6, 1.25),
            0.4: (0.5, 1.3), 0.7: (0.6, 1.5), 0.85: (0.7, 1.7), 1.0: (0.8, 1.7)}
# seeded crossing per cut, on the 0.05 grid (L4 planes 0/0.2/0.4: envelope 0.80-0.83,
# hz=0.4 jump 0.83-0.89, hz=0.7 jump 1.20; hz=1.0 is a crossover centred ~1.45)
_CENTERS = {0.0: 0.85, 0.1: 0.85, 0.2: 0.85, 0.25: 0.85, 0.4: 0.85, 0.7: 1.2, 0.85: 1.3, 1.0: 1.45}
CHAIN_HALF_WINDOW = 0.15
CHAIN_FINE = 0.05
CHAIN_COARSE = 0.1


def chain_window(hz):
    """(lo, hi) of the 0.05-spaced window around the cut's seeded crossing."""
    c = _CENTERS[round(hz, 4)]
    return round(c - CHAIN_HALF_WINDOW, 4), round(c + CHAIN_HALF_WINDOW, 4)


def chain_links(hz, branch):
    """branch: 'up' (low anchor; coarse 0.1 steps up to the window, then the full
    0.05 window ascending) or 'dn' (high anchor; coarse steps down to the window,
    then the full window descending). Every consecutive gap, anchor included, is
    <= CHAIN_COARSE; the window is identical on both branches."""
    lo, hi = _ANCHORS[round(hz, 4)]
    redo = _up_redo(hz)
    if branch == "up" and redo:                    # one 0.05 ladder from the deep anchor to the top of the window
        a, top = redo
        n = int(round((top - a) / CHAIN_FINE))
        return [round(a + k * CHAIN_FINE, 4) for k in range(1, n + 1)]
    wlo, whi = chain_window(hz)
    n_fine = int(round((whi - wlo) / CHAIN_FINE)) + 1
    fine = [round(wlo + k * CHAIN_FINE, 4) for k in range(n_fine)]
    coarse = []
    if branch == "up":
        h = lo + CHAIN_COARSE
        while h < wlo - 1e-9:
            coarse.append(round(h, 4)); h += CHAIN_COARSE
        return coarse + fine
    h = hi - CHAIN_COARSE
    while h > whi + 1e-9:
        coarse.append(round(h, 4)); h -= CHAIN_COARSE
    return coarse + fine[::-1]


# Planes where the topological lobe has shrunk to h_x ~ 0.6 (h_y = 0.8: up chains seeded at 0.6 start AT the
# boundary and the transition is not bracketed). User decision 2026-09-19: REDO the up chains of the topological
# magnetic cuts (h_z <= TOPO_HZ_MAX) from deep inside the lobe -- anchor 0.45, 0.05-spaced links up to 0.95.
_PLANE_UP_REDO = {0.8: (0.45, 0.95), 1.0: (0.45, 0.95)}
_ACTIVE_HY = None                                  # set by plan()/retry_spec: the plane being planned


def _up_redo(hz):
    """(anchor, hi) of the redone up chain on the active plane, or None."""
    if _ACTIVE_HY is None or not isinstance(_ACTIVE_HY, float):
        return None
    span = _PLANE_UP_REDO.get(round(_ACTIVE_HY, 4))
    return span if span and hz <= TOPO_HZ_MAX else None


def chain_anchor(hz, branch):
    lo, hi = _ANCHORS[round(hz, 4)]
    redo = _up_redo(hz)
    if branch == "up" and redo:
        return redo[0]
    return lo if branch == "up" else hi


# ---- y-cuts: sweep hy at fixed (hx, hz) -- notes/phase3d_L4_plan.md SC ------------
# Roof of the topological lobe (hx in {0, 0.5, 0.8} x hz in {0, 0.1, 0.2}), the
# y/z-polarized first-order line (hx=0, hz above the electric wall) and the
# y/x-polarized one (hz=0, hx beyond the magnetic wall). Same link rule as the
# magnetic chains; the pseudo-plane HY=y carries them (their own results dir
# ycuts/, manifest hy column "y").
YCUT_POINTS = [(0.0, 0.0), (0.0, 0.1), (0.0, 0.2), (0.5, 0.0), (0.5, 0.1), (0.5, 0.2),
               (0.8, 0.0), (0.8, 0.1), (0.8, 0.2),
               (0.0, 0.4), (0.0, 0.55), (0.0, 0.7),
               (1.0, 0.0), (1.2, 0.0), (1.4, 0.0),
               # 2026-09-21 (user): the h_x = 0 and h_z = 0 planes as their own maps -- roof rungs at h_z = 0.05/0.15
               # (h_x = 0) and h_x = 0.2/0.4/0.6 (h_z = 0); fine window centred on the spherical-roof estimate
               (0.0, 0.05), (0.0, 0.15), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0),
               # 2026-09-22 (user): trace the trivial->trivial lines leaving each plane's pocket corner, now that
               # both planes are mapped. h_z=0 (x-pol<->y-pol): hx=0.8 jump=0.95 (ok), hx=1.0/1.2/1.4 jump=0.95 but
               # NOT sharp (ok=False) with a shrinking-but-nonzero M_y split (0.09/0.055/0.027) -- the line survives
               # weakly past 1.4, not a clean endpoint; hx=0.7 pins the corner (hx,c(hy) at hz=0 is ~0.67-0.68 near
               # hy~0.9-1.0, so 0.7 sits right at the roof/line boundary) and hx=0.9 brackets where the jump becomes
               # sharp again vs. hx=0.8. h_x=0 (z-pol<->y-pol): only one confirmed point so far, hz=0.2 -> 1.275; at
               # hz=0.4/0.55/0.7 the branches are FULLY merged (sep 0.003-0.018, not just "near the window edge") ->
               # the line likely ends between hz=0.2 and 0.4. hz=0.25/0.3 bracket that endpoint; centred higher than
               # 1.275 (by analogy with the x<->z line, whose h_x,c rises toward its own endpoint) and given an
               # EXTENDED dn anchor (referee, 2026-09-21: "z<->y sheet needs windows to h_y ~ 2.0 at h_z >= 0.3").
               (0.7, 0.0), (0.9, 0.0), (0.0, 0.25), (0.0, 0.3)]
# Trivial-trivial line probes (not roof points): explicit window centres (the roof-sphere formula does not apply
# off the roof) and, for the two h_x=0 points, an extended y-polarized anchor to reach past h_y = 1.5.
YCUT_CENTER_OVERRIDE = {(0.7, 0.0): 0.97, (0.9, 0.0): 1.05, (0.0, 0.25): 1.4, (0.0, 0.3): 1.55}
YCUT_DN_ANCHOR_OVERRIDE = {(0.0, 0.25): 2.0, (0.0, 0.3): 2.0}
_YCUT_ORIG = set(YCUT_POINTS[:15])         # submitted with the fixed centre YCUT_CENTER; keep their windows stable
YCUT_ROOF_R = 1.18                         # referee 2026-09-21: the pocket roof is close to a sphere |h| ~ 1.18 at L=4
YCUT_ANCHORS = (0.6, 1.5)          # up: inside the lobe; dn: y-polarized
YCUT_CENTER = 1.15                 # axis estimate 1.16 (cold points, branch crossing)
YCUT_HY = "y"                      # the pseudo-plane's HY value (launcher, manifests, results dir ycuts/)


def ycut_id(hx, hz):
    return f"ycut_hx{hx:g}_hz{hz:g}"


def _links(lo, hi, c, branch):
    """The chain-link rule around centre c between anchors lo/hi (see chain_links)."""
    wlo, whi = round(c - CHAIN_HALF_WINDOW, 4), round(c + CHAIN_HALF_WINDOW, 4)
    n_fine = int(round((whi - wlo) / CHAIN_FINE)) + 1
    fine = [round(wlo + k * CHAIN_FINE, 4) for k in range(n_fine)]
    coarse = []
    if branch == "up":
        h = lo + CHAIN_COARSE
        while h < wlo - 1e-9:
            coarse.append(round(h, 4)); h += CHAIN_COARSE
        return coarse + fine
    h = hi - CHAIN_COARSE
    while h > whi + 1e-9:
        coarse.append(round(h, 4)); h -= CHAIN_COARSE
    return coarse + fine[::-1]


def ycut_center(hx, hz):
    """Fine-window centre of a y-cut: the original 15 keep YCUT_CENTER (their links are already in the
    manifests); an explicit trivial-trivial-line guess wins next; otherwise new (roof) cuts use the
    spherical-roof estimate sqrt(R^2 - hx^2 - hz^2), floored at 0.95."""
    key = (round(float(hx), 4), round(float(hz), 4))
    if key in _YCUT_ORIG:
        return YCUT_CENTER
    if key in YCUT_CENTER_OVERRIDE:
        return YCUT_CENTER_OVERRIDE[key]
    return round(max(0.95, math.sqrt(max(0.0, YCUT_ROOF_R ** 2 - hx ** 2 - hz ** 2))), 2)


def ycut_links(branch, hx=0.0, hz=0.0):
    return _links(ycut_anchor("up", hx, hz), ycut_anchor("dn", hx, hz), ycut_center(hx, hz), branch)


def ycut_anchor(branch, hx=0.0, hz=0.0):
    if branch == "dn":
        return YCUT_DN_ANCHOR_OVERRIDE.get((round(float(hx), 4), round(float(hz), 4)), YCUT_ANCHORS[1])
    return YCUT_ANCHORS[0]


def is_ycut_plane(hy):
    return str(hy) == YCUT_HY


# ---- cut registry -----------------------------------------------------------
def all_cuts():
    """[(cut_id, kind, fixed_value), ...] -- kind in {electric, magnetic}
    (tail cuts are `kind=magnetic` too; only the hz value distinguishes them,
    per the task's OUT_DIR/naming convention: both live under magnetic_hz{hz})."""
    cuts = [(f"electric_hx{hx}", "electric", hx) for hx in ELECTRIC_HX]
    cuts += [(f"magnetic_hz{hz}", "magnetic", hz) for hz in MAGNETIC_HZ + TAIL_HZ]
    return cuts


def all_ycuts():
    """[(cut_id, "ycut", (hx, hz)), ...] -- the HY=y pseudo-plane's cuts."""
    return [(ycut_id(hx, hz), "ycut", (hx, hz)) for hx, hz in YCUT_POINTS]


_CUTS_BY_ID = {cid: (kind, val) for cid, kind, val in all_cuts() + all_ycuts()}
PLANE_CUT_IDS = [c for c, _, _ in all_cuts()]
YCUT_IDS = [c for c, _, _ in all_ycuts()]


def points_for(cut_id, L, hy):
    """One (cut, L, hy) cell's submission plan.
    electric -> {"kind": "electric", "hx": hx, "hz_points": [7 floats]}
    magnetic -> {"kind": "magnetic", "hz": hz,
                 "up": {"anchor": hx, "links": [...]},
                 "dn": {"anchor": hx, "links": [...]}}
    """
    kind, val = _CUTS_BY_ID[cut_id]
    if kind == "electric":
        return {"kind": "electric", "hx": val, "hz_points": electric_grid(val, L, hy)}
    if kind == "ycut":
        return {"kind": "ycut", "hx": val[0], "hz": val[1],
                "up": {"anchor": ycut_anchor("up", *val), "links": ycut_links("up", *val)},
                "dn": {"anchor": ycut_anchor("dn", *val), "links": ycut_links("dn", *val)}}
    return {"kind": "magnetic", "hz": val,
            "up": {"anchor": chain_anchor(val, "up"), "links": chain_links(val, "up")},
            "dn": {"anchor": chain_anchor(val, "dn"), "links": chain_links(val, "dn")}}


def campaign_grid(hys=None, ls=None, cut_ids=None):
    hys = HY_VALUES if hys is None else hys
    ls = L_VALUES if ls is None else ls
    cut_ids = [c for c, _, _ in all_cuts()] if cut_ids is None else cut_ids
    return {hy: {cid: {L: points_for(cid, L, hy) for L in ls}
                 for cid in (YCUT_IDS if is_ycut_plane(hy) else cut_ids)}
            for hy in hys}


def job_count(grid):
    """(n_jobs, n_points) -- electric: 1 job/point; magnetic: 2 jobs (up/dn) per
    cell, 1 point-row per link+anchor in each (matches the manifest convention:
    chain jobs get one row per point, same jobid)."""
    n_jobs = n_points = 0
    for cuts in grid.values():
        for cells in cuts.values():
            for cell in cells.values():
                if cell["kind"] == "electric":
                    n_jobs += len(cell["hz_points"])
                    n_points += len(cell["hz_points"])
                else:
                    n_jobs += 2
                    n_points += (1 + len(cell["up"]["links"])) + (1 + len(cell["dn"]["links"]))
    return n_jobs, n_points


# ---- QMC hy=0 reference lookup (glob+parse, never format-and-lookup) -------
def build_refs(results_root="results"):
    """{(L, hx, hz): {"E", "E_err", "beta", "source"}} -- highest-beta subset only,
    read straight from each engine JSON's own (L,hx,hz,beta,E,E_err), never from
    the (non-uniform) qmc_hx*_hz* directory name."""
    best = {}
    for d in sorted(glob.glob(os.path.join(results_root, "qmc_hx*_hz*"))):
        for fp in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                doc = json.load(open(fp))
            except (OSError, json.JSONDecodeError):
                continue
            # single-beta run files only -- a beta-ladder *_combined.json (list-valued
            # beta) is the forbidden equal-weight-across-beta combine (see
            # notes/transition_mapping_recipes.md #0: "never equal-weight-combine
            # QMC files across beta"); skip anything not shaped like one run.
            if not isinstance(doc, dict) or not all(
                isinstance(doc.get(k), (int, float)) for k in
                ("L", "hx", "hz", "beta", "E", "E_err")
            ):
                continue
            key = (int(doc["L"]), round(float(doc["hx"]), 4), round(float(doc["hz"]), 4))
            beta = float(doc["beta"])
            cur = best.get(key)
            if cur is None or beta > cur["beta"]:
                best[key] = {"L": key[0], "hx": key[1], "hz": key[2], "beta": beta,
                             "E": float(doc["E"]), "E_err": float(doc["E_err"]),
                             "source": os.path.relpath(fp, results_root)}
    return best


def ref_lookup(refs, hx, hz, L):
    return refs.get((int(L), round(float(hx), 4), round(float(hz), 4)))


# ---- self-tests (grid math only; no filesystem) ----------------------------
def _selftest():
    assert electric_grid(0.2, 4, 0.0) == [0.18, 0.24, 0.27, 0.30, 0.33, 0.36, 0.42]
    assert len(electric_grid(0.8, 6, 0.4)) == 7
    for hx in ELECTRIC_HX:
        for L in L_VALUES:
            for hy in HY_VALUES:
                g = electric_grid(hx, L, hy)
                assert len(g) == 7 and g == sorted(g), (hx, L, hy, g)

    for hz in MAGNETIC_HZ + TAIL_HZ:
        up, dn = chain_links(hz, "up"), chain_links(hz, "dn")
        assert up == sorted(up) and dn == sorted(dn, reverse=True), (hz, up, dn)
        lo, hi = _ANCHORS[hz]
        wlo, whi = chain_window(hz)
        assert lo < min(up) and abs(max(up) - whi) < 1e-9, (hz, lo, up, whi)
        assert abs(min(dn) - wlo) < 1e-9 and max(dn) < hi, (hz, wlo, dn, hi)
        for seq, anchor in ((up, lo), (dn, hi)):                 # every gap <= 0.1, anchor included
            gaps = [abs(b_ - a_) for a_, b_ in zip([anchor] + seq, seq)]
            assert max(gaps) <= CHAIN_COARSE + 1e-9, (hz, seq, gaps)
        fine_up = {h for h in up if wlo - 1e-9 <= h <= whi + 1e-9}
        fine_dn = {h for h in dn if wlo - 1e-9 <= h <= whi + 1e-9}
        assert len(fine_up) == 7 and fine_up == fine_dn, (hz, fine_up, fine_dn)   # both branches cover the window
    assert chain_links(0.7, "up") == [0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]
    assert chain_links(0.7, "dn") == [1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05]
    assert 0.75 in chain_links(0.0, "up") and 0.75 in chain_links(0.0, "dn")      # A2 inserts
    assert 1.2 in chain_links(0.7, "up") and 1.2 in chain_links(0.7, "dn")        # A1 inserts

    assert len(all_cuts()) == 14 and len(all_ycuts()) == 24   # 6 electric + 8 magnetic; 15 + 5 + 4 y-cuts
    assert ycut_center(0.7, 0.0) == 0.97 and ycut_center(0.0, 0.3) == 1.55
    assert ycut_anchor("dn", 0.0, 0.3) == 2.0 and ycut_anchor("dn", 0.7, 0.0) == 1.5 and ycut_anchor("up", 0.0, 0.3) == 0.6
    assert ycut_links("dn", 0.0, 0.3)[0] == 1.9
    assert ycut_center(0.0, 0.0) == YCUT_CENTER and ycut_center(0.6, 0.0) == 1.02 and ycut_center(0.0, 0.15) == 1.17
    assert ycut_links("up", 0.6, 0.0) == [0.7, 0.8, 0.87, 0.92, 0.97, 1.02, 1.07, 1.12, 1.17]
    assert ycut_links("dn", 0.6, 0.0) == [1.4, 1.3, 1.2, 1.17, 1.12, 1.07, 1.02, 0.97, 0.92, 0.87]
    assert zchain_links("up")[0] == 0.05 and zchain_links("dn")[0] == 0.4 and len(zchain_links("up")) == 12
    assert ycut_links("up")[-1] == round(YCUT_CENTER + CHAIN_HALF_WINDOW, 4) and ycut_links("dn")[-1] == round(YCUT_CENTER - CHAIN_HALF_WINDOW, 4)
    assert ycut_links("up") == [0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    assert ycut_links("dn") == [1.4, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1.0]
    assert points_for("ycut_hx0_hz0", 4, "y")["kind"] == "ycut"
    yidx = submitted_index([{"hy": "y", "cut": "ycut_hx0_hz0", "L": "4", "role": "chain_up", "h": "0.6"}])
    assert already_submitted(yidx, "y", "ycut_hx0_hz0", 4, "chain_up", 0.6)
    assert not already_submitted(yidx, "y", "ycut_hx0_hz0", 4, "chain_dn", 1.5)
    cids = {c for c, _, _ in all_cuts()}
    assert cids == {
        "electric_hx0.0", "electric_hx0.2", "electric_hx0.25", "electric_hx0.5", "electric_hx0.65", "electric_hx0.8",
        "magnetic_hz0.0", "magnetic_hz0.1", "magnetic_hz0.2", "magnetic_hz0.25",
        "magnetic_hz0.4", "magnetic_hz0.7", "magnetic_hz0.85", "magnetic_hz1.0"}

    # one hy plane, default (11) cuts: 4 electric hx * 3 L * 7 pts = 84 electric jobs;
    # 7 magnetic/tail hz * 3 L * 2 chains = 42 chain jobs -> 126 jobs/plane.
    default_cuts = [c for c in cids if c not in DEFAULT_SKIP_CUTS]
    g = campaign_grid(hys=[0.0], ls=L_VALUES, cut_ids=default_cuts)
    n_jobs, n_points = job_count(g)
    assert n_jobs == 126, n_jobs
    n_chain_pts = sum(3 * (2 + len(chain_links(hz, "up")) + len(chain_links(hz, "dn")))
                      for hz in MAGNETIC_HZ + TAIL_HZ if f"magnetic_hz{hz}" in default_cuts)
    assert n_points == 84 + n_chain_pts, (n_points, n_chain_pts)
    # stderr only -- --refs/--ref_lookup/--json emit machine-readable stdout
    print("[selftest] ok (grid math)", file=sys.stderr, flush=True)


def _selftest_plan():
    """Pure, filesystem-free tests for the state-driven planner: the three
    plan-time refusals + idempotence primitives + the union-window property.
    (b)'s full cross-branch guarantee -- that the CALLER never hands the other
    branch's points to this pool -- is exercised by the filesystem-based
    "moments" test, since it depends on how `plan()` scopes the manifest index."""
    assert refuse_cold_inside_window(0.9, (0.7, 1.15)) is True
    assert refuse_cold_inside_window(0.6, (0.7, 1.15)) is False       # the anchor itself
    assert refuse_cold_inside_window(1.15, (0.7, 1.15)) is False      # boundary != inside

    assert nearest_same_branch_checkpoint([0.6, 0.7], 0.72) == 0.7
    assert nearest_same_branch_checkpoint([], 0.72) is None           # empty pool -> refuse
    assert nearest_same_branch_checkpoint(None, 0.72) is None

    class _FakeTable:
        def __init__(self, h, diverged):
            self.h, self.diverged = h, diverged

    t = _FakeTable([0.7, 0.8, 0.85, 0.9], [False, False, True, False])
    cutoff = spinodal_cutoff(branch_points_by_distance(t, anchor=0.6))
    assert cutoff == 0.85, cutoff
    assert beyond_spinodal(0.9, 0.6, cutoff) is True                  # past the crash -> refuse
    assert beyond_spinodal(0.85, 0.6, cutoff) is True                 # the crash point -> refuse
    assert beyond_spinodal(0.8, 0.6, cutoff) is False                 # before it -> fine
    assert beyond_spinodal(0.5, 0.6, cutoff) is False                 # other side -> unaffected
    assert spinodal_cutoff(branch_points_by_distance(None, 0.6)) is None

    rows = [{"hy": "0.0", "cut": "electric_hx0.0", "L": "4", "role": "cold", "h": "0.18"}]
    idx = submitted_index(rows)
    assert already_submitted(idx, 0.0, "electric_hx0.0", 4, "cold", 0.18)
    assert not already_submitted(idx, 0.0, "electric_hx0.0", 4, "cold", 0.24)
    assert not already_submitted(idx, 0.2, "electric_hx0.0", 4, "cold", 0.18)    # hy-scoped

    # union window (addendum #1): recentring may only ADD coverage, never drop
    # the plain seed -- checked at both a "same side" and a large L6 shift.
    seed = set(chain_links(0.4, "up"))
    for h_c4 in (0.7, 1.1, None):
        assert seed <= set(chain_link_window(0.4, 5, h_c4))
    assert len(set(chain_link_window(0.4, 6, 1.3)) - seed) > 0, \
        "expected the L6 recentre to add at least one new link"

    # rule 4: electric refine round -- bracketed h_c gets symmetric +-step;
    # an h_c outside the fit's own observed h-range (unbracketed rise) gets
    # ONE point extending outward past the nearest edge instead.
    pts, unb = electric_refine_points(0.30, (0.18, 0.42))
    assert not unb and pts == [0.28, 0.32], pts
    pts, unb = electric_refine_points(0.50, (0.18, 0.42))
    assert unb and pts == [0.44], pts                  # past the high edge -> +step
    pts, unb = electric_refine_points(0.10, (0.18, 0.42))
    assert unb and pts == [0.16], pts                  # past the low edge -> -step
    pts, unb = electric_refine_points(0.18, (0.18, 0.42))
    assert not unb and pts == [0.16, 0.20], pts         # boundary itself counts as bracketed

    print("[selftest] ok (plan/refusals)", file=sys.stderr, flush=True)


def _selftest_health_and_clamps():
    """CRUCIAL #2 (checkpoint_health) + MEDIUM #3 (recentring clamps): each get
    a direct, filesystem-scratch-only unit test (a throwaway tempdir, no repo
    fixtures needed -- kept separate from _selftest_plan's pure-function tests
    since these two genuinely need disk I/O)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        name = "run_a"
        ok, reason = checkpoint_health(d, name, 5)
        assert not ok and "JSON" in reason, reason               # nothing on disk yet

        with open(os.path.join(d, f"{name}.json"), "w") as f:
            json.dump({"diverged": False, "observables": {"E0": -400.0}}, f)
        ok, reason = checkpoint_health(d, name, 5)
        assert not ok and "mpack" in reason, reason               # JSON but no weights

        open(os.path.join(d, f"{name}.mpack"), "w").close()
        ok, reason = checkpoint_health(d, name, 5)
        assert ok, reason                                          # healthy: below bound(5)=-365

        with open(os.path.join(d, f"{name}.json"), "w") as f:
            json.dump({"diverged": True, "observables": {"E0": -400.0}}, f)
        ok, _ = checkpoint_health(d, name, 5)
        assert not ok                                               # diverged -> refuse

        bound5 = -(5 ** 3 + 3 * 4 ** 2 * 5)
        with open(os.path.join(d, f"{name}.json"), "w") as f:
            json.dump({"diverged": False, "observables": {"E0": bound5 + 1.0}}, f)
        ok, _ = checkpoint_health(d, name, 5)
        assert not ok                                               # above bound+0.05 -> refuse

        with open(os.path.join(d, f"{name}.json"), "w") as f:
            json.dump({"diverged": False, "observables": {"E0": bound5 + 0.04}}, f)
        ok, reason = checkpoint_health(d, name, 5)
        assert ok, reason                                           # inside the +0.05 slack -> ok

    # MEDIUM #3: electric fill -- centre clamp (<=0.06 from the seed centre) +
    # the h>=0.02 floor, forced by a deliberately wild L4 estimate.
    c_seed = electric_center(0.0, 5, 0.0)
    c, hz_pts, notes = electric_fill_grid(0.0, 5, 0.0, c_seed + 1.0)
    assert abs(c - c_seed) <= 0.06 + 1e-9, (c, c_seed)
    assert any("clamp" in n for n in notes), notes
    assert all(h >= 0.02 for h in hz_pts), hz_pts
    h_c4_matching = electric_center(0.0, 4, 0.0)                    # L4 fit == the L4 seed
    c, hz_pts, notes = electric_fill_grid(0.0, 5, 0.0, h_c4_matching)   # -> no clamp needed
    assert abs(c - c_seed) < 1e-9 and not notes, (c, notes)

    # MEDIUM #3: chain window shift clamp (<=0.10 from the seed window).
    seed = set(round(x, 4) for x in chain_links(0.4, "up"))
    w = chain_link_window(0.4, 6, _CENTERS[0.4] + 5.0)   # wildly wrong crossing
    for h in set(w) - seed:
        assert min(abs(h - s) for s in seed) <= CHAIN_SHIFT_CLAMP + 1e-6, h

    print("[selftest] ok (health gate + clamps)", file=sys.stderr, flush=True)


def _selftest_chain_link_early_submit():
    """Filesystem-based test for plan()'s early (afterok) chain-link
    submission path: an anchor that's merely SUBMITTED, not yet landed, must
    no longer hold its link job -- it queues now, dependent on the anchor's
    own jobid, read live from watch_state.json. `chain_l4_tables`/
    `chain_l4_crossing` (plan()'s only firstorder_fit-dependent seams) are
    monkeypatched so this needs no real fit input data."""
    import tempfile

    global chain_l4_tables, chain_l4_crossing
    real_tables, real_crossing = chain_l4_tables, chain_l4_crossing

    class _FakeTable:
        def __init__(self, h):
            self.h, self.diverged = h, [False] * len(h)

    hy, cut, hz, L, branch = 0.0, "magnetic_hz0.4", 0.4, 5, "up"
    anchor = chain_anchor(hz, branch)
    seed_centre = _CENTERS[hz]
    ckpt = chain_anchor_run_name(L, anchor, hz, hy)

    def _write_manifest(manifest_dir, jobid):
        os.makedirs(manifest_dir, exist_ok=True)
        with open(os.path.join(manifest_dir, "manifest_test.tsv"), "w") as f:
            f.write("jobid\thy\tcut\tL\trole\th\tname\tout_dir\tsubmitted_at\n")
            f.write(f"{jobid}\t{hy}\t{cut}\t{L}\tchain_{branch}\t{anchor}\tname\tout\tnow\n")

    def _link_spec(specs):
        return [s for s in specs if s["role"] == f"chain_{branch}" and s["L"] == L]

    try:
        chain_l4_tables = lambda hz_, hy_, results_dir_: (
            _FakeTable([anchor]), _FakeTable([chain_anchor(hz, "dn")]))
        chain_l4_crossing = lambda up4, dn4, **kw: seed_centre     # -> no window shift

        # (1) anchor pending: manifest row present, no JSON, no watch_state
        # entry at all -> treated as PENDING -> link emitted WITH afterok.
        with tempfile.TemporaryDirectory() as d:
            base, results_dir = os.path.join(d, "phase3d"), os.path.join(d, "phase3d", f"hy{hy}")
            _write_manifest(os.path.join(base, "manifests"), "58200001")
            specs, _def, _notes = plan(hy, results_dir, os.path.join(base, "manifests"), [cut])
            hit = _link_spec(specs)
            assert hit, "expected a pending-anchor link spec"
            assert hit[0]["dependency"] == "afterok:58200001,singleton", hit[0]["dependency"]

        # (2) anchor landed healthy: final JSON+mpack present, E0 below bound
        # -> emitted WITHOUT afterok (today's path, unchanged).
        with tempfile.TemporaryDirectory() as d:
            base, results_dir = os.path.join(d, "phase3d"), os.path.join(d, "phase3d", f"hy{hy}")
            _write_manifest(os.path.join(base, "manifests"), "58200002")
            ckpt_dir = os.path.join(results_dir, cut, f"L{L}")
            os.makedirs(ckpt_dir, exist_ok=True)
            bound = -(L ** 3 + 3 * (L - 1) ** 2 * L)
            with open(os.path.join(ckpt_dir, f"{ckpt}.json"), "w") as f:
                json.dump({"diverged": False, "observables": {"E0": bound - 1.0}}, f)
            open(os.path.join(ckpt_dir, f"{ckpt}.mpack"), "w").close()
            specs, _def, _notes = plan(hy, results_dir, os.path.join(base, "manifests"), [cut])
            hit = _link_spec(specs)
            assert hit, "expected a healthy-anchor link spec"
            assert hit[0]["dependency"] == "singleton", hit[0]["dependency"]

        # (3) anchor FAILED per watch_state, no JSON -> held (no spec), noted.
        with tempfile.TemporaryDirectory() as d:
            base, results_dir = os.path.join(d, "phase3d"), os.path.join(d, "phase3d", f"hy{hy}")
            _write_manifest(os.path.join(base, "manifests"), "58200003")
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "watch_state.json"), "w") as f:
                json.dump({ckpt: {"state": "FAILED"}}, f)
            specs, _def, notes = plan(hy, results_dir, os.path.join(base, "manifests"), [cut])
            assert not _link_spec(specs), "a FAILED anchor must not emit a link spec"
            assert any("state=FAILED" in n for n in notes), notes

        # (4) anchor submitted, but the L4 fit/window isn't available yet ->
        # held (this cut's whole link tier is skipped upstream of the anchor
        # check -- no spec, regardless of manifest/watch_state content).
        chain_l4_tables = lambda hz_, hy_, results_dir_: (None, None)
        with tempfile.TemporaryDirectory() as d:
            base, results_dir = os.path.join(d, "phase3d"), os.path.join(d, "phase3d", f"hy{hy}")
            _write_manifest(os.path.join(base, "manifests"), "58200004")
            specs, _def, _notes = plan(hy, results_dir, os.path.join(base, "manifests"), [cut])
            assert not _link_spec(specs), "no L4 window -> no link spec"
    finally:
        chain_l4_tables, chain_l4_crossing = real_tables, real_crossing

    print("[selftest] ok (chain-link early submit)", file=sys.stderr, flush=True)


def _dump_dry(grid):
    for hy in sorted(grid):
        for cid in sorted(grid[hy]):
            for L in sorted(grid[hy][cid]):
                cell = grid[hy][cid][L]
                if cell["kind"] == "electric":
                    print(f"hy={hy} {cid:<16} L={L}: hx={cell['hx']} "
                          f"hz_points={cell['hz_points']}  ({len(cell['hz_points'])} cold jobs)")
                else:
                    up, dn = cell["up"], cell["dn"]
                    print(f"hy={hy} {cid:<16} L={L}: hz={cell['hz']} "
                          f"up(anchor={up['anchor']}, links={up['links']})  "
                          f"dn(anchor={dn['anchor']}, links={dn['links']})  (2 chain jobs)")


def emit_cell(cut_id, L, hy):
    """Shell-friendly single-cell dump for nersc/launch_phase3d.sh (avoids scraping
    --dry's human text or shelling out a JSON parser per point):
      electric -> one line  "electric <hx> <hz1> <hz2> ... <hz7>"
      magnetic -> two lines "up <hz> <anchor> <link1> ..." / "dn <hz> <anchor> <link1> ..."
    """
    cell = points_for(cut_id, L, hy)
    if cell["kind"] == "electric":
        print("electric", cell["hx"], *cell["hz_points"])
    else:
        for branch in ("up", "dn"):
            b = cell[branch]
            print(branch, cell["hz"], b["anchor"], *b["links"])


# =============================================================================
# State-driven planner ("plan" subcommand): idempotent, priority-ordered,
# MAX_QUEUE-truncated. ONE python entry point owns every decision (what's
# launchable given the manifests + landed finals + locator fits); the bash
# launcher only turns the returned specs into sbatch calls. Supersedes the
# earlier two-wave (LS=4 then LS="5 6") design entirely.
#
# Fit modules (transition_fit.py, firstorder_fit.py) are sibling scripts in
# analysis/scripts/ -- imported lazily so the grid itself stays scipy-free.
# =============================================================================
WANDB_PROJECT_VAL = "tc3d-phase3d"
SNAP_ARGS = "--snapshot_every 50 --final_eval_rounds 8"   # chains add TOPO_POOLED=1 (in-job O_FM+S2 per point, 2026-09-21)
# Speed levers (p3d/speed-research, merged 2026-09-09): exact unfolded-GEMM invariant block +
# strict float32 forward/VJP with a double QGT twin; gate-verified equivalent (notes/speed_levers.md).
SPEED_ENV = {"INV_IMPL": "dense", "COMPUTE_DTYPE": "float32"}


def speed_env(L, hy):
    """SPEED_ENV plus QGT_SOLVER=kernel where the dense/cholesky QGT does not fit:
    L>=6 complex (h_y != 0) materialises an 11 GB S + its cho_factor copy and OOMs a
    40 GB node (audit 2026-09-09); the Woodbury kernel solve never forms S."""
    env = dict(SPEED_ENV)
    if int(L) >= 6 and float(hy) != 0.0:
        env["QGT_SOLVER"] = "kernel"
    return env
ELECTRIC_FLANK_OFFSETS = (-0.12, 0.0, 0.12)     # L5/6, seed centre, submitted immediately
ELECTRIC_FILL_OFFSETS = (-0.06, -0.03, 0.03, 0.06)   # L5/6, recentred, gated on the L4 fit
OFFSET_L_CHAIN = {5: 0.02, 6: 0.06}              # Phase-B trend 0.83 -> 0.84 -> 0.89


# ---- per-L knobs (ported from the old bash launcher; python is now the only
# place these live) -----------------------------------------------------------
def kernel_for(L):
    return L - 1


def diag_shift_for(L):
    return "3e-3" if L >= 5 else "1e-3"


def chunk_for(L):
    return "2048" if L <= 5 else None   # L6: leave to the wrapper's own default


def resubmit_for(L):
    return "1" if L >= 5 else "0"


def walltime_for(L, hy, electric=False, chain=False):
    # L4 chains (2026-09-17 rule: anchor 500 steps + up to 11 links x 300 steps = 3800
    # steps; L4c 3.1 s/step -> 3.3 h + evals, L4r ~1.5 s/step -> 1.6 h): 4:00 / 2:30,
    # AUTO_RESUBMIT finishes anything that overruns.
    if chain and L == 4:
        return "04:00:00" if float(hy) != 0.0 else "02:30:00"
    # Fast path (dense conv + float32, 2026-09-10): L4c 3.2 s/step, L5r 6.6, L6r 21.9.
    # Budget = 8-link train (200 steps each) + compile + observables, ~1.5x margin;
    # L6 real (4.9 h) still needs the 5 h cap + resubmit.
    # Electric cold points also run the POST_S2_EVAL pass (measured L5: 54 min train
    # + 52 min eval = 1:46) -> 2:30 at L5; an 8-link L5 train measured 1:57 (the
    # wrapper's auto-resubmit had to finish it) -> 3:00 for chains at L5.
    # Complex lane (h_y != 0) electric, DIRECTLY MEASURED 2026-09-14. An electric
    # cold job pays THREE components: the N_ITER step loop, train.py's in-train
    # final_eval_rounds=8 pass, and the separate POST_S2_EVAL=1 addon (eval_snapshots.py
    # --last_only, scores ONE snapshot). Real-lane ground truth (mtime delta on the
    # standalone addon, jobs 58188982/58116756): L5r train ~52 min + in-train eval
    # ~18 min + addon ~9.3 min; L6r train ~182 min + in-train eval ~35.6 min + addon
    # ~23 min. The full-fidelity k-sweep job (58306108: real production L5c train +
    # 8 individually-timed eval rounds) then showed the EVAL machinery does NOT scale
    # with the complex/real per-step ratio the way training does -- L5c in-train eval
    # measured 17.4 min, essentially equal to L5r's 18 min (not 2.51x), because it's
    # ops-build + O(N) measurement, not the SR/QGT-heavy step. Corrected estimates:
    # L5c raw ~= train 131.6 min (measured, 58306108) + in-train eval 17.4 min
    # (measured) + addon (untested for complex, assumed ~real's 9.3 min per the above)
    # ~= 2:38-2:44 -> 2:45:00. L6c raw ~= train 406 min (measured, 48.75 s/step probe
    # 58305359) + in-train eval ~36 min + addon ~23-30 min (same real-like assumption)
    # ~= 7:45-7:52 -- split into two chained 4:00:00 AUTO_RESUBMIT cycles (8 h budget)
    # rather than one 5:00:00 cap + an uncertain 2-3rd cycle.
    nz = float(hy) != 0.0
    if L == 4:
        return "02:00:00" if nz else "01:30:00"
    if L == 5:
        if nz:
            return "02:45:00" if electric else "05:00:00"
        return "02:30:00" if electric else "03:00:00"
    if L == 6 and nz and electric:
        return "04:00:00"
    return "05:00:00"


def exact_e0_for(L):
    return {4: "-172", 5: "-365", 6: "-666"}[L]


def arch_env(L):
    return {"DUAL": "1", "NONINV_HIDDEN": "4 8", "INV": "8 8",
            "KERNEL": str(kernel_for(L)), "BC": "OBC"}


# ---- run-name reconstruction (must match the wrapper's own NAME formula) ---
def electric_run_name(L, hx, hz, hy):
    hy_tag = "" if float(hy) == 0.0 else f"_hy{hy}"
    return (f"gridinv_dual_L{L}_OBC_hx{hx}_hz{hz}{hy_tag}"
            f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}")


def chain_link_run_name(L, hx, hz, hy, branch):
    return (f"gridinv_dual_L{L}_OBC_hx{hx}_hz{hz}_hy{hy}"
            f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")


# the L5/6 anchor is submitted via the PLAIN gridinv path (no NAME_TEMPLATE
# branch suffix) -- same auto-name shape as an electric point.
chain_anchor_run_name = electric_run_name


# ---- manifest reading (idempotence source of truth) -------------------------
def read_manifest_rows(manifest_dir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(manifest_dir, "manifest_*.tsv"))):
        try:
            with open(fp, newline="") as f:
                rows.extend(list(csv.DictReader(f, delimiter="\t")))
        except OSError:
            continue
    return rows


def _parse_hy(v):
    """CLI --hy: the y-cut sentinel stays a str, everything else MUST be a float
    (plan() rounds it; a str crashes the L4 gap-fill tier -- and the launcher
    reads a crashed planner as "0 jobs", silently)."""
    return v if v == YCUT_HY else float(v)


def _hykey(hy):
    """Manifest/plan key for a plane: the rounded float, or the y-cut sentinel."""
    return YCUT_HY if is_ycut_plane(hy) else round(float(hy), 4)


def submitted_index(rows):
    """(hy,cut,L,role) -> {h, ...} already recorded in some manifest."""
    idx = collections.defaultdict(set)
    for r in rows:
        try:
            key = (_hykey(r["hy"]), r["cut"], int(r["L"]), r["role"])
            idx[key].add(round(float(r["h"]), 4))
        except (KeyError, ValueError, TypeError):
            continue
    return idx


def already_submitted(idx, hy, cut, L, role, h):
    return round(float(h), 4) in idx.get((_hykey(hy), cut, L, role), set())


# ---- L4 locator fits (gate the recentring tiers) ----------------------------
def electric_fit_at(hx, hy, L, results_dir):
    """(h_c, h_c_err, (h_min, h_max)) from transition_fit on
    results_dir/electric_hx{hx}/L{L}'s finals (O_FM_paratoric vs hz), or
    (None, None, None) if <5 finals / no convergence. The h-range is the
    fitted curve's own observed span, used by the refine tier to tell a
    bracketed crossing from an unbracketed rise (rule 4)."""
    try:
        import transition_fit as tf
    except ImportError:
        return None, None, None
    d = os.path.join(results_dir, f"electric_hx{hx}", f"L{L}")
    if not os.path.isdir(d):
        return None, None, None
    curves = tf.load_runs([d], "hz", {"hx": float(hx), "hy": float(hy)}, "O_FM_paratoric")
    curve = curves.get(L)
    if curve is None or len(curve.h) < 5:
        return None, None, None
    h_c, err, _meta = tf.combine_default(tf.locate_all(curve))
    if h_c is None or err is None or not (math.isfinite(h_c) and math.isfinite(err)):
        return None, None, None
    return float(h_c), float(err), (float(curve.h.min()), float(curve.h.max()))


def chain_primary_fit_at(hz, hy, L, results_dir):
    """(h_c, h_c_err) from firstorder_fit.locate_cut's PRIMARY marker for this
    cut's kind -- topo-trivial (hz<=0.2): the O_FM_membrane_R1 inflection;
    trivial-trivial (hz>=0.4): the energy crossing -- reusing the SAME
    `_default_kind` policy firstorder_fit itself uses, so the refine gate
    never diverges from the analysis-side definition of "the primary locator
    error" (rule 3). (None, None) if that cut's L dir has no finals yet."""
    try:
        import firstorder_fit as ff
    except ImportError:
        return None, None
    d = os.path.join(results_dir, f"magnetic_hz{hz}", f"L{L}")
    if not os.path.isdir(d):
        return None, None
    rows = ff.locate_cut([d], "hx", {"hz": float(hz), "hy": float(hy)})
    row = next((r for r in rows if r["L"] == L), None)
    if row is None or row.get("h_c") is None or row.get("h_c_err") is None:
        return None, None
    h_c, err = row["h_c"], row["h_c_err"]
    if not (math.isfinite(h_c) and math.isfinite(err)):
        return None, None
    return float(h_c), float(err)


def electric_refine_points(h_c, h_range, step=0.02):
    """Rule 4's single refine round: symmetric h_c +- step when h_c falls
    INSIDE the fit's own observed h-range (a genuinely bracketed crossing);
    otherwise (the sigmoid still rising at the window edge -- an "unbracketed
    rise", the fit extrapolated rather than bracketed a real inflection) ONE
    point `step` beyond the nearest edge, in the rise's direction, so the next
    round's data can finally bracket it. Returns ([h...], unbracketed:bool)."""
    lo, hi = h_range
    if lo <= h_c <= hi:
        return [round(h_c - step, 4), round(h_c + step, 4)], False
    edge, sign = (hi, 1.0) if h_c > hi else (lo, -1.0)
    return [round(edge + sign * step, 4)], True


def chain_tables_at(hz, hy, L, results_dir):
    """(up_Table, dn_Table) at size L from firstorder_fit, or (None, None) if
    that cut's L dir has no finals yet."""
    try:
        import firstorder_fit as ff
    except ImportError:
        return None, None
    d = os.path.join(results_dir, f"magnetic_hz{hz}", f"L{L}")
    if not os.path.isdir(d):
        return None, None
    branches = ff.load_branches([d], "hx", {"hz": float(hz), "hy": float(hy)})
    return branches.get("up", {}).get(L), branches.get("dn", {}).get(L)


def chain_l4_tables(hz, hy, results_dir):
    return chain_tables_at(hz, hy, 4, results_dir)


def chain_l4_crossing(up4, dn4, hz=None):
    """L4 h_c for recentring, by cut kind (the campaign's locator policy): hz <= TOPO_HZ_MAX
    (topological -> trivial) uses the membrane-O_FM inflection on the lowest-energy
    winner curve -- the L4 energy crossing there is a 200-step convergence artifact
    (hz=0: up links 0.85-0.95 sit 5-8 above the dn branch, "crossing" at 0.999);
    hz >= 0.4 (trivial -> trivial) uses the energy crossing. None when merged/no
    overlap/no fit ("if merged, keep the seeded links")."""
    if up4 is None or dn4 is None or len(up4.h) == 0 or len(dn4.h) == 0:
        return None
    import firstorder_fit as ff
    if hz is not None and float(hz) <= TOPO_HZ_MAX:
        wt = ff.winner({"up": {4: up4}, "dn": {4: dn4}}).get(4)
        fits = ff.jump_locators(wt, want_ofm=True).get(ff.OFM_OBS) if wt is not None else None
        if fits:
            h_c, _err, _meta = ff.tf.combine_default(fits)
            if h_c is not None and math.isfinite(h_c):
                return float(h_c)
        return None
    h_c, _h_c_err, _bracket, _info = ff.energy_crossing(up4, dn4)
    return h_c


CHAIN_SHIFT_CLAMP = 0.10   # MEDIUM #3: |window shift| <= this, relative to the seed


def chain_link_window(hz, L, h_c4):
    """Ascending UNION of the seed link window and the L4-recentred window (same
    0.05 inside-spacing) -- never the recentred one alone (design addendum #1):
    a wrong L4 estimate then costs at most a link or two, never a coverage gap.
    The shift itself is clamped to +-CHAIN_SHIFT_CLAMP (MEDIUM #3: a bad fit can
    only nudge the window, never throw it far off the seed)."""
    seed = sorted(round(x, 4) for x in chain_links(hz, "up"))
    if h_c4 is None or L not in OFFSET_L_CHAIN:
        return seed
    seed_center = _CENTERS[round(hz, 4)]
    shift = (h_c4 + OFFSET_L_CHAIN[L]) - seed_center
    if abs(shift) > CHAIN_SHIFT_CLAMP:
        clamped = math.copysign(CHAIN_SHIFT_CLAMP, shift)
        print(f"[plan] clamp magnetic_hz{hz} L{L}: window shift {shift:.4f} -> "
              f"{clamped:.4f} (|Delta|>{CHAIN_SHIFT_CLAMP})", file=sys.stderr)
        shift = clamped
    # Snap to the links' own 0.05 spacing so the recentred grid OVERLAPS the seed
    # grid: the union is then 6-8 links per branch (the trimmed budget), never 12.
    shift = round(shift / 0.05) * 0.05
    recentred = [round(x + shift, 4) for x in seed]
    return sorted(set(seed) | set(recentred))


def checkpoint_health(out_dir, name, L):
    """CRUCIAL #2 gate: is `{out_dir}/{name}` a safe INIT_FROM? (ok, reason).
    Requires the final JSON AND the .mpack weights on disk, diverged == False,
    and E0 finite with E0 < bound(L) + 0.05 (bound = -(L^3+3(L-1)^2 L), the same
    small numerical slack used elsewhere on this bound)."""
    jpath, mpack = os.path.join(out_dir, f"{name}.json"), os.path.join(out_dir, f"{name}.mpack")
    if not os.path.isfile(jpath):
        return False, "no final JSON"
    if not os.path.isfile(mpack):
        return False, "no .mpack weights"
    try:
        d = json.load(open(jpath))
    except (OSError, json.JSONDecodeError):
        return False, "unreadable JSON"
    if d.get("diverged"):
        return False, "diverged"
    E0 = (d.get("observables") or {}).get("E0")
    if E0 is None or not math.isfinite(E0):
        return False, "E0 missing/non-finite"
    bound = -(L ** 3 + 3 * (L - 1) ** 2 * L)
    if E0 >= bound + 0.05:
        return False, f"E0={E0:.2f} >= bound+0.05={bound + 0.05:.2f}"
    return True, "ok"


def electric_fill_grid(hx, L, hy, h_c4):
    """The recentred 4-offset electric fill grid, clamped per MEDIUM #3:
    |centre - seed centre| <= 0.06, every h >= 0.02. Returns (centre, [h...], notes)."""
    c_seed = electric_center(hx, L, hy)
    c = h_c4 + (_BASE_L[L] - _BASE_L[4])
    notes = []
    if abs(c - c_seed) > 0.06:
        clamped = c_seed + math.copysign(0.06, c - c_seed)
        notes.append(f"[plan] clamp electric_hx{hx} L{L}: centre {c:.4f} -> "
                     f"{clamped:.4f} (|Delta|>0.06 from the seed centre)")
        c = clamped
    hz_pts = []
    for d in ELECTRIC_FILL_OFFSETS:
        hz = round(c + d, 2)
        if hz < 0.02:
            notes.append(f"[plan] clamp electric_hx{hx} L{L}: hz={hz} < 0.02 floor -> 0.02")
            hz = 0.02
        hz_pts.append(hz)
    return c, hz_pts, notes


def _outer_checkpoint(table, bracket, side):
    """Refinement's INIT_FROM source: the nearest landed, non-diverged point on
    THIS branch's table, strictly outside the crossing bracket on `side` ('lo'
    below bracket[0], 'hi' above bracket[1]) -- refusal (b) by construction,
    since the caller always passes only one branch's own table."""
    if table is None or len(table.h) == 0:
        return None
    lo, hi = bracket
    if side == "lo":
        cands = [float(table.h[i]) for i in range(len(table.h))
                 if not table.diverged[i] and table.h[i] <= lo + 1e-9]
        return max(cands) if cands else None
    cands = [float(table.h[i]) for i in range(len(table.h))
             if not table.diverged[i] and table.h[i] >= hi - 1e-9]
    return min(cands) if cands else None


# ---- the three plan-time refusals (unit-tested; see _selftest) -------------
def refuse_cold_inside_window(h, window):
    """(a) A chain-cut field value strictly inside its branch's link window may
    NEVER be cold-started (recipe SB: cold + dt<0.02 inside the coexistence
    window diverges, 15/15 at L6) -- only the anchor, kept >=0.15 outside, cold-
    starts. Structurally this planner never builds such a spec; kept as an
    explicit, testable guard against a future regression."""
    lo, hi = window
    return lo < round(float(h), 6) < hi


def nearest_same_branch_checkpoint(same_branch_h, new_h):
    """(b) INIT_FROM must be the nearest ALREADY-SUBMITTED point on the SAME
    branch -- never the other branch, even if numerically closer (the two
    branches sit in different phases; warm-starting across them seeds the wrong
    state). `same_branch_h` is scoped to one branch by the caller; empty/None
    -> refuse (no same-branch checkpoint to warm-start from yet)."""
    pts = list(same_branch_h or [])
    if not pts:
        return None
    return min(pts, key=lambda h: abs(h - new_h))


def branch_points_by_distance(table, anchor):
    """This branch's landed (h, diverged) pairs, ordered outward from the anchor
    -- the scan order refusal (c) needs to find the FIRST crash/shed point."""
    if table is None or len(table.h) == 0:
        return []
    order = sorted(range(len(table.h)), key=lambda i: abs(float(table.h[i]) - anchor))
    return [(float(table.h[i]), bool(table.diverged[i])) for i in order]


def spinodal_cutoff(points_by_distance):
    """(c) the branch's crash/shed point: the first diverged h scanning outward
    from the anchor. None if nothing has diverged (yet)."""
    for h, dv in points_by_distance:
        if dv:
            return h
    return None


def beyond_spinodal(h, anchor, cutoff):
    """(c) refuse any candidate farther from the anchor, on the same side, than
    the branch's recorded spinodal -- that regime is a recorded physics result
    (crash/shed), never a retry target."""
    if cutoff is None:
        return False
    same_side = (h - anchor) * (cutoff - anchor) >= 0
    return same_side and abs(h - anchor) >= abs(cutoff - anchor)


# ---- spec builders (env dicts the bash launcher turns straight into sbatch) -
# User decision 2026-09-19: the h_y = 1.0 electric cold points at h_x = 0 / 0.25 / 0.5 converged badly with the
# standard recipe (Vscore 0.2-0.3 on the topological side, S2 plateau ending near h_z 0.1). Redo them gentler
# and longer: dt 0.01, diag_shift 1e-2, 1000 steps (walltime 3:00 for the 1000-step complex L4 loop + evals).
_PLANE_ELECTRIC_REDO = {(1.0, 0.0), (1.0, 0.2), (1.0, 0.25), (1.0, 0.5)}      # (hy, hx)


def _electric_spec(cut, hx, L, hy, hz, refs=None, role="cold"):
    redo = L == 4 and (round(float(hy), 4), round(float(hx), 4)) in _PLANE_ELECTRIC_REDO
    env = {**arch_env(L), **speed_env(L, hy), "L": str(L), "HX": str(hx), "HZ": str(hz), "HY": str(hy),
           "DT": "0.01" if redo else "0.02", "LR_MIN": "0.002", "N_ITER": "1000" if redo else "500",
           "DIAG_SHIFT": "1e-2" if redo else diag_shift_for(L), "CKPT_EVERY": "10",
           "EXACT_E0": exact_e0_for(L), "EXTRA_ARGS": SNAP_ARGS,
           "AUTO_RESUBMIT": resubmit_for(L), "WANDB_PROJECT": WANDB_PROJECT_VAL,
           "POST_S2_EVAL": "1", "POST_S2_SECTOR": "electric"}
    chunk = chunk_for(L)
    if chunk:
        env["CHUNK"] = chunk
    if refs:
        rec = ref_lookup(refs, hx, hz, L)
        if rec:
            env["REF_E"], env["REF_SIG"] = str(rec["E"]), str(rec["E_err"])
    return {"role": role, "cut": cut, "L": L, "wrapper": "gridinv",
            "jobname": f"p3d_hy{hy}_e{hx}_L{L}", "h_list": [hz], "env": env,
            "dependency": None, "walltime": "03:00:00" if redo else walltime_for(L, hy, electric=True), "array": None,
            "out_dir_rel": f"hy{hy}/{cut}/L{L}"}


def _chain_anchor_spec(cut, hz, L, hy, branch, refs=None):
    hx = chain_anchor(hz, branch)
    # Polarized-side (dn) anchors at L>=5 sit deep in the field-dominated phase, where the
    # cold-start recipe (dt 0.02, ds 3e-3) overshoots: L6 hx=1.3/hz=0.4 oscillated by +-100 and
    # diverged at step 271, L6 hx=1.25/hz=0 needed 10 rollbacks (Vscore 0.095), L5 hx=1.5/hz=0.7
    # blew up at step 4 (2026-09-10). Halve dt, stiffen the shift, add steps to compensate.
    gentle = int(L) >= 5 and branch == "dn"
    env = {**arch_env(L), **speed_env(L, hy), "L": str(L), "HX": str(hx), "HZ": str(hz), "HY": str(hy),
           "DT": "0.01" if gentle else "0.02", "LR_MIN": "0.002", "N_ITER": "600" if gentle else "500",
           "DIAG_SHIFT": "1e-2" if gentle else diag_shift_for(L), "CKPT_EVERY": "10",
           "EXACT_E0": exact_e0_for(L), "EXTRA_ARGS": SNAP_ARGS,
           "AUTO_RESUBMIT": "1", "WANDB_PROJECT": WANDB_PROJECT_VAL}
    chunk = chunk_for(L)
    if chunk:
        env["CHUNK"] = chunk
    if refs:
        rec = ref_lookup(refs, hx, hz, L)
        if rec:
            env["REF_E"], env["REF_SIG"] = str(rec["E"]), str(rec["E_err"])
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "gridinv",
            "jobname": f"p3d_hy{hy}_m{hz}_L{L}_{branch}", "h_list": [hx], "env": env,
            "dependency": None, "walltime": walltime_for(L, hy), "array": None,
            "out_dir_rel": f"hy{hy}/{cut}/L{L}"}


def _chain_l4_job_spec(cut, hz, hy, branch):
    """Tier L4: ONE combined anchor+links batch job per branch (unchanged from
    the original design -- ANCHOR_OVERRIDES applies the cold-start knobs to
    point 0 only)."""
    L = 4
    field_values = [chain_anchor(hz, branch)] + chain_links(hz, branch)
    jobname = f"p3d_hy{hy}_m{hz}_L{L}_{branch}"
    name_tpl = (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")
    ds = diag_shift_for(L)
    # x-polarized (dn) anchors at h_y >= 0.6 diverge with the cold recipe (4/4 on 2026-09-18: hz 0/0.85/1.0 at
    # hy 0.6/0.8/1.0, all a dt-0.02 rollback wall inside 40 steps); the retry recipe dt 0.01 / ds 5e-3 landed 3/3.
    gentle = branch == "dn" and float(hy) >= 0.6
    anchor_ov = (f'{{"dt":0.01,"lr_min":0.002,"n_iter":500,"diag_shift":5e-3}}' if gentle
                 else f'{{"dt":0.02,"lr_min":0.002,"n_iter":500,"diag_shift":{ds}}}')
    env = {**arch_env(L), **speed_env(L, hy), "L": str(L), "SWEEP": "hx", "HZ": str(hz), "HY": str(hy),
           "FIELD_VALUES": " ".join(str(h) for h in field_values),
           "CHUNK_POINTS": str(len(field_values)), "WARM_START": "1",
           "ANCHOR_OVERRIDES": anchor_ov, "NAME_TEMPLATE": name_tpl,
           "DT": "0.005", "LR_MIN": "0.0005", "DIAG_SHIFT": "3e-3", "N_ITER": "300",
           "CKPT_EVERY": "10", "EXTRA_ARGS": SNAP_ARGS,
           "WANDB_PROJECT": WANDB_PROJECT_VAL, "WANDB_GROUP": jobname,
           "AUTO_RESUBMIT": "1", "CHUNK": "2048", "TOPO_POOLED": "1"}
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": field_values, "env": env,
            "dependency": None, "walltime": walltime_for(L, hy, chain=True), "array": "0",
            "out_dir_rel": f"hy{hy}/{cut}/L{L}"}


def _ycut_l4_job_spec(cut, hx, hz, branch):
    """HY=y pseudo-plane: ONE combined anchor+links batch job per branch sweeping
    hy at fixed (hx, hz); complex lane throughout (hy >= 0.6)."""
    L = 4
    field_values = [ycut_anchor(branch, hx, hz)] + ycut_links(branch, hx, hz)
    jobname = f"p3d_y_hx{hx:g}_hz{hz:g}_L{L}_{branch}"
    name_tpl = (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")
    ds = diag_shift_for(L)
    # y-polarized (dn, h_y = 1.5) anchors get the gentle recipe of the h_y >= 0.6 dn anchors
    # (2026-09-19: the cold hx0.5/hz0 dn anchor hit a 26-rollback wall, E0 = -92 with diverged=False).
    anchor_ov = (f'{{"dt":0.01,"lr_min":0.002,"n_iter":500,"diag_shift":5e-3}}' if branch == "dn"
                 else f'{{"dt":0.02,"lr_min":0.002,"n_iter":500,"diag_shift":{ds}}}')
    env = {**arch_env(L), **speed_env(L, YCUT_ANCHORS[0]), "L": str(L), "SWEEP": "hy",
           "HX": str(hx), "HZ": str(hz), "HY": str(field_values[0]),
           "FIELD_VALUES": " ".join(str(h) for h in field_values),
           "CHUNK_POINTS": str(len(field_values)), "WARM_START": "1",
           "ANCHOR_OVERRIDES": anchor_ov, "NAME_TEMPLATE": name_tpl,
           "DT": "0.005", "LR_MIN": "0.0005", "DIAG_SHIFT": "3e-3", "N_ITER": "300",
           "CKPT_EVERY": "10", "EXTRA_ARGS": SNAP_ARGS,
           "WANDB_PROJECT": WANDB_PROJECT_VAL, "WANDB_GROUP": jobname,
           "AUTO_RESUBMIT": "1", "CHUNK": "2048", "TOPO_POOLED": "1"}
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": field_values, "env": env,
            "dependency": None, "walltime": walltime_for(L, YCUT_ANCHORS[0], chain=True), "array": "0",
            "out_dir_rel": f"ycuts/{cut}/L{L}"}


# ---- electric h_z chains (h_x = 0 plane, 2026-09-21) --------------------------------------------------------
# Referee check (exact: E must fall monotonically with h_z): the cold electric points at h_y >= 0.6 are under-
# converged on the topological side (E rises by 0.3-3.3 between neighbours, Vscore 0.2-0.4), which biases the O_FM
# inflection down. ADD warm chains like the magnetic cuts (the cold points stay on disk and in the winner curve): an up chain from h_z = 0.02 (deep topological)
# outward and a dn chain from the z-polarized side (h_z = 0.45) inward, both over the same 12-point grid, so the
# crossing is bracketed by two converged branches (energy crossing + M_z/A_v cross-check the O_FM locator).
ELECTRIC_CHAIN = {(0.6, 0.0), (0.8, 0.0), (1.0, 0.0),
                  (1.0, 0.2), (1.0, 0.25), (1.0, 0.5)}   # (hy, hx); the h_y=1.0 hx0 chain moved h_z,c 0.114->0.164
                                                          # (user, 2026-09-22): redo hx=0.2/0.25/0.5 the same way
ZCHAIN_ANCHORS = (0.02, 0.45)
ZCHAIN_LINKS = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26, 0.29, 0.32, 0.36, 0.40]


def zchain_links(branch):
    return list(ZCHAIN_LINKS) if branch == "up" else list(reversed(ZCHAIN_LINKS))


def _zchain_l4_job_spec(cut, hx, hy, branch):
    """ONE combined anchor+links batch job per branch sweeping h_z at fixed (h_x, h_y); the gentle 1000-step
    recipe on the anchor (both sides), 300-step warm links; outputs land next to the cut's cold points with the
    `_up`/`_dn` name suffix (phase3d_status.py then adds crossing/loop/jump like a magnetic cut)."""
    L = 4
    field_values = [ZCHAIN_ANCHORS[0 if branch == "up" else 1]] + zchain_links(branch)
    jobname = f"p3d_hy{hy}_e{hx:g}_L{L}_{branch}"
    name_tpl = (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")
    anchor_ov = '{"dt":0.01,"lr_min":0.002,"n_iter":1000,"diag_shift":1e-2}'
    env = {**arch_env(L), **speed_env(L, hy), "L": str(L), "SWEEP": "hz", "HX": str(hx), "HY": str(hy),
           "HZ": str(field_values[0]),
           "FIELD_VALUES": " ".join(str(h) for h in field_values),
           "CHUNK_POINTS": str(len(field_values)), "WARM_START": "1",
           "ANCHOR_OVERRIDES": anchor_ov, "NAME_TEMPLATE": name_tpl,
           "DT": "0.005", "LR_MIN": "0.0005", "DIAG_SHIFT": "3e-3", "N_ITER": "300",
           "CKPT_EVERY": "10", "EXTRA_ARGS": SNAP_ARGS,
           "WANDB_PROJECT": WANDB_PROJECT_VAL, "WANDB_GROUP": jobname,
           "AUTO_RESUBMIT": "1", "CHUNK": "2048", "TOPO_POOLED": "1"}
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": field_values, "env": env,
            "dependency": None, "walltime": "04:30:00", "array": "0",
            "out_dir_rel": f"hy{hy}/{cut}/L{L}"}


def _chain_link_job_spec(cut, hz, L, hy, branch, new_h_sorted, init_from_name, role="chain"):
    """L5/6 (or a refine round): a SMALL batch job adding just `new_h_sorted`
    (already deduped by the caller), warm-started from `init_from_name` (a
    same-branch checkpoint basename, no extension) with the SAME --job-name as
    that checkpoint's own job -- `--dependency=singleton` then serializes it
    behind that job (and any of its AUTO_RESUBMIT chunks)."""
    jobname = f"p3d_hy{hy}_m{hz}_L{L}_{branch}"
    name_tpl = (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}")
    env = {**arch_env(L), **speed_env(L, hy), "L": str(L), "SWEEP": "hx", "HZ": str(hz), "HY": str(hy),
           "FIELD_VALUES": " ".join(str(h) for h in new_h_sorted),
           "CHUNK_POINTS": str(len(new_h_sorted)), "WARM_START": "1",
           "INIT_FROM": init_from_name, "NAME_TEMPLATE": name_tpl,
           "DT": "0.005", "LR_MIN": "0.0005", "DIAG_SHIFT": "3e-3", "N_ITER": "300",
           "CKPT_EVERY": "10", "EXTRA_ARGS": SNAP_ARGS,
           "WANDB_PROJECT": WANDB_PROJECT_VAL, "WANDB_GROUP": jobname,
           "AUTO_RESUBMIT": "1", "TOPO_POOLED": "1"}
    chunk = chunk_for(L)
    if chunk:
        env["CHUNK"] = chunk
    return {"role": f"chain_{branch}" if role == "chain" else f"chain_{branch}_refine",
            "cut": cut, "L": L, "wrapper": "batch", "jobname": jobname,
            "h_list": list(new_h_sorted), "env": env, "dependency": "singleton",
            "walltime": walltime_for(L, hy, chain=True), "array": "0",
            "out_dir_rel": f"hy{hy}/{cut}/L{L}"}


# ---- live Slurm state for early (afterok) chain-link submission ------------
# Queue-age fix: an L5/6 chain link whose anchor has been SUBMITTED but not
# yet landed no longer holds -- it queues now, dependent on the anchor's own
# jobid, so it accrues Slurm queue age in parallel instead of waiting for the
# anchor's final JSON. `_BAD_WATCH_STATES` mirrors nersc/watch_phase3d.sh's
# own BAD_STATES vocabulary (that script's python is inline, nothing to
# import from).
_BAD_WATCH_STATES = ("FAILED", "TIMEOUT", "OUT_OF", "NODE_FAIL", "CANCELLED",
                      "DEADLINE", "DependencyNeverSatisfied")


def load_watch_state(results_dir):
    """watch_state.json (written by nersc/watch_phase3d.sh) lives at the
    campaign root, one level above the hy plane dir -- results_dir is
    `$BASE_OUT/hy{HY}`, so the file is `$BASE_OUT/watch_state.json`. {} if
    absent/unreadable (no watcher has run yet)."""
    fp = os.path.join(os.path.dirname(os.path.normpath(results_dir)), "watch_state.json")
    try:
        with open(fp) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def anchor_watch_class(watch_state, run_name):
    """('inflight'|'completed'|'bad', state_label) for `run_name`'s live Slurm
    state. A missing entry counts as 'inflight' (PENDING) -- the watcher may
    simply not have run yet since submission, and the early-submission path
    must stay available rather than defaulting to a hold."""
    state = (watch_state.get(run_name) or {}).get("state")
    if state is None:
        return "inflight", "PENDING"
    if state == "COMPLETED":
        return "completed", state
    if any(b in state for b in _BAD_WATCH_STATES):
        return "bad", state
    return "inflight", state


def anchor_jobid(rows, hy, cut, L, branch, anchor_h):
    """The most recent manifest jobid for this (hy,cut,L,chain_{branch})
    anchor submission, or None if it was never recorded. AUTO_RESUBMIT
    requeues keep the SAME jobid (Slurm resumes the existing allocation), so
    manifest rows are never duplicated for it -- the last match is as good as
    the first, but taking it keeps this robust to any future resubmission
    scheme that does add a fresh row."""
    role = f"chain_{branch}"
    hit = None
    for r in rows:
        try:
            if (round(float(r["hy"]), 4) == round(float(hy), 4) and r["cut"] == cut
                    and int(r["L"]) == L and r["role"] == role
                    and round(float(r["h"]), 4) == round(float(anchor_h), 4)):
                hit = r.get("jobid") or hit
        except (KeyError, ValueError, TypeError):
            continue
    return hit


# ---- the plan itself ---------------------------------------------------------
def plan(hy, results_dir, manifest_dir, cut_ids=None, max_new=None):
    """Idempotent, priority-ordered, state-driven campaign plan.

    Priority order: L4 (electric all 7 + chain combined anchor+links) ->
    L5/6 chain anchors -> L5/6 electric flanks+centre -> recentred electric
    fills -> L5/6 chain link jobs (union window) -> refine. Returns
    (specs, deferred_jobnames, notes) with `specs` already truncated to
    `max_new` (None = no limit); `deferred_jobnames` lists what the ceiling
    pushed to the next re-run (this function does NOT look at the live Slurm
    queue -- the caller subtracts that separately).
    """
    global _ACTIVE_HY
    _ACTIVE_HY = float(hy) if not is_ycut_plane(hy) else None
    if is_ycut_plane(hy):
        cut_ids = [c for c in (cut_ids or YCUT_IDS) if c in YCUT_IDS]
        idx = submitted_index(read_manifest_rows(manifest_dir))
        specs = []
        for cut in cut_ids:
            _kind, (hx, hz) = _CUTS_BY_ID[cut]
            for branch in ("up", "dn"):
                if not already_submitted(idx, hy, cut, 4, f"chain_{branch}", ycut_anchor(branch, hx, hz)):
                    specs.append(_ycut_l4_job_spec(cut, hx, hz, branch))
        if max_new is not None and max_new >= 0:
            specs, deferred = specs[:max_new], specs[max_new:]
        else:
            deferred = []
        return specs, [s["jobname"] for s in deferred], []
    cut_ids = cut_ids or [c for c, _, _ in all_cuts()]
    rows = read_manifest_rows(manifest_dir)
    idx = submitted_index(rows)
    watch_state = load_watch_state(results_dir)
    try:
        refs = build_refs()
    except Exception:                                        # noqa: BLE001
        refs = {}
    notes = []
    tiers = []

    # tier: L4 --------------------------------------------------------------
    t = []
    for cut in cut_ids:
        kind, val = _CUTS_BY_ID[cut]
        if kind == "electric":
            for hz in electric_grid(val, 4, hy):
                if not already_submitted(idx, hy, cut, 4, "cold", hz):
                    t.append(_electric_spec(cut, val, 4, hy, hz, refs))
            if (round(float(hy), 4), round(float(val), 4)) in ELECTRIC_CHAIN:
                for branch in ("up", "dn"):
                    anchor = ZCHAIN_ANCHORS[0 if branch == "up" else 1]
                    if not already_submitted(idx, hy, cut, 4, f"chain_{branch}", anchor):
                        t.append(_zchain_l4_job_spec(cut, val, hy, branch))
        else:
            for branch in ("up", "dn"):
                anchor = chain_anchor(val, branch)
                if not already_submitted(idx, hy, cut, 4, f"chain_{branch}", anchor):
                    t.append(_chain_l4_job_spec(cut, val, hy, branch))
    tiers.append(t)

    # tier: L4 chain gap-fill ----------------------------------------------------
    # Links of the CURRENT link rule that an already-submitted L4 branch lacks --
    # e.g. the 0.05-window links the 2026-09-17 rule added to planes that ran under
    # the old 6-link lists (A1: hx=1.2 on hz=0.7, A2: hx=0.75 on hz<=0.2). Fine
    # window only; one small warm batch job per branch from the nearest same-branch
    # checkpoint, with the L5/6 tier's health gate and spinodal refusal. A branch
    # whose anchor is not in the manifest is covered by the combined L4 job above.
    t = []
    for cut in cut_ids:
        kind, val = _CUTS_BY_ID[cut]
        if kind != "magnetic":
            continue
        wlo, whi = chain_window(val)
        up4, dn4 = chain_l4_tables(val, hy, results_dir)
        for branch in ("up", "dn"):
            anchor = chain_anchor(val, branch)
            if not already_submitted(idx, hy, cut, 4, f"chain_{branch}", anchor):
                continue
            already_h = idx.get((round(hy, 4), cut, 4, f"chain_{branch}"), set())
            new_h = sorted({h for h in chain_links(val, branch)
                            if wlo - 1e-9 <= h <= whi + 1e-9 and round(h, 4) not in already_h},
                           key=lambda h: abs(h - anchor))
            table = up4 if branch == "up" else dn4
            if not new_h or table is None or len(table.h) == 0:
                continue
            init_h = nearest_same_branch_checkpoint(already_h, new_h[0])
            if init_h is None:
                continue
            cutoff = spinodal_cutoff(branch_points_by_distance(table, anchor))
            new_h = [h for h in new_h if not beyond_spinodal(h, anchor, cutoff)]
            if not new_h:
                notes.append(f"[plan] {cut} L4 {branch} gap-fill: all candidates beyond the spinodal ({cutoff}) -- refused")
                continue
            # L4 anchors are point 0 of the combined branch job, so they carry the
            # branch suffix like every link (unlike the L5/6 cold anchors)
            ckpt = chain_link_run_name(4, init_h, val, hy, branch)
            ok, reason = checkpoint_health(os.path.join(results_dir, cut, "L4"), ckpt, 4)
            if not ok:
                notes.append(f"[plan] hold L4 gap-fill {cut} {branch}: {reason}")
                continue
            t.append(_chain_link_job_spec(cut, val, 4, hy, branch, new_h, ckpt))
    tiers.append(t)

    # tier: L5/6 chain anchors (separate cold gridinv jobs) ------------------
    t = []
    for cut in cut_ids:
        kind, val = _CUTS_BY_ID[cut]
        if kind != "magnetic":
            continue
        for L in (5, 6):
            for branch in ("up", "dn"):
                anchor = chain_anchor(val, branch)
                if not already_submitted(idx, hy, cut, L, f"chain_{branch}", anchor):
                    t.append(_chain_anchor_spec(cut, val, L, hy, branch, refs))
    tiers.append(t)

    # tier: L5/6 electric flanks+centre (seed centre, always) ----------------
    t = []
    for cut in cut_ids:
        kind, val = _CUTS_BY_ID[cut]
        if kind != "electric":
            continue
        for L in (5, 6):
            c = electric_center(val, L, hy)
            for d in ELECTRIC_FLANK_OFFSETS:
                hz = round(c + d, 2)
                if not already_submitted(idx, hy, cut, L, "cold", hz):
                    t.append(_electric_spec(cut, val, L, hy, hz, refs))
    tiers.append(t)

    # tier: recentred electric fills (gated on the L4 fit) --------------------
    t = []
    for cut in cut_ids:
        kind, val = _CUTS_BY_ID[cut]
        if kind != "electric":
            continue
        h_c4, err, _rng = electric_fit_at(val, hy, 4, results_dir)
        if h_c4 is None:
            notes.append(f"[plan] {cut}: no usable L4 fit yet -- flanks only")
            continue
        if err >= 0.02:
            notes.append(f"[plan] {cut}: L4 h_c_err={err:.3f} >= 0.02 -- waiting")
            continue
        for L in (5, 6):
            _c, hz_pts, clamp_notes = electric_fill_grid(val, L, hy, h_c4)
            notes.extend(clamp_notes)
            for hz in hz_pts:
                if not already_submitted(idx, hy, cut, L, "cold", hz):
                    t.append(_electric_spec(cut, val, L, hy, hz, refs))
    tiers.append(t)

    # tier: L5/6 chain link jobs (union window, singleton on the anchor) -----
    t = []
    for cut in cut_ids:
        kind, val = _CUTS_BY_ID[cut]
        if kind != "magnetic":
            continue
        up4, dn4 = chain_l4_tables(val, hy, results_dir)
        if up4 is None or dn4 is None or len(up4.h) == 0 or len(dn4.h) == 0:
            continue                                          # L4 chain not landed yet
        h_c4 = chain_l4_crossing(up4, dn4, hz=val)
        tables_at_L = {4: (up4, dn4)}
        for L in (5, 6):
            window = chain_link_window(val, L, h_c4)
            for branch in ("up", "dn"):
                anchor = chain_anchor(val, branch)
                already_h = idx.get((round(hy, 4), cut, L, f"chain_{branch}"), set())
                new_h = sorted({h for h in window if round(h, 4) not in already_h},
                                key=lambda h: abs(h - anchor))
                if not new_h:
                    continue
                # refusal (a) applies to COLD starts only; this tier exclusively
                # emits a WARM batch job (see _chain_link_job_spec) so it never
                # arises here -- exercised directly in _selftest instead.
                # refusal (b): the checkpoint pool is ONLY what this branch has
                # actually had submitted (per the manifest) -- never assume the
                # anchor exists just because it's the seed value; an L5/6 anchor
                # not yet in the manifest means this branch has no checkpoint at
                # all yet, so its link job must be refused, not silently seeded
                # from the anchor's theoretical field value.
                init_h = nearest_same_branch_checkpoint(already_h, new_h[0])
                if init_h is None:
                    continue        # refusal (b): no same-branch checkpoint yet
                table4 = tables_at_L[4][0 if branch == "up" else 1]
                cutoff = spinodal_cutoff(branch_points_by_distance(table4, anchor))
                new_h = [h for h in new_h if not beyond_spinodal(h, anchor, cutoff)]
                if not new_h:
                    notes.append(f"[plan] {cut} L{L} {branch}: all candidates "
                                 f"beyond the L4 spinodal ({cutoff}) -- refused")
                    continue
                is_anchor_ckpt = abs(init_h - anchor) < 1e-9
                ckpt = (chain_anchor_run_name(L, anchor, val, hy) if is_anchor_ckpt
                        else chain_link_run_name(L, init_h, val, hy, branch))
                # CRUCIAL #2: INIT_FROM must reference a LANDED, HEALTHY checkpoint
                # on disk -- manifest presence alone only proves it was submitted,
                # not that it finished cleanly. A FINISHED-but-bad anchor (diverged,
                # or never below the h=0 bound) still holds below; an anchor that's
                # merely still in flight instead gets queued NOW, dependent on its
                # own jobid (queue-age fix) -- tc3d.sweep's own point-0 health gate
                # is the safety net if it lands unhealthy before the link starts.
                ckpt_dir = os.path.join(results_dir, cut, f"L{L}")
                ok, reason = checkpoint_health(ckpt_dir, ckpt, L)
                if not ok:
                    dep = None
                    if is_anchor_ckpt:
                        jid = anchor_jobid(rows, hy, cut, L, branch, anchor)
                        cls, state = anchor_watch_class(watch_state, ckpt)
                        if jid and cls == "inflight":
                            dep = f"afterok:{jid},singleton"
                            notes.append(f"[plan] chain {branch} L{L}: anchor {ckpt} "
                                         f"jobid={jid} state={state} -- queuing link early")
                        elif cls == "bad":
                            notes.append(f"[plan] hold chain {branch}: anchor {ckpt} "
                                         f"jobid={jid} state={state}")
                            continue
                        # cls == "completed" (or no jobid on record): trust the
                        # checkpoint_health verdict above -- fall through to hold.
                    if dep is None:
                        notes.append(f"[plan] hold chain {branch}: anchor not landed/unhealthy ({reason})")
                        continue
                    spec = _chain_link_job_spec(cut, val, L, hy, branch, new_h, ckpt)
                    spec["dependency"] = dep
                    t.append(spec)
                    continue
                t.append(_chain_link_job_spec(cut, val, L, hy, branch, new_h, ckpt))
    tiers.append(t)

    # tier: refine -----------------------------------------------------------
    # electric: any L, AT MOST ONE round (2 points at h_c +- 0.02), triggered
    # only by a fit error > 0.02 or an unbracketed rise (h_c falls outside the
    # fit's own observed h-range -- the sigmoid is still rising at the window
    # edge, so the fit extrapolated rather than bracketed a real inflection);
    # an unbracketed rise gets the outward-extension rule instead (ONE point
    # `step` past the nearest edge, in the rise's direction).
    t = []
    for cut in cut_ids:
        kind, val = _CUTS_BY_ID[cut]
        if kind != "electric":
            continue
        for L in (4, 5, 6):
            h_c, err, h_range = electric_fit_at(val, hy, L, results_dir)
            if h_c is None or err is None:
                continue
            n_prior = len(idx.get((round(hy, 4), cut, L, "refine"), set()))
            if n_prior >= 1:
                continue                                    # at most ONE round, ever
            points, unbracketed = electric_refine_points(h_c, h_range)
            if err <= 0.02 and not unbracketed:
                continue
            if unbracketed:
                notes.append(f"[plan] {cut} L{L}: unbracketed rise (h_c={h_c:.3f} outside "
                             f"[{h_range[0]:.3f},{h_range[1]:.3f}]) -- extending outward "
                             f"to {points[0]}")
            for hz in points:
                if not already_submitted(idx, hy, cut, L, "refine", hz):
                    t.append(_electric_spec(cut, val, L, hy, hz, refs, role="refine"))

    # chain: AT MOST ONE insert per (cut, L, branch) -- the single 0.025 point
    # adjacent to a >0.05 crossing bracket on that branch's own side (up:
    # bracket_lo+0.025; dn: bracket_hi-0.025), gated on the PRIMARY locator
    # error (>0.02, kind-dependent per chain_primary_fit_at) so a
    # well-resolved crossing never triggers a round. Warm-started from the
    # nearest landed healthy link on the SAME branch, outside the bracket.
    for cut in cut_ids:
        kind, val = _CUTS_BY_ID[cut]
        if kind != "magnetic":
            continue
        for L in (4, 5, 6):
            up_L, dn_L = chain_tables_at(val, hy, L, results_dir)
            if up_L is None or dn_L is None or len(up_L.h) == 0 or len(dn_L.h) == 0:
                continue
            try:
                import firstorder_fit as ff
            except ImportError:
                continue
            _h_c, _h_c_err, bracket, _info = ff.energy_crossing(up_L, dn_L)
            if bracket is None or (bracket[1] - bracket[0]) <= 0.05:
                continue
            p_h_c, p_err = chain_primary_fit_at(val, hy, L, results_dir)
            if p_err is None or p_err <= 0.02:
                continue
            lo, hi = bracket
            for branch, side, table, h_insert in (
                    ("up", "lo", up_L, round(lo + 0.025, 4)),
                    ("dn", "hi", dn_L, round(hi - 0.025, 4))):
                role_key = f"chain_{branch}_refine"
                if idx.get((round(hy, 4), cut, L, role_key), set()):
                    continue                                # at most ONE insert, ever
                outer_h = _outer_checkpoint(table, bracket, side)     # refusal (b): same branch only
                if outer_h is None:
                    continue
                anchor = chain_anchor(val, branch)
                cutoff = spinodal_cutoff(branch_points_by_distance(table, anchor))
                if beyond_spinodal(h_insert, anchor, cutoff):          # refusal (c)
                    continue
                is_anchor_ckpt = abs(outer_h - anchor) < 1e-9
                ckpt = (chain_anchor_run_name(L, anchor, val, hy) if is_anchor_ckpt
                        else chain_link_run_name(L, outer_h, val, hy, branch))
                ckpt_dir = os.path.join(results_dir, cut, f"L{L}")
                ok, reason = checkpoint_health(ckpt_dir, ckpt, L)      # CRUCIAL #2
                if not ok:
                    notes.append(f"[plan] hold chain {branch}: anchor not landed/unhealthy ({reason})")
                    continue
                t.append(_chain_link_job_spec(cut, val, L, hy, branch, [h_insert], ckpt, role="refine"))
    tiers.append(t)

    all_specs = [s for tier in tiers for s in tier]
    if max_new is not None and max_new >= 0:
        kept, deferred = all_specs[:max_new], all_specs[max_new:]
    else:
        kept, deferred = all_specs, []
    return kept, [s["jobname"] for s in deferred], notes


def _bash_line(spec):
    # NOTE: "-" (not "") marks an absent array/dependency -- IFS=$'\t' in bash
    # still treats tab as "IFS whitespace" and COLLAPSES consecutive empty
    # fields (unlike a true delimiter IFS), so an empty field here would shift
    # every later column in the reader's `read -r ... <<<"$line"`.
    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in spec["env"].items())
    return "\t".join([
        spec["jobname"], spec["wrapper"], spec["walltime"], spec.get("array") or "-",
        spec.get("dependency") or "-", spec["out_dir_rel"], spec["role"],
        spec["cut"], str(spec["L"]),
        " ".join(str(h) for h in spec["h_list"]), env_str,
    ])


def main_plan(argv):
    p = argparse.ArgumentParser(prog="phase3d_grid.py plan")
    p.add_argument("--hy", type=_parse_hy, required=True,
                   help="plane value (float) or 'y' for the y-cut pseudo-plane")
    p.add_argument("--results", required=True, help="the hy plane's OWN dir, e.g. $BASE_OUT/hy0.0 (ycuts/ for 'y')")
    p.add_argument("--manifests", required=True)
    p.add_argument("--max_new", type=int, default=None)
    p.add_argument("--cuts", default=None)
    p.add_argument("--bash", action="store_true",
                    help="print TAB-separated lines for the bash launcher instead of JSON")
    a = p.parse_args(argv)
    cut_ids = [c.strip() for c in a.cuts.replace(",", " ").split()] if a.cuts else None
    specs, deferred, notes = plan(a.hy, a.results, a.manifests, cut_ids, a.max_new)
    for n in notes:
        print(n, file=sys.stderr)
    if deferred:
        print(f"[plan] deferred (ceiling): {', '.join(deferred)}", file=sys.stderr)
    if a.bash:
        for s in specs:
            print(_bash_line(s))
    else:
        print(json.dumps({"specs": specs, "deferred": deferred}, indent=1))


def retry_spec(final_json, overrides):
    """Rebuild the planner's spec for ONE landed single-point run (electric cold
    point or chain anchor) from its final JSON, with env knob overrides -- the
    retry path for a GENUINE DIVERGENCE / unstable anchor. Returns
    (spec, swept_h, name, out_dir)."""
    with open(final_json) as fh:
        cfg = json.load(fh)["config"]
    L, hx, hy, hz = int(cfg["L"]), float(cfg["hx"]), float(cfg.get("hy", 0.0)), float(cfg["hz"])
    global _ACTIVE_HY
    _ACTIVE_HY = hy
    out_dir = os.path.dirname(os.path.abspath(final_json))
    cut = os.path.basename(os.path.dirname(out_dir))
    if cut.startswith("electric"):
        spec, h = _electric_spec(cut, hx, L, hy, hz), hz
    elif cut.startswith("magnetic"):
        branch = "up" if abs(hx - chain_anchor(hz, "up")) < 1e-9 else "dn"
        if abs(hx - chain_anchor(hz, branch)) > 1e-9:
            raise SystemExit(f"[retry] {final_json}: hx={hx} is a chain LINK, not an anchor -- "
                             "retry the chain job instead")
        spec, h = _chain_anchor_spec(cut, hz, L, hy, branch), hx
        if L == 4:
            # L4 anchors are point 0 of the combined branch job and carry the branch
            # suffix; the gap-fill tier (dn table + INIT_FROM) only sees that name.
            spec["env"]["NAME"] = chain_link_run_name(4, hx, hz, hy, branch)
    else:
        raise SystemExit(f"[retry] unknown cut dir {cut!r}")
    spec["env"].update(overrides)
    return spec, h, os.path.basename(final_json)[:-5], out_dir


def retry_forget(manifests_dir, out_dir, name_prefix, h):
    """Drop the point's rows (same out_dir + h) from every manifest so plan()
    stops deduping it; originals are copied to <manifests_dir>_bak/. Returns
    the old jobids."""
    bak = manifests_dir.rstrip("/") + "_bak"
    os.makedirs(bak, exist_ok=True)
    old = []
    for path in sorted(glob.glob(os.path.join(manifests_dir, "manifest_*.tsv"))):
        with open(path) as fh:
            lines = fh.readlines()
        keep = [lines[0]]
        for ln in lines[1:]:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 8 and f[7] == out_dir and abs(float(f[5]) - h) < 1e-9:
                old.append(f[0])
            else:
                keep.append(ln)
        if len(keep) != len(lines):
            shutil.copy2(path, os.path.join(bak, os.path.basename(path)))
            with open(path, "w") as fh:
                fh.writelines(keep)
    return old


def main_retry(argv):
    p = argparse.ArgumentParser(prog="phase3d_grid.py retry",
                                 description="park one landed point, forget it in the manifests, "
                                             "and print its resubmission line (plan --bash format)")
    p.add_argument("--final", required=True, help="the point's final JSON")
    p.add_argument("--manifests", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="env knob override, e.g. DIAG_SHIFT=5e-3 (repeatable)")
    a = p.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in a.set)
    spec, h, name, out_dir = retry_spec(a.final, overrides)
    old = retry_forget(a.manifests, out_dir, name, h)
    park = os.path.join(out_dir, f"redo_{old[-1] if old else 'manual'}")
    os.makedirs(park, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, name + ".*")):
        shutil.move(f, park)
    print(f"[retry] {name}: forgot jobid(s) {old or '-'}, parked outputs in {park}, "
          f"overrides {overrides}", file=sys.stderr)
    print(_bash_line(spec))


_SWEEP_OF_CUT = (("ycut_", "hy"), ("electric_", "hz"), ("magnetic_", "hx"))


def extend_spec(init_json, values, overrides=None, walltime=None):
    """Warm-started continuation of ONE chain branch (plane chain or y-cut) from a
    healthy landed point: train `values` in order, seeded from `init_json`'s
    checkpoint -- the fix for a chain that diverged/stopped or whose links are
    unconverged. retry_spec refuses chain links; this is their counterpart.
    The caller parks any existing outputs at `values` first (sweep.py SKIPS a
    point whose final JSON exists and RESUMES a leftover .ckpt) and launches
    the line with HY = the plane value, or `y` for a y-cut."""
    with open(init_json) as fh:
        cfg = json.load(fh)["config"]
    name = os.path.basename(init_json)[:-5]
    branch = name.rsplit("_", 1)[-1]
    if branch not in ("up", "dn"):
        raise SystemExit(f"[extend] {name}: not a chain point (no _up/_dn suffix)")
    out_dir = os.path.dirname(os.path.abspath(init_json))
    cut = os.path.basename(os.path.dirname(out_dir))
    plane_dir = os.path.basename(os.path.dirname(os.path.dirname(out_dir)))
    sweep = next((s for p, s in _SWEEP_OF_CUT if cut.startswith(p)), None)
    if sweep is None:
        raise SystemExit(f"[extend] unknown cut dir {cut!r}")
    L, hx, hy, hz = int(cfg["L"]), float(cfg["hx"]), float(cfg.get("hy", 0.0)), float(cfg["hz"])
    fixed = {"hx": hx, "hy": hy, "hz": hz}
    fixed.pop(sweep)
    jobname = {"hy": f"p3d_y_hx{hx:g}_hz{hz:g}_L{L}_{branch}",
               "hz": f"p3d_hy{hy}_e{hx:g}_L{L}_{branch}",
               "hx": f"p3d_hy{hy}_m{hz}_L{L}_{branch}"}[sweep]
    env = {**arch_env(L), **speed_env(L, hy), "L": str(L), "SWEEP": sweep,
           **{k.upper(): str(v) for k, v in fixed.items()}, sweep.upper(): str(values[0]),
           "FIELD_VALUES": " ".join(str(h) for h in values), "CHUNK_POINTS": str(len(values)),
           "WARM_START": "1", "INIT_FROM": name,
           "NAME_TEMPLATE": (f"gridinv_dual_L{{L}}_OBC_hx{{hx}}_hz{{hz}}_hy{{hy}}"
                             f"_n2x4_nh4-8_inv8-8_k{kernel_for(L)}_{branch}"),
           "DT": "0.005", "LR_MIN": "0.0005", "DIAG_SHIFT": "3e-3", "N_ITER": "300",
           "CKPT_EVERY": "10", "EXTRA_ARGS": SNAP_ARGS,
           "WANDB_PROJECT": WANDB_PROJECT_VAL, "WANDB_GROUP": jobname,
           "AUTO_RESUBMIT": "1", "TOPO_POOLED": "1"}
    if chunk_for(L):
        env["CHUNK"] = chunk_for(L)
    env.update(overrides or {})
    return {"role": f"chain_{branch}_extend", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": list(values), "env": env, "dependency": "singleton",
            "walltime": walltime or walltime_for(L, hy, chain=True), "array": "0",
            "out_dir_rel": f"{plane_dir}/{cut}/L{L}"}


def main_extend(argv):
    p = argparse.ArgumentParser(prog="phase3d_grid.py extend",
                                 description="print a plan --bash line continuing one chain branch "
                                             "from a healthy landed point (see extend_spec)")
    p.add_argument("--init", required=True, help="final JSON of the point to warm-start from")
    p.add_argument("--values", required=True, type=float, nargs="+", help="field values, in chain order")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="env knob override, e.g. DIAG_SHIFT=1e-2 (repeatable)")
    p.add_argument("--walltime", default=None)
    a = p.parse_args(argv)
    print(_bash_line(extend_spec(a.init, a.values, dict(kv.split("=", 1) for kv in a.set), a.walltime)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    _selftest_plan()
    _selftest_health_and_clamps()
    _selftest_chain_link_early_submit()
    assert _parse_hy("0.6") == 0.6 and _parse_hy(YCUT_HY) == YCUT_HY   # plan() needs a float hy
    if argv and argv[0] == "plan":
        return main_plan(argv[1:])
    if argv and argv[0] == "retry":
        return main_retry(argv[1:])
    if argv and argv[0] == "extend":
        return main_extend(argv[1:])
    _selftest()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry", action="store_true", help="human-readable dump")
    p.add_argument("--json", action="store_true", help="JSON dump")
    p.add_argument("--refs", action="store_true", help="dump the QMC hy=0 ref table as JSON")
    p.add_argument("--ref_lookup", action="store_true",
                    help="print 'E E_err' for one --hx/--hz/--L point (nothing if absent)")
    p.add_argument("--emit", action="store_true",
                    help="shell-friendly dump of one --cuts/--L/--hy cell (see emit_cell)")
    p.add_argument("--hy", type=_parse_hy, default=None, help="plane value, or y for the y-cuts")
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--hx", type=float, default=None)
    p.add_argument("--hz", type=float, default=None)
    p.add_argument("--cuts", default=None, help="space/comma-separated cut ids (default: all 10)")
    p.add_argument("--results_root", default="results")
    a = p.parse_args(argv)

    if a.emit:
        if None in (a.L, a.hy) or not a.cuts:
            p.error("--emit needs exactly one --cuts, --L and --hy")
        cut_ids = [c.strip() for c in a.cuts.replace(",", " ").split()]
        if len(cut_ids) != 1:
            p.error("--emit needs exactly one --cuts")
        emit_cell(cut_ids[0], a.L, a.hy)
        return

    if a.ref_lookup:
        refs = build_refs(a.results_root)
        rec = ref_lookup(refs, a.hx, a.hz, a.L) if None not in (a.hx, a.hz, a.L) else None
        if rec:
            print(f"{rec['E']} {rec['E_err']}")
        return

    if a.refs:
        refs = build_refs(a.results_root)
        print(json.dumps(sorted(refs.values(), key=lambda r: (r["L"], r["hx"], r["hz"])), indent=1))
        return

    hys = [a.hy] if a.hy is not None else HY_VALUES
    ls = [a.L] if a.L is not None else L_VALUES
    cut_ids = [c.strip() for c in a.cuts.replace(",", " ").split()] if a.cuts else None
    grid = campaign_grid(hys, ls, cut_ids)
    n_jobs, n_points = job_count(grid)
    if a.json:
        print(json.dumps(grid, indent=1))
    else:
        _dump_dry(grid)
        print(f"\n{n_jobs} jobs, {n_points} points total.")


if __name__ == "__main__":
    main()
