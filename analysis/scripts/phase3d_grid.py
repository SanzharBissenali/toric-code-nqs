"""Deterministic Stage-1 grid for the phase3d campaign: hy in {0.0,0.2,0.4} x
L in {4,5,6} x 10 cuts (4 electric + 3 magnetic + 3 tail). Pure python,
NetKet-free -- consumed by nersc/launch_phase3d.sh and inspectable standalone.
Fixed campaign decisions (do not change without re-deriving the physics):
see notes/transition_mapping_recipes.md and the A3 task spec.

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
import sys

HY_VALUES = [0.0, 0.2, 0.4]
L_VALUES = [4, 5, 6]
ELECTRIC_HX = [0.0, 0.2, 0.5, 0.8]
MAGNETIC_HZ = [0.0, 0.1, 0.2]
TAIL_HZ = [0.4, 0.7, 1.0]
DEFAULT_SKIP_CUTS = {"electric_hx0.2", "magnetic_hz0.1"}   # already run at hy=0 (L4-6) / hy=0.2,0.4 (L4)

# ---- electric (2nd-order) grid: 7 points, center + fixed offsets, rounded 0.01 ---
ELECTRIC_OFFSETS = [-0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12]
_BASE_L = {4: 0.30, 5: 0.27, 6: 0.26}
_DHX = {0.0: 0.00, 0.2: 0.00, 0.5: 0.02, 0.8: 0.08}
_DHY = {0.0: 0.0, 0.2: -0.006, 0.4: -0.023}   # empirical, ~quadratic in hy


def electric_center(hx, L, hy):
    return _BASE_L[L] + _DHX[hx] + _DHY[hy]


def electric_grid(hx, L, hy):
    c = electric_center(hx, L, hy)
    return [round(c + d, 2) for d in ELECTRIC_OFFSETS]


# ---- magnetic/tail (1st-order) warm-chain links: 7-8 good points/branch/L is
# enough (trimmed from the original 9-11); up and dn are NOT mirrors of one
# set -- each brackets the crossing from its own side. -----------------------
_ANCHORS = {0.0: (0.6, 1.25), 0.1: (0.6, 1.25), 0.2: (0.6, 1.25),
            0.4: (0.5, 1.3), 0.7: (0.6, 1.5), 1.0: (0.8, 1.7)}

# literal per the task spec (hz in {0.0,0.1,0.2} share one hand-specified PAIR
# of 6-link lists).
_LOW_LINKS_UP = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
_LOW_LINKS_DN = [1.15, 1.05, 1.0, 0.95, 0.9, 0.85]


def _tail_links(hz):
    """Tail (hz in {0.4,0.7,1.0}) 6-link-per-branch chain: mirrors the rule-1
    SHAPE (one 0.1-spaced point, then five 0.05-spaced points spanning the
    crossing +-0.10, ending 0.10 past it on the branch's far side) around that
    cut's own seeded crossing -- the anchor midpoint, the same `seed_center`
    value `chain_link_window` already uses."""
    c = 0.5 * sum(_ANCHORS[hz])
    up = [round(c + d, 4) for d in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1)]
    dn = [round(c + d, 4) for d in (0.2, 0.1, 0.05, 0.0, -0.05, -0.1)]
    return up, dn


_CHAIN_LINKS_UP = {0.0: _LOW_LINKS_UP, 0.1: _LOW_LINKS_UP, 0.2: _LOW_LINKS_UP}
_CHAIN_LINKS_DN = {0.0: _LOW_LINKS_DN, 0.1: _LOW_LINKS_DN, 0.2: _LOW_LINKS_DN}
for _hz in TAIL_HZ:
    _CHAIN_LINKS_UP[_hz], _CHAIN_LINKS_DN[_hz] = _tail_links(_hz)


def chain_links(hz, branch):
    """branch: 'up' (low anchor, links bracket the crossing from below) or
    'dn' (high anchor, links bracket the crossing from above) -- each
    branch's own 6-link list, looked up directly (not a mirrored/reversed
    pair of a single shared set)."""
    table = _CHAIN_LINKS_UP if branch == "up" else _CHAIN_LINKS_DN
    return list(table[round(hz, 4)])


def chain_anchor(hz, branch):
    lo, hi = _ANCHORS[round(hz, 4)]
    return lo if branch == "up" else hi


# ---- cut registry -----------------------------------------------------------
def all_cuts():
    """[(cut_id, kind, fixed_value), ...] -- kind in {electric, magnetic}
    (tail cuts are `kind=magnetic` too; only the hz value distinguishes them,
    per the task's OUT_DIR/naming convention: both live under magnetic_hz{hz})."""
    cuts = [(f"electric_hx{hx}", "electric", hx) for hx in ELECTRIC_HX]
    cuts += [(f"magnetic_hz{hz}", "magnetic", hz) for hz in MAGNETIC_HZ + TAIL_HZ]
    return cuts


_CUTS_BY_ID = {cid: (kind, val) for cid, kind, val in all_cuts()}


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
    return {"kind": "magnetic", "hz": val,
            "up": {"anchor": chain_anchor(val, "up"), "links": chain_links(val, "up")},
            "dn": {"anchor": chain_anchor(val, "dn"), "links": chain_links(val, "dn")}}


def campaign_grid(hys=None, ls=None, cut_ids=None):
    hys = HY_VALUES if hys is None else hys
    ls = L_VALUES if ls is None else ls
    cut_ids = [c for c, _, _ in all_cuts()] if cut_ids is None else cut_ids
    return {hy: {cid: {L: points_for(cid, L, hy) for L in ls} for cid in cut_ids}
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

    assert chain_links(0.1, "up") == _LOW_LINKS_UP
    assert chain_links(0.1, "dn") == _LOW_LINKS_DN
    for hz in MAGNETIC_HZ + TAIL_HZ:
        up, dn = chain_links(hz, "up"), chain_links(hz, "dn")
        assert len(up) == 6 and len(dn) == 6, (hz, up, dn)
        assert up == sorted(up) and dn == sorted(dn, reverse=True), (hz, up, dn)
        lo, hi = _ANCHORS[hz]
        assert lo < min(up) and max(up) < hi, (hz, lo, up, hi)
        assert lo < min(dn) and max(dn) < hi, (hz, lo, dn, hi)

    # tail shape: one 0.1-spaced point then five 0.05-spaced points, ending
    # +-0.10 past the cut's own seeded crossing (rule-1's shape, re-centred).
    for hz in TAIL_HZ:
        c = 0.5 * sum(_ANCHORS[hz])
        up, dn = chain_links(hz, "up"), chain_links(hz, "dn")
        assert up[0] == round(c - 0.2, 4) and up[-1] == round(c + 0.1, 4), (hz, up)
        assert dn[0] == round(c + 0.2, 4) and dn[-1] == round(c - 0.1, 4), (hz, dn)

    assert len(all_cuts()) == 10
    cids = {c for c, _, _ in all_cuts()}
    assert cids == {
        "electric_hx0.0", "electric_hx0.2", "electric_hx0.5", "electric_hx0.8",
        "magnetic_hz0.0", "magnetic_hz0.1", "magnetic_hz0.2",
        "magnetic_hz0.4", "magnetic_hz0.7", "magnetic_hz1.0"}

    # one hy plane, default (8) cuts: 3 electric hx * 3 L * 7 pts = 63 electric jobs;
    # 5 magnetic/tail hz * 3 L * 2 chains = 30 chain jobs -> 93 jobs/plane.
    default_cuts = [c for c in cids if c not in DEFAULT_SKIP_CUTS]
    g = campaign_grid(hys=[0.0], ls=L_VALUES, cut_ids=default_cuts)
    n_jobs, n_points = job_count(g)
    assert n_jobs == 93, n_jobs
    # 63 electric points; every magnetic/tail hz now shares the trimmed 6
    # links/branch (anchor+6=7/branch, 14/cell): 5 hz*3L*14=210 -> 63+210=273.
    assert n_points == 63 + 210 == 273, n_points
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
    w = chain_link_window(0.4, 6, 0.5 * sum(_ANCHORS[0.4]) + 5.0)   # wildly wrong crossing
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
    seed_centre = 0.5 * sum(_ANCHORS[hz])
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
# Peer fit modules (transition_fit.py, firstorder_fit.py) are dropped into this
# worktree UNTRACKED by convention (identical copies live untracked in every
# p3d/* worktree) -- imported lazily, NEVER git-added here.
# =============================================================================
WANDB_PROJECT_VAL = "tc3d-phase3d"
SNAP_ARGS = "--snapshot_every 50 --final_eval_rounds 8"
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


def walltime_for(L, hy):
    # Fast path (dense conv + float32, 2026-09-10): L4c 3.2 s/step, L5r 6.6, L6r 21.9.
    # Budget = 8-link train (200 steps each) + compile + observables, ~1.5x margin;
    # L6 real (4.9 h) and L5 complex (15 s/step) still need the 5 h cap + resubmit.
    nz = float(hy) != 0.0
    if L == 4:
        return "02:00:00" if nz else "01:30:00"
    if L == 5:
        return "05:00:00" if nz else "02:00:00"
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


def submitted_index(rows):
    """(hy,cut,L,role) -> {h, ...} already recorded in some manifest."""
    idx = collections.defaultdict(set)
    for r in rows:
        try:
            key = (round(float(r["hy"]), 4), r["cut"], int(r["L"]), r["role"])
            idx[key].add(round(float(r["h"]), 4))
        except (KeyError, ValueError, TypeError):
            continue
    return idx


def already_submitted(idx, hy, cut, L, role, h):
    return round(float(h), 4) in idx.get((round(float(hy), 4), cut, L, role), set())


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
    """L4 h_c for recentring, by cut kind (the campaign's locator policy): hz <= 0.2
    (topological -> trivial) uses the membrane-O_FM inflection on the lowest-energy
    winner curve -- the L4 energy crossing there is a 200-step convergence artifact
    (hz=0: up links 0.85-0.95 sit 5-8 above the dn branch, "crossing" at 0.999);
    hz >= 0.4 (trivial -> trivial) uses the energy crossing. None when merged/no
    overlap/no fit ("if merged, keep the seeded links")."""
    if up4 is None or dn4 is None or len(up4.h) == 0 or len(dn4.h) == 0:
        return None
    import firstorder_fit as ff
    if hz is not None and float(hz) <= 0.2:
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
    seed_center = 0.5 * sum(_ANCHORS[round(hz, 4)])
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
def _electric_spec(cut, hx, L, hy, hz, refs=None, role="cold"):
    env = {**arch_env(L), **speed_env(L, hy), "L": str(L), "HX": str(hx), "HZ": str(hz), "HY": str(hy),
           "DT": "0.02", "LR_MIN": "0.002", "N_ITER": "500",
           "DIAG_SHIFT": diag_shift_for(L), "CKPT_EVERY": "10",
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
            "dependency": None, "walltime": walltime_for(L, hy), "array": None,
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
    anchor_ov = f'{{"dt":0.02,"lr_min":0.002,"n_iter":500,"diag_shift":{ds}}}'
    env = {**arch_env(L), **speed_env(L, hy), "L": str(L), "SWEEP": "hx", "HZ": str(hz), "HY": str(hy),
           "FIELD_VALUES": " ".join(str(h) for h in field_values),
           "CHUNK_POINTS": str(len(field_values)), "WARM_START": "1",
           "ANCHOR_OVERRIDES": anchor_ov, "NAME_TEMPLATE": name_tpl,
           "DT": "0.005", "LR_MIN": "0.0005", "DIAG_SHIFT": "3e-3", "N_ITER": "200",
           "CKPT_EVERY": "10", "EXTRA_ARGS": SNAP_ARGS,
           "WANDB_PROJECT": WANDB_PROJECT_VAL, "WANDB_GROUP": jobname,
           "AUTO_RESUBMIT": "1", "CHUNK": "2048"}
    return {"role": f"chain_{branch}", "cut": cut, "L": L, "wrapper": "batch",
            "jobname": jobname, "h_list": field_values, "env": env,
            "dependency": None, "walltime": walltime_for(L, hy), "array": "0",
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
           "DT": "0.005", "LR_MIN": "0.0005", "DIAG_SHIFT": "3e-3", "N_ITER": "200",
           "CKPT_EVERY": "10", "EXTRA_ARGS": SNAP_ARGS,
           "WANDB_PROJECT": WANDB_PROJECT_VAL, "WANDB_GROUP": jobname,
           "AUTO_RESUBMIT": "1"}
    chunk = chunk_for(L)
    if chunk:
        env["CHUNK"] = chunk
    return {"role": f"chain_{branch}" if role == "chain" else f"chain_{branch}_refine",
            "cut": cut, "L": L, "wrapper": "batch", "jobname": jobname,
            "h_list": list(new_h_sorted), "env": env, "dependency": "singleton",
            "walltime": walltime_for(L, hy), "array": "0",
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
        else:
            for branch in ("up", "dn"):
                anchor = chain_anchor(val, branch)
                if not already_submitted(idx, hy, cut, 4, f"chain_{branch}", anchor):
                    t.append(_chain_l4_job_spec(cut, val, hy, branch))
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
    p.add_argument("--hy", type=float, required=True)
    p.add_argument("--results", required=True, help="the hy plane's OWN dir, e.g. $BASE_OUT/hy0.0")
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    _selftest_plan()
    _selftest_health_and_clamps()
    _selftest_chain_link_early_submit()
    if argv and argv[0] == "plan":
        return main_plan(argv[1:])
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
    p.add_argument("--hy", type=float, default=None)
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
