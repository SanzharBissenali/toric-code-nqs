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
import glob
import json
import os
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


# ---- magnetic/tail (1st-order) warm-chain links: ascending "up" list; "dn" is
# the same set traversed in reverse (mirror), per the task spec. ------------------
def _frange(lo, hi, step):
    n = round((hi - lo) / step)
    return [round(lo + i * step, 4) for i in range(n + 1)]


def _outer_inner_outer(start, end, inner_lo, inner_hi, outer=0.1, inner=0.05):
    """start..end ascending, `outer` spacing outside [inner_lo,inner_hi], `inner`
    spacing inside (both boundaries included in the inner segment)."""
    left, x = [], start
    while x < inner_lo - 1e-9:
        left.append(round(x, 4))
        x = round(x + outer, 4)
    right, x = [], end
    while x > inner_hi + 1e-9:
        right.append(round(x, 4))
        x = round(x - outer, 4)
    return left + _frange(inner_lo, inner_hi, inner) + list(reversed(right))


# literal per the task spec (hz in {0.0,0.1,0.2} share one hand-specified list)
_UP_LINKS_LOW = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
_CHAIN_LINKS_UP = {
    0.0: _UP_LINKS_LOW,
    0.1: _UP_LINKS_LOW,
    0.2: _UP_LINKS_LOW,
    0.4: _outer_inner_outer(0.6, 1.2, 0.75, 1.05),
    0.7: _outer_inner_outer(0.75, 1.35, 0.9, 1.2),
    1.0: _outer_inner_outer(0.95, 1.55, 1.1, 1.4),
}
_ANCHORS = {0.0: (0.6, 1.25), 0.1: (0.6, 1.25), 0.2: (0.6, 1.25),
            0.4: (0.5, 1.3), 0.7: (0.6, 1.5), 1.0: (0.8, 1.7)}


def chain_links(hz, branch):
    """branch: 'up' (low anchor carried toward the crossing) or 'dn' (high anchor,
    mirrored -- same point set, descending order)."""
    up = _CHAIN_LINKS_UP[round(hz, 4)]
    return list(up) if branch == "up" else list(reversed(up))


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

    up_low = chain_links(0.1, "up")
    assert up_low == _UP_LINKS_LOW
    assert chain_links(0.1, "dn") == list(reversed(_UP_LINKS_LOW))
    for hz in MAGNETIC_HZ + TAIL_HZ:
        up, dn = chain_links(hz, "up"), chain_links(hz, "dn")
        assert up == list(reversed(dn)) and dn == list(reversed(up))
        lo, hi = _ANCHORS[hz]
        assert lo < min(up) and max(up) < hi, (hz, lo, up, hi)

    # the two outer/inner ranges from the spec
    assert _outer_inner_outer(0.6, 1.2, 0.75, 1.05) == [
        0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    assert _outer_inner_outer(0.95, 1.55, 1.1, 1.4) == [
        0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.55]

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
    # 63 electric points; magnetic hz{0.0,0.2} (9 links/branch): 2 hz*3L*20=120;
    # tail+hz0.4 (11 links/branch): 3 hz*3L*24=216 -> 63+120+216=399.
    assert n_points == 63 + 120 + 216 == 399, n_points
    # stderr only -- --refs/--ref_lookup/--json emit machine-readable stdout
    print("[selftest] ok (grid math)", file=sys.stderr, flush=True)


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


def main(argv=None):
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
