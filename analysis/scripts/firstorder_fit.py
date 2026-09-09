"""First-order transition locators for field cuts of the 3D toric code (task A7).

A first-order cut (recipe notes/transition_mapping_recipes.md SB) is mapped by TWO
warm chains through the coexistence window -- an "up" branch carried from the
topological/z-polarized side and a "dn" branch carried from the x-polarized side --
plus cold anchors deep in each phase. Chain runs carry `_up` / `_dn` at the end of
their run name; anchors/cold points carry neither (recipe SB.1-2). The PRIMARY
locator is the energy branch crossing (order-parameter-free, SB.4-5): at each field
the lower-energy branch is the ground state, and the crossing field is where the
lower-energy branch switches. On the tail of the line (fixed h_z >= 0.4, sweep h_x)
BOTH sides are trivial (z-polarized -> x-polarized) so O_FM is not an order
parameter there -- M_x = sx_mean and <A_v> = A_v_mean are the requested SECONDARY
locators (a jump/inflection in a local observable also marks the transition).

Built on top of `transition_fit.py` (peer module, NOT modified here -- imported as
`tf`; see the "peer dependencies" note at the bottom of this file for the list of
things its owner should add/expose).

CLI:
    python analysis/scripts/firstorder_fit.py --runs DIR [DIR ...] --sweep hx \\
        --fixed hz=0.1 hy=0.0 [--ofm] [--kind topo-trivial|trivial-trivial] \\
        [--out results/transitions/<tag>.json] [--dry]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis", "scripts"))
import transition_fit as tf                                   # noqa: E402

# local observable -> its error-bar key (NOT a uniform "+_err" suffix rule, see the
# actual `observables` dicts: "sx_mean"/"sx_err" but "O_FM_membrane_R1"/"..._err")
LOCAL_ERR = {
    "sx_mean": "sx_err", "sz_mean": "sz_err", "sy_mean": "sy_err",
    "A_v_mean": "A_v_err", "B_p_mean": "B_p_err",
    "O_FM_membrane_R1": "O_FM_membrane_R1_err",
}
LOCAL_OBS = tuple(LOCAL_ERR)
JUMP_OBS_DEFAULT = ("sx_mean", "A_v_mean", "B_p_mean", "sz_mean")   # always-on secondary locators
OFM_OBS = "O_FM_membrane_R1"                                        # only when --ofm / want_ofm
HF_OBS = {"hx": "sx_mean", "hy": "sy_mean", "hz": "sz_mean"}        # Hellmann-Feynman conjugate obs

_BRANCH_RE = re.compile(r"_(up|dn)$")


def _branch_of(stem):
    m = _BRANCH_RE.search(stem)
    return m.group(1) if m else "cold"


# ----------------------------------------------------------------------------- data model
@dataclass
class Table:
    """Per-(branch, L) data: arrays over the swept field h, aligned index-for-index."""
    L: int
    h: np.ndarray
    E0: np.ndarray
    E_err: np.ndarray
    obs: dict                              # name -> (y, ye) arrays, name in LOCAL_OBS
    names: list
    diverged: np.ndarray
    branch: list = field(default_factory=list)   # only populated by winner()

    def curve(self, key):
        """Non-diverged transition_fit.Curve for one column (key = "E0" or a LOCAL_OBS)."""
        keep = ~self.diverged
        h = self.h[keep]
        names = [n for n, k in zip(self.names, keep) if k]
        if key == "E0":
            y, ye = self.E0[keep], self.E_err[keep]
            label = "E"
        else:
            y, ye = (a[keep] for a in self.obs[key])
            label = key
        return tf.Curve(self.L, h, y, ye, label, src="firstorder_fit", names=names)


# ----------------------------------------------------------------------------- loader
def load_branches(dirs, sweep, fixed, tol=1e-9):
    """{branch: {L: Table}} from per-run final-state JSONs, branch in {"up","dn","cold"}
    decided purely by the name suffix (recipe SB.1-2: chain runs end in `_up`/`_dn`;
    anchors/cold starts carry neither -> "cold"). Multiple runs at one (branch, L, h)
    (reruns, warm variants, seeds): the lowest-energy NON-diverged run wins; if every
    run at that point diverged, the lowest-energy diverged one is kept and FLAGGED --
    diverged points are never silently dropped, they mark a branch's spinodal (SB.3)
    and are excluded from fits downstream (Table.curve()), not from the table itself.
    """
    raw = {"up": {}, "dn": {}, "cold": {}}
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if not tf._is_final_json(f):
                continue
            try:
                j = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            c, o = j.get("config"), j.get("observables")
            if not c or not o or o.get("E0") is None or sweep not in c:
                continue
            if any(abs(float(c.get(k, 0.0)) - v) > tol for k, v in fixed.items()):
                continue
            L, h = int(c["L"]), round(float(c[sweep]), 6)
            row = {"E0": float(o["E0"]),
                   "E_err": float(o["E_err"]) if o.get("E_err") is not None else np.nan,
                   "diverged": bool(j.get("diverged", False)), "name": f.stem}
            for k, ek in LOCAL_ERR.items():
                row[k] = float(o[k]) if o.get(k) is not None else np.nan
                row[ek] = float(o[ek]) if o.get(ek) is not None else np.nan
            raw[_branch_of(f.stem)].setdefault((L, h), []).append(row)

    out = {}
    for branch, pts in raw.items():
        by_L = {}
        for (L, h), rows in pts.items():
            pool = [r for r in rows if not r["diverged"]] or rows   # non-diverged pool, else keep diverged
            best = min(pool, key=lambda r: r["E0"])
            by_L.setdefault(L, []).append((h, best))
        tabs = {}
        for L, plist in sorted(by_L.items()):
            plist.sort(key=lambda p: p[0])
            h = np.array([p[0] for p in plist])
            E0 = np.array([p[1]["E0"] for p in plist])
            E_err = np.array([p[1]["E_err"] for p in plist])
            names = [p[1]["name"] for p in plist]
            diverged = np.array([p[1]["diverged"] for p in plist], bool)
            obs = {k: (np.array([p[1][k] for p in plist]), np.array([p[1][LOCAL_ERR[k]] for p in plist]))
                   for k in LOCAL_OBS}
            tabs[L] = Table(L, h, E0, E_err, obs, names, diverged)
        out[branch] = tabs
    return out


def winner(tables):
    """{L: Table} -- per (L, h) the lowest-energy NON-diverged run across branches (same
    rule as transition_fit.load_runs); Table.branch carries the winning branch's label
    per point. A (L, h) point where every branch diverged has no surviving state and is
    dropped, exactly as load_runs drops all-diverged points."""
    best = {}   # (L, h) -> (E0, branch, i, Table)
    for branch, by_L in tables.items():
        for L, t in by_L.items():
            for i in range(len(t.h)):
                if t.diverged[i]:
                    continue
                key = (L, round(float(t.h[i]), 6))
                if key not in best or t.E0[i] < best[key][0]:
                    best[key] = (float(t.E0[i]), branch, i, t)

    by_L_pts = {}
    for (L, h), (E0, branch, i, t) in best.items():
        obs_here = {k: (t.obs[k][0][i], t.obs[k][1][i]) for k in LOCAL_OBS}
        by_L_pts.setdefault(L, []).append((h, E0, t.E_err[i], branch, t.names[i], obs_here))

    out = {}
    for L, plist in sorted(by_L_pts.items()):
        plist.sort(key=lambda p: p[0])
        h = np.array([p[0] for p in plist])
        E0 = np.array([p[1] for p in plist])
        E_err = np.array([p[2] for p in plist])
        branch = [p[3] for p in plist]
        names = [p[4] for p in plist]
        obs = {k: (np.array([p[5][k][0] for p in plist]), np.array([p[5][k][1] for p in plist]))
               for k in LOCAL_OBS}
        diverged = np.zeros(len(h), bool)   # winners are non-diverged by construction
        out[L] = Table(L, h, E0, E_err, obs, names, diverged, branch=branch)
    return out


# ----------------------------------------------------------------------------- crossing
def _bracket_h_c(h0, h1, d0, d1):
    return h0 - d0 * (h1 - h0) / (d1 - d0)


def energy_crossing(up, dn):
    """(h_c, h_c_err, bracket, info) -- the crossing of the two branches' energies on
    their common field grid (non-diverged points only): the first adjacent pair where
    sign(E_up - E_dn) flips, h_c from LINEAR interpolation of Delta_E(h) inside that
    bracket.

    Chosen over a global polynomial fit of each branch's E(h): recipe SB.6's "resonance
    blind spot" makes E(h) non-smooth right at the crossing (more steps / 2x-dt kicks
    provably don't fix it there), so a bracket-local secant is the assumption-light
    choice for the reported h_c. As a cross-check (only when >=4 non-diverged points
    survive per branch on the common grid), a degree-2 polynomial is ALSO fit to each
    branch and their crossing reported in info["poly_h_c"] -- if it disagrees with the
    bracket estimate outside errors, that's itself a resonance-window symptom.

    Error = the bracket interpolation's propagated E_err (via a finite-difference
    gradient of h_c(d0, d1)), in quadrature with bracket_width/sqrt(12) (the
    "uniform-within-the-bracket" component -- we only know the crossing is somewhere
    in [h0, h1], not where).

    info["reason"] is "no overlap" (no field value survives non-diverged on both
    branches, or a branch is missing/empty -- e.g. a cold-only campaign) or
    "branches merged" (a common grid exists but Delta_E never changes sign).
    """
    if up is None or dn is None or len(up.h) == 0 or len(dn.h) == 0:
        return None, None, None, {"reason": "no overlap", "n_common": 0, "min_abs_dE": None}

    mu, md = ~up.diverged, ~dn.diverged
    hu, Eu, Eeu = up.h[mu], up.E0[mu], up.E_err[mu]
    hd, Ed, Eed = dn.h[md], dn.E0[md], dn.E_err[md]
    hu_r, hd_r = np.round(hu, 6), np.round(hd, 6)
    common = sorted(set(hu_r) & set(hd_r))
    if len(common) < 2:
        return None, None, None, {"reason": "no overlap", "n_common": len(common), "min_abs_dE": None}

    delta, derr = [], []
    for h in common:
        eu, eeu = Eu[hu_r == h][0], Eeu[hu_r == h][0]
        ed, eed = Ed[hd_r == h][0], Eed[hd_r == h][0]
        delta.append(eu - ed)
        derr.append(float(np.hypot(eeu, eed)))
    hs, delta, derr = np.array(common), np.array(delta), np.array(derr)

    info = {"n_common": len(common), "min_abs_dE": float(np.min(np.abs(delta)))}
    if len(hs) >= 4:
        mask_u, mask_d = np.isin(hu_r, common), np.isin(hd_r, common)
        pu = np.polyfit(hu[mask_u], Eu[mask_u], 2)
        pd = np.polyfit(hd[mask_d], Ed[mask_d], 2)
        roots = np.roots(pu - pd)
        real = [r.real for r in roots if abs(r.imag) < 1e-6 and hs.min() <= r.real <= hs.max()]
        info["poly_h_c"] = float(sorted(real, key=lambda r: abs(r - np.median(hs)))[0]) if real else None

    flips = np.where(np.diff(np.sign(delta)) != 0)[0]
    if len(flips) == 0:
        info["reason"] = "branches merged"
        return None, None, None, info

    i = flips[0]
    h0, h1, d0, d1 = float(hs[i]), float(hs[i + 1]), float(delta[i]), float(delta[i + 1])
    e0, e1 = float(derr[i]), float(derr[i + 1])
    width = h1 - h0
    if d1 == d0:
        h_c, stat = 0.5 * (h0 + h1), 0.0
    else:
        h_c = _bracket_h_c(h0, h1, d0, d1)
        eps = 1e-9
        g0 = (_bracket_h_c(h0, h1, d0 + eps, d1) - _bracket_h_c(h0, h1, d0 - eps, d1)) / (2 * eps)
        g1 = (_bracket_h_c(h0, h1, d0, d1 + eps) - _bracket_h_c(h0, h1, d0, d1 - eps)) / (2 * eps)
        stat = float(np.hypot(g0 * e0, g1 * e1))
    h_c_err = float(np.hypot(stat, width / np.sqrt(12)))
    info["reason"] = "crossing"
    return float(h_c), h_c_err, (h0, h1), info


# ----------------------------------------------------------------------------- secondary locators
def jump_locators(curve, want_ofm=False):
    """{obs: {method: Fit}} on the winner Table `curve`, reusing transition_fit's
    locators: logistic + fd_peak always; richards only if it converges (its 6
    parameters are unreliable on the short, <~10-point windows typical of a
    first-order tail). obs = sx_mean, A_v_mean, B_p_mean, sz_mean, plus
    O_FM_membrane_R1 when want_ofm=True -- kept available for cross-checking against a
    topo-trivial cut's banked O_FM locators, even though O_FM is not an order
    parameter on a trivial-trivial tail."""
    out = {}
    for key in (*JUMP_OBS_DEFAULT, *((OFM_OBS,) if want_ofm else ())):
        c = curve.curve(key)
        if len(c.h) < 4 or not np.any(np.isfinite(c.y)):
            continue
        fits = {"logistic": tf.fit_logistic(c), "fd_peak": tf.fit_fd_peak(c)}
        r = tf.fit_richards(c)
        if r.ok():
            fits["richards"] = r
        out[key] = fits
    return out


def hellmann_feynman(curve, sweep="hx"):
    """Per adjacent pair on the (non-diverged, by construction) winner Table `curve`:
    compare -(Delta E / Delta h) with N * mean(obs) at the pair's midpoint (trapezoid
    average of the endpoints), N = 3 L^3 - 3 L^2 (OBC edge count). obs is the field's
    conjugate local operator (sx_mean for hx, sz_mean for hz, sy_mean for hy) -- the
    Hellmann-Feynman theorem dE/dh = -N <obs>, the same check CLAUDE.md's hy-lane uses
    for hy (recipe SC.4), generalized here to whichever field is swept. Returns
    {"rows": [...], "max_rel_dev": ..., "N": ..., "obs": key}."""
    key = HF_OBS.get(sweep, "sx_mean")
    h, E = curve.h, curve.E0
    y, _ye = curve.obs[key]
    N = 3 * curve.L ** 3 - 3 * curve.L ** 2
    rows = []
    for i in range(len(h) - 1):
        dh = h[i + 1] - h[i]
        if dh == 0 or not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue
        dEdh = -(E[i + 1] - E[i]) / dh
        hf = N * 0.5 * (y[i] + y[i + 1])
        rel = abs(dEdh - hf) / max(abs(hf), 1e-9)
        rows.append({"h_lo": float(h[i]), "h_hi": float(h[i + 1]), "dEdh": float(dEdh),
                     "N_mean_obs": float(hf), "rel_dev": float(rel)})
    max_dev = max((r["rel_dev"] for r in rows), default=np.nan)
    return {"rows": rows, "max_rel_dev": float(max_dev), "N": N, "obs": key}


def spinodals(tables):
    """{branch: {L: {"last_ok": h, "first_diverged": h}}} -- the last swept field where
    a branch still has a non-diverged run, and the first field where it diverged (None
    if it never did). Recipe SB.3: crash/shed marks a branch's spinodal."""
    out = {}
    for branch, by_L in tables.items():
        out[branch] = {}
        for L, t in by_L.items():
            ok, bad = t.h[~t.diverged], t.h[t.diverged]
            out[branch][L] = {"last_ok": float(ok.max()) if len(ok) else None,
                               "first_diverged": float(bad.min()) if len(bad) else None}
    return out


# ----------------------------------------------------------------------------- per-cut assembly
def locate_cut(dirs, sweep, fixed, want_ofm=False):
    """Per-L rows: {L, h_c, h_c_err, bracket, crossing_info, secondary, syst_jump,
    spinodals, hf_dev, n_up, n_dn, merged}. central = the energy crossing; if the
    branches never cross (merged, "no overlap", or a cold-only campaign with no
    _up/_dn files at all) h_c/h_c_err are None and the row is flagged merged=True, but
    `secondary` (the winner-curve jump locators) is still populated -- it is the
    fallback locator for that L."""
    tables = load_branches(dirs, sweep, fixed)
    wtabs = winner(tables)
    sp = spinodals(tables)
    rows = []
    for L in sorted(wtabs):
        wt = wtabs[L]
        up_t = tables.get("up", {}).get(L)
        dn_t = tables.get("dn", {}).get(L)
        h_c, h_c_err, bracket, info = energy_crossing(up_t, dn_t)
        jl = jump_locators(wt, want_ofm=want_ofm)
        hf = hellmann_feynman(wt, sweep=sweep)

        overlap = None
        if up_t is not None and dn_t is not None and len(up_t.h) and len(dn_t.h):
            overlap = (float(max(up_t.h.min(), dn_t.h.min())), float(min(up_t.h.max(), dn_t.h.max())))
        infl = [f.h_c for fits in jl.values() for f in fits.values()
                if f.ok() and (overlap is None or overlap[0] <= f.h_c <= overlap[1])]
        syst_jump = 0.5 * (max(infl) - min(infl)) if len(infl) > 1 else 0.0

        rows.append({
            "L": L, "h_c": h_c, "h_c_err": h_c_err,
            "bracket": list(bracket) if bracket else None, "crossing_info": info,
            "secondary": jl, "syst_jump": syst_jump,
            "spinodals": {b: sp.get(b, {}).get(L) for b in ("up", "dn", "cold")},
            "hf_dev": hf["max_rel_dev"],
            "n_up": int(len(up_t.h)) if up_t is not None else 0,
            "n_dn": int(len(dn_t.h)) if dn_t is not None else 0,
            "merged": h_c is None,
        })
    return rows


def _fit_tuple(f):
    return (f.h_c, f.h_c_err, f.chi2red)


def make_firstorder_record(tag, spec, rows, fss_main, fss_sweep_list, free=None, notes="",
                            kind="topo-trivial"):
    """Same JSON schema as transition_fit.make_record (results/transitions/<tag>.json
    consumers keep working), with kind ("topo-trivial" | "trivial-trivial") and
    obs="E_crossing"; per-L "secondary" locator dicts (Fit -> (h_c, h_c_err, chi2red)
    tuples, matching the banked-record "locators" convention) pass through `rows`
    otherwise untouched. Built THROUGH transition_fit.make_record (not reimplemented)
    so its quality gate / syst_exponent computation stays the single source of truth.
    """
    ser_rows = []
    for r in rows:
        rr = dict(r)
        rr["h_c"] = np.nan if rr.get("h_c") is None else rr["h_c"]
        rr["h_c_err"] = np.nan if rr.get("h_c_err") is None else rr["h_c_err"]
        rr["secondary"] = {obs: {m: _fit_tuple(f) for m, f in fits.items()}
                            for obs, fits in r["secondary"].items()}
        ser_rows.append(rr)
    spec = {**spec, "kind": kind, "obs": "E_crossing"}
    return tf.make_record(tag, spec, ser_rows, fss_main, fss_sweep_list, free, notes)


# ----------------------------------------------------------------------------- CLI
def _parse_fixed(pairs):
    fixed = {}
    for p in pairs:
        k, v = p.split("=")
        fixed[k] = float(v)
    return fixed


def _fmt(x, nd=4):
    return "None" if x is None else f"{x:.{nd}f}"


def _print_table(rows, want_ofm=False):
    obs_order = list(JUMP_OBS_DEFAULT) + ([OFM_OBS] if want_ofm else [])
    print(f"{'L':>3} {'h_c':>8} {'h_c_err':>8} {'bracket':>16} {'n_up':>4} {'n_dn':>4} "
          f"{'hf_dev':>7} {'merged':>7}  jump-logistic h_c (" + ",".join(obs_order) + ")")
    for r in rows:
        br = f"[{r['bracket'][0]:.3f},{r['bracket'][1]:.3f}]" if r["bracket"] else "-"
        vals = []
        for k in obs_order:
            fit = r["secondary"].get(k, {}).get("logistic")
            vals.append(f"{fit.h_c:.3f}" if fit is not None and fit.ok() else "-")
        print(f"{r['L']:>3} {_fmt(r['h_c']):>8} {_fmt(r['h_c_err']):>8} {br:>16} "
              f"{r['n_up']:>4} {r['n_dn']:>4} {_fmt(r['hf_dev'], 3):>7} {str(r['merged']):>7}  "
              + " ".join(vals))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="dirs of per-run final-state JSONs")
    ap.add_argument("--sweep", required=True, choices=tf.FIELDS)
    ap.add_argument("--fixed", nargs="+", required=True, help="key=val pairs, e.g. hz=0.1 hy=0.0")
    ap.add_argument("--ofm", action="store_true", help="also run the O_FM_membrane_R1 jump locator")
    ap.add_argument("--kind", default="topo-trivial", choices=["topo-trivial", "trivial-trivial"])
    ap.add_argument("--lane", default="phase3d",
                     help="tag suffix (default phase3d, NOT prod): an energy-crossing record must "
                          "not collide with an existing O_FM-based prod record at the same "
                          "(hy, fixed, sweep) point")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)

    fixed = _parse_fixed(args.fixed)
    hy = fixed.get("hy", 0.0)
    fixed_name = next((k for k in fixed if k != "hy"), args.sweep)
    fixed_val = fixed[fixed_name]
    tag = tf.cut_tag(hy, fixed_name, fixed_val, args.sweep, lane=args.lane)

    rows = locate_cut(args.runs, args.sweep, fixed, want_ofm=args.ofm)
    _print_table(rows, want_ofm=args.ofm)

    Ls = [r["L"] for r in rows if r["h_c"] is not None]
    hc = [r["h_c"] for r in rows if r["h_c"] is not None]
    hce = [r["h_c_err"] for r in rows if r["h_c_err"] is not None]
    fss_main = tf.fss_fit(Ls, hc, hce, x=1.0) if len(Ls) >= 2 else {}
    fss_sw = tf.fss_sweep(Ls, hc, hce, xs=[1.0, 2.0, 3.0]) if len(Ls) >= 2 else []
    free = tf.fss_free(Ls, hc, hce) if len(Ls) >= 3 else None
    if fss_main and np.isfinite(fss_main.get("h_inf", np.nan)):
        print(f"\nFSS (x=1): h_inf = {fss_main['h_inf']:.4f} +/- {fss_main['h_inf_err']:.4f}")

    spec = {"hy": hy, "fixed": (fixed_name, fixed_val), "sweep": args.sweep, "order": 1, "lane": args.lane}
    rec = make_firstorder_record(tag, spec, rows, fss_main, fss_sw, free,
                                  notes="energy branch-crossing locator (task A7)", kind=args.kind)

    if args.dry or not args.out:
        print(f"\n[dry] tag={tag}" + ("" if args.out else " (no --out given)"))
    else:
        tf.save_record(rec, args.out)
        print(f"\nsaved -> {args.out}")
    return rec


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- peer dependencies
# Nothing in transition_fit.py needs to change for this module to work -- it is
# consumed read-only via `_is_final_json`, `Curve`, `fit_logistic`, `fit_richards`,
# `fit_fd_peak`, `fss_fit`, `fss_sweep`, `fss_free`, `make_record`, `save_record`,
# `cut_tag`, `FIELDS`. Two registry-level asks for its owner, from actually using it:
#   1. `cut_tag`/`make_record` have no first-class "kind" or "obs" collision guard --
#      an energy-crossing record and an O_FM record at the same (hy, fixed, sweep)
#      point collide on the same tag unless the caller remembers a distinct `lane`.
#      A `obs`-qualified tag (or a registry check in save_record) would make that a
#      hard error instead of a silent overwrite.
#   2. `load_runs`/`load_snapshot_s2` drop ALL diverged runs outright; there is no
#      loader in transition_fit that keeps a diverged run visible (flagged) the way
#      this module's `load_branches` does. If another consumer ever needs spinodal
#      information from the "runs" lane, a `keep_diverged=False` flag on `load_runs`
#      would let it reuse the same winner-selection code instead of re-deriving it.
