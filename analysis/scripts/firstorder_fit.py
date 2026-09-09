"""First-order transition locators for field cuts of the 3D toric code (task A7).

A first-order cut (recipe notes/transition_mapping_recipes.md SB) is mapped by TWO
warm chains through the coexistence window -- an "up" branch carried from the
topological/z-polarized side and a "dn" branch carried from the x-polarized side --
plus cold anchors deep in each phase. Chain runs carry `_up` / `_dn` at the end of
their run name; anchors/cold points carry neither (recipe SB.1-2).

The PRIMARY marker depends on `kind` (locate_cut picks a default from the fixed h_z,
override with `kind=`):
  * "topo-trivial" (h_z <= 0.2 -- one side of the line is still topological): the
    O_FM_membrane_R1 inflection on the winner curve, via transition_fit.locate_all +
    combine_default (Richards central, exactly the banked O_FM-record convention).
    The energy branch crossing is then a SECONDARY consistency check: reported when
    found, flagged ("crossing_disagree") when it differs from the O_FM h_c by more
    than the combined error, and never itself reported as a value when the branches
    are merged.
  * "trivial-trivial" (h_z >= 0.4 -- BOTH sides trivial, z-polarized -> x-polarized):
    O_FM is not an order parameter there, so the energy branch crossing
    (order-parameter-free, SB.4-5: at each field the lower-energy branch is the
    ground state, and the crossing field is where the lower-energy branch switches)
    is PRIMARY. M_x = sx_mean, <A_v>, <B_p>, M_z = sz_mean jump/inflection locators are
    the SECONDARY markers.

Built on top of `transition_fit.py` (peer module, NOT modified here -- imported as
`tf`; see the "peer dependencies" note at the bottom of this file for the list of
things its owner should add/expose).

CLI:
    python analysis/scripts/firstorder_fit.py --runs DIR [DIR ...] --sweep hx \\
        --fixed hz=0.1 hy=0.0 [--ofm] [--kind topo-trivial|trivial-trivial] \\
        [--window LO HI] [--out results/transitions/<tag>.json] [--dry]

--kind defaults to whatever `_default_kind` picks from `--fixed`'s h_z (<=0.2 ->
topo-trivial, else trivial-trivial); --window restricts the primary O_FM fit (and the
secondary jump locators) to [LO, HI], e.g. `--window 0.5 1.3` to reproduce a specific
banked record's own recorded fit window bit-for-bit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
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

ERR_INFLATE = 3.0
# House convention (CLAUDE.md / notes/transition_mapping_recipes.md SS0): NQS E_err is
# underestimated by roughly a factor of 3. Applied to the crossing's propagated-E_err
# (statistical) component in energy_crossing() before combining with the geometric
# bracket_width/sqrt(12) term.

_BRANCH_RE = re.compile(r"_(up|dn)(?:_s\d+)?$")            # allows a seed-repeat suffix, e.g. "..._up_s1"
_BRANCH_AMBIGUOUS = ("_up", "_dn")                          # substrings that must NOT be silently missed


def _branch_of(stem):
    """"cold" unless the name ends in `_up`/`_dn` (optionally `_s<seed>`-suffixed).
    Never reclassifies silently: if the name contains "_up"/"_dn" ANYWHERE else
    without matching that end-suffix convention, warn once so a human checks it."""
    m = _BRANCH_RE.search(stem)
    if m:
        return m.group(1)
    if any(tok in stem for tok in _BRANCH_AMBIGUOUS):
        warnings.warn(
            f"firstorder_fit: run name {stem!r} contains '_up'/'_dn' but does not match the "
            f"branch-suffix convention (_up|_dn[_s<seed>] at the end) -- classified as 'cold', "
            f"NOT reclassified automatically.", stacklevel=2)
    return "cold"


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


def _local_stat(h0, h1, d0, d1, e0, e1):
    """Propagated h_c error from the two bracket endpoints' combined E_err (e0, e1),
    via a finite-difference gradient of the secant crossing h_c(d0, d1) -- 0.0 for a
    degenerate (d1 == d0) bracket, where the geometric width term alone carries it."""
    if d1 == d0:
        return 0.0
    eps = 1e-9
    g0 = (_bracket_h_c(h0, h1, d0 + eps, d1) - _bracket_h_c(h0, h1, d0 - eps, d1)) / (2 * eps)
    g1 = (_bracket_h_c(h0, h1, d0, d1 + eps) - _bracket_h_c(h0, h1, d0, d1 - eps)) / (2 * eps)
    return float(np.hypot(g0 * e0, g1 * e1))


def energy_crossing(up, dn):
    """(h_c, h_c_err, bracket, info) -- the crossing of the two branches' energies on
    their common field grid (non-diverged points only): h_c from LINEAR interpolation
    of Delta_E(h) at the bracket where sign(E_up - E_dn) flips.

    Chosen over a global polynomial fit of each branch's E(h): recipe SB.6's "resonance
    blind spot" makes E(h) non-smooth right at the crossing (more steps / 2x-dt kicks
    provably don't fix it there), so a bracket-local secant is the assumption-light
    choice for the reported h_c. As a cross-check (only when >=4 non-diverged points
    survive per branch on the common grid), a degree-2 polynomial is ALSO fit to each
    branch and their crossing reported in info["poly_h_c"]; if it disagrees with the
    bracket estimate by more than the combined h_c_err, info["flags"] gains
    "poly_disagree" (printed in the CLI table -- a resonance-window symptom to look at,
    not auto-resolved).

    MULTIPLE sign flips (info["n_flips"] > 1, info["flip_positions"] lists each bracket)
    happen in a near-degenerate ΔE zone where noise flips the sign of several adjacent
    points -- taking the FIRST flip unconditionally is biased (it preferentially picks
    the leftmost noise excursion, not the true crossing). Instead h_c is the CENTRE of
    the region spanning the first to the last flip, and the error is widened to cover
    that whole region (its bracket is [hs[first_flip], hs[last_flip + 1]], and the
    geometric width/sqrt(12) term grows with it).

    Error = the propagated E_err term (single bracket, or RMS over all flip-brackets in
    a multi-flip region), INFLATED by ERR_INFLATE (house convention: NQS E_err is
    underestimated ~x3), in quadrature with bracket_width/sqrt(12) (the
    "uniform-within-the-bracket" component -- we only know the crossing is somewhere in
    [h0, h1], not where).

    info["reason"] is "no overlap" (no field value survives non-diverged on both
    branches, or a branch is missing/empty -- e.g. a cold-only campaign) or
    "branches merged" (a common grid exists but Delta_E never changes sign).
    """
    if up is None or dn is None or len(up.h) == 0 or len(dn.h) == 0:
        return None, None, None, {"reason": "no overlap", "n_common": 0, "min_abs_dE": None,
                                   "n_flips": 0, "flip_positions": [], "flags": []}

    mu, md = ~up.diverged, ~dn.diverged
    hu, Eu, Eeu = up.h[mu], up.E0[mu], up.E_err[mu]
    hd, Ed, Eed = dn.h[md], dn.E0[md], dn.E_err[md]
    hu_r, hd_r = np.round(hu, 6), np.round(hd, 6)
    common = sorted(set(hu_r) & set(hd_r))
    if len(common) < 2:
        return None, None, None, {"reason": "no overlap", "n_common": len(common), "min_abs_dE": None,
                                   "n_flips": 0, "flip_positions": [], "flags": []}

    delta, derr = [], []
    for h in common:
        eu, eeu = Eu[hu_r == h][0], Eeu[hu_r == h][0]
        ed, eed = Ed[hd_r == h][0], Eed[hd_r == h][0]
        delta.append(eu - ed)
        derr.append(float(np.hypot(eeu, eed)))
    hs, delta, derr = np.array(common), np.array(delta), np.array(derr)

    info = {"n_common": len(common), "min_abs_dE": float(np.min(np.abs(delta))), "flags": []}
    info["poly_h_c"] = None
    if len(hs) >= 4:
        mask_u, mask_d = np.isin(hu_r, common), np.isin(hd_r, common)
        pu = np.polyfit(hu[mask_u], Eu[mask_u], 2)
        pd = np.polyfit(hd[mask_d], Ed[mask_d], 2)
        roots = np.roots(pu - pd)
        real = [r.real for r in roots if abs(r.imag) < 1e-6 and hs.min() <= r.real <= hs.max()]
        info["poly_h_c"] = float(sorted(real, key=lambda r: abs(r - np.median(hs)))[0]) if real else None

    flips = np.where(np.diff(np.sign(delta)) != 0)[0]
    info["n_flips"] = int(len(flips))
    info["flip_positions"] = [[float(hs[i]), float(hs[i + 1])] for i in flips]
    if len(flips) == 0:
        info["reason"] = "branches merged"
        return None, None, None, info

    if len(flips) == 1:
        i = int(flips[0])
        h0, h1, d0, d1 = float(hs[i]), float(hs[i + 1]), float(delta[i]), float(delta[i + 1])
        width = h1 - h0
        h_c = 0.5 * (h0 + h1) if d1 == d0 else _bracket_h_c(h0, h1, d0, d1)
        stat = _local_stat(h0, h1, d0, d1, float(derr[i]), float(derr[i + 1]))
        bracket = (h0, h1)
    else:
        # Near-degenerate zone: several sign changes from noise, not a real
        # multi-crossing. The centre of the whole flip-spanning region is unbiased
        # where "take the first flip" is not; widen the error to cover the region.
        i0, i1 = int(flips[0]), int(flips[-1])
        h0, h1 = float(hs[i0]), float(hs[i1 + 1])
        width = h1 - h0
        h_c = 0.5 * (h0 + h1)
        local = [_local_stat(float(hs[i]), float(hs[i + 1]), float(delta[i]), float(delta[i + 1]),
                              float(derr[i]), float(derr[i + 1])) for i in flips]
        stat = float(np.sqrt(np.mean(np.square(local)))) if local else 0.0
        bracket = (h0, h1)

    h_c_err = float(np.hypot(stat * ERR_INFLATE, width / np.sqrt(12)))
    if info["poly_h_c"] is not None and abs(info["poly_h_c"] - h_c) > h_c_err:
        info["flags"].append("poly_disagree")
    info["reason"] = "crossing"
    return float(h_c), h_c_err, bracket, info


# ----------------------------------------------------------------------------- secondary locators
def jump_locators(curve, want_ofm=False, window=(None, None)):
    """{obs: {method: Fit}} on the winner Table `curve`, reusing transition_fit's
    locators: logistic + fd_peak always; richards only if it converges (its 6
    parameters are unreliable on the short, <~10-point windows typical of a
    first-order tail). obs = sx_mean, A_v_mean, B_p_mean, sz_mean, plus
    O_FM_membrane_R1 when want_ofm=True -- kept available for cross-checking against a
    topo-trivial cut's banked O_FM locators, even though O_FM is not an order
    parameter on a trivial-trivial tail. `window` restricts every fit to [lo, hi]
    (matches a cut's own recorded fit window, e.g. (0.5, 1.3) for hz=0.1 sweep-hx)."""
    out = {}
    for key in (*JUMP_OBS_DEFAULT, *((OFM_OBS,) if want_ofm else ())):
        c = curve.curve(key).clip(window)
        if len(c.h) < 4 or not np.any(np.isfinite(c.y)):
            continue
        fits = {"logistic": tf.fit_logistic(c), "fd_peak": tf.fit_fd_peak(c)}
        r = tf.fit_richards(c)
        if r.ok():
            fits["richards"] = r
        out[key] = fits
    return out


def hellmann_feynman(curve, sweep="hx"):
    """Per adjacent pair on `curve`: compare -(Delta E / Delta h) with N * mean(obs) at
    the pair's midpoint (trapezoid average of the endpoints), N = 3 L^3 - 3 L^2 (OBC
    edge count). obs is the field's conjugate local operator (sx_mean for hx, sz_mean
    for hz, sy_mean for hy) -- the Hellmann-Feynman theorem dE/dh = -N <obs>, the same
    check CLAUDE.md's hy-lane uses for hy (recipe SC.4), generalized here to whichever
    field is swept. Returns {"rows": [...], "max_rel_dev": ..., "N": ..., "obs": key}.

    `curve` MUST be a single branch's own homogeneous Table (e.g. from load_branches,
    not winner()) -- the winner curve switches branch between adjacent h wherever the
    GS branch changes (e.g. hy_cuts_L4: up at hx=0.75, dn at hx=0.80), and dE/dh across
    that handoff is the energy jump between two DIFFERENT variational states at
    (almost) the same field, not a physical derivative; it is not a Hellmann-Feynman
    violation, just the wrong quantity. Call hellmann_feynman_by_branch() to get this
    right automatically. Diverged points (curve.diverged) are dropped before any
    difference is taken -- a diverged run's E0 is "numerical garbage" (recipe SS0), and
    a raw per-branch Table (unlike winner()'s output) can carry one: e.g.
    phaseB_rerun/right/L6 has real diverged=True entries, and differencing across one
    blew dE/dh up to ~1e18 before this filter was added."""
    key = HF_OBS.get(sweep, "sx_mean")
    keep = ~curve.diverged
    h, E = curve.h[keep], curve.E0[keep]
    y, _ye = (a[keep] for a in curve.obs[key])
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


def hellmann_feynman_by_branch(tables, L, sweep="hx"):
    """{branch: hellmann_feynman(...)} at size L, computed WITHIN each branch's own
    (homogeneous) Table separately -- up, dn and cold never mixed, so no adjacent pair
    ever crosses a branch handoff. Only branches with >= 2 points at this L are
    included. This is the fix for calling hellmann_feynman() on the (branch-mixing)
    winner() curve: a branch handoff there injects a spurious dE/dh from comparing two
    different variational states, not the physics (see hellmann_feynman's docstring)."""
    out = {}
    for branch, by_L in tables.items():
        t = by_L.get(L)
        if t is not None and len(t.h) >= 2:
            out[branch] = hellmann_feynman(t, sweep=sweep)
    return out


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


def _default_kind(fixed):
    """Default per-cut marker POLICY from the fixed field values (override with
    locate_cut(..., kind=...)): h_z <= 0.2 keeps at least one side of the line
    topological -> "topo-trivial" (O_FM primary); h_z > 0.2 (recipe: h_z >= 0.4 is
    fully in the trivial-trivial tail) -> "trivial-trivial" (energy crossing primary).
    Falls back to "trivial-trivial" if h_z isn't fixed at all (nothing to judge
    topological character by)."""
    hz = fixed.get("hz")
    return "topo-trivial" if (hz is not None and hz <= 0.2) else "trivial-trivial"


# ----------------------------------------------------------------------------- per-cut assembly
def locate_cut(dirs, sweep, fixed, want_ofm=False, kind=None, window=(None, None)):
    """Per-L rows, policy selected by `kind` (default from `fixed`'s h_z via
    _default_kind; pass kind= to override):

    * "topo-trivial": PRIMARY h_c/h_c_err = the O_FM_membrane_R1 inflection on the
      winner curve, via transition_fit.locate_all + combine_default (Richards
      central) -- exactly the banked O_FM-record convention (also carries the banked
      per-L fields stat/syst/central/spread_over/n_points/amp/S2 for direct schema
      parity). "merged" means combine_default failed to converge. The energy crossing
      is then a SECONDARY-only field (crossing_h_c/crossing_h_c_err): reported when
      found, never reported as a value when the branches themselves are merged/have
      no overlap, and flagged "crossing_disagree" in crossing_info["flags"] when it
      differs from the O_FM h_c by more than their combined error.
    * "trivial-trivial": PRIMARY h_c/h_c_err = the energy crossing; "merged" means the
      branches never cross (or one/both are absent, e.g. a cold-only campaign). The
      winner-curve jump locators (secondary) are the fallback when merged.

    Every row also carries: bracket, crossing_info, secondary (jump-locator Fits per
    local obs -- and, for "topo-trivial", O_FM's own Fits too, so the primary's inputs
    are visible in the same place), syst_jump, spinodals, hf_dev ({branch:
    max_rel_dev}, see hellmann_feynman_by_branch -- computed within each branch
    separately, never across the winner curve's branch handoffs), n_up, n_dn, kind.
    `window` restricts every fit (primary O_FM included) to [lo, hi] -- pass a cut's
    own recorded window (e.g. (0.5, 1.3) for hz=0.1 sweep-hx) to reproduce a banked
    record bit-for-bit."""
    kind = kind or _default_kind(fixed)
    tables = load_branches(dirs, sweep, fixed)
    wtabs = winner(tables)
    sp = spinodals(tables)
    rows = []
    for L in sorted(wtabs):
        wt = wtabs[L]
        up_t = tables.get("up", {}).get(L)
        dn_t = tables.get("dn", {}).get(L)
        h_c_x, h_c_x_err, bracket, info = energy_crossing(up_t, dn_t)   # always computed
        jl = jump_locators(wt, want_ofm=want_ofm, window=window)
        if kind == "topo-trivial":
            # PRIMARY source of truth: always transition_fit.locate_all directly (not
            # jump_locators' own richards-gated variant), so this cut's central marker
            # is unambiguously the banked O_FM-record convention regardless of --ofm.
            ofm_curve = wt.curve(OFM_OBS)
            if len(ofm_curve.clip(window).h) >= 4:
                jl[OFM_OBS] = tf.locate_all(ofm_curve, window=window)
            else:
                jl.pop(OFM_OBS, None)
        hf_by_branch = hellmann_feynman_by_branch(tables, L, sweep=sweep)
        hf_dev = {b: v["max_rel_dev"] for b, v in hf_by_branch.items()}

        overlap = None
        if up_t is not None and dn_t is not None and len(up_t.h) and len(dn_t.h):
            overlap = (float(max(up_t.h.min(), dn_t.h.min())), float(min(up_t.h.max(), dn_t.h.max())))
        infl = [f.h_c for obs, fits in jl.items() for f in fits.values()
                if obs != OFM_OBS and f.ok() and (overlap is None or overlap[0] <= f.h_c <= overlap[1])]
        syst_jump = 0.5 * (max(infl) - min(infl)) if len(infl) > 1 else 0.0

        row = {
            "L": L, "kind": kind,
            "bracket": list(bracket) if bracket else None, "crossing_info": info,
            "secondary": jl, "syst_jump": syst_jump,
            "spinodals": {b: sp.get(b, {}).get(L) for b in ("up", "dn", "cold")},
            "hf_dev": hf_dev,
            "n_up": int(len(up_t.h)) if up_t is not None else 0,
            "n_dn": int(len(dn_t.h)) if dn_t is not None else 0,
        }

        if kind == "topo-trivial" and OFM_OBS in jl:
            h_c, h_c_err, extra = tf.combine_default(jl[OFM_OBS])
            h_c = float(h_c) if np.isfinite(h_c) else None
            h_c_err = float(h_c_err) if np.isfinite(h_c_err) else None
            logi = jl[OFM_OBS].get("logistic")
            row.update({
                "h_c": h_c, "h_c_err": h_c_err, "stat": extra.get("stat"), "syst": extra.get("syst"),
                "central": extra.get("central"), "spread_over": extra.get("spread_over"),
                "n_points": int(len(wt.curve(OFM_OBS).h)), "S2": None,
                "amp": float(logi.popt[1]) if logi is not None and logi.ok() and logi.popt else None,
                "crossing_h_c": h_c_x, "crossing_h_c_err": h_c_x_err,
                "merged": h_c is None,
            })
            if h_c is not None and h_c_x is not None:
                combined_err = float(np.hypot(h_c_err or 0.0, h_c_x_err or 0.0))
                if abs(h_c - h_c_x) > combined_err:
                    info["flags"].append("crossing_disagree")
        else:
            row.update({"h_c": h_c_x, "h_c_err": h_c_x_err, "merged": h_c_x is None})

        rows.append(row)
    return rows


def _fit_tuple(f):
    return (f.h_c, f.h_c_err, f.chi2red)


def make_firstorder_record(tag, spec, rows, fss_main, fss_sweep_list, free=None, notes="",
                            kind=None):
    """Same JSON schema as transition_fit.make_record (results/transitions/<tag>.json
    consumers keep working); per-L "secondary" locator dicts (Fit -> (h_c, h_c_err,
    chi2red) tuples, matching the banked-record "locators" convention) pass through
    `rows` otherwise untouched, and so do the extra "topo-trivial" fields
    (stat/syst/central/spread_over/n_points/amp/S2/crossing_h_c/crossing_h_c_err) rows
    may carry. kind defaults to rows[0]["kind"] (the policy locate_cut actually used)
    if not given explicitly, and obs is "O_FM_membrane_R1" for a topo-trivial record
    or "E_crossing" for a trivial-trivial one -- so a saved record's `obs` always names
    its own PRIMARY marker. Built THROUGH transition_fit.make_record (not
    reimplemented) so its quality gate / syst_exponent computation stays the single
    source of truth."""
    kind = kind or (rows[0]["kind"] if rows else "trivial-trivial")
    ser_rows = []
    for r in rows:
        rr = dict(r)
        rr["h_c"] = np.nan if rr.get("h_c") is None else rr["h_c"]
        rr["h_c_err"] = np.nan if rr.get("h_c_err") is None else rr["h_c_err"]
        rr["secondary"] = {obs: {m: _fit_tuple(f) for m, f in fits.items()}
                            for obs, fits in r["secondary"].items()}
        ser_rows.append(rr)
    obs_name = OFM_OBS if kind == "topo-trivial" else "E_crossing"
    spec = {**spec, "kind": kind, "obs": obs_name}
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


def _fmt_hf(hf_dev):
    return "/".join(f"{b}:{v:.3f}" for b, v in hf_dev.items()) or "-"


def _print_table(rows, want_ofm=False):
    obs_order = list(JUMP_OBS_DEFAULT) + ([OFM_OBS] if want_ofm else [])
    kind = rows[0]["kind"] if rows else "trivial-trivial"
    primary_label = "h_c(O_FM)" if kind == "topo-trivial" else "h_c(x-ing)"
    print(f"kind={kind}  (primary marker: {primary_label})")
    print(f"{'L':>3} {'h_c':>8} {'h_c_err':>8} {'central':>9} {'x-ing_h_c':>10} {'bracket':>16} "
          f"{'flips':>5} {'flags':>16} {'n_up':>4} {'n_dn':>4} {'hf_dev(by branch)':>24} "
          f"{'merged':>7}  jump-logistic h_c (" + ",".join(obs_order) + ")")
    for r in rows:
        br = f"[{r['bracket'][0]:.3f},{r['bracket'][1]:.3f}]" if r["bracket"] else "-"
        vals = []
        for k in obs_order:
            fit = r["secondary"].get(k, {}).get("logistic")
            vals.append(f"{fit.h_c:.3f}" if fit is not None and fit.ok() else "-")
        flags = ",".join(r["crossing_info"].get("flags", [])) or "-"
        central = r.get("central", "E_crossing") or "-"
        x_ing = _fmt(r.get("crossing_h_c", r["h_c"]))
        print(f"{r['L']:>3} {_fmt(r['h_c']):>8} {_fmt(r['h_c_err']):>8} {central:>9} {x_ing:>10} {br:>16} "
              f"{r['crossing_info'].get('n_flips', 0):>5} {flags:>16} "
              f"{r['n_up']:>4} {r['n_dn']:>4} {_fmt_hf(r['hf_dev']):>24} {str(r['merged']):>7}  "
              + " ".join(vals))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="dirs of per-run final-state JSONs")
    ap.add_argument("--sweep", required=True, choices=tf.FIELDS)
    ap.add_argument("--fixed", nargs="+", required=True, help="key=val pairs, e.g. hz=0.1 hy=0.0")
    ap.add_argument("--ofm", action="store_true",
                     help="also run the O_FM_membrane_R1 jump locator (secondary; always on for "
                          "kind=topo-trivial regardless of this flag, since it's the primary there)")
    ap.add_argument("--kind", default=None, choices=["topo-trivial", "trivial-trivial"],
                     help="marker policy; default picked from --fixed's hz (<=0.2 -> topo-trivial, "
                          "else trivial-trivial) -- see _default_kind")
    ap.add_argument("--window", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                     help="restrict every fit (primary O_FM included) to [LO, HI], e.g. 0.5 1.3 -- "
                          "pass a cut's own recorded window to reproduce a banked record bit-for-bit")
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
    kind = args.kind or _default_kind(fixed)
    window = tuple(args.window) if args.window else (None, None)

    rows = locate_cut(args.runs, args.sweep, fixed, want_ofm=args.ofm, kind=kind, window=window)
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
                                  notes="task A7 first-order locator", kind=kind)

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
