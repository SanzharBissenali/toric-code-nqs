"""Campaign progress + status generator for the 3D toric-code phase-diagram campaign (A4).

NetKet-free (numpy/scipy/pandas only). Reads the run tree the launcher/watcher write

    results/phase3d/hy{hy}/{electric_hx{hx}|magnetic_hz{hz}}/L{L}/*.json   (tc3d.train finals)
    results/phase3d/manifests/manifest_*.tsv                              (submitted jobs)
    results/phase3d/watch_state.json                                     (live Slurm/guard state)

and answers "where does the campaign stand right now": coverage of the planned grid,
health of what has landed (divergence, the h=0 bound, hot Vscore), and a running read of
the transition location per (cut, L) from whatever has landed so far. Everything degrades
to an empty-but-shaped result on a missing/partial tree -- never an exception, since this
is read live while the campaign is still running.

`electric` cuts fix h_x and sweep h_z; `magnetic` cuts fix h_z and sweep h_x (CLAUDE.md's
phase_hx{}/phase_hz{} convention). Locator fits reuse the peer's `transition_fit.py`
(untracked sibling module -- import only, never edit/commit it here). First-order (magnetic)
cuts go through the peer's `firstorder_fit.locate_cut`: the energy branch crossing is
primary for the "crossing" field (h_c/merged=null/true when the branches never cross --
never a closest-approach stand-in); for topo-trivial cuts (fixed h_z <= 0.2) the
topological O_FM_membrane_R1 locator on the winner curve (secondary in `locate_cut`, via
`want_ofm=True`) is primary for the "hc" field instead, mirroring how electric cuts use
O_FM_paratoric.

CLI: `python -m analysis.scripts.phase3d_status --root results/phase3d --out results/phase3d/STATUS.md`
     `python -m analysis.scripts.phase3d_status --selftest [--tmp DIR]`
     `python -m analysis.scripts.phase3d_status --export-viewer HY --root ... --curves-root ... --out viewer_hyHY.json`
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import transition_fit as tf                     # noqa: E402  (untracked sibling; import-only)
import firstorder_fit as fof                    # noqa: E402  (peer module; import-only)

TOPO_TRIVIAL_HZ_MAX = 0.2   # first-order cuts fixed at hz <= this are topo->trivial (O_FM primary);
                            # above it both sides are trivial (O_FM not an order parameter there)

CUT_SWEEP = {"electric": "hz", "magnetic": "hx"}  # cut -> field it varies; the other is `fixed_field`

FINAL_COLUMNS = [
    "hy", "cut", "fixed_field", "fixed_val", "L", "h", "branch",
    "E0", "E_err", "E_var", "Vscore", "E_im",
    "A_v", "A_v_err", "B_p", "B_p_err", "sx", "sx_err", "sy", "sy_err", "sz", "sz_err",
    "O_FM_paratoric", "O_FM_paratoric_err", "O_FM_membrane_R1", "O_FM_membrane_R1_err",
    "ref_E", "dE_ref", "diverged", "n_rollbacks", "runtime_s", "name", "path", "mtime",
]
MANIFEST_COLUMNS = ["jobid", "hy", "cut", "L", "role", "h", "name", "out_dir", "submitted_at"]

_HY_RE = re.compile(r"^hy(-?\d+\.?\d*)$")
_CUT_RE = re.compile(r"^(electric|magnetic)_h([xz])(-?\d+\.?\d*)$")
_L_RE = re.compile(r"^L(\d+)$")


def bound(L):
    """OBC h=0 exact anchor E0 = -(#A_v + #B_p) = -(L^3 + 3(L-1)^2 L). Any converged
    finite-field run must sit strictly below it (nqs-finite-field-e0-benchmark)."""
    L = np.asarray(L)
    return -(L ** 3 + 3 * (L - 1) ** 2 * L)


def n_sites(L):
    """Physical qubits (edges) of the LxLxL OBC lattice: geometry.py's
    self.N = 3*Lx*Ly*Lz - (Lx*Ly + Lx*Lz + Ly*Lz) at Lx=Ly=Lz=L -> 3*L^2*(L-1)."""
    L = np.asarray(L)
    return 3 * L ** 3 - 3 * L ** 2


def _is_final(f: Path) -> bool:
    return (f.name != "watch_state.json" and "finaleval" not in f.name
            and not f.name.endswith((".snapshots.json", ".curve.json")))


def _branch(name: str) -> str:
    """Chain runs carry `_up`/`_dn` at the end of the name; everything else is cold."""
    if name.endswith("_up"):
        return "up"
    if name.endswith("_dn"):
        return "dn"
    return "cold"


def _iter_run_dirs(root: Path):
    """Yield (hy, cut, fixed_field, fixed_val, L, dirpath) for every L-leaf directory
    matching the contract layout under `root`; skips anything else (manifests/,
    watch_state.json, an absent root) without raising."""
    if not root.exists():
        return
    for hy_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m_hy = _HY_RE.match(hy_dir.name)
        if not m_hy:
            continue
        hy = float(m_hy.group(1))
        for cut_dir in sorted(p for p in hy_dir.iterdir() if p.is_dir()):
            m_cut = _CUT_RE.match(cut_dir.name)
            if not m_cut:
                continue
            cut, fixed_field, fixed_val = m_cut.group(1), "h" + m_cut.group(2), float(m_cut.group(3))
            for l_dir in sorted(p for p in cut_dir.iterdir() if p.is_dir()):
                m_l = _L_RE.match(l_dir.name)
                if not m_l:
                    continue
                yield hy, cut, fixed_field, fixed_val, int(m_l.group(1)), l_dir


# ------------------------------------------------------------------------------- loaders
def load_finals(root) -> pd.DataFrame:
    """One row per landed run under results/phase3d/hy{hy}/{electric|magnetic}_.../L{L}/.
    Empty tree (root missing, nothing landed yet) -> empty DataFrame, full column set,
    never an exception."""
    rows = []
    for hy, cut, fixed_field, fixed_val, L, l_dir in _iter_run_dirs(Path(root)):
        sweep = CUT_SWEEP[cut]
        for f in sorted(l_dir.glob("*.json")):
            if not _is_final(f):
                continue
            try:
                j = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            c, o = j.get("config") or {}, j.get("observables") or {}
            name = j.get("name", f.stem)
            rows.append({
                "hy": hy, "cut": cut, "fixed_field": fixed_field, "fixed_val": fixed_val, "L": L,
                "h": c.get(sweep, np.nan), "branch": _branch(name),
                "E0": o.get("E0", np.nan), "E_err": o.get("E_err", np.nan), "E_var": o.get("E_var", np.nan),
                "Vscore": o.get("Vscore", np.nan), "E_im": o.get("E_im", np.nan),
                "A_v": o.get("A_v_mean", np.nan), "A_v_err": o.get("A_v_err", np.nan),
                "B_p": o.get("B_p_mean", np.nan), "B_p_err": o.get("B_p_err", np.nan),
                "sx": o.get("sx_mean", np.nan), "sx_err": o.get("sx_err", np.nan),
                "sy": o.get("sy_mean", np.nan), "sy_err": o.get("sy_err", np.nan),
                "sz": o.get("sz_mean", np.nan), "sz_err": o.get("sz_err", np.nan),
                "O_FM_paratoric": o.get("O_FM_paratoric", np.nan),
                "O_FM_paratoric_err": o.get("O_FM_paratoric_err", np.nan),
                "O_FM_membrane_R1": o.get("O_FM_membrane_R1", np.nan),
                "O_FM_membrane_R1_err": o.get("O_FM_membrane_R1_err", np.nan),
                "ref_E": o.get("ref_E", c.get("ref_E", np.nan)), "dE_ref": o.get("dE_ref", np.nan),
                "diverged": bool(j.get("diverged", False)), "n_rollbacks": j.get("n_rollbacks") or 0,
                "runtime_s": j.get("runtime_s", np.nan), "name": name, "path": str(f),
                "mtime": f.stat().st_mtime,
            })
    return pd.DataFrame(rows, columns=FINAL_COLUMNS)


def add_health(df: pd.DataFrame) -> pd.DataFrame:
    """+bound (h=0 OBC anchor for that L), above_bound (E0 sits at/above it -- a red
    flag for a finite-field converged run), vscore_hot (Vscore beyond the h_y
    sign-structure cost 0.5*hy^2 plus a 0.3 slack)."""
    df = df.copy()
    if df.empty:
        df["bound"] = pd.Series(dtype=float)
        df["above_bound"] = pd.Series(dtype=bool)
        df["vscore_hot"] = pd.Series(dtype=bool)
        return df
    df["bound"] = bound(df["L"].to_numpy())
    df["above_bound"] = df["E0"] > df["bound"] + 1e-6
    df["vscore_hot"] = df["Vscore"] > 0.5 * df["hy"] ** 2 + 0.3
    return df


def load_manifests(root) -> pd.DataFrame:
    man_dir = Path(root) / "manifests"
    frames = []
    if man_dir.exists():
        for f in sorted(man_dir.glob("manifest_*.tsv")):
            try:
                frames.append(pd.read_csv(f, sep="\t"))
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
    if not frames:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["hy"] = df["hy"].astype(float)
    df["L"] = df["L"].astype(int)
    df["h"] = df["h"].astype(float)
    return df[MANIFEST_COLUMNS]


def load_watch_state(root) -> dict:
    f = Path(root) / "watch_state.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def job_table(root, df=None, manifests=None, watch=None) -> pd.DataFrame:
    """Per-run status merging manifest submission + watch_state (live) + finals (landed),
    joined on `name`. A name only in the manifest (not yet landed) still gets a row, with
    every final-only field NaN/None and state read from watch_state (else 'UNKNOWN')."""
    df = add_health(load_finals(root) if df is None else df)
    man = load_manifests(root) if manifests is None else manifests
    ws = load_watch_state(root) if watch is None else watch
    fin_by_name = {r["name"]: r for r in df.to_dict("records")}
    man_by_name = {r["name"]: r for r in man.to_dict("records")}
    names = list(dict.fromkeys(list(man_by_name) + list(fin_by_name)))
    rows = []
    for name in names:
        m, fi, w = man_by_name.get(name, {}), fin_by_name.get(name, {}), ws.get(name, {})
        rows.append({
            "name": name, "jobid": m.get("jobid"), "hy": m.get("hy", fi.get("hy")),
            "cut": m.get("cut", fi.get("cut")), "L": m.get("L", fi.get("L")), "role": m.get("role"),
            "h": m.get("h", fi.get("h")), "state": w.get("state", "COMPLETED" if fi else "UNKNOWN"),
            "landed": bool(fi), "diverged": w.get("diverged", fi.get("diverged")),
            "warm_loaded": w.get("warm_loaded"), "last_step": w.get("last_step"),
            "E0": fi.get("E0"), "bound": fi.get("bound"), "above_bound": fi.get("above_bound"),
            "Vscore": fi.get("Vscore"), "log": w.get("log"),
        })
    cols = ["name", "jobid", "hy", "cut", "L", "role", "h", "state", "landed", "diverged",
            "warm_loaded", "last_step", "E0", "bound", "above_bound", "Vscore", "log"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(["hy", "cut", "L", "h"], na_position="last").reset_index(drop=True)
    return out


# ------------------------------------------------------------------------------- planned grid + coverage
def planned_from_grid(script=None, timeout=30):
    """HOOK: analysis/scripts/phase3d_grid.py --json is another agent's planned-grid
    generator; wire it in once it lands. Returns DataFrame[hy, cut, L, h] or None if the
    script doesn't exist yet / doesn't run cleanly -- callers fall back to the manifest."""
    script = Path(script) if script else Path(__file__).with_name("phase3d_grid.py")
    if not script.exists():
        return None
    try:
        out = subprocess.run([sys.executable, str(script), "--json"], capture_output=True,
                              text=True, timeout=timeout, check=True)
        rows = json.loads(out.stdout)
        df = pd.DataFrame(rows)
    except Exception:                                                # noqa: BLE001
        return None
    return df[["hy", "cut", "L", "h"]] if {"hy", "cut", "L", "h"} <= set(df.columns) else None


def planned_from_manifests(root, manifests=None) -> pd.DataFrame:
    """Fallback planned source: every (hy, cut, L, h) any manifest ever submitted."""
    man = load_manifests(root) if manifests is None else manifests
    if man.empty:
        return pd.DataFrame(columns=["hy", "cut", "L", "h"])
    return man[["hy", "cut", "L", "h"]].drop_duplicates().reset_index(drop=True)


def load_planned(root, grid_script=None) -> pd.DataFrame:
    df = planned_from_grid(grid_script)
    if df is None or df.empty:
        df = planned_from_manifests(root)
    return df


def coverage(planned: pd.DataFrame, landed: pd.DataFrame) -> pd.DataFrame:
    """Per (hy, cut, L): planned vs. landed point counts (+ diverged count among landed).
    `planned` is widened to at least `landed` so a point landed outside the known grid
    (refine/manifest-less rerun) never shows a fraction above 1. Either input empty ->
    an empty-but-shaped table."""
    cols = ["hy", "cut", "L", "planned", "landed", "frac", "diverged"]
    if planned.empty and landed.empty:
        return pd.DataFrame(columns=cols)
    p = planned.drop_duplicates(["hy", "cut", "L", "h"]).groupby(["hy", "cut", "L"]).size() if len(planned) else pd.Series(dtype=int)
    la = landed.drop_duplicates(["hy", "cut", "L", "h"]).groupby(["hy", "cut", "L"]).size() if len(landed) else pd.Series(dtype=int)
    ld = (landed[landed["diverged"]].drop_duplicates(["hy", "cut", "L", "h"]).groupby(["hy", "cut", "L"]).size()
          if len(landed) else pd.Series(dtype=int))
    rows = []
    for key in sorted(set(p.index) | set(la.index)):
        hy, cut, L = key
        pn, ln, dn = int(p.get(key, 0)), int(la.get(key, 0)), int(ld.get(key, 0))
        planned_n = max(pn, ln)
        rows.append({"hy": hy, "cut": cut, "L": int(L), "planned": planned_n, "landed": ln,
                     "frac": (ln / planned_n) if planned_n else np.nan, "diverged": dn})
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------------------- partial locators
def partial_locators(root, df=None, min_points=5, obs="O_FM_paratoric"):
    """Electric cuts: per (hy, L) with >= min_points landed points, {h_c, err} from
    `transition_fit.load_runs` + `locate_all` + `combine_default` (richards-inflection
    central value -- the production marker policy). Magnetic (chain) cuts: per (hy, L,
    branch) the last landed field, plus whether the up/dn energy branches have crossed
    (adjacent-point sign change of E_up - E_dn on their common h grid)."""
    df = load_finals(root) if df is None else df
    electric_rows, chain_rows = [], []
    if not df.empty:
        for (hy, ffield, fval, L), g in df[df.cut == "electric"].groupby(
                ["hy", "fixed_field", "fixed_val", "L"], dropna=False):
            dirs = sorted({str(Path(p).parent) for p in g["path"]})
            curve = tf.load_runs(dirs, sweep="hz", fixed={ffield: fval, "hy": hy}, obs=obs).get(int(L))
            if curve is None or len(curve.h) < min_points:
                continue
            fits = tf.locate_all(curve)
            h_c, err, meta = tf.combine_default(fits)
            central = meta.get("central")
            electric_rows.append({
                "hy": hy, "fixed_field": ffield, "fixed_val": fval, "L": int(L),
                "h_c": h_c, "h_c_err": err, "method": central, "n": len(curve.h),
                "chi2red": fits[central].chi2red if central in fits else np.nan,
            })

        for (hy, ffield, fval, L), g in df[df.cut == "magnetic"].groupby(
                ["hy", "fixed_field", "fixed_val", "L"], dropna=False):
            branches = {"up": g[g.branch == "up"].sort_values("h"), "dn": g[g.branch == "dn"].sort_values("h")}
            crossed, bracket = False, None
            common = sorted(set(branches["up"]["h"]) & set(branches["dn"]["h"]))
            if len(common) >= 2:
                eu = branches["up"].set_index("h")["E0"].reindex(common).to_numpy()
                ed = branches["dn"].set_index("h")["E0"].reindex(common).to_numpy()
                diff = eu - ed
                flips = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
                if len(flips):
                    crossed, bracket = True, (float(common[flips[0]]), float(common[flips[0] + 1]))
            for branch, gb in branches.items():
                if not len(gb):
                    continue
                last_h = float(gb["h"].max() if branch == "up" else gb["h"].min())
                chain_rows.append({"hy": hy, "fixed_field": ffield, "fixed_val": fval, "L": int(L),
                                   "branch": branch, "last_h": last_h, "n": len(gb),
                                   "crossed": crossed, "bracket": bracket})
    e_cols = ["hy", "fixed_field", "fixed_val", "L", "h_c", "h_c_err", "method", "n", "chi2red"]
    c_cols = ["hy", "fixed_field", "fixed_val", "L", "branch", "last_h", "n", "crossed", "bracket"]
    return {"electric": pd.DataFrame(electric_rows, columns=e_cols),
            "chain": pd.DataFrame(chain_rows, columns=c_cols)}


# ------------------------------------------------------------------------------- viewer export
def _jn(x):
    """JSON-safe scalar: NaN/inf -> None, numpy scalars -> plain python."""
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return x


def _s2_for_run(path: Path):
    """S2 (+err) for one run: <name>.finaleval_electric.json if present (schema TBD --
    tried as either a top-level or `observables`-nested S2/S2_err), else the last entry
    of <name>.snapshots.json (mirrors transition_fit.load_snapshot_s2), else (None, None)."""
    fe = path.with_name(path.stem + ".finaleval_electric.json")
    if fe.exists():
        try:
            j = json.loads(fe.read_text())
            o = j.get("observables", j)
            if o.get("S2") is not None:
                return o.get("S2"), o.get("S2_err")
        except (json.JSONDecodeError, OSError):
            pass
    snap = path.with_name(path.stem + ".snapshots.json")
    if snap.exists():
        try:
            j = json.loads(snap.read_text())
            ser = [s for s in j.get("series", []) if "error" not in s and s.get("S2") is not None]
            if ser:
                last = ser[-1]
                return last.get("S2"), last.get("S2_err")
        except (json.JSONDecodeError, OSError):
            pass
    return None, None


def _export_curve(row, root, curves_root, max_points=600):
    """<name>.curve.json from the data/tc_nqs mirror only (no inline-'curve' fallback --
    that lane is the raw per-step W&B/curve mirror, not the committed results/ tree);
    None when absent. Subsamples evenly to `max_points` steps; adds a per-step Vscore
    series (N * energy_spread^2 / energy^2, matching validation.py's Vscore definition)
    when the curve carries `energy_spread`."""
    try:
        rel = Path(row["path"]).relative_to(Path(root))
    except ValueError:
        return None
    f = Path(curves_root) / rel.parent / f"{row['name']}.curve.json"
    if not f.exists():
        return None
    try:
        cv = json.loads(f.read_text()).get("curve")
    except (json.JSONDecodeError, OSError):
        return None
    if not cv or not cv.get("step") or not cv.get("energy"):
        return None
    n = len(cv["step"])
    idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).round().astype(int))
    e_arr = np.asarray(cv["energy"], float)
    out = {"step": [int(cv["step"][i]) for i in idx], "E": [_jn(e_arr[i]) for i in idx]}
    spread = cv.get("energy_spread")
    if spread:
        s_arr = np.asarray(spread, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            v_arr = n_sites(int(row["L"])) * s_arr ** 2 / e_arr ** 2
        out["Vscore"] = [_jn(v_arr[i]) for i in idx]
    else:
        out["Vscore"] = None
    return out


def export_viewer(root, curves_root, hy, min_points=5, tol=1e-9) -> dict:
    """One JSON per (campaign, hy plane) for the drill-down viewer Artifact: every cut at
    that hy, its landed points (full observable set + a subsampled learning curve), and
    per-L locators. Electric cuts get "hc" from the O_FM_paratoric partial-locator table.
    First-order (magnetic) cuts always get "crossing" (the energy branch crossing via
    `firstorder_fit.locate_cut`; h_c/err are null and merged=true when the branches never
    cross -- never a closest-approach stand-in); topo-trivial cuts (fixed h_z <=
    TOPO_TRIVIAL_HZ_MAX) additionally get "hc" from the topological O_FM_membrane_R1
    locator on the winner curve (>= min_points non-diverged points), the PRIMARY locator
    there per the banked Phase-B convention. `curves_root` is the data/tc_nqs mirror
    (sibling of the committed results/ tree). Empty/partial campaign -> a well-shaped
    JSON with empty `cuts`."""
    root = Path(root)
    df_all = add_health(load_finals(root))
    hy_all = sorted(set(df_all["hy"])) if len(df_all) else []

    times = list(df_all["mtime"]) if len(df_all) else []
    man_dir = root / "manifests"
    if man_dir.exists():
        times += [f.stat().st_mtime for f in man_dir.glob("manifest_*.tsv")]
    ws_file = root / "watch_state.json"
    if ws_file.exists():
        times.append(ws_file.stat().st_mtime)
    data_as_of = (datetime.fromtimestamp(max(times), tz=timezone.utc).isoformat(timespec="seconds")
                  if times else None)

    df = df_all[np.isclose(df_all["hy"], hy, atol=tol)] if len(df_all) else df_all
    e_loc = partial_locators(root, df_all, min_points=min_points)["electric"]

    cuts, Ls_seen = [], set()
    if len(df):
        for (cut, ffield, fval), g in df.groupby(["cut", "fixed_field", "fixed_val"]):
            sweep = CUT_SWEEP[cut]
            points = []
            for _, row in g.sort_values(["L", "h"]).iterrows():
                Ls_seen.add(int(row["L"]))
                s2, s2_err = _s2_for_run(Path(row["path"]))
                o_fm, o_fm_err = ((row["O_FM_paratoric"], row["O_FM_paratoric_err"]) if cut == "electric"
                                  else (row["O_FM_membrane_R1"], row["O_FM_membrane_R1_err"]))
                points.append({
                    "L": int(row["L"]), "h": _jn(row["h"]), "branch": row["branch"], "name": row["name"],
                    "E0": _jn(row["E0"]), "E_err": _jn(row["E_err"]), "Vscore": _jn(row["Vscore"]),
                    "E_im": _jn(row["E_im"]), "diverged": bool(row["diverged"]),
                    "above_bound": bool(row["above_bound"]), "n_rollbacks": int(row["n_rollbacks"] or 0),
                    "runtime_s": _jn(row["runtime_s"]), "O_FM": _jn(o_fm), "O_FM_err": _jn(o_fm_err),
                    "S2": _jn(s2), "S2_err": _jn(s2_err), "sx": _jn(row["sx"]), "sx_err": _jn(row["sx_err"]),
                    "sz": _jn(row["sz"]), "A_v": _jn(row["A_v"]), "B_p": _jn(row["B_p"]),
                    "ref_E": _jn(row["ref_E"]), "curve": _export_curve(row, root, curves_root),
                })
            cut_dict = {"id": f"{cut}_{ffield}{fval:g}", "kind": "electric" if cut == "electric" else "first-order",
                        "fixed": {ffield: fval}, "sweep": sweep, "order": 2 if cut == "electric" else 1,
                        "points": points}
            if cut == "electric":
                sub = (e_loc[np.isclose(e_loc.hy, hy) & (e_loc.fixed_field == ffield) & np.isclose(e_loc.fixed_val, fval)]
                       if len(e_loc) else e_loc)
                cut_dict["hc"] = {str(int(r["L"])): {"h_c": _jn(r["h_c"]), "err": _jn(r["h_c_err"])}
                                  for _, r in sub.iterrows()}
            else:
                topo = fval <= TOPO_TRIVIAL_HZ_MAX
                dirs = sorted({str(Path(p).parent) for p in g["path"]})
                fo_rows = fof.locate_cut(dirs, sweep=sweep, fixed={ffield: fval, "hy": hy}, want_ofm=topo)
                cut_dict["crossing"] = {
                    str(r["L"]): {"h_c": _jn(r["h_c"]), "err": _jn(r["h_c_err"]), "merged": bool(r["merged"])}
                    for r in fo_rows
                }
                if topo:
                    # primary locator here is the topological O_FM_membrane_R1 fit on the
                    # winner curve (locate_cut's `secondary`, want_ofm=True), NOT the energy
                    # crossing -- mirrors the electric cuts' O_FM_paratoric "hc" above.
                    hc = {}
                    for r in fo_rows:
                        fits = r["secondary"].get(fof.OFM_OBS)
                        if not fits:
                            continue
                        n = next((f.n for f in fits.values() if f.n), 0)
                        if n < min_points:
                            continue
                        h_c, err, _meta = tf.combine_default(fits)
                        hc[str(r["L"])] = {"h_c": _jn(h_c), "err": _jn(err)}
                    cut_dict["hc"] = hc
            cuts.append(cut_dict)

    return {
        "hy": float(hy), "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "planes": [float(x) for x in hy_all], "data_as_of": data_as_of,
        "anchors": {"hz_c_hx0": tf.EXACT["hz_c(hx=0,hy=0)"], "hx_c_hz0": tf.EXACT["hx_c(hz=0,hy=0)"]},
        "bound": {str(L): int(bound(L)) for L in sorted(Ls_seen)},
        "N": {str(L): int(n_sites(L)) for L in sorted(Ls_seen)},
        "cuts": cuts,
    }


# ------------------------------------------------------------------------------- STATUS.md
def write_status_md(root, out, grid_script=None, min_points=5) -> str:
    root = Path(root)
    df = add_health(load_finals(root))
    planned = load_planned(root, grid_script)
    cov = coverage(planned, df)
    locators = partial_locators(root, df, min_points=min_points)
    e_loc, c_loc = locators["electric"], locators["chain"]

    lines = ["# Phase-3D campaign status", "",
             f"_generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `{root}`_", ""]
    hys = sorted((set(cov["hy"]) if len(cov) else set()) | (set(df["hy"]) if len(df) else set()))
    if not hys:
        lines += ["No runs have landed and nothing is planned yet.", ""]
    for hy in hys:
        lines += [f"## hy = {hy:g}", "",
                  "| cut | L | landed/planned | diverged | above-bound | current h_c(L) / crossing | last update |",
                  "|---|---|---|---|---|---|---|"]
        sub_cov = cov[cov.hy == hy].sort_values(["cut", "L"]) if len(cov) else cov
        for _, row in sub_cov.iterrows():
            cut, L = row["cut"], int(row["L"])
            g = df[(df.hy == hy) & (df.cut == cut) & (df.L == L)] if len(df) else df
            n_ab = int(g["above_bound"].sum()) if len(g) else 0
            last = (datetime.fromtimestamp(g["mtime"].max()).isoformat(sep=" ", timespec="minutes")
                    if len(g) else "--")
            if cut == "electric":
                m = e_loc[(e_loc.hy == hy) & (e_loc.L == L)] if len(e_loc) else e_loc
                cell = (f"{m.iloc[0]['h_c']:.4f} ± {m.iloc[0]['h_c_err']:.4f} ({m.iloc[0]['method']})"
                        if len(m) else "--")
            else:
                m = c_loc[(c_loc.hy == hy) & (c_loc.L == L)] if len(c_loc) else c_loc
                if len(m) and bool(m.iloc[0]["crossed"]):
                    br = m.iloc[0]["bracket"]
                    cell = f"crossed in [{br[0]:g}, {br[1]:g}]"
                elif len(m):
                    cell = "no crossing yet"
                else:
                    cell = "--"
            lines.append(f"| {cut} | {L} | {int(row['landed'])}/{int(row['planned'])} | "
                         f"{int(row['diverged'])} | {n_ab} | {cell} | {last} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


# ------------------------------------------------------------------------------- self-test
def _write_final(path, name, hy, hx, hz, L, E0, diverged=False, Vscore=0.01, extra_obs=None, ref_E=None):
    path.mkdir(parents=True, exist_ok=True)
    obs = {"E0": E0, "E_err": 0.01, "E_var": 1.0, "Vscore": Vscore, "A_v_mean": 0.95, "A_v_err": 1e-3,
           "B_p_mean": 0.96, "B_p_err": 1e-3, "sx_mean": 0.1, "sx_err": 1e-3, "sy_mean": 0.01,
           "sy_err": 1e-3, "sz_mean": 0.1, "sz_err": 1e-3,
           "O_FM_paratoric": None, "O_FM_paratoric_err": None,
           "O_FM_membrane_R1": None, "O_FM_membrane_R1_err": None}
    if hy:
        obs["E_im"] = 0.0
    if ref_E is not None:
        obs["ref_E"] = ref_E
        obs["dE_ref"] = E0 - ref_E
    obs.update(extra_obs or {})
    j = {"name": name, "config": {"L": L, "bc": "OBC", "hx": hx, "hy": hy, "hz": hz, "ref_E": ref_E},
         "observables": obs, "diverged": diverged, "n_rollbacks": 0, "runtime_s": 100.0}
    (path / f"{name}.json").write_text(json.dumps(j))


def _selftest(tmp=None):
    tmp = Path(tmp) if tmp else Path(tempfile.mkdtemp(prefix="phase3d_status_selftest_"))
    root = tmp / "results" / "phase3d"

    # --- empty tree: everything must degrade gracefully, no exception -----------------
    empty_df = load_finals(root)
    assert list(empty_df.columns) == FINAL_COLUMNS and empty_df.empty
    assert add_health(empty_df).empty
    assert load_manifests(root).empty and load_watch_state(root) == {}
    assert coverage(load_planned(root), empty_df).empty
    loc0 = partial_locators(root, empty_df)
    assert loc0["electric"].empty and loc0["chain"].empty
    txt0 = write_status_md(root, tmp / "STATUS_empty.md")
    assert "No runs have landed" in txt0

    # --- bound() sanity: L=4 OBC h=0 anchor is -172 (memory: exact-h0-energies) --------
    assert bound(4) == -172
    assert np.array_equal(bound(np.array([4, 6])), np.array([-172, -(6**3 + 3 * 5**2 * 6)]))

    # --- populated tree: one electric cut (hy=0.2, hx=0.2, sweep hz), one magnetic
    #     chain cut (hy=0.2, hz=0.1, sweep hx, up/dn branches with a deliberate crossing)
    hzs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    e_dir = root / "hy0.2" / "electric_hx0.2" / "L4"
    for i, hz in enumerate(hzs):
        # a logistic-ish O_FM rise across hz, so richards/logistic both have signal
        O = 0.02 + 0.9 / (1.0 + np.exp(-(hz - 0.22) / 0.05))
        diverged = (i == 3)                                    # one flagged run in the mix
        name = f"selftest_L4_hx0.2_hz{hz:g}_hy0.2"
        _write_final(e_dir, name, 0.2, 0.2, hz, 4, E0=-180.0 - i, diverged=diverged, Vscore=0.05,
                     extra_obs={"O_FM_paratoric": O, "O_FM_paratoric_err": 0.01})
    # one deliberately unhealthy point: converged (not diverged) but E0 above the bound
    _write_final(e_dir, "selftest_L4_hx0.2_hz0.50_hy0.2", 0.2, 0.2, 0.50, 4, E0=-1.0,
                 Vscore=5.0, extra_obs={"O_FM_paratoric": 0.9, "O_FM_paratoric_err": 0.01})

    m_dir = root / "hy0.2" / "magnetic_hz0.1" / "L4"
    hxs_common = [0.80, 0.85, 0.90, 0.95, 1.00]
    # up branch lower at small hx, dn branch lower at large hx -> one clean interior
    # sign flip of E_up - E_dn between hx=0.90 and hx=0.95 (both branches stay < bound);
    # O_FM_membrane_R1 falls topological -> trivial across the same window (winner-curve
    # jump locator, this cut is fixed hz=0.1 <= TOPO_TRIVIAL_HZ_MAX -> "hc" expected).
    eu = [-174.00, -173.60, -173.20, -172.80, -172.40]
    ed = [-172.35, -172.75, -173.15, -173.55, -173.95]
    ofm = [0.85, 0.70, 0.45, 0.15, 0.05]
    for j, hx in enumerate(hxs_common):
        extra = {"O_FM_membrane_R1": ofm[j], "O_FM_membrane_R1_err": 0.02}
        _write_final(m_dir, f"selftest_L4_hx{hx:g}_hz0.1_hy0.2_up", 0.2, hx, 0.1, 4, E0=eu[j], extra_obs=extra)
        _write_final(m_dir, f"selftest_L4_hx{hx:g}_hz0.1_hy0.2_dn", 0.2, hx, 0.1, 4, E0=ed[j], extra_obs=extra)
    _write_final(m_dir, "selftest_L4_hx0.6_hz0.1_hy0.2", 0.2, 0.6, 0.1, 4, E0=-176.0)  # cold anchor

    man_dir = root / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    header = "\t".join(MANIFEST_COLUMNS)
    man_rows = [header]
    jid = 1
    for hz in hzs + [0.50, 0.55]:                               # 0.55 is planned-not-landed
        man_rows.append("\t".join(str(v) for v in
                        (jid, 0.2, "electric", 4, "cold", hz, f"m{jid}", "", "2026-09-09T00:00:00")))
        jid += 1
    (man_dir / "manifest_selftest.tsv").write_text("\n".join(man_rows) + "\n")

    (root / "watch_state.json").write_text(json.dumps({
        "selftest_L4_hx0.2_hz0.2_hy0.2": {"state": "COMPLETED", "diverged": False, "warm_loaded": True,
                                          "last_step": 500, "log": "ok"},
        "m11": {"state": "RUNNING", "diverged": False, "warm_loaded": False, "last_step": 120, "log": "..."},
    }))                                            # m11 = the hz=0.55 manifest row, not yet landed

    df = load_finals(root)
    assert len(df) == 9 + 1 + 10 + 1, f"expected 21 finals, got {len(df)}"
    assert set(df.cut) == {"electric", "magnetic"}
    hdf = add_health(df)
    assert hdf["diverged"].sum() == 1
    assert bool(hdf.loc[hdf.name == "selftest_L4_hx0.2_hz0.50_hy0.2", "above_bound"].iloc[0])
    assert not bool(hdf.loc[hdf.name == "selftest_L4_hx0.2_hz0.2_hy0.2", "above_bound"].iloc[0])
    # vscore_hot threshold at hy=0.2 is 0.5*0.04+0.3 = 0.32; the Vscore=5.0 point trips it, 0.05 doesn't
    assert bool(hdf.loc[hdf.name == "selftest_L4_hx0.2_hz0.50_hy0.2", "vscore_hot"].iloc[0])
    assert not bool(hdf.loc[hdf.name == "selftest_L4_hx0.2_hz0.2_hy0.2", "vscore_hot"].iloc[0])

    man = load_manifests(root)
    assert len(man) == 11 and set(man.columns) == set(MANIFEST_COLUMNS)
    ws = load_watch_state(root)
    assert ws["m11"]["state"] == "RUNNING"

    jt = job_table(root, df=hdf, manifests=man, watch=ws)
    assert "m11" in set(jt["name"])                              # manifest-only row survives the join
    m11 = jt[jt.name == "m11"].iloc[0]
    assert not m11["landed"] and m11["state"] == "RUNNING"       # submitted, not yet landed, watch-live
    row = jt[jt.name == "selftest_L4_hx0.2_hz0.2_hy0.2"].iloc[0]
    assert row["state"] == "COMPLETED" and row["warm_loaded"] is True

    planned = load_planned(root)
    cov = coverage(planned, hdf)
    e_cov = cov[cov.cut == "electric"].iloc[0]
    assert e_cov["landed"] == 10 and e_cov["planned"] == 11 and abs(e_cov["frac"] - 10 / 11) < 1e-9

    locs = partial_locators(root, hdf, min_points=5)
    assert len(locs["electric"]) == 1
    hc_row = locs["electric"].iloc[0]
    assert 0.15 < hc_row["h_c"] < 0.30, f"h_c out of the injected-rise window: {hc_row['h_c']}"
    assert hc_row["method"] in ("richards", "logistic", "fd_peak")
    chain = locs["chain"]
    assert set(chain.branch) == {"up", "dn"}
    assert bool(chain.iloc[0]["crossed"]) is True
    br = chain.iloc[0]["bracket"]
    assert br is not None and hxs_common[0] <= br[0] < br[1] <= hxs_common[-1]

    text = write_status_md(root, tmp / "STATUS.md", min_points=5)
    assert "hy = 0.2" in text and "electric" in text and "magnetic" in text
    assert (tmp / "STATUS.md").exists()

    # --- viewer export: shape + hc/crossing populated, curve=None (no curves_root data) --
    exp = export_viewer(root, tmp / "data" / "tc_nqs" / "phase3d", 0.2, min_points=5)
    assert exp["hy"] == 0.2 and exp["planes"] == [0.2] and exp["data_as_of"] is not None
    assert exp["bound"][str(4)] == -172 and exp["N"][str(4)] == 144
    exp_e = next(c for c in exp["cuts"] if c["kind"] == "electric")
    assert "4" in exp_e["hc"] and exp_e["hc"]["4"]["h_c"] is not None
    assert exp_e["points"][0]["curve"] is None                    # no curves_root fixture here
    exp_m = next(c for c in exp["cuts"] if c["kind"] == "first-order")
    assert "4" in exp_m["crossing"] and exp_m["crossing"]["4"]["merged"] is False
    assert exp_m["crossing"]["4"]["h_c"] is not None
    assert hxs_common[0] <= exp_m["crossing"]["4"]["h_c"] <= hxs_common[-1]
    # topo-trivial (hz=0.1 <= TOPO_TRIVIAL_HZ_MAX): "hc" from the O_FM_membrane_R1
    # winner-curve locator, independent of (and here far from) the energy crossing
    assert "4" in exp_m["hc"] and exp_m["hc"]["4"]["h_c"] is not None
    assert hxs_common[0] <= exp_m["hc"]["4"]["h_c"] <= hxs_common[-1]

    exp_empty = export_viewer(root / "nope", tmp / "nope_curves", 0.2)
    assert exp_empty["cuts"] == [] and exp_empty["planes"] == []

    print(f"SELFTEST OK ({tmp})")


# ------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/phase3d")
    ap.add_argument("--out", default="results/phase3d/STATUS.md")
    ap.add_argument("--grid-script", default=None, help="path to phase3d_grid.py (default: sibling script)")
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tmp", default=None, help="--selftest only: fixture dir (default: a fresh tempdir)")
    ap.add_argument("--export-viewer", type=float, default=None, metavar="HY",
                     help="write the drill-down-viewer JSON for one hy plane to --out")
    ap.add_argument("--curves-root", default=None,
                     help="data/tc_nqs mirror (default: swap --root's results/... for data/tc_nqs/...)")
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.tmp)
        return
    if args.export_viewer is not None:
        curves_root = args.curves_root or str(Path(args.root).parent.parent / "data" / "tc_nqs" / "phase3d")
        data = export_viewer(args.root, curves_root, args.export_viewer, args.min_points)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=1))
        print(f"wrote {out} ({out.stat().st_size} bytes, {len(data['cuts'])} cuts, curves_root={curves_root})")
        return
    text = write_status_md(args.root, args.out, args.grid_script, args.min_points)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
