"""2D convergence + "not garbage" triage for model/train_2d runs -- the 2D analogue
of check_convergence.py, to run BEFORE extracting O_FM.

Reuses that module's acceptance ladder (`classify`) but with the **2D** OBC anchor
E0(0) = -(#A_v + #B_p) = -(L^2 + (L-1)^2), and it AUTO-DETECTS the swept field per
directory -- hz for the electric (hx-fixed) sweep, hx for the magnetic (hz-fixed)
sweep -- so both analytic cuts print correctly (the 3D tool assumes hz is always
swept, which inverts the magnetic sweep).

  per-dir:   --dir $PSCRATCH/tc_nqs_2d/phase_hx0.0/L6
  campaign:  --tree $PSCRATCH/tc_nqs_2d              (globs phase_*/L*)

A run passes (`ok` / `ok(ckpt)`) iff: finite E and spread, E strictly below the
trivial-state bound E0(0), <A_v> <= 1 (values > 1 are impossible for a normalized
state), Vscore below the blow-up gate, and top-level `diverged` is false. Prefers the
final {name}.json; falls back to {name}.curve.json for a run scored from its latest
checkpoint (shown as `ok(ckpt)` / `descending`). Exits nonzero if anything is flagged.
"""
import argparse
import glob
import json
import math
import os

from analysis.check_convergence import (
    classify, load_runs, last_finite, is_ok, is_flagged, fmt_E, _f, VSCORE_MAX)


def anchor_2d(L):
    """E0(h=0) = -(#A_v + #B_p) = -(L^2 + (L-1)^2) for the OBC 2D surface code."""
    return -(L ** 2 + (L - 1) ** 2)


def detect_swept(runs):
    """The config field that VARIES across a dir is the sweep axis (hz electric / hx
    magnetic); the other is held fixed. Ties (single point) default to hz."""
    seen = {"hx": set(), "hz": set()}
    for d, _ in runs.values():
        cfg = d.get("config", {})
        for k in seen:
            if cfg.get(k) is not None:
                seen[k].add(round(float(cfg[k]), 4))
    return "hx" if len(seen["hx"]) > len(seen["hz"]) else "hz"


def build_rows(runs, field, anchor, vscore_max=VSCORE_MAX):
    """One row per run, sorted by the swept field. Reuses the 3D acceptance ladder."""
    rows = []
    for base, (d, final) in runs.items():
        cfg = d.get("config", {})
        curve = d.get("curve", {})
        obs = d.get("observables", {}) if final else {}
        E = obs.get("E0")
        if E is None:
            E = last_finite(curve, "energy")
        var = obs.get("Var")
        spread = (math.sqrt(var) if isinstance(var, (int, float)) and var >= 0
                  else last_finite(curve, "energy_spread"))
        Vs = obs.get("Vscore")
        if Vs is None:
            Vs = last_finite(curve, "vscore")
        Av = obs.get("A_v_mean")
        step = (curve.get("step", [None])[-1] if curve.get("step")
                else d.get("completed_steps"))
        status = classify(d, final, E, spread, Av, anchor, Vs, vscore_max)
        rows.append({"field": cfg.get(field), "E": E, "spread": spread, "step": step,
                     "Vs": Vs, "Av": Av, "sz": obs.get("sz_mean"),
                     "status": status, "name": os.path.basename(base), "final": final})
    rows.sort(key=lambda r: r["field"] if r["field"] is not None else 1e9)
    return rows


def print_table(rows, field, anchor, header):
    """Per-point table. Returns (ok_count, flagged_rows)."""
    print(header)
    print(f"{field:>7} {'E0':>12} {'spread':>9} {'margin':>9} {'step':>5} "
          f"{'<A_v>':>7} {'<M_z>':>7} {'Vscore':>10}  status")
    ok, flagged = 0, []
    for r in rows:
        margin = (anchor - r["E"]) if isinstance(r["E"], (int, float)) else None
        flag = "   <-- CHECK" if is_flagged(r["status"]) else ""
        print(f"{r['field']!s:>7} {fmt_E(r['E']):>12} {_f(r['spread'], '.4f'):>9} "
              f"{_f(margin, '.3f'):>9} {r['step']!s:>5} {_f(r['Av'], '.4f'):>7} "
              f"{_f(r['sz'], '.3f'):>7} {_f(r['Vs'], '.2e'):>10}  {r['status']}{flag}")
        if is_ok(r["status"]):
            ok += 1
        elif is_flagged(r["status"]):
            flagged.append(r)
    return ok, flagged


def check_dir(dirpath, L=None, vscore_max=VSCORE_MAX):
    """Triage one L-directory. Returns the flagged rows (empty = all good)."""
    if L is None:
        base = os.path.basename(dirpath.rstrip("/"))
        L = int(base[1:]) if base.startswith("L") and base[1:].isdigit() else None
    if L is None:
        raise SystemExit(f"can't infer L from {dirpath!r}; pass --L")
    runs = load_runs(dirpath)
    if not runs:
        print(f"  (no run JSONs in {dirpath})")
        return []
    field = detect_swept(runs)
    anchor = anchor_2d(L)
    sector = "electric" if field == "hz" else "magnetic"
    rows = build_rows(runs, field, anchor, vscore_max)
    ok, flagged = print_table(
        rows, field, anchor,
        f"L={L}  {sector} ({field} swept)  anchor E0(0)={anchor}   "
        f"({len(rows)} runs)  {dirpath}")
    print(f"  -> {ok}/{len(rows)} converged "
          f"(finite E, below {anchor}, <A_v><=1, Vscore ok, not diverged)."
          + (f"  FLAGGED: {', '.join(r['name'] for r in flagged)}" if flagged else ""))
    return flagged


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", help="one L-dir of train_2d runs (per-dir mode)")
    p.add_argument("--L", type=int, default=None, help="system size (else inferred from L<n> dir)")
    p.add_argument("--tree", help="2D campaign root (globs phase_*/L*); default $PSCRATCH/tc_nqs_2d")
    p.add_argument("--vscore-max", type=float, default=VSCORE_MAX,
                   help=f"flag BAD-VSCORE above this (default {VSCORE_MAX})")
    a = p.parse_args(argv)

    if a.dir:
        flagged = check_dir(a.dir, a.L, a.vscore_max)
        raise SystemExit(1 if flagged else 0)

    root = a.tree or os.path.expandvars("$PSCRATCH/tc_nqs_2d")
    dirs = sorted(glob.glob(os.path.join(root, "phase_*", "L*")))
    if not dirs:
        raise SystemExit(f"no phase_*/L* dirs under {root}")
    any_bad = False
    for i, d in enumerate(dirs):
        if i:
            print()
        any_bad |= bool(check_dir(d, None, a.vscore_max))
    print("\n" + ("some runs FLAGGED (see above)." if any_bad
                  else "all runs converged and physical."))
    raise SystemExit(1 if any_bad else 0)


if __name__ == "__main__":
    main()
