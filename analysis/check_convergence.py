"""Convergence + "not garbage" triage for hz-sweep runs -- BEFORE extracting O_FM.

Two modes, same acceptance criteria (all read from the run JSONs, no NetKet/weights):

  * per-dir:  --dir $PSCRATCH/tc_nqs/phase_hx0.2/L6 --L 6
  * campaign: --tree $PSCRATCH/tc_nqs [--hz-min 0.1 --hz-max 0.4 --hz-n 13
                                       --hx-vals 0.0,0.2,0.4,0.6,0.8,1.0 --l-vals 4,5,6,7]

A run is trustworthy iff: finite E and spread, E strictly below the trivial-state
bound E0(0) = -(#A_v + #B_p)  [OBC: -(L^3 + 3(L-1)^2 L)], <A_v> <= 1 (values > 1 are
impossible for a normalized state -> a broken cold-chain estimator), and the
top-level `diverged` flag is false. Anything else is flagged so it can't poison the
sweep. In --tree mode, if the expected hz grid is given, absent points are reported
MISSING (the resubmit list for run_phase_campaign.sh). Exits nonzero if anything is
flagged or missing, so it can gate a script.

Prefers the final {name}.json; falls back to {name}.curve.json for in-flight runs.
"""
import argparse
import collections
import glob
import json
import math
import os

AV_TOL = 1e-3          # <A_v> above 1+tol is unphysical (MC noise stays within tol)
VSCORE_MAX = 1.0       # Vscore above this = variance blow-up the in-run guard missed.
                       # Physical near-critical Vscores top out ~0.05; diverged ~1e5+,
                       # so 1.0 sits in the wide gap (a noisy-but-healthy ~0.1 stays ok).
FLAG_STATUSES = ("DIVERGED", "BAD-VSCORE", "BAD-ESTIMATOR", "ABOVE-BOUND")  # -> `!`


def is_ok(status):
    """Passing states: finished-good `ok`, in-flight-good `ok(ckpt)`."""
    return status.startswith("ok")


def is_flagged(status):
    """Genuine failures only. `descending` (in-flight, not yet below bound) is
    neither ok nor flagged -- it's just not done, so it must not raise `!`."""
    return status in FLAG_STATUSES

Row = collections.namedtuple(
    "Row", "hz E spread step Vs Av Bp sz sx status name d final")


def anchor_obc(L):
    """E0(h=0) = -(#A_v + #B_p) for the OBC 3D toric code."""
    return -(L ** 3 + 3 * (L - 1) ** 2 * L)


def load_runs(directory):
    """One record per run: prefer {name}.json, else {name}.curve.json.

    Robust to a JSON that is being written concurrently (a run still checkpointing):
    an unreadable / partially-flushed file is skipped with a warning rather than
    aborting the whole directory's triage (mirrors fm.iter_matching_checkpoints)."""
    runs = {}
    for jp in sorted(glob.glob(os.path.join(directory, "*.json"))):
        base = jp[:-len(".curve.json")] if jp.endswith(".curve.json") else jp[:-len(".json")]
        final = not jp.endswith(".curve.json")
        if base in runs and not final:
            continue                      # a final .json already won this base
        try:
            with open(jp) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {os.path.basename(jp)}: unreadable ({type(e).__name__}) "
                  f"— partial write? excluded", flush=True)
            continue
        runs[base] = (d, final)
    return runs


def last_finite(curve, key):
    vals = curve.get(key, [])
    return vals[-1] if vals else None


def classify(d, final, E, spread, Av, anchor, Vs=None, vscore_max=VSCORE_MAX):
    """Acceptance ladder -> status string. `ok`/`ok(ckpt)` are the passing states."""
    if d.get("diverged"):
        return "DIVERGED"                        # guard gave up -> garbage state
    if E is None or not math.isfinite(E) or (spread is not None and not math.isfinite(spread)):
        return "DIVERGED"
    if Vs is not None and math.isfinite(Vs) and Vs > vscore_max:
        return "BAD-VSCORE"                      # variance blow-up guard didn't catch
    if Av is not None and math.isfinite(Av) and Av > 1.0 + AV_TOL:
        return "BAD-ESTIMATOR"                   # <A_v> > 1 impossible for a real state
    if E > anchor:
        # finished + above bound = real failure; in-flight = just still descending
        return "ABOVE-BOUND" if final else "descending"
    return "ok" if final else "ok(ckpt)"


def detect_field(runs, candidates=("hz", "hx")):
    """The config knob that actually VARIES across a dir's runs (the swept axis).

    hz-sweeps hold hx fixed and vary hz; the hx-sweeps (phase_hz0.0/L*) do the
    opposite. Returns whichever candidate has the most distinct finite values so the
    per-dir table labels/sorts by the real sweep parameter; ties keep the historical
    'hz'. The Row still stores this value in its `.hz` slot (generic swept value)."""
    best, best_n = candidates[0], -1
    for f in candidates:
        vals = {d.get("config", {}).get(f) for (d, _) in runs.values()
                if isinstance(d.get("config", {}).get(f), (int, float))}
        if len(vals) > best_n:
            best, best_n = f, len(vals)
    return best


def build_rows(runs, anchor, vscore_max=VSCORE_MAX, field="hz"):
    """runs: {base: (doc, final)} -> sorted list of Row, one per run. `field` is the
    swept config key stored in Row.hz (hz for hz-sweeps, hx for the hx-sweeps)."""
    rows = []
    for base, (d, final) in runs.items():
        cfg = d.get("config", {})
        curve = d.get("curve", {})
        obs = d.get("observables", {}) if final else {}
        E = last_finite(curve, "energy")
        spread = last_finite(curve, "energy_spread")
        step = curve.get("step", [None])[-1] if curve.get("step") else d.get("completed_steps")
        Av = obs.get("A_v_mean")
        Vs = obs.get("Vscore")
        status = classify(d, final, E, spread, Av, anchor, Vs, vscore_max)
        rows.append(Row(cfg.get(field), E, spread, step, Vs,
                        Av, obs.get("B_p_mean"), obs.get("sz_mean"), obs.get("sx_mean"),
                        status, os.path.basename(base), d, final))
    rows.sort(key=lambda r: r.hz if r.hz is not None else 1e9)
    return rows


def expected_hz(hz_min, hz_max, hz_n):
    """The hz grid submit_nqs_hz_sweep.sh produces (same round(...,4))."""
    if hz_n <= 1:
        return [round(hz_min, 4)]
    return [round(hz_min + i * (hz_max - hz_min) / (hz_n - 1), 4) for i in range(hz_n)]


def _f(x, fmt):
    return format(x, fmt) if isinstance(x, (int, float)) and math.isfinite(x) else str(x)


def fmt_E(x):
    """Energy: fixed-point when readable, scientific for garbage-large magnitudes."""
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return str(x)
    return f"{x:.3f}" if abs(x) < 1e6 else f"{x:.3e}"


def print_table(rows, anchor, header, xlabel="hz"):
    """Per-sweep-point table (shared by both modes). Returns (ok_count, flagged_rows).

    The two observable columns follow the sweep: an hz-sweep violates the STARS, so
    <A_v> (falls from 1) and <M_z> (rises from 0) are the movers; an hx-sweep violates
    the PLAQUETTES, so <B_p> and <M_x> are — the e<->m mirror. The other pair is pinned
    (A_v commutes with h_x; M_z=0 by the global-X symmetry at hz=0, and vice versa) and
    carries no signal, so we show the informative pair for the detected sweep axis."""
    print(header)
    if xlabel == "hx":
        c1, c2, g1, g2, f2 = "<B_p>", "<M_x>", (lambda r: r.Bp), (lambda r: r.sx), ".3f"
    else:
        c1, c2, g1, g2, f2 = "<A_v>", "<M_z>", (lambda r: r.Av), (lambda r: r.sz), ".3f"
    print(f"{xlabel:>7} {'E0':>12} {'spread':>9} {'step':>5} {c1:>7} "
          f"{c2:>7} {'Vscore':>10}  status")
    ok = 0
    flagged = []
    for r in rows:
        flag = "   <-- CHECK" if is_flagged(r.status) else ""
        print(f"{r.hz!s:>7} {fmt_E(r.E):>12} {_f(r.spread, '.3f'):>9} "
              f"{r.step!s:>5} {_f(g1(r), '.4f'):>7} {_f(g2(r), f2):>7} "
              f"{_f(r.Vs, '.2e'):>10}  {r.status}{flag}")
        if is_ok(r.status):
            ok += 1
        elif is_flagged(r.status):
            flagged.append(r)              # `descending` counts as neither
    return ok, flagged


def print_traces(flagged, xlabel="hz"):
    for r in flagged:
        dstep, lines = trace_run(r.d)
        print(f"\n  {r.name} ({xlabel}={r.hz}): {r.status}"
              + (f", first non-finite at step {dstep}" if dstep is not None else ""))
        for ln in lines:
            print(ln)


def trace_run(d, window=8):
    """Return (diverge_step, lines) showing energy around the first non-finite step."""
    curve = d.get("curve", {})
    steps, En, Sp = curve.get("step", []), curve.get("energy", []), curve.get("energy_spread", [])
    bad_i = next((i for i, e in enumerate(En) if e is None or not math.isfinite(e)), None)
    lines = []
    if bad_i is None:
        return None, lines
    lo = max(0, bad_i - window)
    for i in range(lo, min(len(En), bad_i + 2)):
        mark = "  <-- first non-finite" if i == bad_i else ""
        e = f"{En[i]:.3f}" if math.isfinite(En[i]) else str(En[i])
        s = f"{Sp[i]:.3f}" if i < len(Sp) and math.isfinite(Sp[i]) else (str(Sp[i]) if i < len(Sp) else "-")
        lines.append(f"      step {steps[i]:>4}  E={e:>12}  spread={s:>10}{mark}")
    return steps[bad_i], lines


# --------------------------------------------------------------------------- #
#  modes
# --------------------------------------------------------------------------- #
def dump_energy_curve(rows, field, L, anchor, path):
    """Write a compact energy curve (only `ok`/`ok(ckpt)` rows) for the notebook's
    energy-kink diagnostic: dE/dh = -N<sigma^sweep> (Hellmann-Feynman), so we ship
    the final energy AND the conjugate magnetization for the free cross-check.

    `mag` is the magnetization conjugate to the swept field (<M_x> for an hx-sweep,
    <M_z> for an hz-sweep -- the other is ~0 by symmetry and carries no signal). We
    also ship BOTH stabilizers (<A_v>, <B_p>): the one that anticommutes with the
    field is the topological-order mover (A_v for hz, B_p for hx), the hysteresis
    partner of the magnetization. Pure JSON in/out; no NetKet -> safe on a login node.
    (In-flight `ok(ckpt)` rows have no observables yet -> mag/A_v/B_p/Vscore are null.)"""
    keep = [r for r in rows if is_ok(r.status)]
    mag_key = "mx" if field == "hx" else "mz"
    mag = (lambda r: r.sx) if field == "hx" else (lambda r: r.sz)
    out = {
        "L": L, "field_name": field, "anchor_E0": anchor,
        "field": [r.hz for r in keep],
        "E": [r.E for r in keep],
        "E_spread": [r.spread for r in keep],
        mag_key: [mag(r) for r in keep],
        "A_v": [r.Av for r in keep],
        "B_p": [r.Bp for r in keep],
        "Vscore": [r.Vs for r in keep],
        "n_kept": len(keep), "n_total": len(rows),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # don't fail if the dir is absent
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[dump] {len(keep)}/{len(rows)} points -> {path}")


def run_dir(a):
    anchor = a.anchor if a.anchor is not None else anchor_obc(a.L)
    runs = load_runs(a.dir)
    if not runs:
        raise SystemExit(f"no run JSONs in {a.dir}")
    field = a.field or detect_field(runs)              # hz-sweep -> 'hz', hx-sweep -> 'hx'
    rows = build_rows(runs, anchor, a.vscore_max, field=field)
    ok, flagged = print_table(
        rows, anchor, f"L={a.L}  sweep={field}  anchor E0(0) = {anchor:.1f}   "
                      f"({len(rows)} runs in {a.dir})", xlabel=field)
    print(f"\n{ok}/{len(rows)} converged (finite E, below bound, <A_v><=1, not diverged).")
    if flagged:
        print("flagged:", ", ".join(r.name for r in flagged))
    if a.trace:
        print_traces(flagged, xlabel=field)
    if a.dump:
        dump_energy_curve(rows, field, a.L, anchor, a.dump)
    raise SystemExit(1 if flagged else 0)


def _parse_list(s, cast):
    return [cast(x) for x in s.replace(",", " ").split()] if s else None


def run_tree(a):
    hx_want = _parse_list(a.hx_vals, float)
    l_want = _parse_list(a.l_vals, int)
    grid = (expected_hz(a.hz_min, a.hz_max, a.hz_n)
            if None not in (a.hz_min, a.hz_max, a.hz_n) else None)

    # group runs by (L, hx) read from each run's config (authoritative, as fm.py does)
    groups = collections.defaultdict(dict)          # (L, hx) -> {base: (d, final)}
    for dpath in sorted(glob.glob(os.path.join(a.tree, "phase_hx*", "L*"))):
        for base, rec in load_runs(dpath).items():
            cfg = rec[0].get("config", {})
            if cfg.get("L") is None or cfg.get("hx") is None:
                continue
            groups[(int(cfg["L"]), round(float(cfg["hx"]), 4))][base] = rec
    if not groups and not (hx_want and l_want):
        raise SystemExit(f"no phase_hx*/L* run JSONs under {a.tree}")

    # expected cells: full product if both grids given, else whatever is on disk
    Ls = sorted(l_want) if l_want else sorted({L for L, _ in groups})
    hxs = sorted(hx_want) if hx_want else sorted({hx for _, hx in groups})
    cells = {(L, round(hx, 4)): build_rows(groups.get((L, round(hx, 4)), {}),
                                           anchor_obc(L), a.vscore_max)
             for L in Ls for hx in hxs}

    # summary matrix: ok / expected(-or-found), with ! flagged / ? missing markers
    print(f"campaign QA under {a.tree}"
          + (f"   expected hz grid: {grid[0]}..{grid[-1]} x{len(grid)}" if grid else "")
          + "\n")
    print("  hx \\ L " + "".join(f"{'L=' + str(L):>10}" for L in Ls))
    any_bad = False
    detail = []
    for hx in hxs:
        cellstrs = []
        for L in Ls:
            rows = cells[(L, round(hx, 4))]
            ok = sum(1 for r in rows if is_ok(r.status))
            flagged = [r for r in rows if is_flagged(r.status)]
            found_hz = {round(r.hz, 4) for r in rows if r.hz is not None}
            missing = [h for h in grid if h not in found_hz] if grid else []
            den = len(grid) if grid else len(rows)
            mark = ("!" if flagged else "") + ("?" if missing else "")
            cellstrs.append(f"{ok}/{den}{mark}")
            if flagged or missing:
                any_bad = True
                detail.append((L, hx, flagged, missing))
        print(f"  {hx:<5}  " + "".join(f"{c:>10}" for c in cellstrs))
    print("\n  (ok/expected; ! = flagged run, ? = missing point)")

    for L, hx, flagged, missing in sorted(detail):
        print(f"\n  L={L} hx={hx}:")
        for r in flagged:
            print(f"    FLAG {r.status:<13} hz={r.hz!s:>6}  E={fmt_E(r.E)}  "
                  f"<A_v>={_f(r.Av, '.4f')}  ({r.name})")
        if missing:
            print(f"    MISSING hz: {', '.join(str(h) for h in missing)}")
        if a.trace:
            print_traces(flagged)

    if a.progress:
        print_progress(cells, Ls, hxs, grid)

    if a.summary:
        print_summary(cells, Ls, hxs, len(grid) if grid else None)

    if not any_bad:
        print("\nall expected points present, converged, and physical.")
    raise SystemExit(1 if any_bad else 0)


def print_progress(cells, Ls, hxs, grid):
    """Per-(hx x L) progress matrix from disk: finished / in-flight / not-started.
    finished = final {name}.json (has Vscore); in-flight = only a .curve.json
    checkpoint (running or killed mid-run); not-started = no file for that hz
    (still queued). Needs the expected hz grid to count not-started."""
    print("\n  progress per (hx x L)  [finished / in-flight / not-started]:")
    print("  hx \\ L " + "".join(f"{'L=' + str(L):>12}" for L in Ls))
    tf = ti = tm = 0
    for hx in hxs:
        parts = []
        for L in Ls:
            rows = cells[(L, round(hx, 4))]
            fin = sum(1 for r in rows if r.final)
            flight = sum(1 for r in rows if not r.final)
            found = {round(r.hz, 4) for r in rows if r.hz is not None}
            miss = len([h for h in grid if h not in found]) if grid else None
            tf += fin
            ti += flight
            tm += miss or 0
            parts.append(f"{fin}/{flight}/{miss if miss is not None else '?'}")
        print(f"  {hx:<5}  " + "".join(f"{c:>12}" for c in parts))
    exp = f" (of {len(Ls) * len(hxs) * len(grid)} expected)" if grid else ""
    print(f"\n  totals: {tf} finished, {ti} in-flight, {tm} not-started{exp}.")


def print_summary(cells, Ls, hxs, hz_expected):
    """Flat table of every FINISHED run (final {name}.json -> has Vscore) with its
    final energy, margin below the E0(0) bound, and Vscore. Answers 'how many are
    done and how good are they' at a glance; in-flight (ckpt-only) runs are excluded."""
    print("\n  finished runs (final JSON with observables):")
    print(f"    {'L':>2} {'hx':>4} {'hz':>6} {'E0':>12} {'margin':>9} "
          f"{'<A_v>':>7} {'<M_z>':>7} {'Vscore':>10}")
    total_finished = 0
    for L in Ls:
        anchor = anchor_obc(L)
        for hx in hxs:
            for r in cells[(L, round(hx, 4))]:
                if not r.final:
                    continue                 # ckpt-only == still in flight
                total_finished += 1
                margin = (anchor - r.E) if isinstance(r.E, (int, float)) else None
                flag = "" if r.status.startswith("ok") else f"  <-- {r.status}"
                print(f"    {L:>2} {hx:>4} {r.hz!s:>6} {fmt_E(r.E):>12} "
                      f"{_f(margin, '.2f'):>9} {_f(r.Av, '.4f'):>7} "
                      f"{_f(r.sz, '.3f'):>7} {_f(r.Vs, '.2e'):>10}{flag}")
    denom = (len(Ls) * len(hxs) * hz_expected) if hz_expected else "?"
    print(f"\n  {total_finished} finished (of {denom} expected)."
          "  margin = E0(0) - E  (positive = below the trivial bound).")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", help="one L-dir of hz-sweep runs (per-dir mode)")
    p.add_argument("--L", type=int, help="system size (required with --dir)")
    p.add_argument("--tree", help="campaign root (globs phase_hx*/L*); campaign mode")
    p.add_argument("--hz-min", type=float, default=None)
    p.add_argument("--hz-max", type=float, default=None)
    p.add_argument("--hz-n", type=int, default=None,
                   help="expected hz-grid size (with --hz-min/max) for MISSING detection")
    p.add_argument("--hx-vals", default=None, help="expected hx cuts, e.g. 0.0,0.2,0.4")
    p.add_argument("--l-vals", default=None, help="expected sizes, e.g. 4,5,6,7")
    p.add_argument("--anchor", type=float, default=None,
                   help="override the E0(0) bound (per-dir mode; default OBC formula)")
    p.add_argument("--field", default=None, choices=["hz", "hx"],
                   help="(--dir) swept parameter for the table column/sort; default "
                        "auto-detects (hx-sweeps vary hx at fixed hz, and vice versa)")
    p.add_argument("--vscore-max", type=float, default=VSCORE_MAX,
                   help=f"flag a finished run BAD-VSCORE if its Vscore exceeds this "
                        f"(default {VSCORE_MAX}; catches guard-missed variance blow-ups)")
    p.add_argument("--progress", action="store_true",
                   help="(--tree) per-(hx x L) matrix of finished/in-flight/not-started "
                        "run counts (needs --hz-min/max/n for not-started)")
    p.add_argument("--summary", action="store_true",
                   help="(--tree) after the matrix, list every FINISHED run with its "
                        "final E, margin below the bound, and Vscore")
    p.add_argument("--trace", action="store_true",
                   help="for each flagged run, print the energy curve around the "
                        "first non-finite step (early=unlucky init, late=instability)")
    p.add_argument("--dump", default=None, metavar="OUT.json",
                   help="(--dir) also write a compact energy curve (field, E, "
                        "E_spread, conjugate magnetization, Vscore) of the converged "
                        "points, for the energy-kink diagnostic notebook")
    a = p.parse_args(argv)

    if bool(a.tree) == bool(a.dir):
        p.error("give exactly one of --dir (with --L) or --tree")
    if a.dir:
        if a.L is None:
            p.error("--L is required with --dir")
        run_dir(a)
    else:
        run_tree(a)


if __name__ == "__main__":
    main()
