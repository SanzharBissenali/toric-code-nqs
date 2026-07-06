"""Quick convergence triage for an hz-sweep run directory — BEFORE extracting O_FM.

Scans one L-dir (e.g. $PSCRATCH/tc_nqs/phase_hx0.2/L6), reads each run's energy
curve (no NetKet, no weights — just the JSONs), and flags any run that diverged
(E = -inf / nan) or never cleared the trivial-state bound E0(0) = -(#A_v+#B_p).
For OBC that anchor is -(L^3 + 3(L-1)^2 L); a converged finite-field run MUST sit
strictly below it. Prints a per-hz table sorted by field so a poisoned point is
obvious before it contaminates the O_FM sweep.

    python analysis/check_convergence.py --dir $PSCRATCH/tc_nqs/phase_hx0.2/L6 --L 6

Prefers the final {name}.json; falls back to {name}.curve.json for runs still
in flight / timed out (uses the latest checkpointed energy).
"""
import argparse
import glob
import json
import math
import os


def anchor_obc(L):
    """E0(h=0) = -(#A_v + #B_p) for the OBC 3D toric code."""
    return -(L ** 3 + 3 * (L - 1) ** 2 * L)


def load_runs(directory):
    """One record per run: prefer {name}.json, else {name}.curve.json."""
    runs = {}
    for jp in sorted(glob.glob(os.path.join(directory, "*.json"))):
        base = jp[:-len(".curve.json")] if jp.endswith(".curve.json") else jp[:-len(".json")]
        final = not jp.endswith(".curve.json")
        if base in runs and not final:
            continue                      # a final .json already won this base
        with open(jp) as f:
            d = json.load(f)
        runs[base] = (d, final)
    return runs


def last_finite(curve, key):
    vals = curve.get(key, [])
    return vals[-1] if vals else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="one L-dir of hz-sweep runs")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--anchor", type=float, default=None,
                   help="override the E0(0) bound (default: OBC formula)")
    a = p.parse_args(argv)

    anchor = a.anchor if a.anchor is not None else anchor_obc(a.L)
    runs = load_runs(a.dir)
    if not runs:
        raise SystemExit(f"no run JSONs in {a.dir}")

    rows = []
    for base, (d, final) in runs.items():
        cfg = d.get("config", {})
        curve = d.get("curve", {})
        hz = cfg.get("hz")
        E = last_finite(curve, "energy")
        spread = last_finite(curve, "energy_spread")
        step = curve.get("step", [None])[-1] if curve.get("step") else d.get("completed_steps")
        Vs = d.get("observables", {}).get("Vscore") if final else None

        bad_E = E is None or not math.isfinite(E)
        bad_sp = spread is not None and not math.isfinite(spread)
        if bad_E or bad_sp:
            status = "DIVERGED"
        elif E > anchor:
            status = "ABOVE-BOUND"          # cleared nothing — worse than trivial
        else:
            status = "ok" if final else "ok(ckpt)"
        rows.append((hz if hz is not None else 1e9, hz, E, spread, step, Vs, status,
                     os.path.basename(base)))

    rows.sort()
    print(f"L={a.L}  anchor E0(0) = {anchor:.1f}   ({len(rows)} runs in {a.dir})")
    print(f"{'hz':>6} {'E0':>12} {'spread':>9} {'step':>5} {'Vscore':>10}  status")
    ok = 0
    for _, hz, E, sp, step, Vs, status, name in rows:
        Es = f"{E:.3f}" if isinstance(E, (int, float)) and math.isfinite(E) else str(E)
        sps = f"{sp:.3f}" if isinstance(sp, (int, float)) and math.isfinite(sp) else str(sp)
        Vss = f"{Vs:.2e}" if isinstance(Vs, (int, float)) else "-"
        flag = "" if status.startswith("ok") else "   <-- CHECK"
        print(f"{hz!s:>6} {Es:>12} {sps:>9} {step!s:>5} {Vss:>10}  {status}{flag}")
        if status.startswith("ok"):
            ok += 1
    bad = [r[-1] for r in rows if not r[6].startswith("ok")]
    print(f"\n{ok}/{len(rows)} converged (finite E, below bound).")
    if bad:
        print("flagged:", ", ".join(bad))


if __name__ == "__main__":
    main()
