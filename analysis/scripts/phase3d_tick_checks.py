"""Numeric health checks for the 2-hourly phase3d tick (user, 2026-09-23: numbers first, plots only on a flag).

Over the pulled finals (<root>/{hy*,ycuts}/<cut>/L4/*.json, top level only -- anchor_trials/ and redo_*/ are
skipped), for every chain branch that gained a point in the last --since-min minutes:
  - unhealthy points: diverged, E0 at/above the h=0 bound, >= 5 guard rollbacks;
  - E must not rise with the swept field (dE/dh = -N<sigma> <= 0; a rise = a stuck or relaxing state);
  - Hellmann-Feynman between neighbours in the same state (|d sigma| < 0.1): slope vs -N<sigma> (h_x/h_z
    sweeps only: <sigma_y> is too state-sensitive, the y-cut topological branches read ~2x off);
  - deep-y-polarized anchors (<sigma_y> >= 0.85, h_y >= 1.3): phase3d_reseed.anchor_verdict
    (<B_p> vs its leading order at any h_x; energy vs the strong-field series at h_x = 0).
--all-anchors lists every y-polarized anchor on disk, not just the new ones.

    python analysis/scripts/phase3d_tick_checks.py --root <main>/results/phase3d --since-min 125
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from phase3d_reseed import anchor_verdict

N, BOUND = 144, -172.0                 # L=4 OBC edges, exact h=0 energy
SWEEP_OF = (("ycut_", "hy"), ("electric_", "hz"), ("magnetic_", "hx"))
SIGMA = {"hx": "sx_mean", "hy": "sy_mean", "hz": "sz_mean"}


def load(root):
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(root, "*", "*", "L4", "*.json")):
        if f.endswith((".curve.json", ".snapshots.json")) or "finaleval" in f:
            continue
        cut = os.path.basename(os.path.dirname(os.path.dirname(f)))
        sweep = next((s for p, s in SWEEP_OF if cut.startswith(p)), None)
        m = re.search(r"_(up|dn)\.json$", f)
        if sweep is None or not m:
            continue
        with open(f) as fh:
            d = json.load(fh)
        o, c = d["observables"], d["config"]
        groups[(os.path.relpath(os.path.dirname(f), root), m.group(1), sweep)].append({
            "h": c[sweep], "hx": c["hx"], "hy": c.get("hy", 0.0), "hz": c["hz"], "E0": o.get("E0"),
            "sig": o.get(SIGMA[sweep]), "sy": o.get("sy_mean"), "B_p": o.get("B_p_mean"),
            "rb": d.get("n_rollbacks") or 0, "div": d.get("diverged"), "mtime": os.path.getmtime(f)})
    return groups


def check_branch(pts, branch, sweep):
    flags = []
    for p in pts:
        if p["div"] or p["E0"] is None or p["E0"] >= BOUND:
            flags.append(f"UNHEALTHY h={p['h']:g} (diverged={p['div']}, E0={p['E0']})")
        elif p["rb"] >= 5:
            flags.append(f"rollbacks h={p['h']:g}: {p['rb']}")
    ok = [p for p in pts if not p["div"] and p["E0"] is not None and p["E0"] < BOUND]
    for a, b in zip(ok, ok[1:]):
        slope = (b["E0"] - a["E0"]) / (b["h"] - a["h"])
        if b["E0"] - a["E0"] > 0.5:
            flags.append(f"E RISES with field {a['h']:g}->{b['h']:g}: {a['E0']:.2f} -> {b['E0']:.2f}")
        elif sweep != "hy" and abs(b["sig"] - a["sig"]) < 0.1:      # <sigma_y> is too state-sensitive for HF (CLAUDE.md)
            exp = -N * (a["sig"] + b["sig"]) / 2
            if abs(slope - exp) > 0.3 * abs(exp) + 3:
                flags.append(f"HF {a['h']:g}->{b['h']:g}: slope {slope:.1f} vs -N<s> {exp:.1f}")
    anchor = pts[0] if branch == "up" else pts[-1]
    return flags, anchor


def anchor_line(key, a):
    if a["sy"] is None or a["sy"] < 0.85 or a["hy"] < 1.3 or a["div"] or a["E0"] is None:
        return None
    ok, dE, ratio = anchor_verdict(a["hx"], a["hy"], a["hz"], a["E0"], a["B_p"])
    de = f"dE {dE:+.1f}" if dE is not None else "dE n/a"
    return (f"  anchor {'ok   ' if ok else 'STUCK'} {key[0]} {key[1]} @ {key[2]}={a['h']:g}: "
            f"<B_p> {a['B_p']:.3f} ({ratio:.2f}x lead), {de}")


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--since-min", type=float, default=125)
    p.add_argument("--all-anchors", action="store_true")
    a = p.parse_args(argv)
    cutoff = time.time() - 60 * a.since_min
    groups = load(a.root)
    touched = {k: v for k, v in groups.items() if any(q["mtime"] >= cutoff for q in v)}
    print(f"[checks] {sum(len(v) for v in groups.values())} chain finals, {len(touched)} branch(es) touched "
          f"in the last {a.since_min:g} min")
    for key in sorted(groups):
        pts = sorted(groups[key], key=lambda q: q["h"])
        flags, anchor = check_branch(pts, key[1], key[2])
        new = [q["h"] for q in pts if q["mtime"] >= cutoff]
        if key in touched:
            print(f"- {key[0]} {key[1]}: +{len(new)} pts {new}  ({len(pts)} total)")
            for f in flags:
                print(f"    ! {f}")
        line = anchor_line(key, anchor)
        if line and (key in touched and anchor["mtime"] >= cutoff or a.all_anchors):
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
