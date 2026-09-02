"""Build a (h_x, h_z) x arm summary table for the fermionic plane campaign.

Generic version of hx_ladder_summary.py: given a results directory containing
gridinv_fermionic_L{L}_{bc}_hx{hx}_hz{hz}_k{k}_{arm}.json runs and
exact_diag_fermionic_L{L}_{bc}_hx{hx}_hz{hz}.json ED referees, builds one row
per (hx, hz, arm) with E, E_exact, rel, fidelity, sign_match, diverged, last_step
(same fields as results/fermionic_hx_ladder/summary.json).

h_y != 0 runs insert "_hy{hy}" between hx and hz (matching
analysis/scripts/eval_snapshots.py's `_ed_tag` / ed_electric_line.py's
conditional naming); hy=0.0 keeps the exact-form filenames above. Each row
carries "hy" (0.0 when absent from the filename); pass `hy=` to `build()` to
keep only rows at that field value.

Tolerant by construction: unmatched filenames are skipped, a missing ED
referee leaves E_exact/rel as None, a missing/empty snapshots sidecar leaves
fidelity/sign_match as None. Callers reindex the returned rows onto their own
(hx, hz, arm) grid and NaN-fill absent cells -- this loader never pads.

Usage (importable, or standalone):
    PYTHONPATH=<worktree> .venv/bin/python analysis/scripts/plane_summary.py \
        --dir results/fermionic_plane_L2 --out results/fermionic_plane_L2/summary.json
"""
import argparse
import json
import re
from pathlib import Path

RUN_RE = re.compile(
    r"^gridinv_fermionic_L(\d+)_(OBC|PBC)_hx([\d.]+)(?:_hy([\d.]+))?_hz([\d.]+)_k(\d+)_(.+)$")


def find_run_jsons(root):
    """Final-state run JSONs, excluding curve/snapshot sidecars and ED referees."""
    root = Path(root)
    out = []
    for f in sorted(root.glob("*.json")):
        name = f.stem
        if name.endswith((".curve", ".snapshots")):
            continue
        if name.startswith(("exact_diag", "ed_L")):
            continue
        out.append(f)
    return out


def build(root, arms=None, hy=None):
    """rows for every matched run file; arms=None keeps whatever arm the
    filename carries (no restriction), else filters to that allow-list.
    hy=None keeps every hy found (0.0 for filenames with no "_hy" tag); pass
    a float to keep only rows at that field value."""
    root = Path(root)
    ed_cache = {}

    def ed_ref(L, bc, hx, hy_, hz):
        key = (L, bc, hx, hy_, hz)
        if key not in ed_cache:
            hy_part = f"_hy{hy_}" if hy_ else ""
            fn = root / f"exact_diag_fermionic_L{L}_{bc}_hx{hx}{hy_part}_hz{hz}.json"
            ed_cache[key] = json.load(open(fn)) if fn.exists() else None
        return ed_cache[key]

    rows = []
    for f in find_run_jsons(root):
        m = RUN_RE.match(f.stem)
        if not m:
            continue
        L, bc, hx_s, hy_s, hz_s, k, arm = m.groups()
        if arms is not None and arm not in arms:
            continue
        hx, hz = float(hx_s), float(hz_s)
        hy_ = float(hy_s) if hy_s is not None else 0.0
        if hy is not None and abs(hy_ - hy) > 1e-9:
            continue

        d = json.load(open(f))
        E = d["observables"]["E0"]
        Vscore = d["observables"]["Vscore"]
        diverged = d["diverged"]
        last_step = d["curve"]["step"][-1] if d["curve"]["step"] else None

        ed = ed_ref(L, bc, hx, hy_, hz)
        E_exact = ed["E0"] if ed else None
        rel = abs(E - E_exact) / abs(E_exact) if E_exact else None

        fidelity = sign_match = None
        snap_f = root / f"{f.stem}.snapshots.json"
        if snap_f.exists():
            series = json.load(open(snap_f))["series"]
            if series:
                last_snap = series[-1]
                fidelity = last_snap.get("exact", {}).get("fidelity")
                sign_match = last_snap.get("exact", {}).get("sign_match_weighted")

        rows.append(dict(
            name=f.stem, L=int(L), bc=bc, hx=hx, hy=hy_, hz=hz, k=int(k), arm=arm,
            E=E, E_exact=E_exact, rel=rel,
            fidelity=fidelity, sign_match=sign_match,
            Vscore=Vscore, diverged=diverged, last_step=last_step,
        ))

    rows.sort(key=lambda r: (r["hz"], r["hx"], r["arm"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="results/fermionic_plane_L2")
    ap.add_argument("--arms", nargs="+", default=None,
                     help="restrict to these arm names; default keeps all found")
    ap.add_argument("--hy", type=float, default=None,
                     help="restrict to this hy value; default keeps all found")
    ap.add_argument("--out", default=None, help="write rows as JSON here")
    args = ap.parse_args()

    rows = build(args.dir, args.arms, args.hy)
    print(f"loaded {len(rows)} rows from {args.dir}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
