"""Build results/fermionic_hx_ladder/summary.json for the fermionic L=2 OBC
architecture ladder along the magnetic line (h_z=0), h_x in
{0.1,0.2,0.3,0.5,0.7,1.0} x tiers {plain, asymm, anaC_k0, anaC_k6}.

One row per run: final energy vs the dense-ED referee, final exact-eval
fidelity/sign-match (from the last available snapshot -- snapshot_every=25,
so a diverged run's last snapshot can lag its last completed SR step),
Vscore, diverged flag, and last completed SR step.

Usage (from the worktree root):
    PYTHONPATH=<worktree> .venv/bin/python analysis/scripts/hx_ladder_summary.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results" / "fermionic_hx_ladder"
TIER_ORDER = ["plain", "asymm", "anaC_k0", "anaC_k6"]


def tier_of(name):
    for t in ("anaC_k0", "anaC_k6", "asymm", "plain"):
        if name.endswith("_" + t):
            return t
    raise ValueError(f"cannot infer tier from run name: {name}")


def find_run_jsons():
    """Final-state run JSONs, excluding curve/snapshot sidecars and ED referees."""
    out = []
    for f in sorted(ROOT.glob("*.json")):
        name = f.stem
        if name.endswith((".curve", ".snapshots")):
            continue
        if name.startswith("exact_diag") or name.startswith("ed_L2"):
            continue
        out.append(f)
    return out


def build():
    ed_cache = {}

    def ed_ref(hx, hz):
        key = (hx, hz)
        if key not in ed_cache:
            fn = ROOT / f"exact_diag_fermionic_L2_OBC_hx{hx}_hz{hz}.json"
            ed_cache[key] = json.load(open(fn))
        return ed_cache[key]

    rows = []
    for f in find_run_jsons():
        name = f.stem
        d = json.load(open(f))
        cfg = d["config"]
        hx, hz = cfg["hx"], cfg["hz"]
        tier = tier_of(name)
        kappa = cfg.get("flux_penalty", 0.0)

        E = d["observables"]["E0"]
        Vscore = d["observables"]["Vscore"]
        diverged = d["diverged"]
        last_step = d["curve"]["step"][-1]

        ed = ed_ref(hx, hz)
        E_exact = ed["E0"]
        rel = abs(E - E_exact) / abs(E_exact)

        snaps = json.load(open(ROOT / f"{name}.snapshots.json"))["series"]
        last_snap = snaps[-1]
        fidelity = last_snap["exact"]["fidelity"]
        sign_match = last_snap["exact"]["sign_match_weighted"]

        rows.append(dict(
            name=name, hx=hx, hz=hz, tier=tier, kappa=kappa,
            E=E, E_exact=E_exact, rel=rel,
            fidelity=fidelity, sign_match=sign_match,
            Vscore=Vscore, diverged=diverged, last_step=last_step,
        ))

    rows.sort(key=lambda r: (r["hx"], TIER_ORDER.index(r["tier"])))
    return rows


if __name__ == "__main__":
    rows = build()
    out = ROOT / "summary.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"wrote {len(rows)} rows to {out}")
