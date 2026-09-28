"""Exact-fidelity scorer for the L=2 OBC hy-axis cold-start certification.

Loads a finished train.py run (its {name}.json config + {name}.ckpt.mpack
weights, plus any {name}.step*.mpack snapshots), rebuilds the variational
state via tc3d.builders.build_state, densifies it with MCState.to_array()
(4096-dim at L=2 OBC — the ONLY size this is allowed at), and scores it
against a ground vector saved by analysis/scripts/ed_referee_hy.py (--out
writes gs_*.npz in netket hi.all_states() order, the same convention
to_array() uses; see the referee's `basis_convention` note).

Two scores per state: fidelity to psi0, and fidelity to the ground MANIFOLD
(projector onto every eigenvector within --deg_tol of E0) — on the pure-hy
line the L=2 gap collapses to ~1e-3 by hy=1.2 (first-order level crossing),
where any state in the near-degenerate doublet is an equally correct answer.

    python analysis/scripts/hy_cert_fidelity.py \
        --json $OUT/hy_axis_l2cert_hy0.4_ds1e-3_wf0_s0.json \
        --gs results/hy_l2_certification/gs_L2_OBC_hx0.0_hy0.4_hz0.0_dual.npz
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from tc3d.builders import build_state
from tc3d.io import load_weights

SNAPSHOT_RE = re.compile(r"\.step(\d+)\.mpack$")


def scores(vs, manifold):
    """(F_ground, F_manifold): |<psi|v0>|^2 and sum_i |<psi|v_i>|^2."""
    psi = np.asarray(vs.to_array(normalize=True))
    amps = np.abs(manifold.conj().T @ psi) ** 2
    return float(amps[0]), float(amps.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", required=True, help="train.py run JSON")
    ap.add_argument("--gs", required=True, help="ed_referee_hy.py gs_*.npz")
    ap.add_argument("--deg_tol", type=float, default=1e-2,
                    help="eigenvalues within this of E0 count as the ground "
                         "manifold (default 1e-2 absorbs the hy=1.2 doublet)")
    ap.add_argument("--no_snapshots", action="store_true",
                    help="score only the final checkpoint, skip .step*.mpack")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <run>.fidelity.json)")
    args = ap.parse_args()

    with open(args.json) as f:
        meta = json.load(f)
    cfg = dict(meta["config"])
    assert cfg["L"] == 2 and cfg["bc"] == "OBC", (
        "memory safety: to_array() is L=2 OBC ONLY (4096-dim); anything "
        "bigger must never be densified (see CLAUDE.md)")

    ref = np.load(args.gs)
    w = np.asarray(ref["eigenvalues"])
    E0_ed = float(w[0])
    psi0 = np.asarray(ref["psi"])
    vecs = np.asarray(ref["vecs"]) if "vecs" in ref.files else psi0[:, None]
    n_deg = max(1, int(np.sum(w - w[0] < args.deg_tol)))
    manifold = vecs[:, :min(n_deg, vecs.shape[1])]
    manifold = manifold / np.linalg.norm(manifold, axis=0, keepdims=True)

    geo, hi, Ham, vs, xz_stabs = build_state(cfg)
    assert hi.n_states == manifold.shape[0], (
        f"Hilbert mismatch: state {hi.n_states} vs ED vectors {manifold.shape[0]}")

    base = args.json[:-len(".json")]
    series = []
    if not args.no_snapshots:
        snaps = sorted(
            (int(m.group(1)), p) for p in glob.glob(f"{base}.step*.mpack")
            for m in [SNAPSHOT_RE.search(p)] if m)
        for step, p in snaps:
            vs = load_weights(vs, p[:-len(".mpack")])
            F0, Fm = scores(vs, manifold)
            series.append({"step": step, "fidelity": F0, "fidelity_manifold": Fm})
            print(f"[fidelity] step {step:4d}: F0={F0:.8f}  Fman={Fm:.8f}", flush=True)

    vs = load_weights(vs, base + ".ckpt")
    F0, Fm = scores(vs, manifold)
    print(f"[fidelity] final ckpt : F0={F0:.8f}  Fman={Fm:.8f}  "
          f"(E_MC={meta.get('observables', {}).get('E0')}  E0_ED={E0_ed:.9f}  "
          f"n_deg={manifold.shape[1]})", flush=True)

    result = {
        "name": meta.get("name"), "source_json": os.path.basename(args.json),
        "gs_npz": args.gs, "E0_ED": E0_ed, "ED_eigenvalues": w.tolist(),
        "deg_tol": args.deg_tol, "n_deg": manifold.shape[1],
        "hx": cfg["hx"], "hy": cfg["hy"], "hz": cfg["hz"],
        "seed": cfg.get("seed"), "diag_shift": cfg.get("diag_shift"),
        "warmup_frac": cfg.get("warmup_frac"),
        "E_MC": meta.get("observables", {}).get("E0"),
        "diverged": meta.get("diverged"),
        "fidelity_final": F0, "fidelity_manifold_final": Fm,
        "fidelity_series": series,
    }
    out = args.out or base + ".fidelity.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fidelity] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
