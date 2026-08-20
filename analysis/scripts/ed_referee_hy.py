"""Dense ED referee for the bosonic 3D toric code with h_y != 0 (sign-full
regime, where QMC fails), L=2 OBC only (12 qubits, 4096-dim -- densifying
and diagonalizing this is safe on an 8 GB dev machine; see CLAUDE.md).

Builds H via tc3d.hamiltonian.create_hamiltonian (NetKet PauliStrings, complex
dtype), densifies with `.to_dense()` (~256 MB at this size), and diagonalizes
with `scipy.linalg.eigh`.

Basis convention: row/column i of the dense H (and of the saved ground vector)
follows NetKet's `hi.all_states()[i]` ordering. This is NOT the little-endian
bit convention used by `tc3d.exact_diag`'s matrix-free kernels (site i <-> bit
i of the index) -- verified different by direct comparison. A matching NQS
wavefunction must be built the same way, e.g. `MCState.to_array()` (which
enumerates in `hi.all_states()` order internally), not a hand-rolled bit loop.

Self-checks (always active, not opt-in):
  - at hx=hz=hy=0: E0 must equal the analytic h=0 anchor -(n_stars+n_plaqs).
  - with --dual: build BOTH the primal and Hadamard-conjugated (dual) H at
    this field point and assert their E0 agree to <=1e-10 (same physical
    spectrum, unitary transform -- a real bug here is OUR construction, not
    the physics).

Usage:
  python analysis/scripts/ed_referee_hy.py --hx 0 --hy 0 --hz 0 --dual \
      --out results/hy_l2_certification/ed_L2_OBC_h0.json
  python analysis/scripts/ed_referee_hy.py --hx 0.2 --hy 0.2 --hz 0.1 --dual \
      --out results/hy_l2_certification/ed_L2_OBC_hx0.2_hy0.2_hz0.1.json
"""
import argparse
import json
import os

import numpy as np
import netket as nk
import scipy.linalg

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.hamiltonian import create_hamiltonian


def build_dense_H(geo, hi, hx, hy, hz, dual):
    H = create_hamiltonian(hi=hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all,
                           bonds=geo.bonds, hx=hx, hy=hy, hz=hz, J=1.0,
                           dtype=complex, dual=dual)
    return np.asarray(H.to_dense())


def lowest_eigs(Hd, k):
    k = min(k, Hd.shape[0])
    return scipy.linalg.eigh(Hd, subset_by_index=[0, k - 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--bc", default="OBC", choices=["OBC"],
                    help="OBC only -- L=2 PBC is 2^24 states and OOMs an 8 GB "
                         "dev machine (see CLAUDE.md)")
    ap.add_argument("--hx", type=float, default=0.0)
    ap.add_argument("--hy", type=float, default=0.0)
    ap.add_argument("--hz", type=float, default=0.0)
    ap.add_argument("--dual", action="store_true",
                    help="build H in the Hadamard-conjugated (dual) basis -- "
                         "matches --dual_basis training -- and self-check its "
                         "E0 against the primal build")
    ap.add_argument("--k", type=int, default=4, help="number of lowest eigenpairs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert args.L == 2 and args.bc == "OBC", (
        "memory safety: this referee is L=2 OBC ONLY (4096-dim). NEVER build "
        "L=2 PBC here (2^24 states, ~2.7 GB Lanczos workspace -- OOMs this box).")

    geo = ThreeD_ToricCodeGeometry(args.L, args.L, args.L, bc=args.bc)
    hi = nk.hilbert.Spin(s=1 / 2, N=geo.N)
    n_stars, n_plaqs = len(geo.vertex_all), len(geo.plaq_all)
    print(f"L={args.L} {args.bc}: N={geo.N} qubits, dim={hi.n_states}, "
          f"{n_stars} stars + {n_plaqs} plaquettes", flush=True)

    Hd = build_dense_H(geo, hi, args.hx, args.hy, args.hz, args.dual)
    w, v = lowest_eigs(Hd, args.k)
    E0, psi0 = float(w[0]), v[:, 0]
    gap = float(w[1] - w[0]) if len(w) > 1 else None
    print(f"[ed] E0={E0:.12f}  gap={gap}  eigenvalues={w.tolist()}", flush=True)

    # ---- self-check 1: h=0 analytic anchor -------------------------------
    h0_check = None
    if args.hx == 0.0 and args.hy == 0.0 and args.hz == 0.0:
        anchor = -float(n_stars + n_plaqs)
        delta = abs(E0 - anchor)
        h0_check = {"anchor": anchor, "E0": E0, "delta": delta, "ok": bool(delta < 1e-8)}
        print(f"[selfcheck h=0] {'ok  ' if h0_check['ok'] else 'FAIL'} "
              f"E0={E0:.12f} anchor={anchor} delta={delta:.2e}", flush=True)
        assert h0_check["ok"], f"h=0 anchor mismatch: E0={E0} != {anchor}"

    # ---- self-check 2: primal vs dual agreement --------------------------
    dual_check = None
    if args.dual:
        Hd_primal = build_dense_H(geo, hi, args.hx, args.hy, args.hz, dual=False)
        w_p, _ = lowest_eigs(Hd_primal, 1)
        delta = abs(float(w_p[0]) - E0)
        dual_check = {"E0_primal": float(w_p[0]), "E0_dual": E0, "delta": delta,
                     "ok": bool(delta <= 1e-10)}
        print(f"[selfcheck dual] {'ok  ' if dual_check['ok'] else 'FAIL'} "
              f"E0_primal={w_p[0]:.12f} E0_dual={E0:.12f} delta={delta:.3e}", flush=True)
        assert dual_check["ok"], f"primal/dual E0 mismatch: {dual_check}"

    result = {
        "L": args.L, "bc": args.bc, "hx": args.hx, "hy": args.hy, "hz": args.hz,
        "dual": args.dual, "J": 1.0, "N": geo.N, "dim": int(hi.n_states),
        "n_stars": n_stars, "n_plaqs": n_plaqs,
        "basis_convention": (
            "row/col i of the dense H and the saved ground vector follow "
            "netket hi.all_states()[i] ordering; feed the SAME array (or use "
            "MCState.to_array()) to compare an NQS wavefunction -- this is NOT "
            "the little-endian bit convention (site i <-> bit i) used by "
            "tc3d.exact_diag's matrix-free kernels."),
        "k": args.k, "eigenvalues": w.tolist(), "E0": E0, "gap": gap,
        "h0_anchor_check": h0_check, "primal_dual_check": dual_check,
    }
    if args.out:
        out_dir = os.path.dirname(args.out) or "."
        os.makedirs(out_dir, exist_ok=True)
        npz_path = os.path.join(
            out_dir, f"gs_L{args.L}_{args.bc}_hx{args.hx}_hy{args.hy}_hz{args.hz}"
                    f"{'_dual' if args.dual else ''}.npz")
        np.savez(npz_path, psi=psi0, eigenvalues=w)
        result["gs_npz"] = npz_path
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[ed] wrote {args.out} + {npz_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
