"""
Exact diagonalisation reference for the 3D toric code Hamiltonian.

Builds the Hamiltonian at a given (L, hx, hy, hz), runs Lanczos to obtain
the few lowest eigenvalues, recovers the ground-state vector, and computes
reference values for every observable we'll later log in the NQS runs:

  - E_0, E_1 (and the gap)
  - <A_v>, <B_p> for every stabilizer
  - <sigma_x>, <sigma_y>, <sigma_z> for every qubit

The output is a single JSON file at:
    test_exact_diag_L{L}_hx{hx}_hy{hy}_hz{hz}.json
which the NQS test scripts can load to get a ground-truth reference
without recomputing.

Feasible up to ~25 qubits for full sparse Lanczos -- so L=2 PBC is fine
(N=24), L=3 PBC is NOT (N=81, 2^81 states).
"""

import argparse
import json
import time

import numpy as np
import netket as nk

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.hamiltonian import create_hamiltonian


def _safe_real(x):
    """Take the real part of a possibly-complex netket Stats / scalar."""
    return float(np.real(np.asarray(x)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--Lz", type=int, default=2)
    p.add_argument("--bc", type=str, default="PBC")
    p.add_argument("--hx", type=float, default=0.2)
    p.add_argument("--hy", type=float, default=0.0)
    p.add_argument("--hz", type=float, default=0.2)
    p.add_argument("--J",  type=float, default=1.0)
    p.add_argument("--k",  type=int,   default=2,
                   help="Number of lowest eigenvalues to compute (>=2 to get the gap)")
    p.add_argument("--dtype", type=str, default="float64",
                   choices=["float64", "complex"])
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # ---------------------------------------------------------------------
    # Build geometry, Hilbert space, Hamiltonian
    # ---------------------------------------------------------------------
    geo = ThreeD_ToricCodeGeometry(args.Lx, args.Ly, args.Lz, args.bc)
    hi  = nk.hilbert.Spin(s=1/2, N=geo.N)

    print(f"3D toric code: L=({args.Lx},{args.Ly},{args.Lz}) {args.bc}")
    print(f"  N qubits      = {geo.N}")
    print(f"  N vertices    = {len(geo.vertex_all)}")
    print(f"  N plaquettes  = {len(geo.plaq_all)}")
    print(f"  Hilbert dim   = 2^{geo.N} = {2**geo.N}")

    if geo.N > 26:
        raise SystemExit(
            f"\nN = {geo.N} qubits is too large for exact diag on a laptop. "
            f"Bail out — only L=2 PBC (N=24) is realistic here.")

    Ham = create_hamiltonian(
        hi=hi,
        vertex_all=geo.vertex_all,
        plaq_all=geo.plaq_all,
        bonds=geo.bonds,
        hx=args.hx, hy=args.hy, hz=args.hz,
        J=args.J, dtype=args.dtype,
    )

    # ---------------------------------------------------------------------
    # Lanczos: compute the lowest `k` eigenvalues and the ground-state vector
    # ---------------------------------------------------------------------
    print(f"\nRunning Lanczos (k={args.k}) ...")
    t0 = time.time()
    evals, evecs = nk.exact.lanczos_ed(Ham, k=args.k, compute_eigenvectors=True)
    t_lanczos = time.time() - t0
    evals = np.real(np.asarray(evals)).ravel()
    psi0  = np.asarray(evecs)[:, 0]
    print(f"  Lanczos took {t_lanczos:.2f} s")
    for j, e in enumerate(evals):
        print(f"  E_{j} = {e:.8f}")
    gap = float(evals[1] - evals[0]) if len(evals) >= 2 else None
    if gap is not None:
        print(f"  gap = E_1 - E_0 = {gap:.6f}")

    # ---------------------------------------------------------------------
    # Reference observables in the ground state
    # ---------------------------------------------------------------------
    def expect_dense(op):
        """⟨psi0| op |psi0⟩ via sparse mat-vec; works for any nk operator."""
        sp = op.to_sparse()
        return float(np.real(psi0.conj() @ (sp @ psi0)))

    print("\nComputing ground-state observables ...")
    # Per-qubit magnetisations
    sx = [expect_dense(nk.operator.spin.sigmax(hi, i, dtype=args.dtype))
          for i in range(geo.N)]
    sz = [expect_dense(nk.operator.spin.sigmaz(hi, i, dtype=args.dtype))
          for i in range(geo.N)]
    # sigma_y only meaningful for complex dtype
    if args.dtype == "complex":
        sy = [expect_dense(nk.operator.spin.sigmay(hi, i, dtype=args.dtype))
              for i in range(geo.N)]
    else:
        sy = [0.0] * geo.N

    # Stabilizer expectations
    A_v = []
    for v in geo.vertex_all:
        op = 1
        for i in v:
            if i == -1:
                continue
            op = op * nk.operator.spin.sigmax(hi, int(i), dtype=args.dtype)
        A_v.append(expect_dense(op))

    B_p = []
    for p_ in geo.plaq_all:
        op = 1
        for i in p_:
            if i == -1:
                continue
            op = op * nk.operator.spin.sigmaz(hi, int(i), dtype=args.dtype)
        B_p.append(expect_dense(op))

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    out_path = args.out or (
        f"test_exact_diag_L{args.Lx}_hx{args.hx}_hy{args.hy}_hz{args.hz}.json"
    )
    result = {
        "Lx": args.Lx, "Ly": args.Ly, "Lz": args.Lz, "bc": args.bc, "N": geo.N,
        "hx": args.hx, "hy": args.hy, "hz": args.hz, "J": args.J,
        "dtype": args.dtype,
        "N_vertices": len(geo.vertex_all),
        "N_plaquettes": len(geo.plaq_all),
        "E0": float(evals[0]),
        "E1": float(evals[1]) if len(evals) >= 2 else None,
        "gap": gap,
        "sx_per_qubit": sx,
        "sy_per_qubit": sy,
        "sz_per_qubit": sz,
        "sx_mean": float(np.mean(sx)),
        "sx_max_abs": float(np.max(np.abs(sx))),
        "sy_mean": float(np.mean(sy)),
        "sz_mean": float(np.mean(sz)),
        "A_v_per_vertex": A_v,
        "B_p_per_plaq":   B_p,
        "A_v_mean": float(np.mean(A_v)),
        "A_v_min":  float(np.min(A_v)),
        "B_p_mean": float(np.mean(B_p)),
        "B_p_min":  float(np.min(B_p)),
        "lanczos_seconds": t_lanczos,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ---------------------------------------------------------------------
    # Friendly summary
    # ---------------------------------------------------------------------
    print("\n--- Reference summary ---")
    print(f"  E_0           = {result['E0']:.6f}")
    print(f"  gap           = {result['gap']:.6f}" if gap else "  gap           = n/a")
    print(f"  <sigma_x>     mean={result['sx_mean']:+.4f}   max|.|={result['sx_max_abs']:.4f}")
    print(f"  <sigma_y>     mean={result['sy_mean']:+.4f}")
    print(f"  <sigma_z>     mean={result['sz_mean']:+.4f}")
    print(f"  <A_v>         mean={result['A_v_mean']:+.4f}   min={result['A_v_min']:+.4f}")
    print(f"  <B_p>         mean={result['B_p_mean']:+.4f}   min={result['B_p_min']:+.4f}")


if __name__ == "__main__":
    main()
