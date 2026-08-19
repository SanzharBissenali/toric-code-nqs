"""Export the 3D toric code in a field as a PMRQMC Hamiltonian file (H.txt).

PMRQMC (github.com/LevBarash/PMRQMC) takes one Pauli string per line,
``J q_1 sigma_1 q_2 sigma_2 ...`` with 1-indexed qubits and sigma in {X,Y,Z}.
Our H = -Sum_v A_v - Sum_p B_p - hx Sum_i sx_i - hz Sum_i sz_i is emitted as
L^3 X-star lines, 3L(L-1)^2 Z-plaquette lines, and one field line per edge
per nonzero field (PMRQMC segfaults on zero-coefficient lines).

The cubic OBC geometry is rebuilt here in pure python (no NetKet) so the same
code can run on Colab next to the PMRQMC binaries; ``--verify`` checks it is
isomorphic to ``ThreeD_ToricCodeGeometry`` by matching ground-state energies
at L=2 OBC (N=12) through the shared matrix-free ED.
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def cubic_obc_stabilizers(L):
    """Edges, stars and plaquettes of the open L^3 cubic lattice.

    Returns (N, stars, plaqs): stars = list of edge-index lists (one per
    vertex, 3..6 edges), plaqs = list of 4-edge lists (three orientations).
    Edge (v, d) = link from vertex v in direction d, kept only if in range.
    """
    def vid(x, y, z):
        return x + L * y + L * L * z

    edge_id = {}
    for z in range(L):
        for y in range(L):
            for x in range(L):
                if x < L - 1:
                    edge_id[(vid(x, y, z), 0)] = len(edge_id)
                if y < L - 1:
                    edge_id[(vid(x, y, z), 1)] = len(edge_id)
                if z < L - 1:
                    edge_id[(vid(x, y, z), 2)] = len(edge_id)

    steps = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    stars = []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                star = []
                for d, (dx, dy, dz) in enumerate(steps):
                    e = edge_id.get((vid(x, y, z), d))
                    if e is not None:
                        star.append(e)
                    e = edge_id.get((vid(x - dx, y - dy, z - dz), d))
                    if e is not None:
                        star.append(e)
                stars.append(star)

    plaqs = []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                v = vid(x, y, z)
                # (a,b)-plaquette at v: edges v->a, v->b, (v+a)->b, (v+b)->a
                for da, db in ((0, 1), (0, 2), (1, 2)):
                    sa, sb = steps[da], steps[db]
                    if (x, y, z)[da] < L - 1 and (x, y, z)[db] < L - 1:
                        va = vid(x + sa[0], y + sa[1], z + sa[2])
                        vb = vid(x + sb[0], y + sb[1], z + sb[2])
                        plaqs.append([edge_id[(v, da)], edge_id[(v, db)],
                                      edge_id[(va, db)], edge_id[(vb, da)]])
    return len(edge_id), stars, plaqs


def write_h_txt(path, L, hx, hz):
    """Write PMRQMC H.txt for the L^3 OBC toric code at (hx, hz)."""
    N, stars, plaqs = cubic_obc_stabilizers(L)
    lines = []
    for star in stars:
        lines.append("-1 " + " ".join(f"{e + 1} X" for e in sorted(star)))
    for plaq in plaqs:
        lines.append("-1 " + " ".join(f"{e + 1} Z" for e in sorted(plaq)))
    for h, s in ((hx, "X"), (hz, "Z")):
        if h != 0.0:
            lines.extend(f"{-h} {e + 1} {s}" for e in range(N))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return N, len(stars), len(plaqs)


class _GeomShim:
    """Duck-typed geometry for model.exact_diag.hamiltonian_linop."""

    def __init__(self, L):
        self.N, self.vertex_all, self.plaq_all = cubic_obc_stabilizers(L)


def verify():
    import numpy as np
    from scipy.sparse.linalg import eigsh
    from tc3d.exact_diag import hamiltonian_linop
    from tc3d.geometry import ThreeD_ToricCodeGeometry

    # Structural counts at L=4 (and every edge in exactly 2 stars).
    N, stars, plaqs = cubic_obc_stabilizers(4)
    assert (N, len(stars), len(plaqs)) == (144, 64, 108), (N, len(stars), len(plaqs))
    per_edge = np.zeros(N, int)
    for s in stars:
        per_edge[s] += 1
    assert (per_edge == 2).all()
    for p in plaqs:                      # commutation: even overlap with stars
        for s in stars:
            assert len(set(p) & set(s)) % 2 == 0

    # Ground-state isomorphism check at L=2 OBC (N=12, dim 4096) vs repo geometry.
    hx, hz = 0.2, 0.1
    H_mine, _ = hamiltonian_linop(_GeomShim(2), hx=hx, hz=hz)
    H_repo, _ = hamiltonian_linop(ThreeD_ToricCodeGeometry(2, 2, 2, bc="OBC"), hx=hx, hz=hz)
    e_mine = eigsh(H_mine, k=1, which="SA", return_eigenvectors=False)[0]
    e_repo = eigsh(H_repo, k=1, which="SA", return_eigenvectors=False)[0]
    assert abs(e_mine - e_repo) < 1e-9, (e_mine, e_repo)

    e0, _ = hamiltonian_linop(_GeomShim(2))       # h=0 anchor: -(#A_v + #B_p)
    assert abs(eigsh(e0, k=1, which="SA", return_eigenvectors=False)[0] - (-14.0)) < 1e-9

    print(f"verify OK: counts L=4 (144,64,108); L=2 OBC E0({hx},{hz}) = {e_mine:.10f} "
          f"(repo {e_repo:.10f}); h=0 anchor -14")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--hx", type=float, default=0.2)
    ap.add_argument("--hz", type=float, default=0.1)
    ap.add_argument("--out", default="H.txt")
    ap.add_argument("--verify", action="store_true",
                    help="run self-checks (needs the repo venv) instead of exporting")
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        N, n_v, n_p = write_h_txt(args.out, args.L, args.hx, args.hz)
        print(f"wrote {args.out}: L={args.L} OBC, N={N} spins, "
              f"{n_v} stars + {n_p} plaquettes + fields (hx={args.hx}, hz={args.hz})")
