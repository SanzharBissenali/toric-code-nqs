"""Exact S₂ anchor of the in-job Rényi locator (tc3d/renyi.py), from geometry alone.

At h_z = 0 the ground state is the toric-code stabilizer state, whose entropy is flat:
S₂(A) = (|A| − |S_A|)·ln2, with |S_A| the number of independent stabilizers supported
entirely in A. Over GF(2) in symplectic (X|Z) form, |S_A| = rank(G) − rank(G restricted to
the complement's columns). For the central-plaquette patch used by `renyi.patch_partitions`
this must equal the anchor `S2_EXACT_HZ0 = 3 ln2` in every plane, and the patch must sit
strictly inside the OBC lattice. (Ported from the removed `renyi.s2_stabilizer_exact` /
`verify_s2_geometry`, publication cleanup 2026-09-24.) Pure linear algebra — safe anywhere.
"""
import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.renyi import S2_EXACT_HZ0, patch_partitions

LN2 = np.log(2.0)


def gf2_rank(M) -> int:
    M = (np.asarray(M).astype(np.uint8) & 1).copy()
    if M.size == 0:
        return 0
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        mask = M[:, c].astype(bool).copy()
        mask[r] = False
        M[mask] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def s2_stabilizer_nats(geo, A_edges) -> float:
    N = int(geo.N)
    A = {int(e) for e in A_edges}
    gens = []
    for v in geo.get_vertex_all_hetero():            # A_v: X-type
        row = np.zeros(2 * N, np.uint8); row[np.asarray(v, int)] = 1; gens.append(row)
    for p in geo.plaq_all:                            # B_p: Z-type
        row = np.zeros(2 * N, np.uint8); row[N + np.asarray(p, int)] = 1; gens.append(row)
    G = np.array(gens, np.uint8)
    B = [q for q in range(N) if q not in A]
    n_in_A = gf2_rank(G) - gf2_rank(G[:, B + [N + q for q in B]])
    return (len(A) - n_in_A) * LN2


def test_central_plaquette_s2_is_3ln2():
    for L in (4, 5):
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        for plane, edges in patch_partitions(geo, ("xy", "xz", "yz")).items():
            coords = [np.asarray(geo.arr_coord[int(e)], float) for e in edges]
            assert len(edges) == 4, (L, plane, edges)
            assert all(0.0 < c[ax] < L - 1 for c in coords for ax in range(3)), (L, plane, "patch touches an OBC face")
            s2 = s2_stabilizer_nats(geo, edges)
            assert abs(s2 - S2_EXACT_HZ0) < 1e-12, (L, plane, s2)
    print("  central-plaquette S2 = 3 ln2 at L=4,5 in xy/xz/yz  OK")


if __name__ == "__main__":
    test_central_plaquette_s2_is_3ln2()
    print("test_renyi_exact: PASS")
