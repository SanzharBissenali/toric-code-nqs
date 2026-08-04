"""
Pure-Python geometry tests for the 3D toric code. No netket, no Hilbert space.

Run directly:
    python test_geometry.py
"""

from collections import Counter
from tc3d.geometry import ThreeD_ToricCodeGeometry


def expected_counts(L):
    """(N qubits, N vertices, N plaquettes, N bonds) for an L^3 PBC lattice."""
    V = L * L * L
    return 3 * V, V, 3 * V, 12 * V


def test_counts(geom, L):
    """Total counts match the closed-form formulas."""
    Nq, Nv, Np, Nb = expected_counts(L)
    assert geom.N             == Nq, f"N: {geom.N} != {Nq}"
    assert len(geom.vertex_all) == Nv
    assert len(geom.plaq_all)   == Np
    assert len(geom.bonds)      == Nb


def test_no_missing_lookups(geom):
    """Every stabilizer/bond index was resolved (no -1 sentinels in PBC)."""
    assert all(-1 not in v for v in geom.vertex_all)
    assert all(-1 not in p for p in geom.plaq_all)
    assert all(-1 not in b for b in geom.bonds)


def test_stabilizer_shapes(geom):
    """Vertices touch 6 distinct qubits; plaquettes touch 4 distinct qubits."""
    assert all(len(set(v)) == 6 for v in geom.vertex_all)
    assert all(len(set(p)) == 4 for p in geom.plaq_all)


def test_qubit_incidence(geom):
    """Each qubit lies in exactly 2 vertex stabilizers and 4 plaquettes."""
    cv = Counter(q for v in geom.vertex_all for q in v)
    cp = Counter(q for p in geom.plaq_all   for q in p)
    assert all(cv[q] == 2 for q in range(geom.N))
    assert all(cp[q] == 4 for q in range(geom.N))


def test_stabilizers_commute(geom):
    """
    A_v and B_p commute  iff  |support(A_v) ∩ support(B_p)| is even.
    This is the symplectic-form check; failure means the plaquette/vertex
    enumeration is geometrically inconsistent.
    """
    plaq_sets = [set(p) for p in geom.plaq_all]
    for v in geom.vertex_all:
        vset = set(v)
        for p, pset in enumerate(plaq_sets):
            overlap = len(vset & pset)
            assert overlap % 2 == 0, f"non-commuting A_v, B_{p}: |∩|={overlap}"


def test_product_all_vertices_is_identity(geom):
    """∏_v A_v = I  because each qubit lies in exactly 2 vertices (X^2=I)."""
    c = Counter(q for v in geom.vertex_all for q in v)
    assert all(c[q] % 2 == 0 for q in range(geom.N))


def test_xy_layer_plaquette_product_is_identity(geom):
    """
    3D-specific PBC dependency: the product of all xy-plaquettes at a fixed z
    is the identity (each z-edge contributes 0 times, each x/y-edge twice).
    Catches plaquette-orientation bugs that the 2D code can't see.
    """
    L = geom.Lx
    # An xy-plaquette has all 4 edges with the same integer z-coord.
    # Group plaquettes by the z-coord of their first edge; xy-plaquettes are
    # those whose 4 edges all share that z value as an integer.
    def is_xy_plaq_at_z(plaq, z):
        coords = geom.arr_coord[plaq]
        zs = coords[:, 2]
        return all(zs == z)  # all four edges live in the same z-slice

    for z in range(L):
        layer = [p for p in geom.plaq_all if is_xy_plaq_at_z(p, z)]
        assert len(layer) == L * L, f"xy-layer at z={z} has {len(layer)} plaquettes"
        c = Counter(q for p in layer for q in p)
        assert all(n % 2 == 0 for n in c.values()), \
            f"xy-layer product at z={z} is not identity: {dict(c)}"


def run_all(L=2):
    geom = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc="PBC")
    tests = [
        ("counts",                       lambda: test_counts(geom, L)),
        ("no missing lookups",           lambda: test_no_missing_lookups(geom)),
        ("stabilizer shapes",            lambda: test_stabilizer_shapes(geom)),
        ("qubit incidence",              lambda: test_qubit_incidence(geom)),
        ("A_v and B_p commute",          lambda: test_stabilizers_commute(geom)),
        ("∏A_v = I",                     lambda: test_product_all_vertices_is_identity(geom)),
        ("xy-layer ∏B_p = I",            lambda: test_xy_layer_plaquette_product_is_identity(geom)),
    ]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")


# ---------------------------------------------------------------------------
# OBC: truncated vertex stars (-1 padded) + complete-face plaquettes only.
# ---------------------------------------------------------------------------

def expected_counts_obc(L):
    """(N qubits, N plaquettes) for an L^3 OBC lattice (Lx=Ly=Lz=L).

    Edges of each orientation live on a (L-1)*L*L subgrid; complete faces of each
    orientation on a (L-1)^2*L subgrid.
    """
    return 3 * L * L * L - 3 * L * L, 3 * (L - 1) ** 2 * L


def test_obc(geom, L):
    Nq, Np = expected_counts_obc(L)
    assert geom.N == Nq, f"OBC N: {geom.N} != {Nq}"
    assert len(geom.plaq_all) == Np, f"OBC N_plaq: {len(geom.plaq_all)} != {Np}"
    assert len(geom.vertex_all) == L ** 3, "one (possibly truncated) star per vertex"
    # plaquettes are complete 4-edge faces with no -1 padding
    assert all(-1 not in p for p in geom.plaq_all), "OBC plaq_all must have no -1"
    assert all(len(set(p)) == 4 for p in geom.plaq_all)
    # centres parallel to plaq_all and round-trip through the index map
    assert len(geom.plaq_centers) == len(geom.plaq_all) == len(geom.plaq_orient)
    assert all(geom._plaq_center_to_idx(geom.plaq_centers[i]) == i
               for i in range(len(geom.plaq_all)))
    # commutation still holds (truncated stars vs complete faces)
    test_stabilizers_commute(geom)
    # each edge sits in exactly 2 vertex stars (∏A_v = I)
    test_product_all_vertices_is_identity(geom)


def run_all_obc(L=2):
    geom = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc="OBC")
    test_obc(geom, L)
    print(f"  ok  OBC counts/commute/centres (N={geom.N}, "
          f"N_plaq={len(geom.plaq_all)})")


if __name__ == "__main__":
    for L in (2, 3):
        print(f"L = {L} PBC")
        run_all(L)
    for L in (2, 3):
        print(f"L = {L} OBC")
        run_all_obc(L)
    print("All geometry tests passed.")
