"""
Edge-for-edge tests of the frozen ParaToric-convention FM families (tc3d/fm.py).
Geometry + index combinatorics only — NO ED, NO sampling, NO netket operators.

The load-bearing check: an INDEPENDENT pure-Python replication of ParaToric's C++
constructions — the stock Z-string path (lattice.cpp construct_fredenhagen_marcu_loops,
cubic/3D/z-basis branch, vertex index v = z*L*L + y*L + x) and the membrane patch's
face construction — translated vertex-pair -> coordinates -> edge midpoint ->
`fm._edge`, then asserted equal to the tc3d.fm builders. This closes the
vertex-pair <-> edge-index translation gap between the two codes: if either side's
indexing convention drifts, these tests fail before any physics is compared.

Run directly:
    python test_fm_paratoric.py
"""

import numpy as np
from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fm import (_edge, magnetic_cube_edges, paratoric_corner_rule,
                     paratoric_fm_edges, paratoric_membrane_kwargs,
                     verify_paratoric_fm_geometry)


# ---------------------------------------------------------------------------
# Independent replication of the C++ constructions (kept deliberately close to
# the C++ control flow, NOT to fm.py's — that's the point of the test).
# ---------------------------------------------------------------------------

def _vertex_coord(v, L):
    """ParaToric vertex index v = z*L*L + y*L + x -> (x, y, z)."""
    return np.array([v % L, (v // L) % L, v // (L * L)], dtype=float)


def _pairs_to_edges(geo, pairs, L):
    """Translate ParaToric vertex pairs to tc3d edge indices via the midpoint."""
    out = []
    for v1, v2 in pairs:
        mid = 0.5 * (_vertex_coord(v1, L) + _vertex_coord(v2, L))
        q = _edge(geo, mid)
        assert q != -1, f"vertex pair ({v1},{v2}) -> midpoint {mid} is not an edge"
        out.append(q)
    return out


def cpp_stock_loop_pairs(L):
    """lattice.cpp construct_fredenhagen_marcu_loops, cubic/3D/z-basis branch,
    with the caller's corner formulas (init_lattice_graph): start=(L-1)/4,
    end=3(L-1)/4, middle=(start+end)/2, plane z=(L-1)/2. Returns
    (half_pairs, full_pairs) exactly in ParaToric's construction order."""
    mx = L - 1
    sy, ey = int(mx / 4.0), int(3 * mx / 4.0)
    my = int((sy + ey) / 2.0)
    sx, ex = int(mx / 4.0), int(3 * mx / 4.0)
    mz = int(mx / 2.0)

    def V(x, y):
        return mz * L * L + y * L + x

    full = []
    prev = V(sx, my)
    for y in range(my + 1, ey + 1):          # left leg, upward
        full.append((prev, V(sx, y))); prev = V(sx, y)
    for x in range(sx + 1, ex + 1):          # top
        full.append((prev, V(x, ey))); prev = V(x, ey)
    for y in range(ey - 1, my - 1, -1):      # right leg, downward
        full.append((prev, V(ex, y))); prev = V(ex, y)
    half = list(full)                        # "Store half_loop after half of the path"
    for y in range(my - 1, sy - 1, -1):
        full.append((prev, V(ex, y))); prev = V(ex, y)
    for x in range(ex - 1, sx - 1, -1):
        full.append((prev, V(x, sy))); prev = V(x, sy)
    for y in range(sy + 1, my + 1):
        full.append((prev, V(sx, y))); prev = V(sx, y)
    return half, full


def cpp_membrane_pairs(L, R, corner):
    """The membrane patch's face construction: for each of the 6 faces the
    R+1 x R+1 `ax`-axis edges piercing it (outer vertex at corner[ax]-1 / +R+1);
    half = bottom (low-z) face + the lower `nlay` z-layers of the 4 side faces in
    the order (x,low),(x,high),(y,low),(y,high) with nlay = base + (k < rem),
    base, rem = divmod(2(R+1), 4). Returns (half_pairs, full_pairs)."""
    def V(c):
        return int(c[2]) * L * L + int(c[1]) * L + int(c[0])

    def face(ax, side, zmax=None):
        a, b = [j for j in range(3) if j != ax]
        pairs = []
        for ia in range(R + 1):
            for ib in range(R + 1):
                c_in = np.zeros(3, dtype=int)
                c_in[a], c_in[b] = corner[a] + ia, corner[b] + ib
                c_in[ax] = corner[ax] if side == "low" else corner[ax] + R
                c_out = c_in.copy()
                c_out[ax] += -1 if side == "low" else 1
                if zmax is not None and c_in[2] - corner[2] >= zmax:
                    continue
                pairs.append((V(c_out), V(c_in)))
        return pairs

    full = [p for ax in range(3) for s in ("low", "high") for p in face(ax, s)]
    base, rem = divmod(2 * (R + 1), 4)
    nlay = [base + (1 if k < rem else 0) for k in range(4)]
    half = face(2, "low")
    for (ax, s), n in zip([(0, "low"), (0, "high"), (1, "low"), (1, "high")], nlay):
        half += face(ax, s, zmax=n)
    return half, full


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_corner_rule_identity():
    """Quarter-box corner == centered corner, and the frozen R table (L=4..12)."""
    want_R = {4: 2, 5: 2, 6: 2, 7: 3, 8: 4, 9: 4, 10: 4, 11: 5, 12: 6}
    for L in range(4, 17):
        s, e, R = paratoric_corner_rule(L)
        assert (L - 1 - R) // 2 == s, f"L={L}: centered {(L-1-R)//2} != s {s}"
        assert s + e in (L - 2, L - 1), f"L={L}: s+e={s+e}"
        if L in want_R:
            assert R == want_R[L], f"L={L}: R={R} != frozen {want_R[L]}"


def test_stock_loop_matches_cpp(geo):
    """paratoric_fm_edges == the C++ vertex-pair path, edge for edge."""
    L = geo.Lx
    half, full = cpp_stock_loop_pairs(L)
    cpp_open = _pairs_to_edges(geo, half, L)
    cpp_closed = _pairs_to_edges(geo, full, L)
    closed, open_ = paratoric_fm_edges(geo)
    assert len(set(cpp_closed)) == len(cpp_closed), "C++ closed path revisits an edge"
    assert set(cpp_closed) == set(closed), f"L={L}: closed loop differs from C++"
    assert set(cpp_open) == set(open_), f"L={L}: open string differs from C++"
    s, e, R = paratoric_corner_rule(L)
    assert len(set(closed)) == 4 * R
    assert len(set(open_)) == (2 * R if R % 2 == 0 else 2 * R + 1)


def test_membrane_matches_cpp(geo, R=None):
    """magnetic_cube_edges(**paratoric_membrane_kwargs) == the patch's face
    construction (both families), edge for edge, including the half split."""
    L = geo.Lx
    kw = paratoric_membrane_kwargs(geo, R)
    half, full = cpp_membrane_pairs(L, kw["R"], kw["corner"])
    cpp_open = set(_pairs_to_edges(geo, half, L))
    cpp_closed = set(_pairs_to_edges(geo, full, L))
    closed, open_ = magnetic_cube_edges(geo, R=kw["R"], corner=kw["corner"],
                                        vertical=kw["vertical"])
    assert cpp_closed == set(closed), f"L={L} R={kw['R']}: closed membrane differs"
    assert cpp_open == set(open_), f"L={L} R={kw['R']}: open membrane differs"
    assert len(cpp_closed) == 6 * (kw["R"] + 1) ** 2
    assert len(cpp_open) == len(cpp_closed) // 2


def test_membrane_closed_is_coboundary(geo, R=None):
    """Independent brute force: the closed membrane = ALL lattice edges with
    exactly one endpoint inside the vertex cube (== supp ∏_{v∈cube} A_v)."""
    L = geo.Lx
    kw = paratoric_membrane_kwargs(geo, R)
    x0, Rc = np.array(kw["corner"]), kw["R"]
    def inside(c):
        return bool(np.all(c >= x0 - 1e-9) and np.all(c <= x0 + Rc + 1e-9)
                    and np.allclose(c, np.round(c)))
    brute = []
    for q, mid in enumerate(geo.arr_coord):
        mid = np.asarray(mid, dtype=float)
        ax = int(np.argmax(np.abs(mid - np.round(mid)) > 1e-6))
        e_ax = np.eye(3)[ax]
        if inside(mid - 0.5 * e_ax) != inside(mid + 0.5 * e_ax):
            brute.append(q)
    closed, _ = magnetic_cube_edges(geo, R=Rc, corner=kw["corner"],
                                    vertical=kw["vertical"])
    assert set(brute) == set(closed), f"L={L} R={Rc}: coboundary mismatch"


def test_constructibility_matrix():
    """Family existence per L: string L>=4; pt-cube L>=5; anchor R needs L>=R+3."""
    for L in range(4, 9):
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        paratoric_fm_edges(geo)                              # never raises for L>=4
        for R, Lmin in ((None, 5), (1, 4), (2, 5), (3, 6)):
            try:
                paratoric_membrane_kwargs(geo, R)
                ok = True
            except ValueError:
                ok = False
            assert ok == (L >= Lmin), f"L={L}, R={R}: exists={ok}, want L>={Lmin}"


def main():
    test_corner_rule_identity()
    print("[PASS] corner rule: centered identity + frozen R table, L=4..16")
    for L in range(4, 9):
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        test_stock_loop_matches_cpp(geo)
        print(f"[PASS] stock Z-string == C++ vertex-pair path, L={L}")
        rep = verify_paratoric_fm_geometry(geo)
        assert rep["ok"], f"L={L}: {rep}"
        print(f"[PASS] paratoric FM invariants (parity/halving/subset), L={L}")
        for R in (None, 1):
            try:
                paratoric_membrane_kwargs(geo, R)
            except ValueError:
                continue                                     # family absent at this L
            test_membrane_matches_cpp(geo, R)
            test_membrane_closed_is_coboundary(geo, R)
            fam = "pt-cube" if R is None else f"anchor R={R}"
            print(f"[PASS] membrane {fam} == C++ faces + coboundary, L={L}")
    test_constructibility_matrix()
    print("[PASS] constructibility matrix, L=4..8")
    print("All ParaToric-convention FM tests passed.")


if __name__ == "__main__":
    main()
