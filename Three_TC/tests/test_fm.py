"""
GF(2) index-identity tests for the Fredenhagen–Marcu builders in Three_TC/fm.py.
Geometry + index combinatorics + a logistic-fit check — NO ED, NO sampling.

Run directly:
    python test_fm.py
"""

import _path  # noqa: F401
import numpy as np
from Three_TC.model.geometry import ThreeD_ToricCodeGeometry
from Three_TC.fm import (electric_loop_edges, magnetic_membrane_edges,
                         fit_transition, _bulk_square, PLANE_NORMAL)


def _xor(sets):
    acc = set()
    for s in sets:
        acc ^= set(s)
    return acc


def _odd_overlap(string, stabilizers):
    """How many stabilizers share an odd number of edges with `string`
    (i.e. anticommute with it)."""
    return sum(len(set(string) & set(s)) % 2 == 1 for s in stabilizers)


def test_electric_closed_loop_is_product_of_enclosed_plaquettes(geo, R=2):
    """∏σ^z around the rectangle == ∏ enclosed B_p (the magnetic Wilson loop)."""
    plane_axis, plane_at, corner = 2, 0, (0, 0)
    closed, _ = electric_loop_edges(geo, plane_axis=plane_axis,
                                    plane_at=plane_at, corner=corner, R=R)
    enclosed = []
    for p, (cen, ori) in enumerate(zip(geo.plaq_centers, geo.plaq_orient)):
        if ori != plane_axis:
            continue
        x, y, z = cen
        if abs(z - plane_at) < 1e-9 and corner[0] < x < corner[0] + R \
                and corner[1] < y < corner[1] + R:
            enclosed.append(set(geo.plaq_all[p]))
    assert len(enclosed) == R * R
    assert _xor(enclosed) == set(closed)
    assert len(set(closed)) == 4 * R


def test_electric_open_string_is_half_square_with_two_charges(geo, R=2):
    """BFFM open string = HALF the square: |open| = 2R = ½·|closed| (the perimeter-
    law cancellation that gives a finite ℓ→∞ limit). It flips exactly 2 vertex
    stars (its e-charge ends); the closed loop commutes with all A_v."""
    closed, open_ = electric_loop_edges(geo, plane_axis=2, plane_at=0, R=R)
    assert len(set(open_)) == 2 * R == len(set(closed)) // 2
    verts = geo.get_vertex_all_hetero()
    assert _odd_overlap(open_, verts) == 2
    assert _odd_overlap(closed, verts) == 0


def test_magnetic_membrane_flux(geo):
    """Option A half-sheet: the full σ^x sheet is boundary-free (commutes with
    every B_p — it is ∏A_v over the slab, so =1 on the GS); the half-sheet opens a
    non-empty flux loop along the cut, and its area is L_a//2 of the full sheet."""
    closed, open_ = magnetic_membrane_edges(geo, normal=2, plane_at=0)
    plaqs = [set(p) for p in geo.plaq_all]
    assert _odd_overlap(closed, plaqs) == 0          # closed sheet: no flux
    assert _odd_overlap(open_, plaqs) > 0            # half sheet: a flux loop
    assert len(set(closed)) == geo.Lx * geo.Ly       # full xy sheet of x-edges
    assert len(set(open_)) == (geo.Lx // 2) * geo.Ly  # exactly half (a-cut)


def _enclosed_plaqs(geo, plane_axis, plane_at, corner, R):
    """B_p faces of orientation `plane_axis` at height `plane_at`, strictly inside the loop."""
    a, b = [ax for ax in range(3) if ax != plane_axis]
    out = []
    for p, (cen, ori) in enumerate(zip(geo.plaq_centers, geo.plaq_orient)):
        if ori != plane_axis or abs(cen[plane_axis] - plane_at) > 1e-9:
            continue
        if corner[0] < cen[a] < corner[0] + R and corner[1] < cen[b] < corner[1] + R:
            out.append(set(geo.plaq_all[p]))
    return out


def test_bulk_square_centered_interior(geo):
    """`_bulk_square` gives the largest fully-interior centered square in every plane:
    R=L-3, corner=(1,1), plane at L//2; the loop never touches an OBC surface face."""
    L = geo.Lx
    for c in range(3):
        kw = _bulk_square(geo, c)
        assert kw["R"] == L - 3, kw
        assert kw["corner"] == (1, 1), kw
        assert kw["plane_at"] == L // 2, kw
        closed, open_ = electric_loop_edges(geo, **kw)
        R = kw["R"]
        assert len(set(closed)) == 4 * R
        assert len(set(open_)) == 2 * R == len(set(closed)) // 2
        assert set(open_).issubset(set(closed))
        # strictly interior: every loop-edge coordinate lies in [1, L-2] (surfaces are 0, L-1)
        for q in closed:
            assert all(1 - 1e-9 <= x <= (L - 2) + 1e-9 for x in geo.arr_coord[q]), \
                (c, q, geo.arr_coord[q])


def test_bulk_loop_identities_all_planes(geo):
    """At the bulk placement, in each plane: ∏σ^z(closed)=∏ enclosed B_p, and the open
    half-string flips exactly 2 vertex stars (its e-charge ends)."""
    verts = geo.get_vertex_all_hetero()
    for c in range(3):
        kw = _bulk_square(geo, c)
        closed, open_ = electric_loop_edges(geo, **kw)
        enclosed = _enclosed_plaqs(geo, kw["plane_axis"], kw["plane_at"],
                                   kw["corner"], kw["R"])
        assert len(enclosed) == kw["R"] * kw["R"]
        assert _xor(enclosed) == set(closed)
        assert _odd_overlap(open_, verts) == 2
        assert _odd_overlap(closed, verts) == 0


def test_three_plane_isotropy(geo):
    """The xy/xz/yz bulk loops are index-set symmetric: equal edge counts and equal
    star-overlap parities across orientations (what the O_FM average relies on)."""
    verts = geo.get_vertex_all_hetero()
    sigs = set()
    for lbl in ("xy", "xz", "yz"):
        kw = _bulk_square(geo, PLANE_NORMAL[lbl])
        closed, open_ = electric_loop_edges(geo, **kw)
        sigs.add((len(set(closed)), len(set(open_)),
                  _odd_overlap(closed, verts), _odd_overlap(open_, verts)))
    assert len(sigs) == 1, sigs           # all three orientations identical by symmetry


def test_bulk_loop_raises_below_L4():
    """No bulk-centered loop exists for L<=3 (R=L-3<1) -> clear ValueError, so L=3 drops."""
    for L in (2, 3):
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        for c in range(3):
            try:
                _bulk_square(geo, c)
            except ValueError:
                pass
            else:
                assert False, f"expected ValueError at L={L}, plane_axis={c}"


def test_logistic_fit_recovers_inflection():
    """fit_transition's logistic inflection h_c recovers a known midpoint."""
    h = np.linspace(0.0, 1.0, 11)
    h0 = 0.42
    O = 1.0 / (1.0 + np.exp(-(h - h0) / 0.05))
    fit = fit_transition(h, O, Oe=0.01 * np.ones_like(h))
    assert abs(fit["h_c"] - h0) < 0.02


def main():
    for bc in ("OBC", "PBC"):
        geo = ThreeD_ToricCodeGeometry(3, 3, 3, bc=bc)
        test_electric_closed_loop_is_product_of_enclosed_plaquettes(geo)
        test_electric_open_string_is_half_square_with_two_charges(geo)
        test_magnetic_membrane_flux(geo)
        print(f"[PASS] index identities, bc={bc}")
    for L in (4, 5):                       # bulk-centered loop is defined only for L>=4
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        test_bulk_square_centered_interior(geo)
        test_bulk_loop_identities_all_planes(geo)
        test_three_plane_isotropy(geo)
        print(f"[PASS] bulk-centered 3-plane loop, L={L} OBC")
    test_bulk_loop_raises_below_L4()
    print("[PASS] bulk loop raises for L<=3")
    test_logistic_fit_recovers_inflection()
    print("[PASS] logistic fit recovers inflection")
    print("All FM tests passed.")


if __name__ == "__main__":
    main()
