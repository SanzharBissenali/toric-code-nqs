"""
GF(2) index-identity tests for the Fredenhagen–Marcu builders in Three_TC/fm.py.
Geometry + index combinatorics + a logistic-fit check — NO ED, NO sampling.

Run directly:
    python test_fm.py
"""

import _path  # noqa: F401
import numpy as np
from Three_TC.model.geometry import ThreeD_ToricCodeGeometry
from Three_TC.fm import (electric_loop_edges, fit_transition, _bulk_square,
                         PLANE_NORMAL, _bulk_cube, magnetic_cube_edges,
                         verify_membrane_geometry, verify_membrane_charge_flux)


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


def test_membrane_cube_closed_is_product_of_enclosed_vertex_stars(geo):
    """CLOSED cube surface == ∏_{v∈cube} A_v (the σ^x dual of the electric ∏B_p loop).

    Edges piercing the surface = edges with exactly one endpoint among the (R+1)^3
    interior cube vertices; interior-interior edges appear twice and cancel. Hence
    the closed membrane is a product of vertex stars ⇒ commutes with every B_p ⇒
    ⟨M_closed⟩=1 on the ground state (verified via the B_p parity below)."""
    kw = _bulk_cube(geo)
    R, x0 = kw["R"], kw["corner"]
    closed, _ = magnetic_cube_edges(geo, R=R, corner=x0, vertical=2)
    verts = geo.get_vertex_all_hetero()
    pos = geo.dg_v.positions                        # parallel to vertex_all
    inside = [k for k in range(len(verts))
              if all(x0[a] <= pos[k][a] <= x0[a] + R for a in range(3))]
    assert len(inside) == (R + 1) ** 3
    acc = _xor(verts[k] for k in inside)            # ∏ A_v over the cube
    assert acc == set(closed)
    assert len(set(closed)) == 6 * (R + 1) ** 2
    assert _odd_overlap(closed, [set(p) for p in geo.plaq_all]) == 0   # ⟨closed⟩=1


def test_membrane_cube_open_half_with_flux_loop(geo):
    """OPEN bucket = exactly HALF the cube surface (|open|=|closed|/2, the area-law
    cancellation), bounded by a non-empty, even (closed-loop) flux boundary of B_p ⇒
    ⟨M_open⟩=0 on the GS ⇒ O_FM^m(h_x=0)=0. The flux count is 4(R+1) for a planar
    equator (R=1,3) and steps by a layer for odd R+1 (R=2) — it is checked to be a
    nonzero even bulk loop, not pinned to a fixed value."""
    kw = _bulk_cube(geo)
    closed, open_ = magnetic_cube_edges(geo, R=kw["R"], corner=kw["corner"], vertical=2)
    assert len(set(open_)) == len(set(closed)) // 2 == 3 * (kw["R"] + 1) ** 2
    flux = _odd_overlap(open_, [set(p) for p in geo.plaq_all])
    assert flux > 0 and flux % 2 == 0
    assert _odd_overlap(closed, [set(p) for p in geo.plaq_all]) == 0


def test_membrane_self_checks_pass_and_bulk_safe(geo):
    """verify_membrane_{geometry,charge_flux} agree with the exact-limit contract, and
    all membrane edges + the flux loop live strictly in the OBC bulk (the C3 anti-leak
    guard) — identically across the 3 'up' orientations (isotropy)."""
    g = verify_membrane_geometry(geo)
    f = verify_membrane_charge_flux(geo)
    assert g["ok"] and f["ok"]
    assert f["OFM_hx0_topological"] == 0.0 and f["OFM_hxinf_trivial"] == 1.0
    sig = {(v["n_closed"], v["n_open"]) for v in g["per_vertical"].values()}
    assert len(sig) == 1                            # 3 orientations congruent by symmetry
    assert all(v["flux_loop_bulk"] for v in f["per_vertical"].values())


def test_membrane_excludes_L4_at_half_aspect():
    """Aspect-½ excludes L=4 (⌊4/2⌋=2 > L-3=1) with a clear ValueError — it is *not*
    silently downgraded to R=1 (a different aspect), the labeling trap Phase A flags."""
    geo = ThreeD_ToricCodeGeometry(4, 4, 4, bc="OBC")
    try:
        _bulk_cube(geo)                             # R=None -> aspect-½ default
    except ValueError:
        pass
    else:
        assert False, "expected ValueError excluding L=4 at aspect-½"
    assert _bulk_cube(geo, R=1)["R"] == 1           # explicit small R still allowed


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
        print(f"[PASS] electric index identities, bc={bc}")
    for L in (4, 5):                       # bulk-centered loop is defined only for L>=4
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        test_bulk_square_centered_interior(geo)
        test_bulk_loop_identities_all_planes(geo)
        test_three_plane_isotropy(geo)
        print(f"[PASS] bulk-centered 3-plane loop, L={L} OBC")
    test_bulk_loop_raises_below_L4()
    print("[PASS] bulk loop raises for L<=3")
    for L in (5, 6, 7):                    # cube membrane at aspect-½ (L=4 excluded)
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        test_membrane_cube_closed_is_product_of_enclosed_vertex_stars(geo)
        test_membrane_cube_open_half_with_flux_loop(geo)
        test_membrane_self_checks_pass_and_bulk_safe(geo)
        print(f"[PASS] cube membrane, L={L} OBC")
    test_membrane_excludes_L4_at_half_aspect()
    print("[PASS] membrane excludes L=4 at aspect-½")
    test_logistic_fit_recovers_inflection()
    print("[PASS] logistic fit recovers inflection")
    print("All FM tests passed.")


if __name__ == "__main__":
    main()
