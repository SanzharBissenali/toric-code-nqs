"""
tc3d/fm.py
─────────────────────────────────────────────────────────────────────────────
Fredenhagen–Marcu (BFFM) phase-transition detection from *trained* NQS
checkpoints for the 3D toric code.

Pipeline (one fixed L at a time; stack over L afterwards for FSS):

    checkpoints {name}.mpack + {name}.json   (one per (L, hx, hz))
       │  load_vstate : build_state(config) + flax.from_bytes(mpack)
       ▼
    fm_sweep(dir, sector, L, hx, hy=0.0, field="hz")  → table  field, O_FM ± err, ⟨σz⟩
       │  per checkpoint: build the loop/membrane operators, fm_ratio(vs, …)
       ▼
    fit_transition(field, O, Oe)  → h_c  (logistic inflection = derivative peak),
                                    with a finite-difference derivative cross-check
       ▼
    plot_fm_sweep(...)            (matplotlib, optional)

Two sectors, ONE shared consumer (the 3D e/m duality is not symmetric):
  • electric (hz sweep): σ^z **loop/string** in a lattice plane — the 2D BFFM
    embedded in 3D. Diagonal ⇒ cheap, low MC variance.
  • magnetic (hx sweep): σ^x **membrane** (σ^x on the axis-edges piercing an
    R×R patch; its boundary is the flux loop). Off-diagonal ⇒ noisier.

Only the index-set builder differs; `fm_ratio`, the loader, the sweep and the
analysis are shared.

Fermionic model (electric only): the same loop/string geometry, but the bare σ^z
string anticommutes with the decorated B̃_p, so each operator is GF(2)-dressed
into a Z(z)·X(x) string (`dressed_electric_edges`) before entering `fm_ratio`.

Never run 3D ED/sweeps locally (see CLAUDE.md). This module is for Colab,
where the trained checkpoints live; `_validate.py`-style index checks are the
only thing meant to run on the dev box.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import netket as nk
import flax

from tc3d.builders import build_state
from tc3d.fermionic_decoration import dressed_string, fermionic_plaquettes

VSCORE_MAX = 1.0    # skip finished runs whose Vscore exceeds this: a variance blow-up
                    # the in-run guard missed (diverged:false but garbage state). Matches
                    # analysis/scripts/check_convergence.py's BAD-VSCORE gate.


# =============================================================================
# Geometry → edge index sets (the only thing that differs between sectors)
# =============================================================================

def _edge(geo, coord) -> int:
    """Qubit index of the edge whose midpoint is `coord` (PBC-wrapped). -1 if absent."""
    c = np.asarray(coord, dtype=float)
    if geo.bc == "PBC":
        c = c % np.array([geo.Lx, geo.Ly, geo.Lz], dtype=float)
    return geo._mapping3Dto1D(c)


def _in_plane_axes(plane_axis: int) -> Tuple[int, int]:
    a, b = [ax for ax in range(3) if ax != plane_axis]
    return a, b


# Plane label -> normal axis (an xy-plane has normal z=2, etc.) and its inverse.
PLANE_NORMAL = {"xy": 2, "xz": 1, "yz": 0}
NORMAL_PLANE = {v: k for k, v in PLANE_NORMAL.items()}


def _bulk_square(geo, plane_axis: int, plane_at: Optional[int] = None,
                 R: Optional[int] = None) -> Dict[str, Any]:
    """Kwargs for a centered σ^z square that fits *entirely in the bulk* of `plane_axis`.

    Centered in all three directions: ``corner`` centers the side-``R`` square per
    in-plane axis and the plane sits at the middle layer ``L//2`` (overridable via
    `plane_at`). Feeds straight into `electric_loop_edges(**_bulk_square(...))`.

    `R` controls the side:
      • ``R=None`` (default) → the **largest** bulk square, ``R = min(L_a,L_b) - 3`` (so
        vertices span the interior ``1 .. L-2`` and never touch the OBC surface). This
        *grows with L*, so each L evaluates a different-perimeter operator.
      • ``R=<int>`` → a **fixed** side at every L (e.g. ``R=1`` = one plaquette,
        perimeter 4) — the same physical loop across sizes, still centered/bulk.
    Requires ``1 <= R <= min(L_a,L_b) - 3`` (so L>=4; L<=3 has no bulk loop).
    """
    a, b = _in_plane_axes(plane_axis)
    L = (geo.Lx, geo.Ly, geo.Lz)
    Rmax = min(L[a], L[b]) - 3          # largest side that stays strictly in the bulk
    if R is None:
        R = Rmax
    if R < 1 or Rmax < 1:
        raise ValueError(
            f"bulk-centered FM loop needs L>=4 (R=min(L_a,L_b)-3); got in-plane "
            f"extents ({L[a]},{L[b]}) -> Rmax={Rmax}. Use placement='boundary' for small L.")
    if R > Rmax:
        raise ValueError(
            f"fixed FM loop R={R} leaves the bulk: needs R<=min(L_a,L_b)-3={Rmax} for "
            f"in-plane extents ({L[a]},{L[b]}). Shrink R or grow L.")
    corner = ((L[a] - 1 - R) // 2, (L[b] - 1 - R) // 2)   # centered; (1,1) for R=L-3 cubic
    pa = L[plane_axis] // 2 if plane_at is None else plane_at
    return dict(plane_axis=plane_axis, plane_at=pa, corner=corner, R=R)


def electric_loop_edges(geo, *, plane_axis: int = 2, plane_at: int = 0,
                        corner: Tuple[int, int] = (0, 0),
                        R: Optional[int] = None) -> Tuple[List[int], List[int]]:
    """Edges of an electric (σ^z) Wilson rectangle in a lattice plane.

    Returns ``(closed, open_)``:
      • ``closed`` — the 4 sides of an R×R rectangle (4R edges). Product of σ^z
        over it equals ∏ of the enclosed B_p (a contractible magnetic Wilson
        loop); =1 in the pure ground state, perimeter-law decaying with field.
      • ``open_`` — the BFFM **half-square** (2R edges): the lower U running from
        the midpoint of the left side, down and across the bottom, up to the
        midpoint of the right side. Its two ends carry the e-charges (separated
        by R). Because its length 2R is exactly half the 4R perimeter, the open
        string's perimeter law cancels √⟨closed⟩, so O_FM = ⟨open⟩/√|⟨closed⟩|
        has a finite ℓ→∞ limit (take R as large as the lattice allows / extrapolate).

    `plane_axis` is the rectangle's normal (2 = z-plane by default); `plane_at`
    is the integer coordinate of that plane; `corner` is the (a,b) base vertex.
    `R` defaults to the **largest square the box holds** (min in-plane extent − 1)
    — the ℓ→∞ order parameter is taken as this biggest available loop (we do not
    R-sweep). For odd R the U is split floor/ceil (ends differ by one row, as in
    the 2D `half_length_wilson`).
    """
    a, b = _in_plane_axes(plane_axis)
    if R is None:
        ext = (geo.Lx, geo.Ly, geo.Lz)
        R = min(ext[a], ext[b]) - 1          # biggest loop the lattice allows
    e = np.eye(3)
    x0, y0 = corner

    def vbase(ia, ib):
        v = np.zeros(3)
        v[a], v[b], v[plane_axis] = ia, ib, plane_at
        return v

    def edge(ia, ib, axis):
        return _edge(geo, vbase(ia, ib) + 0.5 * e[axis])

    bottom = [edge(x0 + i, y0,     a) for i in range(R)]
    top    = [edge(x0 + i, y0 + R, a) for i in range(R)]
    left   = [edge(x0,     y0 + j, b) for j in range(R)]
    right  = [edge(x0 + R, y0 + j, b) for j in range(R)]
    closed = bottom + top + left + right             # full square, 4R edges

    # BFFM open string = HALF the square (2R edges): lower-left half + bottom +
    # lower-right half. hL + hR = R, so |open| = R + R = 2R exactly.
    hL, hR = R // 2, R - R // 2
    left_low  = [edge(x0,     y0 + j, b) for j in range(hL)]
    right_low = [edge(x0 + R, y0 + j, b) for j in range(hR)]
    open_ = left_low + bottom + right_low
    if -1 in closed:
        raise ValueError("electric loop runs off the lattice — shrink R/corner "
                         "or move plane_at into the bulk")
    return closed, open_


def paratoric_corner_rule(L: int) -> Tuple[int, int, int]:
    """ParaToric's quarter-box corners: ``(s, e, R)`` with s=(L-1)//4, e=3(L-1)//4.

    Single source of truth for BOTH sectors of the ParaToric-convention FM family
    (lattice.cpp init_lattice_graph's start/end formulas): the Z-string square
    spans [s,e]² in the plane z=(L-1)//2; the X-membrane cube spans [s,e]³.
    R = e-s grows ≈(L-1)/2 — 2,2,2,3,4,4,4,5,6 for L=4..12, aspect → ½ — the
    *growing* family whose ℓ→∞ limit is the genuine order parameter (a fixed-R
    family would converge to a smooth finite-R correlator instead). Identity used
    by both codes: s+e ∈ {L-2, L-1}, hence (L-1-R)//2 == s — the quarter-box
    corner IS the centered corner.
    """
    s, e = (L - 1) // 4, (3 * (L - 1)) // 4
    return s, e, e - s


def paratoric_fm_edges(geo) -> Tuple[List[int], List[int]]:
    """(closed, open_) σ^z edge sets matching ParaToric's hard-coded cubic FM loops
    EDGE-FOR-EDGE, for the NQS-vs-QMC Fredenhagen-Marcu comparison (z-basis --fm runs).

    ParaToric (lattice.cpp construct_fredenhagen_marcu_loops, cubic/3D/z-basis):
    square loop in the plane z = (L-1)//2 with corners x,y ∈ [s, e] from
    `paratoric_corner_rule` (R = e-s: 2 for L=4..6, 3 for L=7); the open
    string is the UPPER half-U from (s, m) over the top to (e, m), m = (s+e)//2.
    Note two deviations from `electric_loop_edges`' conventions: the rectangle is
    NOT centered (corner touches the boundary at L=4 — accepted by convention),
    and for odd R the upper U has 2*(e-m) + R = 2R+1 edges — deliberately more
    than half the perimeter, reproducing ParaToric's convention rather than the
    BFFM exact-half rule (a smooth O(1) factor, harmless for the transition)."""
    L = geo.Lx
    if not (geo.Lx == geo.Ly == geo.Lz):
        raise ValueError("ParaToric FM comparison assumes a cubic box")
    s, e, R = paratoric_corner_rule(L)
    m, z0 = (s + e) // 2, (L - 1) // 2
    if R < 2:
        raise ValueError("ParaToric FM loop degenerates below L=4")
    closed, _ = electric_loop_edges(geo, plane_axis=2, plane_at=z0,
                                    corner=(s, s), R=R)
    ey = np.eye(3)

    def edge(ix, iy, axis):
        return _edge(geo, np.array([ix, iy, z0], float) + 0.5 * ey[axis])

    open_ = ([edge(s, m + j, 1) for j in range(e - m)]       # left leg, upward
             + [edge(s + i, e, 0) for i in range(R)]         # top
             + [edge(e, m + j, 1) for j in range(e - m)])    # right leg
    if -1 in open_:
        raise ValueError("ParaToric FM open string ran off the lattice")
    return closed, open_


def paratoric_membrane_kwargs(geo, R: Optional[int] = None) -> Dict[str, Any]:
    """kwargs for `magnetic_cube_edges` in the ParaToric-convention membrane families.

    ``R=None`` → the **growing corner-rule family**: cube vertices span [s,e]³ with
    the SAME corners as the Z-string (`paratoric_corner_rule`), vertical=z, single
    orientation — matches the ParaToric membrane patch edge-for-edge. Needs L>=5:
    at L=4 s=0 and the side-2 cube's coboundary is truncated by the OBC surface
    (27 edges — odd, so the exact-half open membrane the FM ratio needs does not
    exist).
    ``R=<int>`` → the **fixed-size anchor family** (R=1 in the campaign): centered
    cube, corner=((L-1-R)//2,)³, needs 1<=R<=L-3 (R=1 exists from L=4 up). For the
    corner-rule R the two corner formulas provably coincide (see
    `paratoric_corner_rule`), so both families share one construction.
    """
    L = geo.Lx
    if not (geo.Lx == geo.Ly == geo.Lz):
        raise ValueError("ParaToric FM comparison assumes a cubic box")
    if R is None:
        s, _e, R = paratoric_corner_rule(L)
        if s < 1:
            raise ValueError(
                f"ParaToric corner-rule membrane needs L>=5: at L={L} the corner s=0 "
                f"puts the cube on the OBC surface and its coboundary is truncated "
                f"(odd edge count — no exact-half open membrane). Use the R=1 anchor "
                f"family at L=4.")
        corner = (s, s, s)
    else:
        R = int(R)
        if not (1 <= R <= L - 3):
            raise ValueError(
                f"anchor membrane R={R} needs 1<=R<=L-3 for L={L} (R=1 needs L>=4)")
        corner = tuple((L - 1 - R) // 2 for _ in range(3))
    return dict(R=int(R), corner=corner, vertical=2)


def dressed_electric_edges(geo, **kw) -> Tuple[Tuple[List[int], List[int], List[int]],
                                               Tuple[List[int], List[int], List[int]]]:
    """Fermionic electric FM edge sets: ``(closed, open_)``, each a
    ``(z_edges, x_edges, flux_plaqs)`` triple (the `dressed_string` return).

    The σ^z sets are the *bosonic* ones from ``electric_loop_edges(geo, **kw)``
    (same kwargs, same geometry); each is then GF(2)-dressed with a σ^x support
    via `fermionic_decoration.dressed_string` so Z(z)·X(x) commutes with every
    decorated plaquette B̃_p (a bare σ^z string anticommutes with the B̃_p it
    crosses, so its expectation is identically 0 in the fermionic model):
      • ``closed`` → ``flux_plaqs == []`` — a conserved dressed Wilson loop
        (enforced; a fluxed "closed" loop would make the FM normalisation ⟨W⟩≡0);
      • ``open_``  → ``flux_plaqs`` = one endpoint-localized 3-plaquette flux
        cluster per endpoint (6 total, size-independent). That residual is
        unavoidable — the fermion is a charge+flux composite, so a flux-free
        open string cannot exist (expected, not a bug; see
        `fermionic_decoration._localize_open_flux`).
    The dressing is disjoint from the z-line (`dressed_string` raises on overlap).
    PBC only: `fermionic_plaquettes` wraps coordinates unconditionally.
    """
    stabs = fermionic_plaquettes(geo)
    closed_z, open_z = electric_loop_edges(geo, **kw)
    closed = dressed_string(geo, stabs, closed_z)
    open_ = dressed_string(geo, stabs, open_z)
    if closed[2]:
        raise ValueError(f"dressed closed loop is not conserved (flux plaquettes "
                         f"{closed[2]}) — cannot serve as the FM normalisation")
    return closed, open_


def magnetic_membrane_edges(geo, *, normal: int = 2, plane_at: int = 0,
                            cut_at: Optional[int] = None
                            ) -> Tuple[List[int], List[int]]:
    """Edges of a magnetic (σ^x) membrane normal to axis `normal` — the BFFM dual
    of the electric half-square (Option A).

    σ^x acts on the **`normal`-axis edges** at height ``plane_at+½``. Returns
    ``(closed, open_)``:
      • ``closed`` — the **full** σ^x sheet spanning the box. On OBC it equals
        ∏ A_v over the slab beneath it, so it is boundary-free (commutes with
        every B_p) and ``⟨closed⟩ = 1`` in the pure ground state — the exact dual
        of the electric ``∏B_p`` closed loop, hence the FM normalisation.
      • ``open_`` — **half** that sheet (the columns with in-plane a-coord < cut).
        Its only bulk boundary is the straight cut at ``a = cut`` (length L_b):
        that cut is the **flux loop** the open membrane creates. Because its area
        is ½ the closed sheet, the area laws cancel and O_FM^m = ⟨open⟩/√|⟨closed⟩|
        has a finite ℓ→∞ limit (largest membrane the box holds).

    `cut_at` defaults to L_a // 2 (cut through the middle).
    """
    a, b = _in_plane_axes(normal)
    L = (geo.Lx, geo.Ly, geo.Lz)
    ha = L[a] // 2 if cut_at is None else cut_at

    def xedge(ia, ib):
        coord = np.zeros(3)
        coord[a], coord[b], coord[normal] = ia, ib, plane_at + 0.5
        return _edge(geo, coord)

    closed = [xedge(ia, ib) for ia in range(L[a]) for ib in range(L[b])]
    open_ = [xedge(ia, ib) for ia in range(ha) for ib in range(L[b])]
    if -1 in closed or -1 in open_:
        raise ValueError("magnetic membrane runs off the lattice — check "
                         "normal/plane_at (need plane_at in 0..L-2 for OBC)")
    return closed, open_


# -----------------------------------------------------------------------------
# Cube-surface 't Hooft membrane (Option B) — the production magnetic operator.
#
# The flat sheet above (Option A) touches the OBC surface and its open cut is a
# straight line terminating on the boundary, not a closed bulk loop. The cube
# membrane fixes both: a genuine closed surface in the strict bulk, dual to the
# electric half-square (open string ↔ half-cube; e-charge ends ↔ flux loop).
# -----------------------------------------------------------------------------

def _bulk_cube(geo, R: Optional[int] = None) -> Dict[str, Any]:
    """Kwargs for a centered R-unit cube whose vertices all sit in the OBC bulk.

    Vertices span ``corner[ax] .. corner[ax]+R`` per axis (``(R+1)^3`` vertices);
    the bulk constraint ``1 <= vertex <= L-2`` gives ``R <= L-3`` (so L>=4), with
    ``corner = ((L-1-R)//2, ...)`` centering it. ``R=None`` picks the campaign default
    ``R = ⌊L/2⌋`` (aspect ≈ ½). This is *not* capped to the bulk: aspect-½ gives
    L=5→2, L=6→3, L=7→3 (all bulk), while L=4 wants R=2 > L-3=1 and is **excluded**
    (raises) rather than silently downgraded to a different aspect. Feeds
    ``magnetic_cube_edges(**_bulk_cube(...))``.
    """
    L = (geo.Lx, geo.Ly, geo.Lz)
    Lm = min(L)
    Rmax = Lm - 3
    if R is None:
        R = Lm // 2                       # aspect-½; exclude (not cap) if it leaves the bulk
    if Rmax < 1:
        raise ValueError(
            f"bulk cube membrane needs L>=4 (R<=min(L)-3); got L={L} -> Rmax={Rmax}.")
    if R < 1 or R > Rmax:
        raise ValueError(
            f"cube membrane R={R} leaves the bulk: need 1<=R<=min(L)-3={Rmax} for L={L} "
            f"(aspect-½ excludes L=4).")
    corner = tuple((L[ax] - 1 - R) // 2 for ax in range(3))
    return dict(R=int(R), corner=corner)


def _cube_face_edges(geo, corner, R: int, ax: int, side: str,
                     vlimit: Optional[Tuple[int, int]] = None) -> List[int]:
    """The `ax`-axis edges piercing one face of the cube (`side` in {'low','high'}).

    A face normal to `ax` is pierced by the `ax`-axis edges just outside it: midpoint
    at ``corner[ax]-½`` (low) or ``corner[ax]+R+½`` (high). The two in-face axes run
    over the vertex grid ``corner[c] .. corner[c]+R``. `vlimit=(c_axis, n)` restricts
    the in-face axis ``c_axis`` to its bottom ``n`` layers (used to take the lower half
    of a side face); ``None`` = the full ``(R+1)^2`` face.
    """
    a, b = [j for j in range(3) if j != ax]
    x0 = corner
    mid = x0[ax] - 0.5 if side == "low" else x0[ax] + R + 0.5
    rng = {a: range(R + 1), b: range(R + 1)}
    if vlimit is not None:
        c_axis, n = vlimit
        rng[c_axis] = range(n)
    edges = []
    for ia in rng[a]:
        for ib in rng[b]:
            c = np.zeros(3)
            c[ax] = mid
            c[a] = x0[a] + ia
            c[b] = x0[b] + ib
            edges.append(_edge(geo, c))
    return edges


def _side_layers(R: int) -> List[int]:
    """Vertical-layer counts for the 4 side faces of the open membrane.

    The open surface is the bottom face + lower halves of the 4 side faces; exact
    area-halving needs the side faces to contribute ``2(R+1)`` vertical layers in
    total (bottom face already gives ``(R+1)^2 = ½·6(R+1)^2 - 2(R+1)^2``). We split
    that evenly, distributing the remainder for odd ``R+1`` — the membrane analogue
    of the electric open string's floor/ceil ``hL,hR`` split (`electric_loop_edges`).
    R=3 → [2,2,2,2] (planar equator); R=2 → [2,2,1,1] (stepped by one layer, exactly
    as the electric U's two ends differ by a row).
    """
    total = 2 * (R + 1)
    base, rem = divmod(total, 4)
    return [base + (1 if i < rem else 0) for i in range(4)]


def cube_membrane_faces(geo, *, R: int, corner: Tuple[int, int, int],
                        vertical: int = 2) -> Tuple[List[List[int]], List[List[int]]]:
    """Face decomposition of the cube membrane: ``(closed_faces, open_faces)``.

    ``closed_faces`` = the 6 cube faces (low/high × 3 axes), each a list of the edges
    piercing it. ``open_faces`` = the bottom face + the 4 side-face lower halves
    (5 pieces). Flattened these give `magnetic_cube_edges`; kept split so the
    telescoping estimator can grow the membrane face-by-face and monitor per-face
    amplitude-ratio health (B3).
    """
    x0 = tuple(int(c) for c in corner)
    horiz = [h for h in range(3) if h != vertical]
    closed_faces = [_cube_face_edges(geo, x0, R, ax, s)
                    for ax in range(3) for s in ("low", "high")]
    open_faces = [_cube_face_edges(geo, x0, R, vertical, "low")]
    side_faces = [(h, s) for h in horiz for s in ("low", "high")]
    for (h, s), nlay in zip(side_faces, _side_layers(R)):
        open_faces.append(_cube_face_edges(geo, x0, R, h, s, vlimit=(vertical, nlay)))
    return closed_faces, open_faces


def magnetic_cube_edges(geo, *, R: int, corner: Tuple[int, int, int],
                        vertical: int = 2) -> Tuple[List[int], List[int]]:
    """Edges of the cube-surface 't Hooft membrane. Returns ``(closed, open_)``.

    ``closed`` = σ^x on every edge with **exactly one endpoint** among the cube's
    ``(R+1)^3`` interior vertices — i.e. the edges piercing the 6 cube faces. By the
    toric-code identity this product equals ``∏_{v∈cube} A_v``, so it commutes with
    every B_p and ``⟨closed⟩ = 1`` on the pure ground state (the magnetic dual of the
    electric ``∏B_p`` closed loop).

    ``open_`` = the bottom face (low-`vertical`) + the lower halves of the 4 side
    faces (`_side_layers`), so ``|open_| = |closed|//2`` exactly. Its only bulk
    boundary is the equatorial flux loop — the loop whose condensation the ratio
    detects. Area laws cancel ⇒ O_FM^m = ⟨open⟩/√|⟨closed⟩| has a finite ℓ→∞ limit.
    `vertical` selects which axis is "up" (for isotropy averaging over the 3
    orientations, mirroring the electric xy/xz/yz average).
    """
    closed_faces, open_faces = cube_membrane_faces(geo, R=R, corner=corner,
                                                   vertical=vertical)
    closed = [e for f in closed_faces for e in f]
    open_ = [e for f in open_faces for e in f]

    if -1 in closed or -1 in open_:
        raise ValueError("cube membrane runs off the lattice — need a bulk cube "
                         "(1<=R<=min(L)-3); use _bulk_cube(geo, R).")
    n_closed, n_open = len(set(closed)), len(set(open_))
    if n_closed % 2 or n_open != n_closed // 2:      # exact halving — fail loudly
        raise ValueError(f"cube membrane exact-halving failed for R={R}: "
                         f"|open|={n_open}, |closed|={n_closed} (want |open|=|closed|/2).")
    return closed, open_


# =============================================================================
# Geometry self-checks (cheap, NetKet-free proxies — see CLAUDE.md)
# =============================================================================

def _aspect_sizes(geo, plane_axis: int, aspect: float
                  ) -> Tuple[List[int], List[int], int]:
    """Loop sides for a fixed aspect ratio R/L≈`aspect` in a plane, split floor/ceil.

    Returns (keep, dropped, Rmax): the bulk-fitting sides (``1 <= R <= L-3``) to average,
    the out-of-bulk ones dropped, and Rmax. For odd L the floor/ceil pair straddles L·aspect
    (averaging their FM ratios symmetrises the parity wobble to an *effective* R/L=aspect);
    for even/exact L the pair collapses to one size. NB at aspect=0.5 the L=5 ceil (R=3)
    exceeds Rmax=2 and is dropped, so L=5 falls back to floor-only (R=2, aspect 0.4).
    """
    a, b = _in_plane_axes(plane_axis)
    L = (geo.Lx, geo.Ly, geo.Lz)
    Lm = min(L[a], L[b])
    Rmax = Lm - 3
    cand = sorted({int(np.floor(Lm * aspect)), int(np.ceil(Lm * aspect))})
    keep = [R for R in cand if 1 <= R <= Rmax]
    dropped = [R for R in cand if R < 1 or R > Rmax]
    return keep, dropped, Rmax


def verify_fm_geometry(geo, R, *, plane_axis: int = 2,
                       plane_at: Optional[int] = None) -> Dict[str, Any]:
    """Check the FM-loop invariants for a side-R bulk square (edge sets only, no operators).

    The FM ratio's perimeter-law cancellation *requires* the open string be exactly half
    the closed loop, so this reports rather than fixes. Returns facts + an ``ok`` flag:
      - ``half_ok``  — closed perimeter even and ``len(open) == len(closed)//2`` (= 2R),
      - ``open_subset_closed`` — every open edge lies on the loop (open is a sub-path of
        the closed square, so its endpoints sit on the loop),
      - ``vertices_interior`` — all loop vertices strictly inside the OBC box (each coord
        in ``[1, L-2]``, never on the surface at 0 or L-1).
    """
    L = (geo.Lx, geo.Ly, geo.Lz)
    kw = _bulk_square(geo, plane_axis, plane_at=plane_at, R=R)
    closed, open_ = electric_loop_edges(geo, **kw)
    a, b = _in_plane_axes(plane_axis)
    x0, y0 = kw["corner"]; pa = kw["plane_at"]
    coords = [(x0, a), (x0 + R, a), (y0, b), (y0 + R, b), (pa, plane_axis)]
    interior = all(1 <= c <= L[ax] - 2 for c, ax in coords)
    half = (len(closed) % 2 == 0) and (len(open_) == len(closed) // 2)
    subset = set(open_).issubset(set(closed))
    out = {"R": int(R), "plane_at": int(pa), "corner": (int(x0), int(y0)),
           "n_closed": len(closed), "n_open": len(open_),
           "aspect": R / min(L[a], L[b]),
           "half_ok": bool(half), "open_subset_closed": bool(subset),
           "vertices_interior": bool(interior)}
    out["ok"] = bool(half and subset and interior)
    return out


def verify_fm_charge_flux(geo, R, *, plane_axis: int = 2,
                          plane_at: Optional[int] = None) -> Dict[str, Any]:
    """Operator-algebra check of the exactly-solvable FM limits — no ED, just edge parities.

    A σ^z string commutes with a σ^x vertex operator A_v iff they overlap on an EVEN number
    of edges. On the toric-code ground state (all A_v=+1):
      - CLOSED loop overlaps every A_v evenly → commutes → ``⟨closed⟩ = +1``;
      - OPEN string overlaps A_v oddly at EXACTLY its 2 endpoints → creates 2 e-charges →
        maps the GS to an orthogonal state → ``⟨open⟩ = 0`` → **O_FM(hz=0) = 0**.
    (Both are products of σ^z, so both commute with every B_p — charge, no flux: the bosonic
    e-particle.) On the z-polarised product state (hz→∞) every σ^z=+1 → ``⟨open⟩=⟨closed⟩=1``
    → **O_FM(hz→∞) = 1**. Note this is the opposite of a "topological order parameter": the
    FM ratio marks the *trivial* (condensed) phase. Returns the parity counts + pass flag.
    """
    kw = _bulk_square(geo, plane_axis, plane_at=plane_at, R=R)
    closed, open_ = electric_loop_edges(geo, **kw)
    cset, oset = set(closed), set(open_)
    verts = geo.get_vertex_all_hetero()          # edges per A_v (OBC -1 padding stripped)
    closed_odd = sum(len(cset & set(v)) % 2 for v in verts)
    open_odd = sum(len(oset & set(v)) % 2 for v in verts)
    out = {"R": int(R),
           "closed_anticommuting_Av": int(closed_odd),   # want 0 (commutes with all A_v)
           "open_anticommuting_Av": int(open_odd),        # want 2 (the string's 2 endpoints)
           "OFM_hz0_topological": (0.0 if (closed_odd == 0 and open_odd == 2) else None),
           "OFM_hzinf_trivial": 1.0}                      # z-product state: all σ^z=+1
    out["ok"] = bool(closed_odd == 0 and open_odd == 2)
    return out


def verify_paratoric_fm_geometry(geo) -> Dict[str, Any]:
    """Invariants of the frozen ParaToric-convention FM families at this L — no ED.

    Electric (stock Z-string): ``|closed| = 4R``; ``|open| = 2R`` for even R (exact
    half) or ``2R+1`` for odd R (ParaToric's upper-U convention); open ⊂ closed;
    A_v overlap parity — the closed loop overlaps EVERY vertex star evenly
    (including truncated surface stars: a path vertex carries exactly 2 loop edges
    even on the boundary), the open string oddly at exactly its 2 endpoints.
    ``vertices_interior`` is REPORTED, not asserted — the L=4 loop touches the OBC
    surface by convention.

    Magnetic: `verify_membrane_geometry` + `verify_membrane_charge_flux` at the
    exact placements of both families (`paratoric_membrane_kwargs`): corner-rule
    cube (L>=5, skipped with a reason below) and the R=1 anchor cube (L>=4).

    Location/orientation pinning (2026-08-10 audit — parity/count checks alone
    pass a rigid shift or an upper<->lower flip, which is physically inequivalent
    on even-L OBC): the loop's midpoint bounding box must be exactly
    [s,e]²×{z0}, the open string must contain top-row (y=e) edges and no
    bottom-row (y=s) ones; the cube's box must be [corner-½, corner+R+½]³ and
    the open bucket must hold the full bottom face and none of the top.
    """
    L = geo.Lx
    s, e, R = paratoric_corner_rule(L)
    closed, open_ = paratoric_fm_edges(geo)
    cset, oset = set(closed), set(open_)
    verts = geo.get_vertex_all_hetero()
    closed_odd = sum(len(cset & set(v)) % 2 for v in verts)
    open_odd = sum(len(oset & set(v)) % 2 for v in verts)
    n_open_want = 2 * R if R % 2 == 0 else 2 * R + 1
    cm = np.array([geo.arr_coord[q] for q in cset], dtype=float)
    om = np.array([geo.arr_coord[q] for q in oset], dtype=float)
    z0 = (L - 1) // 2
    at_box = bool(np.all(cm[:, 2] == z0)
                  and cm[:, 0].min() == s and cm[:, 0].max() == e
                  and cm[:, 1].min() == s and cm[:, 1].max() == e)
    upper_u = bool(np.any(om[:, 1] == e) and not np.any(om[:, 1] == s))
    elec = {"corners": (int(s), int(e)), "R": int(R),
            "n_closed": len(cset), "n_open": len(oset), "n_open_want": n_open_want,
            "open_subset_closed": bool(oset.issubset(cset)),
            "closed_anticommuting_Av": int(closed_odd),    # want 0 (⟨closed⟩=+1)
            "open_anticommuting_Av": int(open_odd),        # want 2 (the endpoints)
            "loop_at_spec_box": at_box,                    # pins rigid shifts
            "open_is_upper_U": upper_u,                    # pins the U orientation
            "vertices_interior": bool(s >= 1 and e <= L - 2)}   # False at L=4: reported only
    elec["ok"] = bool(len(cset) == 4 * R and len(oset) == n_open_want
                      and elec["open_subset_closed"]
                      and closed_odd == 0 and open_odd == 2
                      and at_box and upper_u)
    mag = {}
    for fam, Rkw in (("pt-cube", None), ("pt-anchor-R1", 1)):
        try:
            kw = paratoric_membrane_kwargs(geo, Rkw)
        except ValueError as err:              # family not defined at this L
            mag[fam] = {"ok": None, "skipped": str(err)}
            continue
        g = verify_membrane_geometry(geo, R=kw["R"], corner=kw["corner"])
        cf = verify_membrane_charge_flux(geo, R=kw["R"], corner=kw["corner"])
        closed_m, open_m = magnetic_cube_edges(geo, R=kw["R"], corner=kw["corner"],
                                               vertical=kw["vertical"])
        cmm = np.array([geo.arr_coord[q] for q in set(closed_m)], dtype=float)
        omm = np.array([geo.arr_coord[q] for q in set(open_m)], dtype=float)
        c0 = np.array(kw["corner"], dtype=float)
        cube_box = bool(np.all(cmm.min(axis=0) == c0 - 0.5)
                        and np.all(cmm.max(axis=0) == c0 + kw["R"] + 0.5))
        zb, zt = c0[2] - 0.5, c0[2] + kw["R"] + 0.5
        bottom = bool(int((omm[:, 2] == zb).sum()) == (kw["R"] + 1) ** 2
                      and not np.any(omm[:, 2] == zt))
        mag[fam] = {"R": kw["R"], "corner": tuple(kw["corner"]),
                    "geometry_ok": bool(g["ok"]), "charge_flux_ok": bool(cf["ok"]),
                    "cube_at_spec_box": cube_box,          # pins rigid shifts
                    "open_is_bottom_bucket": bottom,       # pins the z-orientation
                    "ok": bool(g["ok"] and cf["ok"] and cube_box and bottom)}
    ok = bool(elec["ok"] and all(v["ok"] is not False for v in mag.values()))
    return {"L": int(L), "electric": elec, "magnetic": mag, "ok": ok}


def verify_membrane_geometry(geo, R: Optional[int] = None,
                             corner: Optional[Tuple[int, int, int]] = None
                             ) -> Dict[str, Any]:
    """Cube-membrane geometry invariants (edge sets only, no operators/ED).

    The FM area-law cancellation *requires* the open surface be exactly half the
    closed cube surface, so this reports rather than fixes. For each of the 3 "up"
    orientations it records: exact halving (``|open| == |closed|//2``), that every
    membrane edge is a strict-bulk edge (no coordinate on the OBC surface 0 or L-1),
    and the raw edge counts (``|closed| = 6(R+1)^2``). ``R=None`` uses the aspect-½
    default (`_bulk_cube`); an explicit ``corner`` (with explicit ``R``) checks that
    exact placement instead — e.g. `paratoric_membrane_kwargs` families.
    """
    L = (geo.Lx, geo.Ly, geo.Lz)
    if corner is not None:
        if R is None:
            raise ValueError("explicit corner needs an explicit R")
        kw = dict(R=int(R), corner=tuple(int(c) for c in corner))
    else:
        kw = _bulk_cube(geo, R)
    R = kw["R"]
    per = {}
    for vert in range(3):
        closed, open_ = magnetic_cube_edges(geo, R=R, corner=kw["corner"], vertical=vert)
        nc, no = len(set(closed)), len(set(open_))
        coords = [geo.arr_coord[q] for q in set(closed) | set(open_)]
        bulk = all(all(0 < x < Lax - 1 for x, Lax in zip(c, L)) for c in coords)
        per[vert] = {"n_closed": nc, "n_open": no,
                     "half_ok": bool(nc % 2 == 0 and no == nc // 2),
                     "edges_interior": bool(bulk)}
    ok = all(v["half_ok"] and v["edges_interior"] for v in per.values())
    return {"R": int(R), "corner": tuple(int(c) for c in kw["corner"]),
            "per_vertical": per, "ok": bool(ok)}


def verify_membrane_charge_flux(geo, R: Optional[int] = None,
                                corner: Optional[Tuple[int, int, int]] = None
                                ) -> Dict[str, Any]:
    """Operator-algebra check of the exactly-solvable membrane FM limits — no ED.

    A σ^x membrane commutes with a σ^z plaquette B_p iff they overlap on an EVEN
    number of edges. On the toric-code ground state (all B_p=+1, all A_v=+1):
      - CLOSED cube surface = ``∏_{v∈cube} A_v`` ⇒ overlaps every B_p evenly ⇒ commutes
        ⇒ ``⟨M_closed⟩ = +1``;
      - OPEN bucket (half the surface) is bounded by the **equatorial flux loop**, so it
        anticommutes with the ``>0`` B_p threaded by that loop ⇒ maps the GS to an
        orthogonal state ⇒ ``⟨M_open⟩ = 0`` ⇒ **O_FM^m(h_x=0) = 0**.
    On the x-polarised product state (h_x→∞) every σ^x=+1 ⇒ ``⟨open⟩=⟨closed⟩=1`` ⇒
    **O_FM^m(h_x→∞) = 1**. Like the electric ratio, this marks the *trivial* (m-condensed)
    phase. The flux loop threads ``4(R+1)`` B_p when the equator is planar (even ``R+1``,
    i.e. R=1,3) and slightly more when the floor/ceil split steps it by one layer (odd
    ``R+1``, R=2) — the count is *not* asserted to a fixed value; what matters physically
    is that it is nonzero and even (a closed loop) and stays in the bulk. Returns counts
    + a pass flag, worst case over the 3 orientations. ``corner`` (with explicit ``R``)
    overrides the centered `_bulk_cube` placement, as in `verify_membrane_geometry`.
    """
    if corner is not None:
        if R is None:
            raise ValueError("explicit corner needs an explicit R")
        kw = dict(R=int(R), corner=tuple(int(c) for c in corner))
    else:
        kw = _bulk_cube(geo, R)
    R = kw["R"]
    plaqs = [set(p) for p in geo.plaq_all]
    per = {}
    for vert in range(3):
        closed, open_ = magnetic_cube_edges(geo, R=R, corner=kw["corner"], vertical=vert)
        cset, oset = set(closed), set(open_)
        c_odd = sum(len(cset & p) % 2 for p in plaqs)
        # boundary = B_p anticommuting with the open bucket = the equatorial flux loop
        bnd = [pl for pl, p in zip(range(len(plaqs)), plaqs) if len(oset & p) % 2]
        o_odd = len(bnd)
        bnd_bulk = all(all(0 < x < Lax - 1 for x, Lax in
                           zip(geo.plaq_centers[pl], (geo.Lx, geo.Ly, geo.Lz)))
                       for pl in bnd)
        per[vert] = {"closed_anticommuting_Bp": int(c_odd),   # want 0 (⟨closed⟩=+1)
                     "open_flux_loop_Bp": int(o_odd),          # >0 & even (⟨open⟩=0)
                     "flux_loop_bulk": bool(bnd_bulk)}
    ok = all(v["closed_anticommuting_Bp"] == 0 and v["open_flux_loop_Bp"] > 0
             and v["open_flux_loop_Bp"] % 2 == 0 and v["flux_loop_bulk"]
             for v in per.values())
    return {"R": int(R), "per_vertical": per,
            "OFM_hx0_topological": (0.0 if ok else None),
            "OFM_hxinf_trivial": 1.0, "ok": bool(ok)}


# =============================================================================
# Operators + the FM ratio (shared by both sectors)
# =============================================================================

def _pauli_product(hi, indices: Sequence[int], pauli: str):
    """∏ σ^{pauli} over `indices` as a NetKet operator (deduplicated)."""
    sigma = nk.operator.spin.sigmaz if pauli == "z" else nk.operator.spin.sigmax
    op = None
    for i in dict.fromkeys(int(j) for j in indices):     # preserve order, drop dups
        term = sigma(hi, i)
        op = term if op is None else op * term
    return op


def _pauli_zx_product(hi, z_edges: Sequence[int], x_edges: Sequence[int]):
    """Z(z_edges)·X(x_edges) as a NetKet operator — the mixed string built exactly
    the way `validation._mean_operators` builds the decorated ⟨B̃_p⟩. Supports are
    disjoint (`dressed_string` raises otherwise), so factor order is immaterial."""
    zop = _pauli_product(hi, z_edges, "z")
    xop = _pauli_product(hi, x_edges, "x")
    if zop is None:
        return xop
    return zop if xop is None else zop * xop


def sector_edges(geo, sector: str, **kw) -> Tuple[List[int], List[int]]:
    """(closed, open_) edge-index sets for the requested sector's FM operator.

    sector="electric" → σ^z Wilson square, kw: ``plane_axis, plane_at, corner, R``
                        (`electric_loop_edges`).
    sector="magnetic" → σ^x cube-surface 't Hooft membrane, kw: ``R, corner, vertical``
                        (`magnetic_cube_edges`, Option B).
    """
    if sector == "electric":
        return electric_loop_edges(geo, **kw)
    if sector == "magnetic":
        return magnetic_cube_edges(geo, **kw)
    raise ValueError(f"sector must be 'electric' or 'magnetic', got {sector!r}")


def sector_operators(geo, hi, sector: str, *, dual: bool = False,
                     model: str = "bosonic", **kw):
    """Build (open_op, closed_op) NetKet operators for the requested sector.

    Sector names stay PHYSICAL; `dual=True` (Hadamard-conjugated run) swaps the
    Pauli letter the physical operator is built from, so the JSON keys keep
    basis-independent meaning. Diagonality follows the letter, not the sector:
    primal electric / dual magnetic are diagonal σ^z products (cheap `vs.expect`);
    primal magnetic prefers the telescoped estimator (`fm_ratio_telescoped`);
    dual electric is an off-diagonal σ^x string — fine through `vs.expect` for the
    small production loops (R=1 → 4 edges), heavier-tailed as the perimeter grows.

    `model="fermionic"` (electric only): both operators become dressed Z(z)·X(x)
    strings (`dressed_electric_edges`) so they commute with the decorated B̃_p —
    off-diagonal but small, scored via `vs.expect` exactly like the dual electric
    loop. Same kwargs as the bosonic electric path.
    """
    if model == "fermionic":
        if sector != "electric":
            raise NotImplementedError(
                "fermionic FM: only the electric (dressed σ^z) sector is implemented")
        if dual:
            raise ValueError("dual basis is bosonic-only")
        (cz, cx, _), (oz, ox, _flux) = dressed_electric_edges(geo, **kw)
        return _pauli_zx_product(hi, oz, ox), _pauli_zx_product(hi, cz, cx)
    closed, open_ = sector_edges(geo, sector, **kw)
    pauli = "z" if (sector == "electric") != dual else "x"
    return _pauli_product(hi, open_, pauli), _pauli_product(hi, closed, pauli)


def uses_telescoped(sector: str, dual: bool = False) -> bool:
    """True when the sector's FM operator is OFF-diagonal in the sampling basis of a
    membrane-shaped support, i.e. `build_loop_operators` returns telescope SPECS
    rather than NetKet operator pairs. Only the primal magnetic membrane qualifies:
    under `dual` the membrane is a diagonal σ^z product scored sample-wise
    (`uses_sampled_diagonal`), and the (dual) electric loop, though off-diagonal,
    is a small string scored via expect."""
    return sector == "magnetic" and not dual


def uses_sampled_diagonal(sector: str, dual: bool = False) -> bool:
    """True when the membrane is DIAGONAL in the sampling basis (dual runs):
    `build_loop_operators` returns edge-set SPECS scored by `fm_ratio_avg_sampled`
    (plain ±1 products over MC samples), never NetKet operator pairs — a
    membrane-support LocalOperator materializes a 2^|support| sparse block
    (2^54 at the smallest L>=5 family: the L=6 inline O_FM OOM; 2026-08-10 audit)."""
    return sector == "magnetic" and dual


def build_loop_operators(geo, hi, sector: str, *, placement: str = "bulk",
                         planes: Sequence[str] = ("xy", "xz", "yz"),
                         plane_at: Optional[int] = None, R: Optional[int] = None,
                         aspect: Optional[float] = None,
                         op_kwargs: Optional[Dict] = None,
                         dual: bool = False, model: str = "bosonic"
                         ) -> Tuple[List[Tuple[str, Any, Any]], Dict[str, Any]]:
    """The (label, open_op, closed_op) list to average over, plus a placement meta dict.

    `model="fermionic"` (electric only): every operator pair is the dressed
    Z(z)·X(x) string of `sector_operators` — same placements/labels/averaging,
    only the operators change. The magnetic membrane has no dressing yet.

    placement="bulk" (electric only): a bulk-centered square in each requested plane
    ('xy'/'xz'/'yz'); their FM ratios are averaged (see `fm_ratio_avg`). Requires L>=4.
    Loop side, pick one:
      • `R=None` (default) → largest bulk square, L-3 (aspect R/L drifts → 1 with L);
      • `R=<int>`          → a fixed side at every L (e.g. 1 = perimeter-4 plaquette);
      • `aspect=<float>`   → a **fixed aspect ratio** R/L≈aspect (e.g. 0.5 = L/2): for odd
        L both floor(L·aspect) and ceil(L·aspect) loops are built and enter the average
        **on the same samples** (`_aspect_sizes`), symmetrising the parity wobble to an
        effective R/L=aspect. Out-of-bulk sides are dropped with a warning (so L=5 at 0.5
        falls back to floor-only, R=2). Overrides `R`. Labels become 'plane:R{side}'.
    placement="boundary": the single legacy loop from `op_kwargs` (label ''), unchanged.
    placement="paratoric": the frozen NQS-vs-QMC comparison family — ParaToric's stock
    Z-string (electric) / corner-rule or fixed-R anchor cube membrane (magnetic), one
    operator, one orientation, no averaging. `R` is membrane-anchor-only here.

    TODO(telescoping): ⟨W_closed⟩ is measured directly, so its relative MC error grows
    exponentially with the perimeter. For large loops a nested-ratio (telescoping)
    estimator ∏ ⟨W_{ℓ+1}⟩/⟨W_ℓ⟩ would be far better conditioned; not yet implemented.
    """
    op_kwargs = op_kwargs or {}
    if placement == "boundary":
        open_op, closed_op = sector_operators(geo, hi, sector, dual=dual, model=model,
                                              **op_kwargs)
        meta = {"placement": "boundary", "planes": [], "plane_at": op_kwargs.get("plane_at"),
                "R": op_kwargs.get("R"), "aspect": None}
        return [("", open_op, closed_op)], meta
    if placement == "paratoric":
        # The frozen NQS-vs-QMC convention (2026-08-10): operators bit-identical to
        # ParaToric's, single loop/membrane, single orientation, NO plane averaging.
        # Electric = the stock lattice.cpp Z-string (boundary-touching at L=4 by
        # convention); magnetic = corner-rule cube (R=None, L>=5) or the fixed-R
        # anchor cube (R=<int>, e.g. 1). Each (sector, R-family) is its own FSS
        # curve — families are never mixed in one extrapolation fit.
        if model == "fermionic":
            raise NotImplementedError(
                "placement='paratoric' is bosonic-only (QMC comparison family)")
        if aspect is not None or plane_at is not None:
            raise ValueError("placement='paratoric': geometry is fixed by the corner "
                             "rule — drop --aspect/--plane_at")
        # Runtime self-check on every operator build (cheap, geometry-only):
        # counts/parity/subset + location and orientation pinning. Geometry
        # identity vs the actual C++ remains tests/test_fm_paratoric.py's job.
        rep = verify_paratoric_fm_geometry(geo)
        if not rep["ok"]:
            raise ValueError(f"paratoric FM geometry self-check FAILED: {rep}")
        s, e, Rpt = paratoric_corner_rule(geo.Lx)
        if sector == "electric":
            if R is not None:
                raise ValueError("placement='paratoric' electric: the loop is "
                                 "ParaToric's stock geometry — drop --R (the R "
                                 "override is membrane-anchor-only)")
            closed, open_ = paratoric_fm_edges(geo)
            pauli = "x" if dual else "z"
            pairs = [("pt", _pauli_product(hi, open_, pauli),
                      _pauli_product(hi, closed, pauli))]
            meta = {"placement": "paratoric", "sector": "electric",
                    "convention": "pt-string", "corners": [s, e],
                    "planes": ["pt"], "plane_at": (geo.Lx - 1) // 2,
                    "R": Rpt, "aspect": None}
            return pairs, meta
        if sector != "magnetic":
            raise ValueError(f"sector must be 'electric' or 'magnetic', got {sector!r}")
        kw = paratoric_membrane_kwargs(geo, R)
        label = "pt" if R is None else f"ptR{kw['R']}"
        meta = {"placement": "paratoric", "sector": "magnetic",
                "convention": "pt-cube" if R is None else f"pt-anchor-R{kw['R']}",
                "corners": ([s, e] if R is None else None),
                "planes": [label], "plane_at": None, "aspect": None,
                "R": kw["R"], "corner": list(kw["corner"]),
                "vertical": kw["vertical"]}
        # Both estimators consume the same edge-set SPECS: primal -> telescoped
        # (off-diagonal), dual -> sample-wise diagonal products. Never NetKet
        # operator pairs here (2^|support| OOM — see `uses_sampled_diagonal`).
        return [(label, kw)], meta
    if placement != "bulk":
        raise ValueError(f"placement must be 'bulk', 'boundary' or 'paratoric', "
                         f"got {placement!r}")
    if sector == "magnetic":
        if model == "fermionic":
            raise NotImplementedError(
                "fermionic FM: only the electric (dressed σ^z) sector is implemented")
        # Cube-surface 't Hooft membrane, aspect-½ (⌊L/2⌋) or explicit R, averaged over
        # the 3 "up" orientations (isotropy, analog of the electric xy/xz/yz average).
        # ALWAYS per-orientation SPECS, never NetKet operators: primal is scored by
        # the telescoped off-diagonal estimator (`uses_telescoped`), dual by plain
        # sample-wise diagonal products (`uses_sampled_diagonal` — a LocalOperator
        # membrane materializes a 2^|support| block and OOMs from L=5 up).
        kw = _bulk_cube(geo, R)
        specs = [(f"v{ax}", dict(R=kw["R"], corner=kw["corner"], vertical=ax))
                 for ax in range(3)]
        meta = {"placement": "bulk", "sector": "magnetic",
                "planes": [s[0] for s in specs], "plane_at": None,
                "aspect": aspect, "R": kw["R"], "corner": list(kw["corner"])}
        return specs, meta
    pairs, kw0, sizes_seen = [], None, None
    for label in planes:
        if aspect is not None:
            sizes, dropped, Rmax = _aspect_sizes(geo, PLANE_NORMAL[label], aspect)
            if dropped:
                print(f"[fm] aspect={aspect}: plane {label} drops out-of-bulk R={dropped} "
                      f"(need R<=Rmax={Rmax}); keeping R={sizes}", flush=True)
            if not sizes:
                raise ValueError(f"aspect={aspect}: no bulk-fitting loop in plane {label} "
                                 f"(Rmax={Rmax}); L too small for this aspect ratio")
        else:
            sizes = [R]                       # None -> _bulk_square default (largest bulk)
        for Rs in sizes:
            kw = _bulk_square(geo, PLANE_NORMAL[label], plane_at=plane_at, R=Rs)
            kw0 = kw0 or kw
            open_op, closed_op = sector_operators(geo, hi, "electric", dual=dual,
                                                  model=model, **kw)
            lbl = f"{label}:R{kw['R']}" if aspect is not None else label
            pairs.append((lbl, open_op, closed_op))
        sizes_seen = sizes                    # uniform across planes for a cubic box
    meta = {"placement": "bulk",
            "planes": [p[0] for p in pairs] if aspect is not None else list(planes),
            "plane_at": kw0["plane_at"], "aspect": aspect,
            "R": (sizes_seen if aspect is not None else kw0["R"])}
    return pairs, meta


def _stat_err(stat, n_samples: int) -> float:
    """Standard error of the mean, robust to NetKet's autocorrelation-corrected
    `error_of_mean` returning NaN.

    That NaN happens for a short-chain, low-cardinality *diagonal* estimator (our
    σ^z string takes values in {±1}): when a chain's samples are all-equal the
    within-chain variance is 0 and the autocorrelation/split-R̂ normalisation is
    0/0. We fall back to the plain sqrt(variance / n_samples), which ignores the
    autocorrelation time and is therefore *mildly optimistic* — flag it as such —
    but finite. Best paired with long chains (few `n_chains`) so the primary,
    autocorrelation-aware estimate is the one that's actually used.
    """
    e = float(np.real(stat.error_of_mean))
    if np.isfinite(e):
        return e
    var = float(np.real(getattr(stat, "variance", np.nan)))
    if np.isfinite(var) and n_samples > 0:
        return float(np.sqrt(var / n_samples))
    return float("nan")


def fm_ratio(vstate, open_op, closed_op, return_den: bool = False):
    """Fredenhagen–Marcu ratio O = ⟨S_open⟩/√⟨W_closed⟩, NaN if ⟨W_closed⟩ ≤ 0.

    Pooled convention (2026-08-11 audit): a non-positive closed-loop expectation
    means the ratio is undefined at this budget — loud NaN, never a |·|-fold.
    `return_den=True` appends the raw closed-loop expectation (W_mean, W_err) to
    the return — the denominator must travel with every ratio so the analysis
    layer can den-gate near-critical points.

    Both expectations are sampled from the same variational state. The error is
    first-order propagation through O(S,W) = S·W^(-1/2) (W > 0 on this path):
        σ_O² = (∂O/∂S σ_S)² + (∂O/∂W σ_W)²,
        ∂O/∂S = W^(-1/2),  ∂O/∂W = -½ S W^(-3/2).
    Per-expectation errors go through `_stat_err` (NetKet `.error_of_mean`, with a
    variance-based fallback so a near-constant chain can't NaN out the whole point).
    """
    n = int(getattr(vstate, "n_samples", 0) or 0)
    S = vstate.expect(open_op)
    W = vstate.expect(closed_op)
    Sm, Se = float(np.real(S.mean)), _stat_err(S, n)
    Wm, We = float(np.real(W.mean)), _stat_err(W, n)
    if Wm <= 0.0:
        # pooled convention (2026-08-11 audit, CRUCIAL 1): a non-positive
        # ⟨closed⟩ means the ratio is undefined at this budget — loud NaN,
        # never the silent |·|-fold (which biases + near transitions). The
        # den fields let the analysis layer gate these points on both axes.
        return (float("nan"), float("nan"), (Wm, We)) if return_den \
            else (float("nan"), float("nan"))
    denom = np.sqrt(Wm)
    O = Sm / denom
    dO_dS = 1.0 / denom
    dO_dW = -0.5 * Sm / Wm ** 1.5
    Oe = float(np.hypot(dO_dS * Se, dO_dW * We))
    return (O, Oe, (Wm, We)) if return_den else (O, Oe)


def fm_ratio_avg(vstate, pairs: Sequence[Tuple[str, Any, Any]]
                 ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """Mean FM ratio over several loop orientations (the xy/xz/yz bulk average).

    `pairs` = [(label, open_op, closed_op), ...]. Scores each with `fm_ratio`, then
    returns (O_mean, O_err, per_plane) where per_plane[label] = (O_i, e_i). The error is
    the propagated MC error of the mean, sqrt(Σ e_i²)/N, treating the orientations as
    independent — they share samples, so the per-plane spread (inspect per_plane) is the
    honest anisotropy check.
    """
    per = {}
    for label, open_op, closed_op in pairs:
        per[label] = fm_ratio(vstate, open_op, closed_op)
    Os = np.array([o for o, _ in per.values()], float)
    Oes = np.array([e for _, e in per.values()], float)
    O_mean = float(np.mean(Os))
    O_err = float(np.sqrt(np.sum(Oes ** 2)) / len(Oes))
    return O_mean, O_err, per


# =============================================================================
# Telescoped membrane estimator (off-diagonal σ^x) — B2/B3
#
# ⟨M⟩ = ⟨∏_e σ^x_e⟩ = E_{σ~|ψ|²}[ ψ(σ⊕m)/ψ(σ) ]. Flipping all ~50 membrane edges
# at once gives a heavy-tailed, area-law-suppressed ratio. We flip the membrane
# face-by-face (nested M_1 ⊂ … ⊂ M_K = target) so the per-sample ratio is assembled
# as a product of per-face increments ψ(σ⊕M_k)/ψ(σ⊕M_{k-1}); this is grouping-
# invariant (telescopes to the exact direct estimator, UNBIASED) and lets B3 flag a
# single heavy-tailed face. NOTE: shared-|ψ|²-sample telescoping does not by itself
# reduce variance below the direct estimator — genuine reduction needs multilevel
# (intermediate) resampling; the B3 diagnostics are what decide whether that is
# warranted on Colab. Error via block-jackknife through the whole open/√closed ratio.
# =============================================================================

def _batched_log_amp(vs, x, chunk: Optional[int] = None) -> np.ndarray:
    """logψ over a batch `x` (…, N) → flat (M,) complex, evaluated in `chunk`-row
    blocks to bound memory (L>=6 needs this, cf. the S2 pipeline / --chunk_size)."""
    x2 = np.asarray(x).reshape(-1, np.asarray(x).shape[-1])
    if not chunk or chunk >= x2.shape[0]:
        return np.asarray(vs.log_value(x2))
    return np.concatenate([np.asarray(vs.log_value(x2[i:i + chunk]))
                           for i in range(0, x2.shape[0], chunk)])


def _nested_log_ratios(vs, samples, faces: Sequence[Sequence[int]],
                       chunk: Optional[int] = None) -> np.ndarray:
    """Per-sample log-amplitude increments for growing the membrane face-by-face.

    Returns `g` of shape (n_samples, K) where ``g[:, k] = logψ(σ⊕M_{k+1}) - logψ(σ⊕M_k)``
    (M_0 = ∅), so ``exp(g.sum(axis=1)) = ψ(σ⊕M_K)/ψ(σ)`` is the per-sample estimator of
    ⟨M⟩ and each column is one face's contribution (B3 monitors these). σ is ±1; a flip
    on edge e multiplies σ[...,e] by -1.
    """
    S = np.asarray(samples).reshape(-1, np.asarray(samples).shape[-1])
    n, N = S.shape
    cum = np.zeros(N, dtype=bool)                    # cumulative membrane mask
    logs = [_batched_log_amp(vs, S, chunk)]          # logψ(σ) at M_0 = ∅
    for face in faces:
        cum = cum.copy()
        cum[np.fromiter((int(e) for e in face), dtype=int)] = True
        flip = np.where(cum, -1.0, 1.0)              # ±1 multiplier on membrane edges
        logs.append(_batched_log_amp(vs, S * flip, chunk))
    L = np.stack(logs, axis=1)                       # (n, K+1)
    return np.diff(L, axis=1)                         # (n, K) per-face increments


# Minimum per-face phase coherence |⟨e^{i·Im g}⟩| below which the membrane product's
# central value / error bar are not trustworthy (the phase winds faster than the sample
# set resolves, so the complex mean is dominated by cancellation noise). 1.0 = perfectly
# coherent (always true for a real logψ, Im g = 0); → 0 = phase-scrambled. The 0.2 default
# is a heuristic first line; tighten once the h_y campaign shows real distributions.
PHASE_COHERENCE_MIN = 0.2


def membrane_estimator_health(g: np.ndarray) -> List[Dict[str, float]]:
    """B3: per-face health of the amplitude-ratio increments `g` (n_samples, K).

    For each face k reports the ratio r=exp(g[:,k])'s variance, the batch-mean
    excess-kurtosis (Gaussian≈0; large ⇒ heavy tail ⇒ the product's error bar is not
    trustworthy), the effective sample size ESS = mean(|r|)²/mean(|r|²)·n, and — for the
    sign-full (complex logψ) regime — the **phase coherence** ``coherence`` = |⟨e^{i·Im
    g}⟩| with a boolean ``coh_ok`` (coherence ≥ PHASE_COHERENCE_MIN). One heavy-tailed OR
    phase-incoherent face invalidates the whole product — this cell is permanent, not a
    one-off (see the S2 pipeline's variance lessons).

    Complex-aware: for a complex ansatz (h_y != 0) the per-face ratio is complex, so the
    heavy-tail/ESS statistics are taken on its magnitude |exp(g)| (NOT exp(Re g), which
    is the same magnitude only by coincidence of notation) and the variance is the
    complex E|r-⟨r⟩|². For a real logψ these reduce exactly to the original formulas and
    coherence ≡ 1 (Im g = 0), so the check is a no-op on the sign-free path.
    """
    r = np.exp(g)                                    # complex amplitude-ratio per face
    a = np.abs(r)                                     # magnitude (== r when logψ is real)
    im = np.imag(g)                                   # per-face phase increment (0 if real)
    n = r.shape[0]
    out = []
    for k in range(r.shape[1]):
        rk, ak = r[:, k], a[:, k]
        m1, m2 = float(np.mean(ak)), float(np.mean(ak ** 2))
        var = float(np.var(rk))                      # E|r-⟨r⟩|²  (== Var for real r)
        mu, sd = float(np.mean(ak)), float(np.std(ak) + 1e-300)
        kurt = float(np.mean(((ak - mu) / sd) ** 4) - 3.0)
        ess = float(n * (m1 ** 2) / m2) if m2 > 0 else 0.0
        coherence = float(np.abs(np.mean(np.exp(1j * im[:, k]))))   # 1 = coherent, 0 = scrambled
        out.append({"face": k, "variance": var, "excess_kurtosis": kurt,
                    "ess": ess, "ess_frac": ess / n if n else 0.0,
                    "coherence": coherence,
                    "coh_ok": bool(coherence >= PHASE_COHERENCE_MIN)})
    return out


def _jackknife_fm_ratio(r_open: np.ndarray, r_closed: np.ndarray,
                        n_blocks: int = 32) -> Tuple[float, float]:
    """Block-jackknife O_FM = mean(r_open)/√mean(r_closed) through the whole ratio,
    NaN when mean(r_closed) ≤ 0 (pooled convention, 2026-08-11 audit — no |·|-fold;
    an undefined delete-one replicate makes the ERROR loud-NaN too, with a warning).

    r_open, r_closed are the per-sample estimators of ⟨M_open⟩, ⟨M_closed⟩ on the SAME
    configurations, so the ratio's numerator and denominator are correlated — jackknifing
    the assembled ratio (not each mean separately) propagates that correlation honestly.
    Returns (O, Oe).

    Complex-aware: for a complex ansatz the per-sample ratios are complex; ⟨M_open⟩ and
    ⟨M_closed⟩ are real (Hermitian membrane operators), so we average the COMPLEX ratios
    and take Re[·] of the numerator mean at the end — Re(mean) removes the vanishing MC
    imaginary noise, whereas the old per-sample np.real dropped a genuine variance
    contribution. The denominator uses Re[⟨M_closed⟩] of the complex mean. Reduces
    exactly to the original real path when logψ is real.
    """
    ro, rc = np.asarray(r_open), np.asarray(r_closed)   # keep complex
    n = ro.shape[0]

    def ratio(o, c):
        # pooled convention (2026-08-11 audit): Re of the complex mean (the
        # imaginary part is vanishing MC noise for Hermitian membranes); a
        # non-positive ⟨closed⟩ → NaN, never |·|-folded.
        d = np.real(np.mean(c))
        return np.real(np.mean(o)) / np.sqrt(d) if d > 0 else float("nan")

    full = ratio(ro, rc)
    b = int(min(n_blocks, n))
    if b < 2:
        return float(full), float("nan")
    blocks = np.array_split(np.arange(n), b)
    jk = np.array([ratio(np.delete(ro, blk), np.delete(rc, blk)) for blk in blocks])
    bad = int(np.sum(~np.isfinite(jk)))
    if bad:
        # delete-one ⟨closed⟩ crossed ≤ 0: the vanishing replicates are the most
        # influential ones, so nan-skipping would bias the error DOWN exactly in
        # the marginal regime — loud NaN instead (mirrors the QMC pooled_fm
        # policy; the den-gate has already failed wherever this triggers).
        print(f"[fm] WARNING: {bad}/{b} jackknife replicates undefined "
              f"(delete-one <closed> <= 0) -- err = NaN", flush=True)
        return float(full), float("nan")
    Oe = float(np.sqrt((b - 1) / b * np.sum((jk - jk.mean()) ** 2)))
    return float(full), Oe


def fm_ratio_telescoped(vs, geo, *, R: int, corner: Tuple[int, int, int],
                        vertical: int = 2, n_blocks: int = 32,
                        chunk: Optional[int] = None
                        ) -> Tuple[float, float, Dict[str, Any]]:
    """Telescoped membrane FM ratio O_FM^m = ⟨M_open⟩/√⟨M_closed⟩ from `vs`'s samples
    (NaN when ⟨M_closed⟩ ≤ 0 — inherits `_jackknife_fm_ratio`'s pooled convention).

    Grows the open and closed cube membranes face-by-face, computes the per-sample
    amplitude-ratio product on the state's current MC samples, block-jackknifes the
    assembled ratio, and returns (O, Oe, diag) where diag carries the B3 per-face
    health for both membranes. Uses `vs.samples` (reset first for fresh samples).
    """
    samples = np.asarray(vs.samples)
    closed_faces, open_faces = cube_membrane_faces(geo, R=R, corner=corner,
                                                   vertical=vertical)
    g_open = _nested_log_ratios(vs, samples, open_faces, chunk)
    g_closed = _nested_log_ratios(vs, samples, closed_faces, chunk)
    r_open = np.exp(g_open.sum(axis=1))
    r_closed = np.exp(g_closed.sum(axis=1))
    O, Oe = _jackknife_fm_ratio(r_open, r_closed, n_blocks)
    diag = {"health_open": membrane_estimator_health(g_open),
            "health_closed": membrane_estimator_health(g_closed),
            "mean_closed": float(np.real(np.mean(r_closed))),
            "mean_open": float(np.real(np.mean(r_open)))}
    return O, Oe, diag


def _json_nonfinite_safe(obj):
    """Replace non-finite floats with None for json.dump (RFC-safe; mirrors
    analysis/scripts/paratoric_driver._json_safe -- see its docstring)."""
    if isinstance(obj, dict):
        return {k: _json_nonfinite_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_nonfinite_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _quadrature_err(Oes, who):
    """Quadrature-average per-spec errors, NaN-propagating: nansum silently
    turned an all-NaN error list into 0.0 -- for the single-spec paratoric
    placement that meant Oe=0.0 exactly, which fit_transition's saturated-point
    floor then MAX-weighted (2026-08-11 re-verification, finding 1). Any
    non-finite per-spec error now makes the combined error loudly NaN."""
    Oes = np.asarray(Oes, float)
    if not np.all(np.isfinite(Oes)):
        bad = int(np.sum(~np.isfinite(Oes)))
        print(f"[fm] WARNING {who}: {bad}/{len(Oes)} spec errors undefined "
              f"-- combined O_err = NaN", flush=True)
        return float("nan")
    return float(np.sqrt(np.sum(Oes ** 2)) / len(Oes))


def fm_ratio_avg_telescoped(vs, geo, specs: Sequence[Tuple[str, Dict]], *,
                            n_blocks: int = 32, chunk: Optional[int] = None
                            ) -> Tuple[float, float, Dict[str, Tuple[float, float]],
                                       Dict[str, Any]]:
    """Mean telescoped membrane FM ratio over the 3 "up" orientations (magnetic bulk
    average), the σ^x analog of `fm_ratio_avg`.

    `specs` = [(label, kwargs_for_fm_ratio_telescoped), ...]; all share the state's
    current samples. Returns (O_mean, O_err, per, diags): per[label]=(O_i,e_i) is the
    isotropy spread, diags[label] the B3 health for that orientation. Because the
    membrane estimator is heavy-tailed, always read `diags` — a blown-up excess
    kurtosis / tiny ess_frac means O_err understates the true uncertainty.
    """
    per, diags = {}, {}
    for label, kw in specs:
        O, Oe, d = fm_ratio_telescoped(vs, geo, n_blocks=n_blocks, chunk=chunk, **kw)
        per[label], diags[label] = (O, Oe), d
    Os = np.array([o for o, _ in per.values()], float)
    Oes = np.array([e for _, e in per.values()], float)
    return (float(np.mean(Os)), _quadrature_err(Oes, "fm_ratio_avg_telescoped"),
            per, diags)


def fm_ratio_sampled(vs, geo, *, R: int, corner: Tuple[int, int, int],
                     vertical: int = 2, n_blocks: int = 32,
                     return_den: bool = False):
    """Diagonal membrane FM ratio from per-sample ±1 products (dual frame only).

    The physical σ^x membrane is a σ^z product on the Hadamard-rotated state, so
    ⟨M⟩ is a plain mean of sample-column products — no operator is ever built
    (a membrane-support LocalOperator is a 2^|support| object; see
    `uses_sampled_diagonal`). Error via `_jackknife_fm_ratio` through the
    assembled ratio: open and closed products share the samples AND half the
    support, so their correlation must be propagated (an independent-error
    formula over-estimates by ~1.6× at campaign scales)."""
    closed, open_ = magnetic_cube_edges(geo, R=R, corner=corner, vertical=vertical)
    x = np.asarray(vs.samples)
    x = x.reshape(-1, x.shape[-1])
    r_open = np.prod(x[:, list(dict.fromkeys(open_))], axis=1)
    r_closed = np.prod(x[:, list(dict.fromkeys(closed))], axis=1)
    O, Oe = _jackknife_fm_ratio(r_open, r_closed, n_blocks)
    if not return_den:
        return O, Oe
    # raw ⟨closed⟩ + block SEM: the den-gate fields (audit MEDIUM b), blocked
    # like the jackknife so the two error scales are comparable
    b = int(min(n_blocks, r_closed.shape[0]))
    bm = np.array([np.real(np.mean(blk)) for blk in np.array_split(r_closed, b)])
    den = float(np.real(np.mean(r_closed)))
    den_err = float(bm.std(ddof=1) / np.sqrt(b)) if b > 1 else float("nan")
    return O, Oe, (den, den_err)


def fm_ratio_avg_sampled(vs, geo, specs: Sequence[Tuple[str, Dict]], *,
                         n_blocks: int = 32
                         ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """Mean sampled-diagonal membrane FM ratio over orientations (dual analog of
    `fm_ratio_avg_telescoped`; same spec format, shared samples)."""
    per = {}
    for label, kw in specs:
        per[label] = fm_ratio_sampled(vs, geo, n_blocks=n_blocks, **kw)
    Os = np.array([o for o, _ in per.values()], float)
    Oes = np.array([e for _, e in per.values()], float)
    return (float(np.mean(Os)), _quadrature_err(Oes, "fm_ratio_avg_sampled"), per)


# =============================================================================
# Checkpoint loader + grid sweep
# =============================================================================

def _weights_path(json_path: str) -> str:
    """Sibling `.mpack` for a `train.py` artifact, falling back to `.ckpt.mpack`
    (the periodic checkpoint weights) for a run that timed out before its final."""
    if json_path.endswith(".curve.json"):      # checkpoint: {name}.curve.json -> base {name}
        base = json_path[:-len(".curve.json")]
    elif json_path.endswith(".json"):
        base = json_path[:-len(".json")]
    else:
        base = json_path
    mpack = base + ".mpack"
    if not os.path.exists(mpack):
        alt = base + ".ckpt.mpack"
        if not os.path.exists(alt):
            raise FileNotFoundError(
                f"no weights for {json_path}: tried {base}.mpack and {base}.ckpt.mpack")
        mpack = alt
    return mpack


def _load_weights(vs, json_path: str):
    """Deserialize the checkpoint weights into `vs`'s structure (returns the new vs).

    Same network/sampler structure -> reusing one `vs` template across an hz sweep
    keeps JAX's compiled `expect` warm; only the parameters change per checkpoint.

    Guarded like `io.load_weights` (4d3d479), and further: `from_bytes` restores
    the CHECKPOINT's sampling config (n_samples, n_discard_per_chain, chunk_size)
    AND its sampler RNG — unguarded, every eval override is silently voided:
    `--eval_samples 65536` re-evaluated at the training 8192, and `--seed`
    reproduced the checkpoint's sample stream bit-for-bit across "replicas"
    (2026-08-10 audit, confirmed on production .eval65k/.fm65k artifacts).
    We keep the caller's config, keep the checkpoint's equilibrated chain
    positions as the MC init (they equilibrate the very weights being loaded),
    and re-key the sampler with the CALLER's RNG so seeds are honored and
    replica error checks measure a real spread.
    """
    keep = (vs.n_samples, vs.n_discard_per_chain, vs.chunk_size)
    fresh_state = vs.sampler_state                  # carries the caller's seed
    with open(_weights_path(json_path), "rb") as f:
        vs = flax.serialization.from_bytes(vs, f.read())
    vs.n_samples, vs.n_discard_per_chain, vs.chunk_size = keep
    try:
        vs.sampler_state = vs.sampler_state.replace(rng=fresh_state.rng)
    except (AttributeError, TypeError):             # exotic sampler-state layout:
        vs.sampler_state = fresh_state              # fresh init; n_discard burns in
    vs.reset()                                      # drop any cached sample stream
    return vs


def _struct_sig(cfg: Dict[str, Any]) -> str:
    """Signature of everything that fixes the network/sampler/state *shape* (all the
    build_state inputs except hz and n_samples). Checkpoints in one hz sweep share
    it, so they can reuse a single built `vs`; a mismatch forces a fresh rebuild.
    Includes `hy`/`force_complex`/`dtype` — these flip the model between real and
    complex weights, so a dir mixing hy=0 and hy!=0 runs must never reuse a
    dtype-inconsistent template."""
    keys = ("L", "bc", "model", "arch", "hidden", "noninv_channels", "n_noninv",
            "noninv_hidden", "inv_hidden", "cnn_hidden", "kernel_size",
            "radius_edge", "radius_plaq", "n_chains", "n_sweeps", "n_discard",
            "chunk_size", "vanilla_depth", "noninv_identity", "dual_basis",
            "hy", "force_complex", "dtype")
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)


def load_vstate(json_path: str, *, eval_samples: Optional[int] = None,
                eval_chains: Optional[int] = None, seed: Optional[int] = None):
    """Rebuild and reload a trained NQS from a `train.py` artifact pair.

    Reads `{json_path}` (config + observables), rebuilds the exact VMC stack via
    `builders.build_state(config)` (H skipped — FM extraction never uses it), then
    loads the sibling `.mpack` weights. `eval_samples` overrides n_samples for a
    more precise expectation; `eval_chains` overrides n_chains — GPU runs default to
    n_chains=1024, i.e. only ~8 samples/chain at eval, too short to estimate the
    autocorrelation time (→ NaN `error_of_mean`); a small value (e.g. 16) makes long
    chains so the primary error estimate is valid. `seed` re-seeds the sampler.
    Weights are sampler-shape-independent, so both overrides reload cleanly. Returns
    (config, geo, hi, vstate).
    """
    with open(json_path) as f:
        meta = json.load(f)
    cfg = dict(meta["config"])
    if eval_samples is not None:
        cfg["n_samples"] = eval_samples
    if eval_chains is not None:
        cfg["n_chains"] = eval_chains
    if seed is not None:
        cfg["seed"] = seed
    geo, hi, _Ham, vs, _xz = build_state(cfg, build_ham=False)
    vs = _load_weights(vs, json_path)
    return cfg, geo, hi, vs


def _matches(cfg: Dict[str, Any], L, hx, model, bc, hy: float = 0.0) -> bool:
    def eq(a, b):
        return b is None or (a is not None and abs(float(a) - float(b)) < 1e-9)
    if L is not None and int(cfg.get("L", -1)) != int(L):
        return False
    if model is not None and cfg.get("model", "bosonic") != model:
        return False
    if bc is not None and cfg.get("bc", "PBC") != bc:
        return False
    # hy must NEVER use eq()'s None-means-any idiom (that's intentional for hx/model/bc
    # sweeps) — None here would silently re-admit mixing every hy cut into one curve.
    hy = 0.0 if hy is None else hy
    if not eq(cfg.get("hy", 0.0), hy):     # missing key ~ hy=0.0; never mix hy cuts
        return False
    return eq(cfg.get("hx"), hx)


def iter_matching_checkpoints(checkpoint_dir: str, *, L=None, hx=None, hy: float = 0.0,
                              model: str = "bosonic", bc: Optional[str] = None,
                              verbose: bool = True):
    """Yield ``(json_path, config, doc)`` for each checkpoint in `checkpoint_dir`
    matching ``(L, hx, hy, model, bc)`` — the shared front-end of every per-checkpoint
    sweep (FM and Rényi). `hy` matches with 1e-9 tolerance and treats a missing key
    as 0.0, defaulting to 0.0 so pre-hy directories (and hy=0 runs) keep matching
    without a flag; pass the campaign's fixed hy (e.g. 0.2) to select that cut only.

    One entry per run: prefer the final ``{name}.json``; fall back to the latest
    ``{name}.curve.json`` (+ ``{name}.ckpt.mpack``) for a run that timed out before
    writing its final artifact. Skips — with a printed reason — runs flagged
    ``diverged:true`` (the self-healing guard gave up) or whose finished ``Vscore``
    exceeds ``VSCORE_MAX`` (a guard-missed variance blow-up). Callers load the NQS
    via ``load_vstate(jp, ...)`` and do their observable-specific work.
    """
    by_base = {}
    for jp in sorted(glob.glob(os.path.join(checkpoint_dir, "*.json"))):
        if jp.endswith(".curve.json"):
            base, final = jp[:-len(".curve.json")], False
        else:
            base, final = jp[:-len(".json")], True
        if final or base not in by_base:
            by_base[base] = jp

    for jp in sorted(by_base.values()):
        try:
            with open(jp) as f:
                doc = json.load(f)
            cfg0 = doc.get("config", {})
        except (json.JSONDecodeError, KeyError):
            continue
        if not cfg0 or not _matches(cfg0, L, hx, model, bc, hy=hy):
            continue
        if doc.get("diverged"):            # self-healing guard gave up -> garbage state
            print(f"  [skip] {os.path.basename(jp)}: diverged:true — excluded "
                  f"from the sweep", flush=True)
            continue
        _vs = doc.get("observables", {}).get("Vscore")           # guard-missed blow-up
        if isinstance(_vs, (int, float)) and np.isfinite(_vs) and _vs > VSCORE_MAX:
            print(f"  [skip] {os.path.basename(jp)}: Vscore={_vs:.2e} > {VSCORE_MAX} "
                  f"— variance blow-up, excluded from the sweep", flush=True)
            continue
        if jp.endswith(".curve.json") and verbose:
            _done = doc.get("completed_steps", "?")
            print(f"  [checkpoint] {os.path.basename(jp)}: run unfinished "
                  f"({_done} steps) — using latest .ckpt.mpack weights", flush=True)
        yield jp, cfg0, doc


def fm_sweep(checkpoint_dir: str, *, sector: str = "electric", field: str = "hz",
             L: Optional[int] = None, hx: Optional[float] = None, hy: float = 0.0,
             model: str = "bosonic", bc: Optional[str] = None,
             eval_samples: int = 8192, eval_chains: Optional[int] = None,
             op_kwargs: Optional[Dict] = None,
             placement: str = "bulk", planes: Sequence[str] = ("xy", "xz", "yz"),
             plane_at: Optional[int] = None, R: Optional[int] = None,
             aspect: Optional[float] = None,
             verbose: bool = True) -> Dict[str, np.ndarray]:
    """Score every matching checkpoint in `checkpoint_dir`, sorted by `field`.

    Selects `{*.json}` whose config matches (L, hx, hy, model, bc) and sweeps the
    swept parameter `field` (default "hz"). `hy` fixes the sign-full cut (default
    0.0); it is NOT a sweepable field here — a directory holding several hy cuts
    of the same (L, hx) needs one `fm_sweep` call per hy. For each match it loads
    the NQS, builds the loop operators once, and evaluates the FM ratio plus ⟨σz⟩
    (a cheap diagonal cross-check whose susceptibility should peak at the same h_c).

    placement="bulk" (default, electric only): the largest bulk-centered square in each
    plane in `planes`, averaged over orientations (needs L>=4). placement="boundary":
    the single legacy loop from `op_kwargs` (works at any L; reproduces old curves).

    Returns a dict of equal-length arrays: field, O, Oe, mz, mz_e, name; for bulk
    placement also O_<plane>/Oe_<plane> per orientation, plus a non-array "_meta" entry
    (placement, planes, plane_at, R).

    All checkpoints in one hz sweep share the same network/sampler/operators (only
    the weights differ), so the stack and the loop operators are built **once** and
    reused: each subsequent checkpoint only swaps in its weights, which keeps JAX's
    compiled `expect` warm. A checkpoint whose structural config differs (`_struct_sig`)
    triggers a one-off rebuild rather than corrupting reuse.
    """
    op_kwargs = op_kwargs or {}
    tmpl_sig = tmpl = None       # (geo, hi, vs, pairs, mz_op, meta)
    sweep_meta: Dict[str, Any] = {}
    rows = []
    diag_by_name: Dict[str, Any] = {}                   # per-checkpoint B3 health (magnetic)
    for jp, cfg0, _doc in iter_matching_checkpoints(
            checkpoint_dir, L=L, hx=hx, hy=hy, model=model, bc=bc, verbose=verbose):
        t0 = time.perf_counter()
        sig = _struct_sig(cfg0)
        if tmpl is None or sig != tmpl_sig:            # first match, or a shape change
            _cfg, geo, hi, vs = load_vstate(jp, eval_samples=eval_samples,
                                            eval_chains=eval_chains)
            dual = bool(cfg0.get("dual_basis", False))
            pairs, sweep_meta = build_loop_operators(
                geo, hi, sector, placement=placement, planes=planes,
                plane_at=plane_at, R=R, aspect=aspect, op_kwargs=op_kwargs,
                dual=dual, model=cfg0.get("model", "bosonic"))
            # physical M_z = Σσ^z/N, built from σ^x in the dual representation
            _sig_mz = nk.operator.spin.sigmax if dual else nk.operator.spin.sigmaz
            mz_op = sum(_sig_mz(hi, i) for i in range(geo.N)) / geo.N
            tmpl_sig, tmpl = sig, (geo, hi, vs, pairs, mz_op, sweep_meta, dual)
        else:                                          # reuse: swap weights only
            geo, hi, _vs, pairs, mz_op, sweep_meta, dual = tmpl
            vs = _load_weights(_vs, jp)
        vs.reset()                                     # fresh samples for these weights
        if uses_telescoped(sector, dual):              # off-diagonal σ^x membrane
            O, Oe, per, diags = fm_ratio_avg_telescoped(
                vs, geo, pairs, chunk=cfg0.get("chunk_size"))
        elif uses_sampled_diagonal(sector, dual):      # dual membrane: ±1 products
            O, Oe, per = fm_ratio_avg_sampled(vs, geo, pairs)
            diags = None
        else:                                          # operator pairs via vs.expect
            O, Oe, per = fm_ratio_avg(vs, pairs)
            diags = None
        mz = vs.expect(mz_op)
        # dtype convention for this checkpoint: complex ansatz (sign-full h_y) vs real.
        # Nominally-real expectations (⟨M_z⟩, ⟨S⟩, ⟨W⟩, O_FM) are reported as Re after the
        # MC average; the imaginary part is a free consistency channel and should be ~0 up
        # to MC noise (pt 13). A persistently large mz_im_frac flags a sign/convention bug.
        is_complex = bool(np.iscomplexobj(np.asarray(mz.mean)))
        mz_im = float(np.imag(mz.mean))
        row = {
            "field": float(cfg0[field]), "O": O, "Oe": Oe,
            "mz": float(np.real(mz.mean)), "mz_e": float(np.real(mz.error_of_mean)),
            "mz_im": mz_im,
            "mz_im_frac": abs(mz_im) / (abs(float(np.real(mz.mean))) + 1e-12),
            "convention": ("complex ansatz; nominally-real expectations are Re(⟨·⟩) after "
                           "MC average, *_im are the discarded imaginary parts (∼0 expected)"
                           if is_complex else "real ansatz (h_y=0); expectations are real"),
            "name": cfg0.get("name", os.path.basename(jp)[:-5]),
        }
        for lbl, (Oi, Oei) in per.items():             # per-orientation cols (bulk only)
            if lbl:
                row[f"O_{lbl}"], row[f"Oe_{lbl}"] = Oi, Oei
        b3 = ""
        if diags is not None:                          # B3: worst-case tail health this point
            faces = [f for d in diags.values() for f in d["health_open"] + d["health_closed"]]
            row["b3_max_kurt"] = max((f["excess_kurtosis"] for f in faces), default=float("nan"))
            row["b3_min_ess_frac"] = min((f["ess_frac"] for f in faces), default=float("nan"))
            # pt 14: min phase coherence across all faces; if any face is incoherent the
            # membrane product's error bar (and central value) are not trustworthy here.
            row["b3_min_coherence"] = min((f["coherence"] for f in faces), default=float("nan"))
            row["phase_incoherent"] = bool(any(not f["coh_ok"] for f in faces))
            diag_by_name[row["name"]] = diags
            pc = "" if not row["phase_incoherent"] else f" PHASE-INCOHERENT(min={row['b3_min_coherence']:.2f})!"
            b3 = (f"  [B3 maxkurt={row['b3_max_kurt']:.1f} min_ess={row['b3_min_ess_frac']:.3f}"
                  f" min_coh={row['b3_min_coherence']:.2f}]{pc}")
        rows.append(row)
        if verbose:
            spread = ("  planes={" +
                      ", ".join(f"{l}:{per[l][0]:.3f}" for l in per if l) + "}"
                      if len(per) > 1 else "")
            print(f"  {rows[-1]['name']}: {field}={rows[-1]['field']:.4g}  "
                  f"O_FM={O:.4f}±{Oe:.4f}  <σz>={rows[-1]['mz']:.4f}{spread}{b3}  "
                  f"[{time.perf_counter() - t0:.1f}s]", flush=True)
    if not rows:
        raise ValueError(f"no checkpoints in {checkpoint_dir} match "
                         f"(L={L}, hx={hx}, hy={hy}, model={model}, bc={bc})")
    rows.sort(key=lambda r: r["field"])
    keys = {k for r in rows for k in r}                # magnetic adds b3_* cols on some rows
    # String metadata columns can't go into a float array. "name" was always one; the
    # complex-aware path adds "convention" (a human-readable dtype note) — both need
    # object dtype or np.array(..., float) raises ValueError on the string.
    str_keys = {"name", "convention"}
    out = {k: np.array([r.get(k, np.nan) for r in rows],
                       dtype=object if k in str_keys else float)
           for k in keys}
    out["_meta"] = sweep_meta
    if diag_by_name:
        out["_b3_health"] = diag_by_name
    return out


# =============================================================================
# Per-L analysis: logistic fit + derivative peak
# =============================================================================

def _logistic(h, a, b, h0, w):
    return a + b / (1.0 + np.exp(-(h - h0) / w))


def fit_transition(field: np.ndarray, O: np.ndarray,
                   Oe: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Locate h_c for one L: logistic fit (h_c = inflection) + derivative peak.

    The FM order parameter rises monotonically through the transition, so a
    logistic a + b/(1+e^{-(h-h0)/w}) captures it; its inflection h0 — which is
    also the peak of the analytic derivative — is the pseudo-critical h_c(L).
    A finite-difference derivative peak is returned as a model-free cross-check.

    Returns: h_c (=h0), width w, popt, a finely-sampled (h, O_fit, dO_fit) curve,
    and the finite-difference (h_mid, dOdh, h_c_fd).
    """
    from scipy.optimize import curve_fit

    field = np.asarray(field, float)
    O = np.asarray(O, float)
    Oe = None if Oe is None else np.asarray(Oe, float)
    # NaN VALUES (den-gate failures under the pooled convention: ⟨closed⟩ ≤ 0
    # → ratio undefined) carry no information — drop the points entirely
    # (2026-08-11 audit; NaNs would otherwise corrupt/abort curve_fit).
    keep = np.isfinite(O)
    if not keep.all():
        print(f"[fit_transition] dropping {int((~keep).sum())}/{len(O)} "
              f"non-finite point(s) (den-gate failures)", flush=True)
        field, O = field[keep], O[keep]
        Oe = None if Oe is None else Oe[keep]
    if len(O) < 4:                                # logistic has 4 parameters
        print(f"[fit_transition] only {len(O)} finite point(s) — no fit", flush=True)
        return {"h_c": float("nan"), "h_c_err": float("nan"),
                "width": float("nan"), "popt": None, "curve": None,
                "h_c_fd": float("nan"), "fd": (np.array([]), np.array([]))}
    p0 = [O[0], O[-1] - O[0], float(np.median(field)),
          0.1 * (field[-1] - field[0]) or 0.1]
    kw = {}
    if Oe is not None:
        # Two distinct degenerate-error cases (2026-08-10 + 2026-08-11 audits):
        # * Oe == 0.0 exactly (deep-polarized constant chain) — genuinely
        #   low-variance: floor at the smallest positive error, keeping the
        #   point maximally (but finitely) weighted.
        # * Oe non-finite (undefined jackknife, marginal ⟨closed⟩) — the LEAST
        #   trustworthy points: ceiling at the largest positive error (minimum
        #   weight), never the old smallest-positive floor that handed them
        #   MAXIMUM weight.
        s = Oe.copy()
        pos = s[np.isfinite(s) & (s > 0)]
        if len(pos):
            s = np.where(np.isfinite(s), np.where(s > 0, s, pos.min()), pos.max())
            kw = dict(sigma=s, absolute_sigma=True)
    try:
        popt, pcov = curve_fit(_logistic, field, O, p0=p0, maxfev=20000, **kw)
        h0_err = float(np.sqrt(abs(pcov[2, 2])))
    except Exception as exc:                      # fall back to derivative peak only
        popt, h0_err = None, float("nan")
        print(f"[fit_transition] logistic fit failed ({exc}); FD peak only")

    # finite-difference derivative (model-free)
    h_mid = 0.5 * (field[1:] + field[:-1])
    dOdh = np.diff(O) / np.diff(field)
    h_c_fd = float(h_mid[int(np.argmax(np.abs(dOdh)))]) if len(h_mid) else float("nan")

    out: Dict[str, Any] = {"h_c_fd": h_c_fd, "fd": (h_mid, dOdh)}
    if popt is not None:
        hh = np.linspace(field[0], field[-1], 400)
        a, b, h0, w = popt
        dO = (b / w) * np.exp(-(hh - h0) / w) / (1 + np.exp(-(hh - h0) / w)) ** 2
        out.update(h_c=float(h0), h_c_err=h0_err, width=float(w), popt=popt,
                   curve=(hh, _logistic(hh, *popt), dO))
    else:
        out.update(h_c=h_c_fd, h_c_err=float("nan"), width=float("nan"),
                   popt=None, curve=None)
    return out


def plot_fm_sweep(field, O, Oe, fit, *, sector="electric", L=None, ax=None):
    """Two-panel plot: O_FM(field) with the logistic fit, and dO/dfield with h_c.

    Reusable but import-light: matplotlib is imported here so the numerics above
    stay usable without a display.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    label = f"{sector} FM" + (f", L={L}" if L is not None else "")

    ax[0].errorbar(field, O, yerr=Oe, fmt="o", capsize=3, label="data")
    if fit.get("curve") is not None:
        hh, Ofit, _ = fit["curve"]
        ax[0].plot(hh, Ofit, "-", label="logistic fit")
    ax[0].axvline(fit["h_c"], ls="--", c="k", label=f"h_c={fit['h_c']:.3f}")
    ax[0].set(xlabel="field", ylabel="$O_{FM}$", title=label)
    ax[0].legend()

    h_mid, dOdh = fit["fd"]
    ax[1].plot(h_mid, dOdh, "s-", label="finite diff")
    if fit.get("curve") is not None:
        hh, _, dO = fit["curve"]
        ax[1].plot(hh, dO, "-", label="d(fit)")
    ax[1].axvline(fit["h_c"], ls="--", c="k")
    ax[1].axvline(fit["h_c_fd"], ls=":", c="r", label=f"FD peak={fit['h_c_fd']:.3f}")
    ax[1].set(xlabel="field", ylabel="$dO_{FM}/d$field", title="derivative")
    ax[1].legend()
    return ax


# =============================================================================
# CLI: extract one L's O_FM(field) curve + transition fit to a compact JSON.
#
# Runs on the cluster (a GPU node, where the checkpoints + NetKet live): it is
# the ONLY on-cluster analysis step. The tiny output JSON — arrays + fit only,
# no weights — is what gets pulled local for the multi-L overlay plot
# (analysis/scripts/plot_phase_diagram.py, which needs no NetKet).
#
#   python -m tc3d.fm --dir $PSCRATCH/tc_nqs/phase_hx0.2/L6 --L 6 --hx 0.2 \
#       --placement bulk --out $PSCRATCH/tc_nqs/phase_hx0.2/fm_L6_bulk.json
# =============================================================================

def extract_curve(checkpoint_dir, *, L, hx, hy=0.0, sector="electric", field="hz",
                  model="bosonic", bc="OBC", eval_samples=8192, eval_chains=None,
                  placement="bulk", planes=("xy", "xz", "yz"), plane_at=None, R=None,
                  aspect=None):
    """fm_sweep + fit_transition for one L -> a JSON-serializable dict.

    `hy` fixes the sign-full cut this curve is drawn from (default 0.0; see
    `fm_sweep`) — recorded in the output so curves at different hy are never
    accidentally overlaid downstream.
    Loop side: `R=None` → largest (L-3); `R=<int>` → fixed; `aspect=<float>` → fixed
    aspect ratio R/L (floor/ceil averaged for odd L, overrides R). `eval_chains` overrides
    n_chains at eval (small = long chains = valid error_of_mean).
    """
    res = fm_sweep(checkpoint_dir, sector=sector, field=field, L=L, hx=hx, hy=hy,
                   model=model, bc=bc, eval_samples=eval_samples, eval_chains=eval_chains,
                   placement=placement, planes=planes, plane_at=plane_at, R=R, aspect=aspect)
    fit = fit_transition(res["field"], res["O"], res["Oe"])
    meta = res.get("_meta", {})
    _R = meta.get("R")                                  # int (fixed), list (aspect), or None
    R_out = ([int(x) for x in _R] if isinstance(_R, (list, tuple))
             else None if _R is None else int(_R))
    rec = {
        "L": int(L), "hx": _num(hx), "hy": _num(hy), "sector": sector, "field_name": field,
        "bc": bc, "model": model, "eval_samples": int(eval_samples),
        "placement": meta.get("placement", placement),
        "planes": meta.get("planes", []), "plane_at": _num(meta.get("plane_at")),
        "R": R_out, "aspect": _num(meta.get("aspect")),
        # `aspect` is the *requested* value; `aspect_true` is the *realized* R/L
        # (they differ when a side is dropped/capped — e.g. L=5 at aspect 0.5 keeps
        # R=2 -> true 0.40; see Phase A A3(iv)). Judge by aspect_true, not aspect.
        "aspect_true": ([x / int(L) for x in R_out] if isinstance(R_out, list)
                        else None if R_out is None else R_out / int(L)),
        "field": res["field"].tolist(), "O": res["O"].tolist(),
        "Oe": res["Oe"].tolist(), "mz": res["mz"].tolist(),
        "mz_e": res["mz_e"].tolist(), "names": [str(x) for x in res["name"]],
        "h_c": _num(fit.get("h_c")), "h_c_err": _num(fit.get("h_c_err")),
        "h_c_fd": _num(fit.get("h_c_fd")), "width": _num(fit.get("width")),
    }
    # ParaToric-convention meta (placement="paratoric" only): every curve JSON
    # self-identifies its operator family — FSS fits must never mix families.
    for k in ("convention", "corners", "corner", "vertical"):
        if meta.get(k) is not None:
            rec[k] = meta[k]
    # Per-orientation curves (populated only for bulk placement) — the isotropy check.
    o_planes = {lbl: res[f"O_{lbl}"].tolist() for lbl in meta.get("planes", [])
                if f"O_{lbl}" in res}
    if o_planes:
        rec["O_planes"] = o_planes
        rec["Oe_planes"] = {lbl: res[f"Oe_{lbl}"].tolist() for lbl in o_planes}
    # B3 estimator health (magnetic membrane only): per-point worst-case tail metrics
    # + the full per-face detail. A blown-up kurtosis / tiny ess_frac at a point means
    # its O_FM error bar is not trustworthy (heavy-tailed σ^x amplitude ratio).
    if "b3_max_kurt" in res:
        rec["b3_max_kurt"] = res["b3_max_kurt"].tolist()
        rec["b3_min_ess_frac"] = res["b3_min_ess_frac"].tolist()
    if "_b3_health" in res:
        rec["b3_health"] = res["_b3_health"]
    hm, dodh = fit["fd"]
    rec["fd"] = {"h_mid": np.asarray(hm).tolist(), "dOdh": np.asarray(dodh).tolist()}
    if fit.get("curve") is not None:
        hh, ofit, dO = fit["curve"]
        rec["fit_curve"] = {"h": hh.tolist(), "O_fit": ofit.tolist(), "dO": dO.tolist()}
    return rec


def _num(x):
    """None-safe float (JSON can't hold numpy scalars / NaN survives as null-ish)."""
    return None if x is None else float(x)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Extract O_FM(field) + transition fit for one L.")
    p.add_argument("--dir", required=True, help="checkpoint dir ({name}.json + .mpack)")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--hx", type=float, default=None,
                   help="fix hx (electric hz-sweep filters to this cut). OMIT for an "
                        "hx-sweep (--field hx): matches ALL hx in the dir.")
    p.add_argument("--hy", type=float, default=0.0,
                   help="fix hy (sign-full cut; NOT a sweepable field). Default 0.0 "
                        "matches hy=0/missing-key runs; set e.g. 0.2 to select that "
                        "hy cut out of a dir holding several.")
    p.add_argument("--sector", default="electric", choices=["electric", "magnetic"])
    p.add_argument("--field", default="hz",
                   help="swept parameter: 'hz' (electric string) or 'hx' (magnetic membrane)")
    p.add_argument("--bc", default="OBC", choices=["OBC", "PBC"])
    p.add_argument("--model", default="bosonic", choices=["bosonic", "fermionic"])
    p.add_argument("--eval_samples", type=int, default=8192)
    p.add_argument("--eval_chains", type=int, default=None,
                   help="override n_chains at eval (default: keep the run's value). "
                        "GPU runs default to 1024 -> ~8 samples/chain, too short for a "
                        "valid autocorrelation error; set e.g. 16 for long chains.")
    p.add_argument("--placement", default="bulk",
                   choices=["bulk", "boundary", "paratoric"],
                   help="bulk: largest bulk-centered square, averaged over --planes "
                        "(electric, needs L>=4); boundary: legacy z=0 largest loop (any L); "
                        "paratoric: the frozen QMC-comparison family — stock Z-string / "
                        "corner-rule cube membrane (--R 1 for the anchor family), single "
                        "operator, no averaging")
    p.add_argument("--planes", default="xy,xz,yz",
                   help="comma-separated planes to average for bulk placement")
    p.add_argument("--plane_at", type=int, default=None,
                   help="loop plane index (default: middle layer L//2); bulk only")
    p.add_argument("--R", type=int, default=None,
                   help="bulk loop side: default None = largest (L-3, grows with L); "
                        "fix it (e.g. --R 1 = perimeter-4 plaquette) for a size-"
                        "independent order parameter. Needs 1<=R<=L-3.")
    p.add_argument("--aspect", type=float, default=None,
                   help="fixed aspect ratio R/L (e.g. 0.5 = L/2): floor/ceil sides averaged "
                        "on the same samples for odd L. Overrides --R. Out-of-bulk sides "
                        "are dropped (so L=5 at 0.5 -> R=2). The clean FSS-crossing choice.")
    p.add_argument("--out", required=True, help="output JSON path")
    a = p.parse_args(argv)
    if a.aspect is not None and a.R is not None:
        p.error("give at most one of --R and --aspect")
    if a.placement == "paratoric":
        if a.aspect is not None or a.plane_at is not None:
            p.error("--placement paratoric fixes the geometry (corner rule): "
                    "drop --aspect/--plane_at")
        if a.R is not None and a.sector != "magnetic":
            p.error("--placement paratoric: --R selects the membrane ANCHOR family "
                    "(--sector magnetic only); the Z-string is always stock geometry")
    planes = tuple(s.strip() for s in a.planes.split(",") if s.strip())
    rec = extract_curve(a.dir, L=a.L, hx=a.hx, hy=a.hy, sector=a.sector, field=a.field,
                        model=a.model, bc=a.bc, eval_samples=a.eval_samples,
                        eval_chains=a.eval_chains, placement=a.placement, planes=planes,
                        plane_at=a.plane_at, R=a.R, aspect=a.aspect)
    with open(a.out, "w") as f:
        # NaN is reachable in O/Oe under the den<=0 -> NaN convention; keep the
        # JSON RFC-parseable (jq corrupts bare NaN/Infinity tokens silently)
        json.dump(_json_nonfinite_safe(rec), f, indent=2)
    asp = ("" if rec.get("aspect") is None
           else f" aspect={rec['aspect']}(true {rec['aspect_true']})")
    print(f"[fm] L={a.L} hx={a.hx} hy={a.hy} placement={rec['placement']} R={rec['R']}{asp}"
          f": {len(rec['field'])} points, "
          f"h_c={rec['h_c']}  h_c_fd={rec['h_c_fd']}  ->  {a.out}")


if __name__ == "__main__":
    main()
