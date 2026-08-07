"""
3D fermionic toric code: bosonic plaquettes decorated with two sigma^x.

Each plaquette stabilizer is
    B~_p = (prod_{e in dp} sigma^z_e) * sigma^x_{e+} * sigma^x_{e-}
where e+ and e- are the perpendicular ("transverse") corner edges at the
(+a,+b) corner on the +perp side and the (-a,-b) corner on the -perp side
(a body diagonal; a, b are the in-plane axes, c the plaquette normal).  The
same pattern is used for all three orientations; vertex stars A_v are unchanged.

This is the minimal decoration giving a valid commuting-stabilizer model
(verify_xz_commutation -> 0 violations at L=2,3 PBC; the pattern is translation
invariant, so it holds for all L).  It replaces the Wang-Levin 10-edge dressing
of PRL 113, 080403 (2014) with two sigma^x per plaquette.
"""

from __future__ import annotations

import numpy as np

_E = np.eye(3)


def _idx(geom, coord) -> int:
    """Qubit index at edge-midpoint `coord`, PBC-wrapped (2x-integer keys)."""
    L = (geom.Lx, geom.Ly, geom.Lz)
    key = tuple(int(round(2 * coord[d])) % (2 * L[d]) for d in range(3))
    return int(geom._coord_to_idx[key])


def _mask(qubits) -> int:
    """Bitmask over a list of qubit indices (skips -1 padding)."""
    m = 0
    for q in qubits:
        if q != -1:
            m |= 1 << int(q)
    return m


def fermionic_plaquettes(geom, J: float = 1.0):
    """Decorated plaquette stabilizers as (z_edges, x_edges, coef) triples.

    Drop-in for hamiltonian_linop(geom, ..., xz_stabs=fermionic_plaquettes(geom)).
    Per plaquette: the 4 sigma^z boundary edges plus two sigma^x on the corner
    edges at ctr +/- 0.5*(e_a + e_b + e_c).
    """
    out = []
    for c in range(3):                                   # plaquette normal axis
        a, b = (d for d in range(3) if d != c)           # in-plane axes
        for ix in range(geom.Lx):
            for iy in range(geom.Ly):
                for iz in range(geom.Lz):
                    ctr = np.array([ix, iy, iz], float) + 0.5 * _E[a] + 0.5 * _E[b]
                    z_edges = [_idx(geom, ctr + s * 0.5 * _E[ax])
                               for ax in (a, b) for s in (+1, -1)]
                    diag = 0.5 * (_E[a] + _E[b] + _E[c])
                    x_edges = [_idx(geom, ctr + diag), _idx(geom, ctr - diag)]
                    out.append((z_edges, x_edges, -float(J)))
    return out


def verify_xz_commutation(stabs, vertex_all) -> dict:
    """Pairwise commutation check on (decorated plaquettes) + (vertex stars).

    Returns {"ok": bool, "violations": list, "n_stabilizers": int}.
    """
    entries = [(_mask(z), _mask(x)) for z, x, _ in stabs]
    entries += [(0, _mask(v)) for v in vertex_all]

    viol = []
    n = len(entries)
    for i in range(n):
        z1, x1 = entries[i]
        for j in range(i + 1, n):
            z2, x2 = entries[j]
            if (bin(z1 & x2).count("1") + bin(z2 & x1).count("1")) & 1:
                viol.append((i, j))
    return {"ok": not viol, "violations": viol, "n_stabilizers": n}


# ---------------------------------------------------------------------------
# Dressed Wilson loop / Fredenhagen-Marcu string
# ---------------------------------------------------------------------------
def _gf2_solve(rows, targets, ncols) -> int:
    """Best-effort GF(2) solve of  popcount(rows[i] & s) == targets[i].

    `rows` are column-bitmasks (one per equation).  Returns a particular
    solution `s` (a bitmask) of the *consistent* subsystem; equations that
    can't be satisfied are left as a residual (see `dressed_string`).
    """
    R = len(rows)
    aug = [rows[i] | (int(targets[i]) << ncols) for i in range(R)]
    pivot_of = {}          # pivot column -> reduced-row index
    r = 0
    for col in range(ncols):
        sel = next((k for k in range(r, R) if (aug[k] >> col) & 1), None)
        if sel is None:
            continue
        aug[r], aug[sel] = aug[sel], aug[r]
        for k in range(R):
            if k != r and (aug[k] >> col) & 1:
                aug[k] ^= aug[r]
        pivot_of[col] = r
        r += 1
    s = 0
    for col, row in pivot_of.items():
        if (aug[row] >> ncols) & 1:   # free vars = 0, pivot var = its target bit
            s |= 1 << col
    return s


def _flux_of(stabs, zmask, s):
    """Plaquettes p with which Z(zmask)·X(s) anticommutes (the flux residual)."""
    return [p for p, (zb, xb, _) in enumerate(stabs)
            if (bin(zmask & _mask(xb)).count("1") + bin(s & _mask(zb)).count("1")) & 1]


def _vertex_edges(geom, v) -> int:
    """Bitmask of the 6 edges incident to the integer vertex `v` (PBC-wrapped)."""
    return _mask([_idx(geom, np.asarray(v, float) + sgn * 0.5 * _E[ax])
                  for ax in range(3) for sgn in (+1, -1)])


def _string_endpoints(geom, z_edges):
    """Odd-incidence vertices of a sigma^z edge set — the open-string endpoints
    ([] for a closed loop). Integer coords, PBC-wrapped."""
    L = (geom.Lx, geom.Ly, geom.Lz)
    par = {}
    for q in z_edges:
        m = np.asarray(geom.arr_coord[int(q)], float)
        ax = next(d for d in range(3) if int(round(2 * m[d])) % 2)
        for sgn in (+1, -1):
            key = tuple(int(round((m + sgn * 0.5 * _E[ax])[d])) % L[d]
                        for d in range(3))
            par[key] = par.get(key, 0) ^ 1
    return [k for k, odd in par.items() if odd]


def _plaq_index(geom, normal: int, cell) -> int:
    """Index into the canonical `fermionic_plaquettes` order: the plaquette with
    normal axis `normal` whose center is cell + 0.5(e_a + e_b), PBC-wrapped."""
    L = (geom.Lx, geom.Ly, geom.Lz)
    i, j, k = (int(c) % L[d] for d, c in enumerate(cell))
    return ((normal * L[0] + i) * L[1] + j) * L[2] + k


def _staircase_faces(geom, c1, c2):
    """Dual-lattice staircase from cube `c1` to cube `c2`: the plaquettes crossed.

    A plaquette is a dual edge between its two adjacent cubes, so this is a
    minimal dual path pairing two frustrated cubes. PBC-shortest signed
    displacement per axis, canonical x→y→z order — deterministic.
    """
    L = (geom.Lx, geom.Ly, geom.Lz)
    faces, cur = [], list(c1)
    for ax in range(3):
        d = (c2[ax] - cur[ax] + L[ax] // 2) % L[ax] - L[ax] // 2   # in [-L/2, L/2)
        step = 1 if d > 0 else -1
        for _ in range(abs(d)):
            cell = list(cur)
            cell[ax] = cur[ax] + (1 if step > 0 else 0)   # shared face's layer
            faces.append(_plaq_index(geom, ax, cell))
            cur[ax] += step
    return faces


def _localize_open_flux(geom, stabs, z_edges, zmask, rows, targets, s, flux):
    """Endpoint-localize an open string's flux residual.

    The violated-equation set of the inconsistent GF(2) system is only defined
    up to Im(M): the raw particular solution usually leaves a delocalized
    residual (solver artifact). Physically each string endpoint frustrates a
    body-diagonal PAIR of cubes — (v, v-(1,1,1)) — the dipole imprint of the
    two-edge body-diagonal decoration. The canonical residual is a shortest
    dual-lattice pairing of those four cubes; candidates in order of preference:
      1. within-endpoint pairing (3+3 staircase) — a localized flux cluster per
         endpoint, size/separation-independent: the right BFFM string operator;
      2./3. cross-endpoint pairings — win only via wrap shortcuts at small L.
    Which pairing is consistent is decided by the string's homology class, so
    each candidate is verified by an exact line-free re-solve (no sigma^y); the
    first consistent one wins. Falls back to the raw residual when none fits
    (e.g. L=2 PBC, where no weight-2 residual exists at all).
    """
    ends = _string_endpoints(geom, z_edges)
    if len(ends) != 2:
        return s, flux
    (a1, a2), (b1, b2) = [(tuple(v), tuple(c - 1 for c in v)) for v in ends]
    pairings = [((a1, a2), (b1, b2)),        # within-endpoint (localized)
                ((a1, b1), (a2, b2)),        # cross, + with +
                ((a1, b2), (a2, b1))]        # cross, + with -
    for pairing in pairings:
        r: set = set()
        for c1, c2 in pairing:
            r ^= set(_staircase_faces(geom, c1, c2))
        if sorted(r) == flux:
            return s, flux                    # raw residual already canonical
        t2 = list(targets)
        for p in r:
            t2[p] ^= 1
        s2 = _gf2_solve([rw & ~zmask for rw in rows], t2, geom.N)
        if _flux_of(stabs, zmask, s2) == sorted(r):
            return s2, sorted(r)
    return s, flux


def dressed_string(geom, stabs, z_edges):
    """Dress a sigma^z string so it commutes with every decorated plaquette.

    Solves for a sigma^x support `s` with  parity(|s ∩ boundary_p|) ==
    parity(|z_edges ∩ decoration_p|)  for each plaquette p, so that
    Z(z_edges)·X(s) commutes with all tilde B_p (vertex stars are automatic:
    sigma^x commutes with the all-sigma^x stars).

    Returns (z_edges, x_edges, flux_plaqs):
      - closed loop  -> flux_plaqs == []  (a conserved Wilson loop),
      - open string  -> flux_plaqs = the endpoint-localized flux residual it
        still anticommutes with — the unavoidable flux of the charge+flux
        fermion (each endpoint frustrates a body-diagonal cube dipole; see
        `_localize_open_flux` for the canonical pairing).
    The dressing never overlaps the z-line (that would be sigma^y, not Z·X).
    """
    zmask = _mask(z_edges)
    rows = [_mask(zb) for zb, xb, _ in stabs]
    targets = [bin(zmask & _mask(xb)).count("1") & 1 for zb, xb, _ in stabs]
    s = _gf2_solve(rows, targets, geom.N)
    if zmask & s:
        # The canonical particular solution overlaps the line. Re-solve with the
        # line's columns excluded — s is then forced to 0 there, picking a
        # disjoint representative of the same coset (for a contractible loop one
        # always exists: the X-part of the enclosed prod B~_p). Only reachable
        # when the first solve overlaps, so previously-valid dressings are
        # unchanged.
        s = _gf2_solve([r & ~zmask for r in rows], targets, geom.N)
    flux_plaqs = _flux_of(stabs, zmask, s)
    if flux_plaqs:
        s, flux_plaqs = _localize_open_flux(geom, stabs, z_edges, zmask, rows,
                                            targets, s, flux_plaqs)
    if zmask & s:
        raise ValueError("dressing overlaps the sigma^z line (would give sigma^y)")
    x_edges = [i for i in range(geom.N) if (s >> i) & 1]
    return z_edges, x_edges, flux_plaqs
