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


def dressed_string(geom, stabs, z_edges):
    """Dress a sigma^z string so it commutes with every decorated plaquette.

    Solves for a sigma^x support `s` with  parity(|s ∩ boundary_p|) ==
    parity(|z_edges ∩ decoration_p|)  for each plaquette p, so that
    Z(z_edges)·X(s) commutes with all tilde B_p (vertex stars are automatic:
    sigma^x commutes with the all-sigma^x stars).

    Returns (z_edges, x_edges, flux_plaqs):
      - closed loop  -> flux_plaqs == []  (a conserved Wilson loop),
      - open string  -> flux_plaqs lists the endpoint plaquettes it still
        anticommutes with (the unavoidable flux of the charge+flux fermion).
    """
    zmask = _mask(z_edges)
    rows = [_mask(zb) for zb, xb, _ in stabs]
    targets = [bin(zmask & _mask(xb)).count("1") & 1 for zb, xb, _ in stabs]
    s = _gf2_solve(rows, targets, geom.N)
    if zmask & s:
        raise ValueError("dressing overlaps the sigma^z line (would give sigma^y)")
    x_edges = [i for i in range(geom.N) if (s >> i) & 1]
    flux_plaqs = [p for p, (zb, xb, _) in enumerate(stabs)
                  if (bin(zmask & _mask(xb)).count("1") + bin(s & _mask(zb)).count("1")) & 1]
    return z_edges, x_edges, flux_plaqs
