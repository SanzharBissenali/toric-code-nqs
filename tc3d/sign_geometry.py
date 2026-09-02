"""Exact fermionic h=0 sign as a lattice cup product: s = Q(a) = int a cup delta a.

Port of the verified recipe in `notes/fermionic_sign_geometry.md` (theory) and
`notes/fermionic_sign_geometry_numerics.md` SS4/SS5 (the exact all-L, PBC-and-OBC
formula, brute-force verified 3000/3000 on every geometry checked there). Reference
implementation: scratchpad `evenL/{common,geom,cup,gaugefix,final}.py`.

Setup. Primal cubic lattice, qubits on edges; `b in C^1(F2)` is the flipped-edge
set of a raw spin configuration (bit 1 = spin down = x_i < 0, matching
`tc3d.exact_diag`/`tc3d.sign_frame`'s convention). `v = (1,1,1)`. Every decorated
plaquette p's two sigma^x partners e+(p), e-(p) = center(p) +/- v/2 lie on the SAME
(1,1,1) body-diagonal line (they differ by exactly v) -- so a BULK pair-move flips
two edges of one line, conserving that line's parity n_l(b) = sum_{e in l} b_e. At
a PBC line (a closed cycle) this holds for every pair-move, so the 3L^2 line
indicators 1_l span null(X) (X = plaquette-pair edge incidence) exactly: X @ 1_l
= 0, because a pair's two edges are either both on l or both off it, never split.
At OBC a line is a finite, boundary-truncated segment, and a pair-move sitting at
that boundary has ALREADY lost its partner (`fermionic_plaquettes`'s truncation
rule keeps only the surviving corner edge) -- it flips a SINGLE edge, not two, so
that line's indicator is NOT generally in null(X) by itself. `Phi` is therefore
computed as the actual null space of X (`_gf2_nullspace`, Gaussian elimination;
dimension 3L(L-1) at OBC, matching the numerics note) rather than assembled from
the geometric line partition -- `lines`/`P_lift` (the diagonal-line structure used
for the LIFT step below) stay geometric, only the detector basis `Phi` is solved
for algebraically.

Recipe (SS4 of the numerics note), all vectorized as GF(2) matmuls:
  1. detect:   y = Phi @ b            (line parities; Phi = null(X), by elimination)
  2. gauge-fix b -> b' = b + (star correction) so that Phi @ b' = 0 whenever
     that is reachable by the model's vertex stars (a FIXED linear map of y,
     precomputed once by Gaussian elimination on Phi @ Sv^T; see `_solve_map`)
  3. lift:     a = P_lift @ b'        (suffix-XOR along +v: a(e) = sum_{k>=0}
     b'(e + k v), anchored at the +v end of each line -- a seam for PBC cycles,
     the physical +v boundary for OBC segments)
  4. s = Q(a) = sum_p (I @ a)_p * a(e+(p))  mod 2,  return (-1)^s

Off-support behaviour (a config whose line-parity "syndrome" cannot be zeroed by
any combination of the model's stars -- e.g. an arbitrary/non-physical bitstring
fed to the sampler before a flux penalty projects it down). The gauge-fix map is
LINEAR (built once from Phi @ Sv^T by fixed-pivot Gaussian elimination with free
variables set to 0), so it is evaluated identically whether or not the target is
reachable; when it is not, the leftover mismatch (the "syndrome", see
`CupSign.syndrome`) survives into b' and is picked up, per line, by the edge at
that line's -v end (a(e) there sums the WHOLE line, so it inherits the residual --
matching numerics SS5's "the defect lands on the single-X boundary faces at the
-v end of each line"). This makes `sign()` a total, deterministic function of any
raw N-bit configuration, with the residual pinned to a fixed geometric location.

Never build 2^N objects here: everything is dense GF(2) linear algebra on N, NP
sized arrays (<= 3 L^3 <= 375 for L <= 5), independent of the 2^N configuration
space.
"""

from __future__ import annotations

import numpy as np

from tc3d.fermionic_decoration import _E, fermionic_plaquettes

__all__ = ["CupSign", "make_cup_sign", "orbit_sign_from_applications",
           "random_orbit_states"]

_MAX_N = 3 * 5 ** 3     # L <= 5 cap (see module docstring / CLAUDE.md)


# =============================================================================
# GF(2) dense linear algebra (small: N, NP <= 375 here)
# =============================================================================

def _gf2_rref(A):
    """Row-reduced echelon form (A % 2) -> (R[:rank], pivot_columns)."""
    A = (A % 2).astype(np.uint8).copy()
    m, n = A.shape
    r = 0
    piv = []
    for c in range(n):
        rows = np.nonzero(A[r:, c])[0]
        if rows.size == 0:
            continue
        p = rows[0] + r
        A[[r, p]] = A[[p, r]]
        sel = np.nonzero(A[:, c])[0]
        sel = sel[sel != r]
        A[sel] ^= A[r]
        piv.append(c)
        r += 1
        if r == m:
            break
    return A[:r], piv


def _gf2_nullspace(A):
    """Basis (rows) of {z : A z = 0 mod 2}, A: (m, n)."""
    R, piv = _gf2_rref(A)
    n = A.shape[1]
    free = [j for j in range(n) if j not in piv]
    B = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        B[i, f] = 1
        for k, c in enumerate(piv):
            B[i, c] = R[k, f]
    return B


def _solve_map(A):
    """Fixed 0/1 matrix Z (k x m) linearising "solve A z = y for z" (A: m x k).

    Gaussian elimination on A picks pivot columns and, via the SAME row
    operations applied to an identity matrix, records a fixed linear map
    y -> z: pivot components of z are read off the transformed identity rows,
    free (non-pivot) components are 0. Because the map is built purely from row
    XORs (no data-dependent branching on y itself), z = Z @ y is valid -- and
    deterministic -- for EVERY y, not only ones for which A z = y is solvable.
    """
    A = (A % 2).astype(np.uint8).copy()
    m, k = A.shape
    T = np.eye(m, dtype=np.uint8)          # tracks the row-operation transform
    piv = []
    r = 0
    for c in range(k):
        rows = np.nonzero(A[r:, c])[0]
        if rows.size == 0:
            continue
        p = rows[0] + r
        A[[r, p]] = A[[p, r]]
        T[[r, p]] = T[[p, r]]
        sel = np.nonzero(A[:, c])[0]
        sel = sel[sel != r]
        A[sel] ^= A[r]
        T[sel] ^= T[r]
        piv.append(c)
        r += 1
        if r == m:
            break
    Z = np.zeros((k, m), dtype=np.uint8)
    for i, c in enumerate(piv):
        Z[c] = T[i]
    return Z


# =============================================================================
# Geometry precompute
# =============================================================================

def _plaquette_geometry(geom, stabs):
    """(I, X, eplus, eminus): Z/X-incidence + the +/-v push-off edge per plaquette.

    I, X are (NP, N) 0/1 (Z-support, X-pair-move edges) built from `stabs`.
    eplus[p]/eminus[p] are the edge indices of e+(p) = ctr(p) + v/2 and
    e-(p) = ctr(p) - v/2 (v the body diagonal of the plaquette's own normal +
    in-plane axes), -1 where that corner falls outside an OBC box. Recomputed
    from scratch (not read back from `stabs`' x_edges, which drop OBC's missing
    partner and so can't tell + from - once only one survives); the loop order
    mirrors `fermionic_plaquettes` exactly, checked by the assert below.
    """
    from tc3d.fermionic_decoration import _idx

    NP, N = len(stabs), geom.N
    I = np.zeros((NP, N), dtype=np.uint8)
    X = np.zeros((NP, N), dtype=np.uint8)
    for p, (z, x, _) in enumerate(stabs):
        I[p, z] = 1
        X[p, x] = 1

    eplus = np.full(NP, -1, dtype=np.int64)
    eminus = np.full(NP, -1, dtype=np.int64)
    k = 0
    for c in range(3):
        a, b = (d for d in range(3) if d != c)
        for ix in range(geom.Lx):
            for iy in range(geom.Ly):
                for iz in range(geom.Lz):
                    ctr = np.array([ix, iy, iz], float) + 0.5 * _E[a] + 0.5 * _E[b]
                    z_edges = [_idx(geom, ctr + s * 0.5 * _E[ax])
                               for ax in (a, b) for s in (+1, -1)]
                    if geom.bc == "OBC" and -1 in z_edges:
                        continue
                    assert set(z_edges) == set(stabs[k][0]), \
                        "plaquette loop order diverged from fermionic_plaquettes"
                    diag = 0.5 * (_E[a] + _E[b] + _E[c])
                    eplus[k] = _idx(geom, ctr + diag)
                    eminus[k] = _idx(geom, ctr - diag)
                    k += 1
    assert k == NP
    return I, X, eplus, eminus


def _edge_axis_site(geom):
    """Per-edge (axis, integer base-vertex coordinate) from `geom.arr_coord`."""
    N = geom.N
    eaxis = np.zeros(N, dtype=np.int8)
    esite = np.zeros((N, 3), dtype=np.int64)
    for e in range(N):
        c2 = np.round(2 * np.asarray(geom.arr_coord[e])).astype(np.int64)
        ax = int(np.nonzero(c2 % 2)[0][0])
        s = c2.copy()
        s[ax] -= 1
        eaxis[e] = ax
        esite[e] = s // 2
    return eaxis, esite


def _diag_lines(geom, eaxis, esite):
    """Partition edges into (1,1,1)-direction lines (PBC: cycles; OBC: segments).

    Each line is a list of edge indices ordered along +v = (1,1,1); adjacent
    entries differ by one step of v (wrapped at PBC, truncated at OBC box
    edges). Every edge belongs to exactly one line -- used for the lift step
    (`_lift_matrix`). NOT the same as null(X) at OBC (see module docstring):
    a boundary line's own indicator need not be annihilated by X there, since
    a boundary pair-move can flip just one of its edges.
    """
    N = geom.N
    bc = geom.bc
    L = (geom.Lx, geom.Ly, geom.Lz)
    eidx = {(int(eaxis[e]), tuple(int(t) for t in esite[e])): e for e in range(N)}
    nxt = np.full(N, -1, dtype=np.int64)
    prv = np.full(N, -1, dtype=np.int64)
    for e in range(N):
        ax = int(eaxis[e])
        s = esite[e]
        for sgn, arr in ((+1, nxt), (-1, prv)):
            t = s + sgn
            if bc == "PBC":
                key = (ax, tuple(int(t[d] % L[d]) for d in range(3)))
            else:
                key = (ax, tuple(int(v) for v in t))
            arr[e] = eidx.get(key, -1)

    lines = []
    seen = set()
    for e in range(N):
        if e in seen:
            continue
        st = e
        while prv[st] != -1 and prv[st] not in (e,) and prv[st] not in seen:
            st = prv[st]
            if st == e:
                break
        ln, cur = [], st
        while cur != -1 and cur not in seen:
            ln.append(int(cur))
            seen.add(int(cur))
            cur = nxt[cur]
        lines.append(ln)
    assert sum(len(ln) for ln in lines) == N
    return lines


def _lift_matrix(N, lines):
    """P_lift (N,N): a = P_lift @ b', a(e) = sum_{k>=0} b'(e + k v) to the line's
    +v end (the "seam"/boundary anchor of SS4)."""
    P = np.zeros((N, N), dtype=np.uint8)
    for ln in lines:
        for i, e in enumerate(ln):
            P[e, ln[i:]] = 1
    return P


def _vertex_star_matrix(geom):
    """(NV, N) 0/1: all vertex stars A_v (edges incident to v, -1 padding skipped)."""
    N = geom.N
    S = np.zeros((len(geom.vertex_all), N), dtype=np.uint8)
    for v, es in enumerate(geom.vertex_all):
        for e in es:
            if e != -1:
                S[v, e] = 1
    return S


# =============================================================================
# CupSign
# =============================================================================

class CupSign:
    """Precomputed cup-product sign map for one geometry.

    All O(1)-per-config work is a GF(2) matmul against matrices built once here
    (edge count N and plaquette count NP scale as O(L^3), never 2^N).
    """

    def __init__(self, geom, stabs=None):
        if geom.N > _MAX_N:
            raise ValueError(f"CupSign: N={geom.N} exceeds the dense-GF(2) cap "
                              f"{_MAX_N} (L<=5) -- see CLAUDE.md working rules")
        stabs = fermionic_plaquettes(geom) if stabs is None else stabs
        self.geom = geom
        self.N = geom.N
        self.NP = len(stabs)

        I, X, eplus, eminus = _plaquette_geometry(geom, stabs)
        self.I, self.X = I, X
        self.eplus, self.eminus = eplus, eminus
        self._has_eplus = (eplus >= 0).astype(np.int64)
        self._eplus_safe = np.where(eplus >= 0, eplus, 0)
        self.M = ((I.astype(np.int64) @ X.T.astype(np.int64)) % 2).astype(np.uint8)

        eaxis, esite = _edge_axis_site(geom)
        self.lines = _diag_lines(geom, eaxis, esite)
        self.Phi = _gf2_nullspace(self.X)                                 # detectors
        self.P_lift = _lift_matrix(self.N, self.lines)

        self.stars_all = _vertex_star_matrix(geom)
        comm = (I.astype(np.int64) @ self.stars_all.T.astype(np.int64)) % 2
        keep = np.nonzero(comm.sum(axis=0) == 0)[0]
        self.Sv = self.stars_all[keep]                                    # (n_keep, N)

        A = (self.Phi.astype(np.int64) @ self.Sv.T.astype(np.int64)) % 2  # (n_line, n_keep)
        self.Zcoef = _solve_map(A)                                        # (n_keep, n_line)

    # -- internal pipeline --------------------------------------------------

    def _bits(self, configs):
        x = np.asarray(configs)
        lead, flat = x.shape[:-1], x.reshape(-1, x.shape[-1])
        return lead, (flat < 0).astype(np.int64)

    def _gauge_fix(self, b):
        """b (B,N) int64 0/1 -> b' (B,N) with line parities zeroed wherever the
        model's stars can reach the target (see module docstring for the
        off-support residual convention)."""
        y = (b @ self.Phi.T.astype(np.int64)) % 2                 # (B, n_line)
        z = (y @ self.Zcoef.T.astype(np.int64)) % 2                # (B, n_keep)
        corr = (z @ self.Sv.astype(np.int64)) % 2                  # (B, N)
        return (b + corr) % 2

    def _Q(self, u):
        """Q(u) = sum_p (I u)_p * u(e+(p))  mod 2,  u: (B,N) int64 0/1."""
        du = (u @ self.I.T.astype(np.int64)) % 2                   # (B, NP)
        pick = u[:, self._eplus_safe] * self._has_eplus[None, :]   # (B, NP)
        return (du * pick).sum(axis=1) % 2

    # -- public API -----------------------------------------------------

    def sign(self, configs):
        """configs: (...,N) array of +-1 spins (bit = x_i<0). Returns (-1)^Q(a)
        as a float array of shape (...,)."""
        lead, b = self._bits(configs)
        bp = self._gauge_fix(b)
        a = (bp @ self.P_lift.T.astype(np.int64)) % 2
        s = self._Q(a)
        return (1.0 - 2.0 * s).reshape(lead)

    def line_parities(self, configs):
        """Raw detector readout Phi @ b (no gauge fix), shape (..., n_lines)."""
        lead, b = self._bits(configs)
        y = (b @ self.Phi.T.astype(np.int64)) % 2
        return y.reshape(lead + (self.Phi.shape[0],))

    def syndrome(self, configs):
        """Line parities of the gauge-FIXED config: the residual that no
        combination of the model's stars can zero -- a fixed GF(2) projection
        of `line_parities`, identically 0 on the physical (star+pair-move)
        orbit and nonzero exactly where the recipe's off-support convention
        (see module docstring) pins a residual defect."""
        lead, b = self._bits(configs)
        bp = self._gauge_fix(b)
        r = (bp @ self.Phi.T.astype(np.int64)) % 2
        return r.reshape(lead + (self.Phi.shape[0],))


def make_cup_sign(geom, stabs=None):
    return CupSign(geom, stabs)


# =============================================================================
# Exact reference (test-only): sign in terms of PAIR-MOVE APPLICATION variables
# =============================================================================

def orbit_sign_from_applications(cs, x):
    """Exact sign bit s(x) = x^T triu(M,1) x mod 2 (0/1), M = (I X^T) mod 2.

    `x` is a (...,NP) 0/1 "application vector" (which decorated plaquettes B~_p
    were applied to build the state from the all-up vacuum) -- the ground-truth
    sign used only to generate test data, independent of the cup-product
    machinery above.
    """
    U = np.triu(cs.M.astype(np.int64), 1)
    x = np.asarray(x)
    lead, flat = x.shape[:-1], x.reshape(-1, x.shape[-1]).astype(np.int64)
    s = np.einsum("ij,jk,ik->i", flat, U, flat) % 2
    return s.reshape(lead).astype(np.uint8)


def random_orbit_states(cs, rng, n):
    """n random physical states: b = X^T x + Sv^T y (random pair moves x, random
    star gauge y), paired with the exact sign s(x). Returns (configs, s) with
    configs as +-1 float arrays (n, N) and s as 0/1 uint8 (n,)."""
    n_keep = cs.Sv.shape[0]
    xs = rng.integers(0, 2, size=(n, cs.NP)).astype(np.int64)
    b = (xs @ cs.X.astype(np.int64)) % 2
    if n_keep:
        ys = rng.integers(0, 2, size=(n, n_keep)).astype(np.int64)
        b = (b + ys @ cs.Sv.astype(np.int64)) % 2
    s = orbit_sign_from_applications(cs, xs)
    configs = 1.0 - 2.0 * b.astype(np.float64)
    return configs, s
