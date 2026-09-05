"""Per-configuration ("on the fly") sign heads: cup | linear | vote | pt2.

`analysis/scripts/sign_fidelity_ftc.py` defines these decoders exhaustively over
the 2^N computational basis (it needs the full ED vector anyway). This module is
the SAME semantics evaluated row by row, so a sign-framed run can use them at any
L without a 2^N lookup table -- `--sign_frame {cup,linear,vote,pt2}`.

The four heads
--------------
`cup`      the exact fermionic h=0 sign, `tc3d.sign_geometry.CupSign.sign`
           (gauge fix -> (1,1,1) suffix lift -> cup product Q).
`linear`   GF(2)-linear decoder: push the config back onto the h=0 support with
           ONE fixed representative edge per lit line class (lowest edge index)
           and read the base sign there.
`vote`     majority of the base sign over ALL minimal recoveries (one edge per
           lit class, every combination); an exact tie falls back to `linear`.
`pt2`      leading-order perturbation theory with exact denominators
           DeltaE(A) = 2J * #{p : |dp cap A| odd}: the sign of
           sum_eps (-1)^{q(sigma+eps)} D(eps) over the minimal recoveries; when
           that vanishes exactly the next order ((k+1)-flip subsets carrying the
           same coset label) decides; still zero -> `linear`.
The heads are h_x / h_z / J independent (each config is decided at its own
leading order, where h_x^k factors out of the sum) -- hence one geometry
precompute serves a whole field sweep.

Base sign
---------
The three recovered heads read the base sign only at ON-SUPPORT configurations
(sigma XOR eps carries coset label 0 by construction), where `CupSign` and the
frozen analytic token-quadratic form (`sign_frame.anaC_theta`) provably coincide
-- checked bit for bit on all 2^12 / 2^20 configs of 2x2x2 / 2x2x3 OBC in
`tests/test_sign_decoders.py`. That is why this module reproduces the
anaC-based reference tables exactly while using the cup-product base (which is
the one verified exact at every L, and needs no NP^3 RREF).

The base sign is a GF(2) quadratic form q(b) = b^T K b (mod 2) on the flipped-edge
bit vector b (bit 1 = spin down): the gauge fix and the suffix lift are fixed
LINEAR maps M and Q(u) = sum_p (I u)_p u(e+(p)) = u^T A u, so K = M^T A M. That
turns the recovery arithmetic into one matmul -- with Ksym = K + K^T,

    q(b XOR e) = q(b) + sum_i l_b(c_i) + sum_{i<j} Ksym[c_i, c_j]   (mod 2)
    l_b(c)     = K[c, c] + (b Ksym)[c]

for a recovery e = XOR_i e_{c_i}, so one (B, N) x (N, N) product per block gives
every candidate edge's contribution for every row. `cup` therefore needs ONE
float32 GEMM plus a row-wise dot -- not `CupSign.sign`'s four N x N int64
products, which it reproduces exactly (regression-tested).

Arithmetic. Every quantity here is a small integer, so all of it runs in float32
GEMMs (exact below 2^24) with the mod-2 reduction pushed to the LAST step:
q(b) = sum_i b_i (b K)_i is reduced once per ROW rather than once per (row,
column), which is what made the first implementation ~25x slower than this one
(a `% 2` on a (B, N) float64 array costs more than the GEMM that produced it).

Recovery sums as tensor contractions
------------------------------------
The first-order sum over the |C_0| x ... x |C_{m-1}| minimal recoveries is
multilinear in the per-class sign vectors: with u_k(a) = (-1)^{l_b(C_k[a])},

    sum_eps (-1)^{q(sigma+eps)} w(eps)
        = (-1)^{q(b)} sum_{a_0..a_{m-1}} prod_k u_k(a_k) * W[a_0,...,a_{m-1}]

where W is a GEOMETRY-ONLY tensor: (-1)^{sum_{i<j} Ksym[C_i[a_i], C_j[a_j]]}
times the path weight D(eps) (identically 1 for `vote`). `_contract` evaluates
that as one BLAS GEMM plus m-1 broadcast reductions -- O(rows x n_recoveries)
work at GEMM speed instead of a Python loop over recoveries.

For `vote` the tensor factorizes further whenever two classes never couple
through Ksym, so the sum splits over the connected components of the lit classes
(`_components`) and only each component is contracted.

`pt2` weights are exact rationals (D = sum over orderings of prod 1/DeltaE, and
DeltaE = 2J * #plaquettes is rational), so they are scaled by the LCM of their
denominators to INTEGERS. The contraction is then exact integer arithmetic and
the "first order vanishes exactly" test is order-independent and exact -- no
float-cancellation ambiguity. (`_MAX_EXACT` guards the float mantissa.)

Lit line classes at OBC
-----------------------
A "lit line class" is NOT in general a (1,1,1) diagonal line. It is defined
exactly as in `sign_fidelity_ftc.support_structure`: the h=0 support
W = span(stars, x-pairs) has W^perp spanned by the Z-strings of
`flux_constraint_masks`, so the coset label of a config is its parity against an
independent detector set, and edges group into classes by the label of their own
single flip. At PBC that partition IS the diagonal-line partition (3L^2 classes
of L edges). At OBC the pair-move truncation removes most detectors -- an L^3 OBC
box has exactly L-1, so an OBC class is a large union of edges spread over the
lattice, not a line:

    L      2      3        4            5              6
    N     12     54      144          300            540
    lit    1      2        3            4              5
    sizes [6]  [12,12] [18,18,24] [24,24,36,36] [30,30,48,48,54]
    prod   6    144     7 776      746 496      111 974 400   <- recoveries/row

The candidate edges of a lit class are ALL its edges. `linear`/`vote`/`pt2` are
OBC-only for exactly this reason: PBC's 3L^2 line labels live in a much smaller
detector space, so a coset label has no unique decomposition into lit classes and
"one edge per lit class" is not a well-defined recovery (the reference asserts
the same independence condition). `cup` has no recovery stage and works at PBC.

Cost and caps
-------------
Python loops run over the distinct coset labels present in a block and over the
small per-class enumerations -- never over rows. Two knobs bound the work, and a
row that exceeds either falls back to the `linear` head and is counted in
`pop_stats()["n_fallback"]`:

  `k_cap`      (default 8) max lit classes -- per connected component for
               `vote`, in total for `pt2` (whose weights do not factorize).
  `max_terms`  (default 200_000) max recoveries actually contracted: the class
               product per component (`vote`) or in total (`pt2`), and the
               second-order candidate count. This is the cap that matters at
               L >= 5, where one coset asks for 7.5e5 (L=5) or 1.1e8 (L=6)
               recoveries per row -- see the table above. `linear` never falls
               back and `cup` has nothing to cap.

Measured steady-state microseconds per row (20k uniform-random configs, one
thread, `tests/test_sign_decoders.py`'s `bench()`; the per-coset structures are
built lazily on the first pass and cached, so a training step pays only this):

    L (OBC)      2     3     4     5     6
    cup       0.04  0.18  0.35  1.05  1.78     <- one N x N GEMM, flat in k
    linear    0.05  0.36  0.47  1.24  2.06     <- ditto plus n_lit columns
    vote      0.10  0.35  0.79  2.09  4.19
    pt2       0.10  1.03  5.33  3.91 10.00     <- k-dependent (2nd-order stage)

`cup`/`linear` are dominated by the fixed N x N product, so they are flat in the
number of flipped spins; `vote`/`pt2` grow with the per-row lit-class count.

`pop_stats()` returns and RESETS
{"n_rows", "n_fallback", "n_pt2", "n_tie", "k_max", "k_sum"}: rows evaluated,
rows that hit a cap, rows where the pt2 second-order stage ran, rows whose
first-order sum was an exact tie, and the max / sum over rows of the lit-class
count k (so k_sum / n_rows is the mean lit lines per row).

Never builds a 2^N object: everything is dense GF(2) linear algebra on N, NP
(<= 3 L^3 <= 648 for L <= 6, the `CupSign` dense-matrix cap).
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from functools import reduce
from operator import xor

import numpy as np

from tc3d.fermionic_decoration import (_mask, fermionic_plaquettes,
                                       flux_constraint_masks)
from tc3d.sign_geometry import CupSign, _gf2_nullspace, _gf2_rref, _solve_map

__all__ = ["make_decoder_sign", "DecoderSign", "CupHead", "KINDS",
           "DEFAULT_K_CAP", "DEFAULT_MAX_TERMS"]

KINDS = ("cup", "linear", "vote", "pt2")

DEFAULT_K_CAP = 8            # max lit classes per contraction (see module docstring)
DEFAULT_MAX_TERMS = 200_000  # max recoveries contracted per row (ditto)

_BLOCK = 1 << 16             # hard cap on rows per host-side chunk
_MAX_DET = 62                # coset labels are packed into an int64
_MAX_ORDER2 = 1 << 21        # C(N, m+1) ceiling on second-order CANDIDATE generation
_MAX_PT_WORK = 1 << 26       # per-coset (recoveries x sub-flips) path-weight cap
_MAX_EXACT = 1 << 50         # |contraction| must stay exactly representable (float64)
_ELEM_BUDGET = 1 << 22       # elements per transient (rows x cols) work array


def _block_rows(width, budget=_ELEM_BUDGET):
    """Rows per chunk so that a transient (rows, width) array stays ~budget."""
    return int(min(_BLOCK, max(256, budget // max(1, int(width)))))


# =============================================================================
# GF(2) helpers
# =============================================================================

def _gf2_reduce(vecs):
    """Full-RREF pivot dict {pivot bit -> reduced row} of integer-bitmask vectors.

    Same routine -- and therefore the same reduced basis -- as
    `sign_fidelity_ftc.gf2_reduce`, so the detector set, the coset labels, the
    class order and the recovery enumeration order all match the reference.
    """
    piv: dict = {}
    for v in vecs:
        v = int(v)
        for c in list(piv):
            if (v >> c) & 1:
                v ^= piv[c]
        if v:
            c = (v & -v).bit_length() - 1
            for k in list(piv):
                if (piv[k] >> c) & 1:
                    piv[k] ^= v
            piv[c] = v
    return piv


def _base_quadratic_form(cs: CupSign):
    """K (N, N) uint8 with (-1)^{b^T K b mod 2} == `cs.sign` on every config.

    gauge fix  b -> G b,  G = 1 + Sv^T Zcoef Phi    (a fixed linear map)
    lift       a = P_lift G b                        (also linear)
    Q(a)       = sum_p (I a)_p * a(e+(p)) = a^T A a, A[:, e+(p)] += I[p]
    """
    N = cs.N
    G = (np.eye(N, dtype=np.int64)
         + cs.Sv.T.astype(np.int64) @ cs.Zcoef.astype(np.int64)
         @ cs.Phi.astype(np.int64)) % 2
    M = (cs.P_lift.astype(np.int64) @ G) % 2
    A = np.zeros((N, N), dtype=np.int64)
    for p in range(cs.NP):
        if cs.eplus[p] >= 0:
            A[:, cs.eplus[p]] += cs.I[p].astype(np.int64)
    A %= 2
    return ((M.T @ A % 2) @ M % 2).astype(np.uint8)


def _path_weights(eps, fx, J):
    """Exact D(eps) for a BATCH of flip subsets (n, k), in input order.

    D(eps) = sum over orderings of prod_j 1/DeltaE(A_j), A_j = the first j flips,
    DeltaE(A) = 2J * #{p : |dp cap A| odd} (sigma^x commutes with every star, so
    only plaquettes are excited); a path through the ground sector (DeltaE = 0)
    is projected out by the resolvent and contributes 0. Port of
    `sign_fidelity_ftc.path_weight` in EXACT rational arithmetic -- `Fraction(J)`
    is the float's exact binary value -- so the "sum vanishes exactly" test is
    independent of summation order.

    D depends on eps only through its EXCITATION PROFILE: the plaquette count of
    every one of the 2^k - 1 non-empty sub-flips. The profiles are computed with
    2^k - 1 vectorized XOR/popcount passes over the whole batch and deduplicated,
    so the (slow) k! Fraction sum runs once per DISTINCT profile -- of which
    there are a handful even for tens of thousands of recoveries.
    """
    eps = np.asarray(eps, dtype=np.int64)
    eps = eps.reshape(eps.shape[0], -1)
    n, k = eps.shape
    if n == 0:
        return []
    if k == 0:
        return [Fraction(1)] * n

    NP = fx.shape[1]
    prof = np.empty((n, (1 << k) - 1), dtype=np.int32)
    step = max(1, _ELEM_BUDGET // max(1, NP))
    for a in range(0, n, step):
        blk = eps[a:a + step]
        for msk in range(1, 1 << k):
            acc = np.zeros((blk.shape[0], NP), dtype=np.uint8)
            for i in range(k):
                if (msk >> i) & 1:
                    acc ^= fx[blk[:, i]]
            prof[a:a + step, msk - 1] = acc.sum(axis=1)

    uniq, inv = np.unique(prof, axis=0, return_inverse=True)
    Jf, vals = Fraction(J), []
    for row in uniq:
        tot = Fraction(0)
        for order in itertools.permutations(range(k)):
            term, msk = Fraction(1), 0
            for i in order:
                msk |= 1 << i
                c = int(row[msk - 1])
                if c == 0:
                    term = Fraction(0)
                    break
                term /= 2 * Jf * c
            tot += term
        vals.append(tot)
    return [vals[i] for i in np.asarray(inv).ravel()]


def _integer_weights(fracs):
    """Exact rationals -> (integer array scaled by the common LCM, ok flag).

    Scaling by a POSITIVE common denominator preserves both the sign and the
    "is exactly zero" verdict of any signed sum of them, so the pt2 contraction
    becomes exact integer arithmetic. `ok` is False when the scaled magnitudes
    would leave the float64 exact-integer range (`_MAX_EXACT`).
    """
    dens = [f.denominator for f in fracs if f != 0]
    lcm = math.lcm(*dens) if dens else 1
    w = np.array([int(f * lcm) for f in fracs], dtype=object)
    tot = int(np.abs(w).sum()) if w.size else 0
    if tot > _MAX_EXACT:
        return None, False
    return w.astype(np.float64), True


def _components(cls_edges, Ksym):
    """Connected components of the lit classes under the Ksym coupling.

    Classes i and j are coupled iff some candidate edge of i couples to some
    candidate edge of j; the recovery phase sum factorizes over components.
    Returns a list of sorted lists of class positions.
    """
    m = len(cls_edges)
    adj = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if Ksym[np.ix_(cls_edges[i], cls_edges[j])].any():
                adj[i].append(j)
                adj[j].append(i)
    seen, comps = [False] * m, []
    for i in range(m):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def _pair_const(choices, Ksym):
    """sum_{a<b} Ksym[choice_a, choice_b] mod 2 for each row of `choices`."""
    n, r = choices.shape
    out = np.zeros(n, dtype=np.int64)
    for a in range(r):
        for b in range(a + 1, r):
            out += Ksym[choices[:, a], choices[:, b]]
    return out % 2


def _pair_tensor(cls_edges, Ksym):
    """(-1)^{sum_{i<j} Ksym[C_i[a_i], C_j[a_j]]} over the class product, flat.

    Built as a broadcast product of the pairwise (-1)^Ksym blocks -- the entries
    of the flat `itertools.product(*cls_edges)` enumeration, in that order.
    """
    shape = tuple(len(c) for c in cls_edges)
    out = np.ones(shape, dtype=np.float64)
    for i in range(len(cls_edges)):
        for j in range(i + 1, len(cls_edges)):
            g = 1.0 - 2.0 * Ksym[np.ix_(cls_edges[i], cls_edges[j])]
            bshape = [1] * len(shape)
            bshape[i], bshape[j] = shape[i], shape[j]
            out *= g.reshape(bshape)
    return out.reshape(-1)


# =============================================================================
# support structure (detectors, lit line classes, coset decomposition)
# =============================================================================

class _Support:
    """Coset detectors and lit line classes of the h=0 support.

    Mirrors `sign_fidelity_ftc.support_structure`, minus its 2^n_lit
    enumeration of coset labels: the decomposition of a label into lit classes
    is a fixed GF(2) solve (`_solve_map`) applied per row instead.

    Detector set. The reference ASSERTS that the `flux_constraint_masks`
    Z-strings already span W^perp, W = span(stars, x-pairs) -- true at OBC, but
    false at PBC, where the homological (logical) Z-strings are a further
    N - rank(W) - n_flux directions (L=2 PBC: 8 flux + 1). Here W^perp is solved
    for and appended, so "coset label 0" always means "on the h=0 support". The
    appended vectors reduce to zero wherever the reference's assert holds, so
    the OBC detector set -- and with it every label, class and enumeration order
    -- is bit-identical to the reference's.
    """

    def __init__(self, geo, stabs):
        N = geo.N
        zm = [_mask(z) for z, _x, _c in stabs]
        u_masks = [reduce(xor, (zm[p] for p in c), 0)
                   for c in flux_constraint_masks(stabs)]
        W = np.zeros((len(geo.vertex_all) + len(stabs), N), dtype=np.uint8)
        for v, es in enumerate(geo.vertex_all):
            for e in es:
                if e != -1:
                    W[v, e] = 1
        for p, (_z, x, _c) in enumerate(stabs):
            for e in x:
                if e != -1:
                    W[len(geo.vertex_all) + p, e] = 1
        perp = [int(sum(int(v[e]) << e for e in range(N)))
                for v in _gf2_nullspace(W)]
        self.n_flux_det = len(_gf2_reduce(u_masks))
        det = sorted(_gf2_reduce(u_masks + perp).values())
        assert len(det) == N - len(_gf2_rref(W)[1]), "detector count != dim W^perp"
        if len(det) > _MAX_DET:
            raise ValueError(f"{len(det)} support detectors exceeds the int64 "
                             f"coset-label packing cap ({_MAX_DET})")

        self.N = N
        self.det = det
        self.lab = [sum((((d >> e) & 1) << k) for k, d in enumerate(det))
                    for e in range(N)]
        classes: dict = {}
        for e in range(N):
            classes.setdefault(self.lab[e], []).append(e)   # ascending -> [0] = min
        self.classes = classes
        self.lit = sorted(k for k in classes if k)
        n_det, n_lit = len(det), len(self.lit)

        self.Dmat = np.array([[(d >> e) & 1 for d in det] for e in range(N)],
                             dtype=np.float32).reshape(N, n_det)
        self._pow2 = 1 << np.arange(n_det, dtype=np.int64)
        A = np.array([[(k >> j) & 1 for k in self.lit] for j in range(n_det)],
                     dtype=np.uint8).reshape(n_det, n_lit)
        self.Zmap = _solve_map(A).T.astype(np.float32)                # (n_det, n_lit)
        if n_lit and not np.array_equal(
                (self.Zmap.T.astype(np.int64) @ A.astype(np.int64)) % 2,
                np.eye(n_lit, dtype=np.int64)):
            raise ValueError(
                f"{n_lit} lit line-class labels span only {len(_gf2_reduce(self.lit))} "
                f"of {n_det} detector bits: a coset label does not decompose "
                "uniquely into lit classes, so the minimal-recovery decoders "
                "(linear/vote/pt2) are undefined here. This is the PBC case "
                "(3L^2 diagonal lines vs. a much smaller detector space) -- the "
                "reference sign_fidelity_ftc.support_structure asserts the same "
                "condition. Use sign_frame='cup' (or OBC).")

    def label(self, b):
        """b (B, N) 0/1 float32 -> (u int64 (B,), coef int64 (B, n_lit))."""
        ub = (b @ self.Dmat).astype(np.int64) & 1
        coef = (ub.astype(np.float32) @ self.Zmap).astype(np.int64) & 1
        return ub @ self._pow2, coef


# =============================================================================
# heads
# =============================================================================

class _Head:
    """Shared geometry precompute, blocking and `pop_stats()` bookkeeping."""

    kind = None

    def __init__(self, geo, stabs=None, k_cap=DEFAULT_K_CAP,
                 max_terms=DEFAULT_MAX_TERMS, J=1.0):
        self.geo = geo
        self.stabs = fermionic_plaquettes(geo) if stabs is None else stabs
        self.k_cap = int(k_cap)
        self.max_terms = int(max_terms)
        self.J = float(J)
        self.cup = CupSign(geo, self.stabs)
        self.K = _base_quadratic_form(self.cup)
        self._K32 = self.K.astype(np.float32)
        self._reset_stats()

    # -- stats --------------------------------------------------------------

    def _reset_stats(self):
        self._stats = {"n_rows": 0, "n_fallback": 0, "n_pt2": 0, "n_tie": 0,
                       "k_max": 0, "k_sum": 0}

    def pop_stats(self):
        """Return the accumulated bookkeeping and reset the counters."""
        s = self._stats
        self._reset_stats()
        return s

    def _account_k(self, coef):
        kk = coef.sum(axis=1)
        self._stats["k_sum"] += int(kk.sum())
        if kk.size:
            self._stats["k_max"] = max(self._stats["k_max"], int(kk.max()))

    # -- evaluation ---------------------------------------------------------

    def __call__(self, configs):
        x = np.asarray(configs)
        lead, flat = x.shape[:-1], x.reshape(-1, x.shape[-1])
        out = np.empty(flat.shape[0], dtype=np.float64)
        step = _block_rows(flat.shape[-1])
        for a in range(0, flat.shape[0], step):
            out[a:a + step] = self._eval_block(flat[a:a + step])
        self._stats["n_rows"] += flat.shape[0]
        return out.reshape(lead)

    def _q0(self, b):
        """q(b) = b^T K b mod 2, one float32 GEMM + a row dot (mod once, at the
        end: the intermediate row sums are <= N^2 < 2^24, exact in float32)."""
        return np.einsum("ij,ij->i", b @ self._K32, b).astype(np.int64) & 1


class CupHead(_Head):
    """The exact h=0 cup-product sign, evaluated as the quadratic form q(b).

    Bit-identical to `CupSign.sign` (regression-tested) but one GEMM instead of
    its gauge-fix / lift / Q pipeline. It still reports the per-row lit-class
    count (k_sum / k_max) -- one small (B, N) x (N, n_det) product -- because
    that is the cheapest "how far off-support are my samples" diagnostic
    available. n_fallback = n_pt2 = n_tie = 0 by construction. Where the class
    decomposition does not exist (PBC, see `_Support`) the head still works and
    k_sum / k_max stay 0.
    """

    kind = "cup"

    def __init__(self, geo, stabs=None, k_cap=DEFAULT_K_CAP,
                 max_terms=DEFAULT_MAX_TERMS, J=1.0):
        super().__init__(geo, stabs, k_cap, max_terms, J)
        try:
            self.support = _Support(geo, self.stabs)
        except ValueError:                     # PBC: no unique class decomposition
            self.support = None

    def _eval_block(self, block):
        b = (block < 0).astype(np.float32)
        if self.support is not None:
            self._account_k(self.support.label(b)[1])
        return 1.0 - 2.0 * self._q0(b)


class DecoderSign(_Head):
    """linear | vote | pt2 evaluated per configuration (no 2^N table)."""

    def __init__(self, kind, geo, stabs=None, k_cap=DEFAULT_K_CAP,
                 max_terms=DEFAULT_MAX_TERMS, J=1.0):
        if kind not in ("linear", "vote", "pt2"):
            raise ValueError("DecoderSign kind must be linear|vote|pt2, "
                             f"got {kind!r}")
        super().__init__(geo, stabs, k_cap, max_terms, J)
        self.kind = kind
        self.support = _Support(geo, self.stabs)
        self.Ksym = (self.K + self.K.T) % 2
        self._fx = np.ascontiguousarray(self.cup.I.T)   # (N, NP) single-flip flux

        # l_b(c) is only ever read at these edges, so the (B, N) x (N, N) Ksym
        # product shrinks to the columns this head actually uses
        sup = self.support
        if kind == "linear":
            cols = sorted(sup.classes[k][0] for k in sup.lit)
        elif kind == "vote":
            cols = sorted(e for k in sup.lit for e in sup.classes[k])
        else:                                   # pt2's 2nd order reaches unlit edges
            cols = list(range(sup.N))
        self._cols = np.array(cols, dtype=np.int64)
        self._colof = np.full(sup.N, -1, dtype=np.int64)
        self._colof[self._cols] = np.arange(self._cols.size)
        self._Ksym_cols = self.Ksym[:, self._cols].astype(np.float32)
        self._diag_cols = np.diag(self.K)[self._cols].astype(np.float32)
        self._cache: dict = {}                  # coset label -> recovery structure

    # -- per-coset structures (built lazily, cached) -------------------------

    def _struct(self, u, coef):
        """Recovery structure for coset label `u` (coef = its 0/1 lit vector)."""
        st = self._cache.get(u)
        if st is not None:
            return st
        sup = self.support
        cls = [sup.classes[sup.lit[i]] for i in range(len(sup.lit)) if coef[i]]
        m = len(cls)
        rep = [c[0] for c in cls]                                  # lowest index
        st = {"u": u, "m": m, "cls": cls,
              "rep_cols": self._colof[np.array(rep, dtype=np.int64)],
              "rep_const": int(_pair_const(np.array(rep, np.int64)[None, :],
                                           self.Ksym)[0]) if m else 0,
              "capped": False, "second": None}
        if m:
            if self.kind == "vote":
                st.update(self._vote_struct(cls))
            elif self.kind == "pt2":
                st.update(self._pt_struct(cls, m))
        self._cache[u] = st
        return st

    def _blocks(self, cls_edges, weights=None):
        """(per-class column indices, flat contraction tensor) for `_contract`.

        The tensor is the pair phase (-1)^{sum_{i<j} Ksym} times the optional
        exact-integer path weights. Its dtype is the narrowest one that still
        represents every partial sum exactly (float32 below 2^23).
        """
        W = _pair_tensor(cls_edges, self.Ksym)
        if weights is not None:
            W = W * weights
        dt = np.float32 if float(np.abs(W).sum()) < float(1 << 23) else np.float64
        return ([self._colof[np.array(c, dtype=np.int64)] for c in cls_edges],
                W.astype(dt))

    def _vote_struct(self, cls):
        """One contraction per connected component of the lit classes."""
        comps = _components(cls, self.Ksym)
        for comp in comps:
            if len(comp) > self.k_cap or \
                    math.prod(len(cls[i]) for i in comp) > self.max_terms:
                return {"capped": True}
        out = []
        for comp in comps:
            sub = [cls[i] for i in comp]
            cols, W = self._blocks(sub)
            out.append((cols, W, tuple(len(c) for c in sub)))
        return {"comps": out, "capped": False}

    def _pt_struct(self, cls, m):
        """First-order recovery tensor: (-1)^{pair phase} * D(eps), exact ints.

        The D(eps) weights do not factorize over classes, so the whole
        |C_0| x ... x |C_{m-1}| product is contracted at once (no component
        split) -- hence `max_terms` bites here first.
        """
        n_rec = math.prod(len(c) for c in cls)
        if (m > self.k_cap or n_rec > self.max_terms
                or n_rec * (1 << m) > _MAX_PT_WORK):   # D(eps) profile passes
            return {"capped": True}
        recs = np.array(list(itertools.product(*cls)), dtype=np.int64)
        w, ok = _integer_weights(_path_weights(recs, self._fx, self.J))
        if not ok:
            return {"capped": True}
        cols, W = self._blocks(cls, weights=w)
        return {"pt": (cols, W, tuple(len(c) for c in cls)), "capped": False}

    def _second_order(self, st):
        """(m+1)-flip subsets carrying the same coset label + their weights.

        Not a class product, so this one is a plain (rows x candidates) sign
        matrix times the integer weight vector.
        """
        if st["second"] is not None:
            return st["second"]
        N, k, u = self.support.N, st["m"] + 1, st["u"]
        if math.comb(N, k) > _MAX_ORDER2:
            st["second"] = {"capped": True}
            return st["second"]
        lab = np.array(self.support.lab, dtype=np.int64)
        if k == 2:                                    # lexicographic, vectorized
            ii, jj = np.triu_indices(N, 1)
            keep = (lab[ii] ^ lab[jj]) == u
            eps = np.stack([ii[keep], jj[keep]], axis=1)
        else:
            eps = np.array([c for c in itertools.combinations(range(N), k)
                            if reduce(xor, (int(lab[e]) for e in c), 0) == u],
                           dtype=np.int64).reshape(-1, k)
        if eps.shape[0] > self.max_terms:
            st["second"] = {"capped": True}
            return st["second"]
        w, ok = _integer_weights(_path_weights(eps, self._fx, self.J))
        if not ok:
            st["second"] = {"capped": True}
            return st["second"]
        nz = np.nonzero(w)[0]
        phase = (1.0 - 2.0 * _pair_const(eps[nz], self.Ksym)) * w[nz]
        dt = np.float32 if float(np.abs(phase).sum()) < float(1 << 23) else np.float64
        st["second"] = {"capped": False, "cols": self._colof[eps[nz]],
                        "phase": np.ascontiguousarray(phase, dtype=dt)}
        return st["second"]

    # -- evaluation ---------------------------------------------------------

    def _eval_block(self, block):
        b = (block < 0).astype(np.float32)                         # (B, N)
        q0 = self._q0(b)                                           # (B,) 0/1
        # l_b(c) unreduced would be fine for a single lookup, but vote/pt2 read
        # it tens of thousands of times per row, so reduce once here (values are
        # <= N+1, exact in float32) and keep it int8: the recovery kernels are
        # gather-bound, so the 4x narrower rows are a 4x traffic cut
        ell = ((b @ self._Ksym_cols + self._diag_cols).astype(np.int32) & 1
               ).astype(np.int8)
        u, coef = self.support.label(b)
        self._account_k(coef)

        out = np.empty(b.shape[0], dtype=np.float64)
        for uv in np.unique(u):
            idx = np.nonzero(u == uv)[0]
            st = self._struct(int(uv), coef[idx[0]])
            out[idx] = self._eval_group(st, q0[idx], ell[idx])
        return out

    def _linear(self, st, q0, ell):
        e = q0 + st["rep_const"]
        if st["m"]:
            e = e + ell[:, st["rep_cols"]].sum(axis=1, dtype=np.int64)
        return 1.0 - 2.0 * (e & 1)

    def _eval_group(self, st, q0, ell):
        """+-1 head values for the rows of one coset label."""
        lin = self._linear(st, q0, ell)
        if self.kind == "linear" or st["m"] == 0:
            return lin                       # on support: every head is the base sign
        if st["capped"]:
            self._stats["n_fallback"] += q0.shape[0]
            return lin
        base = 1.0 - 2.0 * q0

        if self.kind == "vote":
            tot = np.ones(q0.shape[0])
            for cols, W, shape in st["comps"]:
                tot *= self._contract(ell, cols, W, shape)
            tie = tot == 0
            self._stats["n_tie"] += int(tie.sum())
            return np.where(tie, lin, np.sign(tot) * base)

        # ---- pt2 -----------------------------------------------------------
        cols, W, shape = st["pt"]
        s = np.sign(self._contract(ell, cols, W, shape))
        zero = s == 0
        self._stats["n_tie"] += int(zero.sum())
        if zero.any():
            sec = self._second_order(st)
            if sec["capped"]:
                self._stats["n_fallback"] += int(zero.sum())
            else:
                self._stats["n_pt2"] += int(zero.sum())
                s[zero] = np.sign(self._flat_sum(ell[zero], sec["cols"],
                                                 sec["phase"]))
        return np.where(s == 0, lin, s * base)

    # -- contraction kernels ------------------------------------------------

    def _contract(self, ell, cols, W, shape):
        """sum over the class product of prod_k u_k(a_k) * W[a_0,...,a_{m-1}].

        One BLAS GEMM against the trailing class axis, then broadcast reductions
        against the remaining ones. Every quantity is an exact integer (|W| sums
        are bounded by `_MAX_EXACT`), so the result -- and the "== 0" test on it
        -- is order-independent and exact.
        """
        n_rec = int(np.prod(shape))
        dt = W.dtype
        Wm = np.ascontiguousarray(W.reshape(-1, shape[-1]).T)
        step = _block_rows(max(1, n_rec // shape[-1]))
        out = np.empty(ell.shape[0], dtype=np.float64)
        for a in range(0, ell.shape[0], step):
            u = [1 - 2 * ell[a:a + step, c].astype(dt) for c in cols]
            t = u[-1] @ Wm                                  # (b, prod(shape[:-1]))
            for k in range(len(shape) - 2, -1, -1):
                t = (t.reshape(t.shape[0], -1, shape[k]) * u[k][:, None, :]).sum(-1)
            out[a:a + step] = t.reshape(-1)
        return out

    def _flat_sum(self, ell, cols, phase):
        """sum_j (-1)^{sum_k l(cols[j,k])} phase[j] -- the non-product case.

        The second-order candidate set is a union of class products with
        weight-dependent entries, not one product, so there is no low-rank
        contraction: this is gather-bound at O(rows x candidates x k). Cost is
        cut by XOR-folding k narrow int8 gathers instead of materialising a
        (rows, candidates, k) block, and by keeping the GEMV in the narrowest
        dtype that is still exact (see `_second_order`).
        """
        n, k = cols.shape
        dt = phase.dtype
        step = max(1, _ELEM_BUDGET // max(1, ell.shape[0]))
        out = np.zeros(ell.shape[0], dtype=np.float64)
        for a in range(0, n, step):
            c = cols[a:a + step]
            d = ell[:, c[:, 0]]
            for i in range(1, k):
                d = d ^ ell[:, c[:, i]]                          # (B, nj) int8
            out += (1 - 2 * d.astype(dt)) @ phase[a:a + step]
        return out


def make_decoder_sign(kind, geo, stabs=None, k_cap=DEFAULT_K_CAP,
                      max_terms=DEFAULT_MAX_TERMS, J=1.0):
    """`sign_fn(configs) -> (...,) +-1 floats` for kind in cup|linear|vote|pt2.

    The returned callable carries `pop_stats()` (see the module docstring) and
    satisfies the `SignFramedOperator` contract: host numpy in, host numpy out,
    shape (..., N) -> (...). It is stateful only in its counters and its lazy
    per-coset cache, so a later `sign_fn_conn(x, xp)` fast path (decode once per
    sample, reuse the coset structure for its connected configs) can be added as
    an extra method without touching that contract.
    """
    if kind == "cup":
        return CupHead(geo, stabs, k_cap=k_cap, max_terms=max_terms, J=J)
    return DecoderSign(kind, geo, stabs, k_cap=k_cap, max_terms=max_terms, J=J)
