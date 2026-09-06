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

The pt2 TIE-BREAKER is the same shape, one axis longer. The lit-class labels are
independent, so an (m+1)-flip subset can carry the coset label only by taking one
edge from each of the m lit classes plus one edge of label 0 (any other multiset
needs >= m + 2 flips) -- the "next order" set is exactly the class product
C_0 x ... x C_{m-1} x C_unlit, not the unstructured union a C(N, m+1) enumeration
suggests (`test_second_order_is_a_product` checks it against that enumeration).
So `_second_struct` contracts it too, instead of gathering over ~30k candidates
per tied row; the classes are ordered largest-last so the single GEMM does as
much of the work as possible.

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
               second-order product. This is the cap that matters at L >= 5,
               where one coset asks for 7.5e5 (L=5) or 1.1e8 (L=6) recoveries
               per row -- see the table above. `linear` never falls back and
               `cup` has nothing to cap.
  `_MAX_ORDER2` a C(N, m+1) ceiling on the second order. It no longer bounds
               any enumeration (the product is generated directly) and is kept
               only to FREEZE the head's verdicts: dropping it would newly
               enable the L=5 two-lit-class tie-breaker, which is a semantics
               change, not a speed one. The other case it still refuses is the
               L=4 THREE-lit-class tie, whose product is (18, 18, 24, 84) =
               653_184 candidates: measured 4.9 s to build once and 25 us per
               tied row, on 5e-6 of uniform-random rows (1 in 200_000) -- so
               `_MAX_ORDER2 = 1 << 40` plus `--sign_max_terms 700000` buys it
               for ~0.1 ms/step at L=4, at the price of a verdict change.

Measured steady-state microseconds per row for the direct `sign_fn(x)` path (20k
uniform-random configs, `tests/test_sign_decoders.py`'s `bench()`, M-series /
Accelerate; the per-coset structures are built lazily on the first pass and
cached, so a steady step pays only this):

    L (OBC)      2     3     4     5     6
    cup       0.03  0.15  0.34  0.79  1.59     <- one N x N GEMM, flat in k
    linear    0.06  0.21  0.47  1.08  1.95     <- ditto plus n_lit columns
    vote      0.07  0.22  0.69  1.40  2.86
    pt2       0.06  0.27  0.67  1.66  3.49     <- k-dependent (2nd-order stage)

and for the `sign_conn(x, xp)` path a training step actually uses -- B samples
plus their B * n_conn connected rows, same machine, old = this module before the
fast path existed:

    L (OBC)        2      3      4      5      6
    N             12     54    144    300    540
    n_conn        21     94    263    570   1057
    cup      .057/.039 .087/.154 .122/.338 .216/.805 .322/1.686
    linear   .061/.054 .091/.201 .132/.402 .228/.960 .357/1.858
    vote     .070/.099 .136/.281 .291/.707 .501/1.612 .989/3.542
    pt2      .072/.101 .165/1.010 .339/4.833 .682/3.028 1.273/6.156
                                  ^ 14x, and 1.2x of vote instead of 7x

`cup`/`linear` are dominated by the fixed N x N product, so they are flat in the
number of flipped spins; `vote`/`pt2` grow with the per-row lit-class count. At
L = 2 the fast path is a small LOSS (the mask matvec is not amortised by an
N = 12 quadratic form) -- 0.07 us/row either way, i.e. milliseconds per step.

`pop_stats()` returns and RESETS
{"n_rows", "n_decoded", "n_fallback", "n_pt2", "n_tie", "k_max", "k_sum"}: rows
asked for, rows actually decoded (smaller than n_rows exactly when `sign_conn`
served a connected row from its sample -- see `_Head`), rows that hit a cap,
rows where the pt2 second-order stage ran, rows whose first-order sum was an
exact tie, and the max / sum over rows of the lit-class count k (so
k_sum / n_rows is the mean lit lines per row). n_tie / n_pt2 / n_fallback count
DECODED rows only; k_sum / k_max still cover every row.

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
_MAX_ORDER2 = 1 << 21        # second-order SEMANTICS freeze, not a cost cap (see above)
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

def _stack_classes(cls):
    """(edge universe, per-class column SLICES into it, offsets).

    The slices are contiguous by construction, so `_contract` reads each class'
    l-values as a strided view instead of a fancy-index gather.
    """
    off = np.cumsum([0] + [len(c) for c in cls]).astype(np.int64)
    ecols = np.concatenate([np.array(c, dtype=np.int64) for c in cls]) if cls \
        else np.zeros(0, dtype=np.int64)
    return ecols, [slice(int(off[i]), int(off[i + 1])) for i in range(len(cls))], off


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

    `gid` packs the lit-class coefficient vector of a config into one small
    integer (bit i = "lit class i is on"). It is an equivalent -- and linear --
    relabelling of the coset label `u`: every reachable u is a GF(2) combination
    of the (independent) lit-class labels, so gid <-> u is a bijection there.
    Grouping rows by gid instead of u keeps the group key in [0, 2^n_lit), which
    is what makes the connected-set fast path's grouping a bincount rather than
    a sort, and makes the mask update gid(b XOR M) = gid(b) XOR gid(M) trivial.
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
        Wr, _piv = _gf2_rref(W)
        assert len(det) == N - len(_piv), "detector count != dim W^perp"
        if len(det) > _MAX_DET:
            raise ValueError(f"{len(det)} support detectors exceeds the int64 "
                             f"coset-label packing cap ({_MAX_DET})")

        self.N = N
        self.det = det
        self.Wbasis = np.asarray(Wr, dtype=np.uint8).reshape(-1, N)  # basis of W
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
        self._pow2lit = 1 << np.arange(n_lit, dtype=np.int64)
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

    def gid(self, b):
        """b (B, N) 0/1 float32 -> packed lit-class coefficient vector (B,)."""
        ub = (b @ self.Dmat).astype(np.float32)
        ub -= 2.0 * np.floor(0.5 * ub)                       # ub mod 2, in float32
        coef = (ub @ self.Zmap).astype(np.int64) & 1
        return coef @ self._pow2lit

    def gid_of_edges(self, edges):
        """Packed lit-class vector of the mask with exactly `edges` flipped."""
        g = 0
        for e in np.asarray(edges, dtype=np.int64).ravel().tolist():
            lb = self.lab[int(e)]
            if lb:
                g ^= 1 << self.lit.index(lb)
        return int(g)


# =============================================================================
# connected-set XOR-mask dictionary
# =============================================================================

_HASH_BITS = 40              # width of each random mask hash (exact in float64)
_MAX_MASKS = 1 << 14         # dictionary ceiling before the fast path gives up


class _ConnMasks:
    """The XOR masks between a sample and its connected configurations.

    `get_conn_padded` hands the head (x, xp) where every xp[b, c] differs from
    x[b] by the X-support of ONE Pauli term of the wrapped operator (padding
    columns repeat x, i.e. the empty mask). That mask SET is a property of the
    operator -- a few hundred entries -- while the mask of a given COLUMN is
    not fixed (NetKet drops zero matrix elements and left-packs, so the columns
    shift row by row). So each connected row is identified, after which its
    (q, l) follow from the sample's by the exact GF(2) update in `_Head`.

    Identification is one batched (B, C, N) x (B, N, 2) matvec against
    [x*r1, x*r2]: with d = (1 - x*xp)/2 (valid because the local states are
    +-1) those columns give two independent 40-bit random hashes of the mask,
    as exact integers in float64 (the partial sums stay below sum(r) <=
    N * 2^40 < 2^53). hash1 keys the dictionary and hash2 is checked against
    the stored value on EVERY row of every call, so a mis-identification would
    need BOTH hashes to collide; a row that fails either check is routed to a
    full decode, which is why a collision could only ever cost time.

    The (B, N, 2) projector and the (B, C, 2) dot block are kept and reused
    between calls: allocating them fresh next to a multi-GB xp costs more in
    page faults than the matvec itself does in arithmetic.
    """

    def __init__(self, N, seed=20260906):
        self.N = int(N)
        rng = np.random.default_rng(seed)
        self.r = rng.integers(1, 1 << _HASH_BITS, size=(2, self.N)).astype(np.float64)
        self.R = self.r.sum(axis=1)
        self.keys: dict = {}                 # int64 key -> mask id
        self.idx: list = []                  # mask id -> flipped edge indices
        self._kk = np.zeros(0, dtype=np.int64)   # sorted keys      (lookup table)
        self._kid = np.zeros(0, dtype=np.int32)  # their mask ids
        self._h2 = np.zeros(0, dtype=np.int64)   # per-mask verification hash
        self._buf: dict = {}                 # (B, C) -> reusable scratch
        self.disabled = False                # local states are not +-1
        self.full = False                    # dictionary hit _MAX_MASKS

    def _scratch(self, B, C):
        buf = self._buf.get((B, C))
        if buf is None:
            while len(self._buf) >= 2:       # ragged last chunk + full chunks
                self._buf.pop(next(iter(self._buf)))
            buf = self._buf[(B, C)] = (np.empty((B, self.N, 2), dtype=np.float64),
                                       np.empty((B, C, 2), dtype=np.float64),
                                       np.empty((B, C), dtype=np.int64),
                                       np.empty((B, C), dtype=np.int64))
        return buf

    def _fingerprint(self, xb, xpb):
        """(key, hash2) int64 arrays, flat over xpb.shape[:-1]."""
        B, C, _N = xpb.shape
        proj, dots, key, h2 = self._scratch(B, C)
        np.multiply(xb, self.r[0], out=proj[:, :, 0])
        np.multiply(xb, self.r[1], out=proj[:, :, 1])
        np.matmul(np.asarray(xpb, dtype=np.float64), proj, out=dots)
        np.subtract(self.R[0], dots[:, :, 0], out=key, casting="unsafe")
        np.subtract(self.R[1], dots[:, :, 1], out=h2, casting="unsafe")
        key >>= 1                            # (R - dot) / 2, exact and even
        h2 >>= 1
        return key.reshape(-1), h2.reshape(-1)

    def identify(self, xb, xpb):
        """(mask ids (B*C,) int32, verified (B*C,) bool). An unverified row's
        id is meaningless -- the caller must decode it from scratch."""
        flat, h2f = self._fingerprint(xb, xpb)
        pos = np.searchsorted(self._kk, flat)
        np.clip(pos, 0, max(self._kk.size - 1, 0), out=pos)
        ok = self._kk[pos] == flat if self._kk.size else np.zeros(flat.size, bool)
        if not ok.all() and not self.full:
            self._learn(xb, xpb, flat, h2f, ok)
            pos = np.searchsorted(self._kk, flat)
            np.clip(pos, 0, max(self._kk.size - 1, 0), out=pos)
            ok = self._kk[pos] == flat if self._kk.size else np.zeros(flat.size, bool)
        ids = np.where(ok, self._kid[pos], np.int32(0))
        ok &= self._h2[ids] == h2f                     # collision guard
        return ids, ok

    def _learn(self, xb, xpb, flat, h2f, known):
        """Record the masks behind the keys not yet in the dictionary."""
        C = xpb.shape[1]
        miss = np.nonzero(~known)[0]
        uniq, first = np.unique(flat[miss], return_index=True)
        for k, p in zip(uniq.tolist(), miss[first].tolist()):
            if len(self.idx) >= _MAX_MASKS:  # give up: those rows decode in full
                self.full = True
                break
            if k in self.keys:
                continue
            r, c = divmod(p, C)
            edges = np.nonzero(xpb[r, c] != xb[r])[0].astype(np.int64)
            self.keys[k] = len(self.idx)
            self.idx.append(edges)
            self._h2 = np.append(self._h2, np.int64(h2f[p]))
        ks = np.fromiter(self.keys.keys(), dtype=np.int64, count=len(self.keys))
        vs = np.fromiter(self.keys.values(), dtype=np.int32, count=len(self.keys))
        order = np.argsort(ks)
        self._kk, self._kid = ks[order], vs[order]


# =============================================================================
# heads
# =============================================================================

class _Head:
    """Shared geometry precompute, blocking and `pop_stats()` bookkeeping.

    Connected-set fast path
    -----------------------
    The base sign is a GF(2) quadratic form, so a configuration that differs
    from an already-decoded one by a KNOWN mask d needs no matrix product at
    all -- with l_b(c) = K[c, c] + (b Ksym)[c],

        q(b XOR d) = q(b) + [q(d) + sum_{e in d} K_ee] + sum_{e in d} l_b(e)
        l_{b XOR d}(c) = l_b(c) XOR (d Ksym)[c]
        gid(b XOR d)   = gid(b) XOR gid(d)

    where the bracket and (d Ksym) are mask constants. `sign_conn(x, xp)` pays
    ONE (B, N) x (N, N) pair of products for the B SAMPLES and then O(|d|) per
    connected row -- at L=4 OBC that is 263 connected rows served by one sample
    decode instead of 263 of them.

    Star gauge. For linear/vote/pt2 a mask that lies in the h=0 support W and
    satisfies W (Ksym d) = 0 shifts EVERY recovered term by the same constant
    (b XOR eps is on support, so <b XOR eps, Ksym d> = 0), hence
    s(b XOR d) = (-1)^{q(d)} s(b): the vertex stars are exactly this case
    (checked: all of them, with (-1)^{q(d)} = +1, at L = 2, 3, 4 OBC), so their
    connected rows cost nothing at all. The decorated plaquettes' x-pairs are
    NOT (they move the terms relative to each other) and neither is `cup`,
    which reads the base sign OFF support -- both go through the mask update
    above instead. `pop_stats()["n_decoded"]` counts the rows that were really
    decoded, `n_rows` still counts every row asked for.
    """

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
        self.Ksym = (self.K + self.K.T) % 2
        self._Ksym32 = self.Ksym.astype(np.float32)
        self._diagK = np.diag(self.K).astype(np.int32)
        self.support = None
        self._masks = None                      # lazy _ConnMasks
        self._n_tab = -1                        # masks already tabulated below
        self._reset_stats()

    # -- stats --------------------------------------------------------------

    def _reset_stats(self):
        self._stats = {"n_rows": 0, "n_decoded": 0, "n_fallback": 0, "n_pt2": 0,
                       "n_tie": 0, "k_max": 0, "k_sum": 0}

    def pop_stats(self):
        """Return the accumulated bookkeeping and reset the counters."""
        s = self._stats
        self._reset_stats()
        return s

    def _account_gid(self, gid):
        """Accumulate the per-row lit-class count from packed gids."""
        if gid is None or not len(gid):
            return
        nz, cnt = np.unique(gid, return_counts=True)
        cnt = dict(zip(nz.tolist(), cnt.tolist()))
        cnt = np.array([cnt[int(v)] for v in nz.tolist()], dtype=np.int64)
        k = np.array([int(v).bit_count() for v in nz.tolist()], dtype=np.int64)
        self._stats["k_sum"] += int((k * cnt).sum())
        if k.size:
            self._stats["k_max"] = max(self._stats["k_max"], int(k.max()))

    # -- evaluation ---------------------------------------------------------

    def __call__(self, configs):
        x = np.asarray(configs)
        lead, flat = x.shape[:-1], x.reshape(-1, x.shape[-1])
        self._stats["n_rows"] += flat.shape[0]
        return self._eval_raw(flat).reshape(lead)

    def _eval_raw(self, flat):
        """Blocked evaluation of (M, N) rows; counts decodes, not rows."""
        out = np.empty(flat.shape[0], dtype=np.float64)
        step = _block_rows(flat.shape[-1])
        for a in range(0, flat.shape[0], step):
            out[a:a + step] = self._eval_block(flat[a:a + step])
        self._stats["n_decoded"] += flat.shape[0]
        return out

    def _q0(self, b):
        """q(b) = b^T K b mod 2, one float32 GEMM + a row dot (mod once, at the
        end: the intermediate row sums are <= N^2 < 2^24, exact in float32)."""
        return np.einsum("ij,ij->i", b @ self._K32, b).astype(np.int64) & 1

    def _ell(self, b, cols=None):
        """l_b(c) = K[c, c] + (b Ksym)[c] mod 2, int8, over `cols` (default all).

        Values are <= N + 1, exact in float32; int8 because the recovery kernels
        are gather-bound, so the 4x narrower rows are a 4x traffic cut.
        """
        Ks = self._Ksym32 if cols is None else self._Ksym32[:, cols]
        dg = self._diagK if cols is None else self._diagK[cols]
        return (((b @ Ks).astype(np.int32) + dg) & 1).astype(np.int8)

    # -- connected-set fast path -------------------------------------------

    def _mask_tables(self):
        """Per-mask constants of the fast path, rebuilt when the dict grows."""
        ms = self._masks
        if self._n_tab == len(ms.idx):
            return
        n, N = len(ms.idx), self.geo.N
        width = max(1, max((e.size for e in ms.idx), default=1))
        pad = np.full((n, width), N, dtype=np.int32)
        cq = np.zeros(n, dtype=np.int64)
        dK = np.zeros((n, N), dtype=np.int8)
        gid = np.zeros(n, dtype=np.int64)
        gauge = np.zeros(n, dtype=bool)
        cst = np.ones(n, dtype=np.float64)
        Ki = self.K.astype(np.int64)
        for m, e in enumerate(ms.idx):
            pad[m, :e.size] = e
            v = np.zeros(N, dtype=np.int64)
            v[e] = 1
            qd = int(v @ Ki @ v) & 1
            cq[m] = (qd + int(Ki[e, e].sum())) & 1
            row = ((self.Ksym[e].sum(axis=0) & 1).astype(np.int8) if e.size
                   else np.zeros(N, dtype=np.int8))
            dK[m] = row
            cst[m] = 1.0 - 2.0 * qd
            if self.support is not None:
                gid[m] = self.support.gid_of_edges(e)
            gauge[m] = self._is_gauge(e, row, gid[m])
        self._mpad, self._mcq, self._mdK = pad, cq, dK
        self._mgid, self._mgauge, self._mc = gid, gauge, cst
        self._n_tab = n

    def _is_gauge(self, edges, dKrow, gid):
        """Does XOR-ing this mask in multiply the head by a CONSTANT sign?"""
        return not dKrow.any()          # cup: only when q(b + d) - q(d) = q(b)

    def sign_conn(self, x, xp):
        """(s(x), s(xp)) with every connected row served from its sample's decode.

        Bit-identical to (self(x), self(xp)) -- the mask update is exact GF(2)
        arithmetic and any row whose mask cannot be verified is decoded in full.
        """
        x, xp = np.asarray(x), np.asarray(xp)
        N, C = x.shape[-1], xp.shape[-2]
        lead = x.shape[:-1]
        xf, xpf = x.reshape(-1, N), xp.reshape(-1, C, N)
        B = xf.shape[0]
        if self._masks is None:
            self._masks = _ConnMasks(N)
            self._masks.disabled = not (np.abs(xf) == 1).all()
        if self._masks.disabled or C == 0:
            return self(x), self(xp)
        s = np.empty(B, dtype=np.float64)
        sp = np.empty((B, C), dtype=np.float64)
        # rows per chunk; when xp is not already float64 the mask matvec has to
        # cast it, so bound the chunk by ELEMENTS instead (that temporary is the
        # only place the fast path ever touches a full (rows, N) block of xp)
        step = max(1, min(B, (1 << 24) // max(1, C * N)
                          if xpf.dtype != np.float64 else (1 << 21) // max(1, C)))
        for a in range(0, B, step):
            s[a:a + step], sp[a:a + step] = self._conn_block(xf[a:a + step],
                                                             xpf[a:a + step])
        self._stats["n_rows"] += B * (C + 1)
        return s.reshape(lead), sp.reshape(lead + (C,))

    def _conn_block(self, xb, xpb):
        B, C, N = xpb.shape
        bb = (xb < 0).astype(np.float32)
        q0 = self._q0(bb)                                          # (B,)
        ell = self._ell(bb)                                        # (B, N) int8
        gid = self.support.gid(bb) if self.support is not None \
            else np.zeros(B, dtype=np.int64)
        self._account_gid(gid)
        s = self._eval_rows(gid, q0, _EllRows(ell))
        self._stats["n_decoded"] += B

        ids, ok = self._masks.identify(xb, xpb)
        self._mask_tables()
        samp = np.repeat(np.arange(B, dtype=np.int32), C)
        # parity of l_b over the mask support: one flat gather per mask slot
        # (column N of the padded l table is a permanent zero, so pad slots
        # contribute nothing) -- no (rows, width) temporary
        ellp = np.concatenate([ell, np.zeros((B, 1), np.int8)], axis=1).ravel()
        base = samp.astype(np.int64) * (self.geo.N + 1)
        mp = self._mpad
        pell = ellp[base + mp[ids, 0]]
        for j in range(1, mp.shape[1]):
            pell ^= ellp[base + mp[ids, j]]
        q0c = q0[samp] ^ self._mcq[ids] ^ pell
        gidc = gid[samp] ^ self._mgid[ids]
        self._account_gid(gidc if ok.all() else gidc[ok])   # `bad` self-accounts

        out = np.empty(B * C, dtype=np.float64)
        gsel = ok & self._mgauge[ids]
        out[gsel] = self._mc[ids[gsel]] * s[samp[gsel]]
        run = np.nonzero(ok & ~self._mgauge[ids])[0]
        if run.size:
            out[run] = self._eval_rows(
                gidc[run], q0c[run],
                _EllConn(ell, samp[run], self._mdK, ids[run]))
            self._stats["n_decoded"] += run.size
        bad = np.nonzero(~ok)[0]
        if bad.size:                                # never in practice; exact anyway
            out[bad] = self._eval_raw(xpb.reshape(-1, N)[bad])
        return s, out.reshape(B, C)

    def _eval_rows(self, gid, q0, src):
        raise NotImplementedError


class _EllRows:
    """l over a (M, N) table, addressed by row index.

    Both sources narrow to the coset's edge universe FIRST (a few hundred rows
    at most) and then take whole rows: a row gather of an (n, len(cols)) int8
    block is memcpy-shaped, while the (rows[:, None], cols[None, :]) broadcast
    it replaces is an element-by-element fancy index.
    """

    def __init__(self, ell):
        self.ell = ell

    def at(self, rows, cols):
        return self.ell[:, cols][np.asarray(rows)]


class _EllConn:
    """l of a connected row = l of its sample XOR the mask's (d Ksym) row."""

    def __init__(self, ell, samp, dK, ids):
        self.ell, self.samp, self.dK, self.ids = ell, samp, dK, ids

    def at(self, rows, cols):
        r = np.asarray(rows)
        return self.ell[:, cols][self.samp[r]] ^ self.dK[:, cols][self.ids[r]]


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
            self._account_gid(self.support.gid(b))
        return 1.0 - 2.0 * self._q0(b)

    def _eval_rows(self, gid, q0, src):
        return 1.0 - 2.0 * q0


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
        self._fx = np.ascontiguousarray(self.cup.I.T)   # (N, NP) single-flip flux
        self._cache: dict = {}                  # gid -> recovery structure

    def _is_gauge(self, edges, dKrow, gid):
        """A mask in the support W with W (Ksym d) = 0 shifts every recovered
        term by the same (-1)^{q(d)} -- so the whole head does (docstring)."""
        if gid:                                    # off support: classes move
            return False
        return not ((self.support.Wbasis.astype(np.int64)
                     @ dKrow.astype(np.int64)) & 1).any()

    # -- per-coset structures (built lazily, cached) -------------------------

    def _struct(self, gid):
        """Recovery structure for the coset whose lit-class vector packs to gid.

        `ecols` is the EDGE universe the row evaluation reads (the class edges,
        or just the representatives for `linear`); every column index stored
        here is local to it, so a caller only ever has to materialise l on those
        columns -- the whole point of the connected-set gather.
        """
        st = self._cache.get(gid)
        if st is not None:
            return st
        sup = self.support
        cls = [sup.classes[sup.lit[i]] for i in range(len(sup.lit))
               if (gid >> i) & 1]
        cls.sort(key=len)                    # largest class last: `_contract`'s
        m = len(cls)                         # single GEMM then does the most work
        rep = np.array([c[0] for c in cls], dtype=np.int64)
        if self.kind == "linear" or m == 0:
            ecols, rep_local = rep, np.arange(m, dtype=np.int64)
            local = []
        else:
            ecols, local, off = _stack_classes(cls)
            rep_local = off[:-1]
        st = {"gid": gid, "m": m, "cls": cls, "ecols": ecols, "local": local,
              "rep_local": rep_local, "capped": False, "second": None,
              "rep_const": int(_pair_const(rep[None, :], self.Ksym)[0]) if m else 0}
        if m:
            if self.kind == "vote":
                st.update(self._vote_struct(cls, local))
            elif self.kind == "pt2":
                st.update(self._pt_struct(cls, local, m))
        self._cache[gid] = st
        return st

    def _tensor(self, cls_edges, weights=None):
        """Flat contraction tensor: pair phase x optional exact-integer weights.

        Its dtype is the narrowest one that still represents every partial sum
        exactly (float32 below 2^23).
        """
        W = _pair_tensor(cls_edges, self.Ksym)
        if weights is not None:
            W = W * weights
        dt = np.float32 if float(np.abs(W).sum()) < float(1 << 23) else np.float64
        return W.astype(dt)

    def _vote_struct(self, cls, local):
        """One contraction per connected component of the lit classes."""
        comps = _components(cls, self.Ksym)
        for comp in comps:
            if len(comp) > self.k_cap or \
                    math.prod(len(cls[i]) for i in comp) > self.max_terms:
                return {"capped": True}
        out = []
        for comp in comps:
            sub = [cls[i] for i in comp]
            out.append(([local[i] for i in comp], self._tensor(sub),
                        tuple(len(c) for c in sub)))
        return {"comps": out, "capped": False}

    def _pt_struct(self, cls, local, m):
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
        return {"pt": (local, self._tensor(cls, weights=w),
                       tuple(len(c) for c in cls)), "capped": False}

    def _second_struct(self, st):
        """The (m+1)-flip tie-breaking order, as ONE MORE class product.

        The lit-class labels are independent, so an (m+1)-flip subset carrying
        the coset label can only be "one edge from each of the m lit classes,
        plus one edge of label 0" (any other multiset needs >= m + 2 flips).
        The tie-breaking candidates are therefore the class product C_0 x ... x
        C_{m-1} x C_unlit -- the same shape as the first-order stage, one axis
        longer -- and NOT the unstructured union the C(N, m+1) enumeration
        produced. Checked against that enumeration at L = 2, 3, 4 OBC.

        That turns the second order from a (rows x ~30k x k) gather into one
        (k+1)-fold contraction, which is what made `pt2` 55% of an L=4 step.
        The largest class is contracted LAST so the single GEMM does as much of
        the work as possible. `_MAX_ORDER2` is kept purely to FREEZE the head's
        verdicts: it no longer bounds any enumeration (the product is generated
        directly), but dropping it would newly enable the L=5 two-lit-class
        second order, which is a semantics change, not a speed one.
        """
        if st["second"] is not None:
            return st["second"]
        sup, m = self.support, st["m"]
        cls0 = sup.classes.get(0, [])
        cls = list(st["cls"]) + [cls0]
        n2 = math.prod(len(c) for c in cls)
        if (not cls0 or n2 == 0 or n2 > self.max_terms
                or n2 * (1 << (m + 1)) > _MAX_PT_WORK
                or math.comb(sup.N, m + 1) > _MAX_ORDER2):
            st["second"] = {"capped": True}
            return st["second"]
        cls.sort(key=len)                                        # largest last
        ecols2, local, _off = _stack_classes(cls)
        recs = np.array(list(itertools.product(*cls)), dtype=np.int64)
        w, ok = _integer_weights(_path_weights(recs, self._fx, self.J))
        if not ok:
            st["second"] = {"capped": True}
            return st["second"]
        st["ecols2"] = ecols2
        st["second"] = {"capped": False, "cols": local,
                        "W": self._tensor(cls, weights=w),
                        "shape": tuple(len(c) for c in cls)}
        return st["second"]

    # -- evaluation ---------------------------------------------------------

    def _eval_block(self, block):
        b = (block < 0).astype(np.float32)                         # (B, N)
        q0 = self._q0(b)                                           # (B,) 0/1
        gid = self.support.gid(b)
        self._account_gid(gid)
        return self._eval_rows(gid, q0, _EllBlock(self, b))

    def _eval_rows(self, gid, q0, src):
        """+-1 head values for rows already reduced to (gid, q0) + an l source."""
        out = np.empty(gid.shape[0], dtype=np.float64)
        n_lit = len(self.support.lit)
        groups = (np.nonzero(np.bincount(gid, minlength=1 << n_lit))[0]
                  if n_lit <= 20 and gid.size else np.unique(gid))
        for gv in groups.tolist():
            idx = np.nonzero(gid == gv)[0]
            if not idx.size:
                continue
            st = self._struct(int(gv))
            ell = src.at(idx, st["ecols"]) if st["ecols"].size else None
            out[idx] = self._eval_group(st, q0[idx], ell,
                                        lambda z, i=idx, s=st: src.at(
                                            i[z], self._ecols2(s)))
        return out

    def _ecols2(self, st):
        return st.get("ecols2", st["ecols"])

    def _linear(self, st, q0, ell):
        e = q0 + st["rep_const"]
        if st["m"]:
            e = e + ell[:, st["rep_local"]].sum(axis=1, dtype=np.int64)
        return 1.0 - 2.0 * (e & 1)

    def _eval_group(self, st, q0, ell, ell2):
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
            sec = self._second_struct(st)
            if sec["capped"]:
                self._stats["n_fallback"] += int(zero.sum())
            else:
                self._stats["n_pt2"] += int(zero.sum())
                z = np.nonzero(zero)[0]
                s[zero] = np.sign(self._contract(ell2(z), sec["cols"],
                                                 sec["W"], sec["shape"]))
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


class _EllBlock:
    """l materialised on demand from the raw bits (the no-conn block path)."""

    def __init__(self, head, b):
        self.head, self.b = head, b

    def at(self, rows, cols):
        return self.head._ell(self.b[np.asarray(rows)], cols)


def make_decoder_sign(kind, geo, stabs=None, k_cap=DEFAULT_K_CAP,
                      max_terms=DEFAULT_MAX_TERMS, J=1.0):
    """`sign_fn(configs) -> (...,) +-1 floats` for kind in cup|linear|vote|pt2.

    The returned callable carries `pop_stats()` (see the module docstring) and
    satisfies the `SignFramedOperator` contract: host numpy in, host numpy out,
    shape (..., N) -> (...). It also carries `sign_conn(x, xp) -> (s, sp)`,
    the connected-set fast path `SignFramedOperator.get_conn_padded` prefers;
    it is stateful only in its counters, its lazy per-coset cache and its
    connected-mask dictionary.
    """
    if kind == "cup":
        return CupHead(geo, stabs, k_cap=k_cap, max_terms=max_terms, J=J)
    return DecoderSign(kind, geo, stabs, k_cap=k_cap, max_terms=max_terms, J=J)
