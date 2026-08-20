"""Stage 1 decoder-extended sign head prototype (notes/decoder_sign_head_plan.md §3).

Extends the exact h=0 fixed-point sign S(b) = (-1)^{q(b)} (q a GF(2)-quadratic
form in edge bits b) from the h=0 support V to ALL 2^N configurations, via a
canonical error decomposition b = e(b) (+) v, v in Im(F) = span{stars, decorated
plaquette x-pairs} (a GF(2)-linear subspace of edge space), e(b) the coset
representative from a full-RREF-with-preimage-tracking reduction (mirrors
analysis/scripts/prefit_phase_head.py:202-217, including its reduce-against-EVERY-pivot
note).  Two sign gauges (error_last / error_first, plan §2) plus the production
head's off-support "v1 shadow" extrapolation (token-space RREF pullback,
prefit_phase_head.py:189-241) are built and cross-validated by the ladder V0-V4
(plan §3).  Pure numpy + Python-int bitmasks; no netket, no ED, no Hamiltonian.

Run: .venv/bin/python analysis/scripts/decoder_sign_prototype.py [--L 2|3] [--gauge ...]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque

import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import fermionic_plaquettes, _mask


# =============================================================================
# Generic GF(2) bitmask helpers
# =============================================================================
def popcount(m: int) -> int:
    return bin(m).count("1")


def random_mask(rng, nbits: int, p: float = 0.5) -> int:
    """Uniform-ish random subset of nbits bit positions as a Python-int mask.

    (numpy can't hold `1 << nbits` for nbits > ~63, so this stays in Python ints.)
    """
    if nbits == 0:
        return 0
    bits = rng.random(nbits) < p
    m = 0
    for i in np.flatnonzero(bits):
        m |= 1 << int(i)
    return m


def combine_rows(rows, mask: int) -> int:
    """XOR of rows[i] for i with bit i set in `mask`."""
    v, m = 0, mask
    while m:
        i = (m & -m).bit_length() - 1
        m &= m - 1
        v ^= rows[i]
    return v


# =============================================================================
# Geometry + stabilizer setup (mirrors analysis/scripts/prefit_phase_head.py:88-100)
# =============================================================================
def build_geom(L: int):
    geo = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc="PBC")
    xz = fermionic_plaquettes(geo, J=1.0)
    stars = [_mask(v) for v in geo.vertex_all]
    zxm = [(_mask(z), _mask(x)) for z, x, _ in xz]        # (z_mask, x_mask) per plaquette
    return geo, stars, zxm


# =============================================================================
# Step 1 (spec §1.1/1.2): RREF of Im(F) over edge space, with preimage tracking
#   F = {stars (sign-free), plaquette x-pair masks (sign-carrying)}
# =============================================================================
def build_pipeline(NSTAR: int, zxm, G):
    """RREF-with-preimage-tracking of the generator image Im(F) subset edge space.

    Full reduction against EVERY existing pivot (not just while the lowest bit
    is set) -- the prefit_phase_head.py:205-207 bug trap: partial RREF leaves
    higher pivot bits in reduced rows and breaks pivot-coordinate extraction.

    piv[c] = (row, preimage): `row` is the RREF basis vector with leading edge
    bit c; `preimage` is an NG-bit mask over generator indices (0..NSTAR-1 =
    stars, NSTAR..NSTAR+NP-1 = plaquette x-pairs) s.t. XOR of those generators'
    masks == row.  ker_basis collects generator-index relations found along the
    way (subsets whose masks XOR to zero -- used by V0a).
    """
    NP = len(zxm)
    NG = len(G)
    assert NG == NSTAR + NP
    piv: dict = {}
    ker_basis = []
    for gi in range(NG):
        v, pre = G[gi], 1 << gi
        for c in list(piv):
            if (v >> c) & 1:
                v ^= piv[c][0]
                pre ^= piv[c][1]
        if v:
            c = (v & -v).bit_length() - 1
            for k in list(piv):
                if (piv[k][0] >> c) & 1:
                    piv[k][0] ^= v
                    piv[k][1] ^= pre
            piv[c] = [v, pre]
        else:
            ker_basis.append(pre)
    piv = {c: (row, pre) for c, (row, pre) in piv.items()}

    # C_pq = |dp cap xpair_q| mod 2 (p!=q): the local TI sign form (application
    # variables). Must be symmetric -- that IS the plaquette commutation identity.
    C = np.zeros((NP, NP), dtype=np.int64)
    for p in range(NP):
        for q in range(NP):
            if p != q:
                C[p, q] = popcount(zxm[p][0] & zxm[q][1]) & 1
    assert np.array_equal(C, C.T), "C must be symmetric (commutation identity)"

    return dict(piv=piv, zxm=zxm, C=C, NSTAR=NSTAR, NP=NP, NG=NG, G=G,
                ker_basis=ker_basis, dimV=len(piv))


def reduce_with_preimage(pipe, b: int):
    """(e(b), generator-preimage of b^e(b)) via full pivot reduction."""
    r, pre = b, 0
    for c, (row, prerow) in pipe["piv"].items():
        if (r >> c) & 1:
            r ^= row
            pre ^= prerow
    return r, pre


def e_and_x(pipe, b: int):
    """Coset residual e(b) and plaquette application pattern x(b^e(b)) (spec §1.1)."""
    r, pre = reduce_with_preimage(pipe, b)
    NSTAR, NP = pipe["NSTAR"], pipe["NP"]
    x_bits = (pre >> NSTAR) & ((1 << NP) - 1)
    x_on = [p for p in range(NP) if (x_bits >> p) & 1]
    return r, x_on


def q_pairs(C, x_on) -> int:
    q = 0
    for i in range(len(x_on)):
        for j in range(i + 1, len(x_on)):
            q ^= int(C[x_on[i], x_on[j]])
    return q


def sign_error_last(pipe, b: int) -> int:
    """q(b) = sum_{p<q in x(b^e(b))} C_pq (spec §2 error-LAST)."""
    _, x_on = e_and_x(pipe, b)
    return -1 if q_pairs(pipe["C"], x_on) & 1 else 1


def sign_error_first(pipe, b: int) -> int:
    """q(b) = sum C_pq x_p x_q + sum_p x_p |dp cap e(b)| (spec §2 error-FIRST)."""
    e, x_on = e_and_x(pipe, b)
    q = q_pairs(pipe["C"], x_on)
    cross = 0
    for p in x_on:
        cross ^= popcount(pipe["zxm"][p][0] & e) & 1
    return -1 if (q ^ cross) & 1 else 1


def sign_gauge(pipe, b: int, gauge: str) -> int:
    return sign_error_last(pipe, b) if gauge == "error_last" else sign_error_first(pipe, b)


def apply_gen_subset(pipe, b: int, gen_mask: int):
    """Apply generators in gen_mask (increasing index order) to b, tracking sign
    (stars pure-X/sign-free; plaquette pairs pick up (-1) if |z & current|odd)."""
    s, sg = b, 1
    NSTAR, G, zxm = pipe["NSTAR"], pipe["G"], pipe["zxm"]
    m = gen_mask
    while m:
        gi = (m & -m).bit_length() - 1
        m &= m - 1
        if gi < NSTAR:
            s ^= G[gi]
        else:
            zb, xb = zxm[gi - NSTAR]
            if popcount(zb & s) & 1:
                sg = -sg
            s ^= xb
    return s, sg


def canonical_sign(pipe, v: int) -> int:
    """Sign of v in Im(F) via its RREF-canonical generator preimage."""
    r, pre = reduce_with_preimage(pipe, v)
    assert r == 0, "canonical_sign requires v in Im(F)"
    _, sg = apply_gen_subset(pipe, 0, pre)
    return sg


def draw_class(pipe, rng):
    """One uniform-random product of plaquette pairs applied to |0..0>, sign
    tracked (mirrors prefit_phase_head.py:123-135; stars omitted -- sign-free,
    token-free, and V0b independently verifies star invariance)."""
    s, sg = 0, 1
    NP, zxm = pipe["NP"], pipe["zxm"]
    for k in np.flatnonzero(rng.integers(0, 2, size=NP)):
        zb, xb = zxm[int(k)]
        if popcount(zb & s) & 1:
            sg = -sg
        s ^= xb
    return s, sg


def bfs_signs(pipe):
    """Exact BFS enumeration of the h=0 support with signs (L=2 only; mirrors
    prefit_phase_head.py:155-177, kept as dict instead of the numpy-array form)."""
    NSTAR = pipe["NSTAR"]
    stars = pipe["G"][:NSTAR]
    sign = {0: 1}
    q = deque([0])
    while q:
        s = q.popleft()
        for st in stars:
            t = s ^ st
            if t not in sign:
                sign[t] = sign[s]
                q.append(t)
        for zb, xb in pipe["zxm"]:
            t = s ^ xb
            if t not in sign:
                sign[t] = sign[s] * (-1 if popcount(zb & s) & 1 else 1)
                q.append(t)
    return sign


# =============================================================================
# v1 "shadow" gauge: token-space RREF pullback (prefit_phase_head.py:189-241)
# =============================================================================
def build_v1(pipe):
    NP, C = pipe["NP"], pipe["C"]
    cols = []
    for p in range(NP):
        v = 0
        for q in range(NP):
            if C[q, p]:
                v |= 1 << q
        cols.append(v)
    piv: dict = {}
    for p in range(NP):
        v, pre = cols[p], 1 << p
        for c in list(piv):
            if (v >> c) & 1:
                v ^= piv[c][0]
                pre ^= piv[c][1]
        if v:
            c = (v & -v).bit_length() - 1
            for k in list(piv):
                if (piv[k][0] >> c) & 1:
                    piv[k][0] ^= v
                    piv[k][1] ^= pre
            piv[c] = [v, pre]
    pivots = sorted(piv)
    d = len(pivots)
    P = np.zeros((d, NP), dtype=np.int64)
    for i, c in enumerate(pivots):
        for p in range(NP):
            P[i, p] = (piv[c][1] >> p) & 1
    Cu = np.triu(C, 1)
    l = np.zeros(NP, dtype=np.int64)
    Q = np.zeros((NP, NP), dtype=np.int64)
    lp_ = (np.einsum("ip,pq,iq->i", P, Cu, P) & 1)
    Bm = (P @ C @ P.T) & 1
    for i in range(d):
        l[pivots[i]] = lp_[i]
        for j in range(i + 1, d):
            Q[pivots[i], pivots[j]] = Bm[i, j]
    zedges = [zb for zb, _ in pipe["zxm"]]
    return dict(l=l, Q=Q, zedges=zedges, dim=d)


def sign_v1(v1ctx, b: int) -> int:
    zedges = v1ctx["zedges"]
    t_on = [p for p, zb in enumerate(zedges) if popcount(zb & b) & 1]
    q = 0
    for p in t_on:
        q ^= int(v1ctx["l"][p])
    for i in range(len(t_on)):
        for j in range(i + 1, len(t_on)):
            q ^= int(v1ctx["Q"][t_on[i], t_on[j]])
    return -1 if q & 1 else 1


def sign_any(pipe, v1ctx, b: int, gauge: str) -> int:
    if gauge == "v1_shadow":
        return sign_v1(v1ctx, b)
    return sign_gauge(pipe, b, gauge)


# =============================================================================
# V0a: generator relations (ker(F) carries phase +1)
# =============================================================================
def run_V0a(pipe, rng, n_k=200, n_x=100, N=None, restrict_support=False):
    """b restricted to Im(F) (restrict_support=True) diagnoses whether a <1.0
    match rate is a genuine ghost-flux-sector phase (relation carries a
    state-dependent phase off the physical support -- flux_constraint_masks'
    "Gauss laws" story) rather than a bug: it should read 1.0 on-support even
    if the unrestricted (b uniform over 2^N) test does not."""
    ker_basis = pipe["ker_basis"]
    NG = pipe["NG"]
    if not ker_basis:
        return {"n_pairs": 0, "match_rate": None,
                "note": "ker(F) trivial (generators independent)"}
    pivot_rows = [row for row, _ in pipe["piv"].values()]
    d = len(pivot_rows)
    total = match = 0
    for _ in range(n_k):
        kmask = random_mask(rng, len(ker_basis))
        k = combine_rows(ker_basis, kmask)
        for _ in range(n_x):
            x = random_mask(rng, NG)
            b = combine_rows(pivot_rows, random_mask(rng, d)) if restrict_support \
                else random_mask(rng, N)
            s1, sg1 = apply_gen_subset(pipe, b, x)
            s2, sg2 = apply_gen_subset(pipe, b, x ^ k)
            assert s1 == s2, "k not actually in ker(F) -- construction bug"
            total += 1
            match += int(sg1 == sg2)
    return {"n_pairs": total, "match_rate": match / total,
            "n_ker_basis": len(ker_basis), "restrict_support": restrict_support}


# =============================================================================
# V0b: star invariance S(b) == S(b ^ star)
# =============================================================================
def run_V0b(pipe, v1ctx, gauges, rng, N, n_samples=10000):
    stars = pipe["G"][:pipe["NSTAR"]]
    out = {}
    for g in gauges + ["v1_shadow"]:
        match = 0
        for _ in range(n_samples):
            b = random_mask(rng, N)
            st = stars[int(rng.integers(len(stars)))]
            s1 = sign_any(pipe, v1ctx, b, g)
            s2 = sign_any(pipe, v1ctx, b ^ st, g)
            match += int(s1 == s2)
        out[g] = {"n": n_samples, "match_rate": match / n_samples}
    return out


# =============================================================================
# V0c: edge-relabeling gauge-dependence -> affine (-1)^ell(b) pattern per sector
# =============================================================================
def run_V0c(pipe, gauges, rng, N, n_weight2=20, n_test=200):
    perm = rng.permutation(N)

    def pm(m):
        out, mm = 0, m
        while mm:
            i = (mm & -mm).bit_length() - 1
            mm &= mm - 1
            out |= 1 << int(perm[i])
        return out

    G2 = [pm(g) for g in pipe["G"]]
    zxm2 = [(pm(z), pm(x)) for z, x in pipe["zxm"]]
    pipe2 = build_pipeline(pipe["NSTAR"], zxm2, G2)
    C_relabel_invariant = bool(np.array_equal(pipe2["C"], pipe["C"]))

    free_bits = [c for c in range(N) if c not in pipe["piv"]]
    pivot_rows = [row for row, _ in pipe["piv"].values()]
    d = len(pivot_rows)

    sectors = [("weight0", 0)]
    for fb in free_bits:
        sectors.append(("weight1", 1 << fb))
    if len(free_bits) >= 2:
        for _ in range(min(n_weight2, len(free_bits) * (len(free_bits) - 1) // 2)):
            i, j = rng.choice(free_bits, size=2, replace=False)
            sectors.append(("weight2", (1 << int(i)) | (1 << int(j))))

    report = {}
    for gauge in gauges:
        def g_of(vmask, e0):
            v = combine_rows(pivot_rows, vmask)
            b = e0 ^ v
            return sign_gauge(pipe, b, gauge) * sign_gauge(pipe2, pm(b), gauge)

        per_sector = []
        for label, e0 in sectors:
            g0 = g_of(0, e0)
            abits = 0
            for i in range(d):
                if g_of(1 << i, e0) != g0:
                    abits |= 1 << i
            c0 = 0 if g0 == 1 else 1
            mism = 0
            for _ in range(n_test):
                vmask = random_mask(rng, d)
                actual = g_of(vmask, e0)
                pred_bit = c0 ^ (popcount(vmask & abits) & 1)
                predicted = 1 if pred_bit == 0 else -1
                mism += int(actual != predicted)
            per_sector.append({"label": label, "e0": e0, "n_test": n_test,
                                "mismatches": mism, "mismatch_fraction": mism / n_test})
        total_mism = sum(s["mismatches"] for s in per_sector)
        total_n = sum(s["n_test"] for s in per_sector)
        report[gauge] = {"n_sectors": len(sectors),
                          "overall_mismatch_fraction": total_mism / total_n,
                          "per_sector": per_sector}
    return {"C_relabel_invariant": C_relabel_invariant, "dimV": d,
            "n_free_bits": len(free_bits), "gauges": report}


# =============================================================================
# V1: on-support exactness vs BFS
# =============================================================================
def run_V1_full(pipe, v1ctx, ground, gauges):
    out = {}
    for gauge in gauges:
        mism = sum(1 for b, sg in ground.items() if sign_any(pipe, v1ctx, b, gauge) != sg)
        out[gauge] = {"n": len(ground), "mismatches": mism,
                       "match_rate": 1 - mism / len(ground)}
    return out


def run_V1_sampled(pipe, gauges, rng, n_sample):
    out = {g: {"n": 0, "mismatches": 0} for g in gauges}
    for _ in range(n_sample):
        s, sg = draw_class(pipe, rng)
        for g in gauges:
            out[g]["n"] += 1
            if sign_gauge(pipe, s, g) != sg:
                out[g]["mismatches"] += 1
    for g in gauges:
        out[g]["match_rate"] = 1 - out[g]["mismatches"] / out[g]["n"]
    return out


# =============================================================================
# V2: weight-1 sector frustration + PT match
# =============================================================================
def enumerate_weight1_full(pipe, support, N):
    w1 = {}
    for v, sv in support.items():
        for i in range(N):
            b = v ^ (1 << i)
            if b not in support:
                e_b, _ = reduce_with_preimage(pipe, b)
                if popcount(e_b) == 1:
                    w1.setdefault(b, []).append((v, sv))
    return w1


def sample_weight1(pipe, rng, N, n_target, max_tries_mult=25):
    """L>=3: sample states one flip from Im(F); exact local N-neighbor parent
    scan per candidate (membership test is cheap: O(dimV) per neighbor)."""
    pivot_rows = [row for row, _ in pipe["piv"].values()]
    d = len(pivot_rows)
    w1, sign_cache = {}, {}
    tries = 0
    while len(w1) < n_target and tries < n_target * max_tries_mult:
        tries += 1
        v = combine_rows(pivot_rows, random_mask(rng, d))
        i = int(rng.integers(N))
        b = v ^ (1 << i)
        if b in w1:
            continue
        e_b, _ = reduce_with_preimage(pipe, b)
        if popcount(e_b) != 1:
            continue
        parents = []
        for j in range(N):
            cand = b ^ (1 << j)
            e_c, pre_c = reduce_with_preimage(pipe, cand)
            if e_c == 0:
                if cand not in sign_cache:
                    sign_cache[cand] = canonical_sign(pipe, cand)
                parents.append((cand, sign_cache[cand]))
        w1[b] = parents
    return w1


def analyze_weight1(w1):
    """Frustration + parent-structure diagnostics for the weight-1 pool.

    Oracle bond-match ceiling (plan §3 task-3): a gauge assigns ONE sign S(b)
    per state b, then is scored bond-by-bond against every support parent of b
    (bond_match_rate, as computed downstream). For a fixed state with k parents
    split n_+ agreeing / n_- disagreeing (k = n_+ + n_-), the best any single
    per-state sign choice can do is match max(n_+, n_-) of the k bonds (choose
    the majority sign; on an exact tie n_+==n_- this reduces to k/2, i.e. 50% --
    no fixed choice beats a coin flip). Aggregated over the whole pool:

        ceiling = [ sum_b max(n_+(b), n_-(b)) ] / [ sum_b k(b) ]

    which is exactly "match every majority-side bond, tie-break states at 50%",
    computed exactly (not by simulating a tie-break) since max(n_+,n_-) already
    equals k/2 in the tied case.
    """
    frustrated, common = 0, {}
    parent_count_hist: dict = {}
    oracle_num = oracle_den = 0
    for b, plist in w1.items():
        signs = {sg for _, sg in plist}
        k = len(plist)
        parent_count_hist[k] = parent_count_hist.get(k, 0) + 1
        n_plus = sum(1 for _, sg in plist if sg == 1)
        n_minus = k - n_plus
        oracle_num += max(n_plus, n_minus)
        oracle_den += k
        if len(signs) > 1:
            frustrated += 1
        else:
            common[b] = next(iter(signs))
    return {"n": len(w1), "n_frustrated": frustrated,
            "frustrated_fraction": (frustrated / len(w1)) if w1 else None,
            "common": common,
            "parent_count_histogram": dict(sorted(parent_count_hist.items())),
            "oracle_bond_match_ceiling": (oracle_num / oracle_den) if oracle_den else None,
            "oracle_num": oracle_num, "oracle_den": oracle_den}


def v2_gauge_metrics(pipe, v1ctx, w1, common, gauge):
    s_cache = {}

    def S(b):
        if b not in s_cache:
            s_cache[b] = sign_any(pipe, v1ctx, b, gauge)
        return s_cache[b]

    match_common = sum(1 for b, cs in common.items() if S(b) == cs)
    bond_total = sum(len(pl) for pl in w1.values())
    bond_match = sum(1 for b, pl in w1.items() for _, sg in pl if S(b) == sg)
    return {"match_rate_common_parent": (match_common / len(common)) if common else None,
            "n_unfrustrated": len(common),
            "bond_match_rate": (bond_match / bond_total) if bond_total else None,
            "n_bonds": bond_total}


# =============================================================================
# V3: stoquasticity / violation audit
# =============================================================================
def sample_onsupport(pipe, rng, n_target, max_tries_mult=25):
    """L>=3 V3 pool component (i): repeated draw_class draws, deduplicated by
    resulting state (draw_class's own tracked sign is irrelevant here -- V3
    recomputes S(b) per gauge itself; only the pool of states is needed)."""
    out = set()
    tries = 0
    while len(out) < n_target and tries < n_target * max_tries_mult:
        tries += 1
        s, _ = draw_class(pipe, rng)
        out.add(s)
    return out


def sample_weight2(pipe, rng, N, n, max_tries_mult=20):
    pivot_rows = [row for row, _ in pipe["piv"].values()]
    d = len(pivot_rows)
    out = set()
    tries = 0
    while len(out) < n and tries < n * max_tries_mult:
        tries += 1
        v = combine_rows(pivot_rows, random_mask(rng, d))
        i, j = rng.choice(N, size=2, replace=False)
        b = v ^ (1 << int(i)) ^ (1 << int(j))
        r, _ = reduce_with_preimage(pipe, b)
        if popcount(r) == 2:
            out.add(b)
    return out


def run_V3(pipe, v1ctx, E, N, NP, hx=0.3):
    gauges = ["v1_shadow", "error_last", "error_first"]
    caches = {g: {} for g in gauges}

    def S(g, b):
        c = caches[g]
        if b not in c:
            c[b] = sign_any(pipe, v1ctx, b, g)
        return c[b]

    wcache = {}

    def wt(b):
        if b not in wcache:
            r, _ = reduce_with_preimage(pipe, b)
            wcache[b] = popcount(r)
        return wcache[b]

    zxm = pipe["zxm"]
    report = {}
    for g in gauges:
        cBp = {"n": 0, "viol": 0, "wsum": 0.0, "wviol": 0.0}
        cSx = {"n": 0, "viol": 0, "wsum": 0.0, "wviol": 0.0}
        for b in E:
            sb = S(g, b)
            wb = wt(b)
            for p in range(NP):
                zb, xb = zxm[p]
                par = popcount(zb & b) & 1
                esign = -1 if par == 0 else 1
                bp = b ^ xb
                sbp = S(g, bp)
                viol = (sbp * esign * sb) > 0
                w = hx ** (wb + wt(bp))
                cBp["n"] += 1
                cBp["viol"] += int(viol)
                cBp["wsum"] += w
                cBp["wviol"] += w * int(viol)
            for e in range(N):
                bp = b ^ (1 << e)
                sbp = S(g, bp)
                viol = (sbp * (-1) * sb) > 0
                w = hx ** (wb + wt(bp))
                cSx["n"] += 1
                cSx["viol"] += int(viol)
                cSx["wsum"] += w
                cSx["wviol"] += w * int(viol)
        report[g] = {
            "Bp": {"raw_violation_fraction": cBp["viol"] / cBp["n"],
                   "pt_weighted_violation_fraction": cBp["wviol"] / cBp["wsum"],
                   "n_bonds": cBp["n"]},
            "sx": {"raw_violation_fraction": cSx["viol"] / cSx["n"],
                   "pt_weighted_violation_fraction": cSx["wviol"] / cSx["wsum"],
                   "n_bonds": cSx["n"]},
        }
    return report


def report_V3(rows, results, v3):
    """Append go/no-go + per-gauge/family rows for a V3 report dict (shared by
    the L=2 exhaustive-pool path and the L>=3 sampled-pool path -- identical
    logic, extracted so the L=2 output is provably unchanged by the L>=3
    generalization)."""
    results["V3"] = v3
    results["V3"]["note_hz"] = ("h_z terms are diagonal and cannot violate "
                                 "stoquasticity; both field points (hx=0.3,hz=0) and "
                                 "(0.3,0.15) give IDENTICAL V3 numbers since only hx "
                                 "enters the PT weighting -- reported once.")
    # go/no-go on PT-WEIGHTED fractions (the physically relevant metric --
    # raw fractions are dominated by generic weight-1 bulk states far off
    # support, where no gauge has meaningful sign structure; reported too).
    def pt_sum(g):
        return (v3[g]["Bp"]["pt_weighted_violation_fraction"]
                + v3[g]["sx"]["pt_weighted_violation_fraction"])

    def raw_sum(g):
        return v3[g]["Bp"]["raw_violation_fraction"] + v3[g]["sx"]["raw_violation_fraction"]

    best_v2_pt = min(pt_sum("error_last"), pt_sum("error_first"))
    v1_pt = pt_sum("v1_shadow")
    best_v2_raw = min(raw_sum("error_last"), raw_sum("error_first"))
    v1_raw = raw_sum("v1_shadow")
    rows.append(("V3 go/no-go (PT-weighted, v2 << v1)", pf(best_v2_pt < v1_pt),
                 f"best_v2_pt_sum={best_v2_pt:.4f}  v1_shadow_pt_sum={v1_pt:.4f}"))
    rows.append(("V3 go/no-go (raw, v2 << v1)", pf(best_v2_raw < v1_raw),
                 f"best_v2_raw_sum={best_v2_raw:.4f}  v1_shadow_raw_sum={v1_raw:.4f}"))
    for g in ("v1_shadow", "error_last", "error_first"):
        for fam in ("Bp", "sx"):
            d = v3[g][fam]
            rows.append((f"V3 {g}/{fam}", "INFO",
                         f"raw={d['raw_violation_fraction']:.4f}  "
                         f"PT-weighted={d['pt_weighted_violation_fraction']:.4f}  "
                         f"n={d['n_bonds']}"))


# =============================================================================
# Reporting
# =============================================================================
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def pf(ok):
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gauge", choices=["error_last", "error_first", "both"], default="both")
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--n_sample", type=int, default=2000)
    ap.add_argument("--n_v3", type=int, default=None,
                     help="L>=3 only: target size per V3 pool component (on-support, "
                          "weight1, weight2); defaults to --n_sample. Decoupled because "
                          "run_V3 cost is O(pool_size*(N+NP)) and much steeper than V1/V2.")
    ap.add_argument("--hx", type=float, default=0.3)
    args = ap.parse_args()
    if args.json_out is None:
        args.json_out = f"results/fermionic_h0/decoder_gauge_L{args.L}.json"

    rng = np.random.default_rng(args.seed)
    gauges = ["error_last", "error_first"] if args.gauge == "both" else [args.gauge]

    t_all = time.perf_counter()
    geo, stars, zxm = build_geom(args.L)
    N, NP, NSTAR = geo.N, len(zxm), len(stars)
    G = stars + [xb for _, xb in zxm]
    pipe = build_pipeline(NSTAR, zxm, G)
    v1ctx = build_v1(pipe)
    print(f"L={args.L}  N={N}  NP={NP}  NSTAR={NSTAR}  dimV={pipe['dimV']} "
          f"(|support|=2^{pipe['dimV']})  v1 pivot dim={v1ctx['dim']}")

    results = {"L": args.L, "seed": args.seed, "N": N, "NP": NP, "NSTAR": NSTAR,
               "dimV": pipe["dimV"], "gauge_arg": args.gauge, "n_sample": args.n_sample,
               "timing_sec": {}}
    rows = []  # (name, status, detail)

    def timed(name, fn):
        t0 = time.perf_counter()
        r = fn()
        dt = time.perf_counter() - t0
        results["timing_sec"][name] = dt
        print(f"  [{name}] {dt:.1f}s")
        return r

    # ---- V0a ----
    print("V0a: generator relations (ker(F) phase)...")
    v0a = timed("V0a", lambda: run_V0a(pipe, rng, N=N))
    v0a_supp = timed("V0a_on_support", lambda: run_V0a(pipe, rng, N=N, restrict_support=True))
    results["V0a"] = v0a
    results["V0a_on_support"] = v0a_supp
    ok = v0a.get("match_rate") is None or v0a["match_rate"] == 1.0
    ok_supp = v0a_supp.get("match_rate") is None or v0a_supp["match_rate"] == 1.0
    rows.append(("V0a ker(F) phase +1 (b~U(2^N))", pf(ok),
                 f"match_rate={v0a.get('match_rate')} n={v0a.get('n_pairs')}"))
    rows.append(("V0a ker(F) phase +1 (b in support)", pf(ok_supp),
                 f"match_rate={v0a_supp.get('match_rate')} n={v0a_supp.get('n_pairs')}"))

    # ---- V0b ----
    print("V0b: star invariance...")
    v0b = timed("V0b", lambda: run_V0b(pipe, v1ctx, gauges, rng, N))
    results["V0b"] = v0b
    for g, r in v0b.items():
        rows.append((f"V0b star-inv [{g}]", pf(r["match_rate"] == 1.0),
                     f"match_rate={r['match_rate']:.6f} n={r['n']}"))

    # ---- V0c ----
    print("V0c: edge-relabeling gauge dependence...")
    v0c = timed("V0c", lambda: run_V0c(pipe, gauges, rng, N))
    results["V0c"] = v0c
    for g, r in v0c["gauges"].items():
        rows.append((f"V0c affine-relabel [{g}]", pf(r["overall_mismatch_fraction"] == 0.0),
                     f"mismatch_frac={r['overall_mismatch_fraction']:.6f} "
                     f"n_sectors={r['n_sectors']}"))

    if args.L == 2:
        # ---- V1 (full BFS) ----
        print("V1: BFS enumeration + exactness check (L=2 exhaustive)...")
        support = timed("bfs_enum", lambda: bfs_signs(pipe))
        print(f"  |support| = {len(support)}")
        v1 = timed("V1", lambda: run_V1_full(pipe, v1ctx, support, gauges + ["v1_shadow"]))
        results["V1"] = v1
        for g, r in v1.items():
            rows.append((f"V1 on-support exact [{g}]", pf(r["mismatches"] == 0),
                         f"match_rate={r['match_rate']:.8f} mismatches={r['mismatches']}/{r['n']}"))

        # ---- V2 (full) ----
        print("V2: weight-1 sector frustration + PT match (exhaustive)...")
        w1 = timed("V2_enum", lambda: enumerate_weight1_full(pipe, support, N))
        wa = analyze_weight1(w1)
        print(f"  |weight1| = {wa['n']}  frustrated = {wa['n_frustrated']} "
              f"({wa['frustrated_fraction']:.4f})")
        v2 = {"n_weight1": wa["n"], "n_frustrated": wa["n_frustrated"],
              "frustrated_fraction": wa["frustrated_fraction"], "gauges": {}}
        for g in gauges:
            m = timed(f"V2_{g}", lambda g=g: v2_gauge_metrics(pipe, v1ctx, w1, wa["common"], g))
            v2["gauges"][g] = m
            rows.append((f"V2 PT-match [{g}]",
                         "INFO", f"common_parent_match={m['match_rate_common_parent']:.4f} "
                         f"bond_match={m['bond_match_rate']:.4f} "
                         f"(n_unfrustrated={m['n_unfrustrated']}, n_bonds={m['n_bonds']})"))
        rows.append(("V2 frustration fraction", "INFO",
                     f"{wa['frustrated_fraction']:.4f} ({wa['n_frustrated']}/{wa['n']})"))
        rows.append(("V2 parent-count histogram", "INFO", f"{wa['parent_count_histogram']}"))
        rows.append(("V2 oracle bond-match ceiling", "INFO",
                     f"{wa['oracle_bond_match_ceiling']:.4f} "
                     f"(num={wa['oracle_num']}, den={wa['oracle_den']})"))
        for g in gauges:
            gap = wa["oracle_bond_match_ceiling"] - v2["gauges"][g]["bond_match_rate"]
            rows.append((f"V2 ceiling gap [{g}]", "INFO",
                         f"bond_match={v2['gauges'][g]['bond_match_rate']:.4f} "
                         f"ceiling={wa['oracle_bond_match_ceiling']:.4f} gap={gap:.4f}"))
        v2["parent_count_histogram"] = wa["parent_count_histogram"]
        v2["oracle_bond_match_ceiling"] = wa["oracle_bond_match_ceiling"]
        v2["oracle_num"], v2["oracle_den"] = wa["oracle_num"], wa["oracle_den"]
        results["V2"] = v2

        # ---- V3 ----
        print("V3: stoquasticity / violation audit (go/no-go)...")
        E = set(support.keys()) | set(w1.keys())
        w2 = timed("V3_sample_weight2", lambda: sample_weight2(pipe, rng, N, args.n_sample))
        E |= w2
        print(f"  |E| = {len(E)}  (support {len(support)}, weight1 {len(w1)}, "
              f"weight2 sample {len(w2)})")
        v3 = timed("V3", lambda: run_V3(pipe, v1ctx, E, N, NP, hx=args.hx))
        report_V3(rows, results, v3)
    else:
        # ---- V4 reduced ladder (L>=3) ----
        print(f"V4 (L={args.L}) reduced ladder: V1-sampled, V2-sampled...")
        v1s = timed("V1_sampled", lambda: run_V1_sampled(pipe, gauges, rng, args.n_sample))
        results["V1_sampled"] = v1s
        for g, r in v1s.items():
            rows.append((f"V4 V1-sampled [{g}]", pf(r["mismatches"] == 0),
                         f"match_rate={r['match_rate']:.6f} "
                         f"mismatches={r['mismatches']}/{r['n']}"))

        w1s = timed("V2_sample_weight1", lambda: sample_weight1(pipe, rng, N, args.n_sample))
        wa = analyze_weight1(w1s)
        print(f"  |weight1 sampled| = {wa['n']}  frustrated = {wa['n_frustrated']} "
              f"({wa['frustrated_fraction']:.4f})")
        v2s = {"n_weight1": wa["n"], "n_frustrated": wa["n_frustrated"],
               "frustrated_fraction": wa["frustrated_fraction"], "gauges": {}}
        for g in gauges:
            m = timed(f"V2_sampled_{g}", lambda g=g: v2_gauge_metrics(pipe, v1ctx, w1s, wa["common"], g))
            v2s["gauges"][g] = m
            rows.append((f"V4 V2-sampled PT-match [{g}]", "INFO",
                         f"common_parent_match={m['match_rate_common_parent']:.4f} "
                         f"bond_match={m['bond_match_rate']:.4f} "
                         f"(n_unfrustrated={m['n_unfrustrated']}, n_bonds={m['n_bonds']})"))
        rows.append(("V4 frustration fraction", "INFO",
                     f"{wa['frustrated_fraction']:.4f} ({wa['n_frustrated']}/{wa['n']})"))
        rows.append(("V4 parent-count histogram", "INFO", f"{wa['parent_count_histogram']}"))
        rows.append(("V4 oracle bond-match ceiling", "INFO",
                     f"{wa['oracle_bond_match_ceiling']:.4f} "
                     f"(num={wa['oracle_num']}, den={wa['oracle_den']})"))
        for g in gauges:
            gap = wa["oracle_bond_match_ceiling"] - v2s["gauges"][g]["bond_match_rate"]
            rows.append((f"V4 ceiling gap [{g}]", "INFO",
                         f"bond_match={v2s['gauges'][g]['bond_match_rate']:.4f} "
                         f"ceiling={wa['oracle_bond_match_ceiling']:.4f} gap={gap:.4f}"))
        v2s["parent_count_histogram"] = wa["parent_count_histogram"]
        v2s["oracle_bond_match_ceiling"] = wa["oracle_bond_match_ceiling"]
        v2s["oracle_num"], v2s["oracle_den"] = wa["oracle_num"], wa["oracle_den"]
        results["V2_sampled"] = v2s

        # ---- V3 (generalized: sampled pool, plan §3 task-2) ----
        # n_v3 is decoupled from n_sample: run_V3 costs O(|E|*(N+NP)) RREF
        # reductions (empirically ~0.02s/state at L=3, ~0.29s/state at L=4),
        # so a smaller dedicated pool keeps L=4 wall-clock sane while V1/V2
        # (cheap) still get the full n_sample for good frustration statistics.
        n_v3 = args.n_v3 if args.n_v3 is not None else args.n_sample
        results["n_v3"] = n_v3
        print(f"V3 (L={args.L}) sampled stoquasticity/violation audit (go/no-go), "
              f"n_v3={n_v3} per pool component...")
        onsupp = timed("V3_sample_onsupport", lambda: sample_onsupport(pipe, rng, n_v3))
        w1_v3 = timed("V3_sample_weight1", lambda: sample_weight1(pipe, rng, N, n_v3))
        w2 = timed("V3_sample_weight2", lambda: sample_weight2(pipe, rng, N, n_v3))
        E = onsupp | set(w1_v3.keys()) | w2
        print(f"  |E| = {len(E)}  (on-support sample {len(onsupp)}, weight1 sample "
              f"{len(w1_v3)}, weight2 sample {len(w2)})")
        v3 = timed("V3", lambda: run_V3(pipe, v1ctx, E, N, NP, hx=args.hx))
        report_V3(rows, results, v3)

    results["runtime_sec_total"] = time.perf_counter() - t_all

    # ---- print table ----
    print("\n" + "=" * 78)
    print(f"DECODER SIGN PROTOTYPE -- L={args.L}  gauge_arg={args.gauge}  seed={args.seed}")
    print("=" * 78)
    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"  {name:<{width}}  [{status:<4}]  {detail}")
    print("=" * 78)
    print(f"total runtime: {results['runtime_sec_total']:.1f}s")

    import os
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(to_jsonable(results), f, indent=2)
    print(f"metrics dumped to {args.json_out}")


if __name__ == "__main__":
    main()
