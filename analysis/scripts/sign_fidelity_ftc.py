"""Gate-0 sign-fidelity diagnostic for the 3D fermionic toric code (ED sizes).

Question: how much of the exact ground state's SIGN structure can a deterministic,
training-free head reproduce once h_x populates the off-support (flux-violating)
sectors?  Everything is exhaustive over the 2^N computational basis, so the
numbers are ceilings, not estimates.

Heads compared at every (h_x, h_z) point (all anchored, no free global flip):
  anaC   the frozen analytic token-quadratic q(t) (exact on the h=0 support,
         solver-gauge shadow off it) -- the production head
  linear GF(2)-linear decoder: one FIXED representative edge per lit line class
         (lowest index; `linear_hi` = highest index is the opposite tie-break)
  vote   majority vote of (-1)^q over ALL minimal recoveries (nonlinear,
         background-dependent); exact ties fall back to `linear`
  pt2    leading-order perturbation theory with EXACT energy denominators:
         sign of sum_eps (-1)^{q(sigma+eps)} D(eps), D = sum over orderings of
         prod_j 1/DeltaE(A_j), DeltaE(A) = 2J * #{p : |dp cap A| odd}.  When the
         first-order sum vanishes exactly, the next order (one extra flip; for a
         single lit class that is the two-flip paths sigma <- e1 <- e2 <- support)
         decides.
  exact  sign(psi_ED) (ceiling = 1);  plus  all +1 (sign-blind ceiling)

Support/syndrome structure (computed, not assumed): the h=0 support is
W = span(stars, x-pairs) over edge bits; W = (W^perp)^perp, so membership is a
handful of Z-string parities -- and those parities are exactly the closed-surface
flux constraints `flux_constraint_masks`.  Edges group into line classes by their
single-flip syndrome; a minimal recovery picks one edge per lit class.

Run from the repo root:
    python analysis/scripts/sign_fidelity_ftc.py --Lx 2 --Ly 2 --Lz 2 --bc OBC
    python analysis/scripts/sign_fidelity_ftc.py --Lx 2 --Ly 2 --Lz 3 --bc OBC
"""
import argparse
import itertools
import json
import os

import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import fermionic_plaquettes, flux_constraint_masks, _mask
from tc3d.exact_diag import hamiltonian_linop

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#D55E00", "#0072B2", "#CC79A7", "#F0E442"]
RECOVERED = ("anaC", "linear", "linear_hi", "vote", "pt2")   # heads read off q(t)


# ---------------------------------------------------------------------------
# GF(2) helpers + the analytic head form
# ---------------------------------------------------------------------------
def gf2_reduce(vecs):
    """Full-RREF pivot dict {pivot bit -> reduced row} of integer-bitmask vectors."""
    piv = {}
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


def gf2_nullspace(vecs, n):
    """Basis (as bitmasks over n columns) of {h : popcount(h & v) even for all v}."""
    A = np.array([[(v >> e) & 1 for e in range(n)] for v in vecs], dtype=np.uint8)
    if A.size == 0:
        return [1 << e for e in range(n)]
    piv, r = [], 0
    for c in range(n):
        sel = next((i for i in range(r, A.shape[0]) if A[i, c]), None)
        if sel is None:
            continue
        A[[r, sel]] = A[[sel, r]]
        for i in range(A.shape[0]):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        piv.append(c)
        r += 1
    out = []
    for f in (c for c in range(n) if c not in piv):
        v = np.zeros(n, np.uint8)
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = A[i, f]
        out.append(int(sum(int(v[e]) << e for e in range(n))))
    return out


def head_form(zxm):
    """(l, Q) of the frozen analytic head: q(t) = l.t + t^T triu(Q,1) t over GF(2).

    Same C-form pullback as prefit_phase_head.py --analytic_C / ed_electric_line.py
    (duplicated on purpose so this diagnostic is an independent path), and generic
    in the plaquette count -- it never sees L, so non-cubic boxes work unchanged.
    """
    NP = len(zxm)
    C = np.array([[0 if p == q else bin(zxm[p][0] & zxm[q][1]).count("1") & 1
                   for q in range(NP)] for p in range(NP)], dtype=np.int64)
    assert np.array_equal(C, C.T), "C must be symmetric (commutation)"
    cols = [sum(((bin(zxm[q][0] & zxm[p][1]).count("1") & 1) << q) for q in range(NP))
            for p in range(NP)]
    piv = {}                                   # pivot bit -> [reduced col, preimage]
    for p in range(NP):
        v, pre = cols[p], 1 << p
        for c in list(piv):
            if (v >> c) & 1:
                v ^= piv[c][0]; pre ^= piv[c][1]
        if v:
            c = (v & -v).bit_length() - 1
            for k in list(piv):
                if (piv[k][0] >> c) & 1:
                    piv[k][0] ^= v; piv[k][1] ^= pre
            piv[c] = [v, pre]
    pivots = sorted(piv)
    P = np.array([[(piv[c][1] >> p) & 1 for p in range(NP)] for c in pivots],
                 dtype=np.int64).reshape(len(pivots), NP)
    l = np.zeros(NP, dtype=np.int64)
    Q = np.zeros((NP, NP), dtype=np.int64)
    lp = np.einsum("ip,pq,iq->i", P, np.triu(C, 1), P) & 1
    B = (P @ C @ P.T) & 1
    for i in range(len(pivots)):
        l[pivots[i]] = lp[i]
        for j in range(i + 1, len(pivots)):
            Q[pivots[i], pivots[j]] = B[i, j]
    return l, Q


def parity_of_masked(basis, mask):
    """Vectorized parity of popcount(basis & mask)."""
    v = (basis & np.int64(mask)).astype(np.uint64)
    for s in (32, 16, 8, 4, 2, 1):
        v ^= v >> np.uint64(s)
    return (v & np.uint64(1)).astype(np.uint8)


# ---------------------------------------------------------------------------
# support / syndrome / recovery structure
# ---------------------------------------------------------------------------
def support_structure(geo, stabs, zxm):
    """Coset detectors, line classes and minimal recoveries of the h=0 support.

    W = span(stars, x-pairs) is the h=0 orbit of |0...0>; W^perp is spanned by the
    Z-strings of `flux_constraint_masks` (asserted), so the coset label of a config
    -- packed parities against an independent detector set -- is both the syndrome
    and an exact support-membership test (label 0 <=> on support).
    """
    N = geo.N
    W = list(gf2_reduce([_mask(v) for v in geo.vertex_all] +
                        [_mask(x) for _, x, _ in stabs]).values())
    perp = gf2_nullspace(W, N)
    # the physical detectors: u_c = parity(b & XOR_{p in c} dp)
    u_masks = [int(np.bitwise_xor.reduce([zxm[p][0] for p in c])) for c in
               flux_constraint_masks(stabs)]
    det = sorted(gf2_reduce(u_masks).values())
    assert len(det) == len(perp), (
        f"flux constraints span {len(det)} of {len(perp)} support-detector bits")
    assert set(gf2_reduce(det)) == set(gf2_reduce(perp)), "detector spans differ"

    lab = [sum(((bin(d & (1 << e)).count("1") & 1) << k) for k, d in enumerate(det))
           for e in range(N)]                              # coset label of one flip
    classes = {}
    for e in range(N):
        classes.setdefault(lab[e], []).append(e)
    lit = sorted(k for k in classes if k)                  # nonzero line classes
    assert len(gf2_reduce(lit)) == len(lit), "line-class labels are not independent"

    # minimal recoveries per coset label: one edge per lit class
    rec, rep_lo, rep_hi, nlit = {}, {}, {}, {}
    for r in range(1, 1 << len(lit)):
        on = [lit[i] for i in range(len(lit)) if (r >> i) & 1]
        u = int(np.bitwise_xor.reduce(on))
        rec[u] = [tuple(c) for c in itertools.product(*[classes[k] for k in on])]
        rep_lo[u] = tuple(min(classes[k]) for k in on)
        rep_hi[u] = tuple(max(classes[k]) for k in on)
        nlit[u] = len(on)
    rec[0], rep_lo[0], rep_hi[0], nlit[0] = [()], (), (), 0
    return {"det": det, "classes": classes, "lit": lit, "rec": rec,
            "rep_lo": rep_lo, "rep_hi": rep_hi, "nlit": nlit, "dimW": len(W)}


def edge_flux(zxm, N):
    """flux[e] = plaquette-violation pattern of a single sigma^x flip on edge e."""
    return np.array([[(zm >> e) & 1 for zm, _ in zxm] for e in range(N)], np.uint8)


def path_weight(eps, fx, J):
    """D(eps) = sum over orderings of prod_j 1/DeltaE(A_j), A_j = first j flips.

    DeltaE(A) = 2J * #{p : |dp cap A| odd} (sigma^x commutes with every star, so
    only plaquettes are excited).  A path through the ground sector (DeltaE = 0)
    is projected out by the resolvent and contributes 0.
    """
    tot = 0.0
    for order in itertools.permutations(eps):
        acc, term = np.zeros(fx.shape[1], np.uint8), 1.0
        for e in order:
            acc = acc ^ fx[e]
            dE = 2.0 * J * int(acc.sum())
            if dE == 0.0:
                term = 0.0
                break
            term /= dE
        tot += term
    return tot


def next_order_recoveries(u, nlit_u, lab, N):
    """All (nlit+1)-flip subsets with coset label u -- the tie-breaking order.

    For a single lit class this is exactly the two-flip paths sigma <- e1 <- e2
    <- support (one lit + one unlit edge).
    """
    return [c for c in itertools.combinations(range(N), nlit_u + 1)
            if int(np.bitwise_xor.reduce([lab[e] for e in c])) == u]


# ---------------------------------------------------------------------------
# ED
# ---------------------------------------------------------------------------
def ground_state(geo, stabs, hx, hz, J, dense_max_N, tol=1e-11):
    """(psi, E0, E1, degeneracy) with psi gauge-fixed to psi(all-up) > 0."""
    from scipy.sparse.linalg import eigsh
    dim = 1 << geo.N
    H, basis = hamiltonian_linop(geo, hx=hx, hz=hz, J=J, xz_stabs=stabs)
    if geo.N <= dense_max_N:
        Hd = np.zeros((dim, dim))
        e = np.zeros(dim)
        for i in range(dim):
            e[i] = 1.0
            Hd[:, i] = H.matvec(e)
            e[i] = 0.0
        assert np.max(np.abs(Hd - Hd.T)) == 0.0, "H is not symmetric"
        ev, evec = np.linalg.eigh(Hd)
        psi, E0, E1 = evec[:, 0], float(ev[0]), float(ev[1])
        deg = int(np.sum(ev < E0 + 1e-9))
    else:
        ev, evec = eigsh(H, k=2, which="SA", tol=tol,
                         v0=np.ones(dim) / np.sqrt(dim))
        order = np.argsort(ev)
        psi, E0, E1 = evec[:, order[0]], float(ev[order[0]]), float(ev[order[1]])
        deg = int(1 + (E1 - E0 < 1e-9))
    psi = np.ascontiguousarray(psi, dtype=np.float64)
    assert abs(psi[0]) > 1e-12, f"|psi(all-up)| = {abs(psi[0]):.2e} -- gauge undefined"
    psi *= np.sign(psi[0])
    E_check = float(psi @ H.matvec(psi))
    assert abs(E_check - E0) < 1e-10, f"<psi|H|psi> = {E_check} != E0 = {E0}"
    return psi, E0, E1, deg, H


# ---------------------------------------------------------------------------
# heads + metrics at one field point
# ---------------------------------------------------------------------------
def head_signs(sq, basis, coset, struct, fx, lab, J, N):
    """Deterministic +-1 head values for every config, plus tie bookkeeping.

    sq = (-1)^{q(t)} of the analytic form on every config (the anaC head itself);
    recovered heads read sq at sigma XOR eps.
    """
    dim = sq.shape[0]
    out = {"anaC": sq.astype(np.int8)}
    lo = np.empty(dim, np.int8)
    hi = np.empty(dim, np.int8)
    vote_sum = np.zeros(dim, np.int32)
    pt1 = np.zeros(dim, np.float64)
    nlit_arr = np.zeros(dim, np.int8)

    for u, recs in struct["rec"].items():
        sel = coset == u
        if not sel.any():
            continue
        idx = basis[sel]
        nlit_arr[sel] = struct["nlit"][u]
        lo[sel] = sq[idx ^ _emask(struct["rep_lo"][u])]
        hi[sel] = sq[idx ^ _emask(struct["rep_hi"][u])]
        for eps in recs:
            s = sq[idx ^ _emask(eps)].astype(np.int32)
            vote_sum[sel] += s
            pt1[sel] += s * path_weight(eps, fx, J)

    out["linear"], out["linear_hi"] = lo, hi
    tie = vote_sum == 0
    out["vote"] = np.where(tie, lo, np.sign(vote_sum)).astype(np.int8)

    # pt: first order where it is non-vanishing, next order on the exact ties
    pt = np.sign(pt1)
    zero = pt == 0
    pt2_used = np.zeros(dim, bool)
    for u, nl in struct["nlit"].items():
        sel = zero & (coset == u)
        if not sel.any() or u == 0:
            continue
        idx = basis[sel]
        s2 = np.zeros(idx.shape[0], np.float64)
        for eps in next_order_recoveries(u, nl, lab, N):
            s2 += sq[idx ^ _emask(eps)] * path_weight(eps, fx, J)
        pt[sel] = np.sign(s2)
        pt2_used[sel] = True
    out["pt2"] = np.where(pt == 0, lo, pt).astype(np.int8)
    return out, {"vote_sum": vote_sum, "tie": tie, "nlit": nlit_arr,
                 "pt2_used": pt2_used, "pt_zero": zero}


def _emask(eps):
    m = 0
    for e in eps:
        m |= 1 << int(e)
    return m


def point_metrics(psi, heads, aux, coset, struct):
    """|psi|^2-weighted sign fidelities + the tie/vote diagnostics at one point."""
    w = psi ** 2
    s_ed = np.sign(psi).astype(np.int8)
    off = coset != 0
    nlit = aux["nlit"]

    m = {"weight_off_support": float(w[off].sum()),
         "weight_on_support": float(w[~off].sum())}
    F = {"exact": float(w[s_ed != 0].sum()),
         "plus": float(w[s_ed > 0].sum())}
    for h in RECOVERED:
        F[h] = float(w[heads[h] == s_ed].sum())
    m["F_s"] = F
    m["one_minus_F_s"] = {h: max(0.0, 1.0 - v) for h, v in F.items()}
    m["plus_best_gauge"] = max(F["plus"], 1.0 - F["plus"])

    # per-head breakdown by number of lit line classes
    bins = {"0": nlit == 0, "1": nlit == 1, "2": nlit == 2, "ge3": nlit >= 3}
    m["by_lit_classes"] = {
        k: {"weight": float(w[b].sum()),
            "n_configs": int(b.sum()),
            "wrong_weight": {h: float(w[b & (heads[h] != s_ed)].sum())
                             for h in RECOVERED}}
        for k, b in bins.items() if b.any()}

    # tie channel
    tie, vs = aux["tie"], aux["vote_sum"]
    dis = heads["linear"] != heads["linear_hi"]
    m["tie_weight_vote"] = float(w[tie].sum())
    m["tie_disagree_weight"] = float(w[dis].sum())
    m["wrong_weight_linear"] = float(w[heads["linear"] != s_ed].sum())
    m["tie_disagree_half_ratio"] = (m["wrong_weight_linear"] /
                                    (0.5 * m["tie_disagree_weight"])
                                    if m["tie_disagree_weight"] > 0 else None)
    m["pt2_used_weight"] = float(w[aux["pt2_used"]].sum())
    m["pt_undecided_weight"] = float(w[aux["pt_zero"] & (heads["pt2"] == 0)].sum())
    hist = {}
    for v in np.unique(vs[off]) if off.any() else []:
        hist[int(v)] = float(w[off & (vs == v)].sum())
    m["vote_multiplicity_weight"] = hist
    # vote with ties resolved by pt2 instead of by linear
    v_pt2 = np.where(tie, heads["pt2"], heads["vote"]).astype(np.int8)
    m["F_s"]["vote_tie_pt2"] = float(w[v_pt2 == s_ed].sum())
    m["one_minus_F_s"]["vote_tie_pt2"] = max(0.0, 1.0 - m["F_s"]["vote_tie_pt2"])
    return m


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(rows, tag, path, hx_grid, hz_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floor = 1e-17
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    heads = [h for h in rows[0]["one_minus_F_s"] if h != "exact"]

    ax = axes[0]
    for k, h in enumerate(heads):
        for hz, ls in zip(hz_grid, ["-", "--"]):
            xs = [r["hx"] for r in rows if r["hz"] == hz]
            ys = [max(r["one_minus_F_s"][h], floor) for r in rows if r["hz"] == hz]
            ax.plot(xs, ys, ls, color=OKABE_ITO[k % len(OKABE_ITO)], marker="o",
                    ms=3.5, lw=1.4, label=h if hz == hz_grid[0] else None)
    ax.set_yscale("log")
    ax.set_xlabel(r"$h_x$")
    ax.set_ylabel(r"$1 - F_s$")
    ax.set_title(f"sign infidelity, {tag}\n(solid $h_z$={hz_grid[0]}, "
                 f"dashed $h_z$={hz_grid[-1]})")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    order = sorted(heads, key=lambda h: -np.mean([r["one_minus_F_s"][h] for r in rows]))
    cmap = plt.get_cmap("plasma")
    for i, r in enumerate(rows):
        ax.plot(range(len(order)),
                [max(r["one_minus_F_s"][h], floor) for h in order],
                color=cmap(i / max(1, len(rows) - 1)), marker="o", ms=3.5, lw=1.2,
                label=f"({r['hx']}, {r['hz']})" if i % 4 == 0 else None)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(r"$1 - F_s$")
    ax.set_title("decoder ladder (worst -> best)")
    ax.legend(frameon=False, fontsize=7, title=r"$(h_x, h_z)$",
              title_fontsize=7, ncol=2)

    ax = axes[2]
    series = [("weight_off_support", "off-support weight", OKABE_ITO[1]),
              ("tie_weight_vote", "vote-tie weight", OKABE_ITO[2]),
              ("tie_disagree_weight", "lo/hi tie-break disagree", OKABE_ITO[3]),
              ("wrong_weight_linear", r"linear wrong weight", OKABE_ITO[4])]
    for key, lab, col in series:
        for hz, ls in zip(hz_grid, ["-", "--"]):
            xs = [r["hx"] for r in rows if r["hz"] == hz]
            ys = [max(r[key], floor) for r in rows if r["hz"] == hz]
            ax.plot(xs, ys, ls, color=col, marker="o", ms=3.5, lw=1.4,
                    label=lab if hz == hz_grid[0] else None)
    for hz, ls in zip(hz_grid, ["-", "--"]):
        xs = [r["hx"] for r in rows if r["hz"] == hz]
        ys = [max(0.5 * r["tie_disagree_weight"], floor) for r in rows if r["hz"] == hz]
        ax.plot(xs, ys, ls, color="0.55", lw=1.0,
                label="½ · disagree (prediction)" if hz == hz_grid[0] else None)
    ax.set_yscale("log")
    ax.set_xlabel(r"$h_x$")
    ax.set_ylabel(r"$|\psi|^2$ weight")
    ax.set_title("tie channel")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[fig] {path}")


# ---------------------------------------------------------------------------
def loglog_slope(xs, ys):
    """Least-squares log-log slope over strictly positive (x, y) pairs."""
    p = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 1e-15]
    if len(p) < 2:
        return None
    a = np.polyfit(np.log([x for x, _ in p]), np.log([y for _, y in p]), 1)
    return float(a[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--Lx", type=int, default=2)
    ap.add_argument("--Ly", type=int, default=2)
    ap.add_argument("--Lz", type=int, default=2)
    ap.add_argument("--bc", choices=["OBC", "PBC"], default="OBC")
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--hx_grid", type=float, nargs="+",
                    default=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--hz_grid", type=float, nargs="+", default=[0, 0.2])
    ap.add_argument("--fit_hx_max", type=float, default=0.2,
                    help="log-log slope of 1-F_s is fitted over 0 < hx <= this")
    ap.add_argument("--dense_max_N", type=int, default=14,
                    help="dense eigh below this N; eigsh(k=2) above")
    ap.add_argument("--export_tables", action="store_true",
                    help="also dump each head as a 2^N int8 lookup table for "
                         "tc3d.sign_frame.table_sign (--sign_frame table)")
    ap.add_argument("--out_dir", default="results/fermionic_gate0")
    ap.add_argument("--fig_dir", default="analysis/figs")
    args = ap.parse_args()

    geo = ThreeD_ToricCodeGeometry(args.Lx, args.Ly, args.Lz, bc=args.bc)
    stabs = fermionic_plaquettes(geo)
    zxm = [(_mask(z), _mask(x)) for z, x, _ in stabs]
    N, NP, dim = geo.N, len(stabs), 1 << geo.N
    tag = f"{args.Lx}x{args.Ly}x{args.Lz}_{args.bc}"
    struct = support_structure(geo, stabs, zxm)
    fx = edge_flux(zxm, N)
    lab = [sum(((bin(d & (1 << e)).count("1") & 1) << k)
               for k, d in enumerate(struct["det"])) for e in range(N)]
    l, Q = head_form(zxm)

    print(f"[geom] {tag}: N={N} (dim 2^{N}), {len(geo.vertex_all)} stars, "
          f"{NP} decorated plaquettes, E0(h=0) must be "
          f"{-(len(geo.vertex_all) + NP)}")
    print(f"[support] dim W = {struct['dimW']} (|support| = 2^{struct['dimW']}), "
          f"{len(struct['det'])} detector bit(s), line classes "
          f"{ {k: len(v) for k, v in struct['classes'].items()} }")
    print(f"[recover] minimal recoveries per coset: "
          f"{ {u: len(r) for u, r in struct['rec'].items()} }; "
          f"1/DeltaE per recovery edge: "
          f"{ {u: sorted({round(path_weight(e, fx, args.J), 6) for e in r}) for u, r in struct['rec'].items() if u} }")
    print(f"[head] analytic form: {int(l.sum())} linear + {int(np.triu(Q,1).sum())} "
          f"pair couplings")

    basis = np.arange(dim, dtype=np.int64)
    tbits = np.stack([parity_of_masked(basis, zm) for zm, _ in zxm])
    qb = np.zeros(dim, np.uint8)
    for i in np.nonzero(l)[0]:
        qb ^= tbits[i]
    for i, j in zip(*np.nonzero(np.triu(Q, 1))):
        qb ^= tbits[i] & tbits[j]
    sq = (1 - 2 * qb.astype(np.int8)).astype(np.int8)
    coset = np.zeros(dim, np.int32)
    for k, d in enumerate(struct["det"]):
        coset |= parity_of_masked(basis, d).astype(np.int32) << k
    del tbits

    heads, aux = head_signs(sq, basis, coset, struct, fx, lab, args.J, N)

    if args.export_tables:
        # The heads depend only on the geometry (q(t), the recovery sets and the
        # POSITIVE denominators 1/DeltaE) -- never on h_x or h_z: each config is
        # decided at its own leading order, where h_x^k factors out of the sum.
        # So one table per head serves the whole field grid.
        os.makedirs(args.out_dir, exist_ok=True)
        for h in ("pt2", "vote", "linear", "anaC"):
            t = heads[h].astype(np.int8)
            assert np.all(np.abs(t) == 1), f"head {h} has non-+-1 entries"
            np.save(os.path.join(args.out_dir, f"sign_table_{h}_{tag}.npy"), t)
        print(f"[tables] wrote sign_table_{{pt2,vote,linear,anaC}}_{tag}.npy "
              f"({dim} int8 each) -> {args.out_dir}")

    rows, gates = [], []
    for hz in args.hz_grid:
        for hx in args.hx_grid:
            psi, E0, E1, deg, H = ground_state(geo, stabs, hx, hz, args.J,
                                               args.dense_max_N)
            m = point_metrics(psi, heads, aux, coset, struct)
            m.update({"hx": float(hx), "hz": float(hz), "E0": E0, "E1": E1,
                      "gap": E1 - E0, "gs_degeneracy": deg})
            rows.append(m)
            f = m["one_minus_F_s"]
            print(f"  (hx={hx:<5} hz={hz:<4}) E0={E0:14.10f} gap={E1-E0:9.6f} "
                  f"off={m['weight_off_support']:.3e} | 1-F: "
                  + "  ".join(f"{h}={f[h]:.3e}" for h in
                              ("anaC", "linear", "vote", "pt2")), flush=True)

    # ---- validation gates ---------------------------------------------------
    def gate(name, ok, detail=""):
        gates.append({"gate": name, "pass": bool(ok), "detail": detail})
        print(f"[gate] {'ok  ' if ok else 'FAIL'} {name}  {detail}")

    r00 = next((r for r in rows if r["hx"] == 0 and r["hz"] == 0), None)
    if r00:
        gate("h=0 E0 == -(#stars+#plaq)",
             abs(r00["E0"] + len(geo.vertex_all) + NP) < 1e-9, f"E0={r00['E0']:.10f}")
        gate("h=0 unique GS", r00["gs_degeneracy"] == 1, f"deg={r00['gs_degeneracy']}")
        for h in ("anaC", "linear", "vote", "pt2"):
            gate(f"h=0 F_s({h}) == 1", abs(r00["F_s"][h] - 1.0) < 1e-12,
                 f"F_s={r00['F_s'][h]:.15f}")
    # L=2 OBC only: the banked ceilings of the earlier decoder diagnostic
    anchors = {(0.2, 0.0): 0.992455, (0.2, 0.2): 0.992443} if tag == "2x2x2_OBC" else {}
    for (hx, hz), ref in anchors.items():
        r = next((r for r in rows if r["hx"] == hx and r["hz"] == hz), None)
        if r is not None:
            gate(f"linear ceiling at ({hx},{hz}) ~ {ref}",
                 abs(r["F_s"]["linear"] - ref) < 5e-6,
                 f"F_s(linear)={r['F_s']['linear']:.6f}  ref={ref}")
    ref_path = "results/fermionic_obc_L2/ed_L2_OBC_rect.json"
    if tag == "2x2x2_OBC" and os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = {(p["hx"], p["hz"]): p["E0"] for p in json.load(f)["points"]}
        for k, E in ref.items():
            r = next((r for r in rows if (r["hx"], r["hz"]) == k), None)
            if r is not None:
                gate(f"E0{k} matches ed_L2_OBC_rect.json", abs(r["E0"] - E) < 1e-9,
                     f"{r['E0']:.12f} vs {E:.12f}")

    # ---- small-hx exponents -------------------------------------------------
    slopes = {}
    for hz in args.hz_grid:
        sub = [r for r in rows if r["hz"] == hz and 0 < r["hx"] <= args.fit_hx_max]
        xs = [r["hx"] for r in sub]
        slopes[str(hz)] = {
            **{h: loglog_slope(xs, [r["one_minus_F_s"][h] for r in sub])
               for h in sub[0]["one_minus_F_s"]},
            "weight_off_support": loglog_slope(xs, [r["weight_off_support"] for r in sub]),
            "tie_weight_vote": loglog_slope(xs, [r["tie_weight_vote"] for r in sub]),
            "tie_disagree_weight": loglog_slope(xs, [r["tie_disagree_weight"] for r in sub]),
        }
    print(f"[slope] d log(1-F_s)/d log hx over hx <= {args.fit_hx_max}:")
    for hz, s in slopes.items():
        print(f"   hz={hz}: " + "  ".join(f"{k}={v:.2f}" for k, v in s.items()
                                          if v is not None))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    out = {
        "geometry": {"Lx": args.Lx, "Ly": args.Ly, "Lz": args.Lz, "bc": args.bc,
                     "N": N, "n_stars": len(geo.vertex_all), "n_plaquettes": NP,
                     "J": args.J},
        "support": {"dim_W": struct["dimW"], "n_detector_bits": len(struct["det"]),
                    "detector_masks": [int(d) for d in struct["det"]],
                    "line_classes": {str(k): v for k, v in struct["classes"].items()},
                    "minimal_recoveries_per_coset":
                        {str(u): len(r) for u, r in struct["rec"].items()},
                    "edge_plaquette_multiplicity": fx.sum(1, dtype=np.int64).tolist()},
        "head_form": {"n_linear": int(l.sum()), "n_pairs": int(np.triu(Q, 1).sum())},
        "grid": {"hx": list(map(float, args.hx_grid)),
                 "hz": list(map(float, args.hz_grid))},
        "points": rows, "gates": gates, "small_hx_slopes": slopes,
    }
    jpath = os.path.join(args.out_dir, f"{tag}_gate0.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[out] {jpath}")
    make_figure(rows, tag, os.path.join(args.fig_dir, f"fermionic_gate0_{tag}.png"),
                args.hx_grid, args.hz_grid)
    print(f"\nALL GATES {'PASS' if all(g['pass'] for g in gates) else 'FAIL'}")


if __name__ == "__main__":
    main()
