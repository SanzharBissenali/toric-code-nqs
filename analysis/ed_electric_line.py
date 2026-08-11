"""Fermionic ED along the electric line (h_x=0, sweep h_z) at L=2 PBC.

The three-part check of the frozen-head exactness conjecture
(notes/fermionic_architecture.tex Sec. 6.2 / 7, "electric-line stoquasticity"):
  (1) sector: the global GS keeps every Gauss parity u_c = +1 at every h_z
      (the one gap in the analytic argument — a sector crossing is not
      excluded by generalities);
  (2) signs: sign(psi_ED(s)) == (-1)^{q(t(s))} on the support, |psi|^2-weighted
      (end-to-end audit of the head phases, including convention slips);
  (3) E0(h_z): the first exact finite-field benchmark for the NQS sweeps.

CLUSTER ONLY for the ED itself (2^24-dim Lanczos, ~3.5 GB; the 8 GB dev
machine is off-limits per CLAUDE.md). `--selftest` runs the cheap bit-level
pieces (pullback form vs sampled generator signs) locally.

Usage:
  python analysis/ed_electric_line.py --selftest
  python analysis/ed_electric_line.py --hz 0 0.05 0.1 0.194 0.3 0.5 0.8 \
      --out results/fermionic_h0/ed_L2_electric.json
"""
import argparse
import json
import os

import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import (fermionic_plaquettes, flux_constraint_masks,
                                       _mask)
from tc3d.exact_diag import hamiltonian_linop


def head_form(zxm):
    """(l, Q) of the frozen head via the C-form pullback (same construction as
    prefit_phase_head --analytic_C; duplicated here so the ED check is an
    INDEPENDENT path from the production code it audits)."""
    NP = len(zxm)
    Cnp = np.zeros((NP, NP), dtype=np.int64)
    for p in range(NP):
        for q in range(NP):
            if p != q:
                Cnp[p, q] = bin(zxm[p][0] & zxm[q][1]).count("1") & 1
    cols = []
    for p in range(NP):
        v = 0
        for q in range(NP):
            if bin(zxm[q][0] & zxm[p][1]).count("1") & 1:
                v |= 1 << q
        cols.append(v)
    piv = {}
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
    d = len(pivots)
    P = np.zeros((d, NP), dtype=np.int64)
    for i, c in enumerate(pivots):
        for p in range(NP):
            P[i, p] = (piv[c][1] >> p) & 1
    Cu = np.triu(Cnp, 1)
    l = np.zeros(NP, dtype=np.int64)
    Q = np.zeros((NP, NP), dtype=np.int64)
    lp_ = (np.einsum("ip,pq,iq->i", P, Cu, P) & 1)
    Bm = (P @ Cnp @ P.T) & 1
    for i in range(d):
        l[pivots[i]] = lp_[i]
        for j in range(i + 1, d):
            Q[pivots[i], pivots[j]] = Bm[i, j]
    return l, Q


def parity_of_masked(basis, mask):
    """Vectorized parity of popcount(basis & mask) — no popcount needed."""
    v = (basis & np.int64(mask)).astype(np.uint64)
    for s in (32, 16, 8, 4, 2, 1):
        v ^= v >> np.uint64(s)
    return (v & np.uint64(1)).astype(np.uint8)


def head_parity_bits(basis, zxm, l, Q):
    """q(t(s)) mod 2 for every basis state (uint8), from token parities."""
    NP = len(zxm)
    tbits = {}
    def t(p):
        if p not in tbits:
            tbits[p] = parity_of_masked(basis, zxm[p][0])
        return tbits[p]
    q = np.zeros(len(basis), dtype=np.uint8)
    for i in range(NP):
        if l[i]:
            q ^= t(i)
        for j in range(i + 1, NP):
            if Q[i, j]:
                q ^= t(i) & t(j)
    return q


def run_point(geo, stabs, zxm, masks, l, Q, hz, tol):
    from scipy.sparse.linalg import eigsh
    H, basis = hamiltonian_linop(geo, hz=hz, xz_stabs=stabs)
    E, V = eigsh(H, k=1, which="SA", tol=tol)
    E0, psi = float(E[0]), V[:, 0]

    # (1) Gauss parities of the GS: u_c = XOR of token bits over the mask
    u_exp = []
    w = psi * psi
    for c in masks:
        ub = np.zeros(len(basis), dtype=np.uint8)
        for p in c:
            ub ^= parity_of_masked(basis, zxm[p][0])
        u_exp.append(float(np.sum(w * (1.0 - 2.0 * ub))))

    # (2) sign audit vs the head, |psi|^2-weighted, global-phase-fixed
    qbits = head_parity_bits(basis, zxm, l, Q)
    head_sign = 1.0 - 2.0 * qbits
    supp = w > 1e-24
    agree = float(np.sum(w[supp] * (np.sign(psi[supp]) == head_sign[supp])))
    norm = float(np.sum(w[supp]))
    frac = agree / norm
    frac = max(frac, 1.0 - frac)                     # eigsh global sign is arbitrary
    return {"hz": hz, "E0": E0, "support_weight": norm,
            "min_u": min(u_exp), "mean_u": float(np.mean(u_exp)),
            "sign_match_weighted": frac,
            "n_support_states": int(np.sum(supp))}


ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=2)
ap.add_argument("--hz", type=float, nargs="+", default=[0.0])
ap.add_argument("--tol", type=float, default=1e-10)
ap.add_argument("--out", default=None)
ap.add_argument("--selftest", action="store_true",
                help="local: pullback form vs sampled generator signs (no ED)")
args = ap.parse_args()

geo = ThreeD_ToricCodeGeometry(args.L, args.L, args.L, bc="PBC")
stabs = fermionic_plaquettes(geo)
zxm = [(_mask(z), _mask(x)) for z, x, _ in stabs]
masks = flux_constraint_masks(stabs)
l, Q = head_form(zxm)
print(f"L={args.L}: NP={len(zxm)}, {len(masks)} Gauss masks, "
      f"head form: {int(l.sum())} linear + {int(Q.sum())} pair couplings")

if args.selftest:
    rng = np.random.default_rng(0)
    ok = 0
    n = 3000
    NP = len(zxm)
    for _ in range(n):
        s, sg = 0, 1
        for k in np.nonzero(rng.integers(0, 2, size=NP))[0]:
            zb, xb = zxm[k]
            if bin(zb & s).count("1") & 1:
                sg = -sg
            s ^= xb
        t = np.array([bin(s & zb).count("1") & 1 for zb, _ in zxm], dtype=np.int64)
        pred = int(l @ t + t @ np.triu(Q, 1) @ t) & 1
        ok += int(pred == (0 if sg > 0 else 1))
    print(f"SELFTEST: form matches generator signs on {ok}/{n} sampled classes")
    raise SystemExit(0 if ok == n else 1)

results = []
for hz in args.hz:
    r = run_point(geo, stabs, zxm, masks, l, Q, hz, args.tol)
    results.append(r)
    print(f"hz={hz:<6} E0={r['E0']:.10f}  min<u_c>={r['min_u']:.10f}  "
          f"sign match={r['sign_match_weighted']:.10f}  "
          f"(support {r['n_support_states']} states, weight {r['support_weight']:.6f})",
          flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"L": args.L, "model": "fermionic electric line (hx=0)",
                       "tol": args.tol, "points": results}, f, indent=1)
print("done")
