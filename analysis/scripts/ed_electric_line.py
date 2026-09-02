"""Fermionic ED benchmark: electric line (PBC, legacy) + OBC rectangle referee.

Two modes, selected by --bc (default PBC = the ORIGINAL behavior, unchanged):

  PBC (legacy, unchanged): the three-part check of the frozen-head exactness
  conjecture (notes/fermionic_architecture.tex Sec. 6.2 / 7, "electric-line
  stoquasticity") along h_z at h_x=0, L=2 PBC:
    (1) sector: the global GS keeps every Gauss parity u_c = +1 at every h_z
        (the one gap in the analytic argument — a sector crossing is not
        excluded by generalities);
    (2) signs: sign(psi_ED(s)) == (-1)^{q(t(s))} on the support, |psi|^2-weighted
        (end-to-end audit of the head phases, including convention slips);
    (3) E0(h_z): the first exact finite-field benchmark for the NQS sweeps.
  CLUSTER ONLY for the ED itself (2^24-dim Lanczos, ~3.5 GB; the 8 GB dev
  machine is off-limits per CLAUDE.md).

  OBC (new, notes/fermionic_obc_l2_benchmark_plan.md Phase 2): L=2 OBC has
  N=12 (2^12=4096) — dense ED (full spectrum via numpy.linalg.eigh) is
  trivial locally. Sweeps (hx, hy, hz) as a cartesian product over
  --hx x --hy x --hz; computes E0/E1/gap/degeneracy, per-qubit sx/sz + means,
  per-vertex <A_v>, per-plaquette <B~_p>, plus the legacy Gauss-parity/sign-audit
  diagnostics (labeled as diagnostic-only once hx != 0, since their derivation
  assumed hx=0). Writes, under --out_dir (default results/fermionic_obc_L2 when
  --bc OBC):
    (a) one exact_diag_fermionic_L{L}_{bc}_hx{hx}_hz{hz}.json per point (or
        _hx{hx}_hy{hy}_hz{hz}.json once hy != 0 — the hy=0 filenames/content
        are untouched, so existing committed points stay byte-identical), in
        the schema tc3d.validation.find_reference/load_reference consumes;
    (b) a combined ed_L2_OBC_rect.json (style of ed_L2_electric.json,
        extended with hx);
    (c) the ground-state vector per point as compressed npz under
        ed_vectors/ (small — 4096 complex each at L=2 OBC).
  Prints explicit ok/FAIL validation gate lines (see run_gates).

  h_y != 0 (dense path only; NotImplementedError if it lands on the legacy
  eigsh/N>dense_max_N branch): tc3d.exact_diag.hamiltonian_linop has no h_y
  term (NotImplementedError there, left untouched per instructions), so the
  -h_y*Sum_i sigma^y_i field is added here as a SEPARATE dense/matvec
  contribution (`hy_field_matvec`/`hy_field_dense`), in exact_diag's OWN basis
  convention (qubit i = bit i, bit=1 = spin down). This sidesteps any
  NetKet-vs-exact_diag state-numbering mismatch entirely: Sum_i sigma^y_i is
  symmetric under any relabeling of the N sites, so it is basis-order-safe by
  construction (independently cross-checked against
  tc3d.hamiltonian.create_hamiltonian_fermionic(dtype="complex").to_dense(),
  permuted by the site-reversal i -> N-1-i that reconciles NetKet's
  Spin(N=...) site numbering with exact_diag's, to 3e-17 at L=2 OBC).
  GS becomes complex; psi is banked as complex128 in the npz (as it already
  was). The legacy sign-audit (real sign(psi) vs the deterministic head) is
  UNDEFINED for complex psi and is skipped (sign_match_weighted=None) rather
  than silently computed on nonsense — the phase-optimized analogue lives in
  sign_fidelity_ftc.py.

`--selftest` runs the cheap bit-level pieces (pullback form vs sampled
generator signs) locally — no ED, unaffected by --bc/--hx.

Usage (unchanged, legacy):
  python analysis/scripts/ed_electric_line.py --selftest
  python analysis/scripts/ed_electric_line.py --hz 0 0.05 0.1 0.194 0.3 0.5 0.8 \
      --out results/fermionic_h0/ed_L2_electric.json

Usage (new, OBC benchmark rectangle):
  python analysis/scripts/ed_electric_line.py --bc OBC --hx 0.0 0.2 --hz 0.0 0.2
  python analysis/scripts/ed_electric_line.py --bc OBC --hx 0.2 --hy 0.2 --hz 0.0
"""
import argparse
import json
import os

import numpy as np
from scipy.sparse.linalg import LinearOperator

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import (fermionic_plaquettes, flux_constraint_masks,
                                       _mask)
from tc3d.exact_diag import (hamiltonian_linop, expect_x_string, expect_z_string,
                             expect_xz_string, qubits_to_mask)


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


def build_dense(H, dim, dtype=np.float64):
    """Materialize a scipy LinearOperator as a dense ndarray, column by column.

    hamiltonian_linop's H only supports 1-D matvec (no .toarray()/todense()
    ctor for a matrix-free operator) — Hd[:, i] = H @ e_i for each basis
    vector e_i. O(dim) matvecs of O(dim) cost each; trivial at dim=4096
    (L=2 OBC, N=12).
    """
    Hd = np.zeros((dim, dim), dtype=dtype)
    e = np.zeros(dim, dtype=dtype)
    for i in range(dim):
        e[i] = 1.0
        Hd[:, i] = H.matvec(e)
        e[i] = 0.0
    return Hd


def hy_field_matvec(psi, N, hy):
    """(-hy * Sum_i sigma^y_i) @ psi, matrix-free, exact_diag's OWN bit
    convention (qubit i = bit i; bit=1 <=> spin down): from
    sigma^y_i|s> = 1j*(1-2*bit_i(s))|s XOR (1<<i)>, the matvec of the FULL
    field (summed over i, with the Hamiltonian's -hy prefactor) is

        out[r] = 1j*hy * Sum_i sign_i(r) * psi[r XOR (1<<i)],  sign_i(r) = 1-2*bit_i(r)

    (derivation: (sigma^y_i psi)[r] = 1j*sign_i(r^(1<<i))*psi[r^(1<<i)] =
    -1j*sign_i(r)*psi[r^(1<<i)] since sign_i flips under bit_i -> 1-bit_i;
    the field's -hy prefactor cancels the leading minus). Cross-checked
    against tc3d.hamiltonian.create_hamiltonian_fermionic(dtype="complex")
    .to_dense(), permuted into this basis order, to 3e-17 at L=2 OBC
    (2026-09-02, ad hoc verification -- see module docstring).

    Sum_i sigma^y_i is invariant under ANY relabeling of the N sites, so
    this matvec is basis-order-safe regardless of how exact_diag's
    `basis = arange(2^N)` convention lines up with any other code's site
    numbering -- no permutation is needed here (unlike a geometry-specific
    operator such as a vertex star or decorated plaquette).
    """
    psi = np.asarray(psi)
    r = np.arange(psi.shape[0], dtype=np.int64)
    out = np.zeros(psi.shape[0], dtype=np.complex128)
    for i in range(N):
        sign = 1.0 - 2.0 * ((r >> i) & 1).astype(np.float64)
        out += sign * psi[r ^ (1 << i)]
    return 1j * hy * out


class _HyFieldOp:
    """Thin matvec-only handle so `build_dense` can materialize the h_y field
    term the same way it materializes `hamiltonian_linop`'s H (one dense
    matrix, column by column) -- the SAME matvec then also backs the
    scipy LinearOperator used for the independent eigsh cross-check, so the
    dense term and the operator used to audit it can never drift apart."""

    def __init__(self, N, hy):
        self.N, self.hy = N, hy

    def matvec(self, psi):
        return hy_field_matvec(psi, self.N, self.hy)


def expect_y_string(psi, basis, i):
    """<psi| sigma^y_i |psi>, exact_diag's bit convention (qubit i = bit i).

    = -1j * Sum_r sign_i(r) * conj(psi[r]) * psi[r XOR (1<<i)], sign_i(r) =
    1-2*bit_i(r) (same derivation as `hy_field_matvec` with hy -> -1, i.e.
    the bare operator rather than the -hy*Sum_i field). A new, local, single-
    qubit helper (exact_diag.py only has expect_x_string/expect_z_string/
    expect_xz_string; not modifying that file per instructions).
    """
    sign = 1.0 - 2.0 * ((basis >> i) & 1).astype(np.float64)
    val = -1j * np.sum(sign * np.conj(psi) * psi[basis ^ (1 << i)])
    return float(np.real(val))


def run_point(geo, stabs, zxm, masks, l, Q, hx, hy, hz, tol, dense_max_N=20):
    """ED at one (hx, hz) point.

    Dense eigh (full spectrum, exact gap + degeneracy) when 2^geo.N fits
    comfortably in memory (N <= dense_max_N); eigsh(k=1) otherwise — the
    UNCHANGED legacy path (same call signature as the original script), so
    the PBC L=2 (N=24) invocation is byte-for-byte identical in cost/behavior.

    Returns (result_dict, psi) with result_dict holding every scalar/array
    both the legacy print line and the new exact_diag_*.json schema need.
    """
    from scipy.sparse.linalg import eigsh
    N = geo.N
    dim = 1 << N
    complex_gs = hy != 0.0
    dtype = np.complex128 if complex_gs else np.float64
    H, basis = hamiltonian_linop(geo, hx=hx, hz=hz, xz_stabs=stabs, dtype=dtype)
    dense = N <= dense_max_N
    if complex_gs and not dense:
        raise NotImplementedError(
            "h_y != 0 is only wired through the dense path here (hy_field_matvec "
            f"is O(dim); dense_max_N={dense_max_N} < N={N}) -- the PBC L=2 (N=24) "
            "legacy branch is out of scope locally per CLAUDE.md anyway")

    r = {"hx": hx, "hy": hy, "hz": hz, "N": N, "dense": dense}

    if dense:
        Hd = build_dense(H, dim, dtype=dtype)
        if complex_gs:
            Hd = Hd + build_dense(_HyFieldOp(N, hy), dim, dtype=np.complex128)
        r["herm_max_abs_dev"] = float(np.max(np.abs(Hd - Hd.conj().T)))
        evals, evecs = np.linalg.eigh(Hd)          # ascending order, guaranteed (real evals even for complex Hermitian Hd)
        E0 = float(evals[0])
        deg_tol = max(tol, 1e-9)
        deg = int(np.sum(evals < E0 + deg_tol))
        E1 = float(evals[1])
        gap = float(evals[1] - evals[0])
        gap_to_excited = float(evals[deg] - evals[0]) if dim > deg else None
        psi = np.ascontiguousarray(evecs[:, 0], dtype=dtype)
        # Independent eigsh(k=1) cross-check on the SAME matrix-free operator
        # (scipy Lanczos vs numpy dense eigh) — cheap at this size, always run.
        # At hy != 0 the operator must include the hy_field_matvec contribution
        # too (hamiltonian_linop's own matvec never carries it), or this would
        # silently compare eigh(Hd_with_hy) against eigsh(H_without_hy).
        if complex_gs:
            def _matvec_full(v):
                return H.matvec(v) + hy_field_matvec(v, N, hy)
            H_check = LinearOperator((dim, dim), matvec=_matvec_full, dtype=np.complex128)
        else:
            H_check = H
        Ek, _ = eigsh(H_check, k=1, which="SA", tol=tol)
        r["eigsh_E0"] = float(Ek[0])
        r["eigsh_dense_E0_delta"] = abs(float(Ek[0]) - E0)
        r.update({"E0": E0, "E1": E1, "gap": gap, "gs_degeneracy": deg,
                  "gap_to_first_excited": gap_to_excited})
    else:
        E, V = eigsh(H, k=1, which="SA", tol=tol)   # unchanged from the original script
        E0, psi = float(E[0]), V[:, 0]
        r.update({"E0": E0, "E1": None, "gap": None, "gs_degeneracy": None,
                  "gap_to_first_excited": None})

    # ---- observables: per-qubit sx/sz(/sy), per-vertex <A_v>, per-plaquette <B~_p>.
    # Dense mode only: ~2N+NV+NP O(2^N) contractions are trivial at N<=20 but
    # would silently weigh down the legacy eigsh path on 2^24 cluster sweeps. ----
    if dense:
        sx = [expect_x_string(psi, basis, 1 << i) for i in range(N)]
        sz = [expect_z_string(psi, basis, 1 << i, N) for i in range(N)]
        A_v = [expect_x_string(psi, basis, qubits_to_mask(v)) for v in geo.vertex_all]
        B_p = [expect_xz_string(psi, basis, qubits_to_mask(z), qubits_to_mask(x), N)
               for z, x, _ in stabs]
        r.update({
            "sx_per_qubit": sx, "sz_per_qubit": sz,
            "sx_mean": float(np.mean(sx)), "sx_max_abs": float(np.max(np.abs(sx))),
            "sz_mean": float(np.mean(sz)), "sz_max_abs": float(np.max(np.abs(sz))),
            "A_v_per_vertex": A_v, "A_v_mean": float(np.mean(A_v)), "A_v_min": float(np.min(A_v)),
            "B_p_per_plaq": B_p, "B_p_mean": float(np.mean(B_p)), "B_p_min": float(np.min(B_p)),
        })
        if complex_gs:                      # new field: only appears once hy != 0
            sy = [expect_y_string(psi, basis, i) for i in range(N)]
            r.update({"sy_per_qubit": sy, "sy_mean": float(np.mean(sy)),
                      "sy_max_abs": float(np.max(np.abs(sy)))})

    # ---- Gauss-parity <u_c> + sign-audit vs the frozen head (legacy metrics).
    # The derivation (prefit_phase_head --analytic_C, electric-line stoquasticity)
    # assumes h_x = 0; at h_x != 0 these are still well-defined (H stays real
    # symmetric, so the GS is real and signs are unambiguous up to a global
    # sign) but are reported as DIAGNOSTICS ONLY — the head is not expected to
    # match once h_x != 0. At h_y != 0 the GS is genuinely complex and a real
    # +-1 "sign" is undefined; sign_match_weighted is None there (the
    # phase-optimized analogue is sign_fidelity_ftc.py's F_s^C). ----
    w = np.abs(psi) ** 2                    # == psi*psi for real psi (hy=0): unchanged there
    u_exp = []
    for c in masks:
        ub = np.zeros(dim, dtype=np.uint8)
        for p in c:
            ub ^= parity_of_masked(basis, zxm[p][0])
        u_exp.append(float(np.sum(w * (1.0 - 2.0 * ub))))
    qbits = head_parity_bits(basis, zxm, l, Q)
    head_sign = 1.0 - 2.0 * qbits
    supp = w > 1e-24
    norm = float(np.sum(w[supp]))
    if complex_gs:
        frac = None
    else:
        agree = float(np.sum(w[supp] * (np.sign(psi[supp]) == head_sign[supp])))
        frac = agree / norm if norm > 0 else float("nan")
        frac = max(frac, 1.0 - frac)                 # global sign is arbitrary
    r.update({
        "support_weight": norm,
        "min_u": min(u_exp) if u_exp else None,
        "mean_u": float(np.mean(u_exp)) if u_exp else None,
        "sign_match_weighted": frac,
        "n_support_states": int(np.sum(supp)),
        "sign_audit_diagnostic_only": bool(hx != 0.0),
    })
    return r, psi


def _ok(label, cond, detail=""):
    tag = "ok  " if cond else "FAIL"
    print(f"[gate] {tag} {label}{('  ' + detail) if detail else ''}")
    return bool(cond)


def run_gates(rows, tol):
    """Prints explicit ok/FAIL lines for every gate whose precondition point(s)
    are present in `rows` (dict keyed by (hx, hz)); SKIPs the rest. Returns
    whether every checked gate passed."""
    all_ok = True
    key0 = (0.0, 0.0, 0.0)
    if key0 in rows:
        r0 = rows[key0]
        all_ok &= _ok("(0,0) E0 == -14", abs(r0["E0"] - (-14.0)) < 1e-9, f"E0={r0['E0']:.12f}")
        all_ok &= _ok("(0,0) unique GS", r0.get("gs_degeneracy") == 1,
                      f"degeneracy={r0.get('gs_degeneracy')}")
        all_ok &= _ok("(0,0) gap == 4", r0.get("gap") is not None and abs(r0["gap"] - 4.0) < 1e-9,
                      f"gap={r0.get('gap')}")
        all_ok &= _ok("(0,0) Mx < 1e-10", abs(r0["sx_mean"]) < 1e-10, f"Mx={r0['sx_mean']:.3e}")
        all_ok &= _ok("(0,0) Mz < 1e-10", abs(r0["sz_mean"]) < 1e-10, f"Mz={r0['sz_mean']:.3e}")
        maxdevA = max(abs(a - 1.0) for a in r0["A_v_per_vertex"])
        all_ok &= _ok("(0,0) every <A_v> == 1", maxdevA < 1e-9, f"max|dev|={maxdevA:.2e}")
        maxdevB = max(abs(b - 1.0) for b in r0["B_p_per_plaq"])
        all_ok &= _ok("(0,0) every <B~_p> == 1", maxdevB < 1e-9, f"max|dev|={maxdevB:.2e}")
    else:
        print("[gate] SKIP  (0,0) anchor gates -- point not in this run")

    herm_devs = [(k, r["herm_max_abs_dev"]) for k, r in rows.items()
                 if r.get("dense") and "herm_max_abs_dev" in r]
    if herm_devs:
        worst = max(herm_devs, key=lambda kv: kv[1])
        all_ok &= _ok("Hermiticity max|H-H^T| == 0 (all dense points)", worst[1] == 0.0,
                      f"worst={worst[1]:.3e} at {worst[0]}")
    else:
        print("[gate] SKIP  hermiticity -- no dense points computed")

    key_mm = (0.2, 0.0, 0.2)
    if key_mm in rows and "eigsh_dense_E0_delta" in rows[key_mm]:
        d = rows[key_mm]["eigsh_dense_E0_delta"]
        all_ok &= _ok("(0.2,0,0.2) dense eigh vs eigsh(k=1) E0 agree to 1e-9", d < 1e-9,
                      f"|delta E0|={d:.3e}")
    else:
        print("[gate] SKIP  (0.2,0,0.2) eigh/eigsh cross-check -- point not in this run")

    if key0 in rows:
        E00 = rows[key0]["E0"]
        for (hx, hy, hz), r in sorted(rows.items()):
            if (hx, hy, hz) == key0:
                continue
            all_ok &= _ok(f"E0 strictly decreases at (hx={hx}, hy={hy}, hz={hz}) vs (0,0,0)",
                          r["E0"] < E00, f"E0={r['E0']:.10f} vs E0(0,0,0)={E00:.10f}")
    else:
        print("[gate] SKIP  E0 monotonic-decrease -- (0,0,0) anchor not in this run")

    # ---- h_y time-reversal pair: H(-hy) = H(hy)* (complex conjugation flips
    # sigma^y -> -sigma^y, everything else real), so E0 must be IDENTICAL for
    # +hy and -hy at the same (hx, hz) -- a cheap, independent correctness
    # check on the sign of the hy_field_matvec term. ----
    hy_pos = sorted({k[1] for k in rows if k[1] > 0.0})
    for hyv in hy_pos:
        for hx, hz in sorted({(k[0], k[2]) for k in rows}):
            kp, km = (hx, hyv, hz), (hx, -hyv, hz)
            if kp in rows and km in rows:
                Ep, Em = rows[kp]["E0"], rows[km]["E0"]
                all_ok &= _ok(f"time-reversal pair E0(hy=+{hyv})==E0(hy=-{hyv}) "
                              f"at (hx={hx}, hz={hz})", abs(Ep - Em) < 1e-9,
                              f"E0(+)={Ep:.12f}  E0(-)={Em:.12f}")

    print("[gate] field-response table:")
    for (hx, hy, hz), r in sorted(rows.items()):
        print(f"    (hx={hx}, hy={hy}, hz={hz}): Mx={r['sx_mean']:+.8f}  Mz={r['sz_mean']:+.8f}")
        if hx > 0:
            all_ok &= _ok(f"Mx > 0 at hx={hx} (hy={hy}, hz={hz})", r["sx_mean"] > 0, f"Mx={r['sx_mean']:+.8f}")
        if hz > 0:
            all_ok &= _ok(f"Mz > 0 at hz={hz} (hx={hx}, hy={hy})", r["sz_mean"] > 0, f"Mz={r['sz_mean']:+.8f}")

    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--bc", choices=["PBC", "OBC"], default="PBC",
                    help="PBC = legacy electric-line path (unchanged); OBC = the L=2 "
                         "dense-ED benchmark rectangle (Phase 2)")
    ap.add_argument("--hx", type=float, nargs="+", default=[0.0],
                    help="h_x values; cartesian product with --hy x --hz. Default [0.0] "
                         "keeps every existing invocation identical (hx=0 electric line).")
    ap.add_argument("--hy", type=float, nargs="+", default=[0.0],
                    help="h_y values (dense --bc OBC path only; NotImplementedError "
                         "otherwise). Default [0.0] keeps every existing invocation "
                         "identical (real GS, unchanged filenames/schema).")
    ap.add_argument("--hz", type=float, nargs="+", default=[0.0])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--out", default=None,
                    help="legacy combined JSON path (electric-line schema, unchanged "
                         "from the original script — only written if given)")
    ap.add_argument("--out_dir", default=None,
                    help="dir for the OBC benchmark outputs (per-point exact_diag_*.json, "
                         "ed_L2_OBC_rect.json, ed_vectors/*.npz). Defaults to "
                         "results/fermionic_obc_L2 when --bc OBC and --out_dir is not given; "
                         "left unset (no new-style outputs) for --bc PBC.")
    ap.add_argument("--dense_max_N", type=int, default=20,
                    help="use dense numpy.linalg.eigh (full spectrum) when N <= this; "
                         "eigsh(k=1) otherwise (unchanged legacy path)")
    ap.add_argument("--selftest", action="store_true",
                    help="local: pullback form vs sampled generator signs (no ED)")
    args = ap.parse_args()

    geo = ThreeD_ToricCodeGeometry(args.L, args.L, args.L, bc=args.bc)
    stabs = fermionic_plaquettes(geo)
    zxm = [(_mask(z), _mask(x)) for z, x, _ in stabs]
    masks = flux_constraint_masks(stabs)
    l, Q = head_form(zxm)
    print(f"L={args.L} bc={args.bc}: N={geo.N}, NP={len(zxm)}, {len(masks)} Gauss masks, "
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

    if args.bc == "OBC" and args.out_dir is None:
        args.out_dir = "results/fermionic_obc_L2"

    rows = {}
    legacy_points = []
    for hx in args.hx:
        for hy in args.hy:
            for hz in args.hz:
                r, psi = run_point(geo, stabs, zxm, masks, l, Q, hx, hy, hz, args.tol,
                                   dense_max_N=args.dense_max_N)
                rows[(hx, hy, hz)] = r

                # ---- legacy print line: unchanged text/values from the original script
                # (hy=0); sign_match_weighted is None once hy != 0 (complex GS -- see
                # run_point), printed as "n/a (complex)" instead of crashing on format. ----
                sm = (f"{r['sign_match_weighted']:.10f}" if r["sign_match_weighted"] is not None
                      else "n/a (complex)")
                print(f"hz={hz:<6} E0={r['E0']:.10f}  min<u_c>={r['min_u']:.10f}  "
                      f"sign match={sm}  "
                      f"(support {r['n_support_states']} states, weight {r['support_weight']:.6f})",
                      flush=True)
                if r["dense"]:
                    print(f"    hx={hx}  hy={hy}  gap={r['gap']:.10f}  E1={r['E1']:.10f}  "
                          f"gs_degeneracy={r['gs_degeneracy']}  Mx={r['sx_mean']:+.10f}  "
                          f"Mz={r['sz_mean']:+.10f}  <A_v>={r['A_v_mean']:.10f}  "
                          f"<B~_p>={r['B_p_mean']:.10f}  herm_dev={r['herm_max_abs_dev']:.3e}",
                          flush=True)

                legacy_points.append({
                    "hz": hz, "E0": r["E0"], "support_weight": r["support_weight"],
                    "min_u": r["min_u"], "mean_u": r["mean_u"],
                    "sign_match_weighted": r["sign_match_weighted"],
                    "n_support_states": r["n_support_states"],
                })
                if args.out:                       # incremental dump: a mid-sweep crash
                    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                    with open(args.out, "w") as f:  # keeps completed points (original behavior)
                        json.dump({"L": args.L, "model": "fermionic electric line (hx=0)",
                                   "tol": args.tol, "points": legacy_points}, f, indent=1)

                if args.out_dir:
                    vec_dir = os.path.join(args.out_dir, "ed_vectors")
                    os.makedirs(vec_dir, exist_ok=True)
                    # hy=0.0 keeps the ORIGINAL tag (no "_hy..." component) so existing
                    # committed hy=0 files/names are untouched (byte-identical regression);
                    # hy != 0.0 inserts "_hy{hy}_" between hx and hz.
                    hy_part = f"_hy{hy}" if hy != 0.0 else ""
                    tag = f"L{args.L}_{args.bc}_hx{hx}{hy_part}_hz{hz}"

                    # (a) per-point exact_diag_*.json — schema tc3d.validation.find_reference/
                    # load_reference consumes (_REF_KEYS: E0, gap, A_v_mean, B_p_mean, sx_mean,
                    # sz_mean, hx, hz, N; "model" field selects fermionic vs bosonic).
                    payload = {
                        "model": "fermionic", "Lx": geo.Lx, "Ly": geo.Ly, "Lz": geo.Lz,
                        "bc": args.bc, "N_vertices": len(geo.vertex_all),
                        "N_plaquettes": len(stabs), "hy": hy, "J": 1.0,
                        "dtype": "complex128" if hy != 0.0 else "float64",
                        **r,
                    }
                    with open(os.path.join(args.out_dir, f"exact_diag_fermionic_{tag}.json"), "w") as f:
                        json.dump(payload, f, indent=2)

                    # (c) ground-state vector (compressed npz; small at L=2 OBC, dim=4096)
                    np.savez_compressed(
                        os.path.join(vec_dir, f"gs_{tag}.npz"),
                        psi=psi.astype(np.complex128), E0=r["E0"], hx=hx, hy=hy, hz=hz,
                        L=args.L, bc=args.bc, N=geo.N,
                        basis_convention="arange(2^N); sigma_i = 1 - 2*bit_i(b); qubit index == bit position")

    if args.out_dir:
        combined = {
            "L": args.L, "bc": args.bc, "model": "fermionic OBC rectangle",
            "tol": args.tol,
            "points": [rows[k] for k in sorted(rows)],
        }
        combined_path = os.path.join(args.out_dir, "ed_L2_OBC_rect.json")
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved combined rectangle JSON -> {combined_path}")

        print("\n=== 4-point results table ===")
        print(f"{'(hx,hy,hz)':>18} {'E0':>16} {'gap':>12} {'Mx':>14} {'Mz':>14} "
              f"{'<A_v>':>12} {'<B~_p>':>12}")
        for (hx, hy, hz), r in sorted(rows.items()):
            gap_s = f"{r['gap']:.6f}" if r['gap'] is not None else "n/a"
            print(f"({hx:.2f},{hy:.2f},{hz:.2f})   {r['E0']:16.10f} {gap_s:>12} "
                  f"{r['sx_mean']:14.8e} {r['sz_mean']:14.8e} "
                  f"{r['A_v_mean']:12.8f} {r['B_p_mean']:12.8f}")

        print("\n=== validation gates ===")
        gates_ok = run_gates(rows, args.tol)
        print(f"\nALL GATES {'PASS' if gates_ok else 'FAIL'}")

    print("\ndone")
