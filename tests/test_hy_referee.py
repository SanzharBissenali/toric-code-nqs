"""
h_y != 0 fermionic ED referee -- fast local regression + physics gates.

Covers the hy extension of analysis/scripts/ed_electric_line.py (--bc OBC
dense path) / sign_fidelity_ftc.py (F_s^C) / eval_snapshots.py (--exact):
  (a) hy=0 regression: rerunning the real ed_electric_line.py CLI at
      (hx=0.2, hz=0.0) must still reproduce the committed reference JSON
      (byte-identical contract of the hy extension).
  (b) Hermiticity of the -hy*Sum_i sigma^y_i matvec (hy_field_matvec, added
      independently in all three scripts since exact_diag.py's
      hamiltonian_linop raises NotImplementedError for hy != 0 and is not
      modified per instructions).
  (c) time-reversal pair: E0(hx,+hy,hz) == E0(hx,-hy,hz) exactly (H(-hy) =
      H(hy)* = H(hy)^T, same eigenvalues as H(hy)).
  (d) Hellmann-Feynman: dE/dhy (finite difference) ~= -Sum_i <sigma^y_i>.
  (e) phase_optimal_ceiling (the F_s^C ceiling used for complex psi) reduces
      exactly to the real anchored-gauge formula max(F_s, 1-F_s) at hy=0.

All ED here is L=2 OBC (N=12, dim=4096) via scipy eigsh (Lanczos, fast --
NOT the dense np.linalg.eigh path, which was independently timed at
57-166s per point on this machine for SOME field values, an Apple
Accelerate LAPACK performance issue reproduced ad hoc 2026-09-02, not a
correctness one; see ed_electric_line.py's module docstring and
sign_fidelity_ftc.py's ground_state_complex docstring). hy_field_matvec and
phase_optimal_ceiling are duplicated here on purpose (same convention as
`head_form` being duplicated between ed_electric_line.py and
sign_fidelity_ftc.py) so this test is an INDEPENDENT path from the
production scripts it audits, not an import of them (ed_electric_line.py's
CLI section runs unconditionally at import time -- do not import it as a
module; test (a) below shells out to it instead).

Run directly:
    cd tests && ../.venv/bin/python test_hy_referee.py
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import fermionic_plaquettes
from tc3d.exact_diag import hamiltonian_linop

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ED_SCRIPT = os.path.join(REPO_ROOT, "analysis", "scripts", "ed_electric_line.py")


def hy_field_matvec(psi, N, hy):
    """(-hy * Sum_i sigma^y_i) @ psi; exact_diag bit convention (qubit i = bit
    i, bit=1 <=> spin down). out[r] = 1j*hy * Sum_i sign_i(r)*psi[r^(1<<i)],
    sign_i(r) = 1-2*bit_i(r) -- see ed_electric_line.py's module docstring
    for the derivation and the NetKet cross-check (3e-17 agreement, L=2 OBC)."""
    r = np.arange(psi.shape[0], dtype=np.int64)
    out = np.zeros(psi.shape[0], dtype=np.complex128)
    for i in range(N):
        sign = 1.0 - 2.0 * ((r >> i) & 1).astype(np.float64)
        out += sign * psi[r ^ (1 << i)]
    return 1j * hy * out


def phase_optimal_ceiling(psi, s, n_bins=8192):
    """F_s^C = max_theta Sum_sigma max(Re(e^{i theta} s(sigma) psi*(sigma)), 0)^2;
    see sign_fidelity_ftc.py's docstring for the exact-binning derivation."""
    psi = np.asarray(psi, dtype=np.complex128)
    s = np.asarray(s, dtype=np.float64)
    c = s * np.conj(psi)
    wc = np.abs(psi) ** 2
    scale = n_bins / (2.0 * np.pi)
    b = np.floor((np.angle(c) + np.pi) * scale).astype(np.int64) % n_bins
    M0 = np.bincount(b, weights=wc, minlength=n_bins)
    c2 = c * c
    M2 = (np.bincount(b, weights=c2.real, minlength=n_bins)
          + 1j * np.bincount(b, weights=c2.imag, minlength=n_bins))
    A, B = 0.5 * M0, 0.5 * M2
    cA, cB = np.concatenate([A, A]).cumsum(), np.concatenate([B, B]).cumsum()
    half = n_bins // 2
    thetas = np.arange(n_bins) * (2.0 * np.pi / n_bins)
    start = np.floor((np.pi / 2.0 - thetas + np.pi) * scale).astype(np.int64) % n_bins
    sumA = cA[start + half - 1] - np.where(start > 0, cA[start - 1], 0.0)
    sumB = cB[start + half - 1] - np.where(start > 0, cB[start - 1], 0.0)
    vals = sumA + np.real(np.exp(2j * thetas) * sumB)
    width = 2.0 * np.pi / n_bins
    tstar = (-np.angle(sumB) / 2.0) % np.pi
    lo_edge = thetas - width
    inwin = np.zeros_like(vals, dtype=bool)
    for shift in (-np.pi, 0.0, np.pi):
        t = tstar + shift
        inwin |= (t > lo_edge) & (t <= thetas)
    vals = np.where(inwin, sumA + np.abs(sumB), vals)
    return float(vals.max())


def _geo_stabs(L=2, bc="OBC"):
    geo = ThreeD_ToricCodeGeometry(L, L, L, bc=bc)
    return geo, fermionic_plaquettes(geo)


def _ground_state(geo, stabs, hx, hy, hz, k=2, tol=1e-11):
    """Fast eigsh-based (hx,hy,hz) ground state (see module docstring for why
    not dense eigh)."""
    dim = 1 << geo.N
    H, basis = hamiltonian_linop(geo, hx=hx, hz=hz, xz_stabs=stabs, dtype=np.complex128)

    def matvec(v):
        out = H.matvec(v)
        return out + hy_field_matvec(v, geo.N, hy) if hy != 0.0 else out
    Hop = LinearOperator((dim, dim), matvec=matvec, dtype=np.complex128)
    ev, evec = eigsh(Hop, k=k, which="SA", tol=tol)
    order = np.argsort(ev)
    return ev[order], evec[:, order], Hop


def test_hy0_regression_vs_committed_json():
    """(a) rerunning the REAL ed_electric_line.py --bc OBC CLI at (hx=0.2,
    hz=0.0) (hy defaults to 0.0, unchanged) must reproduce the committed
    reference JSON's physics fields exactly (the byte-identical regression
    contract of the hy extension)."""
    ref_path = os.path.join(REPO_ROOT, "results", "fermionic_obc_L2",
                            "exact_diag_fermionic_L2_OBC_hx0.2_hz0.0.json")
    with open(ref_path) as f:
        ref = json.load(f)

    tmp_dir = os.path.join(REPO_ROOT, "tests", "_tmp_hy_referee_regress")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        env = dict(os.environ, PYTHONPATH=REPO_ROOT)
        subprocess.run(
            [sys.executable, ED_SCRIPT, "--bc", "OBC", "--hx", "0.2", "--hz", "0.0",
             "--out_dir", tmp_dir],
            check=True, cwd=REPO_ROOT, env=env, capture_output=True, timeout=120)
        out_path = os.path.join(tmp_dir, "exact_diag_fermionic_L2_OBC_hx0.2_hz0.0.json")
        with open(out_path) as f:
            got = json.load(f)
        for key in ("E0", "gap", "E1", "gs_degeneracy", "A_v_mean", "B_p_mean",
                   "sx_mean", "sz_mean", "hy", "dtype", "herm_max_abs_dev",
                   "sign_match_weighted"):
            gv, rv = got[key], ref[key]
            ok = (gv == rv) if not isinstance(rv, float) else abs(gv - rv) < 1e-9
            assert ok, f"{key}: got={gv!r} != ref={rv!r}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_hermiticity():
    """(b) <u|H(hy)|v> == conj(<v|H(hy)|u>) for hy = 0, +0.2, -0.3."""
    geo, stabs = _geo_stabs()
    dim = 1 << geo.N
    rng = np.random.default_rng(0)
    u = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    for hy in (0.0, 0.2, -0.3):
        _, _, Hop = _ground_state(geo, stabs, hx=0.1, hy=hy, hz=0.05, k=1)
        dev = abs(np.vdot(u, Hop.matvec(v)) - np.conj(np.vdot(v, Hop.matvec(u))))
        assert dev < 1e-9, f"hy={hy}: Hermiticity dev={dev:.3e}"


def test_hy_time_reversal_pair():
    """(c) H(-hy) = H(hy)^T (complex conjugation flips sigma^y only) => same
    eigenvalues, so E0(+hy) == E0(-hy) exactly."""
    geo, stabs = _geo_stabs()
    ev_p, _, _ = _ground_state(geo, stabs, hx=0.0, hy=0.2, hz=0.0, k=1)
    ev_m, _, _ = _ground_state(geo, stabs, hx=0.0, hy=-0.2, hz=0.0, k=1)
    assert abs(ev_p[0] - ev_m[0]) < 1e-9, \
        f"E0(+0.2)={ev_p[0]!r} != E0(-0.2)={ev_m[0]!r}"


def test_hellmann_feynman():
    """(d) dE0/dhy (central finite difference, eps=1e-4) ~= -Sum_i <sigma^y_i>
    at hy=0.2 (hx=hz=0)."""
    geo, stabs = _geo_stabs()
    N = geo.N
    eps = 1e-4
    ev_hi, _, _ = _ground_state(geo, stabs, hx=0.0, hy=0.2 + eps, hz=0.0, k=1)
    ev_lo, _, _ = _ground_state(geo, stabs, hx=0.0, hy=0.2 - eps, hz=0.0, k=1)
    dEdhy = (ev_hi[0] - ev_lo[0]) / (2 * eps)

    _, evec, _ = _ground_state(geo, stabs, hx=0.0, hy=0.2, hz=0.0, k=1)
    psi = evec[:, 0]
    psi = psi / np.linalg.norm(psi)
    r = np.arange(psi.shape[0], dtype=np.int64)
    sy_sum = 0.0
    for i in range(N):
        sign = 1.0 - 2.0 * ((r >> i) & 1).astype(np.float64)
        val = -1j * np.sum(sign * np.conj(psi) * psi[r ^ (1 << i)])
        sy_sum += float(np.real(val))
    assert abs(dEdhy - (-sy_sum)) < 1e-6, \
        f"HF mismatch: dE/dhy={dEdhy!r} vs -Sum<sy>={-sy_sum!r}"


def test_phase_optimal_ceiling_reduces_to_real_formula():
    """(e) at hy=0 (real psi), phase_optimal_ceiling(psi, s) == max(F_s, 1-F_s)
    for a +-1 head s, for BOTH a random head and the trivial all-+1 head."""
    geo, stabs = _geo_stabs()
    _, evec, _ = _ground_state(geo, stabs, hx=0.2, hy=0.0, hz=0.0, k=1)
    psi = evec[:, 0].real
    psi = psi / np.linalg.norm(psi)
    rng = np.random.default_rng(1)
    for s in (np.ones(psi.shape[0]),
             rng.choice([-1.0, 1.0], size=psi.shape[0])):
        F_s = float(np.sum((psi ** 2)[np.sign(psi) == s]))
        anchored = max(F_s, 1.0 - F_s)
        f_complex = phase_optimal_ceiling(psi.astype(np.complex128), s)
        assert abs(f_complex - anchored) < 1e-9, \
            f"phase_optimal_ceiling={f_complex!r} != anchored={anchored!r}"


if __name__ == "__main__":
    test_hy0_regression_vs_committed_json()
    print("  ok  hy=0 regression vs committed exact_diag_fermionic_L2_OBC_hx0.2_hz0.0.json")

    test_hermiticity()
    print("  ok  Hermiticity of H(hy) at hy = 0, +0.2, -0.3")

    test_hy_time_reversal_pair()
    print("  ok  time-reversal pair: E0(hy=+0.2) == E0(hy=-0.2)")

    test_hellmann_feynman()
    print("  ok  Hellmann-Feynman: dE0/dhy == -Sum_i <sigma^y_i>")

    test_phase_optimal_ceiling_reduces_to_real_formula()
    print("  ok  phase_optimal_ceiling reduces to max(F_s,1-F_s) at hy=0")

    print("ALL HY-REFEREE TESTS PASSED")
