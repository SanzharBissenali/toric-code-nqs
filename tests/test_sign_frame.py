"""A/B equivalence witness for the sign-framed Hamiltonian (formulation B).

Formulation A:  log psi = log A_theta + i*pi*s(sigma)   (gridinv --phase_head_frozen,
                theta from the analytic anaC form) evaluated on the bare H.
Formulation B:  a POSITIVE real trunk A_theta with the SAME weights, on the framed
                H~ = S H S (tc3d.sign_frame.SignFramedOperator).

They must agree number-for-number: S carries no parameters, so |psi|^2 = A^2 (same
chains), and every local energy / expectation coincides. Everything here is L=2 OBC
(N=12, 4096 configs) -- safe on the dev machine, well under 2 minutes.

Run directly:
    cd tests && PYTHONPATH=.. ../.venv/bin/python test_sign_frame.py
"""

import os
import tempfile

import numpy as np
import scipy.sparse.linalg as sla
import jax
import jax.numpy as jnp

from tc3d.builders import build_state, run_loop
from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import fermionic_plaquettes, _mask
from tc3d.exact_diag import hamiltonian_linop
from tc3d.io import load_weights, save_model
from tc3d.sign_frame import (SignFramedOperator, anaC_sign, anaC_theta,
                             sign_table, table_sign)
from tc3d.validation import build_eval_operators, nqs_observables
from tc3d.sign_frame import frame_eval_ops
from analysis.scripts.eval_snapshots import prepare_exact_context, exact_observables

# Matches analysis/scripts/prefit_phase_head.py's config, so the committed
# prefit_anaC_k2_L2_OBC.mpack loads into it parameter-for-parameter.
BASE = dict(L=2, bc="OBC", model="fermionic", arch="ToricCNN_gridinv", kernel_size=2,
            noninv_hidden=[4, 8], inv_hidden=[8, 8], n_noninv=2, noninv_channels=4,
            hx=0.2, hz=0.1, n_samples=256, n_chains=8, n_discard=4, seed=1)
PREFIT = os.environ.get(
    "TC3D_PREFIT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "fermionic_obc_L2", "prefit_anaC_k2_L2_OBC"))


def _configs(geo, n_random=192, seed=0):
    """Batch of +-1 configs: the all-up state, the h=0 stabilizer orbit (ON the
    ground-state support), and uniform random ones (mostly OFF it)."""
    rng = np.random.default_rng(seed)
    flips = [_mask(v) for v in geo.vertex_all]                        # A_v
    flips += [_mask(x) for _, x, _ in fermionic_plaquettes(geo)]      # B~_p x-pairs
    bits = [0]
    for _ in range(63):
        m = 0
        for k in np.nonzero(rng.integers(0, 2, size=len(flips)))[0]:
            m ^= flips[k]
        bits.append(m)
    on = np.array([[(m >> i) & 1 for i in range(geo.N)] for m in bits])
    off = rng.integers(0, 2, size=(n_random, geo.N))
    return 1.0 - 2.0 * np.concatenate([on, off]).astype(np.float64)


def _states():
    """(geo, HamA, vsA, HamB, vsB, sign_fn) at shared trunk weights."""
    geo, hi, HamA, vsA, _ = build_state({**BASE, "phase_head_frozen": True})
    tl, tq = anaC_theta(fermionic_plaquettes(geo))
    if os.path.exists(f"{PREFIT}.mpack"):
        vsA = load_weights(vsA, PREFIT)
        ck = vsA.model_state["constants"]
        assert np.array_equal(np.asarray(ck["phase_lin"]).real, tl) and \
            np.array_equal(np.asarray(ck["phase_quad"]).real, tq), \
            "analytic anaC theta != the committed prefit checkpoint's theta"
        src = "prefit checkpoint (theta bit-identical to the analytic form)"
    else:                       # untracked artifact: fall back to the analytic form
        vsA.model_state = {"constants": {
            "phase_lin": jnp.asarray(tl, jnp.complex128),
            "phase_quad": jnp.asarray(tq, jnp.complex128)}}
        vsA.parameters = jax.tree_util.tree_map(
            lambda a: jnp.real(a).astype(a.dtype), vsA.parameters)
        src = f"analytic anaC form ({PREFIT}.mpack absent)"
    geo, hi, HamB, vsB, _ = build_state({**BASE, "sign_frame": "anaC"})
    assert not any(np.iscomplexobj(np.asarray(p))
                   for p in jax.tree_util.tree_leaves(vsB.parameters)), \
        "sign_frame must give the fermionic model a REAL (positive) trunk"
    vsB.parameters = jax.tree_util.tree_map(
        lambda a: jnp.real(a).astype(jnp.float64), vsA.parameters)
    print(f"  [theta source: {src}]")
    return geo, HamA, vsA, HamB, vsB, anaC_sign(geo)


def _eloc(vs, op, x, logx):
    xp, mels = op.get_conn_padded(x)
    lp = np.asarray(vs.log_value(np.asarray(xp).reshape(-1, x.shape[-1])))
    return (np.asarray(mels) * np.exp(lp.reshape(xp.shape[:-1]) - logx[:, None])).sum(1)


def test_trivial_framing_is_transparent(HamB):
    """A +1 sign must reproduce the wrapped operator BITWISE (the wrapper itself
    adds nothing) -- and the framed operator must keep its metadata."""
    geo_n = HamB.hilbert.size
    triv = SignFramedOperator(HamB.base, lambda x: np.ones(np.asarray(x).shape[:-1]))
    x = 1.0 - 2.0 * np.random.default_rng(3).integers(0, 2, size=(64, geo_n)).astype(float)
    for a, b in zip(HamB.base.get_conn_padded(x), triv.get_conn_padded(x)):
        assert np.array_equal(np.asarray(a), np.asarray(b)), "trivial framing altered mels"
    assert HamB.dtype == HamB.base.dtype, "framing changed the dtype"
    assert HamB.max_conn_size == HamB.base.max_conn_size, "framing changed max_conn_size"
    assert HamB.is_hermitian and HamB.hilbert is HamB.base.hilbert


def test_logpsi(vsA, vsB, sign_fn, x):
    """(i) log|psi_A| == log A_B, and the phase of psi_A is exactly the sign."""
    lA, lB = np.asarray(vsA.log_value(x)), np.asarray(vsB.log_value(x))
    d = float(np.abs(lA.real - lB).max())
    assert d <= 1e-12, f"max |Re log psi_A - log A_B| = {d}"
    ph, s = np.exp(1j * lA.imag), sign_fn(x)
    assert np.array_equal(ph.real, s), "exp(i Im log psi_A) != (-1)^s exactly"
    assert set(np.unique(s).tolist()) == {-1.0, 1.0}, "sign head is not +-1"
    return d, float(np.abs(ph.imag).max()), int((s < 0).sum())


def test_local_energies(vsA, HamA, vsB, HamB, x):
    """(ii) per-config E_loc: bare H on the signed state == H~ on the positive one."""
    eA = _eloc(vsA, HamA, x, np.asarray(vsA.log_value(x)))
    eB = _eloc(vsB, HamB, x, np.asarray(vsB.log_value(x)))
    d = float(np.abs(eA.real - eB.real).max())
    assert d <= 1e-10, f"max |Re E_loc^A - E_loc^B| = {d}"
    assert np.abs(np.imag(eB)).max() == 0.0, "framed E_loc must be exactly real"
    return d, float(np.abs(eA.imag).max())


def test_expect_same_samples(vsA, HamA, vsB, HamB):
    """(iii) <H>_A == <H~>_B on the SAME Monte-Carlo samples."""
    vsA.reset()
    vsB._samples = vsA.samples                 # identical batch, no resampling
    EA, EB = complex(vsA.expect(HamA).mean), complex(vsB.expect(HamB).mean)
    d = abs(EA.real - EB.real)
    assert d <= 1e-10, f"|<H>_A - <H~>_B| = {d} (A {EA}, B {EB})"
    assert EB.imag == 0.0, "framed <H~> must be exactly real"
    return EA.real, EB.real, d, abs(EA.imag)


def test_hermiticity(HamB, x):
    """(iv) <x|H~|x'> == <x'|H~|x>* on every connected pair of a random batch."""
    xp, mels = (np.asarray(a) for a in HamB.get_conn_padded(x))
    worst, npairs = 0.0, 0
    for b in range(x.shape[0]):
        rp, rm = (np.asarray(a) for a in HamB.get_conn_padded(xp[b]))
        for k in range(xp.shape[1]):
            if mels[b, k] == 0:
                continue
            hit = np.flatnonzero((rp[k] == x[b]).all(-1) & (rm[k] != 0))
            assert hit.size, f"missing reverse matrix element for pair ({b},{k})"
            worst = max(worst, abs(complex(rm[k, hit].sum()).conjugate() - mels[b, k]))
            npairs += 1
    assert worst <= 1e-12, f"max |H~_xy - conj(H~_yx)| = {worst}"
    return worst, npairs


def test_table_head(geo, sign_fn):
    """table_sign(tabulated anaC) == the anaC head on all 2^12 configurations."""
    tab = sign_table(sign_fn, geo.N)
    all_x = 1.0 - 2.0 * ((np.arange(1 << geo.N)[:, None] >> np.arange(geo.N)) & 1)
    ok = np.array_equal(table_sign(tab, geo.N)(all_x.astype(float)), sign_fn(all_x.astype(float)))
    assert ok, "table head disagrees with the token-quadratic head"
    assert tab[0] == 1.0, "all-up must be index 0 with sign +1 (exact_diag bit order)"
    return int((tab < 0).sum()), tab.size


def test_real_sr_run(vs, Ham):
    """The real-dtype --sign_frame anaC build trains: 5 SR steps, finite energies.
    (Reuses the already-compiled B state -- it is the last thing the witness does,
    so mutating its parameters is safe.)"""
    assert isinstance(Ham, SignFramedOperator) and Ham.dtype == np.float64
    es = []
    run_loop(vs, Ham, n_iter=5, dt=0.02, diag_shift=1e-3, qgt="dense",
             on_step=lambda step, E, _vs: es.append(float(np.real(E.mean))))
    assert len(es) == 5 and all(np.isfinite(es)), f"non-finite SR energies: {es}"
    return es


class _FakeVS:
    """Minimal stand-in for a NetKet MCState -- only the surface exact_psi
    needs (`.variables`, `._apply_fun(variables, block)`) -- so the exact-eval
    regression test below can fix psi = A EXACTLY (a trained network can only
    approximate it) and still exercise the real eval_snapshots.py code path."""
    def __init__(self, logA_table):
        self.variables = {}
        self._logA = logA_table

    def _apply_fun(self, variables, block):
        X = np.asarray(block)
        idx = (((1.0 - X) / 2.0).astype(np.int64)
               * (1 << np.arange(X.shape[-1]))).sum(-1)
        return jnp.asarray(self._logA[idx])


def test_exact_eval_signs_psi():
    """(2026-09 audit, finding 1 regression) eval_snapshots.py's --exact path
    must sign the amplitude vector (psi = S*A) before scoring E0/fidelity/
    Vscore against the bare H -- the network only ever returns A. Ideal
    trunk A = |psi_ED| at h=0 must reproduce the exact ground state EXACTLY:
    E0 = -14, fidelity = 1, Vscore = 0 (was E0=-8, fidelity=0.0625 unframed)."""
    geo = ThreeD_ToricCodeGeometry(2, 2, 2, bc="OBC")
    xz = fermionic_plaquettes(geo)
    N, dim = geo.N, 1 << geo.N
    H, _ = hamiltonian_linop(geo, hx=0.0, hz=0.0, J=1.0, xz_stabs=xz, dtype=np.float64)
    op = sla.LinearOperator((dim, dim), matvec=H.matvec, dtype=np.float64)
    _w, v = sla.eigsh(op, k=1, which="SA")
    psi_ed = v[:, 0]
    A = np.abs(psi_ed)
    with np.errstate(divide="ignore"):
        fake_vs = _FakeVS(np.log(A))

    cfg = {"L": 2, "bc": "OBC", "model": "fermionic", "hx": 0.0, "hz": 0.0,
           "J": 1.0, "sign_frame": "anaC"}
    ctx = prepare_exact_context(geo, cfg, xz, ed_vectors=None)
    ctx["psi_ed"], ctx["ed_npz"] = psi_ed.astype(np.complex128), "test"
    out = exact_observables(fake_vs, ctx, chunk=1024)
    assert abs(out["E0"] - (-14.0)) < 1e-9, f"E0 = {out['E0']} (expected -14)"
    assert abs(out["Vscore"]) < 1e-9, f"Vscore = {out['Vscore']} (expected 0)"
    assert abs(out["fidelity"] - 1.0) < 1e-9, f"fidelity = {out['fidelity']} (expected 1)"
    return out["E0"], out["Vscore"], out["fidelity"]


def test_mean_ops_are_framed(geo, HamB, vsB, cfg):
    """(2026-09 audit, finding 2 regression) the MC-path mean_ops (bank_point.py/
    eval_ckpt.py/eval_snapshots.py's build_eval_operators output) must be routed
    through frame_eval_ops -- an OFF-DIAGONAL observable (B~_p, sx) must change
    under framing, a DIAGONAL one (sz) must not (S O S == O)."""
    mean_ops, _ = build_eval_operators(vsB.hilbert, geo, cfg, xz_stabs=fermionic_plaquettes(geo))
    framed = frame_eval_ops(mean_ops, cfg, geo)
    assert framed is not mean_ops, "frame_eval_ops was a no-op for sign_frame='anaC'"
    vsB.reset()
    vsB._samples = vsB.samples                     # freeze the batch across both calls
    u = nqs_observables(vsB, HamB, geo, xz_stabs=fermionic_plaquettes(geo), mean_ops=mean_ops)
    f = nqs_observables(vsB, HamB, geo, xz_stabs=fermionic_plaquettes(geo), mean_ops=framed)
    assert abs(u["B_p_mean"] - f["B_p_mean"]) > 1e-6, "framing had no effect on B~_p"
    assert abs(u["sx_mean"] - f["sx_mean"]) > 1e-6, "framing had no effect on sx"
    assert abs(u["sz_mean"] - f["sz_mean"]) < 1e-9, "sz (diagonal) must be sign-blind"
    return u["B_p_mean"], f["B_p_mean"], u["sz_mean"], f["sz_mean"]


def test_init_from_dtype_guard():
    """(2026-09 audit, finding 4 regression) --init_from a complex checkpoint
    into a real --sign_frame target must not silently promote the target to
    complex: negligible Im -> cast down (logged); non-negligible Im -> raise."""
    geoA, _hiA, _HamA, vsA, _ = build_state({**BASE, "phase_head_frozen": True})
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "ckpt")

        # (a) genuinely complex (random-init) trunk -> refuse, never promote
        save_model(vsA, base, verbose=False)
        _geoB, _hiB, _HamB, vsB, _ = build_state({**BASE, "sign_frame": "anaC"})
        try:
            load_weights(vsB, base)
        except TypeError as e:
            assert "refusing" in str(e).lower() and "complex" in str(e).lower(), str(e)
        else:
            raise AssertionError("load_weights silently promoted a real target to complex")

        # (b) negligible-imaginary (manually zeroed) trunk -> clean real cast
        vsA_real = build_state({**BASE, "phase_head_frozen": True})[3]
        vsA_real.parameters = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a.real, dtype=a.dtype), vsA.parameters)
        save_model(vsA_real, base, verbose=False)
        _geoB2, _hiB2, _HamB2, vsB2, _ = build_state({**BASE, "sign_frame": "anaC"})
        vsB2 = load_weights(vsB2, base)
        assert not any(np.iscomplexobj(np.asarray(p))
                       for p in jax.tree_util.tree_leaves(vsB2.parameters)), \
            "clean (negligible-Im) load must leave the target real"
    return True


def test_exclusivity():
    """sign_frame + phase_head* would apply the sign twice -> refuse loudly."""
    for k in ("phase_head", "phase_head_frozen"):
        try:
            build_state({**BASE, "sign_frame": "anaC", k: True})
        except ValueError as e:
            assert "twice" in str(e) or "double" in str(e).lower(), str(e)
        else:
            raise AssertionError(f"sign_frame + {k} was accepted")


if __name__ == "__main__":
    geo0 = ThreeD_ToricCodeGeometry(2, 2, 2, bc="OBC")
    tl, tq = anaC_theta(fermionic_plaquettes(geo0))
    print(f"  ok  anaC form: {np.count_nonzero(tl)} linear + {np.count_nonzero(tq)} "
          f"quadratic nonzeros over {len(tl)} tokens")

    geo, HamA, vsA, HamB, vsB, sfn = _states()
    x = _configs(geo)
    print(f"  ok  A/B states built ({x.shape[0]} configs: 64 on-support + "
          f"{x.shape[0] - 64} random)")

    test_trivial_framing_is_transparent(HamB)
    print("  ok  trivial (+1) framing is bitwise transparent; dtype/max_conn/"
          "hermiticity metadata preserved")

    d, dust, nm = test_logpsi(vsA, vsB, sfn, x)
    print(f"  ok  (i) max|Re log psi_A - log A_B| = {d:.3e}; exp(i Im log psi_A) == "
          f"(-1)^s exactly ({nm}/{x.shape[0]} negative; imag dust {dust:.2e})")

    d, dimA = test_local_energies(vsA, HamA, vsB, HamB, x)
    print(f"  ok  (ii) max|Re E_loc^A - E_loc^B| = {d:.3e} (A's exp(i pi) imag dust "
          f"{dimA:.2e}; B exactly real)")

    EA, EB, d, imA = test_expect_same_samples(vsA, HamA, vsB, HamB)
    print(f"  ok  (iii) <H>_A = {EA:.12f}  <H~>_B = {EB:.12f}  |diff| = {d:.3e}")

    w, npair = test_hermiticity(HamB, x[:16])
    print(f"  ok  (iv) hermiticity of H~ on {npair} connected pairs: max err {w:.3e}")

    nm, tot = test_table_head(geo, sfn)
    print(f"  ok  table_sign == anaC head on all {tot} configs ({nm} negative)")

    E0x, Vx, fidx = test_exact_eval_signs_psi()
    print(f"  ok  (2026-09 audit #1) eval_snapshots --exact signs psi before scoring: "
          f"E0={E0x:.6f}  Vscore={Vx:.3e}  fidelity={fidx:.6f} (ideal trunk, h=0)")

    cfgB = {**BASE, "sign_frame": "anaC"}
    Bu, Bf, Zu, Zf = test_mean_ops_are_framed(geo, HamB, vsB, cfgB)
    print(f"  ok  (2026-09 audit #2) MC-path mean_ops framed via frame_eval_ops: "
          f"<B~_p> unframed={Bu:.5f} framed={Bf:.5f} (differ); "
          f"<sz> unframed={Zu:.5f} framed={Zf:.5f} (sign-blind, match)")

    es = test_real_sr_run(vsB, HamB)
    print(f"  ok  real-trunk --sign_frame anaC SR run: E = "
          f"{' '.join(f'{e:.4f}' for e in es)}")

    test_init_from_dtype_guard()
    print("  ok  (2026-09 audit #4) load_weights refuses a genuinely-complex "
          "checkpoint into a real target; casts down a negligible-Im one")

    test_exclusivity()
    print("  ok  sign_frame + phase_head/phase_head_frozen refused")
