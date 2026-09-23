"""Witness for the learned-vs-gated sign-head benchmark arms (tc3d --sign_arm).

  (a) recovery_features: (eps, x) = b F mod 2 is injective on all 2^N configs
      (2x2x2 and 2x2x3 OBC), r = b + eps lies on the h=0 support, and
      b = x Gp + eps exactly;
  (b) the x bits are the application variables the spec means: on the
      support the exact sign is the C-form in the PLAQUETTE bits of x,
      (-1)^{x^T triu(M,1) x} == CupHead(r) on every support config;
  (c) TwoBranchNet: the stable log psi equals log(e^{a1} + s e^{a2}) evaluated
      directly, s is the pt2 table in the table_sign bit convention, and at
      c -> -inf the arm IS the head-only positive-trunk arm;
  (d) MLPSignNet: log psi == log A + log tanh(m) from the separately applied
      trunk and SignMLP, Im log psi in {0, pi};
  (e) load_sign_mlp round-trips pretrained params and refuses a shape mismatch;
  (f) with_defaults refuses the sign_arm combinations that would double-sign
      or run a complex trunk;
  (g) non-cubic OBC trunk (2x2x3: unequal per-orientation edge/plaquette
      counts, padded in KernelManager3D): every real site scattered exactly
      once, and the identity-init gridinv trunk is exactly A_v-invariant.

L=2 OBC NetKet builds + host numpy on N <= 20 only (CLAUDE.md-safe).
    cd tests && PYTHONPATH=.. ../.venv/bin/python test_signbench.py
"""
import os
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tc3d.builders import build_state, with_defaults
from tc3d.fermionic_decoration import fermionic_plaquettes
from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.sign_decoders import CupHead, make_decoder_sign, recovery_features
from tc3d.sign_frame import sign_table, table_sign
from tc3d.sign_geometry import orbit_sign_from_applications

BASE = dict(L=2, bc="OBC", model="fermionic", arch="ToricCNN_gridinv", kernel_size=2,
            noninv_hidden=[4, 8], inv_hidden=[8, 8], hx=0.5, hz=0.2, n_samples=256,
            n_chains=16, seed=0)


def _all_bits(N):
    idx = np.arange(1 << N, dtype=np.int64)
    return ((idx[:, None] >> np.arange(N)) & 1).astype(np.int64)


def test_feature_map_exhaustive():
    for dims in [(2, 2, 2), (2, 2, 3)]:
        geo = ThreeD_ToricCodeGeometry(*dims, bc="OBC")
        F, Gp, labels = recovery_features(geo)
        N = geo.N
        b = _all_bits(N)
        f = b @ F.astype(np.int64) % 2
        eps, x = f[:, :N], f[:, N:]
        assert np.unique(np.packbits(f.astype(np.uint8), axis=1), axis=0).shape[0] == 1 << N
        assert np.array_equal((x @ Gp.astype(np.int64) + eps) % 2, b)
        # eps depends on the syndrome only: at most 1 + #lit-class-combinations masks
        assert np.unique(eps, axis=0).shape[0] <= 2 ** 8
        r = (b + eps) % 2
        head = make_decoder_sign("linear", geo)
        assert not head.support.gid(r.astype(np.float32)).any(), "r = b + eps off the h=0 support"
        print(f"  (a) {dims}: F {F.shape}, injective on 2^{N}, round trip exact, r on support")


def test_x_is_application_variables():
    for dims in [(2, 2, 2), (2, 2, 3)]:
        geo = ThreeD_ToricCodeGeometry(*dims, bc="OBC")
        stabs = fermionic_plaquettes(geo)
        F, Gp, labels = recovery_features(geo)
        N, NP = geo.N, len(stabs)
        assert [l for l in labels[:NP]] == [("plaq", p) for p in range(NP)]
        rng = np.random.default_rng(1)
        xs = rng.integers(0, 2, size=(4000, Gp.shape[0]))
        r = xs @ Gp.astype(np.int64) % 2                  # every support config, randomly
        cup = CupHead(geo, stabs)
        s_cup = cup(1.0 - 2.0 * r)
        s_form = 1.0 - 2.0 * orbit_sign_from_applications(cup.cup, xs[:, :NP])
        # the star bits must not matter and the plaquette bits must be what F reads back
        fx = (r @ F.astype(np.int64) % 2)[:, N:]
        assert np.array_equal(fx, xs)
        assert np.array_equal(s_cup, s_form), \
            f"{dims}: cup sign != C-form in the x plaquette bits on {np.sum(s_cup != s_form)} configs"
        print(f"  (b) {dims}: on-support sign == (-1)^(x^T triu(M) x) in the plaquette bits of x")


def test_twobranch():
    geo, hi, H, vs, _ = build_state({**BASE, "sign_arm": "twobranch"})
    N = geo.N
    X = 1 - 2 * _all_bits(N)
    lp = np.asarray(vs.log_value(jnp.asarray(X, dtype=jnp.int8)))
    m, v = vs.model, vs.variables["params"]
    a1 = float(v["log_mix"]) + np.asarray(m.triv.apply({"params": v["triv"]}, X))
    a2 = np.asarray(m.top.apply({"params": v["top"]}, X))
    s = sign_table(make_decoder_sign("pt2", geo), N)
    assert np.array_equal(np.asarray(m.sign_table.a), s.astype(np.int8))
    assert np.array_equal(table_sign(np.asarray(m.sign_table.a), N)(X), s)
    direct = np.exp(a1) + s * np.exp(a2)
    assert np.max(np.abs(np.exp(lp) - direct) / np.abs(direct)) < 1e-12
    assert float(v["log_mix"]) == -3.0
    # c -> -inf: exactly the head-only arm psi = s A_top
    p = jax.tree_util.tree_map(lambda a: a, dict(vs.parameters))
    p["log_mix"] = jnp.asarray(-1e4)
    vs.parameters = p
    lp0 = np.asarray(vs.log_value(jnp.asarray(X, dtype=jnp.int8)))
    assert np.allclose(np.exp(lp0), s * np.exp(a2), rtol=1e-12, atol=0)
    print(f"  (c) twobranch: stable == direct (2^{N}), table == pt2 decoder, c->-inf == head-only")


def test_mlp_and_loader():
    geo, hi, H, vs, _ = build_state({**BASE, "sign_arm": "mlp"})
    N = geo.N
    X = jnp.asarray(1 - 2 * _all_bits(N), dtype=jnp.int8)
    lp = np.asarray(vs.log_value(X))
    m, v = vs.model, vs.variables["params"]
    from tc3d.networks import SignMLP, recovery_feature_pm
    logA = np.asarray(m.trunk.apply({"params": v["trunk"]}, X))
    mm = np.asarray(SignMLP(m.hidden).apply({"params": v["sign_mlp"]},
                                            recovery_feature_pm(X, m.feat_map)))
    ref = np.exp(logA) * np.tanh(mm)
    assert np.max(np.abs(np.exp(lp) - ref) / np.abs(ref)) < 1e-12
    im = np.mod(np.imag(lp), 2 * np.pi)
    assert np.all(np.isclose(im, 0) | np.isclose(im, np.pi) | np.isclose(im, 2 * np.pi))
    print(f"  (d) mlp: log psi == log A + log tanh m on 2^{N}, Im in {{0, pi}}")

    from flax import serialization
    from tc3d.train import load_sign_mlp
    new = jax.tree_util.tree_map(lambda a: np.asarray(a) + 0.5, dict(v["sign_mlp"]))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mlp.mpack")
        with open(path, "wb") as f:
            f.write(serialization.msgpack_serialize(new))
        vs = load_sign_mlp(vs, path)
        got = jax.tree_util.tree_leaves(vs.parameters["sign_mlp"])
        assert all(np.array_equal(np.asarray(a), b) for a, b in
                   zip(got, jax.tree_util.tree_leaves(new)))
        assert np.array_equal(np.asarray(jax.tree_util.tree_leaves(vs.parameters["trunk"])[0]),
                              np.asarray(jax.tree_util.tree_leaves(v["trunk"])[0]))
        bad = dict(new)
        bad["Dense_0"] = {"kernel": np.zeros((3, 64)), "bias": np.zeros(64)}
        with open(path, "wb") as f:
            f.write(serialization.msgpack_serialize(bad))
        try:
            load_sign_mlp(vs, path)
            raise AssertionError("shape mismatch not refused")
        except ValueError:
            pass
    print("  (e) load_sign_mlp: round trip exact, trunk untouched, shape mismatch refused")


def test_config_guards():
    ok = with_defaults({**BASE, "sign_arm": "mlp"})
    assert ok["dtype"] == "float64", "sign_arm trunk must default to real"
    for bad in ({"sign_frame": "pt2"}, {"phase_head_frozen": True}, {"hy": 0.2},
                {"dtype": "complex"}, {"arch": "GeoCNN"}, {"model": "bosonic"}):
        try:
            with_defaults({**BASE, "sign_arm": "twobranch", **bad})
            raise AssertionError(f"sign_arm accepted {bad}")
        except ValueError:
            pass
    print("  (f) with_defaults: real trunk default, incompatible combos refused")


def test_noncubic_trunk():
    from tc3d.builders import build_model
    from tc3d.networks import KernelManager3D
    geo = ThreeD_ToricCodeGeometry(2, 2, 3, bc="OBC")
    km = KernelManager3D(geo)
    for out, M in ((km.edge_out.ravel(), geo.N), (km.plaq_out.ravel(), km.N_plaq)):
        assert np.array_equal(np.sort(out[out < M]), np.arange(M)) and (out <= M).all()
    assert (km.edge_mask[km.edge_out == geo.N] == 0).all()
    cfg = with_defaults({**BASE, "Lxyz": [2, 2, 3]})
    m = build_model({**cfg, "dtype": "float64"}, geo)
    x = 1.0 - 2.0 * np.random.default_rng(0).integers(0, 2, (128, geo.N))
    params = m.init(jax.random.PRNGKey(1), x[:2])
    y0 = np.asarray(m.apply(params, x))
    for v in geo.vertex_all:
        xs = x.copy()
        xs[:, [e for e in v if e != -1]] *= -1
        assert np.max(np.abs(np.asarray(m.apply(params, xs)) - y0)) < 1e-12
    try:
        with_defaults({**BASE, "Lxyz": [2, 2, 3], "bc": "PBC"})
        from tc3d.builders import build_geometry
        build_geometry({**BASE, "Lxyz": [2, 2, 3], "bc": "PBC"})
        raise AssertionError("non-cubic PBC accepted")
    except ValueError:
        pass
    print("  (g) 2x2x3 OBC trunk: padded stencils scatter each site once, exact A_v invariance")


if __name__ == "__main__":
    test_feature_map_exhaustive()
    test_x_is_application_variables()
    test_config_guards()
    test_twobranch()
    test_mlp_and_loader()
    test_noncubic_trunk()
    print("ALL OK")
