"""Witness for the per-configuration sign heads (`tc3d.sign_decoders`).

The four heads (`cup`, `linear`, `vote`, `pt2`) must reproduce, row by row, what
`analysis/scripts/sign_fidelity_ftc.py` computes by enumerating the whole 2^N
basis -- so the ED-sized reference tables in `results/fermionic_gate0/` are the
ground truth here:

  (a) linear / vote / pt2 == sign_table_*_2x2x2_OBC.npy on all 2^12 configs and
      sign_table_*_2x2x3_OBC.npy on all 2^20 configs, bit for bit, with zero
      cap fallbacks;
  (b) cup == CupSign.sign on random configs at L=3 and L=4 OBC;
  (c) SignFramedOperator(H, pt2 decoder) == SignFramedOperator(H, table_sign(pt2
      table)) matrix element for matrix element at L=2 OBC;
  (d) pop_stats() bookkeeping (n_rows, cap fallbacks, tie/pt2 counters).

Everything is host numpy on N <= 20 plus one small NetKet build at L=2 OBC --
safe on the dev machine (CLAUDE.md: never run 3D TC ED/sweeps locally).

Run directly:
    cd tests && PYTHONPATH=.. ../.venv/bin/python test_sign_decoders.py
"""

import os
import time

import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.sign_decoders import KINDS, make_decoder_sign
from tc3d.sign_frame import SignFramedOperator, table_sign
from tc3d.sign_geometry import CupSign

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "fermionic_gate0")
CHUNK = 1 << 16


def _configs_from_ints(j, N):
    """+-1 configs from basis integers (qubit i = bit i, bit 1 = spin down)."""
    return 1.0 - 2.0 * ((j[:, None] >> np.arange(N)) & 1).astype(np.float64)


def _table_path(head, tag):
    return os.path.join(TABLES, f"sign_table_{head}_{tag}.npy")


def test_tables(Lx, Ly, Lz, tag):
    """(a) every recovered head == its 2^N reference table, bit for bit."""
    geo = ThreeD_ToricCodeGeometry(Lx, Ly, Lz, bc="OBC")
    N, dim = geo.N, 1 << geo.N
    out = {}
    for head in ("linear", "vote", "pt2"):
        path = _table_path(head, tag)
        if not os.path.exists(path):
            print(f"  SKIP {tag} {head}: {path} absent (regenerate with "
                  f"analysis/scripts/sign_fidelity_ftc.py --export_tables)")
            continue
        tab = np.load(path)
        assert tab.size == dim, f"{path}: {tab.size} entries, expected 2^{N}"
        fn = make_decoder_sign(head, geo)
        t0 = time.time()
        got = np.empty(dim, np.int8)
        for a in range(0, dim, CHUNK):
            j = np.arange(a, min(a + CHUNK, dim), dtype=np.int64)
            s = fn(_configs_from_ints(j, N))
            assert set(np.unique(s).tolist()) <= {-1.0, 1.0}, "head is not +-1"
            got[a:a + j.size] = s.astype(np.int8)
        st = fn.pop_stats()
        bad = int((got != tab).sum())
        assert bad == 0, f"{tag} {head}: {bad}/{dim} configs disagree with {path}"
        assert st["n_rows"] == dim, f"n_rows={st['n_rows']} != {dim}"
        assert st["n_fallback"] == 0, f"{head} hit the k_cap {st['n_fallback']}x"
        out[head] = (st, time.time() - t0)
    return out


def test_cup_base_equals_anaC_on_support(Lx, Ly, Lz):
    """The premise behind (a): the cup-product base sign and the frozen analytic
    token-quadratic head (which generated the reference tables) agree on EVERY
    on-support config -- and the recovered heads only ever read the base there,
    so a cup-based decoder reproduces anaC-based tables. They legitimately
    differ off support (solver-gauge shadow), which is why this is scoped."""
    from tc3d.sign_frame import anaC_sign

    geo = ThreeD_ToricCodeGeometry(Lx, Ly, Lz, bc="OBC")
    N, dim = geo.N, 1 << geo.N
    cup, ana = make_decoder_sign("cup", geo), anaC_sign(geo)
    sup = make_decoder_sign("linear", geo).support
    n_on = n_off_diff = 0
    for a in range(0, dim, CHUNK):
        j = np.arange(a, min(a + CHUNK, dim), dtype=np.int64)
        x = _configs_from_ints(j, N)
        on = sup.label((x < 0).astype(np.float64))[0] == 0
        sc, sa = cup(x), ana(x)
        assert np.array_equal(sc[on], sa[on]), \
            f"{Lx}x{Ly}x{Lz}: cup != anaC on {int((sc[on] != sa[on]).sum())} " \
            "ON-SUPPORT configs"
        n_on += int(on.sum())
        n_off_diff += int((sc[~on] != sa[~on]).sum())
    cup.pop_stats()
    return n_on, dim - n_on, n_off_diff


def test_cup_matches_cupsign(Lx, Ly, Lz, n=20000, seed=0):
    """(b) the `cup` head is exactly `CupSign.sign` (plus stats bookkeeping)."""
    geo = ThreeD_ToricCodeGeometry(Lx, Ly, Lz, bc="OBC")
    rng = np.random.default_rng(seed)
    x = 1.0 - 2.0 * rng.integers(0, 2, size=(n, geo.N)).astype(np.float64)
    fn = make_decoder_sign("cup", geo)
    got, ref = fn(x), CupSign(geo).sign(x)
    st = fn.pop_stats()
    assert np.array_equal(got, ref), \
        f"{Lx}x{Ly}x{Lz}: cup head != CupSign.sign on {int((got != ref).sum())} rows"
    assert st["n_rows"] == n and st["n_fallback"] == st["n_pt2"] == st["n_tie"] == 0
    return st


def test_leading_shape_preserved(seed=3):
    """The SignFramedOperator contract: (..., N) in -> (...) out, any leading rank."""
    geo = ThreeD_ToricCodeGeometry(2, 2, 2, bc="OBC")
    rng = np.random.default_rng(seed)
    x = 1.0 - 2.0 * rng.integers(0, 2, size=(7, 5, geo.N)).astype(np.float64)
    for kind in KINDS:
        fn = make_decoder_sign(kind, geo)
        s = fn(x)
        assert s.shape == (7, 5), f"{kind}: got shape {s.shape}, expected (7, 5)"
        flat = fn(x.reshape(-1, geo.N))
        assert np.array_equal(s.reshape(-1), flat), f"{kind}: reshape-dependent"
        assert fn.pop_stats()["n_rows"] == 70, f"{kind}: n_rows != 70"
        assert fn.pop_stats()["n_rows"] == 0, f"{kind}: pop_stats did not reset"


def test_framed_ab(tag="2x2x2_OBC", n=256, seed=5):
    """(c) framed H~ mels are identical for the pt2 decoder and the pt2 table."""
    from tc3d.builders import build_state
    from tests.test_sign_frame import BASE, _configs        # shared fixtures

    path = _table_path("pt2", tag)
    if not os.path.exists(path):
        print(f"  SKIP framed A/B: {path} absent")
        return None
    geo, _hi, Ham, _vs, _ = build_state({**BASE, "sign_frame": "pt2"})
    assert isinstance(Ham, SignFramedOperator), "builders did not frame H"
    tab_op = SignFramedOperator(Ham.base, table_sign(np.load(path), geo.N))
    x = _configs(geo, n_random=n - 64, seed=seed)
    xa, ma = (np.asarray(a) for a in Ham.get_conn_padded(x))
    xb, mb = (np.asarray(a) for a in tab_op.get_conn_padded(x))
    assert np.array_equal(xa, xb), "connected configs differ between the two heads"
    assert np.array_equal(ma, mb), \
        f"max |mel diff| = {np.abs(ma - mb).max()} between decoder and table pt2"
    st = Ham.sign_fn.pop_stats()
    return x.shape[0], int(ma.size), st


def test_k_cap_fallback(L=4, n=4000, seed=1):
    """(d) a tight k_cap must fall back to `linear` -- and be counted."""
    geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
    rng = np.random.default_rng(seed)
    x = 1.0 - 2.0 * rng.integers(0, 2, size=(n, geo.N)).astype(np.float64)
    lin = make_decoder_sign("linear", geo)(x)
    out = {}
    for kind in ("vote", "pt2"):
        wide = make_decoder_sign(kind, geo, k_cap=8)
        s_wide = wide(x)
        st_wide = wide.pop_stats()
        tight = make_decoder_sign(kind, geo, k_cap=1)
        s_tight = tight(x)
        st = tight.pop_stats()
        assert st_wide["n_fallback"] == 0, f"{kind}: k_cap=8 already falls back"
        assert st["n_fallback"] > 0, f"{kind}: k_cap=1 never fell back"
        assert st["n_rows"] == n and st_wide["n_rows"] == n
        assert st["k_max"] == st_wide["k_max"] and st["k_sum"] == st_wide["k_sum"]
        # every capped row must equal the linear head; some uncapped row must not
        assert (s_tight == lin).sum() >= st["n_fallback"], \
            f"{kind}: fewer linear-agreeing rows than fallbacks"
        assert not np.array_equal(s_wide, lin), \
            f"{kind}: k_cap=8 head is indistinguishable from linear (test is blind)"
        out[kind] = (st_wide, st)
    return out


def test_pbc_scope(L=2):
    """`cup` works at PBC; the recovery decoders refuse it with a clear error."""
    geo = ThreeD_ToricCodeGeometry(L, L, L, bc="PBC")
    rng = np.random.default_rng(0)
    x = 1.0 - 2.0 * rng.integers(0, 2, size=(256, geo.N)).astype(np.float64)
    assert np.array_equal(make_decoder_sign("cup", geo)(x), CupSign(geo).sign(x))
    for kind in ("linear", "vote", "pt2"):
        try:
            make_decoder_sign(kind, geo)
        except ValueError as e:
            assert "decompose" in str(e) and "PBC" in str(e), str(e)
        else:
            raise AssertionError(f"{kind} was accepted at PBC")


def bench(n=20000, seed=7, Ls=(2, 3, 4, 5, 6)):
    """Informational: microseconds per row for every head, single thread.

    Reported separately: `ctor` (geometry precompute), `build` (the first pass,
    which lazily builds the per-coset recovery structures) and the STEADY-state
    per-row cost of a second identical pass -- the number a training step pays.
    """
    rows = []
    for L in Ls:
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        rng = np.random.default_rng(seed)
        x = 1.0 - 2.0 * rng.integers(0, 2, size=(n, geo.N)).astype(np.float64)
        per = {}
        for kind in KINDS:
            t0 = time.time()
            fn = make_decoder_sign(kind, geo)
            ctor = time.time() - t0
            t0 = time.time()
            fn(x)                                 # builds every coset structure
            build = time.time() - t0
            fn.pop_stats()
            t0 = time.time()
            fn(x)
            per[kind] = ((time.time() - t0) / n * 1e6, ctor, build)
            per[kind + "_stats"] = fn.pop_stats()
        rows.append((L, geo.N, per))
    return rows


def bench_flips(L=4, n=20000, seed=11, kmax=5):
    """Informational: per-row cost vs the number of flipped spins at fixed L.

    `cup`/`linear` are flat (a fixed N x N product dominates); `vote`/`pt2` grow
    with the number of LIT LINE CLASSES a config carries, which is what the
    recovery stage enumerates over.
    """
    geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
    N = geo.N
    rng = np.random.default_rng(seed)
    heads = {k: make_decoder_sign(k, geo) for k in KINDS}
    warm = 1.0 - 2.0 * rng.integers(0, 2, size=(4000, N)).astype(np.float64)
    for fn in heads.values():                     # build every coset structure
        fn(warm)
        fn.pop_stats()
    out = []
    for nf in range(kmax + 1):
        b = np.zeros((n, N), dtype=np.int64)
        if nf:
            idx = np.argsort(rng.random((n, N)), axis=1)[:, :nf]
            np.put_along_axis(b, idx, 1, axis=1)
        x = 1.0 - 2.0 * b.astype(np.float64)
        per, k_mean = {}, 0.0
        for kind in KINDS:
            fn = heads[kind]
            fn(x)                                 # page the data in
            fn.pop_stats()
            t0 = time.time()
            fn(x)
            per[kind] = (time.time() - t0) / n * 1e6
            st = fn.pop_stats()
            k_mean = st["k_sum"] / st["n_rows"]
        out.append((nf, k_mean, per))
    return geo.N, out


def test_max_terms_fallback(L=3, n=4000, seed=2):
    """`max_terms` must cap the recovery COUNT (not just the class count) and
    route the over-cap rows to `linear`, counted in n_fallback."""
    geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
    rng = np.random.default_rng(seed)
    x = 1.0 - 2.0 * rng.integers(0, 2, size=(n, geo.N)).astype(np.float64)
    lin = make_decoder_sign("linear", geo)(x)
    out = {}
    for kind in ("vote", "pt2"):
        wide = make_decoder_sign(kind, geo)                    # 200k default
        s_wide, st_wide = wide(x), wide.pop_stats()
        # L=3 OBC: two lit classes of 12 -> the k=2 coset asks for 144 recoveries
        tight = make_decoder_sign(kind, geo, k_cap=99, max_terms=100)
        s_tight, st = tight(x), tight.pop_stats()
        assert st_wide["n_fallback"] == 0, f"{kind}: default max_terms falls back"
        assert st["n_fallback"] > 0, f"{kind}: max_terms=100 never fell back"
        assert not np.array_equal(s_wide, s_tight), \
            f"{kind}: capping changed nothing (test is blind)"
        assert np.array_equal(s_tight[s_tight != s_wide], lin[s_tight != s_wide]), \
            f"{kind}: a capped row did not fall back to the linear head"
        out[kind] = (st_wide["n_fallback"], st["n_fallback"], st["n_rows"])
    return out


if __name__ == "__main__":
    for (Lx, Ly, Lz) in [(2, 2, 2), (2, 2, 3)]:
        tag = f"{Lx}x{Ly}x{Lz}_OBC"
        res = test_tables(Lx, Ly, Lz, tag)
        for head, (st, dt) in res.items():
            print(f"  ok  {tag} {head}: bit-exact vs sign_table_{head}_{tag}.npy on "
                  f"all 2^{ThreeD_ToricCodeGeometry(Lx, Ly, Lz, bc='OBC').N} configs "
                  f"({dt:.1f}s; n_tie={st['n_tie']} n_pt2={st['n_pt2']} "
                  f"k_max={st['k_max']} k_mean={st['k_sum'] / st['n_rows']:.3f}, "
                  f"0 fallbacks)")

    for (Lx, Ly, Lz) in [(2, 2, 2), (2, 2, 3)]:
        n_on, n_off, n_diff = test_cup_base_equals_anaC_on_support(Lx, Ly, Lz)
        print(f"  ok  {Lx}x{Ly}x{Lz} OBC: cup base == anaC head on all {n_on} "
              f"on-support configs ({n_diff}/{n_off} off-support configs differ, "
              "as expected)")

    for (Lx, Ly, Lz) in [(3, 3, 3), (4, 4, 4)]:
        st = test_cup_matches_cupsign(Lx, Ly, Lz)
        print(f"  ok  {Lx}x{Ly}x{Lz} OBC: cup head == CupSign.sign on "
              f"{st['n_rows']} random configs (k_max={st['k_max']}, "
              f"k_mean={st['k_sum'] / st['n_rows']:.3f})")

    test_leading_shape_preserved()
    print("  ok  (..., N) -> (...) shape contract + pop_stats reset, all 4 heads")

    test_pbc_scope()
    print("  ok  PBC: cup exact, linear/vote/pt2 refused (no unique coset "
          "decomposition into lit line classes)")

    caps = test_k_cap_fallback()
    for kind, (wide, tight) in caps.items():
        print(f"  ok  {kind} k_cap: 8 -> {wide['n_fallback']} fallbacks, "
              f"1 -> {tight['n_fallback']}/{tight['n_rows']} rows fall back to "
              f"linear (k_max={tight['k_max']})")

    ab = test_framed_ab()
    if ab is not None:
        nx, nmel, st = ab
        print(f"  ok  framed A/B: SignFramedOperator(H, pt2 decoder) == "
              f"SignFramedOperator(H, table_sign(pt2)) on {nx} configs "
              f"({nmel} matrix elements; head saw n_rows={st['n_rows']}, "
              f"n_pt2={st['n_pt2']})")

    mt = test_max_terms_fallback()
    for kind, (wide, tight, tot) in mt.items():
        print(f"  ok  {kind} max_terms: 200000 -> {wide} fallbacks, 100 -> "
              f"{tight}/{tot} rows fall back to linear")

    print("  --  microseconds per row, steady state (20k random configs, "
          "single thread; ctor + first-pass build shown separately):")
    for L, N, per in bench():
        print(f"      L={L} OBC (N={N:3d}): " + "  ".join(
            f"{k}={per[k][0]:6.2f}us" for k in KINDS))
        print(f"                     ctor/build: " + "  ".join(
            f"{k}={per[k][1]:.2f}/{per[k][2]:.2f}s" for k in KINDS))
    N4, flips = bench_flips()
    print(f"  --  us/row vs flipped spins at L=4 OBC (N={N4}), 20k configs:")
    print("      flips  lit-classes  " + "  ".join(f"{k:>8}" for k in KINDS))
    for nf, k_mean, per in flips:
        print(f"      {nf:>5}  {k_mean:>11.3f}  " + "  ".join(
            f"{per[k]:8.2f}" for k in KINDS))
