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
  (d) pop_stats() bookkeeping (n_rows, cap fallbacks, tie/pt2 counters);
  (e) the connected-set fast path `sign_conn(x, xp)` == signing x and xp
      separately, row for row, on the REAL fermionic H's get_conn_padded at
      L = 2, 3, 4 OBC (plus the framed matrix elements it feeds);
  (f) the two geometry identities that fast path and the pt2 tie-breaker rest
      on: the (m+1)-flip second-order candidate set IS the class product
      C_0 x ... x C_{m-1} x C_unlit, and a "gauge" mask really does multiply
      the recovered heads by a constant.

Everything is host numpy on N <= 20 plus one small NetKet build at L=2 OBC --
safe on the dev machine (CLAUDE.md: never run 3D TC ED/sweeps locally).

Run directly:
    cd tests && PYTHONPATH=.. ../.venv/bin/python test_sign_decoders.py
"""

import itertools
import math
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


def _fermionic_H(geo, hx=0.5, hz=0.2):
    """The real fermionic H as NetKet PauliStrings (no variational state)."""
    import netket as nk
    from tc3d.fermionic_decoration import fermionic_plaquettes
    from tc3d.hamiltonian import create_hamiltonian_fermionic

    hi = nk.hilbert.Spin(s=0.5, N=geo.N)
    return create_hamiltonian_fermionic(hi, geo.vertex_all,
                                        fermionic_plaquettes(geo), [],
                                        hx=hx, hz=hz, J=1.0, dtype=float)


def test_conn_fast_path(L=3, n=192, seed=13):
    """(e) sign_conn(x, xp) is bit-identical to signing x and xp separately.

    The head serves each connected row from its SAMPLE's decode (they differ by
    a known XOR mask), so this is the witness that the exact GF(2) mask update
    -- and the mask identification behind it -- never changes a single sign.
    Run against the operator's OWN get_conn_padded, because that is where the
    padding / zero-mel left-packing lives (a connected COLUMN does not carry a
    fixed mask, which is why the head identifies rows rather than columns).
    """
    geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
    Ham = _fermionic_H(geo)
    rng = np.random.default_rng(seed)
    x = 1.0 - 2.0 * rng.integers(0, 2, size=(n, geo.N)).astype(np.float64)
    xp, mels = (np.asarray(a) for a in Ham.get_conn_padded(x))
    out = {}
    for kind in KINDS:
        fn = make_decoder_sign(kind, geo)
        s_ref, sp_ref = fn(x), fn(xp)
        ref = fn.pop_stats()
        s, sp = fn.sign_conn(x, xp)
        st = fn.pop_stats()
        assert np.array_equal(s, s_ref), f"{kind}: sample signs differ"
        assert np.array_equal(sp, sp_ref), \
            f"{kind}: {int((sp != sp_ref).sum())}/{sp.size} connected signs differ"
        assert st["n_rows"] == ref["n_rows"] == n * (1 + xp.shape[1])
        assert st["k_sum"] == ref["k_sum"] and st["k_max"] == ref["k_max"], \
            f"{kind}: lit-class bookkeeping changed on the fast path"
        assert st["n_decoded"] <= ref["n_decoded"], f"{kind}: fast path decoded more"
        out[kind] = (st["n_decoded"], st["n_rows"], len(fn._masks.idx))
    # ... and through the operator, matrix element for matrix element
    fn, bare = make_decoder_sign("pt2", geo), make_decoder_sign("pt2", geo)
    _xp, m_fast = SignFramedOperator(Ham, fn).get_conn_padded(x)
    m_ref = mels * bare(x)[:, None] * bare(xp)
    assert np.array_equal(np.asarray(m_fast), m_ref), "framed mels differ"
    return xp.shape[1], out


def test_second_order_is_a_product(Ls=(2, 3, 4), max_comb=3_000_000):
    """(f1) the (m+1)-flip tie-breaking set == product(lit classes) x class 0.

    The lit-class labels are independent, so an (m+1)-flip subset can only carry
    the coset label by taking one edge from each lit class plus one label-0
    edge. That is what lets `_second_struct` contract the second order instead
    of gathering over an unstructured C(N, m+1) enumeration -- so check it
    against that very enumeration.
    """
    out = []
    for L in Ls:
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        sup = make_decoder_sign("pt2", geo).support
        lab = np.array(sup.lab, dtype=np.int64)
        cls0, n_lit, done = sup.classes.get(0, []), len(sup.lit), 0
        for gid in range(1, 1 << n_lit):
            present = [i for i in range(n_lit) if (gid >> i) & 1]
            u = int(np.bitwise_xor.reduce([sup.lit[i] for i in present]))
            k = len(present) + 1
            if math.comb(geo.N, k) > max_comb:
                continue
            brute = {c for c in itertools.combinations(range(geo.N), k)
                     if int(np.bitwise_xor.reduce([lab[e] for e in c])) == u}
            prod = {tuple(sorted(t)) for t in itertools.product(
                *[sup.classes[sup.lit[i]] for i in present], cls0)}
            assert brute == prod, \
                f"L={L} gid={gid}: {len(brute)} enumerated vs {len(prod)} product"
            done += 1
        out.append((L, geo.N, done, 1 << n_lit))
    return out


def test_gauge_masks(Ls=(2, 3, 4), n=1500, seed=21):
    """(f2) a mask the head calls "gauge" multiplies it by a CONSTANT sign.

    That is what lets the connected-set fast path serve the vertex-star
    neighbours for free. Also pins the negative: `cup` reads the base sign OFF
    support, so it is NOT star-gauge invariant there, and must not claim to be.
    """
    from tc3d.fermionic_decoration import fermionic_plaquettes, _mask

    out = []
    for L in Ls:
        geo = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
        rng = np.random.default_rng(seed)
        b = rng.integers(0, 2, size=(n, geo.N))
        x0 = 1.0 - 2.0 * b.astype(np.float64)
        masks = [[e for e in v if e != -1] for v in geo.vertex_all]
        masks += [list(xe) for _z, xe, _c in fermionic_plaquettes(geo)]
        n_gauge = {}
        for kind in KINDS:
            fn = make_decoder_sign(kind, geo)
            s0 = fn(x0)
            fn._masks = None
            fn.sign_conn(x0[:8], np.repeat(x0[:8, None, :], 2, axis=1))  # init tables
            ng = 0
            for mk in masks:
                v = np.zeros(geo.N, dtype=np.int64)
                v[mk] = 1
                e = np.array(sorted(mk), dtype=np.int64)
                dK = (fn.Ksym[e].sum(axis=0) & 1).astype(np.int8)
                gid = fn.support.gid_of_edges(e) if fn.support is not None else 0
                if not fn._is_gauge(e, dK, gid):
                    continue
                ng += 1
                c = 1.0 - 2.0 * (int(v @ fn.K.astype(np.int64) @ v) & 1)
                s1 = fn(1.0 - 2.0 * ((b ^ v[None, :]) % 2).astype(np.float64))
                assert np.array_equal(s1, c * s0), \
                    f"L={L} {kind}: a 'gauge' mask is not a constant sign shift"
            fn.pop_stats()
            n_gauge[kind] = ng
        assert n_gauge["linear"] == n_gauge["vote"] == n_gauge["pt2"] \
            == len(geo.vertex_all), \
            f"L={L}: expected every vertex star to be gauge, got {n_gauge}"
        assert n_gauge["cup"] < n_gauge["pt2"], \
            f"L={L}: cup must NOT be star-gauge invariant off support ({n_gauge})"
        out.append((L, n_gauge, len(masks)))
    return out


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

    for L, N, done, tot in test_second_order_is_a_product():
        print(f"  ok  {L}x{L}x{L} OBC: the pt2 second-order candidate set is the "
              f"lit-class product x class 0 on {done}/{tot - 1} cosets "
              f"(vs the C({N}, m+1) enumeration)")

    for L, ng, nm in test_gauge_masks():
        print(f"  ok  {L}x{L}x{L} OBC: every gauge mask shifts its head by a "
              f"constant; linear/vote/pt2 call {ng['pt2']}/{nm} of the "
              f"star+x-pair masks gauge, cup only {ng['cup']}")

    for L in (2, 3, 4):
        nc, per = test_conn_fast_path(L=L)
        head = "  ".join(f"{k}={d}/{t}" for k, (d, t, _m) in per.items())
        print(f"  ok  {L}x{L}x{L} OBC: sign_conn == per-row signing on the real "
              f"fermionic H (n_conn={nc}, {per['pt2'][2]} masks); decoded {head}")

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
