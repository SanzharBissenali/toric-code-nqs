"""Tests for tc3d.sign_geometry.CupSign -- the vectorized cup-product fermionic
h=0 sign (notes/fermionic_sign_geometry.md theory,
notes/fermionic_sign_geometry_numerics.md SS4/SS5 the verified recipe).

All geometries here are OBC 2^3/3^3/2x2x3 and PBC L=2,3 -- GF(2) dense algebra on
N,NP <= 81, never a 2^N object except the exhaustive L=2 OBC enumeration (2^12).

Run directly:
    cd tests && PYTHONPATH=.. ../.venv/bin/python test_sign_geometry.py
"""

import time

import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.sign_frame import anaC_sign
from tc3d.sign_geometry import CupSign, random_orbit_states

GEOMS = [(2, 2, 2, "OBC"), (3, 3, 3, "OBC"), (2, 2, 3, "OBC"),
         (2, 2, 2, "PBC"), (3, 3, 3, "PBC")]


def _cupsign(Lx, Ly, Lz, bc):
    return CupSign(ThreeD_ToricCodeGeometry(Lx, Ly, Lz, bc))


def test_orbit_sign_exact(seed=0, n=2000):
    """(1) CupSign.sign == the exact orbit sign (application-variable formula)
    on random pair-move + random-star-gauge physical states."""
    for (Lx, Ly, Lz, bc) in GEOMS:
        cs = _cupsign(Lx, Ly, Lz, bc)
        rng = np.random.default_rng(seed)
        configs, s_true = random_orbit_states(cs, rng, n)
        s_pred = (cs.sign(configs) < 0).astype(np.uint8)
        agree = int((s_pred == s_true).sum())
        assert agree == n, f"{Lx}x{Ly}x{Lz} {bc}: orbit sign {agree}/{n}"
        print(f"  ok  {Lx}x{Ly}x{Lz} {bc}: orbit sign exact on {agree}/{n} "
              f"(N={cs.N}, NP={cs.NP}, n_lines={cs.Phi.shape[0]})")


def test_star_invariance_offsupport(seed=1, n=500):
    """(2) sign(b) vs sign(b + star) for a random star, on random (mostly
    off-support) bitstrings. The theory note's star-invariance claim ([D])
    is scoped to the physical orbit; here we MEASURE it on arbitrary
    bitstrings rather than assume it -- PBC is exact everywhere we checked,
    OBC only away from the physical/gauge orbit is not guaranteed (a
    boundary pair-move flips a single edge, so an arbitrary off-support b
    need not sit in any coset the gauge-fix respects)."""
    for (Lx, Ly, Lz, bc) in GEOMS:
        cs = _cupsign(Lx, Ly, Lz, bc)
        rng = np.random.default_rng(seed)
        N = cs.N
        b = rng.integers(0, 2, size=(n, N))
        s0 = cs.sign(1.0 - 2.0 * b.astype(np.float64))
        j = rng.integers(0, cs.Sv.shape[0], size=n)
        b2 = (b + cs.Sv[j]) % 2
        s2 = cs.sign(1.0 - 2.0 * b2.astype(np.float64))
        agree = int((s0 == s2).sum())
        tag = "exact" if bc == "PBC" else "off-support, see note above"
        print(f"  ..  {Lx}x{Ly}x{Lz} {bc}: star-invariance on random bitstrings "
              f"{agree}/{n} ({tag})")
        if bc == "PBC":
            assert agree == n, f"{Lx}x{Ly}x{Lz} PBC: expected exact star invariance"


def test_anaC_agreement_l2obc():
    """(3) L=2 OBC, ALL 4096 configs vs tc3d.sign_frame.anaC_sign: on-support
    agreement must be 1.0; off-support is reported (may legitimately differ --
    the two heads use unrelated closed forms off the physical orbit)."""
    geom = ThreeD_ToricCodeGeometry(2, 2, 2, "OBC")
    cs = CupSign(geom)
    N = geom.N
    all_x = 1.0 - 2.0 * ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1).astype(np.float64)
    cup_s = cs.sign(all_x)
    ana_s = anaC_sign(geom)(all_x)

    # Exact physical orbit by brute force (N=12, NP=6, n_keep<=8 -- all small).
    NP, n_keep = cs.NP, cs.Sv.shape[0]
    pw = 1 << np.arange(N)
    onsupport = np.zeros(1 << N, dtype=bool)
    for xi in range(1 << NP):
        xb = (xi >> np.arange(NP)) & 1
        b0 = (xb @ cs.X.astype(np.int64)) % 2
        for yi in range(1 << n_keep):
            yb = (yi >> np.arange(n_keep)) & 1
            b = (b0 + yb @ cs.Sv.astype(np.int64)) % 2
            onsupport[int(b @ pw)] = True

    agree_on = float((cup_s[onsupport] == ana_s[onsupport]).mean())
    agree_off = float((cup_s[~onsupport] == ana_s[~onsupport]).mean())
    assert agree_on == 1.0, f"on-support anaC agreement {agree_on} != 1.0"
    print(f"  ok  L=2 OBC vs anaC: on-support agreement {agree_on:.4f} "
          f"({int(onsupport.sum())}/{1 << N} configs); off-support agreement "
          f"{agree_off:.4f} ({int((~onsupport).sum())} configs, may legitimately differ)")


def test_line_parities():
    """(4) all-up config -> zero line parities everywhere; a single sigma^x
    flip lights AT MOST one detector, and lights EXACTLY one at PBC always
    (OBC boundary edges may light zero -- a documented 'zero-syndrome
    boundary flip', notes/fermionic_sign_geometry.md SSC)."""
    for (Lx, Ly, Lz, bc) in GEOMS:
        cs = _cupsign(Lx, Ly, Lz, bc)
        N = cs.N
        allup = np.ones((1, N))
        assert (cs.line_parities(allup) == 0).all(), \
            f"{Lx}x{Ly}x{Lz} {bc}: all-up line parities nonzero"

        hits = np.zeros(N, dtype=int)
        for e in range(N):
            x = np.ones((1, N))
            x[0, e] = -1
            hits[e] = int(cs.line_parities(x)[0].sum())
        assert hits.max() <= 1, f"{Lx}x{Ly}x{Lz} {bc}: a single flip lit >1 detector"
        n_lit = int((hits == 1).sum())
        if bc == "PBC":
            assert n_lit == N, f"{Lx}x{Ly}x{Lz} PBC: not every flip lit exactly one detector"
            print(f"  ok  {Lx}x{Ly}x{Lz} {bc}: all-up clean; every single flip lights "
                  f"exactly one of {cs.Phi.shape[0]} detectors")
        else:
            print(f"  ok  {Lx}x{Ly}x{Lz} {bc}: all-up clean; {n_lit}/{N} single flips light "
                  f"exactly one of {cs.Phi.shape[0]} detectors "
                  f"({N - n_lit} zero-syndrome boundary flips)")


def test_timing():
    """(5) throughput: 2^12 configs at L=2 OBC, 20000 random configs at L=3 OBC."""
    geom2 = ThreeD_ToricCodeGeometry(2, 2, 2, "OBC")
    cs2 = CupSign(geom2)
    N2 = geom2.N
    all_x = 1.0 - 2.0 * ((np.arange(1 << N2)[:, None] >> np.arange(N2)) & 1).astype(np.float64)
    t0 = time.perf_counter()
    cs2.sign(all_x)
    dt2 = time.perf_counter() - t0
    print(f"  ..  L=2 OBC: {all_x.shape[0]} configs in {dt2 * 1e3:.2f} ms "
          f"({dt2 / all_x.shape[0] * 1e6:.4f} ms/1000 configs)")

    geom3 = ThreeD_ToricCodeGeometry(3, 3, 3, "OBC")
    cs3 = CupSign(geom3)
    rng = np.random.default_rng(2)
    x3 = 1.0 - 2.0 * rng.integers(0, 2, size=(20000, geom3.N)).astype(np.float64)
    t0 = time.perf_counter()
    cs3.sign(x3)
    dt3 = time.perf_counter() - t0
    print(f"  ..  L=3 OBC: {x3.shape[0]} configs in {dt3 * 1e3:.2f} ms "
          f"({dt3 / x3.shape[0] * 1e6:.4f} ms/1000 configs)")


if __name__ == "__main__":
    t_start = time.perf_counter()
    test_orbit_sign_exact()
    test_star_invariance_offsupport()
    test_anaC_agreement_l2obc()
    test_line_parities()
    test_timing()
    print(f"ALL SIGN_GEOMETRY TESTS PASSED  ({time.perf_counter() - t_start:.1f} s total)")
