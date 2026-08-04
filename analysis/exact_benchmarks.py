"""
analysis/exact_benchmarks.py
─────────────────────────────────────────────────────────────────────────────
Exact and analytic energy benchmarks for the 3D toric code in a field — the
only independent references that exist for our L>=4 NQS runs.

There is no published QMC, tensor-network, or high-order-series energy for the
3D toric code in a field (the sole 3D reference, Reiss & Schmidt
arXiv:1902.03908, is exact dualities + 2nd-order pCUT + a variational bound,
with no lattice numerics). What *is* available, and what this module encodes:

  1. `E_lowfield`     — the O(h^2) series about the stabilizer ground state.
  2. `E_highfield_x/z`— the O(1/h) series about the polarized states.
  3. `crossing_hc`    — where 1 and 2 cross => a 1st-order transition estimate.
  4. `REFERENCE`      — exactly-known critical fields from the dualities.
  5. `duality_hz`     — an exact energy identity along the pure h_z axis.

All series are **boundary-aware**: they are built from this repo's own
`vertex_all` / `plaq_all`, so they apply to the exact OBC lattice we simulate
rather than to an idealized bulk. That matters — the h_x^2 coefficient at
L=4 OBC is 25.5, where a naive bulk formula gives 18.0.

Normalization is ours throughout: H = -J sum A_v - J sum B_p - sum_a h_a sum_i
sigma^a_i, with J = 1. NOTE the Vidal/Dusuel/Schmidt/Reiss papers use J = 1/2,
so **every field of theirs is half of ours**; `REFERENCE` stores our values.

Pure numpy / scipy — no NQS, no sampling. Self-check:
    .venv/bin/python analysis/exact_benchmarks.py
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

# Repo root on sys.path so this imports the same way from analysis/ (notebooks,
# whose cwd is analysis/) and from the repo root (CLI, other modules).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Three_TC.model.geometry import ThreeD_ToricCodeGeometry  # noqa: E402


# ---------------------------------------------------------------------------
# Stabilizer bookkeeping
# ---------------------------------------------------------------------------
class Counts:
    """Boundary-aware stabilizer counts and the series coefficients built on them.

    Attributes
    ----------
    N, n_Av, n_Bp : int
        Qubits, vertex stars, plaquettes.
    E0 : float
        Exact unperturbed energy, -(n_Av + n_Bp).
    c_z2, c_x2, c_y2 : float
        Low-field O(h^2) coefficients (positive; the series subtracts them).
    c_star : float
        High-field h_z coefficient, sum_v 1/(2 n_star(v)).
    c_z4 : float
        **Exact** 4th-order pure-h_z coefficient (derived; see `E_lowfield`).
    N_adj : int
        Unordered edge pairs sharing a vertex, sum_v C(n_star(v), 2).
    n_v, n_p, n_star : ndarray
        Per-edge star/plaquette incidences and per-vertex star sizes.
    """

    def __init__(self, L: int, bc: str = "OBC"):
        g = ThreeD_ToricCodeGeometry(L, L, L, bc=bc)
        self.L, self.bc, self.N = L, bc, g.N
        self.n_Av, self.n_Bp = len(g.vertex_all), len(g.plaq_all)

        # `vertex_all` pads short boundary stars with -1 sentinels (96 of them at
        # L=4 OBC). Failing to strip them silently corrupts one edge's counter.
        self.n_v = np.zeros(g.N)
        self.n_p = np.zeros(g.N)
        for star in g.vertex_all:
            for e in star:
                if e != -1:
                    self.n_v[e] += 1
        for plaq in g.plaq_all:
            for e in plaq:
                if e != -1:
                    self.n_p[e] += 1
        self.n_star = np.array([sum(1 for e in s if e != -1) for s in g.vertex_all])

        if np.any(self.n_v == 0) or np.any(self.n_p == 0):
            raise ValueError(f"L={L} {bc}: some edge belongs to no star/plaquette")

        self.E0 = float(-(self.n_Av + self.n_Bp))
        # Denominators = 2J x (number of stabilizers the field operator violates).
        self.c_z2 = float(np.sum(1.0 / (2.0 * self.n_v)))
        self.c_x2 = float(np.sum(1.0 / (2.0 * self.n_p)))
        self.c_y2 = float(np.sum(1.0 / (2.0 * (self.n_v + self.n_p))))
        self.c_star = float(np.sum(1.0 / (2.0 * self.n_star)))
        # 4th-order pure-h_z coefficient (see E_lowfield for the derivation).
        self.N_adj = int(sum(k * (k - 1) // 2 for k in self.n_star))
        self.c_z4 = (5.0 / 16.0) * self.n_Bp + self.N_adj / 32.0 - self.N / 64.0

    def __repr__(self) -> str:
        return (f"Counts(L={self.L}, {self.bc}, N={self.N}, E0={self.E0:.0f}, "
                f"c_z2={self.c_z2:g}, c_x2={self.c_x2:g}, c_y2={self.c_y2:g})")


@lru_cache(maxsize=None)
def counts(L: int, bc: str = "OBC") -> Counts:
    """Cached `Counts` — geometry construction costs a few seconds at L=7."""
    return Counts(L, bc)


# ---------------------------------------------------------------------------
# Low-field series: exact through O(h^2)
# ---------------------------------------------------------------------------
def E_lowfield(L: int, bc: str = "OBC", hx: float = 0.0, hy: float = 0.0,
               hz: float = 0.0, order: int = 2) -> float:
    """Ground-state energy about the stabilizer state, exact through O(h^order).

    order=2 (all three field directions):

        E = -(n_Av + n_Bp) - sum_i [ hz^2/(2 n_v) + hx^2/(2 n_p)
                                     + hy^2/(2(n_v + n_p)) ] + O(h^4)

    Second-order PT collapses to counting because each edge contributes exactly
    *one* orthogonal excited state (sigma^z_i|0> flips the n_v(i) stars containing
    i, and nothing else), so there are no cross terms — neither between edges nor
    between field directions. Cross terms like hx^2 hz^2 first appear at O(h^4).

    order=4 (**pure h_z only** — hx = hy = 0 required):

        E = -(n_Av + n_Bp) - c_z2 hz^2 - c_z4 hz^4 + O(hz^6),
        c_z4 = (5/16) n_Bp + N_adj/32 - N/64

    Derivation. In the sigma^z sector every state is labelled by its syndrome S
    (the set of flipped stars) with E = E0 + 2|S|, and every matrix element of
    sigma^z_i is exactly +1 (two edge sets with the same boundary differ by a
    contractible loop = a product of B_p, eigenvalue +1). So Rayleigh-Schroedinger
    at 4th order is pure combinatorics over ordered 4-tuples of edges whose net
    operator returns to the vacuum. Two families contribute: (a) all four edges
    distinct, which forces a closed 4-cycle — and in a cubic lattice every
    4-cycle is a plaquette — giving 5/2 per plaquette over its 24 orderings;
    (b) edges in coincident pairs {a,a,b,b}, giving 1/|d(a,b)| per pair. Family
    (b) sums to O(N^2), which cancels *exactly* against the
    -E^(2) sum_k |V_0k|^2/Delta_k^2 renormalization term, leaving the extensive
    result above. Verified two ways: against a brute-force sum over all 4-tuples,
    and against exact ED at L=2 OBC (residual then scales as hz^6).

    OBC only: the argument needs a simply-connected lattice (so the syndrome
    determines the state) and a non-degenerate ground state. Both fail under PBC,
    where the eightfold degeneracy would require degenerate PT.
    """
    c = counts(L, bc)
    if order == 2:
        return c.E0 - c.c_z2 * hz ** 2 - c.c_x2 * hx ** 2 - c.c_y2 * hy ** 2
    if order == 4:
        if hx != 0.0 or hy != 0.0:
            raise ValueError("order=4 is derived for pure h_z only (hx=hy=0); "
                             "the hx^4 and hx^2 hz^2 coefficients are NOT known")
        if bc.upper() != "OBC":
            raise ValueError("order=4 derivation assumes OBC (simply connected, "
                             "non-degenerate ground state)")
        return c.E0 - c.c_z2 * hz ** 2 - c.c_z4 * hz ** 4
    raise ValueError(f"order must be 2 or 4, got {order}")


# ---------------------------------------------------------------------------
# High-field series: exact through O(1/h)
# ---------------------------------------------------------------------------
def E_highfield_x(L: int, bc: str = "OBC", hx: float = 1.0) -> float:
    """Energy about the x-polarized state |+>^N, exact through O(1/hx).

        E = -hx N - n_Av - n_Bp/(8 hx) + O(hx^-3)

    In |+>^N every A_v = +1 exactly (a product of X eigenvalues), so the stars
    contribute -n_Av and the only off-diagonal perturbation is B_p. Exciting one
    B_p flips 4 spins; each of the plaquette's 4 vertices carries exactly *two*
    of its edges, so every A_v is left unchanged and the energy denominator is
    pure Zeeman cost, 2*hx*4 = 8 hx. Hence no boundary dependence here: every
    plaquette we keep has exactly 4 edges.
    """
    c = counts(L, bc)
    return -hx * c.N - c.n_Av - c.n_Bp / (8.0 * hx)


def E_highfield_z(L: int, bc: str = "OBC", hz: float = 1.0) -> float:
    """Energy about the z-polarized state |up>^N, exact through O(1/hz).

        E = -hz N - n_Bp - sum_v 1/(2 hz n_star(v)) + O(hz^-3)

    Mirror of `E_highfield_x`: now every B_p = +1, and exciting one A_v flips
    n_star(v) spins (6 in the bulk, 3/4/5/6 on an open boundary — hence the
    boundary dependence) while leaving every B_p unchanged, since a plaquette
    through v uses exactly two of v's incident edges.
    """
    c = counts(L, bc)
    return -hz * c.N - c.n_Bp - c.c_star / hz


# ---------------------------------------------------------------------------
# Energy-crossing estimate of a first-order transition
# ---------------------------------------------------------------------------
def crossing_hc(L: int, bc: str = "OBC", sector: str = "x",
                bracket: Tuple[float, float] = (0.05, 5.0)) -> Optional[float]:
    """Field where the low- and high-field series cross.

    For a *first-order* transition the two expansions cross at h_c (the level
    crossing itself); this is the standard pCUT locator, validated in 3D against
    QMC for fracton models (arXiv:1911.13117: pCUT 0.9196 vs QMC 0.922).

    Returns None if the two branches do not cross inside `bracket` (which is the
    honest answer for a second-order transition, where there is no crossing to
    find). `sector` is "x" (h_x cut, h_z=0) or "z" (h_z cut, h_x=0).
    """
    if sector == "x":
        def diff(h):
            return E_lowfield(L, bc, hx=h) - E_highfield_x(L, bc, hx=h)
    elif sector == "z":
        def diff(h):
            return E_lowfield(L, bc, hz=h) - E_highfield_z(L, bc, hz=h)
    else:
        raise ValueError(f"sector must be 'x' or 'z', got {sector!r}")

    lo, hi = bracket
    hs = np.geomspace(lo, hi, 400)
    vals = np.array([diff(h) for h in hs])

    # Select by the DIRECTION of the crossing, not its position. Both truncated
    # branches misbehave outside their domain and each contributes a spurious
    # *downward* root: the high-field -n_Bp/(8h) term diverges as h->0 (root at
    # 0.157 for L=4) and the low-field -c_x2 h^2 parabola eventually dives back
    # under the high-field line (root at 4.785). The physical level crossing is
    # the unique *upward* zero — where the low-field branch stops being the
    # ground state as h increases.
    up = [k for k in range(len(hs) - 1) if vals[k] < 0.0 <= vals[k + 1]]
    if not up:
        return None
    return float(brentq(diff, hs[up[0]], hs[up[0] + 1]))


# ---------------------------------------------------------------------------
# Exact duality along the pure h_z axis
# ---------------------------------------------------------------------------
def _gf2_rank(rows: list, ncols: int) -> int:
    """GF(2) rank by bitmask elimination (same idiom as `_gf2_solve`)."""
    rows = list(rows)
    r = 0
    for col in range(ncols):
        sel = next((k for k in range(r, len(rows)) if (rows[k] >> col) & 1), None)
        if sel is None:
            continue
        rows[r], rows[sel] = rows[sel], rows[r]
        for k in range(len(rows)):
            if k != r and (rows[k] >> col) & 1:
                rows[k] ^= rows[r]
        r += 1
    return r


def duality_consistency(L: int, bc: str = "OBC") -> Dict[str, int]:
    """Verify the h_z-axis duality bookkeeping on our actual geometry.

    In the flux-free sector sigma^z_e = s_v s_v', so A_v -> tau^z_v,
    sigma^z_e -> tau^x_v tau^x_v', and every B_p -> identity. That map is
    consistent iff the number of *independent* plaquette constraints leaves
    exactly the dual spin degrees of freedom (up to a global flip):

        N - rank_GF2(plaquettes) == n_Av - 1 + n_homology

    where `n_homology` is 0 under OBC and **3** under PBC — the three independent
    non-contractible 1-cycles of the 3-torus, i.e. log2 of the eightfold
    topological ground-state degeneracy. Those extra sectors are exactly why the
    clean duality-to-TFIM statement is an OBC statement; under PBC the dual model
    additionally carries winding sectors.

    `rank` is computed directly from `plaq_all`, so this is an assumption-free
    numerical check rather than a cube-counting shortcut.
    """
    g = ThreeD_ToricCodeGeometry(L, L, L, bc=bc)
    rows = []
    for plaq in g.plaq_all:
        m = 0
        for e in plaq:
            if e != -1:
                m |= 1 << int(e)
        rows.append(m)
    rank = _gf2_rank(rows, g.N)
    c = counts(L, bc)
    n_homology = 3 if bc.upper() == "PBC" else 0
    expected = c.n_Av - 1 + n_homology
    return {"N": c.N, "n_Av": c.n_Av, "n_Bp": c.n_Bp, "rank": rank,
            "free": c.N - rank, "n_homology": n_homology, "expected": expected,
            "ok": int(c.N - rank == expected)}


def duality_hz(L: int, bc: str, E_tfim: float) -> float:
    """Exact TC energy on the pure h_z axis from the dual 3D TFIM energy.

        E_TC(hz) = -n_Bp + E_TFIM(Gamma=1, J=hz ; open L^3 cubic lattice)

    `E_tfim` must be the ground energy of  H = -sum_v tau^z_v - hz sum_<vv'>
    tau^x_v tau^x_v'  on the L^3 cubic lattice whose sites are our vertices and
    whose bonds are our edges. Valid at *every* field, not just perturbatively —
    this is the route to a non-perturbative reference along h_z without ever
    simulating a toric code (the dual model is 2-body and sign-problem-free).
    """
    return -counts(L, bc).n_Bp + E_tfim


# ---------------------------------------------------------------------------
# Exactly-known reference values (our normalization, J = 1)
# ---------------------------------------------------------------------------
REFERENCE: Dict[str, Dict] = {
    "3d_hz": dict(
        value=0.193869, order="second", universality="(3+1)D Ising*",
        how="exact duality to the 3D TFIM; (Gamma/J)_c = 5.158136",
        cite="Blote & Deng PRE 66 066110 (2002); Reiss & Schmidt arXiv:1902.03908"),
    "3d_hx": dict(
        value=1.0, order="first", universality="self-dual level crossing",
        how="exact self-duality to 4D Wegner Z2 gauge theory; h_x^c = J exactly",
        cite="Reiss & Schmidt arXiv:1902.03908 Eq. 25; Wegner J.Math.Phys. 12 2259"),
    "3d_hy": dict(
        value=1.23, order="first", universality="unknown (no duality exists)",
        how="VARIATIONAL ONLY — no exact result, no numerics published",
        cite="Reiss & Schmidt arXiv:1902.03908 sec 4.1.3"),
    "2d_hz": dict(
        value=0.328474, order="second", universality="3D Ising*",
        how="CT-QMC; inherited from the 2D TFIM critical point by duality",
        cite="Wu, Deng, Prokof'ev arXiv:1201.6409, PRB 85 195104"),
    "2d_hy": dict(
        value=1.0, order="first", universality="self-dual (Xu-Moore)",
        how="exact self-duality; e0/N = -1.13651 at h_y = J",
        cite="Vidal, Thomale, Schmidt, Dusuel arXiv:0902.3547, PRB 80 081104(R)"),
}


# ---------------------------------------------------------------------------
# Accuracy certificate
# ---------------------------------------------------------------------------
def raw_deviation(field: np.ndarray, E: np.ndarray, L: int, bc: str = "OBC",
                  cut: str = "hz", fixed: float = 0.0) -> np.ndarray:
    """E_NQS - E_lowfield, elementwise. `cut` is "hz" or "hx"; `fixed` is the
    other (constant) field, so the two-field series is used when it is nonzero."""
    field = np.asarray(field, float)
    if cut == "hz":
        ser = np.array([E_lowfield(L, bc, hx=fixed, hz=h) for h in field])
    elif cut == "hx":
        ser = np.array([E_lowfield(L, bc, hx=h, hz=fixed) for h in field])
    else:
        raise ValueError(f"cut must be 'hz' or 'hx', got {cut!r}")
    return np.asarray(E, float) - ser


def fit_residual(field: np.ndarray, dev: np.ndarray, hmax: float,
                 orders: Tuple[int, ...] = (4, 6)) -> Dict[str, np.ndarray]:
    """Fit  dev(h) ~ sum_k c_k h^k  over h <= hmax and return the residual.

    WARNING — this is a SMOOTHNESS diagnostic, NOT an error bound. The fitted
    c_k are contaminated by the NQS error they are supposed to be separated
    from, so a small residual says only that dev(h) is well described by two
    smooth powers. Do not quote it as an NQS accuracy. See `nqs_error_bound`
    for the argument and `c4_extensivity` for how to detect the contamination.
    """
    field, dev = np.asarray(field, float), np.asarray(dev, float)
    m = field <= hmax
    if m.sum() <= len(orders):
        raise ValueError(f"need > {len(orders)} points with h <= {hmax}, got {m.sum()}")
    A = np.vstack([field[m] ** k for k in orders]).T
    coef, *_ = np.linalg.lstsq(A, dev[m], rcond=None)
    resid = dev[m] - A @ coef
    return {"h": field[m], "coef": coef, "orders": np.array(orders),
            "residual": resid, "fit": A @ coef}


def nqs_error_bound(field: np.ndarray, E: np.ndarray, L: int, bc: str = "OBC",
                    cut: str = "hz", fixed: float = 0.0,
                    hmax: Optional[float] = None,
                    E_err: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Compare NQS energies to the exact low-field series. READ THIS FIRST.

    Write  dev(h) = E_NQS - E_series2 = T(h) + eps(h),  where T = c4 h^4 + ...
    is the (unknown, unpublished in 3D) truncation remainder and eps >= 0 is the
    NQS variational error. Only one direction yields a real conclusion:

      * dev > 0  =>  eps = dev - T > dev  whenever T < 0. A genuine LOWER BOUND
        on the NQS error, with nothing fitted. This is how the L=6/L=7 small-hz
        runs were caught: it is a sharper test than the E < E0(h=0) anchor bound
        or Vscore, both of which those runs pass.
      * dev < 0  =>  only |T| >= |dev|. That bounds the unknown 4th-order term,
        NOT the NQS error. **No accuracy claim follows.**

    So this function does NOT return an NQS accuracy for well-behaved points, and
    deliberately so — an earlier version fitted c4, c6 to dev and quoted the
    residual, which is circular (the fit absorbs the very error it should
    isolate; `c4_extensivity` demonstrates it does). `fit` is still returned as a
    smoothness diagnostic only.

    To get a non-circular accuracy you need c4 from an independent source: derive
    4th-order perturbation theory (no NQS input — tractable here because every
    intermediate state is a stabilizer state with a known energy), or use the
    zero-variance extrapolation, or L=2 ED.
    """
    field = np.asarray(field, float)
    dev = raw_deviation(field, E, L, bc=bc, cut=cut, fixed=fixed)
    out: Dict[str, object] = {"h": field, "E": np.asarray(E, float), "dev": dev,
                              "rel_raw": np.abs(dev) / np.abs(np.asarray(E, float))}
    if hmax is None:
        # Default fit window: up to this cut's critical field. Tighter would be
        # safer physics but our hz sweeps only start at 0.1, so 0.6*h_c leaves
        # too few points to fit two coefficients — pass `hmax` explicitly to
        # trade window size against point count.
        hmax = REFERENCE["3d_hz" if cut == "hz" else "3d_hx"]["value"]
    try:
        fr = fit_residual(field, dev, hmax)
        out["fit"] = fr
        # Smoothness only — NOT an accuracy. Named to make misuse obvious.
        out["rel_smoothness"] = np.abs(fr["residual"]) / np.abs(
            np.asarray(E, float)[field <= hmax])
    except ValueError as exc:
        out["fit"] = None
        out["fit_error"] = str(exc)

    m = field <= hmax
    n_above = int((dev[m] > 0).sum())
    out["hmax"] = hmax
    out["n_above"] = n_above
    out["err_lower_bound"] = float(np.max(dev[m] / np.abs(np.asarray(E, float)[m]))
                                   ) if n_above else 0.0
    out["verdict"] = ("UNDER-CONVERGED" if n_above else "no accuracy claim possible")

    # On the PURE h_z cut the O(h^4) coefficient is now known exactly, so the
    # deviation is  eps + O(hz^6)  -- an actual accuracy, not a bound. eps >= 0 is
    # then a hard test: a negative dev4 means the neglected c6 hz^6 term has taken
    # over (compare |dev4| against ~0.45*N*hz^6, the rough c6 scale) or the run is
    # broken. Only defined for cut="hz", fixed=0, OBC.
    if cut == "hz" and fixed == 0.0 and bc.upper() == "OBC":
        ser4 = np.array([E_lowfield(L, bc, hz=h, order=4) for h in field])
        dev4 = np.asarray(E, float) - ser4
        out["dev4"] = dev4
        out["rel_err4"] = dev4 / np.abs(np.asarray(E, float))
        out["c6_scale"] = 0.45 * counts(L, bc).N * field ** 6   # magnitude guide

        # E_NQS is a SINGLE vs.expect(Ham) on the final state (validation.py:169),
        # not an average over optimization steps, so it carries an MC error of the
        # mean. dev4 means nothing until compared against it: at n_samples=8192 the
        # L=4/L=5 deviations sit at 0.3-1.2 sigma (pure noise) while L=6/L=7 are at
        # 4-13 sigma (real). Pass `E_err` (the `E_spread` field of the aggregate
        # JSONs, which is NetKet's autocorrelation-corrected error_of_mean).
        if E_err is not None:
            ee = np.asarray(E_err, float)
            out["E_err"] = ee
            out["sigma"] = dev4 / ee
            # 2-sigma one-sided upper bound on the variational error, and the useful
            # statement when dev4 is unresolved. Since eps = dev4_true + c6*hz^6 and
            # dev4_obs = dev4_true + noise, a CONSERVATIVE bound must ADD the c6
            # estimate, not drop it:  eps <= dev4 + 2*E_err + c6*hz^6.
            out["eps_upper_2sig"] = ((dev4 + 2.0 * ee + out["c6_scale"])
                                     / np.abs(np.asarray(E, float)))
            # Only trust the bound where the (estimated) truncation sits below the MC
            # noise; past that the crude 0.45*N c6 guess dominates and the bound is
            # meaningless (it can even go negative).
            out["bound_valid"] = out["c6_scale"] < ee
            out["resolved"] = np.abs(out["sigma"]) > 3.0
    return out


def c4_extensivity(fitted_c4: Dict[int, float], bc: str = "OBC") -> Dict[int, float]:
    """c4/N per L — the check that reveals whether a fitted c4 is physical.

    The exact c2 is exactly extensive (c2/N = 1/4 at every L), so a genuine c4
    must be extensive too, up to mild boundary drift. On the hz cut the fitted
    values give c4/N = -0.196, -0.305, +1.74, +9.15 for L=4,5,6,7 — spanning two
    orders of magnitude and changing sign. That is proof the fit is absorbing NQS
    error rather than recovering the series, and is why `nqs_error_bound` no
    longer quotes a residual-based accuracy.
    """
    return {L: c4 / counts(L, bc).N for L, c4 in fitted_c4.items()}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    fails = []

    def chk(name, got, want, tol):
        ok = abs(got - want) <= tol
        print(f"  {'ok ' if ok else 'FAIL'} {name:52s} got {got:12.6f}  want {want:12.6f}")
        if not ok:
            fails.append(name)

    print("1. series coefficients, OBC (plan table)")
    for L, E0, cz, cx, cy in [(4, -172, 36.00, 25.50, 14.70),
                              (5, -365, 75.00, 49.50, 29.40),
                              (6, -666, 135.00, 85.00, 51.50),
                              (7, -1099, 220.50, 134.25, 82.50)]:
        c = counts(L, "OBC")
        chk(f"L={L} E0", c.E0, E0, 0)
        chk(f"L={L} c_z2", c.c_z2, cz, 1e-9)
        chk(f"L={L} c_x2", c.c_x2, cx, 1e-9)
        chk(f"L={L} c_y2", c.c_y2, cy, 1e-9)

    print("2. bulk PBC per-spin coefficients  (-4/3, 1/4, 1/8, 1/12)")
    c = counts(4, "PBC")
    chk("PBC e0/N", c.E0 / c.N, -4.0 / 3.0, 1e-12)
    chk("PBC c_z2/N", c.c_z2 / c.N, 0.25, 1e-12)
    chk("PBC c_x2/N", c.c_x2 / c.N, 0.125, 1e-12)
    chk("PBC c_y2/N", c.c_y2 / c.N, 1.0 / 12.0, 1e-12)

    print("3. h=0 anchors match the stored anchor_E0")
    chk("L=2 OBC", E_lowfield(2, "OBC"), -14, 0)
    chk("L=2 PBC", E_lowfield(2, "PBC"), -32, 0)
    chk("L=7 OBC", E_lowfield(7, "OBC"), -1099, 0)

    print("4. this session's validated deviations")
    chk("L=4 hz=0.1  E_lowfield", E_lowfield(4, "OBC", hz=0.1), -172.36, 1e-9)
    chk("L=4 hx=1.3  E_highfield_x", E_highfield_x(4, "OBC", hx=1.3), -261.5846, 1e-3)

    print("4b. exact 4th-order pure-hz coefficient c_z4")
    chk("L=2 OBC c_z4 (vs brute-force 4-tuple sum)", counts(2, "OBC").c_z4, 2.4375, 1e-12)
    chk("L=4 OBC c_z4", counts(4, "OBC").c_z4, 48.0, 1e-12)
    # L=2 OBC exact ED gives E = -14 - 3 hz^2 - 2.4375 hz^4 + O(hz^6)
    chk("L=2 OBC E(hz=0.01) order 4", E_lowfield(2, "OBC", hz=0.01, order=4),
        -14.00030002437500, 1e-12)
    # OBC c_z4/N must extrapolate in 1/L to the independently computed PBC bulk 29/64
    Ls = [4, 5, 6, 7]
    y = np.array([counts(L, "OBC").c_z4 / counts(L, "OBC").N for L in Ls])
    x = 1.0 / np.array(Ls, float)
    a, b = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)[0]
    chk("c_z4/N extrapolated -> PBC bulk 29/64", a, 29.0 / 64.0, 3e-3)
    chk("PBC bulk c_z4/N", counts(4, "PBC").c_z4 / counts(4, "PBC").N, 29.0 / 64.0, 1e-12)

    print("5. energy-crossing h_c (h_x sector)")
    for L, want in [(4, 0.7051), (5, 0.7665), (6, 0.8066), (7, 0.8349)]:
        chk(f"L={L} crossing", crossing_hc(L, "OBC", "x"), want, 1e-3)

    print("6. duality bookkeeping  N - rank(plaq) == n_Av - 1")
    for bc in ("OBC", "PBC"):
        for L in (2, 3, 4, 5):
            d = duality_consistency(L, bc)
            ok = bool(d["ok"])
            print(f"  {'ok ' if ok else 'FAIL'} L={L} {bc}: N={d['N']} rank={d['rank']} "
                  f"free={d['free']} expected={d['expected']}")
            if not ok:
                fails.append(f"duality L={L} {bc}")

    print()
    if fails:
        print(f"FAILED: {len(fails)} check(s): {fails}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
