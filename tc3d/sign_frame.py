"""Sign-framed Hamiltonian (formulation B): conjugate H instead of signing psi.

For a diagonal sign S|s> = (-1)^{sign(s)}|s>, training a POSITIVE trunk A_theta on
H~ = S H S is the SAME optimization as training psi = (-1)^{sign} A_theta on H: S
carries no parameters, so |psi|^2 = A^2 (identical chains), every local energy and
every O_k = dlog A/dtheta coincide. Usage:

    python -m tc3d.train --L 2 --bc OBC --model fermionic --arch ToricCNN_gridinv \
        --sign_frame anaC --flux_penalty 6.0        # real trunk, no phase head
    # programmatic:  Ham = SignFramedOperator(Ham, anaC_sign(geo))

`--sign_frame anaC` is the analytic fTC h=0 sign (parameter-free, host-numpy
cost guard at N_p > 64, i.e. refused from L=4 OBC up -- see
token_quadratic_sign); `table` loads a 2^N lookup table (`--sign_table x.npy`,
N <= 24). Framing and the in-network phase head are mutually exclusive -- both
would apply the sign twice. Also excludes --dual_basis (the fermionic
decoration is not self-dual) and h_y != 0 (a genuine complex phase, not a +-1
sign) -- see builders.with_defaults.

Caveats:
  * Any OFF-DIAGONAL observable must be framed too (`frame_eval_ops`, used by
    train.py's final eval and the analysis-side eval_snapshots/bank_point/
    eval_ckpt entry points). Diagonal ones (sigma^z, Z-strings) are sign-blind.
  * psi-LEVEL estimators (Renyi-S2 swap kernel, fm.py's FM ratios/membranes)
    cannot be fixed by operator framing at all -- they act on psi = S*A
    directly, not through <psi|O|psi>. validation.topological_observables
    refuses to run on a sign_frame checkpoint (train.py's final eval and
    eval_ckpt.py --topological both go through it) instead of silently
    scoring the positive trunk A. fm.py/renyi.py's standalone sweep CLIs
    (fm_sweep/renyi_sweep, load_vstate/iter_matching_checkpoints) do NOT yet
    carry the same guard -- fm.py/renyi.py were out of scope for this fix;
    do not point either CLI at a sign_frame checkpoint directory.
  * `--exact` full-Hilbert-space evals (analysis/scripts/eval_snapshots.py)
    must sign the amplitude vector psi = S*A themselves before scoring E0/
    fidelity/Vscore against a bare H -- the network only ever returns A.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import netket as nk

_BLOCK = 1 << 16          # rows per host-side chunk in the token heads
_MAX_ANAC_NP = 64         # anaC/token-quadratic cost guard (see token_quadratic_sign)
_MAX_TABLE_N = 24         # table-head cost guard (see table_sign)


class SignFramedOperator(nk.operator.DiscreteOperator):
    """S @ op @ S for a diagonal sign function `sign_fn: (..., N) -> +-1`.

    Delegates everything to the wrapped operator and multiplies matrix elements
    by s(x) * s(x'): the connected configurations, their count, the dtype and
    hermiticity are all untouched, so `MCState.expect` / VMC / SR treat this
    exactly like the operator it wraps. Host/numpy path (the one NetKet already
    uses for numba `PauliStrings`); the sampler never sees the head.
    """

    def __init__(self, op, sign_fn: Callable[[np.ndarray], np.ndarray]):
        super().__init__(op.hilbert)
        self._op = op
        self._sign_fn = sign_fn

    @property
    def dtype(self):
        return self._op.dtype

    @property
    def is_hermitian(self):
        return self._op.is_hermitian

    @property
    def max_conn_size(self):
        return self._op.max_conn_size

    @property
    def base(self):
        """The unframed operator (for cross-checks / observables)."""
        return self._op

    @property
    def sign_fn(self):
        return self._sign_fn

    def n_conn(self, x, out=None):
        return self._op.n_conn(x, out)          # +-1 rescaling never kills a term

    def get_conn_padded(self, x):
        xp, mels = self._op.get_conn_padded(x)
        s = self._sign_fn(np.asarray(x))                    # (...,)
        sp = self._sign_fn(np.asarray(xp))                  # (..., n_conn)
        return xp, mels * s[..., None] * sp

    def get_conn_flattened(self, x, sections, pad=False):
        xp, mels = self._op.get_conn_flattened(x, sections, pad)
        s = self._sign_fn(np.asarray(x))                    # (B,)
        counts = np.diff(np.concatenate(([0], np.asarray(sections))))
        return xp, mels * np.repeat(s, counts) * self._sign_fn(np.asarray(xp))

    def __repr__(self):
        return f"SignFramedOperator(S @ {self._op!r} @ S)"


# =============================================================================
# Sign heads: pure numpy functions of +-1 spin configurations
# =============================================================================

def token_quadratic_sign(theta_lin, theta_quad, plaq_masks):
    """The network's frozen phase head, as a parameter-free sign function.

    `ToricCNN_gridinv` adds `1j * (t . theta_lin + t^T theta_quad t)` to log psi,
    with tokens `t_p = prod_{i in plaq_masks[p]} sigma_i` in {+-1} (raw spins, in
    `geo.plaq_all` == `fermionic_plaquettes` order). When theta is a valid sign
    form that phase is a multiple of pi, i.e. exactly a +-1 factor -- this returns
    it, computed over the NONZERO theta entries only (the anaC form is sparse; a
    dense N_p^2 contraction is unaffordable at L >= 3).
    """
    idx = np.asarray(plaq_masks, dtype=np.int64)
    if idx.shape[0] > _MAX_ANAC_NP:
        raise ValueError(
            f"token_quadratic_sign: N_p={idx.shape[0]} exceeds the host-numpy "
            f"cost guard ({_MAX_ANAC_NP}) -- this head runs on the host for every "
            "get_conn_padded call (probe_cost.py: L=3 OBC N_p=36 framed vs bare "
            "overhead is already visible; L=4's N_p=108 is refused here). Use the "
            "in-network phase head (--phase_head/--phase_head_frozen) at this size, "
            "or a future device (JAX) sign head.")
    tl, tq = np.asarray(theta_lin), np.asarray(theta_quad)
    for name, a in (("theta_lin", tl), ("theta_quad", tq)):
        if np.iscomplexobj(a) and np.max(np.abs(a.imag)) > 1e-12:
            raise ValueError(f"{name} has an imaginary part: that is an amplitude "
                             "correction, not a sign -- refusing to frame it")
    tl, tq = np.real(tl).astype(np.float64).ravel(), np.real(tq).astype(np.float64)
    li, = np.nonzero(tl)
    wl = tl[li]
    qi, qj = np.nonzero(tq)
    wq = tq[qi, qj]

    def sign_fn(x):
        x = np.asarray(x)
        lead, flat = x.shape[:-1], x.reshape(-1, x.shape[-1])
        out = np.empty(flat.shape[0])
        for a in range(0, flat.shape[0], _BLOCK):
            t = np.prod(flat[a:a + _BLOCK][:, idx], axis=-1)      # (b, N_p) +-1
            phi = t[:, li] @ wl + (t[:, qi] * t[:, qj]) @ wq
            c = np.cos(phi)
            if not np.all(np.abs(c) > 1 - 1e-6):
                raise ValueError("token phase is not a multiple of pi "
                                 f"(min |cos| = {np.abs(c).min():.3e}): theta is "
                                 "not a sign form")
            out[a:a + _BLOCK] = np.where(c >= 0.0, 1.0, -1.0)
        return out.reshape(lead)

    return sign_fn


def table_sign(table, N):
    """Lookup sign function over all 2^N configurations (ED sizes, N <= 24).

    Bit convention = `tc3d.exact_diag`'s basis order: qubit i is bit i and bit 1
    means sigma^z = -1 (spin down), so the all-up state is index 0. NetKet samples
    are +-1 floats, hence bit = (spin < 0). `table[j]` must be +1 or -1.
    """
    if N > _MAX_TABLE_N:
        raise ValueError(f"sign_frame='table' refuses N > {_MAX_TABLE_N} "
                         f"(host lookup is over 2^N configs; got N={N})")
    tab = np.asarray(table, dtype=np.float64).ravel()
    if tab.size != 1 << N:
        raise ValueError(f"sign table has {tab.size} entries, expected 2^{N}")
    if not np.all(np.abs(tab) == 1.0):
        raise ValueError("sign table entries must be exactly +-1")
    pw = 1 << np.arange(N, dtype=np.int64)

    def sign_fn(x):
        x = np.asarray(x)
        lead, flat = x.shape[:-1], x.reshape(-1, x.shape[-1])
        return tab[(flat < 0).astype(np.int64) @ pw].reshape(lead)

    return sign_fn


def sign_table(sign_fn, N):
    """Tabulate `sign_fn` over all 2^N configurations (same bit order as above)."""
    if N > 24:
        raise ValueError(f"refusing to tabulate 2^{N} configurations")
    out = np.empty(1 << N)
    for a in range(0, 1 << N, _BLOCK):
        j = np.arange(a, min(a + _BLOCK, 1 << N), dtype=np.int64)
        out[a:a + len(j)] = sign_fn(1.0 - 2.0 * ((j[:, None] >> np.arange(N)) & 1))
    return out


# =============================================================================
# The analytic fermionic h=0 sign form (anaC)
# =============================================================================

def anaC_theta(stabs):
    """(theta_lin, theta_quad) of the exact fTC h=0 sign, analytically.

    Port of `analysis/scripts/prefit_phase_head.py --analytic_C` (no fitting, no
    checkpoint): the local sign form in APPLICATION variables,
    C_pq = |dp cap xpair_q| mod 2, pulled back to token space through a GF(2)
    right-inverse (full RREF with preimage tracking) of the token-flip map
    M[q,p] = |dq cap xpair_p| mod 2. Exact on the physical (zero-flux) sector at
    any L; off-sector values are irrelevant (that is what --flux_penalty is for).
    """
    from tc3d.fermionic_decoration import _mask

    zxm = [(_mask(z), _mask(x)) for z, x, _ in stabs]
    NP = len(zxm)
    C = np.zeros((NP, NP), dtype=np.int64)
    for p in range(NP):
        for q in range(NP):
            if p != q:
                C[p, q] = bin(zxm[p][0] & zxm[q][1]).count("1") & 1
    assert np.array_equal(C, C.T), "C must be symmetric (commutation)"

    piv: dict = {}                          # pivot bit -> [reduced column, preimage]
    for p in range(NP):
        v = 0
        for q in range(NP):                 # column p of the token-flip map M
            if bin(zxm[q][0] & zxm[p][1]).count("1") & 1:
                v |= 1 << q
        pre = 1 << p
        for c in list(piv):                 # reduce against EVERY pivot (full RREF)
            if (v >> c) & 1:
                v ^= piv[c][0]
                pre ^= piv[c][1]
        if v:
            c = (v & -v).bit_length() - 1
            for k in list(piv):             # back-reduce
                if (piv[k][0] >> c) & 1:
                    piv[k][0] ^= v
                    piv[k][1] ^= pre
            piv[c] = [v, pre]
    pivots = sorted(piv)
    d = len(pivots)
    P = np.array([[(piv[c][1] >> p) & 1 for p in range(NP)] for c in pivots],
                 dtype=np.int64).reshape(d, NP)

    lp = np.einsum("ip,pq,iq->i", P, np.triu(C, 1), P) & 1
    B = (P @ C @ P.T) & 1
    l = np.zeros(NP)
    Q = np.zeros((NP, NP))
    for i in range(d):
        l[pivots[i]] = lp[i]
        for j in range(i + 1, d):
            Q[pivots[i], pivots[j]] = B[i, j]

    # bits b_p = (1 - t_p)/2 -> +-1 tokens; the constant rides on t_0 t_0 == 1
    theta_lin = -np.pi * l / 2 - np.pi / 4 * (Q.sum(1) + Q.sum(0))
    theta_quad = np.pi * Q / 4
    theta_quad[0, 0] += np.pi * (l.sum() / 2 + Q.sum() / 4)
    return theta_lin, theta_quad


def anaC_sign(geo, J: float = 1.0):
    """The analytic fermionic h=0 sign function for this geometry."""
    from tc3d.fermionic_decoration import fermionic_plaquettes

    stabs = fermionic_plaquettes(geo, J=J)
    return token_quadratic_sign(*anaC_theta(stabs), geo.plaq_all)


def _repo_root():
    """tc3d/sign_frame.py -> repo root (one level up from the tc3d/ package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_sign_table_path(path):
    """A relative `sign_table` path is resolved against the CURRENT directory
    first (analysis/scripts convention: run from repo root), then against the
    repo root itself -- so a run JSON's stored path still loads when the caller
    (e.g. eval_snapshots.py) runs from a different cwd than the training job did."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(_repo_root(), path)
    return candidate if os.path.exists(candidate) else path


def build_sign_fn(config, geo):
    """config -> sign function (None when `sign_frame` is 'none'). Used by builders."""
    kind = config.get("sign_frame", "none") or "none"
    if kind == "none":
        return None
    if kind == "anaC":
        if config.get("model", "bosonic") != "fermionic":
            raise ValueError("sign_frame='anaC' is the fermionic h=0 sign form; "
                             f"got model={config.get('model')!r}")
        return anaC_sign(geo, J=float(config.get("J", 1.0)))
    if kind == "table":
        path = config.get("sign_table")
        if not path:
            raise ValueError("sign_frame='table' needs --sign_table PATH.npy")
        return table_sign(np.load(_resolve_sign_table_path(path)), geo.N)
    raise ValueError(f"unknown sign_frame {kind!r} (none|anaC|table)")


def frame_eval_ops(mean_ops, config, geo):
    """Frame a `validation._mean_operators` tuple for `sign_frame` runs.

    Under formulation B the state is psi = S A, so <psi|O|psi> = <A|S O S|A>:
    every OFF-DIAGONAL observable (A_v, B~_p, M_x) must be conjugated exactly
    like H. Diagonal ones (M_z) satisfy S O S == O, so framing them is a safe
    no-op. Returns `mean_ops` unchanged when `sign_frame` is 'none' -- the single
    call site shared by train.py's final eval and the analysis-side entry points
    (eval_snapshots.py, bank_point.py, eval_ckpt.py), which used to build their
    mean_ops unframed (2026-09 audit finding).
    """
    if (config.get("sign_frame", "none") or "none") == "none":
        return mean_ops
    sfn = build_sign_fn(config, geo)
    return tuple(SignFramedOperator(o, sfn) for o in mean_ops)
