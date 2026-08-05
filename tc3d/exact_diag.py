"""
Matrix-free Hamiltonian and Pauli-string expectations for the perturbed
2D toric / surface code in the computational (sigma^z) basis.

Lifted from tests/colab_exact_diag.py and adapted: geometry-agnostic
(consumes any object with .N, .vertex_all, .plaq_all), no Y-string handling
(this project keeps h_y = 0), defaults to float64.

H = -J Sum_v A_v  -  J Sum_p B_p  -  h_x Sum_i sigma^x_i  -  h_z Sum_i sigma^z_i
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Tuple

import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator


@njit(inline='always', cache=True)
def _bit_parity64(x):
    """Parity (XOR-reduction) of bits of a 64-bit int.  Returns 0 or 1."""
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


@njit(parallel=True, fastmath=True, cache=True)
def _matvec_jit(psi, diag, x_masks, x_coefs, out):
    """Parallel toric / surface-code matvec.

    Computes  out[i] = diag[i] * psi[i] + sum_k x_coefs[k] * psi[i XOR x_masks[k]].
    All Pauli-X strings (vertex stars + h_x single-site terms) are encoded as a
    (mask, coefficient) pair; the JIT walks the dim-long output in parallel.
    """
    dim = psi.shape[0]
    n_masks = x_masks.shape[0]
    for i in prange(dim):
        s = diag[i] * psi[i]
        for k in range(n_masks):
            s += x_coefs[k] * psi[i ^ x_masks[k]]
        out[i] = s


@njit(parallel=True, fastmath=True, cache=True)
def _matvec_jit_xz_add(psi, xz_z_masks, xz_x_masks, xz_coefs, out):
    """Add XZ-string contributions in place.

    For each k, accumulates  coef[k] * (-1)^parity(j & z_masks[k]) * psi[j ^ x_masks[k]]
    onto out[j].  Used for fermionic toric-code plaquettes (which are both X and Z
    products) or any other generalized stabilizer with disjoint X- and Z-supports.
    """
    dim = psi.shape[0]
    n = xz_z_masks.shape[0]
    for j in prange(dim):
        s = 0.0
        for k in range(n):
            sign = 1.0 - 2.0 * _bit_parity64(j & xz_z_masks[k])
            s += xz_coefs[k] * sign * psi[j ^ xz_x_masks[k]]
        out[j] += s


def qubits_to_mask(indices: Iterable[int]) -> int:
    """Bitmask over qubit positions.  Skips -1 padding (OBC boundary stars)."""
    m = 0
    for i in indices:
        if i == -1:
            continue
        m |= (1 << int(i))
    return m


def z_string_eigvals(basis: np.ndarray, mask: int, N: int) -> np.ndarray:
    """Eigenvalues of prod_{i in mask} sigma^z_i on each computational basis state."""
    parity = np.zeros_like(basis, dtype=np.int8)
    for i in range(N):
        if mask & (1 << i):
            parity ^= ((basis >> i) & 1).astype(np.int8)
    return 1.0 - 2.0 * parity.astype(np.float64)


def hamiltonian_linop(
    geom,
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    J: float = 1.0,
    xz_stabs=None,
    dtype=np.float64,
) -> Tuple[LinearOperator, np.ndarray]:
    """Matrix-free toric / surface-code Hamiltonian.

    Returns (H, basis) with H a scipy LinearOperator of shape (2^N, 2^N).
    Memory: O(2^N) for the diagonal and basis arrays.  No sparse matrix is built.

    xz_stabs: optional list of (z_qubits, x_qubits, coef) triples for generalized
        stabilizers that have both a sigma^z and a sigma^x support (e.g. the
        decorated plaquettes of the 3D fermionic toric code).  When provided
        AND non-empty, these REPLACE the default Z-only `geom.plaq_all` term;
        the user is expected to supply the full plaquette set with the correct
        decoration coefficients.  Z-supports and X-supports of each entry must
        be disjoint.
    """
    if hy != 0:
        raise NotImplementedError("Y-field not supported here (h_y = 0 assumed).")

    N = geom.N
    dim = 1 << N
    basis = np.arange(dim, dtype=np.int64)

    use_xz = xz_stabs is not None and len(xz_stabs) > 0

    # Diagonal: -J Sum_p B_p (only if no xz_stabs override)  -  h_z Sum_i sigma^z_i
    diag = np.zeros(dim, dtype=dtype)
    if not use_xz:
        for p in geom.plaq_all:
            diag -= J * z_string_eigvals(basis, qubits_to_mask(p), N)
    if hz != 0.0:
        for i in range(N):
            diag -= hz * z_string_eigvals(basis, 1 << i, N)

    # Off-diagonal X-strings: -J A_v (vertex stars) and -h_x sigma^x_i (single sites)
    x_strings: dict[int, float] = defaultdict(float)
    for v in geom.vertex_all:
        x_strings[qubits_to_mask(v)] -= J
    if hx != 0.0:
        for i in range(N):
            x_strings[1 << i] -= hx

    # Split the user-supplied XZ stabilizers into the three bins by support type.
    xz_z_list, xz_x_list, xz_c_list = [], [], []
    if use_xz:
        for z_qubits, x_qubits, coef in xz_stabs:
            zm = qubits_to_mask(z_qubits)
            xm = qubits_to_mask(x_qubits)
            if zm & xm:
                raise ValueError(f"xz_stabs entry has overlapping z and x supports: z={z_qubits}, x={x_qubits}")
            if xm == 0:  # pure Z -> diagonal
                diag += coef * z_string_eigvals(basis, zm, N)
            elif zm == 0:  # pure X -> off-diagonal
                x_strings[xm] += coef
            else:           # genuine XZ -> separate kernel
                xz_z_list.append(zm); xz_x_list.append(xm); xz_c_list.append(coef)

    x_strings = {m: c for m, c in x_strings.items() if c != 0.0 and m != 0}

    # Pack (mask, coef) pairs into contiguous arrays the JIT kernels can iterate.
    if x_strings:
        x_masks = np.fromiter(x_strings.keys(), dtype=np.int64, count=len(x_strings))
        x_coefs = np.fromiter(x_strings.values(), dtype=dtype, count=len(x_strings))
    else:
        x_masks = np.empty(0, dtype=np.int64)
        x_coefs = np.empty(0, dtype=dtype)

    xz_z_masks = np.array(xz_z_list, dtype=np.int64) if xz_z_list else np.empty(0, dtype=np.int64)
    xz_x_masks = np.array(xz_x_list, dtype=np.int64) if xz_x_list else np.empty(0, dtype=np.int64)
    xz_coefs   = np.array(xz_c_list, dtype=dtype)    if xz_c_list else np.empty(0, dtype=dtype)
    has_xz = xz_z_masks.size > 0

    def matvec(psi):
        psi = np.ascontiguousarray(psi, dtype=dtype)
        out = np.empty_like(psi)
        _matvec_jit(psi, diag, x_masks, x_coefs, out)
        if has_xz:
            _matvec_jit_xz_add(psi, xz_z_masks, xz_x_masks, xz_coefs, out)
        return out

    H = LinearOperator((dim, dim), matvec=matvec, dtype=dtype)
    return H, basis


def expect_x_string(psi: np.ndarray, basis: np.ndarray, mask: int) -> float:
    """<psi| prod_{i in mask} sigma^x_i |psi>  =  sum_b conj(psi[b]) * psi[b XOR mask]."""
    if mask == 0:
        return float(np.real(np.sum(np.abs(psi) ** 2)))
    return float(np.real(np.sum(np.conj(psi) * psi[basis ^ mask])))


def expect_z_string(psi: np.ndarray, basis: np.ndarray, mask: int, N: int) -> float:
    """<psi| prod_{i in mask} sigma^z_i |psi>  =  sum_b |psi[b]|^2 (-1)^popcount(b & mask)."""
    if mask == 0:
        return float(np.sum(np.abs(psi) ** 2))
    return float(np.sum(np.abs(psi) ** 2 * z_string_eigvals(basis, mask, N)))


def expect_xz_string(psi: np.ndarray, basis: np.ndarray,
                     z_mask: int, x_mask: int, N: int) -> float:
    """<psi| Z(z_mask) X(x_mask) |psi> for a generalized stabilizer (disjoint supports).

    = sum_b conj(psi[b]) * (-1)^popcount(b & z_mask) * psi[b XOR x_mask].
    Reduces to expect_z_string / expect_x_string when one support is empty.
    """
    if x_mask == 0:
        return expect_z_string(psi, basis, z_mask, N)
    if z_mask == 0:
        return expect_x_string(psi, basis, x_mask)
    signs = z_string_eigvals(basis, z_mask, N)
    return float(np.real(np.sum(np.conj(psi) * signs * psi[basis ^ x_mask])))
