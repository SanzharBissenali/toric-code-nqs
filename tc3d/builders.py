"""
tc3d/builders.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for turning a `config` dict into a runnable VMC setup.

Both the training pipeline (`tc3d/train.py`) and the validation harness
(`tc3d/validation.py`) construct their geometry / Hamiltonian / ansatz /
sampler / variational state *here*, so the two can never drift apart: the model
you train is exactly the model validation scores.

The optimization loop (`run_loop`) also lives here, shared by both front-ends.

Config keys consumed (all optional except where noted; see DEFAULTS):
    System      : L (req), bc, model ∈ {"bosonic","fermionic"}
    Hamiltonian : hx, hy, hz, J
    Architecture: arch ∈ {"ToricCNN","ToricCNN_full","ToricCNN_gridinv","GeoCNN"},
                  hidden, cnn_hidden (GeoCNN edge-conv widths),
                  kernel_size (ToricCNN_gridinv invariant grid-conv kernel; auto=L)
    Sampling    : n_samples, n_chains, n_discard, chunk_size, n_sweeps, seed
"""
from __future__ import annotations

import functools
import hashlib
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from netket.jax import tree_cast

from tc3d.sampler import WeightedRule, MultiRule
from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.hamiltonian import (
    create_hamiltonian, create_hamiltonian_fermionic)
from tc3d.fermionic_decoration import fermionic_plaquettes, flux_constraint_masks
from tc3d.networks import (
    ToricCNN, ToricCNN_full, ToricCNN_gridinv, ToricCNN_gridinv_dual, GeoCNN,
    VanillaCNN, VanillaWilsonCNN, KernelManager3D, compute_edges_3D,
    plaq_grid_layout, vertex_grid_layout, star_index_arrays)


class DivergenceError(RuntimeError):
    """run_loop's guard exhausted max_rollbacks. `vs` has been restored to the
    last sane parameters before this is raised, so the caller can finalize on a
    clean state."""
    def __init__(self, step, n_rollbacks):
        self.step, self.n_rollbacks = step, n_rollbacks
        super().__init__(f"VMC diverged at step {step} after {n_rollbacks} rollbacks")


def is_bad_step(spread, hist, spike_factor, guard_warmup):
    """Pure per-step divergence test on the energy spread (sqrt of Var[H]).

    True if `spread` is non-finite, or -- once at least `guard_warmup` sane
    spreads have accumulated in `hist` -- if it exceeds `spike_factor` x their
    running median. The non-finite test is always armed; the spike test waits for
    a baseline so early steps never trigger a false rollback. Pure/stdlib so it
    is unit-testable without NetKet against real diverged curves.
    """
    if not np.isfinite(spread):
        return True
    if len(hist) < guard_warmup:
        return False
    return spread > spike_factor * float(np.median(hist))


def _tree_all_finite(tree) -> bool:
    """True iff every leaf of a pytree (e.g. an SR update `dp`) is entirely
    finite. `nk.optimizer.solver.cholesky`/`.solve` (jsp.linalg.cho_factor/
    cho_solve, jsp.linalg.solve(assume_a="pos")) return SILENT NaN -- no
    exception, even under jit -- when the S matrix is indefinite/singular.
    This is the only way to detect that failure mode."""
    return bool(jax.tree_util.tree_reduce(
        lambda acc, x: jnp.logical_and(acc, jnp.all(jnp.isfinite(x))),
        tree, jnp.array(True)))


DEFAULTS: Dict[str, Any] = {
    "bc": "PBC", "model": "bosonic", "dual_basis": False, "phase_head": False,
    "phase_head_frozen": False, "flux_penalty": 0.0, "force_complex": False,
    "hx": 0.0, "hy": 0.0, "hz": 0.0, "J": 1.0,
    "arch": "ToricCNN_full", "hidden": 8,
    "n_samples": 8192, "n_chains": 16, "n_discard": 8,
    "chunk_size": None, "n_sweeps": 48, "seed": 0,
    # Speed levers (2026-09, p3d/speed-research). Neither changes the variational
    # family or the parameter tree:
    #   compute_dtype: None/"float64" (params' precision), "float32" (true single:
    #     Precision.HIGHEST) or "tf32" (XLA default = TF32 tensor cores) -- run the
    #     network forward/backward in complex64/float32 (sampling, local energies,
    #     energy gradient); the dense-QGT Jacobian + SR solve stay in double via
    #     `exact_qgt_apply_fun`. Params/checkpoints stay complex128/float64.
    #   inv_impl: "conv" (nn.Conv) or "dense" (the invariant block's convs as ONE
    #     unfolded GEMM each -- exactly the same function of the same parameters;
    #     see networks.UnfoldedConv3D).
    "compute_dtype": None, "inv_impl": "conv",
}


def with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `config` with DEFAULTS filled in and `dtype` derived."""
    cfg = {**DEFAULTS, **config}
    if "L" not in cfg:
        raise KeyError("config must specify system size 'L'")
    # Complex weights whenever the target state is sign-full (h_y breaks
    # stoquasticity explicitly; the fermionic B~_p does so even at zero field —
    # mixed-sign off-diagonals, GS has negative amplitudes, see
    # tests/test_fermionic.py) OR by explicit request (force_complex: a
    # complex-weights control arm at hy=0, e.g. to check whether the extra phase
    # freedom changes a stoquastic ground state). build_hamiltonian/build_model
    # both key off this single derived "dtype" -- no separate force_complex
    # branch needed downstream.
    signfull = (cfg["hy"] != 0.0 or cfg["model"] == "fermionic"
               or cfg.get("force_complex", False))
    cfg.setdefault("dtype", "complex" if signfull else "float64")
    # qgt_solver default: "cholesky" for the dense-QGT path (fixed-cost direct
    # solve on the materialized S matrix; NetKet's true default -- uncapped
    # jax.scipy.sparse.linalg.cg -- measured up to ~400x/step slower under QGT
    # ill-conditioning in the sign-full hy lane; see run_loop's
    # _resolve_dense_solver docstring). Left unset (None -> NetKet's CG) for
    # anything that isn't literally qgt=="dense": run_loop raises loudly if a
    # non-None qgt_solver is ever actually combined with onthefly/srt/minsr,
    # so this must never inject a value there -- "auto" is handled precisely
    # (using the real n_params, not knowable here) by train.py once vs exists.
    if cfg.get("qgt", "auto") == "dense":
        cfg.setdefault("qgt_solver", "cholesky")
    else:
        cfg.setdefault("qgt_solver", None)
    return cfg


def _to_tuple(x):
    return tuple(_to_tuple(v) for v in x) if isinstance(x, list) else x


# =============================================================================
# Builders
# =============================================================================

def build_geometry(config: Dict[str, Any]):
    L = config["L"]
    return ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc=config.get("bc", "PBC"))


# ── field-independent Pauli-string cache ─────────────────────────────────────
# create_hamiltonian assembles ~4L^3 + 2N LocalOperators with O(n^2) `+=` algebra
# and only then converts to PauliStrings: 191 s at L=4 (rising to 675 s by the
# second point of the same process) and 523–1812 s at L=5, paid AGAIN at every
# field point even though only the (hx, hz) WEIGHTS change (2026-08-11 [t]
# instrumentation, jobs 56641739_1/56641742_1). Cache the field-independent
# string sets once per (geometry, basis) and rebuild per point by rescaling —
# verified vs create_hamiltonian at L=4 OBC dual: identical string set, identical
# max_conn_size, max|dE_loc| = 0 over random configs (bit-identical local
# energies, no new JIT shapes).
#
# ONE combined create_hamiltonian call, not three (2026-08-12 fix): calling it
# per-channel cost MORE than the original single call at single-point chunks
# (measured 1289.5 s for 3 calls vs 191–675 s for 1 at L=4 — a regression for
# the L=6 series, whose 5 h wall gives one field point per process, so the
# per-chunk amortization Patch A relies on never materializes). The J-channel
# is separated from the field channels by SUPPORT SIZE (every A_v/B_p string
# acts on >=3 sites; every hx/hy/hz string acts on exactly 1 -- verified against
# hamiltonian.py: no other term shape exists when Jy_v=Jy_p=Jbond=0), and each
# field channel is separated from the others by a distinct nonzero marker weight
# that create_hamiltonian bakes verbatim (uniformly, no per-site factor) into
# every single-site string it emits.
#
# hy is production, not prototype (promoted 2026-09-09, task A2). hy's
# single-site sigma_y term has IDENTICAL structure to hx/hz (support==1,
# uniform marker weight) and, after the dual+hy sign-law fix (main 111375a),
# is well-defined in dual mode too. A THIRD marker (_HY_MARKER) generates the
# sigma_y strings in the SAME single create_hamiltonian call whenever
# dtype=="complex" (sigma_y requires complex; a float64 cache entry has no
# "hy" key) -- verified bit-identical to create_hamiltonian at L=2 OBC dual
# (max|ΔH|=0 dense, real AND complex/hy!=0) and identical get_conn_padded
# output at L=4 OBC dual (see tests/test_hamiltonian_cache.py and this task's
# verification script). This benefits the hy-axis campaign's pure-hy sweeps
# (hx=hz=0, hy varying every point) and any --resume/restart at hy!=0.
#
# Disk persistence (task A2): the in-memory _PS_PARTS is per-PROCESS, so every
# fresh Slurm chunk / --resume / chain link still pays the full assembly cost
# once. Mirror each entry to a file keyed identically to the in-memory dict, so
# a later process can load it in milliseconds instead of rebuilding. Location:
# $TC3D_PAULI_CACHE_DIR, else $PSCRATCH/tc_nqs/pauli_cache (cluster) or
# ~/.cache/tc3d/pauli_cache (laptop); $TC3D_PAULI_CACHE=0 disables disk use
# entirely (in-memory only, the pre-persistence behaviour). Files are written
# atomically (temp file + os.replace) so a concurrent reader (another chunk
# starting at the same instant) never observes a partial write; a corrupt or
# unreadable file is treated as a miss (rebuild + rewrite), never a crash.
#
# Self-identification (2026-09-09 audit fix): a bare content hash of the KEY
# is not enough -- a file placed at the wrong path (copy/rename mistake) or a
# stale file surviving an edit to create_hamiltonian/geometry/the marker
# constants would otherwise be loaded as a SILENT hit (wrong strings, no
# error). Every file embeds `key_repr` (checked against the caller's key) and
# `code_hash` (checked against the current source); either mismatch is
# rejected and rebuilt, never loaded. `code_hash` is also baked into the file
# NAME, so an edit to the hashed sources produces a brand-new filename next to
# the old one -- the old file is simply dead weight, never addressed again
# (nothing here deletes it; that's an operator/cron concern, not correctness).
_PS_PARTS: Dict[Any, Any] = {}
_HX_MARKER, _HZ_MARKER, _HY_MARKER = 1.0, 7.0, 13.0   # distinct; never real fields
PAULI_CACHE_SCHEMA = 1   # bump whenever the on-disk npz LAYOUT itself changes


class _PauliCacheMismatch(Exception):
    """Raised by `_load_pauli_parts` when the loaded file's embedded key_repr
    or code_hash doesn't match what the caller expects. `args[0]` is "key" or
    "code" -- the caller uses it to pick the log line; either way the file is
    treated as a miss (rebuild + overwrite), never loaded."""


@functools.lru_cache(maxsize=1)
def _code_hash() -> str:
    """Fingerprint of everything that can change the cached STRING SET without
    changing `_pauli_parts`'s key: the source of create_hamiltonian
    (hamiltonian.py), the geometry it consumes (geometry.py), the marker
    constants, and the on-disk schema version. Cached (module source doesn't
    change mid-process)."""
    src_dir = Path(__file__).parent
    h = hashlib.sha256()
    for fname in ("hamiltonian.py", "geometry.py"):
        h.update((src_dir / fname).read_bytes())
    h.update(repr((_HX_MARKER, _HZ_MARKER, _HY_MARKER, PAULI_CACHE_SCHEMA)).encode())
    return h.hexdigest()[:16]


def _pauli_cache_dir() -> Optional[Path]:
    """Disk location for `_PS_PARTS` entries, or None to disable disk use
    (TC3D_PAULI_CACHE=0). See the persistence note above for precedence."""
    if os.environ.get("TC3D_PAULI_CACHE", "1") == "0":
        return None
    d = os.environ.get("TC3D_PAULI_CACHE_DIR")
    if not d:
        pscratch = os.environ.get("PSCRATCH")
        d = f"{pscratch}/tc_nqs/pauli_cache" if pscratch \
            else "~/.cache/tc3d/pauli_cache"
    return Path(os.path.expanduser(d))


def _pauli_cache_path(cache_dir: Path, key: Tuple) -> Path:
    """Readable-prefix (L, bc, dual, dtype) + content-hash + code-hash filename
    for `key`. The content hash covers the FULL key tuple (repr(key) -- J's
    repr(float(...)) is the shortest string that round-trips to the exact
    float, so two distinct J's never collide and identical J hashes identically
    across processes), so key components not shown in the prefix (N, #A_v,
    #B_p, J) still get their own distinct file. The code hash (`_code_hash`)
    means an edit to create_hamiltonian/geometry/the markers produces a NEW
    filename next to the old one -- a fresh process can never address a stale
    file by accident."""
    _, Lx, _, _, bc, _, _, dual, _, dtype = key
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    return cache_dir / f"L{Lx}_{bc}_dual{int(dual)}_{dtype}_{digest}_{_code_hash()}.npz"


def _save_pauli_parts(path: Path, key: Tuple, parts: Dict[str, Tuple]) -> None:
    """Atomic write: build the npz in a per-process temp file, fsync, then
    os.replace onto `path` -- readers see either the old file or the complete
    new one, never a partial one. All channels share one dtype (it's the same
    operator's .dtype for every channel -- see `_pauli_parts`), so it is
    stored once rather than per channel. Embeds `key_repr`/`code_hash`/`schema`
    for `_load_pauli_parts` to self-check on the way back in."""
    dtype_str = str(next(iter(parts.values()))[2])
    arrays = {"channels": np.array(list(parts.keys())), "dtype": np.array([dtype_str]),
              "key_repr": np.array([repr(key)]), "code_hash": np.array([_code_hash()]),
              "schema": np.array([PAULI_CACHE_SCHEMA])}
    for ch, (ops, ws, _dt) in parts.items():
        arrays[f"{ch}__ops"] = np.array(ops, dtype="U")
        arrays[f"{ch}__weights"] = np.asarray(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_pauli_parts(path: Path, key: Tuple) -> Dict[str, Tuple]:
    """Inverse of `_save_pauli_parts`, self-checked against the caller's `key`
    and the current code fingerprint -- raises `_PauliCacheMismatch` on either
    mismatch (wrong path, or stale post-edit file: never loaded silently) and
    propagates any other exception on corruption/truncation; the caller treats
    both as a cache miss."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["key_repr"][0]) != repr(key):
            raise _PauliCacheMismatch("key")
        if str(data["code_hash"][0]) != _code_hash():
            raise _PauliCacheMismatch("code")
        dt = np.dtype(str(data["dtype"][0]))
        parts = {str(ch): ([str(s) for s in data[f"{ch}__ops"]],
                            data[f"{ch}__weights"], dt)
                 for ch in data["channels"]}
    return parts


def _pauli_parts(geo, hi, dual, J, dtype):
    """Cached {channel: (operators, weights, dtype)} for the bosonic hx/hy/hz H.

    `weights` is the PER-UNIT weight for "hx"/"hy"/"hz" (multiply by the actual
    field to get the true contribution) and the ACTUAL weight for "J" (never
    rescaled — J is constant across a sweep). The "hy" channel only exists when
    dtype=="complex" (sigma_y requires it) -- callers must gate on that.

    hx/hy/hz are deliberately NOT in the cache key: because their weights are
    stored per-unit and rescaled by every `build_hamiltonian` call, one cached
    entry serves every field value along a whole sweep -- that's the entire
    point of separating structure from assembly cost. J IS in the key because
    its channel stores the ACTUAL (unscaled) weight, so a different J needs a
    different entry. Everything that changes the STRING SET itself is in the
    key: N (=hi.size), the three L's, bc, #A_v, #B_p (geometry), dual (basis),
    and dtype (whether the hy marker/channel exists at all).
    """
    key = (int(hi.size), geo.Lx, geo.Ly, geo.Lz, geo.bc, len(geo.vertex_all),
           len(geo.plaq_all), bool(dual), float(J), str(dtype))
    if key in _PS_PARTS:
        return _PS_PARTS[key]

    cache_dir = _pauli_cache_dir()
    path = _pauli_cache_path(cache_dir, key) if cache_dir is not None else None
    if path is not None and path.exists():
        try:
            parts = _load_pauli_parts(path, key)
            print(f"[pauli-cache] hit {path}")
            _PS_PARTS[key] = parts
            return parts
        except _PauliCacheMismatch as e:
            print(f"[pauli-cache] {e.args[0]} mismatch at {path} -- rebuilding")
        except Exception as e:
            print(f"[pauli-cache] corrupt {path} ({e!r}) -- rebuilding")

    t0 = time.time()
    markers = {"hx": _HX_MARKER, "hz": _HZ_MARKER}
    if dtype == "complex":
        markers["hy"] = _HY_MARKER
    H = create_hamiltonian(hi=hi, vertex_all=geo.vertex_all,
                           plaq_all=geo.plaq_all, bonds=geo.bonds,
                           dual=dual, J=float(J), dtype=dtype, **markers)
    ops = list(H.operators)
    ws = np.asarray(H.weights)
    support = np.array([len(s) - s.count("I") for s in ops])
    keep = ws != 0            # a zero-weight string must not inflate n_conn
    parts = {}
    m_J = keep & (support > 1)
    parts["J"] = ([s for s, k in zip(ops, m_J) if k], ws[m_J], H.dtype)
    for ch, marker in markers.items():
        m = keep & (support == 1) & (np.abs(np.abs(ws) - marker) < 1e-9)
        parts[ch] = ([s for s, k in zip(ops, m) if k], ws[m] / marker,
                    H.dtype)
    dt_build = time.time() - t0

    if path is None:
        print("[pauli-cache] disabled")
    else:
        try:
            _save_pauli_parts(path, key, parts)
            print(f"[pauli-cache] miss -> built in {dt_build:.1f} s -> wrote {path}")
        except Exception as e:
            print(f"[pauli-cache] miss -> built in {dt_build:.1f} s "
                  f"-> cache write failed ({e!r})")
    _PS_PARTS[key] = parts
    return parts


def build_hamiltonian(config: Dict[str, Any], geo, hi):
    """Returns (Ham, xz_stabs). xz_stabs is None for the bosonic model."""
    dtype = config.get("dtype", "complex" if config.get("hy", 0.0) != 0.0
                       or config.get("model", "bosonic") == "fermionic" else "float64")
    dual = config.get("dual_basis", False)
    common = dict(hx=config.get("hx", 0.0), hy=config.get("hy", 0.0),
                  hz=config.get("hz", 0.0), J=config.get("J", 1.0), dtype=dtype)
    if config.get("model", "bosonic") == "fermionic":
        if dual:
            raise NotImplementedError(
                "dual_basis is bosonic-only (the fermionic decoration is not "
                "self-dual under Hadamard conjugation)")
        xz_stabs = fermionic_plaquettes(geo, J=config.get("J", 1.0))
        Ham = create_hamiltonian_fermionic(
            hi=hi, vertex_all=geo.vertex_all, xz_stabs=xz_stabs,
            bonds=geo.bonds, **common)
        return Ham, xz_stabs
    # Bosonic hx/hy/hz sector (the sweep/campaign workhorse, hy included since
    # the dual+hy sign-law fix): rebuild from the cached strings with rescaled
    # weights instead of re-running the LocalOperator algebra. Anything beyond
    # this sector (Jy_v/Jy_p/Jbond != 0) falls through to the slow path.
    if all(float(config.get(k, 0.0) or 0.0) == 0.0
           for k in ("Jy_v", "Jy_p", "Jbond")):
        parts = _pauli_parts(geo, hi, dual, common["J"], dtype)
        channels = [("J", 1.0), ("hx", common["hx"]), ("hz", common["hz"])]
        if dtype == "complex":
            channels.append(("hy", common["hy"]))
        ops, ws, dt = [], [], None
        for ch, scale in channels:
            o, w, dt_ch = parts[ch]
            if scale == 0.0 or not o:   # create_hamiltonian omits a zero channel
                continue
            ops += o
            ws.append(scale * w)
            dt = dt_ch
        if ops:
            return nk.operator.PauliStrings(
                hi, ops, np.concatenate(ws), dtype=dt), None
    Ham = create_hamiltonian(
        hi=hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all,
        bonds=geo.bonds, dual=dual, **common)
    return Ham, None


def resolve_compute_dtype(compute_dtype, model_dtype):
    """Map the config string to (arithmetic dtype, lax matmul precision) for the
    ansatz. None/"float64"/"double" -> (None, None): compute in the parameter dtype
    (the historical behaviour). "float32"/"single" -> complex64 for a complex
    ansatz, float32 for a real one, with Precision.HIGHEST on the matmuls/convs =
    TRUE single precision (XLA's default for f32 on Ampere is TF32 tensor cores,
    10-bit mantissa: log psi off by 4e-4 and the SR update by ~40 % at L=4, measured
    2026-09-09). "tf32" -> same dtypes, XLA default precision (faster, ~1e-3
    relative arithmetic) -- opt-in only."""
    if compute_dtype in (None, "", "none", "float64", "double"):
        return None, None
    single = jnp.complex64 if model_dtype == jnp.complex128 else jnp.float32
    if compute_dtype in ("float32", "single"):
        return single, jax.lax.Precision.HIGHEST
    if compute_dtype == "tf32":
        return single, None
    raise ValueError(f"compute_dtype must be None/'float64', 'float32' or 'tf32', "
                     f"got {compute_dtype!r}")


def exact_qgt_apply_fun(vs) -> Optional[Callable]:
    """Apply function of the REFERENCE twin of `vs`'s model (same module, same
    parameters; compute_dtype=None, inv_impl="conv") for the dense-QGT Jacobian,
    or None when the model already is the reference (nothing to do -- run_loop
    then uses vs._apply_fun as before). Two reasons the QGT wants the twin:
      * precision: with compute_dtype=float32 the SR geometry and solve stay in
        double while sampling + local energies + energy gradient run fast;
      * memory: NetKet's dense Jacobian vmaps the backward pass PER SAMPLE, so
        every intermediate's cotangent is materialised per sample -- for the
        unfolded (P*C)^2 GEMM matrix that is chunk x (P*C)^2 x 16 B = 98 GB at
        L=6 (measured OOM, 2026-09-09); the conv path's per-sample cotangents are
        the (k^3 C C) kernels. The energy-gradient VJP (expect_and_grad) reduces
        over the batch before reaching the matrix, so the GEMM path is fine there.
    The twin evaluates only the n_samples Jacobian rows, i.e. the QGT stage costs
    what it always did."""
    model = vs.model
    if (getattr(model, "compute_dtype", None) is None
            and getattr(model, "inv_impl", "conv") == "conv"):
        return None
    from netket.utils.jax import wrap_to_support_scalar   # what MCState wraps with
    return wrap_to_support_scalar(
        model.clone(compute_dtype=None, inv_impl="conv", precision=None).apply)


def build_model(config: Dict[str, Any], geo):
    """Instantiate the ansatz named by `config['arch']`.

    The same two ansätze serve both the bosonic and fermionic models: the Wilson
    4-product enforces A_v invariance, and A_v is unchanged by the decoration.
    """
    plaq_tuple = tuple(tuple(p) for p in geo.plaq_all)
    hidden = config.get("hidden", 8)
    arch = config.get("arch", "ToricCNN_full")

    # Dual (Hadamard) basis: star tokens on the vertex grid for the gridinv
    # sandwich. GeoCNN is a pure function of edge spins (no stabilizer-aligned
    # structure), hence basis-agnostic — allowed as the symmetry-unaware control
    # arm. Refuse everything else loudly (BEFORE the Vanilla* early returns, or
    # the flag would be silently ignored).
    if config.get("dual_basis", False) and arch not in ("ToricCNN_gridinv", "GeoCNN"):
        raise NotImplementedError(
            f"dual_basis is implemented for arch='ToricCNN_gridinv' (star tokens) "
            f"and 'GeoCNN' (basis-agnostic control); got arch={arch!r}")
    if config.get("phase_head", False) and (arch != "ToricCNN_gridinv"
                                            or config.get("dual_basis", False)):
        raise NotImplementedError(
            "phase_head (token-quadratic phase) is implemented for the primal "
            f"ToricCNN_gridinv only; got arch={arch!r}"
            f"{' + dual_basis' if config.get('dual_basis') else ''}")

    # Map the config's string dtype ("complex" when h_y != 0, else "float64") to a
    # concrete jax dtype for the ansatz. A complex log ψ is required for the sign-full
    # (h_y != 0 / fermionic) regime; the complex path is implemented for the
    # Wilson-sandwich archs (ToricCNN / ToricCNN_full / ToricCNN_gridinv) and GeoCNN —
    # all use complex-aware split activations and thread dtype through GeoConv3D.
    # Vanilla* are still deferred (nn.elu raises on complex), so refuse them loudly
    # rather than train a real ansatz against a complex Hamiltonian.
    dt_str = config.get("dtype", "complex" if config.get("hy", 0.0) != 0.0 else "float64")
    model_dtype = jnp.complex128 if dt_str == "complex" else jnp.float64
    compute_dtype, precision = resolve_compute_dtype(config.get("compute_dtype"), model_dtype)
    inv_impl = config.get("inv_impl", "conv") or "conv"
    if (compute_dtype is not None or inv_impl != "conv") and arch != "ToricCNN_gridinv":
        raise NotImplementedError(
            f"compute_dtype/inv_impl are implemented for arch='ToricCNN_gridinv' "
            f"(primal and dual); got arch={arch!r}")
    if model_dtype == jnp.complex128 and arch not in (
            "ToricCNN", "ToricCNN_full", "ToricCNN_gridinv", "GeoCNN"):
        raise NotImplementedError(
            f"complex (h_y != 0) ansatz is implemented for ToricCNN / ToricCNN_full / "
            f"ToricCNN_gridinv / GeoCNN so far; got arch={arch!r}. Use one of those or "
            "extend build_model (Vanilla* need complex-aware activations).")

    if geo.bc == "OBC" and arch in ("VanillaCNN", "VanillaWilsonCNN"):
        raise ValueError(
            f"{arch} is PBC-only (CIRCULAR padding + dense (3,L,L,L) fold); "
            "use ToricCNN or ToricCNN_full for OBC.")
    if arch == "VanillaCNN":
        # plain grid CNN baseline — bypasses KernelManager3D entirely
        edges = compute_edges_3D(geo)            # (3, Lx, Ly, Lz)
        return VanillaCNN(
            shape=tuple(edges.shape), edges_flat=tuple(int(i) for i in edges.reshape(-1)),
            hidden=hidden, depth=config.get("vanilla_depth", 2),
            kernel_size=config.get("kernel_size", 3))
    if arch == "VanillaWilsonCNN":
        # Wilson sandwich (noninv → Wilson → inv) with plain grid convs, no GeoConv3D
        edges = compute_edges_3D(geo)            # (3, Lx, Ly, Lz)
        return VanillaWilsonCNN(
            shape=tuple(edges.shape), edges_flat=tuple(int(i) for i in edges.reshape(-1)),
            plaq_all=plaq_tuple,
            noninv_channels=config.get("noninv_channels", 1),
            n_noninv=config.get("n_noninv", 1),
            inv_hidden=tuple(config.get("inv_hidden", (4,)) or ()),
            kernel_size=config.get("kernel_size", 3),
            noninv_identity=config.get("noninv_identity", True))
    km = KernelManager3D(geo,
                         radius_edge=config.get("radius_edge", 1.05),
                         radius_plaq=config.get("radius_plaq", 1.05))
    if arch == "ToricCNN":
        return ToricCNN(km=km, plaq_all=plaq_tuple, hidden=hidden, dtype=model_dtype)
    if arch == "ToricCNN_full":
        return ToricCNN_full(
            km=km, plaq_all=plaq_tuple, hidden=hidden,
            noninv_channels=config.get("noninv_channels", 4),
            n_noninv=config.get("n_noninv", 2),
            inv_hidden=tuple(config.get("inv_hidden", (4, 4)) or ()),
            dtype=model_dtype)
    if arch == "GeoCNN":
        # geometry-exact CNN, NO Wilson 4-product: same kernel, not A_v-invariant
        return GeoCNN(km=km,
                      hidden=tuple(config.get("cnn_hidden", (4, 4, 4)) or ()),
                      dtype=model_dtype)
    # optional per-layer noninv widths (overrides noninv_channels x n_noninv)
    nh = config.get("noninv_hidden")
    noninv_hidden = tuple(nh) if nh else None
    if config.get("dual_basis", False):
        star_idx, star_mask = star_index_arrays(geo)
        grid_dims, vertex_lin = vertex_grid_layout(geo)
        return ToricCNN_gridinv_dual(
            km=km, star_idx=star_idx, star_mask=star_mask,
            grid_dims=grid_dims, vertex_lin=vertex_lin,
            noninv_channels=config.get("noninv_channels", 4),
            n_noninv=config.get("n_noninv", 2),
            noninv_hidden=noninv_hidden,
            inv_hidden=tuple(config.get("inv_hidden", (4, 4)) or ()),
            kernel_size=config.get("kernel_size"),
            padding="CIRCULAR" if geo.bc == "PBC" else "SAME",
            dtype=model_dtype, compute_dtype=compute_dtype, inv_impl=inv_impl,
            precision=precision)
    if arch == "ToricCNN_gridinv":
        # Wilson sandwich with a standard grid nn.Conv3D invariant block,
        # kernel → L (override with kernel_size). PBC: CIRCULAR; OBC: zero pad.
        phase_head = bool(config.get("phase_head", False))
        phase_head_frozen = bool(config.get("phase_head_frozen", False))
        if phase_head and phase_head_frozen:
            raise ValueError("phase_head and phase_head_frozen are exclusive")
        if (phase_head or phase_head_frozen) and model_dtype != jnp.complex128:
            raise ValueError("phase_head adds an imaginary token-quadratic term — "
                             "it requires the complex (sign-full) dtype")
        flux_kappa = float(config.get("flux_penalty", 0.0) or 0.0)
        flux_masks = ()
        if flux_kappa:
            # analytic flux-sector projection (fermionic): kill the ghost flux
            # cosets the token-blind trunk cannot separate (adds NO parameters)
            if config.get("model", "bosonic") != "fermionic":
                raise ValueError("flux_penalty is defined by the decorated-plaquette "
                                 "pair moves — fermionic model only")
            flux_masks = flux_constraint_masks(fermionic_plaquettes(geo))
        grid_dims, grid_lin, grid_mask = plaq_grid_layout(geo)
        return ToricCNN_gridinv(
            km=km, plaq_all=plaq_tuple,
            grid_dims=grid_dims, grid_lin=grid_lin, grid_mask=grid_mask,
            phase_head=phase_head, phase_head_frozen=phase_head_frozen,
            flux_masks=flux_masks, flux_kappa=flux_kappa,
            noninv_channels=config.get("noninv_channels", 4),
            n_noninv=config.get("n_noninv", 2),
            noninv_hidden=noninv_hidden,
            inv_hidden=tuple(config.get("inv_hidden", (4, 4)) or ()),
            kernel_size=config.get("kernel_size"),
            padding="CIRCULAR" if geo.bc == "PBC" else "SAME",
            dtype=model_dtype,   # complex128 in the sign-full (h_y) regime; else float64
            compute_dtype=compute_dtype, inv_impl=inv_impl, precision=precision)
    raise ValueError(
        f"unknown arch {arch!r} (expected ToricCNN, ToricCNN_full, "
        "ToricCNN_gridinv or GeoCNN)")


def build_sampler(config: Dict[str, Any], hi, geo):
    """WeightedRule(LocalRule, cluster MultiRule) — the topological-phase fix.

    The cluster moves flip the generators of the ansatz's ENFORCED symmetry
    group, i.e. the off-diagonal stabilizer family in the sampling basis:

    Primal: each cluster is a vertex star's edges (A_v = X⁶ flips). Under OBC
    the boundary stars are truncated (fewer than 6 edges, padded with -1 in
    geo.vertex_all); we strip the -1 and pad each cluster back to width 6 by
    repeating its last valid edge. The MultiRule flip is `.at[cluster].set(-...)`,
    idempotent under duplicate indices, so a padded cluster flips exactly its
    distinct edges — correct for truncated stars (and at L=2 OBC there are no
    full bulk stars at all). PBC is unaffected: every star already has 6
    distinct edges, so the padding is a no-op.

    Dual basis: the flip family is B_p = X⁴ (faces), so the clusters are the
    plaquette 4-tuples — already fixed-width with no -1 (OBC drops incomplete
    faces), so no padding is needed.
    """
    if config.get("dual_basis", False):
        clusters = np.array(geo.plaq_all)                  # (N_p, 4), no -1
    else:
        hetero = geo.get_vertex_all_hetero()               # -1 stripped, ragged
        width = max(len(v) for v in hetero)
        clusters = np.array([v + [v[-1]] * (width - len(v)) for v in hetero])
    if config.get("model", "bosonic") == "fermionic":
        # The fermionic model has a SECOND off-diagonal stabilizer family: each
        # B~_p carries an X^2 body-diagonal pair (fermionic_plaquettes -> x_edges).
        # On the converged GS those flips are pure sign flips (|psi'/psi|^2 = 1),
        # and they are the only moves that cross star-suborbits at h=0 — without
        # them the chain freezes into a fraction of the GS support. Padding a
        # pair to the star width by repeating its last index is safe: the
        # .at[cluster].set(-...) flip is idempotent under duplicates (same trick
        # as the truncated OBC stars above).
        width = clusters.shape[1]
        pairs = [x + [x[-1]] * (width - len(x))
                 for _, x, _ in fermionic_plaquettes(geo)]
        clusters = np.vstack([clusters, np.array(pairs)])
    samp_ratio = geo.N / len(clusters)
    weighted = WeightedRule(
        (samp_ratio / (samp_ratio + 1), 1 - samp_ratio / (samp_ratio + 1)),
        [nk.sampler.rules.LocalRule(), MultiRule(clusters)],
    )
    n_sweeps = config.get("n_sweeps") or geo.N * 2
    # sweep_size, not the deprecated n_sweeps kwarg: netket >= 3.17 removed it
    # (same meaning; 3.10+ accepts sweep_size, so this covers venv/cluster/Colab)
    return nk.sampler.MetropolisSampler(
        hi, rule=weighted, n_chains=config.get("n_chains", 16),
        sweep_size=n_sweeps, dtype=jnp.int8)


def build_state(config: Dict[str, Any], *, build_ham: bool = True
                ) -> Tuple[Any, Any, Any, Any, Any]:
    """Build everything: returns (geo, hi, Ham, vs, xz_stabs).

    `build_ham=False` skips Hamiltonian construction (Ham, xz_stabs = None) — for
    consumers that only need the ansatz/sampler/state (e.g. `fm.py` order-parameter
    extraction, which never touches H). Constructing the 3D H is a non-trivial
    Python cost, so skipping it matters when reloading many checkpoints in a loop.
    """
    cfg = with_defaults(config)
    geo = build_geometry(cfg)
    hi = nk.hilbert.Spin(s=1/2, N=geo.N)
    Ham, xz_stabs = build_hamiltonian(cfg, geo, hi) if build_ham else (None, None)
    model = build_model(cfg, geo)
    sa = build_sampler(cfg, hi, geo)
    vs = nk.vqs.MCState(sa, model, n_samples=cfg["n_samples"],
                        n_discard_per_chain=cfg["n_discard"],
                        chunk_size=cfg["chunk_size"], seed=cfg["seed"])
    return geo, hi, Ham, vs, xz_stabs


# =============================================================================
# Shared optimization loop (one loop, two front-ends)
# =============================================================================

#  ── PROTOTYPE (speed investigation, not production): pick the linear solver
#  used inside dense-QGT SR. NetKet's `nk.optimizer.SR(qgt=QGTJacobianDense, ...)`
#  defaults its `solver` kwarg to `jax.scipy.sparse.linalg.cg` (uncapped:
#  maxiter defaults to 10 * (2 * n_params) for our non-holomorphic complex
#  ansatz) -- QGTJacobianDense forms the Jacobian O eagerly but never
#  materializes/factorizes the S=O^H O matrix on this path; the "dense" name
#  refers only to O's storage. CG's iteration count scales ~sqrt(cond(S)), so
#  a well-conditioned step (healthy hy) and an ill-conditioned one (deep
#  polarized hy) can differ by 100x in wall-clock at IDENTICAL matrix size --
#  this is the leading hypothesis for the observed hy>=1.4 qgt-stage blowup.
#  `qgt_solver` swaps in a fixed-cost alternative for A/B timing:
#    "cg"        : NetKet's default (uncapped CG) -- the status quo.
#    "cgN"       : CG capped at N iterations (e.g. "cg100", "cg300").
#    "cholesky"  : direct Cholesky solve on the materialized S matrix
#                  (nk.optimizer.solver.cholesky) -- O(n_params_real^3) FIXED
#                  cost, insensitive to conditioning (until numerically
#                  singular).
#    "solve"     : jsp.linalg.solve(assume_a="pos") on the materialized S
#                  matrix (nk.optimizer.solver.solve) -- same cost class as
#                  cholesky, different LAPACK path.
#  None (default) keeps the exact production behaviour (no solver= passed).
def _qgt_dense_with_apply(vstate, *, apply_fun, diag_shift, diag_scale=None,
                          mode=None, holomorphic=None, chunk_size=None, **kwargs):
    """NetKet's `QGTJacobianDense(vstate, ...)` with `apply_fun` in place of
    `vstate._apply_fun` (same parameters, samples, model_state, chunking) --
    the hook that keeps the QGT Jacobian in double when the state's own model
    evaluates in reduced precision (see `exact_qgt_apply_fun`)."""
    from netket.optimizer.qgt.qgt_jacobian import QGTJacobian_DefaultConstructor
    if chunk_size is None:
        chunk_size = getattr(vstate, "chunk_size", None)
    return QGTJacobian_DefaultConstructor(
        apply_fun, vstate.parameters, vstate.model_state, vstate.samples,
        dense=True, mode=mode, holomorphic=holomorphic, diag_shift=diag_shift,
        diag_scale=diag_scale, chunk_size=chunk_size, **kwargs)


def kernel_solve(A, b, *, x0=None):
    """Dense-SR solve by the kernel trick (Woodbury identity), never forming the
    n_params x n_params S matrix. NetKet's dense QGT holds the centred,
    1/sqrt(n)-scaled Jacobian O (real (n_s, n_p) in mode 'real'; (n_s, 2, n_p) real
    in mode 'complex' = Re/Im rows stacked, n_p counting Re and Im parts of
    complex parameters) with S = O^T O + lam*I. For any right-hand side b,
        (O^T O + lam I)^-1 b = (b - O^T (O O^T + lam I)^-1 O b) / lam ,
    which costs O(n_s^2 n_p) flops and O(n_s^2) memory instead of O(n_p^2 n_s)
    and O(n_p^2): the win once n_p exceeds the number of Jacobian rows (L=6
    complex ansatz: n_p = 37.3k vs 2*8192 rows; S alone is 11 GB there, and
    cho_factor needs a second copy -- the 20.8 GiB OOM). Same regularised
    solution as `cholesky` (equal diag_shift), so it is an exact-equivalence
    swap. The 1/lam division amplifies roundoff by ~cond(K+lam)/lam, so one
    step of iterative refinement (re-using the Cholesky factor) is applied:
    that brings the agreement with the direct solve to ~1e-12 relative.
    Signature matches nk.optimizer.solver.* (returns (x, info))."""
    import jax.scipy as jsp
    O = A.O.reshape((-1, A.O.shape[-1]))                  # rows = (samples[, Re/Im])
    lam = A.diag_shift
    K = O @ O.conj().T + lam * jnp.eye(O.shape[0], dtype=O.dtype)
    c_low = jsp.linalg.cho_factor(K)

    def solve(rhs):
        return (rhs - O.conj().T @ jsp.linalg.cho_solve(c_low, O @ rhs)) / lam

    x = solve(b)
    r = b - (O.conj().T @ (O @ x) + lam * x)              # residual of (S + lam) x = b
    return x + solve(r), None


def _resolve_dense_solver(qgt_solver: Optional[str]):
    if not qgt_solver or qgt_solver == "cg":
        return None
    if qgt_solver == "cholesky":
        return nk.optimizer.solver.cholesky
    if qgt_solver == "solve":
        return nk.optimizer.solver.solve
    if qgt_solver == "kernel":
        return kernel_solve
    if qgt_solver.startswith("cg") and qgt_solver[2:].isdigit():
        import jax.scipy.sparse.linalg as jsla
        return functools.partial(jsla.cg, maxiter=int(qgt_solver[2:]))
    raise ValueError(f"unknown qgt_solver={qgt_solver!r} "
                     "(expected None/'cg', 'cgN', 'cholesky', 'solve', or 'kernel')")


def run_loop(vs, Ham, n_iter: int, dt: float, diag_shift: float,
             on_step: Optional[Callable] = None, lr_min: Optional[float] = None,
             qgt: str = "auto", qgt_solver: Optional[str] = None, start_step: int = 0,
             total_iter: Optional[int] = None,
             time_phases: bool = False, on_timing: Optional[Callable] = None,
             grad_guard: bool = False, spike_factor: float = 10.0,
             max_rollbacks: int = 5, rollback_shift_boost: float = 10.0,
             rollback_cooldown: int = 20, baseline_window: int = 20,
             guard_warmup: int = 5, warmup_frac: float = 0.0,
             qgt_apply_fun: Optional[Callable] = None):
    """VMC + Sgd + SR(diag_shift) for n_iter steps. Returns (vs, n_rollbacks) --
    n_rollbacks is 0 on the srt/untimed paths (no guard there; see `grad_guard`
    below) and the guard's rollback count on the instrumented path.

    Learning rate: constant `dt` by default, or — if `lr_min` is given — a cosine
    decay from `dt` down to `lr_min` across the `total_iter` steps
    (`optax.cosine_decay_schedule`, alpha = lr_min/dt). `warmup_frac > 0` prepends
    a linear ramp from 0 to `dt` over the first `warmup_frac * total_iter` steps
    (`optax.warmup_cosine_decay_schedule`) before the same cosine decay to
    `lr_min` — pass a `guard_warmup` comfortably above the warmup step count so
    the divergence guard's baseline isn't established mid-ramp.

    `qgt` selects the SR geometric-tensor representation: "dense"
    (QGTJacobianDense — form the matrix once + direct solve; ~9x faster for the
    few-thousand-parameter nets here), "onthefly" (NetKet's CG default, matrix-
    free; for n_params ≫ n_samples or when the dense n_params^2 matrix would not
    fit), or "auto" (dense when n_params ≤ 8192, else onthefly).

    `qgt_apply_fun` (dense path only): evaluate the QGT Jacobian with this apply
    function instead of `vs._apply_fun` -- `exact_qgt_apply_fun(vs)` returns the
    reference twin (double precision, conv implementation) of a fast
    (compute_dtype / inv_impl="dense") model, so the SR geometry and solve stay
    exact and the per-sample Jacobian never sees the unfolded GEMM matrix (98 GB
    at L=6 otherwise), while everything else runs fast. Ignored (with a printed
    note) on the srt/minsr and onthefly paths.

    `start_step`/`total_iter` support **resuming** a timed-out run: this call
    runs `n_iter` more steps, but the cosine-LR schedule and the step index given
    to `on_step` are offset by `start_step`, and the decay horizon is the original
    `total_iter` (defaults to `n_iter`). A fresh run leaves both at their defaults
    and behaves identically to before.

    If `on_step` is given it is called as on_step(step, E, vs) each iteration;
    pass None to skip per-step expectation when only the final state is needed
    (cheaper).

    `time_phases=True` splits each step into its three real costs — sampling,
    local-energy+gradient (`expect_and_grad`), and the QGT/SR solve — and times
    each with a `block_until_ready` barrier (JAX is async, so without the barrier
    the numbers are meaningless dispatch times). It prints a `[t]` line per step
    and a `[timing]` median summary (step 0 excluded — that's the XLA compile),
    and calls `on_timing(step, {sample,grad,qgt,update,total})` if given. In this
    mode the energy passed to `on_step` is the `expect_and_grad` estimate on the
    step's own samples (NetKet's `driver.energy`), which also avoids the second,
    redundant `vs.expect(Ham)` the untimed path does purely for logging.

    `grad_guard=True` (instrumented path only) makes the loop self-healing against
    the SR blow-up where an ill-conditioned QGT solve turns a normal gradient into
    a giant update that corrupts the parameters. Each step's energy spread is
    checked (`is_bad_step`); on a non-finite value or a `spike_factor`x jump over
    the running median of recent sane spreads, the loop restores the last sane
    snapshot -- **both parameters and the warm sampler state** (restoring cold/
    reseeded chains instead would make local energies explode on a converged
    peaked state and death-spiral the guard) -- boosts `diag_shift` (escalating
    with consecutive rollbacks, capped) for `rollback_cooldown` steps, and retries
    without letting the bad point reach `on_step` (so curve/checkpoint stay clean).
    After `max_rollbacks` consecutive failures it raises `DivergenceError` with
    `vs` left on the last sane state (warm chains), so the caller finalizes on
    valid samples.
    """
    total_iter = total_iter or n_iter
    if warmup_frac > 0:
        import optax
        base = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=dt,
            warmup_steps=int(warmup_frac * total_iter),
            decay_steps=total_iter, end_value=(lr_min if lr_min is not None else 0.0))
        lr = (lambda s: base(s + start_step)) if start_step else base
    elif lr_min is not None and lr_min != dt:
        import optax
        base = optax.cosine_decay_schedule(init_value=dt, decay_steps=total_iter,
                                           alpha=lr_min / dt)
        lr = (lambda s: base(s + start_step)) if start_step else base
    else:
        lr = dt
    opt = nk.optimizer.Sgd(learning_rate=lr)

    def _timed(fn):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        return out, time.perf_counter() - t0

    # SRt / minSR: solve in the n_samples space (NTK) instead of the n_params space.
    # The right solver once n_params >> n_samples (dense QGT stores/solves n_params^2,
    # which OOMs at large kernel). Uses NetKet's stable public VMC_SRt driver, which
    # does sample+solve+update atomically inside advance() -- so we time TOTAL per
    # step (no sample/grad/qgt split) and there is no divergence guard here (VMC_SRt
    # is far more stable than dense SR; the guard exists for the dense blow-up).
    use_srt = qgt in ("srt", "minsr")
    if qgt_apply_fun is not None and (use_srt or not (
            qgt == "dense" or (qgt == "auto" and vs.n_parameters <= 8192))):
        print("[run_loop] note: qgt_apply_fun (double-precision QGT twin) only wires "
              f"into the dense QGT path; qgt={qgt!r} uses the state's own model.",
              flush=True)
    if use_srt:
        # jacobian_mode: force the non-holomorphic 'complex' (real+imag) treatment when
        # the ansatz is complex (sign-full h_y != 0), instead of relying on the
        # dtype-inferred default. Our complex CNN uses complex weights throughout with a
        # split (non-holomorphic) activation, so 'complex' — NOT the holomorphic mode — is
        # the correct QGT geometry; the real (h_y=0) path keeps 'real'. Detected cheaply
        # from the parameter dtype (no forced (re)compilation of the ansatz).
        params_complex = any(np.iscomplexobj(np.asarray(p))
                             for p in jax.tree_util.tree_leaves(vs.parameters))
        jac_mode = "complex" if params_complex else "real"
        driver = nkx.driver.VMC_SRt(Ham, opt, diag_shift=diag_shift,
                                    jacobian_mode=jac_mode, variational_state=vs)
        agg = []
        for step in range(n_iter):
            gstep = start_step + step
            if time_phases:
                _, t = _timed(lambda: driver.advance(1))
                if step > 0:                    # step 0 total is dominated by compile
                    agg.append(t)
                print(f"  [t] step {gstep:4d}: srt(advance) total {t:7.3f} s", flush=True)
                if on_timing is not None:
                    on_timing(gstep, {"total": t})
            else:
                driver.advance(1)
            if on_step is not None:
                on_step(gstep, driver._loss_stats, vs)   # Stats: .mean/.variance/.error_of_mean
        if agg:
            med = float(np.median(agg))
            print(f"[timing] srt median over {len(agg)} steps (excl. compile step 0): "
                  f"{med:.3f} s/step", flush=True)
            print(f"[timing] extrapolated: ~{med * total_iter / 60:.1f} min for "
                  f"{total_iter} steps (+ ~one-off compile)", flush=True)
        return vs, 0     # no guard on this path (see the note above)

    use_dense = qgt == "dense" or (qgt == "auto" and vs.n_parameters <= 8192)
    _solver = _resolve_dense_solver(qgt_solver)   # None -> NetKet default (CG)
    if _solver is not None and not use_dense:
        # qgt_solver only wires into the QGTJacobianDense branch below -- on
        # the onthefly path (qgt="onthefly", or "auto" with n_params > 8192)
        # it would silently no-op, leaving the caller thinking they got their
        # requested solver when they got NetKet's onthefly CG instead.
        raise ValueError(
            f"qgt_solver={qgt_solver!r} has no effect: qgt={qgt!r} resolves to "
            f"the onthefly path (use_dense=False, n_params={vs.n_parameters}). "
            "Pass qgt='dense' explicitly, or drop qgt_solver.")

    qgt_ctor = nk.optimizer.qgt.QGTJacobianDense
    if qgt_apply_fun is not None:
        qgt_ctor = functools.partial(_qgt_dense_with_apply, apply_fun=qgt_apply_fun)

    def _build_sr(shift):
        if use_dense:
            kwargs = dict(diag_shift=shift, holomorphic=False)
            if _solver is not None:
                kwargs["solver"] = _solver
            return nk.optimizer.SR(qgt=qgt_ctor, **kwargs)
        return nk.optimizer.SR(diag_shift=shift)

    sr = _build_sr(diag_shift)
    driver = nk.driver.VMC(Ham, opt, variational_state=vs, preconditioner=sr)

    if not time_phases:
        for step in range(n_iter):
            driver.advance(1)
            if on_step is not None:
                on_step(start_step + step, vs.expect(Ham), vs)
        return vs, 0     # no guard on this path (guard = grad_guard and time_phases)

    # --- instrumented path: split one VMC+SR step into its timed phases -------
    # This replicates VMC._forward_and_backward (reset -> sample -> expect_and_grad
    # -> preconditioner) + update_parameters exactly, so the trajectory is
    # identical to the untimed path; we just insert block_until_ready barriers to
    # attribute wall-clock to sampling vs. local-energy/grad vs. QGT solve.
    def _sample():
        vs.reset()              # advance/keep chains (reset_chains=False) + mark stale
        return vs.samples       # property access forces the (warm-started) sampling

    def _copy(tree):
        return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), tree)

    def _snapshot():
        # Snapshot params AND the WARM sampler state (sigma + rng + counters), so a
        # rollback restores thermalized chains near the peak -- NOT a cold reseed,
        # whose exp-tiny-amplitude configs make local energies explode on a
        # converged wavefunction and death-spiral the guard.
        return (_copy(vs.parameters), _copy(vs.sampler_state))

    def _restore(snap):
        params, sstate = snap
        vs.parameters = params
        vs.sampler_state = sstate          # warm chains back; next _sample() resamples

    guard = grad_guard and time_phases     # guard only wired into the instrumented path
    last_good = _snapshot()                # sane by construction (fresh init or gated resume)
    spread_hist: list = []                 # rolling window of recent *sane* spreads
    n_rollbacks = 0                        # total, for the log line
    consec = 0                             # CONSECUTIVE rollbacks (reset on a sane step)
    cooldown = 0                           # steps of boosted diag_shift remaining

    def _rollback(reason: str):
        """Common bad-step handling, shared by the pre-solve spread/finite
        check and the post-solve dp-finiteness check below: log a uniform
        [guard] line, restore the last sane (params, sampler_state), escalate
        diag_shift for `rollback_cooldown` steps, and raise DivergenceError
        past max_rollbacks. Caller must `continue` immediately after -- the
        bad step must never reach update_parameters/on_step/on_timing
        (curve/checkpoint stay clean)."""
        nonlocal n_rollbacks, consec, sr, cooldown
        n_rollbacks += 1
        consec += 1
        base = float(np.median(spread_hist)) if spread_hist else float("nan")
        print(f"  [guard] step {gstep}: BAD ({reason}, spread={spread:.4g}, "
              f"baseline={base:.4g}) -> rollback #{n_rollbacks} (consec {consec})",
              flush=True)
        _restore(last_good)                # warm params + chains back
        if consec > max_rollbacks:
            print(f"  [guard] exceeded max_rollbacks={max_rollbacks} "
                  f"consecutively; giving up on last sane state.", flush=True)
            raise DivergenceError(gstep, n_rollbacks)
        # escalating, capped regularization for a rare deterministic re-blowup
        sr = _build_sr(diag_shift * rollback_shift_boost ** min(consec, 3))
        cooldown = rollback_cooldown

    agg = defaultdict(list)
    for step in range(n_iter):
        gstep = start_step + step
        _, t_s = _timed(_sample)
        (E, grad), t_g = _timed(lambda: vs.expect_and_grad(Ham))

        if guard:
            em, ev = float(np.real(E.mean)), float(np.real(E.variance))
            finite = np.isfinite(em) and np.isfinite(ev) and ev >= 0.0
            spread = float(np.sqrt(ev)) if finite else np.inf
            if is_bad_step(spread, spread_hist, spike_factor, guard_warmup):
                _rollback(f"finite={finite}")
                continue          # skip update AND on_step -> curve/checkpoint stay clean
            # spread/E is sane, but do NOT advance the snapshot/baseline yet --
            # the step still has to survive the post-solve dp check right
            # below. If we reset consec=0 here unconditionally, a solver that
            # fails EVERY step (E always sane, dp always non-finite) would
            # never accumulate past consec=1 and DivergenceError would never
            # fire -- run_loop would silently burn through n_iter steps with
            # zero effective parameter updates instead of giving up.

        dp, t_q = _timed(lambda: tree_cast(sr(vs, grad, gstep), vs.parameters))
        if guard:
            if not _tree_all_finite(dp):
                # The pre-solve check above already passed (E/grad were sane)
                # -- if dp is non-finite here, the SR SOLVE ITSELF blew up
                # silently (cho_factor/cho_solve, jsp.linalg.solve(assume_a=
                # "pos") never raise on an indefinite S; see _tree_all_finite).
                # Must catch this BEFORE driver.update_parameters(dp) below,
                # or a NaN dp corrupts vs.parameters for one step while this
                # step's (still-sane) E reaches on_step/checkpoint looking
                # healthy. Routes through the SAME _rollback as the pre-solve
                # check, so a persistent failure here still escalates consec
                # and eventually raises (see note above).
                _rollback("dp non-finite (solver NaN)")
                continue      # never reaches update_parameters/on_step/checkpoint
            # step cleared BOTH checks: NOW it's sane -- advance the snapshot +
            # baseline, decay the shift boost
            consec = 0
            last_good = _snapshot()
            spread_hist.append(spread)
            if len(spread_hist) > baseline_window:
                spread_hist.pop(0)
            if cooldown > 0:
                cooldown -= 1
                if cooldown == 0:
                    sr = _build_sr(diag_shift)
        _, t_u = _timed(lambda: (driver.update_parameters(dp), vs.parameters)[-1])
        td = {"sample": t_s, "grad": t_g, "qgt": t_q, "update": t_u,
              "total": t_s + t_g + t_q + t_u}
        print(f"  [t] step {gstep:4d}: sample {t_s:6.3f} | grad {t_g:6.3f} | "
              f"qgt {t_q:6.3f} | upd {t_u:6.3f} | total {td['total']:6.3f} s", flush=True)
        if step > 0:                    # step 0 total is dominated by XLA compile
            for k, v in td.items():
                agg[k].append(v)
        if on_step is not None:
            on_step(gstep, E, vs)
        if on_timing is not None:
            on_timing(gstep, td)

    if agg["total"]:
        med = {k: float(np.median(v)) for k, v in agg.items()}
        print(f"[timing] median over {len(agg['total'])} steps (excl. compile "
              f"step {start_step}):  sample {med['sample']:.3f} | grad {med['grad']:.3f} | "
              f"qgt {med['qgt']:.3f} | upd {med['update']:.3f} | "
              f"total {med['total']:.3f} s/step", flush=True)
        print(f"[timing] extrapolated: ~{med['total'] * (total_iter or n_iter) / 60:.1f} "
              f"min for {total_iter or n_iter} steps (+ ~one-off compile)", flush=True)
    return vs, n_rollbacks
