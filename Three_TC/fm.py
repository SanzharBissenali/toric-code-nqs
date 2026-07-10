"""
Three_TC/fm.py
─────────────────────────────────────────────────────────────────────────────
Fredenhagen–Marcu (BFFM) phase-transition detection from *trained* NQS
checkpoints for the 3D toric code.

Pipeline (one fixed L at a time; stack over L afterwards for FSS):

    checkpoints {name}.mpack + {name}.json   (one per (L, hx, hz))
       │  load_vstate : build_state(config) + flax.from_bytes(mpack)
       ▼
    fm_sweep(dir, sector, L, hx, field="hz")  → table  field, O_FM ± err, ⟨σz⟩
       │  per checkpoint: build the loop/membrane operators, fm_ratio(vs, …)
       ▼
    fit_transition(field, O, Oe)  → h_c  (logistic inflection = derivative peak),
                                    with a finite-difference derivative cross-check
       ▼
    plot_fm_sweep(...)            (matplotlib, optional)

Two sectors, ONE shared consumer (the 3D e/m duality is not symmetric):
  • electric (hz sweep): σ^z **loop/string** in a lattice plane — the 2D BFFM
    embedded in 3D. Diagonal ⇒ cheap, low MC variance.
  • magnetic (hx sweep): σ^x **membrane** (σ^x on the axis-edges piercing an
    R×R patch; its boundary is the flux loop). Off-diagonal ⇒ noisier.

Only the index-set builder differs; `fm_ratio`, the loader, the sweep and the
analysis are shared.

Never run 3D ED/sweeps locally (see CLAUDE.md). This module is for Colab,
where the trained checkpoints live; `_validate.py`-style index checks are the
only thing meant to run on the dev box.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import netket as nk
import flax

from Three_TC.builders import build_state

VSCORE_MAX = 1.0    # skip finished runs whose Vscore exceeds this: a variance blow-up
                    # the in-run guard missed (diverged:false but garbage state). Matches
                    # analysis/check_convergence.py's BAD-VSCORE gate.


# =============================================================================
# Geometry → edge index sets (the only thing that differs between sectors)
# =============================================================================

def _edge(geo, coord) -> int:
    """Qubit index of the edge whose midpoint is `coord` (PBC-wrapped). -1 if absent."""
    c = np.asarray(coord, dtype=float)
    if geo.bc == "PBC":
        c = c % np.array([geo.Lx, geo.Ly, geo.Lz], dtype=float)
    return geo._mapping3Dto1D(c)


def _in_plane_axes(plane_axis: int) -> Tuple[int, int]:
    a, b = [ax for ax in range(3) if ax != plane_axis]
    return a, b


# Plane label -> normal axis (an xy-plane has normal z=2, etc.) and its inverse.
PLANE_NORMAL = {"xy": 2, "xz": 1, "yz": 0}
NORMAL_PLANE = {v: k for k, v in PLANE_NORMAL.items()}


def _bulk_square(geo, plane_axis: int, plane_at: Optional[int] = None,
                 R: Optional[int] = None) -> Dict[str, Any]:
    """Kwargs for a centered σ^z square that fits *entirely in the bulk* of `plane_axis`.

    Centered in all three directions: ``corner`` centers the side-``R`` square per
    in-plane axis and the plane sits at the middle layer ``L//2`` (overridable via
    `plane_at`). Feeds straight into `electric_loop_edges(**_bulk_square(...))`.

    `R` controls the side:
      • ``R=None`` (default) → the **largest** bulk square, ``R = min(L_a,L_b) - 3`` (so
        vertices span the interior ``1 .. L-2`` and never touch the OBC surface). This
        *grows with L*, so each L evaluates a different-perimeter operator.
      • ``R=<int>`` → a **fixed** side at every L (e.g. ``R=1`` = one plaquette,
        perimeter 4) — the same physical loop across sizes, still centered/bulk.
    Requires ``1 <= R <= min(L_a,L_b) - 3`` (so L>=4; L<=3 has no bulk loop).
    """
    a, b = _in_plane_axes(plane_axis)
    L = (geo.Lx, geo.Ly, geo.Lz)
    Rmax = min(L[a], L[b]) - 3          # largest side that stays strictly in the bulk
    if R is None:
        R = Rmax
    if R < 1 or Rmax < 1:
        raise ValueError(
            f"bulk-centered FM loop needs L>=4 (R=min(L_a,L_b)-3); got in-plane "
            f"extents ({L[a]},{L[b]}) -> Rmax={Rmax}. Use placement='boundary' for small L.")
    if R > Rmax:
        raise ValueError(
            f"fixed FM loop R={R} leaves the bulk: needs R<=min(L_a,L_b)-3={Rmax} for "
            f"in-plane extents ({L[a]},{L[b]}). Shrink R or grow L.")
    corner = ((L[a] - 1 - R) // 2, (L[b] - 1 - R) // 2)   # centered; (1,1) for R=L-3 cubic
    pa = L[plane_axis] // 2 if plane_at is None else plane_at
    return dict(plane_axis=plane_axis, plane_at=pa, corner=corner, R=R)


def electric_loop_edges(geo, *, plane_axis: int = 2, plane_at: int = 0,
                        corner: Tuple[int, int] = (0, 0),
                        R: Optional[int] = None) -> Tuple[List[int], List[int]]:
    """Edges of an electric (σ^z) Wilson rectangle in a lattice plane.

    Returns ``(closed, open_)``:
      • ``closed`` — the 4 sides of an R×R rectangle (4R edges). Product of σ^z
        over it equals ∏ of the enclosed B_p (a contractible magnetic Wilson
        loop); =1 in the pure ground state, perimeter-law decaying with field.
      • ``open_`` — the BFFM **half-square** (2R edges): the lower U running from
        the midpoint of the left side, down and across the bottom, up to the
        midpoint of the right side. Its two ends carry the e-charges (separated
        by R). Because its length 2R is exactly half the 4R perimeter, the open
        string's perimeter law cancels √⟨closed⟩, so O_FM = ⟨open⟩/√|⟨closed⟩|
        has a finite ℓ→∞ limit (take R as large as the lattice allows / extrapolate).

    `plane_axis` is the rectangle's normal (2 = z-plane by default); `plane_at`
    is the integer coordinate of that plane; `corner` is the (a,b) base vertex.
    `R` defaults to the **largest square the box holds** (min in-plane extent − 1)
    — the ℓ→∞ order parameter is taken as this biggest available loop (we do not
    R-sweep). For odd R the U is split floor/ceil (ends differ by one row, as in
    the 2D `half_length_wilson`).
    """
    a, b = _in_plane_axes(plane_axis)
    if R is None:
        ext = (geo.Lx, geo.Ly, geo.Lz)
        R = min(ext[a], ext[b]) - 1          # biggest loop the lattice allows
    e = np.eye(3)
    x0, y0 = corner

    def vbase(ia, ib):
        v = np.zeros(3)
        v[a], v[b], v[plane_axis] = ia, ib, plane_at
        return v

    def edge(ia, ib, axis):
        return _edge(geo, vbase(ia, ib) + 0.5 * e[axis])

    bottom = [edge(x0 + i, y0,     a) for i in range(R)]
    top    = [edge(x0 + i, y0 + R, a) for i in range(R)]
    left   = [edge(x0,     y0 + j, b) for j in range(R)]
    right  = [edge(x0 + R, y0 + j, b) for j in range(R)]
    closed = bottom + top + left + right             # full square, 4R edges

    # BFFM open string = HALF the square (2R edges): lower-left half + bottom +
    # lower-right half. hL + hR = R, so |open| = R + R = 2R exactly.
    hL, hR = R // 2, R - R // 2
    left_low  = [edge(x0,     y0 + j, b) for j in range(hL)]
    right_low = [edge(x0 + R, y0 + j, b) for j in range(hR)]
    open_ = left_low + bottom + right_low
    if -1 in closed:
        raise ValueError("electric loop runs off the lattice — shrink R/corner "
                         "or move plane_at into the bulk")
    return closed, open_


def magnetic_membrane_edges(geo, *, normal: int = 2, plane_at: int = 0,
                            cut_at: Optional[int] = None
                            ) -> Tuple[List[int], List[int]]:
    """Edges of a magnetic (σ^x) membrane normal to axis `normal` — the BFFM dual
    of the electric half-square (Option A).

    σ^x acts on the **`normal`-axis edges** at height ``plane_at+½``. Returns
    ``(closed, open_)``:
      • ``closed`` — the **full** σ^x sheet spanning the box. On OBC it equals
        ∏ A_v over the slab beneath it, so it is boundary-free (commutes with
        every B_p) and ``⟨closed⟩ = 1`` in the pure ground state — the exact dual
        of the electric ``∏B_p`` closed loop, hence the FM normalisation.
      • ``open_`` — **half** that sheet (the columns with in-plane a-coord < cut).
        Its only bulk boundary is the straight cut at ``a = cut`` (length L_b):
        that cut is the **flux loop** the open membrane creates. Because its area
        is ½ the closed sheet, the area laws cancel and O_FM^m = ⟨open⟩/√|⟨closed⟩|
        has a finite ℓ→∞ limit (largest membrane the box holds).

    `cut_at` defaults to L_a // 2 (cut through the middle).
    """
    a, b = _in_plane_axes(normal)
    L = (geo.Lx, geo.Ly, geo.Lz)
    ha = L[a] // 2 if cut_at is None else cut_at

    def xedge(ia, ib):
        coord = np.zeros(3)
        coord[a], coord[b], coord[normal] = ia, ib, plane_at + 0.5
        return _edge(geo, coord)

    closed = [xedge(ia, ib) for ia in range(L[a]) for ib in range(L[b])]
    open_ = [xedge(ia, ib) for ia in range(ha) for ib in range(L[b])]
    if -1 in closed or -1 in open_:
        raise ValueError("magnetic membrane runs off the lattice — check "
                         "normal/plane_at (need plane_at in 0..L-2 for OBC)")
    return closed, open_


# =============================================================================
# Operators + the FM ratio (shared by both sectors)
# =============================================================================

def _pauli_product(hi, indices: Sequence[int], pauli: str):
    """∏ σ^{pauli} over `indices` as a NetKet operator (deduplicated)."""
    sigma = nk.operator.spin.sigmaz if pauli == "z" else nk.operator.spin.sigmax
    op = None
    for i in dict.fromkeys(int(j) for j in indices):     # preserve order, drop dups
        term = sigma(hi, i)
        op = term if op is None else op * term
    return op


def sector_operators(geo, hi, sector: str, **kw):
    """Build (open_op, closed_op) NetKet operators for the requested sector.

    sector="electric" → σ^z loop (kw: plane_axis, plane_at, corner, R)
    sector="magnetic" → σ^x membrane (kw: normal, plane_at, corner, R)
    """
    if sector == "electric":
        closed, open_ = electric_loop_edges(geo, **kw)
        pauli = "z"
    elif sector == "magnetic":
        closed, open_ = magnetic_membrane_edges(geo, **kw)
        pauli = "x"
    else:
        raise ValueError(f"sector must be 'electric' or 'magnetic', got {sector!r}")
    return _pauli_product(hi, open_, pauli), _pauli_product(hi, closed, pauli)


def build_loop_operators(geo, hi, sector: str, *, placement: str = "bulk",
                         planes: Sequence[str] = ("xy", "xz", "yz"),
                         plane_at: Optional[int] = None, R: Optional[int] = None,
                         op_kwargs: Optional[Dict] = None
                         ) -> Tuple[List[Tuple[str, Any, Any]], Dict[str, Any]]:
    """The (label, open_op, closed_op) list to average over, plus a placement meta dict.

    placement="bulk" (electric only): a bulk-centered square in each requested plane
    ('xy'/'xz'/'yz'); their FM ratios are averaged (see `fm_ratio_avg`). Requires L>=4.
    `R` sets the loop side (see `_bulk_square`): None = largest (L-3, grows with L),
    or a fixed int (e.g. 1 = perimeter-4 plaquette) for a size-independent operator.
    placement="boundary": the single legacy loop from `op_kwargs` (label ''), unchanged.
    """
    op_kwargs = op_kwargs or {}
    if placement == "boundary":
        open_op, closed_op = sector_operators(geo, hi, sector, **op_kwargs)
        meta = {"placement": "boundary", "planes": [], "plane_at": op_kwargs.get("plane_at"),
                "R": op_kwargs.get("R")}
        return [("", open_op, closed_op)], meta
    if placement != "bulk":
        raise ValueError(f"placement must be 'bulk' or 'boundary', got {placement!r}")
    if sector != "electric":
        raise ValueError("placement='bulk' is implemented for the electric sector only")
    pairs, kw0 = [], None
    for label in planes:
        kw = _bulk_square(geo, PLANE_NORMAL[label], plane_at=plane_at, R=R)
        kw0 = kw0 or kw
        open_op, closed_op = sector_operators(geo, hi, "electric", **kw)
        pairs.append((label, open_op, closed_op))
    meta = {"placement": "bulk", "planes": list(planes),
            "plane_at": kw0["plane_at"], "R": kw0["R"]}   # uniform for a cubic box
    return pairs, meta


def _stat_err(stat, n_samples: int) -> float:
    """Standard error of the mean, robust to NetKet's autocorrelation-corrected
    `error_of_mean` returning NaN.

    That NaN happens for a short-chain, low-cardinality *diagonal* estimator (our
    σ^z string takes values in {±1}): when a chain's samples are all-equal the
    within-chain variance is 0 and the autocorrelation/split-R̂ normalisation is
    0/0. We fall back to the plain sqrt(variance / n_samples), which ignores the
    autocorrelation time and is therefore *mildly optimistic* — flag it as such —
    but finite. Best paired with long chains (few `n_chains`) so the primary,
    autocorrelation-aware estimate is the one that's actually used.
    """
    e = float(np.real(stat.error_of_mean))
    if np.isfinite(e):
        return e
    var = float(np.real(getattr(stat, "variance", np.nan)))
    if np.isfinite(var) and n_samples > 0:
        return float(np.sqrt(var / n_samples))
    return float("nan")


def fm_ratio(vstate, open_op, closed_op) -> Tuple[float, float]:
    """Fredenhagen–Marcu ratio O = ⟨S_open⟩/√|⟨W_closed⟩|, with propagated error.

    Both expectations are sampled from the same variational state. The error is
    first-order propagation through O(S,W) = S·|W|^(-1/2):
        σ_O² = (∂O/∂S σ_S)² + (∂O/∂W σ_W)²,
        ∂O/∂S = |W|^(-1/2),  ∂O/∂W = -½ S |W|^(-3/2).
    Per-expectation errors go through `_stat_err` (NetKet `.error_of_mean`, with a
    variance-based fallback so a near-constant chain can't NaN out the whole point).
    """
    n = int(getattr(vstate, "n_samples", 0) or 0)
    S = vstate.expect(open_op)
    W = vstate.expect(closed_op)
    Sm, Se = float(np.real(S.mean)), _stat_err(S, n)
    Wm, We = float(np.real(W.mean)), _stat_err(W, n)
    denom = np.sqrt(abs(Wm))
    if denom == 0.0:
        return float("nan"), float("nan")
    O = Sm / denom
    dO_dS = 1.0 / denom
    dO_dW = -0.5 * Sm / abs(Wm) ** 1.5
    Oe = float(np.hypot(dO_dS * Se, dO_dW * We))
    return O, Oe


def fm_ratio_avg(vstate, pairs: Sequence[Tuple[str, Any, Any]]
                 ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """Mean FM ratio over several loop orientations (the xy/xz/yz bulk average).

    `pairs` = [(label, open_op, closed_op), ...]. Scores each with `fm_ratio`, then
    returns (O_mean, O_err, per_plane) where per_plane[label] = (O_i, e_i). The error is
    the propagated MC error of the mean, sqrt(Σ e_i²)/N, treating the orientations as
    independent — they share samples, so the per-plane spread (inspect per_plane) is the
    honest anisotropy check.
    """
    per = {}
    for label, open_op, closed_op in pairs:
        per[label] = fm_ratio(vstate, open_op, closed_op)
    Os = np.array([o for o, _ in per.values()], float)
    Oes = np.array([e for _, e in per.values()], float)
    O_mean = float(np.mean(Os))
    O_err = float(np.sqrt(np.sum(Oes ** 2)) / len(Oes))
    return O_mean, O_err, per


# =============================================================================
# Checkpoint loader + grid sweep
# =============================================================================

def _weights_path(json_path: str) -> str:
    """Sibling `.mpack` for a `train.py` artifact, falling back to `.ckpt.mpack`
    (the periodic checkpoint weights) for a run that timed out before its final."""
    if json_path.endswith(".curve.json"):      # checkpoint: {name}.curve.json -> base {name}
        base = json_path[:-len(".curve.json")]
    elif json_path.endswith(".json"):
        base = json_path[:-len(".json")]
    else:
        base = json_path
    mpack = base + ".mpack"
    if not os.path.exists(mpack):
        alt = base + ".ckpt.mpack"
        if not os.path.exists(alt):
            raise FileNotFoundError(
                f"no weights for {json_path}: tried {base}.mpack and {base}.ckpt.mpack")
        mpack = alt
    return mpack


def _load_weights(vs, json_path: str):
    """Deserialize the checkpoint weights into `vs`'s structure (returns the new vs).

    Same network/sampler structure -> reusing one `vs` template across an hz sweep
    keeps JAX's compiled `expect` warm; only the parameters change per checkpoint.
    """
    with open(_weights_path(json_path), "rb") as f:
        return flax.serialization.from_bytes(vs, f.read())


def _struct_sig(cfg: Dict[str, Any]) -> str:
    """Signature of everything that fixes the network/sampler/state *shape* (all the
    build_state inputs except hz and n_samples). Checkpoints in one hz sweep share
    it, so they can reuse a single built `vs`; a mismatch forces a fresh rebuild."""
    keys = ("L", "bc", "model", "arch", "hidden", "noninv_channels", "n_noninv",
            "inv_hidden", "cnn_hidden", "kernel_size", "radius_edge", "radius_plaq",
            "n_chains", "n_sweeps", "n_discard", "chunk_size", "vanilla_depth",
            "noninv_identity")
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)


def load_vstate(json_path: str, *, eval_samples: Optional[int] = None,
                eval_chains: Optional[int] = None, seed: Optional[int] = None):
    """Rebuild and reload a trained NQS from a `train.py` artifact pair.

    Reads `{json_path}` (config + observables), rebuilds the exact VMC stack via
    `builders.build_state(config)` (H skipped — FM extraction never uses it), then
    loads the sibling `.mpack` weights. `eval_samples` overrides n_samples for a
    more precise expectation; `eval_chains` overrides n_chains — GPU runs default to
    n_chains=1024, i.e. only ~8 samples/chain at eval, too short to estimate the
    autocorrelation time (→ NaN `error_of_mean`); a small value (e.g. 16) makes long
    chains so the primary error estimate is valid. `seed` re-seeds the sampler.
    Weights are sampler-shape-independent, so both overrides reload cleanly. Returns
    (config, geo, hi, vstate).
    """
    with open(json_path) as f:
        meta = json.load(f)
    cfg = dict(meta["config"])
    if eval_samples is not None:
        cfg["n_samples"] = eval_samples
    if eval_chains is not None:
        cfg["n_chains"] = eval_chains
    if seed is not None:
        cfg["seed"] = seed
    geo, hi, _Ham, vs, _xz = build_state(cfg, build_ham=False)
    vs = _load_weights(vs, json_path)
    return cfg, geo, hi, vs


def _matches(cfg: Dict[str, Any], L, hx, model, bc) -> bool:
    def eq(a, b):
        return b is None or (a is not None and abs(float(a) - float(b)) < 1e-9)
    if L is not None and int(cfg.get("L", -1)) != int(L):
        return False
    if model is not None and cfg.get("model", "bosonic") != model:
        return False
    if bc is not None and cfg.get("bc", "PBC") != bc:
        return False
    return eq(cfg.get("hx"), hx)


def fm_sweep(checkpoint_dir: str, *, sector: str = "electric", field: str = "hz",
             L: Optional[int] = None, hx: Optional[float] = None,
             model: str = "bosonic", bc: Optional[str] = None,
             eval_samples: int = 8192, eval_chains: Optional[int] = None,
             op_kwargs: Optional[Dict] = None,
             placement: str = "bulk", planes: Sequence[str] = ("xy", "xz", "yz"),
             plane_at: Optional[int] = None, R: Optional[int] = None,
             verbose: bool = True) -> Dict[str, np.ndarray]:
    """Score every matching checkpoint in `checkpoint_dir`, sorted by `field`.

    Selects `{*.json}` whose config matches (L, hx, model, bc) and sweeps the
    swept parameter `field` (default "hz"). For each it loads the NQS, builds the
    loop operators once, and evaluates the FM ratio plus ⟨σz⟩ (a cheap diagonal
    cross-check whose susceptibility should peak at the same h_c).

    placement="bulk" (default, electric only): the largest bulk-centered square in each
    plane in `planes`, averaged over orientations (needs L>=4). placement="boundary":
    the single legacy loop from `op_kwargs` (works at any L; reproduces old curves).

    Returns a dict of equal-length arrays: field, O, Oe, mz, mz_e, name; for bulk
    placement also O_<plane>/Oe_<plane> per orientation, plus a non-array "_meta" entry
    (placement, planes, plane_at, R).

    All checkpoints in one hz sweep share the same network/sampler/operators (only
    the weights differ), so the stack and the loop operators are built **once** and
    reused: each subsequent checkpoint only swaps in its weights, which keeps JAX's
    compiled `expect` warm. A checkpoint whose structural config differs (`_struct_sig`)
    triggers a one-off rebuild rather than corrupting reuse.
    """
    op_kwargs = op_kwargs or {}
    # One entry per run: prefer the final {name}.json; fall back to the latest
    # checkpoint {name}.curve.json (+ {name}.ckpt.mpack) for a run that timed out
    # before writing its final artifact.
    by_base = {}
    for jp in sorted(glob.glob(os.path.join(checkpoint_dir, "*.json"))):
        if jp.endswith(".curve.json"):
            base, final = jp[:-len(".curve.json")], False
        else:
            base, final = jp[:-len(".json")], True
        if final or base not in by_base:
            by_base[base] = jp

    tmpl_sig = tmpl = None       # (geo, hi, vs, pairs, mz_op, meta)
    sweep_meta: Dict[str, Any] = {}
    rows = []
    for jp in sorted(by_base.values()):
        try:
            with open(jp) as f:
                doc = json.load(f)
            cfg0 = doc.get("config", {})
        except (json.JSONDecodeError, KeyError):
            continue
        if not cfg0 or not _matches(cfg0, L, hx, model, bc):
            continue
        if doc.get("diverged"):            # self-healing guard gave up -> garbage state
            print(f"  [skip] {os.path.basename(jp)}: diverged:true — excluded "
                  f"from the sweep", flush=True)
            continue
        _vs = doc.get("observables", {}).get("Vscore")           # guard-missed blow-up
        if isinstance(_vs, (int, float)) and np.isfinite(_vs) and _vs > VSCORE_MAX:
            print(f"  [skip] {os.path.basename(jp)}: Vscore={_vs:.2e} > {VSCORE_MAX} "
                  f"— variance blow-up, excluded from the sweep", flush=True)
            continue
        if jp.endswith(".curve.json") and verbose:
            with open(jp) as f:
                _done = json.load(f).get("completed_steps", "?")
            print(f"  [checkpoint] {os.path.basename(jp)}: run unfinished "
                  f"({_done} steps) — using latest .ckpt.mpack weights", flush=True)
        t0 = time.perf_counter()
        sig = _struct_sig(cfg0)
        if tmpl is None or sig != tmpl_sig:            # first match, or a shape change
            _cfg, geo, hi, vs = load_vstate(jp, eval_samples=eval_samples,
                                            eval_chains=eval_chains)
            pairs, sweep_meta = build_loop_operators(
                geo, hi, sector, placement=placement, planes=planes,
                plane_at=plane_at, R=R, op_kwargs=op_kwargs)
            mz_op = sum(nk.operator.spin.sigmaz(hi, i) for i in range(geo.N)) / geo.N
            tmpl_sig, tmpl = sig, (geo, hi, vs, pairs, mz_op, sweep_meta)
        else:                                          # reuse: swap weights only
            geo, hi, _vs, pairs, mz_op, sweep_meta = tmpl
            vs = _load_weights(_vs, jp)
        vs.reset()                                     # fresh samples for these weights
        O, Oe, per = fm_ratio_avg(vs, pairs)
        mz = vs.expect(mz_op)
        row = {
            "field": float(cfg0[field]), "O": O, "Oe": Oe,
            "mz": float(np.real(mz.mean)), "mz_e": float(np.real(mz.error_of_mean)),
            "name": cfg0.get("name", os.path.basename(jp)[:-5]),
        }
        for lbl, (Oi, Oei) in per.items():             # per-orientation cols (bulk only)
            if lbl:
                row[f"O_{lbl}"], row[f"Oe_{lbl}"] = Oi, Oei
        rows.append(row)
        if verbose:
            spread = ("  planes={" +
                      ", ".join(f"{l}:{per[l][0]:.3f}" for l in per if l) + "}"
                      if len(per) > 1 else "")
            print(f"  {rows[-1]['name']}: {field}={rows[-1]['field']:.4g}  "
                  f"O_FM={O:.4f}±{Oe:.4f}  <σz>={rows[-1]['mz']:.4f}{spread}  "
                  f"[{time.perf_counter() - t0:.1f}s]", flush=True)
    if not rows:
        raise ValueError(f"no checkpoints in {checkpoint_dir} match "
                         f"(L={L}, hx={hx}, model={model}, bc={bc})")
    rows.sort(key=lambda r: r["field"])
    out = {k: np.array([r[k] for r in rows],
                       dtype=object if k == "name" else float)
           for k in rows[0]}
    out["_meta"] = sweep_meta
    return out


# =============================================================================
# Per-L analysis: logistic fit + derivative peak
# =============================================================================

def _logistic(h, a, b, h0, w):
    return a + b / (1.0 + np.exp(-(h - h0) / w))


def fit_transition(field: np.ndarray, O: np.ndarray,
                   Oe: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Locate h_c for one L: logistic fit (h_c = inflection) + derivative peak.

    The FM order parameter rises monotonically through the transition, so a
    logistic a + b/(1+e^{-(h-h0)/w}) captures it; its inflection h0 — which is
    also the peak of the analytic derivative — is the pseudo-critical h_c(L).
    A finite-difference derivative peak is returned as a model-free cross-check.

    Returns: h_c (=h0), width w, popt, a finely-sampled (h, O_fit, dO_fit) curve,
    and the finite-difference (h_mid, dOdh, h_c_fd).
    """
    from scipy.optimize import curve_fit

    field = np.asarray(field, float)
    O = np.asarray(O, float)
    p0 = [O[0], O[-1] - O[0], float(np.median(field)),
          0.1 * (field[-1] - field[0]) or 0.1]
    kw = {}
    if Oe is not None and np.all(np.asarray(Oe) > 0):
        kw = dict(sigma=np.asarray(Oe, float), absolute_sigma=True)
    try:
        popt, pcov = curve_fit(_logistic, field, O, p0=p0, maxfev=20000, **kw)
        h0_err = float(np.sqrt(abs(pcov[2, 2])))
    except Exception as exc:                      # fall back to derivative peak only
        popt, h0_err = None, float("nan")
        print(f"[fit_transition] logistic fit failed ({exc}); FD peak only")

    # finite-difference derivative (model-free)
    h_mid = 0.5 * (field[1:] + field[:-1])
    dOdh = np.diff(O) / np.diff(field)
    h_c_fd = float(h_mid[int(np.argmax(np.abs(dOdh)))]) if len(h_mid) else float("nan")

    out: Dict[str, Any] = {"h_c_fd": h_c_fd, "fd": (h_mid, dOdh)}
    if popt is not None:
        hh = np.linspace(field[0], field[-1], 400)
        a, b, h0, w = popt
        dO = (b / w) * np.exp(-(hh - h0) / w) / (1 + np.exp(-(hh - h0) / w)) ** 2
        out.update(h_c=float(h0), h_c_err=h0_err, width=float(w), popt=popt,
                   curve=(hh, _logistic(hh, *popt), dO))
    else:
        out.update(h_c=h_c_fd, h_c_err=float("nan"), width=float("nan"),
                   popt=None, curve=None)
    return out


def plot_fm_sweep(field, O, Oe, fit, *, sector="electric", L=None, ax=None):
    """Two-panel plot: O_FM(field) with the logistic fit, and dO/dfield with h_c.

    Reusable but import-light: matplotlib is imported here so the numerics above
    stay usable without a display.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    label = f"{sector} FM" + (f", L={L}" if L is not None else "")

    ax[0].errorbar(field, O, yerr=Oe, fmt="o", capsize=3, label="data")
    if fit.get("curve") is not None:
        hh, Ofit, _ = fit["curve"]
        ax[0].plot(hh, Ofit, "-", label="logistic fit")
    ax[0].axvline(fit["h_c"], ls="--", c="k", label=f"h_c={fit['h_c']:.3f}")
    ax[0].set(xlabel="field", ylabel="$O_{FM}$", title=label)
    ax[0].legend()

    h_mid, dOdh = fit["fd"]
    ax[1].plot(h_mid, dOdh, "s-", label="finite diff")
    if fit.get("curve") is not None:
        hh, _, dO = fit["curve"]
        ax[1].plot(hh, dO, "-", label="d(fit)")
    ax[1].axvline(fit["h_c"], ls="--", c="k")
    ax[1].axvline(fit["h_c_fd"], ls=":", c="r", label=f"FD peak={fit['h_c_fd']:.3f}")
    ax[1].set(xlabel="field", ylabel="$dO_{FM}/d$field", title="derivative")
    ax[1].legend()
    return ax


# =============================================================================
# CLI: extract one L's O_FM(field) curve + transition fit to a compact JSON.
#
# Runs on the cluster (a GPU node, where the checkpoints + NetKet live): it is
# the ONLY on-cluster analysis step. The tiny output JSON — arrays + fit only,
# no weights — is what gets pulled local for the multi-L overlay plot
# (analysis/plot_phase_diagram.py, which needs no NetKet).
#
#   python -m Three_TC.fm --dir $PSCRATCH/tc_nqs/phase_hx0.2/L6 --L 6 --hx 0.2 \
#       --placement bulk --out $PSCRATCH/tc_nqs/phase_hx0.2/fm_L6_bulk.json
# =============================================================================

def extract_curve(checkpoint_dir, *, L, hx, sector="electric", field="hz",
                  model="bosonic", bc="OBC", eval_samples=8192, eval_chains=None,
                  placement="bulk", planes=("xy", "xz", "yz"), plane_at=None, R=None):
    """fm_sweep + fit_transition for one L -> a JSON-serializable dict.

    `R` = loop side for bulk placement: None → largest (L-3, grows with L); an int
    → fixed (R=1 is a perimeter-4 plaquette, the same operator at every L).
    `eval_chains` overrides n_chains at eval (small = long chains = valid error_of_mean).
    """
    res = fm_sweep(checkpoint_dir, sector=sector, field=field, L=L, hx=hx,
                   model=model, bc=bc, eval_samples=eval_samples, eval_chains=eval_chains,
                   placement=placement, planes=planes, plane_at=plane_at, R=R)
    fit = fit_transition(res["field"], res["O"], res["Oe"])
    meta = res.get("_meta", {})
    rec = {
        "L": int(L), "hx": float(hx), "sector": sector, "field_name": field,
        "bc": bc, "model": model, "eval_samples": int(eval_samples),
        "placement": meta.get("placement", placement),
        "planes": meta.get("planes", []), "plane_at": _num(meta.get("plane_at")),
        "R": (None if meta.get("R") is None else int(meta["R"])),
        "field": res["field"].tolist(), "O": res["O"].tolist(),
        "Oe": res["Oe"].tolist(), "mz": res["mz"].tolist(),
        "mz_e": res["mz_e"].tolist(), "names": [str(x) for x in res["name"]],
        "h_c": _num(fit.get("h_c")), "h_c_err": _num(fit.get("h_c_err")),
        "h_c_fd": _num(fit.get("h_c_fd")), "width": _num(fit.get("width")),
    }
    # Per-orientation curves (populated only for bulk placement) — the isotropy check.
    o_planes = {lbl: res[f"O_{lbl}"].tolist() for lbl in meta.get("planes", [])
                if f"O_{lbl}" in res}
    if o_planes:
        rec["O_planes"] = o_planes
        rec["Oe_planes"] = {lbl: res[f"Oe_{lbl}"].tolist() for lbl in o_planes}
    hm, dodh = fit["fd"]
    rec["fd"] = {"h_mid": np.asarray(hm).tolist(), "dOdh": np.asarray(dodh).tolist()}
    if fit.get("curve") is not None:
        hh, ofit, dO = fit["curve"]
        rec["fit_curve"] = {"h": hh.tolist(), "O_fit": ofit.tolist(), "dO": dO.tolist()}
    return rec


def _num(x):
    """None-safe float (JSON can't hold numpy scalars / NaN survives as null-ish)."""
    return None if x is None else float(x)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Extract O_FM(field) + transition fit for one L.")
    p.add_argument("--dir", required=True, help="checkpoint dir ({name}.json + .mpack)")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--hx", type=float, required=True)
    p.add_argument("--sector", default="electric", choices=["electric", "magnetic"])
    p.add_argument("--field", default="hz", help="swept parameter (hz for electric)")
    p.add_argument("--bc", default="OBC", choices=["OBC", "PBC"])
    p.add_argument("--model", default="bosonic", choices=["bosonic", "fermionic"])
    p.add_argument("--eval_samples", type=int, default=8192)
    p.add_argument("--eval_chains", type=int, default=None,
                   help="override n_chains at eval (default: keep the run's value). "
                        "GPU runs default to 1024 -> ~8 samples/chain, too short for a "
                        "valid autocorrelation error; set e.g. 16 for long chains.")
    p.add_argument("--placement", default="bulk", choices=["bulk", "boundary"],
                   help="bulk: largest bulk-centered square, averaged over --planes "
                        "(electric, needs L>=4); boundary: legacy z=0 largest loop (any L)")
    p.add_argument("--planes", default="xy,xz,yz",
                   help="comma-separated planes to average for bulk placement")
    p.add_argument("--plane_at", type=int, default=None,
                   help="loop plane index (default: middle layer L//2); bulk only")
    p.add_argument("--R", type=int, default=None,
                   help="bulk loop side: default None = largest (L-3, grows with L); "
                        "fix it (e.g. --R 1 = perimeter-4 plaquette) for a size-"
                        "independent order parameter. Needs 1<=R<=L-3.")
    p.add_argument("--out", required=True, help="output JSON path")
    a = p.parse_args(argv)
    planes = tuple(s.strip() for s in a.planes.split(",") if s.strip())
    rec = extract_curve(a.dir, L=a.L, hx=a.hx, sector=a.sector, field=a.field,
                        model=a.model, bc=a.bc, eval_samples=a.eval_samples,
                        eval_chains=a.eval_chains, placement=a.placement, planes=planes,
                        plane_at=a.plane_at, R=a.R)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[fm] L={a.L} hx={a.hx} placement={rec['placement']} R={rec['R']} "
          f"planes={rec['planes']}: {len(rec['field'])} points, "
          f"h_c={rec['h_c']}  h_c_fd={rec['h_c_fd']}  ->  {a.out}")


if __name__ == "__main__":
    main()
