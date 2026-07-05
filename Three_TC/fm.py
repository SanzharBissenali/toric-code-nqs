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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import netket as nk
import flax

from Three_TC.builders import build_state


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


def fm_ratio(vstate, open_op, closed_op) -> Tuple[float, float]:
    """Fredenhagen–Marcu ratio O = ⟨S_open⟩/√|⟨W_closed⟩|, with propagated error.

    Both expectations are sampled from the same variational state. The error is
    first-order propagation through O(S,W) = S·|W|^(-1/2):
        σ_O² = (∂O/∂S σ_S)² + (∂O/∂W σ_W)²,
        ∂O/∂S = |W|^(-1/2),  ∂O/∂W = -½ S |W|^(-3/2).
    (NetKet's `.error_of_mean` is already the standard error of the mean.)
    """
    S = vstate.expect(open_op)
    W = vstate.expect(closed_op)
    Sm, Se = float(np.real(S.mean)), float(np.real(S.error_of_mean))
    Wm, We = float(np.real(W.mean)), float(np.real(W.error_of_mean))
    denom = np.sqrt(abs(Wm))
    if denom == 0.0:
        return float("nan"), float("nan")
    O = Sm / denom
    dO_dS = 1.0 / denom
    dO_dW = -0.5 * Sm / abs(Wm) ** 1.5
    Oe = float(np.hypot(dO_dS * Se, dO_dW * We))
    return O, Oe


# =============================================================================
# Checkpoint loader + grid sweep
# =============================================================================

def load_vstate(json_path: str, *, eval_samples: Optional[int] = None,
                seed: Optional[int] = None):
    """Rebuild and reload a trained NQS from a `train.py` artifact pair.

    Reads `{json_path}` (config + observables), rebuilds the exact VMC stack via
    `builders.build_state(config)`, then loads the sibling `.mpack` weights.
    `eval_samples` overrides n_samples for a more precise expectation; `seed`
    re-seeds the sampler. Returns (config, geo, hi, vstate).
    """
    with open(json_path) as f:
        meta = json.load(f)
    cfg = dict(meta["config"])
    if eval_samples is not None:
        cfg["n_samples"] = eval_samples
    if seed is not None:
        cfg["seed"] = seed
    geo, hi, _Ham, vs, _xz = build_state(cfg)
    if json_path.endswith(".curve.json"):     # checkpoint: {name}.curve.json -> base {name}
        base = json_path[:-len(".curve.json")]
    elif json_path.endswith(".json"):
        base = json_path[:-len(".json")]
    else:
        base = json_path
    mpack = base + ".mpack"
    if not os.path.exists(mpack):
        alt = base + ".ckpt.mpack"           # fall back to the periodic checkpoint weights
        if not os.path.exists(alt):
            raise FileNotFoundError(
                f"no weights for {json_path}: tried {base}.mpack and {base}.ckpt.mpack")
        mpack = alt
    with open(mpack, "rb") as f:
        vs = flax.serialization.from_bytes(vs, f.read())
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
             eval_samples: int = 8192, op_kwargs: Optional[Dict] = None,
             verbose: bool = True) -> Dict[str, np.ndarray]:
    """Score every matching checkpoint in `checkpoint_dir`, sorted by `field`.

    Selects `{*.json}` whose config matches (L, hx, model, bc) and sweeps the
    swept parameter `field` (default "hz"). For each it loads the NQS, builds the
    sector operators once, and evaluates the FM ratio plus ⟨σz⟩ (a cheap
    diagonal cross-check whose susceptibility should peak at the same h_c).

    Returns a dict of equal-length arrays: field, O, Oe, mz, mz_e, name.
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
    rows = []
    for jp in sorted(by_base.values()):
        try:
            with open(jp) as f:
                cfg0 = json.load(f).get("config", {})
        except (json.JSONDecodeError, KeyError):
            continue
        if not cfg0 or not _matches(cfg0, L, hx, model, bc):
            continue
        if jp.endswith(".curve.json") and verbose:
            with open(jp) as f:
                _done = json.load(f).get("completed_steps", "?")
            print(f"  [checkpoint] {os.path.basename(jp)}: run unfinished "
                  f"({_done} steps) — using latest .ckpt.mpack weights")
        cfg, geo, hi, vs = load_vstate(jp, eval_samples=eval_samples)
        open_op, closed_op = sector_operators(geo, hi, sector, **op_kwargs)
        O, Oe = fm_ratio(vs, open_op, closed_op)
        mz = vs.expect(sum(nk.operator.spin.sigmaz(hi, i) for i in range(geo.N)) / geo.N)
        rows.append({
            "field": float(cfg[field]), "O": O, "Oe": Oe,
            "mz": float(np.real(mz.mean)), "mz_e": float(np.real(mz.error_of_mean)),
            "name": cfg.get("name", os.path.basename(jp)[:-5]),
        })
        if verbose:
            print(f"  {rows[-1]['name']}: {field}={rows[-1]['field']:.4g}  "
                  f"O_FM={O:.4f}±{Oe:.4f}  <σz>={rows[-1]['mz']:.4f}")
    if not rows:
        raise ValueError(f"no checkpoints in {checkpoint_dir} match "
                         f"(L={L}, hx={hx}, model={model}, bc={bc})")
    rows.sort(key=lambda r: r["field"])
    return {k: np.array([r[k] for r in rows],
                        dtype=object if k == "name" else float)
            for k in rows[0]}


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
#       --out $PSCRATCH/tc_nqs/phase_hx0.2/fm_L6_hx0.2.json
# =============================================================================

def extract_curve(checkpoint_dir, *, L, hx, sector="electric", field="hz",
                  model="bosonic", bc="OBC", eval_samples=8192):
    """fm_sweep + fit_transition for one L -> a JSON-serializable dict."""
    res = fm_sweep(checkpoint_dir, sector=sector, field=field, L=L, hx=hx,
                   model=model, bc=bc, eval_samples=eval_samples)
    fit = fit_transition(res["field"], res["O"], res["Oe"])
    rec = {
        "L": int(L), "hx": float(hx), "sector": sector, "field_name": field,
        "bc": bc, "model": model, "eval_samples": int(eval_samples),
        "field": res["field"].tolist(), "O": res["O"].tolist(),
        "Oe": res["Oe"].tolist(), "mz": res["mz"].tolist(),
        "mz_e": res["mz_e"].tolist(), "names": [str(x) for x in res["name"]],
        "h_c": _num(fit.get("h_c")), "h_c_err": _num(fit.get("h_c_err")),
        "h_c_fd": _num(fit.get("h_c_fd")), "width": _num(fit.get("width")),
    }
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
    p.add_argument("--out", required=True, help="output JSON path")
    a = p.parse_args(argv)
    rec = extract_curve(a.dir, L=a.L, hx=a.hx, sector=a.sector, field=a.field,
                        model=a.model, bc=a.bc, eval_samples=a.eval_samples)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[fm] L={a.L} hx={a.hx}: {len(rec['field'])} points, "
          f"h_c={rec['h_c']}  h_c_fd={rec['h_c_fd']}  ->  {a.out}")


if __name__ == "__main__":
    main()
