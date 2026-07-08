"""
analysis/fm_2d.py
─────────────────────────────────────────────────────────────────────────────
2D electric Fredenhagen–Marcu (FM) string order parameter from *trained* 2D NQS.

The 2D analogue of the electric sector in `Three_TC/fm.py`, specialised to
`model.geometry.ToricCodeGeometry` (qubits on edges at half-integer coords).

Construction (σ^z / "electric" sector — the one that detects the hz-driven
e-condensation / topological→polarized transition):

  • closed loop  W = ∏ σ^z over the boundary of an R×R block of plaquettes.
    ∏ σ^z around that rectangle = ∏ of the enclosed B_p, so W = 1 in the pure
    ground state and decays with a perimeter law once hz turns on.
  • open string  S = ∏ σ^z over HALF that rectangle (2R edges, one corner to the
    diagonally-opposite corner). Its two endpoints are vertices, so S creates a
    pair of e-charges.

    O_FM = ⟨S⟩ / √|⟨W⟩|

  The open string's perimeter law is exactly half the closed loop's, so √|W|
  cancels it and O_FM has a finite ℓ→∞ limit: finite in the deconfined
  (topological) phase, → 0 once e condenses. The collapse/crossing of O_FM(L)
  across the hz grid locates h_c.

Bulk placement (see `electric_string_edges`): the block is centred, so the loop
sits in the bulk with a fixed 1-plaquette margin from the open boundary. Default
side R = L-3 (the largest bulk loop that keeps that margin); requires L>=4.

Reuses `fm_ratio` (error-propagated ratio) and `fit_transition` (logistic
inflection + finite-difference derivative peak) from `Three_TC.fm` — both are
pure numerics, geometry-agnostic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import netket as nk

from model.builders_2d import build_state, with_defaults
from utils.io import load_weights
from Three_TC.fm import fm_ratio, fit_transition, _pauli_product


# =============================================================================
# Geometry: σ^z Wilson rectangle + half-string, bulk-centred
# =============================================================================

def _edge(geo, x: float, y: float) -> int:
    """Qubit index of the edge at coordinate (x, y); raises if it is off-lattice."""
    hit = geo._mapping2Dto1D(geo.arr_coord, np.array([x, y]))
    if len(hit) == 0:
        raise ValueError(f"no qubit at ({x}, {y}) — edge runs off the lattice")
    return int(hit[0][0])


def bulk_string_R(L: int, R: Optional[int] = None) -> int:
    """Default loop side: the largest bulk-centred square with a 1-plaquette margin.

    Plaquettes are indexed 0..L-2 (an (L-1)-wide strip). A centred R×R block leaves
    a margin (L-1-R)/2 on each side; R = L-3 makes that margin exactly 1 plaquette.
    """
    R = (L - 3) if R is None else R
    if R < 1:
        raise ValueError(f"electric FM string needs R>=1 (bulk loop) → L>=4; got L={L}, R={R}")
    if (L - 1 - R) < 2 or (L - 1 - R) % 2 != 0:
        # margin must be a positive integer on both sides (keeps the loop off the boundary)
        raise ValueError(
            f"R={R} is not a centred bulk loop for L={L} "
            f"(need L-1-R even and >=2, i.e. R in {{{', '.join(str(r) for r in range(1, L-2, 2))}}})")
    return R


def electric_string_edges(geo, *, R: Optional[int] = None, half: str = "br"
                          ) -> Tuple[List[int], List[int]]:
    """(closed, open) edge-index lists for a bulk-centred σ^z Wilson rectangle.

    R      : side of the R×R plaquette block (default `bulk_string_R`).
    half   : which half is the open string — "br" (bottom+right) or "tl" (top+left);
             both share the same closed loop and the same endpoints, so averaging
             their FM ratios is a cheap variance reducer (see `fm_sweep_2d`).
    """
    L = geo.Lx
    if geo.Lx != geo.Ly:
        raise NotImplementedError("electric_string_edges assumes Lx == Ly")
    R = bulk_string_R(L, R)
    a0 = b0 = (L - 1 - R) // 2                 # lower-left corner vertex of the block

    def hedge(k, y):                            # horizontal edge, k-th along a row
        return _edge(geo, a0 + 0.5 + k, y)

    def vedge(x, k):                            # vertical edge, k-th along a column
        return _edge(geo, x, b0 + 0.5 + k)

    bottom = [hedge(k, b0)       for k in range(R)]
    top    = [hedge(k, b0 + R)   for k in range(R)]
    left   = [vedge(a0,     k)   for k in range(R)]
    right  = [vedge(a0 + R, k)   for k in range(R)]

    closed = bottom + right + top + left        # full perimeter (4R edges)
    open_  = (bottom + right) if half == "br" else (top + left)   # 2R edges, corner→corner
    return closed, open_


def build_string_operators(geo, hi, *, R: Optional[int] = None
                           ) -> List[Tuple[str, Any, Any]]:
    """[(label, open_op, closed_op), ...] for the two diagonal halves of one bulk loop."""
    pairs = []
    for half in ("br", "tl"):
        closed, open_ = electric_string_edges(geo, R=R, half=half)
        pairs.append((half,
                      _pauli_product(hi, open_, "z"),
                      _pauli_product(hi, closed, "z")))
    return pairs


def fm_ratio_avg(vstate, pairs: Sequence[Tuple[str, Any, Any]]
                 ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """Mean FM ratio over the supplied (label, open, closed) pairs, with per-label detail."""
    per = {lbl: fm_ratio(vstate, o, c) for lbl, o, c in pairs}
    Os = np.array([o for o, _ in per.values()], float)
    Oes = np.array([e for _, e in per.values()], float)
    return float(np.mean(Os)), float(np.sqrt(np.sum(Oes ** 2)) / len(Oes)), per


# =============================================================================
# Checkpoint loading (2D artifacts from model/train_2d.py)
# =============================================================================

def _weights_base(json_path: str) -> str:
    """Return the weights base ({name}) for a train_2d artifact, preferring the final
    `.mpack` and falling back to the periodic `.ckpt.mpack` for a timed-out run."""
    base = json_path[:-len(".json")]
    if os.path.exists(base + ".mpack"):
        return base
    if os.path.exists(base + ".ckpt.mpack"):
        return base + ".ckpt"
    raise FileNotFoundError(f"no weights for {json_path}: tried {base}.mpack / {base}.ckpt.mpack")


def load_vstate_2d(json_path: str, *, eval_samples: Optional[int] = None):
    """Rebuild (cfg, geo, hi, vs) from a train_2d artifact and load its weights."""
    with open(json_path) as f:
        cfg = json.load(f)["config"]
    if eval_samples is not None:
        cfg = {**cfg, "n_samples": int(eval_samples)}
    cfg = with_defaults(cfg)
    geo, hi, _Ham, vs = build_state(cfg)
    vs = load_weights(vs, _weights_base(json_path))
    return cfg, geo, hi, vs


def _matches(cfg: Dict[str, Any], L, hx, arch, bc) -> bool:
    def eq(a, b):
        return a is None or abs(float(cfg.get(a, np.nan)) - float(b)) < 1e-9
    return (int(cfg.get("L", -1)) == L
            and eq("hx", hx)
            and (arch is None or cfg.get("arch") == arch)
            and (bc is None or cfg.get("bc") == bc))


# =============================================================================
# Grid sweep over trained checkpoints
# =============================================================================

def fm_sweep_2d(checkpoint_dir: str, *, L: int, hx: float = 0.0, arch: str = "Combo",
                bc: str = "OBC", field: str = "hz", R: Optional[int] = None,
                eval_samples: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
    """FM string O_FM (avg over both halves) + ⟨σz⟩ per swept-field value.

    Scans `checkpoint_dir` for train_2d `*.json` artifacts matching (L, hx, arch, bc),
    builds the bulk σ^z string operators once, and evaluates each checkpoint by
    swapping in its weights. Returns arrays keyed field / O / Oe / mz / mz_e / name.
    """
    jsons = sorted(p for p in glob.glob(os.path.join(checkpoint_dir, "*.json"))
                   if not p.endswith(".curve.json"))
    tmpl = None                                    # (geo, hi, vs, pairs, mz_op)
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    for jp in jsons:
        try:
            with open(jp) as f:
                cfg0 = json.load(f).get("config", {})
        except (json.JSONDecodeError, KeyError):
            continue
        if not cfg0 or not _matches(cfg0, L, hx, arch, bc):
            continue
        t0 = time.perf_counter()
        if tmpl is None:
            _cfg, geo, hi, vs = load_vstate_2d(jp, eval_samples=eval_samples)
            pairs = build_string_operators(geo, hi, R=R)
            mz_op = sum(nk.operator.spin.sigmaz(hi, i) for i in range(geo.N)) / geo.N
            R_used = bulk_string_R(geo.Lx, R)
            meta = {"L": L, "hx": hx, "arch": arch, "bc": bc,
                    "R": R_used, "margin": (geo.Lx - 1 - R_used) // 2}
            tmpl = (geo, hi, vs, pairs, mz_op)
        else:
            geo, hi, vs, pairs, mz_op = tmpl
            vs = load_weights(vs, _weights_base(jp))
        vs.reset()
        O, Oe, per = fm_ratio_avg(vs, pairs)
        mz = vs.expect(mz_op)
        rows.append({"field": float(cfg0[field]), "O": O, "Oe": Oe,
                     "mz": float(np.real(mz.mean)), "mz_e": float(np.real(mz.error_of_mean)),
                     "name": cfg0.get("name", os.path.basename(jp)[:-5])})
        if verbose:
            print(f"  {rows[-1]['name']}: {field}={rows[-1]['field']:.4g}  "
                  f"O_FM={O:.4f}±{Oe:.4f}  <σz>={rows[-1]['mz']:.4f}  "
                  f"[halves {per['br'][0]:.3f}/{per['tl'][0]:.3f}, "
                  f"{time.perf_counter()-t0:.1f}s]", flush=True)
    if not rows:
        raise ValueError(f"no checkpoints in {checkpoint_dir} match "
                         f"(L={L}, hx={hx}, arch={arch}, bc={bc})")
    rows.sort(key=lambda r: r["field"])
    out = {k: np.array([r[k] for r in rows],
                       dtype=object if k == "name" else float) for k in rows[0]}
    out["_meta"] = meta
    return out


# =============================================================================
# CLI
# =============================================================================

def _parse():
    p = argparse.ArgumentParser(description="2D electric FM string sweep from trained NQS")
    p.add_argument("--dir", required=True, help="dir of train_2d *.json checkpoints")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--hx", type=float, default=0.0)
    p.add_argument("--arch", default="Combo")
    p.add_argument("--bc", default="OBC")
    p.add_argument("--field", default="hz")
    p.add_argument("--R", type=int, default=None, help="loop side (default L-3, bulk-centred)")
    p.add_argument("--eval_samples", type=int, default=None)
    p.add_argument("--out", default=None, help="write the sweep + fit to this JSON")
    return p.parse_args()


def main():
    a = _parse()
    sweep = fm_sweep_2d(a.dir, L=a.L, hx=a.hx, arch=a.arch, bc=a.bc, field=a.field,
                        R=a.R, eval_samples=a.eval_samples)
    fit = fit_transition(sweep["field"], sweep["O"], sweep["Oe"])
    hc = fit.get("h_c", fit["h_c_fd"])
    print(f"\n[fm_2d] L={a.L} hx={a.hx}: R={sweep['_meta']['R']} "
          f"(margin {sweep['_meta']['margin']} plaq)  →  h_c ≈ {hc:.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        payload = {"meta": sweep["_meta"],
                   "field": sweep["field"].tolist(), "O": sweep["O"].tolist(),
                   "Oe": sweep["Oe"].tolist(), "mz": sweep["mz"].tolist(),
                   "mz_e": sweep["mz_e"].tolist(), "name": sweep["name"].tolist(),
                   "h_c": float(hc), "h_c_fd": float(fit["h_c_fd"])}
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[fm_2d] wrote {a.out}")


if __name__ == "__main__":
    main()
