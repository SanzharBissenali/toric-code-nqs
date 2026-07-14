"""
analysis/fm_2d.py
─────────────────────────────────────────────────────────────────────────────
2D Fredenhagen–Marcu (FM) string order parameter from *trained* 2D NQS, in both
the electric (σ^z, hz cut) and magnetic (σ^x, hx cut) sectors.

The 2D analogue of `Three_TC/fm.py`, specialised to `model.geometry.
ToricCodeGeometry` (qubits on edges at half-integer coords). Unlike 3D — where the
magnetic object is a *membrane* — in 2D both sectors are *strings*, related by the
exact e–m duality: the magnetic loop is the electric one shifted by (+½,+½)
(`magnetic_string_edges`), which swaps horizontal↔vertical edges and σ^z↔σ^x.

  • electric (hz sweep, fix hx): detects the hz-driven e-condensation.
  • magnetic (hx sweep, fix hz): detects the hx-driven m-condensation.
Both are the same 3D-Ising transition at |h_c|≈0.328 J by self-duality.

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
from Three_TC.fm import fm_ratio, fit_transition


def _pauli_string_op(hi, indices: Sequence[int], pauli: str):
    """∏ σ^{pauli} over `indices` as ONE NetKet PauliStrings operator.

    A product of single-site LocalOperators (the old `_pauli_product`) materializes a
    dense 2^k block over the k-site support, so it OOMs once the loop perimeter grows
    (L>=10: closed loop 4R>=28 sites -> 2^28). PauliStrings stores the string
    symbolically: O(N) memory, trivial get_conn (diagonal for Z, one flip per X)."""
    chars = ["I"] * hi.size
    for i in dict.fromkeys(int(j) for j in indices):     # dedup, preserve order
        chars[i] = pauli.upper()
    return nk.operator.PauliStrings(hi, ["".join(chars)], [1.0])


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


def magnetic_string_edges(geo, *, R: Optional[int] = None, half: str = "br"
                          ) -> Tuple[List[int], List[int]]:
    """(closed, open) edge-index lists for a bulk-centred σ^x *dual* Wilson rectangle.

    The e–m dual of `electric_string_edges`: the identical rectangle shifted by
    (+½,+½), which swaps horizontal↔vertical edges and turns the σ^z-around-
    plaquettes loop into a σ^x-around-vertices loop (the (+½,+½) shift IS the
    square-lattice e–m duality). Consequences, verified combinatorially:

      • closed  = ∏ σ^x over the perimeter = ∏ A_v over the enclosed R×R block of
        (bulk) vertices, so ⟨closed⟩ = 1 in the pure ground state — the σ^x analogue
        of the electric ∏B_p, hence the FM normalisation.
      • open    = HALF that perimeter (2R edges); its two endpoints are plaquette
        centres, i.e. m-charges (fluxes). Detects the hx-driven m-condensation.

    Same R = L-3 bulk loop as the electric sector (matched size for an e/m
    comparison). The dual shift pushes the loop's *edges* one half-unit outward
    (enclosed vertices span 2..L-2), so its bulk margin is 1 vertex on the high
    side — still strictly interior, and the closed loop is a product of bulk-vertex
    stabilizers by construction.
    """
    L = geo.Lx
    if geo.Lx != geo.Ly:
        raise NotImplementedError("magnetic_string_edges assumes Lx == Ly")
    R = bulk_string_R(L, R)
    a0 = b0 = (L - 1 - R) // 2
    s = 0.5                                     # (+½,+½) e–m dual shift

    def hedge(k, y):                            # electric horizontal edge, shifted → vertical
        return _edge(geo, a0 + 0.5 + k + s, y + s)

    def vedge(x, k):                            # electric vertical edge, shifted → horizontal
        return _edge(geo, x + s, b0 + 0.5 + k + s)

    bottom = [hedge(k, b0)       for k in range(R)]
    top    = [hedge(k, b0 + R)   for k in range(R)]
    left   = [vedge(a0,     k)   for k in range(R)]
    right  = [vedge(a0 + R, k)   for k in range(R)]

    closed = bottom + right + top + left        # full perimeter (4R edges)
    open_  = (bottom + right) if half == "br" else (top + left)   # 2R edges, corner→corner
    return closed, open_


# sector → (edge builder, Pauli letter, magnetization Pauli) — the ONLY sector-dependent bits.
_SECTORS = {
    "electric": (electric_string_edges, "z"),   # hz sweep: σ^z string, ⟨σz⟩
    "magnetic": (magnetic_string_edges, "x"),   # hx sweep: σ^x dual string, ⟨σx⟩
}


def build_string_operators(geo, hi, *, sector: str = "electric", R: Optional[int] = None
                           ) -> List[Tuple[str, Any, Any]]:
    """[(label, open_op, closed_op), ...] for the two diagonal halves of one bulk loop.

    sector="electric" → σ^z rectangle (hz cut); "magnetic" → σ^x dual rectangle (hx cut).
    """
    edges_fn, pauli = _SECTORS[sector]
    pairs = []
    for half in ("br", "tl"):
        closed, open_ = edges_fn(geo, R=R, half=half)
        pairs.append((half,
                      _pauli_string_op(hi, open_, pauli),
                      _pauli_string_op(hi, closed, pauli)))
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
    """Weights base for a train_2d artifact, preferring the final `.mpack` and
    falling back to the periodic `.ckpt.mpack` — the run that timed out before its
    final save but whose latest checkpoint we still want to score. Accepts either a
    final `{name}.json` or a `{name}.curve.json` checkpoint marker as the path."""
    if json_path.endswith(".curve.json"):
        base = json_path[:-len(".curve.json")]
    else:
        base = json_path[:-len(".json")]
    if os.path.exists(base + ".mpack"):
        return base
    if os.path.exists(base + ".ckpt.mpack"):
        return base + ".ckpt"
    raise FileNotFoundError(f"no weights for {json_path}: tried {base}.mpack / {base}.ckpt.mpack")


def load_vstate_2d(json_path: str, *, eval_samples: Optional[int] = None,
                   chunk_size: Optional[int] = None):
    """Rebuild (cfg, geo, hi, vs) from a train_2d artifact and load its weights.

    Works for both a final `{name}.json` and a `{name}.curve.json` checkpoint (both
    carry `config`); `_weights_base` picks `.mpack` or `.ckpt.mpack` accordingly.
    `chunk_size` overrides the saved config's chunk_size for evaluation: the Combo
    forward over eval_samples must be chunked at large L (N>=180) or it OOMs the GPU,
    exactly as the local-energy step did in training. None keeps the saved value."""
    with open(json_path) as f:
        cfg = json.load(f)["config"]
    overrides: Dict[str, Any] = {}
    if eval_samples is not None:
        overrides["n_samples"] = int(eval_samples)
    if chunk_size is not None:
        overrides["chunk_size"] = int(chunk_size)
    cfg = with_defaults({**cfg, **overrides})
    geo, hi, _Ham, vs = build_state(cfg)
    vs = load_weights(vs, _weights_base(json_path))
    return cfg, geo, hi, vs


def _matches(cfg: Dict[str, Any], L, fixed_name, fixed_val, arch, bc) -> bool:
    """Select a checkpoint of size L whose *fixed* transverse field matches (the
    field NOT being swept: hx for an electric/hz sweep, hz for a magnetic/hx sweep)."""
    return (int(cfg.get("L", -1)) == L
            and abs(float(cfg.get(fixed_name, np.nan)) - float(fixed_val)) < 1e-9
            and (arch is None or cfg.get("arch") == arch)
            and (bc is None or cfg.get("bc") == bc))


# =============================================================================
# Grid sweep over trained checkpoints
# =============================================================================

def fm_sweep_2d(checkpoint_dir: str, *, L: int, sector: str = "electric",
                fixed: float = 0.0, arch: str = "Combo", bc: str = "OBC",
                field: Optional[str] = None, R: Optional[int] = None,
                eval_samples: Optional[int] = None, eval_chunk: Optional[int] = None,
                verbose: bool = True) -> Dict[str, Any]:
    """FM string O_FM (avg over both halves) + field-aligned ⟨mag⟩ + V-score per swept field.

    sector="electric": σ^z string, fix hx=`fixed`, sweep hz, cross-check ⟨σz⟩.
    sector="magnetic": σ^x dual string, fix hz=`fixed`, sweep hx, cross-check ⟨σx⟩.

    Scans `checkpoint_dir` for train_2d artifacts matching (L, fixed-field, arch, bc),
    builds the bulk string operators once, and evaluates each by swapping in its weights.
    One artifact per run: the final `{name}.json` if present, else the `{name}.curve.json`
    of a run that timed out before its final save (scored from its latest `.ckpt.mpack`).
    The `vscore` column is the convergence gate — trust a timed-out checkpoint's O_FM
    only where V-score is low. Returns arrays keyed field / O / Oe / mag / mag_e / vscore / name.
    """
    field = field or ("hz" if sector == "electric" else "hx")
    fixed_name = "hx" if sector == "electric" else "hz"   # the field held at `fixed`
    mag_pauli = nk.operator.spin.sigmaz if sector == "electric" else nk.operator.spin.sigmax
    by_base: Dict[str, str] = {}
    for jp in sorted(glob.glob(os.path.join(checkpoint_dir, "*.json"))):
        if jp.endswith(".curve.json"):
            base, final = jp[:-len(".curve.json")], False
        else:
            base, final = jp[:-len(".json")], True
        if final or base not in by_base:           # prefer the final over the checkpoint
            by_base[base] = jp
    tmpl = None                                    # (geo, hi, vs, pairs, mz_op)
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    for jp in sorted(by_base.values()):
        try:
            with open(jp) as f:
                doc = json.load(f)
            cfg0 = doc.get("config", {})
        except (json.JSONDecodeError, KeyError):
            continue
        if not cfg0 or not _matches(cfg0, L, fixed_name, fixed, arch, bc):
            continue
        unfinished = jp.endswith(".curve.json")
        vscore = None                              # latest V-score (convergence gate)
        if not unfinished and doc.get("observables", {}).get("Vscore") is not None:
            vscore = float(doc["observables"]["Vscore"])
        elif doc.get("curve", {}).get("vscore"):
            vscore = float(doc["curve"]["vscore"][-1])
        t0 = time.perf_counter()
        if tmpl is None:
            _cfg, geo, hi, vs = load_vstate_2d(jp, eval_samples=eval_samples,
                                               chunk_size=eval_chunk)
            pairs = build_string_operators(geo, hi, sector=sector, R=R)
            mag_op = sum(mag_pauli(hi, i) for i in range(geo.N)) / geo.N
            R_used = bulk_string_R(geo.Lx, R)
            meta = {"L": L, "sector": sector, fixed_name: fixed, "field": field,
                    "arch": arch, "bc": bc,
                    "R": R_used, "margin": (geo.Lx - 1 - R_used) // 2}
            tmpl = (geo, hi, vs, pairs, mag_op)
        else:
            geo, hi, vs, pairs, mag_op = tmpl
            vs = load_weights(vs, _weights_base(jp))
        vs.reset()
        O, Oe, per = fm_ratio_avg(vs, pairs)
        mag = vs.expect(mag_op)
        rows.append({"field": float(cfg0[field]), "O": O, "Oe": Oe,
                     "mag": float(np.real(mag.mean)), "mag_e": float(np.real(mag.error_of_mean)),
                     "vscore": float(vscore) if vscore is not None else float("nan"),
                     "name": cfg0.get("name", os.path.basename(jp)[:-5])})
        if verbose:
            tag = "  [ckpt/unfinished]" if unfinished else ""
            vs_str = f"{vscore:.2e}" if vscore is not None else "n/a"
            msym = "σz" if sector == "electric" else "σx"
            print(f"  {rows[-1]['name']}: {field}={rows[-1]['field']:.4g}  "
                  f"O_FM={O:.4f}±{Oe:.4f}  <{msym}>={rows[-1]['mag']:.4f}  Vscore={vs_str}  "
                  f"[halves {per['br'][0]:.3f}/{per['tl'][0]:.3f}, "
                  f"{time.perf_counter()-t0:.1f}s]{tag}", flush=True)
    if not rows:
        raise ValueError(f"no checkpoints in {checkpoint_dir} match "
                         f"(L={L}, {fixed_name}={fixed}, arch={arch}, bc={bc})")
    rows.sort(key=lambda r: r["field"])
    out = {k: np.array([r[k] for r in rows],
                       dtype=object if k == "name" else float) for k in rows[0]}
    out["_meta"] = meta
    return out


# =============================================================================
# CLI
# =============================================================================

def _parse():
    p = argparse.ArgumentParser(description="2D FM string sweep from trained NQS "
                                            "(electric σ^z / hz cut, or magnetic σ^x / hx cut)")
    p.add_argument("--dir", required=True, help="dir of train_2d *.json checkpoints")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--sector", default="electric", choices=["electric", "magnetic"],
                   help="electric: σ^z string, sweep hz (fix hx); magnetic: σ^x dual "
                        "string, sweep hx (fix hz)")
    p.add_argument("--fixed", type=float, default=0.0,
                   help="value of the NON-swept transverse field (hx for electric, hz "
                        "for magnetic); the analytic cuts are at 0.0")
    p.add_argument("--arch", default="Combo")
    p.add_argument("--bc", default="OBC")
    p.add_argument("--field", default=None,
                   help="swept field (default hz for electric, hx for magnetic)")
    p.add_argument("--R", type=int, default=None, help="loop side (default L-3, bulk-centred)")
    p.add_argument("--eval_samples", type=int, default=None)
    p.add_argument("--eval_chunk", type=int, default=None,
                   help="chunk_size for the eval forward pass; set at L>=10 (N>=180) to "
                        "avoid a GPU OOM in vs.expect (overrides the saved config's value)")
    p.add_argument("--out", default=None, help="write the sweep + fit to this JSON")
    return p.parse_args()


def main():
    a = _parse()
    sweep = fm_sweep_2d(a.dir, L=a.L, sector=a.sector, fixed=a.fixed, arch=a.arch,
                        bc=a.bc, field=a.field, R=a.R, eval_samples=a.eval_samples,
                        eval_chunk=a.eval_chunk)
    fit = fit_transition(sweep["field"], sweep["O"], sweep["Oe"])
    hc = fit.get("h_c", fit["h_c_fd"])
    m = sweep["_meta"]
    print(f"\n[fm_2d] L={a.L} {a.sector} ({m['field']} sweep, "
          f"{'hx' if a.sector == 'electric' else 'hz'}={a.fixed}): R={m['R']} "
          f"(margin {m['margin']})  →  h_c ≈ {hc:.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        payload = {"meta": m, "L": int(a.L), "sector": a.sector,
                   "field_name": m["field"],
                   "field": sweep["field"].tolist(), "O": sweep["O"].tolist(),
                   "Oe": sweep["Oe"].tolist(), "mag": sweep["mag"].tolist(),
                   "mag_e": sweep["mag_e"].tolist(), "vscore": sweep["vscore"].tolist(),
                   "name": sweep["name"].tolist(),
                   "h_c": float(hc), "h_c_fd": float(fit["h_c_fd"])}
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[fm_2d] wrote {a.out}")


if __name__ == "__main__":
    main()
