"""Build the field-independent Pauli-string cache DIRECTLY, bypassing
create_hamiltonian's LocalOperator `+=` algebra.

`create_hamiltonian` accumulates ~(N_v + N_p + 2N) LocalOperators with O(n^1.45)
cost (191 s at L=4, 523 s at L=5, ~1290 s at L=6) and only then converts to
PauliStrings -- at L=8 that is ~80 min and at L=10 ~3.8 h of single-core CPU
before a single VMC step can run. But the string set is trivially enumerable:
every term is a single Pauli letter repeated over a known support. This emits
exactly the same {channel: (ops, weights, dtype)} parts dict and writes it with
the repo's own `_save_pauli_parts`, so `_pauli_parts` picks it up as a normal
cache hit.

  --check L  : rebuild L via create_hamiltonian and assert the direct strings
               match as a multiset (this is the correctness gate)
  --verify-cached : compare against an EXISTING cache .npz (no rebuild)
"""
import argparse, os, sys, time
import numpy as np

from tc3d.builders import (_pauli_parts, _pauli_cache_dir, _pauli_cache_path,
                           _save_pauli_parts, _load_pauli_parts,
                           _HX_MARKER, _HZ_MARKER, _HY_MARKER, build_geometry)
from tc3d.hamiltonian import create_hamiltonian
import netket as nk


def direct_parts(geo, hi, dual, J, dtype):
    """The exact string set create_hamiltonian(+markers).to_pauli_strings()
    produces, enumerated instead of assembled.

    Letter map (dual = Hadamard conjugation, sigma_x <-> sigma_z):
        A_v (star)  -> rep_x  = Z (dual) / X (primal)
        B_p (plaq)  -> rep_z  = X (dual) / Z (primal)
        hx  field   -> rep_x  = Z (dual) / X (primal),  weight -1 per unit
        hz  field   -> rep_z  = X (dual) / Z (primal),  weight -1 per unit
        hy  field   -> Y, weight -sgn_y with sgn_y = -1 in dual  -> +1 dual / -1 primal
    """
    N = int(hi.size)
    lx, lz = ("Z", "X") if dual else ("X", "Z")
    sgn_y = -1.0 if dual else 1.0
    wdt = np.complex128 if dtype == "complex" else np.float64

    def s(letter, sites):
        row = ["I"] * N
        for j in sites:
            row[j] = letter
        return "".join(row)

    J_ops = [s(lx, [j for j in v if j != -1]) for v in geo.vertex_all]
    J_ops += [s(lz, [j for j in p if j != -1]) for p in geo.plaq_all]
    J_w = np.full(len(J_ops), -float(J), dtype=wdt)

    parts = {
        "J":  (J_ops, J_w, np.dtype(wdt)),
        "hx": ([s(lx, [j]) for j in range(N)],
               np.full(N, -1.0, dtype=wdt), np.dtype(wdt)),
        "hz": ([s(lz, [j]) for j in range(N)],
               np.full(N, -1.0, dtype=wdt), np.dtype(wdt)),
    }
    if dtype == "complex":
        parts["hy"] = ([s("Y", [j]) for j in range(N)],
                       np.full(N, -sgn_y, dtype=wdt), np.dtype(wdt))
    return parts


def as_multiset(parts):
    return {ch: sorted(zip(ops, [complex(w) for w in ws]))
            for ch, (ops, ws, _dt) in parts.items()}


def compare(a, b, label):
    ok = True
    if set(a) != set(b):
        print(f"  [{label}] CHANNEL MISMATCH {sorted(a)} vs {sorted(b)}")
        return False
    for ch in sorted(a):
        if len(a[ch]) != len(b[ch]):
            print(f"  [{label}] {ch}: count {len(a[ch])} vs {len(b[ch])}")
            ok = False
            continue
        bad = [(x, y) for x, y in zip(a[ch], b[ch])
               if x[0] != y[0] or abs(x[1] - y[1]) > 1e-12]
        if bad:
            print(f"  [{label}] {ch}: {len(bad)} differing entries, e.g. {bad[0]}")
            ok = False
        else:
            print(f"  [{label}] {ch}: OK ({len(a[ch])} strings)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", required=True)
    ap.add_argument("--bc", default="OBC")
    ap.add_argument("--dtype", nargs="+", default=["float64"])
    ap.add_argument("--dual", type=int, default=1)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--check", action="store_true",
                    help="rebuild via create_hamiltonian and compare (slow)")
    ap.add_argument("--verify-cached", action="store_true",
                    help="compare against an existing cache file, if present")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rc = 0
    for L in a.L:
        for dtype in a.dtype:
            cfg = {"L": L, "bc": a.bc}
            t0 = time.time()
            geo = build_geometry(cfg)
            hi = nk.hilbert.Spin(s=0.5, N=geo.N)
            t_geo = time.time() - t0
            t0 = time.time()
            parts = direct_parts(geo, hi, bool(a.dual), a.J, dtype)
            t_dir = time.time() - t0
            nstr = sum(len(o) for o, _, _ in parts.values())
            print(f"L={L} {a.bc} dual={a.dual} {dtype}: N={geo.N} "
                  f"N_v={len(geo.vertex_all)} N_p={len(geo.plaq_all)} "
                  f"strings={nstr} n_conn={nstr + 1 - 2 * geo.N}"
                  f"  geo={t_geo:.1f}s direct={t_dir:.1f}s")

            key = (int(hi.size), geo.Lx, geo.Ly, geo.Lz, geo.bc,
                   len(geo.vertex_all), len(geo.plaq_all), bool(a.dual),
                   float(a.J), str(dtype))

            if a.verify_cached:
                for d in ("/pscratch/sd/s/sanzharb/tc_nqs/pauli_cache",
                          os.environ.get("TC3D_PAULI_CACHE_DIR", "")):
                    if not d:
                        continue
                    from pathlib import Path
                    p = _pauli_cache_path(Path(d), key)
                    if p.exists():
                        print(f"  vs cached {p.name}")
                        if not compare(as_multiset(_load_pauli_parts(p, key)),
                                       as_multiset(parts), "cached"):
                            rc = 1
                        break
                else:
                    print("  (no existing cache file to verify against)")

            if a.check:
                markers = {"hx": _HX_MARKER, "hz": _HZ_MARKER}
                if dtype == "complex":
                    markers["hy"] = _HY_MARKER
                t0 = time.time()
                H = create_hamiltonian(hi=hi, vertex_all=geo.vertex_all,
                                       plaq_all=geo.plaq_all, bonds=geo.bonds,
                                       dual=bool(a.dual), J=float(a.J),
                                       dtype=dtype, **markers)
                print(f"  create_hamiltonian: {time.time() - t0:.1f} s")
                ops = list(H.operators)
                ws = np.asarray(H.weights)
                support = np.array([len(s) - s.count("I") for s in ops])
                keep = ws != 0
                ref = {}
                m_J = keep & (support > 1)
                ref["J"] = ([s for s, k in zip(ops, m_J) if k], ws[m_J], H.dtype)
                for ch, marker in markers.items():
                    m = keep & (support == 1) & (np.abs(np.abs(ws) - marker) < 1e-9)
                    ref[ch] = ([s for s, k in zip(ops, m) if k],
                               ws[m] / marker, H.dtype)
                if not compare(as_multiset(ref), as_multiset(parts), "rebuilt"):
                    rc = 1

            if a.write:
                d = _pauli_cache_dir()
                p = _pauli_cache_path(d, key)
                _save_pauli_parts(p, key, parts)
                print(f"  wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
