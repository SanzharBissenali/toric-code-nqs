"""Regression: the cached-Pauli-string fast path in builders.build_hamiltonian
must be EXACTLY the operator create_hamiltonian builds — same strings, same
max_conn_size (no new JIT shapes), identical matrix elements on random configs.

Standalone: cd tests && ../.venv/bin/python test_hamiltonian_cache.py
Operators only (host-side, no variational state) — safe on the dev machine.
"""
import shutil

import numpy as np

from tc3d.builders import (build_geometry, build_hamiltonian, _PS_PARTS,
                           _pauli_cache_dir, _pauli_cache_path)
from tc3d.hamiltonian import create_hamiltonian


def _conn_dict(H, x):
    """{connected-config bytes -> summed mel} for one sample row."""
    xp, mels = H.get_conn_padded(x[None, :])
    d = {}
    for row, m in zip(np.asarray(xp[0]), np.asarray(mels[0])):
        k = row.tobytes()
        d[k] = d.get(k, 0.0) + complex(m)
    return {k: v for k, v in d.items() if abs(v) > 1e-14}


def _compare(cfg, geo, hi, rng, tag):
    fast, _ = build_hamiltonian(cfg, geo, hi)
    dtype = cfg.get("dtype", "complex" if cfg.get("hy", 0.0) != 0.0 else "float64")
    slow = create_hamiltonian(hi=hi, vertex_all=geo.vertex_all,
                              plaq_all=geo.plaq_all, bonds=geo.bonds,
                              dual=cfg.get("dual_basis", False),
                              hx=cfg.get("hx", 0.0), hy=cfg.get("hy", 0.0),
                              hz=cfg.get("hz", 0.0),
                              J=cfg.get("J", 1.0), dtype=dtype)
    assert fast.max_conn_size == slow.max_conn_size, \
        (tag, fast.max_conn_size, slow.max_conn_size)
    for _ in range(8):
        x = rng.choice([-1.0, 1.0], hi.size)
        df, ds = _conn_dict(fast, x), _conn_dict(slow, x)
        assert df.keys() == ds.keys(), (tag, "connected sets differ")
        err = max(abs(df[k] - ds[k]) for k in df) if df else 0.0
        assert err < 1e-12, (tag, err)
    print(f"[PASS] fast == create_hamiltonian: {tag} "
          f"(max_conn {fast.max_conn_size})")


def _mismatch_tests(geo, hi, key_real, key_hy):
    """Adversarial-audit fix: a file's embedded key_repr/code_hash must be
    checked on load. (a) a file copied onto the WRONG key's path must be
    rejected (key mismatch) and rebuilt; (b) a file with a tampered code_hash
    must be rejected (code mismatch) and rebuilt. Both must reproduce the
    direct create_hamiltonian result exactly after the rebuild. Skipped when
    disk caching is off (TC3D_PAULI_CACHE=0) -- nothing on disk to tamper."""
    cache_dir = _pauli_cache_dir()
    if cache_dir is None:
        print("[SKIP] mismatch tests (disk cache disabled)")
        return
    path_real, path_hy = (_pauli_cache_path(cache_dir, key_real),
                          _pauli_cache_path(cache_dir, key_hy))
    assert path_real.exists() and path_hy.exists(), \
        "both keys must already be on disk from the calls above"
    x = np.random.default_rng(3).choice([-1.0, 1.0], hi.size)

    shutil.copyfile(path_real, path_hy)   # (a) wrong-path copy
    _PS_PARTS.pop(key_hy, None)           # force a disk re-read, not an in-memory hit
    fast, _ = build_hamiltonian({"model": "bosonic", "dual_basis": True, "J": 1.0,
                                 "hx": 0.1, "hz": 0.1, "hy": 0.2}, geo, hi)
    slow = create_hamiltonian(hi=hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all,
                              bonds=geo.bonds, dual=True, hx=0.1, hz=0.1, hy=0.2,
                              J=1.0, dtype="complex")
    assert fast.max_conn_size == slow.max_conn_size
    df, ds = _conn_dict(fast, x), _conn_dict(slow, x)
    assert df.keys() == ds.keys() and max((abs(df[k] - ds[k]) for k in df), default=0.0) == 0.0
    print("[PASS] (a) wrong-path copy -> key mismatch -> rebuilt correctly")

    _PS_PARTS.pop(key_real, None)
    with np.load(path_real, allow_pickle=False) as data:   # (b) tampered code_hash
        arrays = {k: data[k] for k in data.files}
    arrays["code_hash"] = np.array(["deadbeefdeadbeef"])
    np.savez(path_real, **arrays)
    fast, _ = build_hamiltonian({"model": "bosonic", "dual_basis": True, "J": 1.0,
                                 "hx": 0.2, "hz": 0.1}, geo, hi)
    slow = create_hamiltonian(hi=hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all,
                              bonds=geo.bonds, dual=True, hx=0.2, hz=0.1,
                              J=1.0, dtype="float64")
    assert fast.max_conn_size == slow.max_conn_size
    df, ds = _conn_dict(fast, x), _conn_dict(slow, x)
    assert df.keys() == ds.keys() and max((abs(df[k] - ds[k]) for k in df), default=0.0) == 0.0
    print("[PASS] (b) tampered code_hash -> code mismatch -> rebuilt correctly")


def main():
    rng = np.random.default_rng(11)
    geo = build_geometry({"L": 4, "bc": "OBC"})
    import netket as nk
    hi = nk.hilbert.Spin(s=1 / 2, N=geo.N)

    base = {"model": "bosonic", "dual_basis": True, "J": 1.0}
    for hx, hz in ((0.2, 0.1), (0.8, 0.1), (0.2, 0.0), (0.0, 0.3)):
        _compare({**base, "hx": hx, "hz": hz}, geo, hi, rng,
                 f"dual hx={hx} hz={hz}")
    _compare({"model": "bosonic", "dual_basis": False, "J": 1.0,
              "hx": 0.3, "hz": 0.2}, geo, hi, rng, "primal hx=0.3 hz=0.2")
    # cache actually used: one float64 entry per (geometry, basis) after the
    # runs above (dual and primal each get their own key; the four dual
    # hx/hz combos above share the SAME key -- only weights differ).
    assert len(_PS_PARTS) == 2, len(_PS_PARTS)

    # hy is a PRODUCTION cache channel (task A2, 2026-09), not a fallthrough:
    # dtype=="complex" gets its OWN key (one per basis), verified bit-identical
    # to create_hamiltonian exactly like the real-only channels above.
    n_keys = len(_PS_PARTS)
    _compare({**base, "hx": 0.1, "hz": 0.1, "hy": 0.2}, geo, hi, rng,
             "dual hx=0.1 hz=0.1 hy=0.2")
    assert len(_PS_PARTS) == n_keys + 1, "hy!=0 (dual) must populate its own cache key"
    _compare({"model": "bosonic", "dual_basis": False, "J": 1.0,
              "hx": 0.1, "hz": 0.1, "hy": 0.3}, geo, hi, rng,
             "primal hx=0.1 hz=0.1 hy=0.3")
    assert len(_PS_PARTS) == n_keys + 2, "hy!=0 (primal) must populate its own cache key"
    print("[PASS] cache population, real + hy channels")

    # keys already on disk from the dual real/hy calls above (see _pauli_parts)
    key_real = (int(hi.size), geo.Lx, geo.Ly, geo.Lz, geo.bc, len(geo.vertex_all),
               len(geo.plaq_all), True, 1.0, "float64")
    key_hy = (int(hi.size), geo.Lx, geo.Ly, geo.Lz, geo.bc, len(geo.vertex_all),
             len(geo.plaq_all), True, 1.0, "complex")
    _mismatch_tests(geo, hi, key_real, key_hy)
    print("All Hamiltonian-cache tests passed.")


if __name__ == "__main__":
    main()
