"""Regression: the cached-Pauli-string fast path in builders.build_hamiltonian
must be EXACTLY the operator create_hamiltonian builds — same strings, same
max_conn_size (no new JIT shapes), identical matrix elements on random configs.

Standalone: cd tests && ../.venv/bin/python test_hamiltonian_cache.py
Operators only (host-side, no variational state) — safe on the dev machine.
"""
import numpy as np

from tc3d.builders import build_geometry, build_hamiltonian, _PS_PARTS
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
    print("All Hamiltonian-cache tests passed.")


if __name__ == "__main__":
    main()
