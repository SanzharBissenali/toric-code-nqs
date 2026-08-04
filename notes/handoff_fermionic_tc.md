# Handoff: 2D / 3D toric-code KD sweeps + Wang-Levin fermionic extension

Repo: `/Users/sanzhar123/Desktop/Approximate-Symmetries-TC-main/`
Notebook: `2D_TC_phase_diag.ipynb`

## Goal

Track the topological→polarized phase transition of the perturbed toric code
(2D rotated surface code at d=3,4; 3D bosonic toric code at L=2 PBC) under a
uniform $h_z$ field at fixed $h_x = 0.3$, using a **matrix-free Krylov
diagonalization** (no sparse matrix materialized). Diagnostic is the
Fredenhagen–Marcu (BFFM) string order parameter $O^Z_{BFFM}$ and its
derivative $\partial O^Z_{BFFM} / \partial h_z$ (peak = transition).

Extended to the **3D fermionic toric code**. Bosonic case done; fermionic case
**now done** with a minimal two-edge plaquette decoration (much simpler than the
Wang–Levin 10-edge $\Phi_p$ — see "Fermionic decoration: resolved" below).

## Built and working

### `tc3d/exact_diag.py`
Matrix-free Hamiltonian + Pauli-string expectations. Geometry-agnostic
(consumes any object with `.N`, `.vertex_all`, `.plaq_all`, plus optional
`.arr_coord` and `._coord_to_idx` for the 3D geometry).

- `hamiltonian_linop(geom, hx, hy=0, hz, J=1, xz_stabs=None, dtype=float64)`
  returns `(H, basis)` where `H` is a `scipy.sparse.linalg.LinearOperator`.
- `xz_stabs`: optional list of `(z_qubits, x_qubits, coef)` triples. When
  non-empty, **replaces** the default Z-only plaquette term — supply the
  full plaquette set with the dressing. Z- and X-supports must be disjoint
  in each entry.
- Numba-JIT'd matvec (`_matvec_jit`, `_matvec_jit_xz_add`) — parallel,
  `cache=True`. The XZ kernel uses `_bit_parity64` for branch-free sign
  evaluation.
- `expect_x_string(psi, basis, mask)`, `expect_z_string(psi, basis, mask, N)`,
  `qubits_to_mask(indices)` — same conventions as
  `tests/colab_exact_diag.py:135-217`.

### `model/rotated_surface.py`
2D rotated surface code on a $d \times d$ vertex grid. Implements
`vertex_all`, `plaq_all` (checkerboard X/Z faces + boundary 2-body
stabilizers), `bonds`, `get_vertex_all_hetero`, and:

- `wilson_paths(center=None)` returns `(closed_x, open_x, closed_z, open_z)`.
  Convention: σ^x logical = full vertical column (varies $j$); σ^z logical =
  full horizontal row (varies $i$). Open = first half.

### `tc3d/fermionic_decoration.py`  *(complete — see "Fermionic decoration: resolved")*
Minimal two-edge decoration, replacing the Wang-Levin skeleton:

- `Plaquette` dataclass (cell index + orientation + 4 boundary edges) with
  `center`, `perp_axis`, `in_plane_axes` properties.
- `enumerate_plaquettes(geom)` — 24 plaquette objects at L=2 PBC (8 xy + 8 yz
  + 8 zx), boundary edges computed PBC-correctly.
- `decoration_edges(geom, p)` — the two perpendicular $\sigma^x$ edges of
  $B^\sim_p$ (the body diagonal; see below).
- `fermionic_plaquettes(geom, J)` — produces the drop-in `xz_stabs` list
  `[(z_qubits, x_qubits, coef), ...]` for `hamiltonian_linop`.
- `verify_xz_commutation(stabs, vertex_all)` — pairwise commutation check.

### `tc3d/geometry.py` (pre-existing, used as-is)
3D toric code geometry. PBC L=2: `N=24`, `|vertex_all|=8`, `|plaq_all|=24`.
Exposes `arr_coord`, `_coord_to_idx` (2×-integer keys to avoid float
equality issues — see lines 90, 98–99).

### Notebook (`2D_TC_phase_diag.ipynb`) sections
1. **2D rotated surface code d=3, d=4** — KD sweep + BFFM_Z and
   ∂$O^Z_{BFFM}$/∂$h_z$ plot. Cells `a29a39b9` (sweep fn), `caf664e8` (driver),
   `e4d11edd` (4-panel A/B/gap/Mz plot), `bffmplot` (BFFM 1×2 plot).
2. **3D bosonic TC L=2 PBC** — matrix-free sweep with `hamiltonian_linop`,
   BFFM_Z via a z-line Wilson string. Cells `td3hdr` → `td3plot`. Verified:
   `E0 = -32`, `gap = 4` (the gap is 4, not 2, because of the
   $\prod_v A_v = I$ constraint — single stabilizer flips are forbidden).
3. **CNN-NQS sweep** at L=3 PBC / L=5 PBC — separate workflow using
   `model/networks.py`, `simulation/custom_sampler.py`, `simulation/optimizer.py`
   (TDVP path). Driven from `2D_TC_phase_diag.ipynb` cells `f6dbdb13`
   (config), `a124ea00` (`cnn_sweep`), etc. There's also a bash sweep at
   `scripts/sweep_hz.sh` that uses `main.py` end-to-end.

## Sanity checks already passed

- **2D rotated surface code d=3, d=4** zero field: all stabilizers $+1$,
  $E_0 = -8$ at d=3 (4 X-stabs + 4 Z-stabs + 4 boundary X-stabs?),
  $E_0 = -15$ at d=4. BFFM_Z $\to 0.99$ at $h_z = 2.0$ on d=3 (Z-polarized).
- **3D bosonic TC L=2 PBC**: $E_0 = -32$, gap = 4, all $\langle A_v\rangle = 1$,
  all $\langle B_p\rangle = 1$. 8-fold ground-state degeneracy on $T^3$
  confirmed (need `k>=8` in `eigsh` to see all 8 copies at $-32$).
- **Fermionic skeleton** with placeholder $\Phi_p = I$: `verify_xz_commutation`
  returns `ok=True, violations=0` (just reproduces bosonic).
- **Numba JIT**: first call has ~1–2 s compile overhead, subsequent calls hit
  the cached binary. Speedup vs. pure-numpy matvec: roughly 5-10× on 3D L=2.

## Fermionic decoration: resolved

The Wang-Levin 10-edge dressing $\Phi_p$ turned out to be unnecessary. A much
simpler decoration works: each plaquette $B^\sim_p = (\prod_{e\in\partial p}
\sigma^z_e)\,\sigma^x_{e_+}\,\sigma^x_{e_-}$ where $e_+, e_-$ are two
perpendicular ("transverse") edges at **opposite corners on opposite sides** of
the plaquette plane (a body diagonal):

- $e_+$: perpendicular edge at the $(+a, +b)$ corner, on the $+$perp side;
- $e_-$: perpendicular edge at the $(-a, -b)$ corner, on the $-$perp side,

with $(a, b)$ the in-plane axes and perp the normal axis. The **same** pattern
is used for all three orientations.

Why it commutes (and why the earlier naive guess didn't):
- vs. vertex stars $A_v$: automatic. $\sigma^x$ decoration commutes with the
  all-$\sigma^x$ star, and the $\sigma^z$ boundary shares an even number of
  edges with each star (exactly as bosonic). True for *any* perpendicular-edge
  decoration — so the only real constraint is plaquette–plaquette.
- vs. other plaquettes: same-orientation pairs commute trivially (in-plane Z
  boundary never overlaps perpendicular decoration). Different-orientation pairs
  are the binding constraint; a brute-force search over "any 2 of the 8
  perpendicular corner edges per orientation" found 80 commuting patterns at
  L=2, of which the body-diagonal one above is the cleanest and uniform across
  orientations. The earlier same-side planar-diagonal guess gives **0** working
  patterns (this is why the naive attempt failed).

Verified numerically:
- `verify_xz_commutation` → `ok=True`, 32 stabilizers, 0 violations at L=2; also
  0 violations at L=3 (pattern is translation-invariant ⇒ holds for all L).
- Hamiltonian Hermiticity through the matrix-free pipeline: residual ~3e-9.
- Zero-field ED at L=2 PBC: $E_0 = -32$ (frustration-free, matches bosonic
  minimum) but **gap = 8, not the bosonic 4** — a clear signature the decoration
  is non-trivial; the whole low spectrum comes in exact doublets.

Drop-in usage (matrix-free pipeline unchanged from bosonic):

```python
from tc3d.fermionic_decoration import fermionic_plaquettes, verify_xz_commutation
from tc3d.exact_diag import hamiltonian_linop
from tc3d.geometry import ThreeD_ToricCodeGeometry

geom = ThreeD_ToricCodeGeometry(2, 2, 2, "PBC")
stabs = fermionic_plaquettes(geom, J=1.0)
assert verify_xz_commutation(stabs, geom.vertex_all)["ok"]

H, basis = hamiltonian_linop(geom, hx=0.3, hz=0.3, xz_stabs=stabs)
# eigsh exactly as in bosonic — the matrix-free pipeline doesn't change.
```

## Dressed Wilson loop & fermionic order parameter

The bare $\sigma^z$ Wilson string is **not conserved** in the fermionic model: a
$\sigma^z$ line that runs through a decorated edge anticommutes with that
plaquette's $\sigma^x$ (we measured a straight wrapping line anticommuting with 4
of the $\tilde B_p$). So it can't serve as an order parameter — acting with it on
the ground state sprays plaquette excitations along its body, and any closed loop
that anticommutes with a stabilizer has $\langle W\rangle\equiv0$.

**Fix — dress the line with $\sigma^x$.** Write the operator as $Z(\ell)\,X(s)$:
$\sigma^z$ on the line $\ell$, plus a $\sigma^x$ dressing $s$. It commutes with
$\tilde B_p=Z(\partial p)X(d_p)$ iff $|\ell\cap d_p|+|s\cap\partial p|$ is even.
(Vertex stars are automatic — $\sigma^x$ dressing commutes with the all-$\sigma^x$
stars, and $\ell$ shares an even number of edges with each star except at its
endpoints.) So choose $s$ solving, over GF(2),

$$\sum_{e\in\partial p} s_e \;\equiv\; |\ell\cap d_p| \pmod 2 \quad\forall p,
\qquad\text{i.e. } M\,s = t,\ \ M_{p,e}=[e\in\partial p],\ t_p=|\ell\cap d_p|\bmod2.$$

This is `dressed_string(geom, stabs, z_edges)` in
`tc3d/fermionic_decoration.py` (a ~12-line `_gf2_solve` + the assembly).

- **Closed wrapping loop** → $M s=t$ is consistent → a fully **conserved Wilson
  loop** $W$ (verified: dressing of 4 σˣ at L=2 / 6 at L=3, disjoint from the
  line, 0 residual, commutes with the whole stabilizer group).
- **Open half-string** → $M s=t$ is **inconsistent**: the cubes containing the
  endpoints are frustrated, so a flux-free open string *cannot exist*. The
  solver returns the consistent-subsystem solution, leaving a **localized
  residual of exactly 2 plaquettes** (one flux per endpoint). Each endpoint thus
  carries a charge (violated star) **and** a flux (violated plaquette) — that
  charge+flux composite is the **fermion**. This non-existence of a flux-free
  open string is the operational statement that the excitation is fermionic.

  (The residual of an inconsistent GF(2) system is solver/ordering dependent; the
  *canonical* `fermionic_plaquettes` plaquette order yields the minimal,
  endpoint-localized residual = 2. Both notebooks use it, so they agree.)

**Order parameter.** Fredenhagen–Marcu ratio
$O_{FM}=\langle S\rangle/\sqrt{|\langle W\rangle|}$ with $S$ the open dressed
fermion string and $W$ the conserved loop, measured via
`expect_xz_string`. Plotted with its $h_z$-derivative (transition = peak),
alongside the conserved $\langle W\rangle$, the spectral gap, and
$\langle M_z\rangle$ as robust diagnostics. Implemented in the fermionic sweep of
`2D_TC_phase_diag.ipynb` (cells `tdfsetup`/`tdfsweep`/`tdfplot`) and
`colab/fermionic_TC_colab.ipynb`.

**L=2 caveat.** At $L=2$ PBC the open string is a single edge with no bulk, so
$O_{FM}$ may be inconclusive; $\langle W\rangle$, the gap, and $\langle M_z\rangle$
are the reliable signals. A clean $O_{FM}$ needs $L\ge3$–$4$, which is past ED
reach — run those on Colab if a bigger machine is available.

### Open follow-ups (physics, not blockers)
- Confirm the emergent point excitation is genuinely a fermion via braiding
  statistics (the open-string flux obstruction above is strong evidence).
- Full ground-state degeneracy on $T^3$ (needs `k` large enough in `eigsh`;
  only the lowest 10 were inspected, showing a doublet at $E_0=-32$).
- A provably minimum-weight (ordering-independent) open-string residual via
  syndrome decoding of the cube code, if larger $L$ is ever run.

## Earlier path attempted and ruled out

- **"Decorate with sigma^x on NE-SW diagonal perpendicular corners on +perp
  side"** — naive guess. Failed `verify_xz_commutation` with 96 / 496
  pairwise violations. The issue is that single perpendicular σ^x decorations
  on a z-edge always anticommute with exactly one neighboring xz- or yz-
  plaquette's σ^z support, generating an odd anti-commutation count.
  Wang-Levin's actual recipe must cancel these somehow — that's what the
  precise formula is for.
- **Walker-Wang multi-qubit construction** from the errorcorrectionzoo
  picture: deliberately deferred. If 3 qubits per edge (×3 edges per cell ×
  L³ = 9 L³), then L=2 PBC = 72 qubits = $2^{72}$ states, infeasible. If 3
  qubits per vertex, L=2 PBC = 24 qubits = $2^{24}$ states, feasible but a
  totally different framework from what we built.

## Memory budget on user's machine (8 GB Mac)

- 2D rotated surface d=3, d=4: trivial.
- 2D rotated surface d=5 ($2^{25}$ = 33M states, ~268 MB / vector at
  float64): **does not fit** on this machine with default `ncv=20` Lanczos
  workspace (5.4 GB). Workarounds: `ncv=6`, `float32`, GPU. **User
  explicitly said not to optimize for d=5** — keep it commented out.
- 3D bosonic TC L=2 PBC ($2^{24}$ = 16.8M states, ~134 MB / vector at
  float64): fits with `ncv=10`. Each eigsh call ~10 minutes with k=4 at
  default tol. Numba JIT brings this down meaningfully.
- 3D L=3 PBC = 81 qubits — way out of reach.

## User preferences observed

- Code style: concise, clean, readable. Comments only when they add real
  signal. Functions with one clear purpose, no over-abstraction.
- Implementation style: edits to existing modules over creating new ones,
  unless a clean separation justifies a new file.
- Diagnostics over claims: when something doesn't behave, verify with a
  small inline test, don't just speculate.
- Plotting: matplotlib, simple `o-` markers with legends, math labels in
  LaTeX.
- Strongly dislikes when I conflate independent concepts (e.g., "you mixed
  $O_X$ and $O_Z$ — we're only sweeping $h_z$, drop $O_X$"). Match the
  observable to the sweep axis.
- When uncertain, asks for the source rather than guess. The Wang-Levin
  decoration is the prime example.

## Single-file glossary

| Path | Role |
|---|---|
| `tc3d/exact_diag.py` | Matrix-free Hamiltonian + Pauli expectations. Numba JIT. |
| `model/rotated_surface.py` | 2D rotated surface code geometry + Wilson paths. |
| `tc3d/fermionic_decoration.py` | Fermionic plaquette decoration (two-edge body diagonal) + commutation verifier. |
| `model/geometry.py` | Edge-qubit 2D TC geometry (pre-existing). |
| `model/hamiltonian.py` | NetKet PauliStrings Hamiltonian builder (used by CNN side). |
| `model/networks.py` | 2D CNN ansatz factory (`create_model`, `KernelManager`). |
| `tc3d/geometry.py` | 3D TC geometry (pre-existing). |
| `tests/colab_exact_diag.py` | Original matrix-free Lanczos reference (3D, complex). |
| `simulation/custom_sampler.py` | Vertex-cluster + local Metropolis sampler. |
| `simulation/optimizer.py` | TDVP loop (`run_tdvp`). |
| `scripts/sweep_hz.sh` | Bash wrapper for `main.py` to sweep $h_z$. |
| `2D_TC_phase_diag.ipynb` | The main analysis notebook. |
