# NQS architecture considerations — 3D toric-code networks

Intuition notes on `tc3d/networks.py`: what the conv actually does to an
input and why the tensor shapes look the way they do. For the training/SR loop see
`vmc_internals.md`; for failure modes see `training_gotchas.md`.

Started 2026-06-24.

---

## Why the conv acts on `(C, N)`, not `(L, L, L)`

A vanilla `nn.Conv3D` wants a `(C, L, L, L)` tensor and reads "who is my neighbour"
from **axis adjacency** — `(ix+1, iy, iz)` is next to `(ix, iy, iz)`. `GeoConv3D`
does **not** do this. It keeps features flat and supplies the neighbour relation as
an explicit lookup table.

**The grid did not disappear — it is the flat index.**

    N = 3 · Lx·Ly·Lz = 3 L³      (3 edge orientations × L³ cube vertices)

So `(1, N)` is really `(C_in=1, [orientation × ix × iy × iz])` in
`geo._mapping3Dto1D` order. For L=2: `L³=8`, `N=24`, input `(1, 24)`. Same for
plaquettes (`N_plaq = 3L³`). See `compute_edges_3D` (`networks.py:68`).

## Why not keep `(L, L, L)`?

The three edge orientations are **interpenetrating sublattices** offset by ½ a
lattice vector — they are *not* co-located at a vertex. Stuffing them into a
channel axis `(3, L, L, L)` and running a grid conv makes "neighbour" only
*approximate* (the half-offset error). This is the whole reason the custom kernel
exists (docstring `networks.py:8`).

## What `GeoConv3D` does instead (one sample, batch dropped)

The geometry is precomputed once in `KernelManager3D`:

- `edge_gather`  `(O=3, P=L³, S)` — for each output orientation & vertex, the S
  flat indices of its stencil neighbours (Euclidean radius ≤ `radius`, default 1.05).
- `edge_out`     `(3, L³)`        — where each `(orientation, vertex)` output lands
  in flat order (a permutation of `0..N-1`).

Forward pass (`networks.py:302`):

    x        : (C_in, N)                  flat sites
    xg       : (C_in, O=3, P=L³, S)       x[:, gather]  — GATHER neighbours
    W        : (O=3, C_out, C_in, S)      one weight set per output orientation
    y        : (O=3, C_out, P=L³)         einsum over C_in and the S taps
    out      : (C_out, N)                 SCATTER (orient,vertex) back to flat N

The `(O=3, P=L³)` pair *is* the `(L,L,L)`-like structure, carried as gather/scatter
axes instead of tensor dimensions. Weight sharing across the P=L³ sites (one set per
orientation) gives exact integer-translation equivariance — the property a grid
conv gets for free from sliding.

## Mental model

A standard conv = (1) gather a fixed neighbour stencil at every site, (2) contract
with shared weights, (3) write back; the `(L,L,L)` shape just makes step (1) an
index shift. `GeoConv3D` runs the **same three steps**, but the toric-code neighbour
relation is the irregular half-offset one, so it can't be an axis shift — the gather
table replaces it and the data stays flat.

**One-liner:** `N` is `3L³` flattened; `edge_gather` is the explicit "where are my
neighbours" map that a grid conv would otherwise get implicitly from the `(L,L,L)`
layout.

## The stencil size S (=15) — the conv footprint

`S` is the kernel footprint: how many taps each conv sums over per output site.
It's the 3D-toric analog of "a 3×3×3 grid conv has 27 taps", but defined by
**physical Euclidean distance** on the half-offset edge lattice, not axis steps.
Set entirely by `radius_edge` / `radius_plaq` (default 1.05, `networks.py:116`).

Reference edge = x-edge at `(0.5, 0, 0)`. Within radius 1.05:

| shell | dist | count | what they are |
|---|---|---|---|
| self        | 0.000 | 1 | the edge itself (identity tap, `self_index=0`) |
| nearest     | 0.707 | 8 | perpendicular edges sharing a vertex (4 y + 4 z) |
| same-orient | 1.000 | 6 | same-orientation x-edges one cell away, ±x±y±z |

→ **S = 1 + 8 + 6 = 15.** Plaquette stencil S_p = 15 the same way.

Key distances:
- `0.707 = √½` = gap between an x-edge midpoint and a perpendicular y/z midpoint
  (displacement `(0.5,−0.5,0)`). These 8 are the genuine physical nearest
  neighbours — **exactly the shell a grid conv gets wrong** (half-offset error).
- `1.000` = same-orientation edge one full lattice step away.

Radius as the knob:

    radius < 0.707      → S = 1   (self only, pointwise)
    0.707 ≤ r < 1.0     → S = 9   (self + 8 perpendicular)
    1.0  ≤ r < 1.06     → S = 15  ← default (adds 6 same-orient ring)
    larger              → pulls in diagonal shells, S grows fast

1.05 is chosen to sit just past 1.0 (minimal physically-complete neighbourhood)
but below √2 ≈ 1.414 so it doesn't blow up into diagonal taps.

S multiplies every weight tensor `(O=3, C_out, C_in, S)`, so it scales every
param count: e.g. a 4→4 layer is `3·4·4·15 = 720`. Bump the radius and every
layer grows with the new S.

## L=2 caveat

Even this half-offset-exact kernel cannot separate the `+ê` and `−ê` neighbour at
L=2 (same site under PBC). Intrinsic to L=2, not to the kernel; the stencil radius
buys nothing extra there (`networks.py:49`).

## OBC: mask, don't wrap

`KernelManager3D` takes `bc` from the geometry. **PBC** wraps neighbour lookups
(`coord % L`) — every tap exists, so `edge_mask`/`plaq_mask` are all-ones. **OBC**
does *not* wrap: a stencil tap that lands outside the open box `[0,L−1]³` gets
gather index 0 (a dummy) and **mask 0**, so it contributes nothing to the einsum.
The per-orientation stencil (its taps and order, including `self_index=0`) is the
same one computed on the infinite lattice — only the per-site mask changes. So a
single weight set per orientation still applies everywhere; translation
equivariance just stops being exact at the boundary (as it must under OBC).

Two OBC-specific points:

- **Output sites.** Edges/plaquettes that don't exist in the open box are not
  emitted as outputs (the builder skips taps whose centre maps to `-1`). At
  Lx=Ly=Lz the surviving count is the same for all three orientations, so the
  `(O=3, P, S)` tensors stay rectangular (L=2: `P_edge=4`, `P_plaq=2`; N=12,
  N_plaq=6).
- **Plaquette gather is coordinate-based.** It no longer uses the dense-`L³`
  `pidx` arithmetic (which assumed every face exists); it looks neighbours up by
  centre coordinate via `geo._plaq_center_to_idx` (built from `geo.plaq_centers`),
  which works for both BCs and returns `-1` → mask 0 off the box. OBC `plaq_all`
  is filtered to **complete faces only**, so the Wilson 4-product has no `-1` to
  trip over. The PBC gather tables are byte-identical to the old `pidx` path.

`VanillaCNN`/`VanillaWilsonCNN` stay PBC-only (CIRCULAR padding + dense
`(3,L,L,L)` fold); `build_model` raises on Vanilla\*+OBC.

---

## How the net becomes A_v-symmetry-aware (the actual mechanism)

The target symmetry is the vertex operator: `ψ(s) = ψ(A_v s)`, where `A_v` flips
the spins on the 6 edges meeting a vertex (4 in 2D). Old approaches *tied weights*
to enforce this. This codebase instead does a **change of coordinates** so the
symmetry becomes automatic — no constraint.

**The trick = the Wilson 4-product** (`jnp.prod(x[..., plaq_idx], axis=-1)`,
`networks.py:555`; 2D `_Wilson_4spin_plaq`). Each plaquette `p ↦ B_p = ∏_{i∈p}s_i`.
`A_v` flips **0 or 2** edges of any plaquette, so `B_p ↦ (−1)^{0 or 2} B_p = B_p`.
The map E→P is therefore **identically A_v-invariant for any input**, with no
weight constraint. This survives the 4→6 edges/vertex jump unchanged: the count is
still 0-or-2 per plaquette in 3D.

Everything **downstream of Wilson** sees only `B_p`, so it is exactly invariant for
*any* weights — that's why the post-Wilson stack can be an unconstrained CNN.

### Three-stage flow (`ToricCNN_full`, `networks.py:522`)

    spins ±1  (E)
      │  CNN_noninvariant_3D  (E→E)   optional, identity-init   ← symmetry-BREAKING knob
      ▼
    edge features (E)
      │  Wilson 4-product     (E→P)                             ← the invariant change of coords
      ▼
    plaquette fluxes (P)   ── A_v-invariant from here down, any weights ──
      │  CNN_invariant_3D ×k (P→P) + ELU
      ▼
    mean → log ψ

### Pre-Wilson block = the approximate-symmetry knob

For the Wilson product to stay invariant, each output edge `x_i` must be **per-edge
sign-equivariant**: flip sign iff its *own* spin flips, untouched by neighbour
flips. That holds **only at identity-init**: kernel = 1 on the self tap, 0
elsewhere (`_geo_identity_init`), zero bias, and an **odd** activation
(`_normalised_sigmoid`, scaled so ±1↦±1, `networks.py:271`). Then `x_i = f(s_i)` and
the whole net is **exactly** A_v-invariant at step 0 (reduces to `ToricCNN`).

The instant training moves an off-diagonal weight, `x_i` mixes neighbour spins →
no longer a clean sign flip under `A_v` → Wilson product no longer invariant. So the
non-invariant block is the *tunable, warm-started* symmetry breaker — needed off the
`h_z=0` line where the true ground state isn't A_v-symmetric. Drop it entirely
(`ToricCNN`) for the exact-symmetry `h_x`-only sector.

Same 15-tap conv, opposite consequence by placement: mixing **raw spins** breaks
symmetry; mixing **fluxes** can't.

### Post-Wilson conv — worked example (L=3 PBC, plaq lattice)

`CNN_invariant_3D` = `GeoConv3D(lattice="plaq")`: identical gather/sum/scatter
machinery, now on the **3 face-normal plaquette sublattices**. Output plaquette
`p_0` = idx 0, orientation `c_out=0` (normal x). Its 15-tap gather row:

    [0,  45,48,27,30,  72,73,54,55,  18,6,2,1,3,9]
     self  └ 4 y-normal ┘ └ 4 z-normal ┘ └─ 6 x-normal ─┘
     d=0   d=0.707         d=0.707         d=1.0

    h[j,p0] = elu( Σ_{s=0..14} W[0, j, 0, s] · B_gather[s]  +  b[0,j] )
            = elu( w0·B(0) + w1·B(45) + … + w14·B(9) + b )

The d=0.707 taps are plaquettes **sharing an edge** with `p_0` (perpendicular faces
hinged on a common edge); the d=1.0 taps are **parallel faces** one cell over. Each
`B_p` is already A_v-invariant, so this weighted neighbour-sum builds *flux
correlations* without ever breaking symmetry. Weights `W[0,…]` are shared across all
27 x-normal plaquettes; `W[1]`,`W[2]` are independent stencils for y,z normals —
that per-orientation sharing is the 3-sublattice structure.

### Grid-conv invariant alternative (`ToricCNN_gridinv`, `networks.py:621`)

The geometry-exact `GeoConv3D(lattice="plaq")` above carries an **`O=3` orientation
axis** in its weights `W[o,c_out,c_in,s]` (independent stencils per face-normal) and
grows its footprint by adding 15-tap taps. To reach the topological long-range order
the receptive field must span the system (`Θ(L)`), which with the geometry-exact
stencil means either many layers or a hand-grown bowl — and the `O=3`×`S` cost.

`ToricCNN_gridinv` is the **2D-paper architecture generalised the obvious way**: after
the per-channel Wilson product it **folds the flux field onto the cube-cell grid** and
runs a *standard* `nn.Conv3D` with kernel scaled toward `L`.

- **The fold** (`plaq_grid_layout`, `networks.py`): each plaquette (centre
  `corner + ½ê_a + ½ê_b`, normal `o`) is anchored to cell `floor(centre)` and placed
  in orientation-channel `o` → a dense `(L,L,L,3·C)` tensor + an occupancy mask. This
  is the 2D "plaquettes live on the dual-lattice vertices" picture, except a 3D cell
  carries up to **3** plaquettes (one per normal), folded into channels. The
  within-cell ½-offsets between the three normals collapse onto the same vertex — the
  **half-offset approximation** that `GeoConv3D` avoids.
- **The conv**: `nn.Conv3D(features=w·O, kernel_size=L)` (override `--kernel_size`),
  `padding="CIRCULAR"` for PBC, zero (`"SAME"`) for OBC. Kernel `→ L` spans the lattice
  in one layer; the orientation mixing is now ordinary `3C→3C` channel mixing.
- **Readout**: final conv → `O` channels (width-1 per orientation), then a
  **masked mean** over occupied cells (OBC boundary cells excluded) → real `log ψ`.
- **Tradeoff**: not geometry-exact (the half-offset is dropped), but a fast,
  well-optimised conv with trivial kernel-to-`L` scaling. Still translation-equivariant
  on the cube grid; the pre-Wilson (noninv) block stays geometry-exact (small kernel).
  A/B it against `ToricCNN_full` (geometry-exact invariant block) at matched depth.

## 2D vs 3D — what actually differs

| | 2D (`model/networks.py`) | 3D (`tc3d/networks.py`) |
|---|---|---|
| pre-Wilson conv | `CNN_noninvariant` (link lattice) | `GeoConv3D(lattice="edge")` |
| post-Wilson conv | `CNN_invariant` (single square dual lattice) | `GeoConv3D(lattice="plaq")`, **3 face-normal sublattices** |
| conv class | two separate hand-built classes | one `GeoConv3D`, switched by `lattice` flag |
| Wilson rescale | only in **complex** branch (`10**1.5`); real branch none | none — real (`h_y=0`) sector |
| log ψ | complex supported | real |

Two real conceptual differences hide here: (1) 3D plaquettes are a **3-sublattice**
object (half-offset), so the geometry-exact stencil is needed *post*-Wilson too, not
just pre; (2) 3D commits to the **real positive (Perron–Frobenius, h_y=0) sector**,
so the complex-only `rescale` never enters. The `_normalised_sigmoid` factor
`(2+2e)/(e−1)≈2.79` is *not* a rescale analog — it only keeps the identity-init
pass-through (±1↦±1) in the pre-Wilson block.

## Extending to the fermionic TC — what transfers, what breaks

Decoration: `B̃_p = (∏_{∂p}σᶻ)·σˣ_{e+}·σˣ_{e−}` (`fermionic_decoration.py`).
**Vertex stars A_v are unchanged** — that single fact decides everything.

**The A_v machinery transfers unchanged.** ∏σᶻ over a plaquette boundary is still
exactly A_v-invariant (same 4 boundary edges, A_v still flips 0/2 of them), and the
fermionic GS is still a +1 A_v-eigenstate with A_v all-σˣ ⇒ `ψ(A_v s)=ψ(s)`
**exactly, phase included**. So the Wilson change-of-coordinates + invariant CNN
backbone stays correct. **Keep Wilson.**

**No "decorated Wilson" nonlinearity — and none is needed.** B̃_p contains σˣ
(off-diagonal): `B̃_p|s⟩ ∝ |s'⟩` with two flipped bits, so it is *not* a function of
one bitstring — there's no pointwise nonlinearity for it. But the diagonal
A_v-invariants are products of σᶻ over cycles, generated by plaquette boundaries +
global Wilson loops — **identical** to bosonic. The σˣ decoration adds **no new
diagonal invariant**, so the bare flux features already form a *complete*
A_v-invariant basis.

**What breaks: the sign (non-stoquasticity).** `⟨s'|(−J B̃_p)|s⟩ = −J·(∏_{∂p}σᶻ on
s) = ∓J` — positive off-diagonal elements occur ⇒ no Perron–Frobenius positivity ⇒
the fermionic GS has genuine negative/complex amplitudes. Since the diagonal feature
basis is unchanged, the **entire** bosonic→fermionic difference lives in the
amplitude **sign/phase** over each A_v orbit. The current **real `log ψ`** (h_y=0)
sector cannot represent it. The sign is a function on A_v orbits = a function of
`{B_p}` + Wilson loops, so a *complex* invariant CNN can represent it in principle;
the hard part is **optimisation** (sign-problem-flavoured).

**Plan (highest leverage first):**
1. **Clifford disentangler test** — does a finite-depth Clifford `U` map the bosonic
   stabiliser group to the fermionic one? Cheap GF(2) (reuse `_gf2_solve`). **Yes** →
   conjugate inputs by `U†`, sign vanishes, the real net works verbatim. **No** → the
   sign is topologically intrinsic; must be learned.
2. **Complexify** `GeoConv3D` (port the 2D complex branch: split activations, complex
   params), keep Wilson + invariant CNN as the A_v-invariant complex backbone; a
   separate phase head may help.
3. **Sampler / `E_loc`**: add the B̃_p 2-edge-flip off-diagonal moves (like extra hx).
4. **Validate** vs ED on the **FM ratio** (dressed σˣ string), gap, ⟨Mz⟩ — phases now
   matter, not just magnitudes (L=2 local; L≥3 Colab only).

**One-liner:** A_v is unchanged ⇒ Wilson + the whole symmetry trick survive; the
fermion lives entirely in the wavefunction **sign**, so the real-positive ansatz must
go complex — unless a finite-depth Clifford disentangler conjugates it back to the
stoquastic bosonic problem.

