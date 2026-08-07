# How we know the fermionic h=0 plateau is a positive-sector state

*2026-08-07, branch `feat/fermionic-h0`. Companion to the L=2 benchmark runs
(`gridinv_fermionic_L2_PBC_hx0.0_hz0.0_*`, W&B `ea8c5c06ad6c` / `420331c3abba`)
and the sign-structure tests in `tests/test_fermionic.py`.*

## Context

At fermionic h=0, both hyperparameter sets (tune-rect winner and the old
workhorse) converged under SR to the **same** state at L=2 PBC:
E = −22.5211 vs exact E₀ = −32 (δ = 29.6%), with Vscore down to ~1e-9 — a
near-exact eigenstate of *something*, but not the ground state. A guard-off
escape probe (spike_factor 1e6, 400 hot-schedule iterations resumed from the
plateau) attempted two crossings and relaxed back: the trap is a genuine
optimization barrier, not a divergence-guard artifact.

The claim: this plateau state is the optimal wavefunction **confined to the
positive-sign half of the true GS support, with flat phase**. This note records
how that was established, since no single measurement proves it alone.

## Step 1 — fresh samples from the trapped state

Rebuild the variational state locally (L=2 evaluation is cheap; no ED
involved), load the plateau checkpoint's weights, and draw a fresh batch of
8192 configurations with the MCMC sampler. These are configurations the trapped
state's own |ψ|² considers probable — the question is where they live relative
to the *exact* ground state.

## Step 2 — the exact sign map (the load-bearing half)

The network cannot certify support membership or true signs; the exact solution
can. At h=0 the model is a commuting-stabilizer Hamiltonian, so the GS is a
stabilizer state and its amplitude structure is exactly computable by sign
propagation — no diagonalization:

- BFS over the stabilizer orbit from |0…0⟩ with sign +1:
  A_v flips (pure X⁶) preserve the sign; each B̃_p flip (X² on the
  body-diagonal pair) multiplies by (−1)^{z-parity of the state on ∂p}.
- This enumerates the full 32,768-state support at L=2 with the exact sign of
  every member (14,336 negative — the state is sign-full).
- The sign is an exact **function of the 24 flux tokens** (plaquette-boundary
  z-parities): the orbit collapses to 64 token classes, each with a unique
  sign. (Structural reason: the sign-carrying pair flips overlap plaquette
  boundaries oddly, so they *move* the tokens; star flips don't change tokens
  and don't change signs.)

Then, per sampled configuration: compute its token vector → **membership in
the 64 valid classes** decides "on support" (8192/8192 were), and the **class
lookup gives the exact GS sign** (all 8192 came back +1). The chain never left
the positive-sign half.

## Step 3 — the network's own phases

With log ψ(s) = a(s) + i·b(s), the magnitude is e^a and the phase e^{ib}. The
spread of b across the 8192 samples was ~1e-4 rad: every sampled amplitude is a
positive real number times one global phase. Caveat: a wavefunction is defined
only up to a *global* phase, so "the amplitudes are real" is not by itself
meaningful — the meaningful statements are (i) the phase **spread** is zero
(the state is positive up to gauge) and (ii) the exact GS demands **relative**
phase π between specific pairs of configurations, which a zero-spread phase
field cannot supply.

## Why the pieces together are a proof

- Step 3 alone: "the state is effectively positive where it has weight."
- Step 2 alone: "the chain never visited the negative-sign half."
- Sampler side: the move set *was* proposing entries into the negative half
  (every pair flip with odd boundary parity leads there — these are the B̃_p
  x-pair cluster moves added in `builders.build_sampler`), and Metropolis was
  rejecting them ⇒ |ψ(t)/ψ(s)|² ≈ 0 ⇒ the state has ~zero magnitude there.

Quantitative closure: for the uniform positive state restricted to the
positive-sign sector, ⟨A_v⟩ = 1 exactly (star flips act within the sector) and
⟨B̃_p⟩ = P(ε_p = +1 | positive sector). Measured: ⟨B̃_p⟩ = 0.605, uniform
across all 24 plaquettes, and

E = −8 (stars) − 24 × 0.605 = −22.52,

matching all three runs to the digit. Within the positive sector the
Hamiltonian acts stoquastically, which is why SR polishes this state to
Vscore ~1e-9: it has solved the sign-free restriction of the problem exactly.

## Consequence

The sign problem does not disappear with complex weights — it migrates from
the sampler into the optimizer. Escaping requires coherently growing π-phased
amplitude on ~44% of the support, and every smooth path there first raises the
energy variance (the "spikes" the divergence guard misreads as blow-ups).
Options, ranked in `MEMORY`/BLOG: token-pair quadratic phase head (stabilizer
phases are GF(2) quadratic forms over the tokens, so this is exact at h=0),
decoration annealing (1−λ)B_p + λB̃_p, or supervised phase pre-fit to the
exact token-sign map as a capacity test and initialization.
