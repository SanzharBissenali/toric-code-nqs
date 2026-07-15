# Distinguishing first- vs second-order transitions in the 3D toric-code NQS data

Working note for the analysis in `analysis/transition_order.ipynb`. Goal: given
the trained NQS sweeps we already have (per-field order parameters, Rényi-2,
magnetization, and — once pulled — final energies), decide whether we can tell a
**first-order** transition from a **second-order** one, and if not, say exactly
what extra data closes the gap.

## 1. The physics: why the two directions differ (and it's real)

$H = -J\sum_v A_v - J\sum_p B_p - h_x\sum_i\sigma^x_i - h_z\sum_i\sigma^z_i$.

In the **3D** toric code the two elementary excitations are *not* dual, unlike
2D: the electric charge $e$ is a **point** particle, the magnetic flux $m$ is a
**loop**. Condensing a point charge is a continuous (3D-Ising) affair; condensing
loops/membranes can be **first order**. QMC on the 3D TC in a field confirms this
directional asymmetry (Vidal et al., arXiv:1902.03908): one field direction gives
a 3D-Ising continuous transition, the other a first-order one. (Adding a *y*-field
component makes the transition strongly first order — arXiv:2402.15389 — but our
field is purely in the $xz$ plane, so the asymmetry here is the $e$/$m$ one.)

That the datasets already use a **string** Fredenhagen–Marcu ratio on the
second-order side (`bulkR1`, electric $\sigma^z$ loop) and a **membrane** on the
first-order side (`memA0.5`, magnetic $\sigma^x$ membrane) is itself the fingerprint
of point-vs-loop condensation — the right order parameter is dimensionally
different in the two cases.

**Our mapping (user's framing, consistent with the data on disk):**

| Transition | Sweep | Fixed | Order parameter dirs |
|---|---|---|---|
| **2nd order** (charge, 3D-Ising) | `hz` | `hx`∈{0,0.2,…,1.0} | `phase_hx*_bulkR1` (FM string), `phase_hx*_s2plaq` (S2) |
| **1st order** (flux/membrane) | `hx` | `hz`=0 | `phase_hz0.0_memA0.5` (FM membrane), `phase_hz0.0_s2plaq` (S2) |

Caveat carried into the notebook: the first-order side is sampled on a **single
line** (`hz`=0). One line cannot show the order *persists* along the transition;
a finite-`hz` first-order cut would strengthen any claim.

## 2. The key identity — energy kink ≡ magnetization jump

The field couples linearly, $H(h)=H_0-h\,M$ with $M=\sum_i\sigma^{(\text{sweep})}_i$.
Hellmann–Feynman gives

$$\frac{dE}{dh} = -\langle M\rangle = -N\,m,\qquad \frac{d^2E}{dh^2} = -N\frac{dm}{dh} = -\chi.$$

So the **energy's first derivative is (minus) the magnetization**, and:

- **1st order:** $m$ is discontinuous at $h_c$ → $dE/dh$ jumps → $E(h)$ has a
  **kink**. The jump in $m$ is the ground-state analogue of a *latent heat*.
  $d^2E/dh^2$ (a susceptibility) forms a $\delta$-function: at finite $L$ a peak of
  height $\sim L^{d}=L^3$ and width $\sim L^{-3}$.
- **2nd order:** $m$ is continuous → $E$ smooth, no kink; the peak in $d^2E/dh^2$
  grows only weakly (set by $\alpha/\nu$, tiny for 3D-Ising).

Two consequences for us: (i) pulling the final energy $E(h)$ per field point is
enough to run the primary diagnostic; (ii) computed $dE/dh$ is a **free
cross-check** against $-N\langle\sigma^{(\text{sweep})}\rangle$ from the stored
magnetization ($\langle M_z\rangle$ on the `hz`-sweeps, $\langle M_x\rangle$ =
`sx_mean` on the `hz`=0 `hx`-sweep, which is ~0 in the FM extracts and must come
from the run JSONs).

**Finite-size honesty:** every finite-$L$ ground state is analytic in $h$; the
kink/discontinuity is *emergent*. The tell is the sharpening — does the $d^2E/dh^2$
peak grow like the **volume** $L^3$ (first order) or barely (second order), and
does the $m$-jump stay finite as $L$ grows or shrink toward zero.

## 3. Diagnostics, and which our data can drive

| Diagnostic | 1st order | 2nd order | Have now? |
|---|---|---|---|
| **Energy kink** — jump in $dE/dh$; peak of $d^2E/dh^2\sim L^3$ | discontinuous slope | smooth | **after energy pull** |
| Order-param / S2 **jump** across $h_c$ | finite discontinuity as $L\to\infty$ | continuous | ✅ curves |
| **Derivative-peak height** $\max\lvert dO/dh\rvert$, $\lvert dS_2/dh\rvert$, $\lvert dm/dh\rvert$ | volume law $\sim L^3$ | $\sim L^{1/\nu}$/$L^{\gamma/\nu}$ (≈$L^{1.6}$) | ✅ `fd` |
| Transition **width** $w(L)$ | $\sim L^{-3}$ | $\sim L^{-1/\nu}\approx L^{-1.6}$ | ✅ `width` |
| $h_c(L)$ **drift** exponent | $\sim L^{-3}$ | $\sim L^{-1/\nu}$ | ✅ per-L `h_c` |
| **Rényi-S2 FSS** form | distinct 1st / weak-1st | 2nd-order form | ✅ S2 (matches arXiv:2506.16111) |
| **Binder** negative dip | dip deepens $\sim -L^{d}$ | fixed crossing | ❌ needs $\langle m^2\rangle,\langle m^4\rangle$ |
| Energy-histogram **bimodality** / hysteresis | double peak / branches | single | ❌ needs samples / re-train |

Gold-standard first-order signatures (latent heat via energy kink, bimodal
energy histogram, Binder negative dip, hysteresis from ordered/disordered
inits) — the first is reachable by pulling energies; the rest need moments,
sample histograms, or paired re-training runs.

## 4. What to pull, and how

The final energy and both magnetizations already live in each run JSON on the
cluster (`$PSCRATCH/tc_nqs/.../L*/{name}.json`): `curve.energy` (final = last
finite), `curve.energy_spread`, `observables.{sx_mean,sz_mean,A_v_mean,B_p_mean,
Vscore}`. `analysis/check_convergence.py` already reads exactly these fields and
knows the acceptance ladder (finite E below the trivial bound, `<A_v>≤1`, not
diverged), so energy extraction is a `--dump` option on it — pure JSON reads, no
NetKet, safe on a **login node** (never run 3D ED locally).

Pull the compact per-L curves into `results/energy_hz0.0/` (first order) and
`results/energy_hx${HX}/` (second order); the notebook globs them.

## 5. The honest bottom line

- $L=4\!\to\!7$ is under a factor of 2 in $L$. Cleanly separating $L^3$ from
  $L^{1.6}$ scaling over that range is hard; both merely "sharpen." Treat the
  power-law FSS as *suggestive*.
- The **energy kink / $m$-jump** is the strongest thing the existing runs can
  deliver: a *finite* magnetization jump that does **not** shrink with $L$, plus a
  $d^2E/dh^2$ peak growing toward $L^3$, is a first-order signature that does not
  rely on nailing a critical exponent. The absence of any jump (smooth $E$, $m$)
  on the `hz`-sweeps is the second-order counterpart.
- NQS caveat: variational states can **miss** a first-order jump by getting stuck
  in one phase (no tunneling across the coexistence barrier), or *fake* sharpness
  from optimization discontinuities. Cross-checking the energy kink against three
  independent probes (FM, S2, $m$) guards against reading an NQS artifact as
  physics; disagreement between probes is itself a red flag.

## References

- Vidal et al., *Quantum robustness and phase transitions of the 3D toric code in
  a field*, arXiv:1902.03908 — $e$/$m$ directional asymmetry; first- vs
  second-order.
- *Toric code in a parallel field on honeycomb/triangular lattices*,
  arXiv:2402.15389 — $y$-field → strongly first order.
- *Complete finite-size scaling theory of Rényi thermal entropy for second,
  first and weak-first order QPTs*, arXiv:2506.16111 — Rényi-entropy FSS forms
  by transition order; derivative-based unified FSS ($L^{d+z}$ peaks).
- *Entanglement scaling at first-order QPTs*, New J. Phys. (2018),
  10.1088/1367-2630/aab2db.
- *First- and second-order QPTs in the long-range AFM Ising chain*,
  arXiv:2409.02165 — NQS/VMC detecting a tricritical change of transition order.
- Standard first-order FSS: Binder & Landau; Janke, *Statistical Analysis of
  Simulations: Data Correlations and Error Estimation* — negative Binder dip,
  volume-law peaks, bimodal histograms.
