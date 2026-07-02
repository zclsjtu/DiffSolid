# Dynamic solid mechanics

**Scope:** Explicit structural dynamics for nonlinear solids (mechanics-only or coupled to phase-field damage).

---

## 1. Semi-discrete momentum balance

After finite element discretisation of the displacement field $\mathbf{u}(\mathbf{X},t)$, the semi-discrete equations read

$$
\mathbf{M}\,\ddot{\mathbf{u}} + \mathbf{C}\,\dot{\mathbf{u}} + \mathbf{R}_\text{int}(\mathbf{u}) = \mathbf{f}_\text{ext}(t),
$$

where:

| Symbol | Meaning |
|--------|---------|
| $\mathbf{M}$ | Mass matrix (lumped diagonal in explicit integration) |
| $\mathbf{C}$ | Damping matrix (optional Rayleigh mass term) |
| $\mathbf{R}_\text{int}$ | Internal force vector from the weak form of equilibrium |
| $\mathbf{f}_\text{ext}$ | External loads (body forces, tractions, concentrated forces) |

For hyperelastic or UMAT-based materials, $\mathbf{R}_\text{int}$ is assembled from the same weak form as quasi-static analysis (see [Quasi-static mechanics](formulations.md)), evaluated at the current configuration $\mathbf{u}_n$.

Density $\rho$ enters through the lumped mass vector assembled from element contributions.

---

## 2. Central-difference (leap-frog) scheme

DiffSolid uses an explicit **central-difference / Verlet** integrator for structural dynamics:

$$
\mathbf{a}_n = \mathbf{M}^{-1}\!\left(\mathbf{f}_\text{ext}(t_n,\mathbf{u}_n) - \mathbf{R}_\text{int}(\mathbf{u}_n)\right) - \alpha_M\,\mathbf{v}_{n-1/2},
$$

$$
\mathbf{v}_{n+1/2} = \mathbf{v}_{n-1/2} + \Delta t\,\mathbf{a}_n,
\qquad
\mathbf{u}_{n+1} = \mathbf{u}_n + \Delta t\,\mathbf{v}_{n+1/2},
$$

where $\mathbf{v}_{n\pm1/2}$ are staggered half-step velocities and $\alpha_M$ is optional mass-proportional Rayleigh damping ($\mathbf{C} = \alpha_M \mathbf{M}$).

Dirichlet boundary conditions are enforced on $\mathbf{u}_{n+1}$ and on acceleration DOFs after each update.

This is the default explicit mechanics integrator when `dynamics="explicit"` is set on a simulation step (pure mechanics) or when `mechanics_integrator="explicit_central_difference"` is used in dynamic fracture.

---

## 3. Stability and time-step selection

For linearised wave propagation with maximum element frequency $\omega_\max$, the central-difference scheme is stable when

$$
\Delta t < \frac{2}{\omega_\max}.
$$

In practice DiffSolid estimates a critical step from mesh geometry and the elastic wave speed $c_p$:

$$
c_p = \sqrt{\frac{\lambda + 2\mu}{\rho}},
$$

with optional **CFL safety factor** $\gamma_\text{CFL} \in (0,1]$:

$$
\Delta t \leq \gamma_\text{CFL}\,\Delta t_\text{crit}.
$$

Element-wise CFL estimates are supported for heterogeneous meshes. Users may override `dt`, `cfl_safety`, or `wave_speed` on the step descriptor.

---

## 4. Lumped mass matrix

Explicit integration uses a **diagonal lumped mass** $\mathbf{M}_\text{diag}$ obtained by row-sum (or consistent) lumping of the element mass matrices. The acceleration update reduces to element-wise division:

$$
a_i = \frac{f_{\text{ext},i} - R_{\text{int},i}}{M_{\text{diag},i}}.
$$

Lumped mass is required for explicit dynamics with standard low-order elements. Density must be supplied through the material model or `SolidMechanics(density=...)`.

---

## 5. Weak form vs. time integrator

| Regime | Governing equation at each step | Time treatment |
|--------|--------------------------------|----------------|
| Quasi-static | $\mathbf{R}_\text{int}(\mathbf{u}) = \mathbf{f}_\text{ext}$ | Pseudo-time load stepping; Newton–Raphson |
| Explicit dynamic | $\mathbf{M}\ddot{\mathbf{u}} + \mathbf{R}_\text{int}(\mathbf{u}) = \mathbf{f}_\text{ext}(t)$ | Central difference; no Newton loop per step |
| Implicit dynamic | Same as explicit | Newmark-$\beta$ / $\gamma$ (when enabled) |

The **spatial discretisation** (small-strain, finite-strain, F-bar, B-bar, EAS, hourglass control) is shared with quasi-static mechanics. Only the time-discrete update changes.

---

## 6. Explicit material kernels and JAX scan backends

Stateless hyperelastic materials can integrate with **energy-based** explicit kernels (`scan_energy`). Materials with history variables (J2 plasticity, crystal plasticity, Fe–Fp) route to **stateful** scans that advance $(\mathbf{u}, \mathbf{v}, \text{state})$ jointly.

| Integrator keyword | Typical use |
|--------------------|-------------|
| `scan_energy` | Linear / Neo-Hookean elasticity, hourglass-stabilised elements |
| `scan_residual` | Stateless potentials with MPC or periodic BC offsets |
| `scan_stateful` | J2, crystal plasticity, F-bar / EAS with history |
| `python` | Debug loop; fallback when scan is disabled |

Periodic or MPC-driven boundary motion supplies a per-step offset sequence consumed by the scan driver.

---

## 7. Coupling to phase-field fracture

When damage field $d$ is active, explicit mechanics is typically combined with a **one-pass stagger** per time step:

1. Advance $\mathbf{u}$ with frozen or lagged $d_n$
2. Update driving force history $H_n$
3. Integrate damage PDE for $d_{n+1}$

See [Phase-field fracture](phase-field-fracture.md) for damage PDEs and the validated strategy matrix **S2–S4, S7**. Monolithic coupling of $(\mathbf{u}, d)$ in a single residual is **not** supported.

---

## References

1. Belytschko, T., Liu, W. K., & Moran, B. (2014). *Nonlinear Finite Elements for Continua and Structures*. Wiley. (Ch. 6 — explicit dynamics)
2. Hughes, T. J. R. (2000). *The Finite Element Method*. Dover. (Central difference, stability)
