# Phase-field fracture

**Scope:** Diffuse crack modelling coupled to nonlinear solid mechanics via a scalar damage field $d \in [0,1]$.

---

## 1. Coupled state variables

| Field | Role |
|-------|------|
| $\mathbf{u}$ | Displacement (mechanics) |
| $d$ | Phase-field damage ($d=0$ intact, $d\to1$ fully broken) |
| $H$ (optional) | Crack-driving history variable at Gauss points |

Mechanics and damage are solved with **staggered coupling only** (fixed-point or one-pass). Monolithic simultaneous solution of $(\mathbf{u}, d)$ is not implemented.

---

## 2. Degradation of elastic energy

Elastic strain energy is reduced by a degradation function $g(d)$:

$$
\Psi_\text{eff} = g(d)\,\mathcal{H}^+ + \mathcal{H}^- + \Psi_\text{frac}(d,\nabla d),
$$

where $\mathcal{H}^+$ and $\mathcal{H}^-$ are tensile and compressive (or volumetric / deviatoric) parts of the elastic energy, and $\Psi_\text{frac}$ is the regularised crack surface functional.

**Standard AT2 degradation:**

$$
g(d) = (1-d)^2 + \eta, \qquad \eta \sim 10^{-10}\ \text{(residual stiffness)}.
$$

AT1 and cohesive-zone variants use different crack surface densities $\gamma(d,\nabla d)$; see [Quasi-static constitutive §14.6](formulations.md#146-phase-field-fracture-coupling).

---

## 3. Strain energy splits (crack driving force)

The tensile energy $\mathcal{H}^+$ drives crack growth. Two common splits:

**Spectral (Miehe et al. 2010):**

$$
\mathcal{H}^\pm = \frac{\lambda}{2}\langle\mathrm{tr}\,\boldsymbol{\varepsilon}\rangle_\pm^2 + \mu\,\|\boldsymbol{\varepsilon}^\pm\|^2,
$$

with spectral projectors $\boldsymbol{\varepsilon}^\pm = \sum_\alpha \langle\varepsilon_\alpha\rangle_\pm\,\mathbf{n}_\alpha\otimes\mathbf{n}_\alpha$.

**Volumetric–deviatoric (Amor et al. 2009):**

$$
\mathcal{H}^+ = \frac{\kappa}{2}\langle\mathrm{tr}\,\boldsymbol{\varepsilon}\rangle_+^2 + \mu\,\|\mathrm{dev}\,\boldsymbol{\varepsilon}\|^2, \qquad
\mathcal{H}^- = \frac{\kappa}{2}\langle\mathrm{tr}\,\boldsymbol{\varepsilon}\rangle_-^2.
$$

Finite-strain splits (`vol_dev`, hybrid) follow the same principle on the appropriate strain measure.

The **driving force** $H$ used in the damage equation is derived from $\mathcal{H}^+$ (current value or historical maximum, depending on configuration).

---

## 4. Phase-field crack surface density

For the Ambrosio–Tortorelli (AT2) model with length scale $l_0$ and geometric function $c_w$:

$$
\gamma(d,\nabla d) = \frac{1}{c_w}\left(\frac{d^2}{l_0} + l_0|\nabla d|^2\right), \qquad c_w = 2.
$$

The critical energy release rate $G_c$ and $l_0$ set the fracture toughness and regularisation width. AT1 uses a linear $d$ term in $\gamma$ instead of $d^2/l_0$.

---

## 5. Damage field equations

Let $H$ denote the crack-driving history. The damage field satisfies one of the following PDE types:

### 5.1 Elliptic (quasi-static, S1 / S2)

$$
\frac{2G_c}{c_w l_0}\left(d - l_0^2 \Delta d\right) = 2 H, \qquad d \in [0,1].
$$

Solved implicitly with a variational inequality to enforce **irreversibility** $d_{n+1} \geq d_n$.

### 5.2 Parabolic viscous (S3 / S5)

$$
\eta\,\dot{d} = 2(H - H_c(d,\nabla d)), \qquad H_c = \frac{G_c}{c_w l_0}\left(d - l_0^2\Delta d\right).
$$

Viscosity $\eta > 0$ regularises fast crack growth. Explicit Euler or implicit integration depending on strategy.

### 5.3 Pseudo-parabolic (S6 / S7)

Adds a gradient-viscosity term with parameter $\tau$ for higher-order regularisation of the damage rate.

### 5.4 Inertial damage (S4)

Second-order damage dynamics with mass $\rho_d$ and damping $\mu_d$, integrated with a Verlet-type scheme coupled to explicit mechanics.

---

## 6. Staggered coupling algorithms

### 6.1 Fixed-point stagger (quasi-static, S1 / S5 / S6)

At each load step:

1. Given $d^k$, solve mechanics equilibrium for $\mathbf{u}^{k+1}$
2. Update $H$ from $\mathcal{H}^+(\boldsymbol{\varepsilon}(\mathbf{u}^{k+1}))$
3. Solve damage equation for $d^{k+1}$ with irreversibility
4. Repeat until $\|d^{k+1}-d^k\|$ and residual norms meet tolerance

Used for implicit quasi-static fracture with elliptic or parabolic damage.

### 6.2 One-pass stagger (dynamic, S2–S4 / S7)

At each time step $\Delta t$:

1. Advance explicit mechanics with frozen or lagged damage
2. Update history $H_n$
3. Advance damage field once (implicit elliptic sub-step or explicit viscous update)

No inner fixed-point loop — suitable for wave-driven dynamic fracture.

---

## 7. Strategy matrix (S1–S7)

Valid combinations enforced at solve time:

| ID | Mechanics | Damage PDE | Damage integrator | Coupling |
|----|-----------|------------|-------------------|----------|
| **S1** | quasi_static | elliptic | implicit | stagger_fixed_point |
| **S2** | explicit_central_difference | elliptic | implicit | stagger_one_pass |
| **S3** | explicit_central_difference | parabolic_viscous | explicit_euler | stagger_one_pass |
| **S4** | explicit_central_difference | inertial | explicit_verlet | stagger_one_pass |
| **S5** | quasi_static | parabolic_viscous | implicit | stagger_fixed_point |
| **S6** | quasi_static | pseudo_parabolic | implicit | stagger_fixed_point |
| **S7** | explicit_central_difference | pseudo_parabolic | explicit_euler_tau | stagger_one_pass |

Worked API examples: [Quick Start — Phase-field fracture](../quickstart.md#phase-field-fracture).

---

## 8. Irreversibility and damage solvers

Damage growth is **irreversible**: $d_{n+1} \ge d_n$ at every quadrature point / node. Elliptic damage problems are cast as **variational inequalities** solved with active-set Newton (VI solver).

Mechanics uses standard Newton–Raphson (quasi-static) or explicit integration (dynamic). Phase-field systems often employ separate linear solvers (e.g. AMGCL) for the damage Laplacian.

---

## 9. Regional control

Damage evolution and degradation laws may be restricted to mesh subsets via **phase-field sections** and **active zones** — useful for pre-cracked regions or preventing spurious damage away from the intended path.

---

## References

1. Ambrosio, L., & Tortorelli, V. M. (1990). Approximation of functional depending on jumps by elliptic functional via $\Gamma$-convergence. *Communications on Pure and Applied Mathematics*, 43(8), 999–1036.
2. Miehe, C., Hofacker, M., & Welschinger, F. (2010). A phase field model for rate-independent crack propagation. *CMAME*, 199(45–48), 2765–2778.
3. Amor, H., Marigo, J.-J., & Maurini, C. (2009). Regularized formulation of the variational brittle fracture with unilateral contact. *JMPS*, 57(8), 1209–1228.
