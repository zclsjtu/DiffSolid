# Quasi-static solid mechanics

**Version:** 2026-05 
**Scope:** 3D, plane-strain, plane-stress, and axisymmetric **quasi-static** formulations; constitutive models; nonlinear and linear solvers.

> **Related theory:** [Dynamic mechanics](dynamic-mechanics.md) · [Phase-field fracture](phase-field-fracture.md) · [Theory overview](index.md)

---

## 1. Introduction

This document covers quasi-static finite element formulations and shared infrastructure. For time-dependent solid mechanics and phase-field fracture, see the dedicated chapters linked above.

Topics in this file:

- **Finite element infrastructure** (§2–§3): notation, isoparametric mapping, quadrature, and displacement-gradient assembly.
- **Quasi-static formulations** (§4–§6): 3D, 2D, and axisymmetric solid mechanics under small-strain and finite-strain kinematics.
- **Anti-locking and special methods** (§4.3–§4.4, §6.3–§6.4, §10–§13): F-bar, B-bar, plane-stress condensation, periodic BCs, EAS, and F-bar patch.
- **Consistent tangent** (§7): automatic differentiation via `jax.jvp`.
- **Boundary conditions** (§8): Dirichlet and Neumann.
- **Post-processing** (§9): stress recovery, Hencky strain, VTK output.
- **Constitutive models** (§14): elasticity, plasticity, viscoelasticity, crystal plasticity; brief phase-field degradation summary (full theory in [phase-field-fracture.md](phase-field-fracture.md)).
- **Solvers** (§15–§16): Newton–Raphson, arc-length, linear backends.

---

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| $\Omega_0$, $\Omega$ | Reference and current configurations |
| $\mathbf{X}$, $\mathbf{x}$ | Material and spatial position vectors |
| $\mathbf{u}$ | Displacement field, $\mathbf{u} = \mathbf{x} - \mathbf{X}$ |
| $\mathbf{F}$ | Deformation gradient, $F_{iJ} = \partial x_i / \partial X_J$ |
| $J$ | Jacobian of deformation, $J = \det \mathbf{F}$ |
| $\boldsymbol{\varepsilon}$ | Small-strain tensor, $\varepsilon_{ij} = \tfrac{1}{2}(u_{i,j} + u_{j,i})$ |
| $\mathbf{P}$ | First Piola–Kirchhoff (PK1) stress, $P_{iJ}$ |
| $\boldsymbol{\sigma}$ | Cauchy stress, $\sigma_{ij}$ |
| $\Psi(\mathbf{F})$ | Strain-energy density function |
| $N_a(\boldsymbol{\xi})$ | Shape function of node $a$ in reference coordinates |
| $n_\text{en}$ | Number of nodes per element |
| $n_q$ | Number of quadrature points per element |
| $w_q$ | Quadrature weight at point $q$ |
| $\mathbf{j}$ | Isoparametric Jacobian matrix (reference-to-physical mapping) |
| $\text{JxW}_q$ | Quadrature weight scaled by physical volume, $\det(\mathbf{j}_q)\,w_q$ |
| $V_e$ | Physical volume of element $e$ |
| $\boldsymbol{\xi}$ | Reference (parametric) coordinates |
| $\delta(\cdot)$ | Variation of a field |
| $\text{sym}(\mathbf{A})$ | Symmetric part, $\tfrac{1}{2}(\mathbf{A} + \mathbf{A}^T)$ |
| $\mathbf{I}$ | Identity tensor |
| $\text{tr}(\mathbf{A})$ | Trace of tensor $\mathbf{A}$ |

Indices $i,j,k$ denote spatial (current-configuration) directions; $I,J,K$ denote material (reference-configuration) directions. Einstein summation convention is used unless otherwise noted.

---

## 3. Finite Element Discretization

### 3.1 Mesh and Shape Functions

The domain is partitioned into $n_e$ non-overlapping elements. Within each element the geometry and displacement are interpolated using the same set of shape functions (isoparametric formulation):

$$
\mathbf{X}(\boldsymbol{\xi}) = \sum_{n=1}^{n_\text{en}} N_n(\boldsymbol{\xi})\, \mathbf{X}_n, \qquad
\mathbf{u}(\boldsymbol{\xi}) = \sum_{n=1}^{n_\text{en}} N_n(\boldsymbol{\xi})\, \mathbf{u}_n.
$$

Shape function values and reference-coordinate gradients are precomputed by the `basix` library (FEniCS) and stored as arrays `shape_vals[q, n]` and `shape_grads_ref[q, n, i]`.

### 3.2 Isoparametric Mapping and Quadrature

The isoparametric Jacobian at quadrature point $q$ is

$$
j_{ij} = \sum_{n=1}^{n_\text{en}} X_n^i \frac{\partial N_n}{\partial \xi^j},
$$

and the physical quadrature weight is

$$
\text{JxW}_q = \det(\mathbf{j}_q)\, w_q.
$$

Physical shape-function gradients are obtained by the chain rule:

$$
\frac{\partial N_n}{\partial x^i} = \sum_j \frac{\partial N_n}{\partial \xi^j}\, (j^{-1})_{ji},
\quad \text{i.e.,} \quad
\texttt{shape\_grads} = \texttt{shape\_grads\_ref} \cdot \mathbf{j}^{-1}.
$$

Default Gauss quadrature orders:

| Element type | Quadrature points |
|---|---|
| TRI3, TET4 | 1 |
| QUAD4 | $2 \times 2$ |
| HEX8 | $2 \times 2 \times 2$ |
| Higher-order elements | Correspondingly higher order |

### 3.3 Displacement Gradient Computation

The displacement gradient at quadrature point $q$ of element $e$ is assembled as

$$
(u_\text{grad})_{q,i,j} = \sum_{n=1}^{n_\text{en}} u_n^i \frac{\partial N_n}{\partial x^j}\bigg|_q,
$$

implemented via the einsum `cni,cqnj->cqij` over cells $c$, nodes $n$, and spatial indices $i,j$.

---

## 4. Three-Dimensional Formulations

### 4.1 Small-Strain Mechanics (`equations/mechanics.py` — `MechanicsEquation`)

Under the small-strain assumption the linearised strain tensor is

$$
\boldsymbol{\varepsilon} = \text{sym}(\nabla \mathbf{u}) = \tfrac{1}{2}\!\left(\nabla \mathbf{u} + (\nabla \mathbf{u})^T\right), \quad \varepsilon_{ij} = \tfrac{1}{2}(u_{i,j} + u_{j,i}).
$$

The Cauchy stress $\boldsymbol{\sigma} = \boldsymbol{\sigma}(\boldsymbol{\varepsilon})$ is supplied by the user-defined constitutive model (UMAT). The weak form of quasi-static equilibrium (body forces neglected) reads

$$
\int_\Omega \boldsymbol{\sigma} : \delta\boldsymbol{\varepsilon}\, dV = 0 \quad \forall\, \delta\mathbf{u} \in \mathcal{V}_0,
$$

which, after discretisation, yields the nodal residual

$$
R_a^i = \sum_{q=1}^{n_q} \sigma^{ij}(q)\, \frac{\partial N_a}{\partial x^j}\bigg|_q \text{JxW}_q.
$$

This is implemented via the einsum `qij,qnj,q->ni`.

### 4.2 Finite-Strain Hyperelasticity (`equations/mechanics.py` — `HyperElasticEquation`)

The deformation gradient is

$$
\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}, \quad F_{iJ} = \delta_{iJ} + \frac{\partial u_i}{\partial X_J},
$$

where $\nabla$ denotes the material gradient. For a hyperelastic material with strain-energy density $\Psi(\mathbf{F})$, the first Piola–Kirchhoff stress is obtained by automatic differentiation:

$$
\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}, \quad P_{iJ} = \frac{\partial \Psi}{\partial F_{iJ}},
$$

computed via `jax.grad` applied to the user-supplied energy function. The weak form in the reference configuration (total Lagrangian) is

$$
\int_{\Omega_0} \mathbf{P} : \delta\mathbf{F}\, dV = \int_{\Omega_0} P_{iJ}\, \delta u_{i,J}\, dV = 0,
$$

giving the nodal residual

$$
R_a^i = \sum_{q=1}^{n_q} P^{iJ}(q)\, \frac{\partial N_a}{\partial X^J}\bigg|_q \text{JxW}_q.
$$

### 4.3 F-bar Volumetric Anti-Locking — 3D (`equations/fbar.py` — `Fbar3DMechanicsEquation`)

Low-order elements (HEX8, TET4) suffer from volumetric locking when the material is nearly incompressible. The F-bar method (de Souza Neto, Perić & Owen 1996) replaces the local Jacobian with an element-averaged value while preserving the isochoric part of the deformation gradient.

**Algorithm:**

1. Compute the local Jacobian at each quadrature point:
$$
J_q = \det(\mathbf{F}_q).
$$

2. Compute the element-average Jacobian (volume-weighted):
$$
\bar{J} = \frac{1}{V_e} \sum_{q=1}^{n_q} J_q\, \text{JxW}_q, \qquad V_e = \sum_{q=1}^{n_q} \text{JxW}_q.
$$

3. Compute the scaling factor:
$$
\alpha_q = \left(\frac{\bar{J}}{J_q}\right)^{1/3}.
$$

4. Form the modified deformation gradient:
$$
\bar{\mathbf{F}}_q = \alpha_q\, \mathbf{F}_q, \qquad \det(\bar{\mathbf{F}}_q) = \alpha_q^3 J_q = \bar{J}.
$$

5. Evaluate the PK1 stress from the modified gradient: $\mathbf{P} = \text{UMAT}(\bar{\mathbf{F}})$.

The residual has the same form as in Section 4.2 with $\mathbf{P}$ evaluated at $\bar{\mathbf{F}}$.

### 4.4 B-bar Mean-Dilatation — 3D (`equations/bbar.py` — `BbarMechanicsEquation`)

The B-bar method (Hughes 1980; Simo & Hughes §4.4) modifies the strain operator to use an element-averaged volumetric part, preventing locking in the small-strain setting.

**Mean-dilatation shape-function gradient** (precomputed per element):

$$
\bar{\mathbf{b}}_a = \frac{1}{V_e} \int_{\Omega_e} \nabla N_a\, dV = \frac{1}{V_e} \sum_{q=1}^{n_q} \nabla N_a|_q\, \text{JxW}_q.
$$

**Element-average divergence:**

$$
\overline{\text{div}}(\mathbf{u}) = \sum_{a=1}^{n_\text{en}} \bar{\mathbf{b}}_a \cdot \mathbf{u}_a.
$$

**Modified displacement gradient** at quadrature point $q$:

$$
\overline{\nabla \mathbf{u}}(q) = \nabla \mathbf{u}(q) + \frac{1}{d}\!\left[\overline{\text{div}}(\mathbf{u}) - \text{div}(\mathbf{u})(q)\right] \mathbf{I},
$$

where $d = 3$ in three dimensions. The modified strain is $\bar{\boldsymbol{\varepsilon}} = \text{sym}(\overline{\nabla \mathbf{u}})$, and the residual is

$$
R_a^i = \sum_{q=1}^{n_q} \sigma^{ij}(\bar{\boldsymbol{\varepsilon}}(q))\, \frac{\partial N_a}{\partial x^j}\bigg|_q \text{JxW}_q.
$$

---

## 5. Two-Dimensional Formulations

### 5.1 Plane Strain — Finite Strain with F-bar (`equations/fbar.py` — `PlaneStrainFbarMechanicsEquation`)

Under the plane-strain constraint the out-of-plane stretch is identically unity, $F_{33} = 1$. The 2D mesh carries DOFs $(u_x, u_y)$; the full 3D deformation gradient is reconstructed as

$$
\mathbf{F}_\text{3D} = \begin{pmatrix} \mathbf{F}_\text{2D} & \mathbf{0} \\ \mathbf{0}^T & 1 \end{pmatrix},
$$

where $\mathbf{F}_\text{2D}$ is the $2\times 2$ in-plane gradient computed from the 2D mesh.

**F-bar algorithm for plane strain:**

1. $J_q = \det(\mathbf{F}_\text{2D})$ (in-plane area ratio).
2. $\bar{J} = \dfrac{1}{V_e}\displaystyle\sum_q J_q\,\text{JxW}_q$.
3. Scaling exponent is $\tfrac{1}{2}$ (two in-plane degrees of freedom):
$$
\alpha_q = \left(\frac{\bar{J}}{J_q}\right)^{1/2}.
$$
4. $\bar{\mathbf{F}}_\text{2D} = \alpha_q\,\mathbf{F}_\text{2D}$, so $\det(\bar{\mathbf{F}}_\text{2D}) = \alpha_q^2 J_q = \bar{J}$.
5. $\bar{\mathbf{F}}_\text{3D} = \text{diag}(\bar{\mathbf{F}}_\text{2D},\, 1)$, giving $\det(\bar{\mathbf{F}}_\text{3D}) = \bar{J}$.
6. $\mathbf{P}_\text{3D} = \text{UMAT}(\bar{\mathbf{F}}_\text{3D})$.

Only the in-plane block of the PK1 stress contributes to the 2D residual:

$$
R_a^i = \sum_{q=1}^{n_q} P_\text{2D}^{ij}(q)\, \frac{\partial N_a}{\partial X^j}\bigg|_q \text{JxW}_q, \quad i,j \in \{1,2\},
$$

where $\mathbf{P}_\text{2D} = \mathbf{P}_\text{3D}[:2,:2]$.

### 5.2 Plane Stress

Under the plane-stress constraint $\sigma_{33} = \sigma_{13} = \sigma_{23} = 0$. For small-strain problems the out-of-plane strain $\varepsilon_{33}$ is determined by the constitutive relation; for finite-strain problems $F_{33}$ is solved as an additional local unknown (condensed at the element level). The 2D DOFs remain $(u_x, u_y)$; the out-of-plane component is handled internally by the constitutive driver.

### 5.3 B-bar for 2D

The B-bar formulation extends directly to two dimensions with $d = 2$ in the modified gradient formula:

$$
\overline{\nabla \mathbf{u}}(q) = \nabla \mathbf{u}(q) + \frac{1}{2}\!\left[\overline{\text{div}}(\mathbf{u}) - \text{div}(\mathbf{u})(q)\right] \mathbf{I}_\text{2D}.
$$

All other steps are identical to the 3D case (Section 4.4).

---

## 6. Axisymmetric Formulations

The 2D mesh lies in the $r$–$z$ plane (code convention: $x \equiv r$, $y \equiv z$). DOFs are $(u_r, u_z)$. The circumferential direction $\theta$ is eliminated by symmetry, but the hoop components of stress and strain must be retained explicitly.

### 6.1 Kinematics: Deformation Gradient and Strain

#### 6.1.1 Finite-Strain Axisymmetric

The full 3D deformation gradient in the $r$–$\theta$–$z$ ordering is

$$
\mathbf{F} = \begin{pmatrix}
F_{rr} & 0 & F_{rz} \\
0 & F_{\theta\theta} & 0 \\
F_{zr} & 0 & F_{zz}
\end{pmatrix},
$$

with components

$$
F_{rr} = 1 + \frac{\partial u_r}{\partial R}, \quad
F_{rz} = \frac{\partial u_r}{\partial Z}, \quad
F_{zr} = \frac{\partial u_z}{\partial R}, \quad
F_{zz} = 1 + \frac{\partial u_z}{\partial Z},
$$

and the hoop stretch (no spatial derivative required):

$$
F_{\theta\theta} = 1 + \frac{u_r}{R}.
$$

The Jacobian of deformation is

$$
J = \det(\mathbf{F}) = \det\!\begin{pmatrix} F_{rr} & F_{rz} \\ F_{zr} & F_{zz} \end{pmatrix} \cdot F_{\theta\theta}
= (F_{rr} F_{zz} - F_{rz} F_{zr})\, F_{\theta\theta}.
$$

To avoid a singularity on the axis of symmetry, the code uses $r_\text{safe} = \max(r,\, 10^{-14})$ when evaluating $F_{\theta\theta}$.

#### 6.1.2 Small-Strain Axisymmetric

The 3D strain tensor in the $r$–$\theta$–$z$ ordering is

$$
\boldsymbol{\varepsilon} = \begin{pmatrix}
\varepsilon_{rr} & 0 & \varepsilon_{rz} \\
0 & \varepsilon_{\theta\theta} & 0 \\
\varepsilon_{rz} & 0 & \varepsilon_{zz}
\end{pmatrix},
$$

with

$$
\varepsilon_{rr} = \frac{\partial u_r}{\partial r}, \quad
\varepsilon_{zz} = \frac{\partial u_z}{\partial z}, \quad
\varepsilon_{rz} = \tfrac{1}{2}\!\left(\frac{\partial u_r}{\partial z} + \frac{\partial u_z}{\partial r}\right), \quad
\varepsilon_{\theta\theta} = \frac{u_r}{r}.
$$

### 6.2 Weak Form and Hoop Term

#### 6.2.1 Finite-Strain Weak Form

The physical volume element in axisymmetry is $dV = R\,dR\,dZ\,d\theta$; integrating over $\theta \in [0, 2\pi)$ gives $dV_\text{phys} = 2\pi R\,\text{JxW}$. The factor $2\pi$ cancels from both sides of the equilibrium equation, leaving an effective weight $R\,\text{JxW}$.

The variation of the deformation gradient is

$$
\delta F_{iJ} = \frac{\partial \delta u_i}{\partial X_J} \quad (i,J \neq \theta),
\qquad
\delta F_{\theta\theta} = \frac{\delta u_r}{R}.
$$

The virtual work $\delta W = \int_{\Omega_0} P_{iJ}\,\delta F_{iJ}\,dV$ yields the nodal residuals:

$$
R_I^r = \sum_q \left[
 \left(P_{rR}\,\frac{\partial N_I}{\partial R} + P_{rZ}\,\frac{\partial N_I}{\partial Z}\right) R_q
 + P_{\theta\Theta}\, N_I
\right] \text{JxW}_q,
$$

$$
R_I^z = \sum_q \left[
 P_{zR}\,\frac{\partial N_I}{\partial R} + P_{zZ}\,\frac{\partial N_I}{\partial Z}
\right] R_q\, \text{JxW}_q.
$$

The hoop term $P_{\theta\Theta}\,\delta u_r / R$ contributes $P_{\theta\Theta}\,N_I / R$ to the integrand; multiplied by the physical weight $R\,\text{JxW}$, the factor $R$ cancels, leaving $P_{\theta\Theta}\,N_I\,\text{JxW}$.

#### 6.2.2 Small-Strain Weak Form

The analogous residuals for the small-strain case are

$$
R_I^r = \sum_q \left[
 \left(\sigma_{rr}\,\frac{\partial N_I}{\partial r} + \sigma_{rz}\,\frac{\partial N_I}{\partial z}\right) r_q
 + \sigma_{\theta\theta}\, N_I
\right] \text{JxW}_q,
$$

$$
R_I^z = \sum_q \left[
 \sigma_{zr}\,\frac{\partial N_I}{\partial r} + \sigma_{zz}\,\frac{\partial N_I}{\partial z}
\right] r_q\, \text{JxW}_q.
$$

### 6.3 F-bar for Axisymmetric Problems (`equations/axisymmetric.py` — `AxisymmetricFbarKinematicMechanicsEquation`)

Because the axisymmetric problem is physically three-dimensional, the F-bar exponent is $\tfrac{1}{3}$.

**Algorithm:**

1. Compute the local Jacobian:
$$
J_q = (F_{rr} F_{zz} - F_{rz} F_{zr})\, F_{\theta\theta}.
$$

2. Compute the element-average Jacobian with the physical volume weight $R_q\,\text{JxW}_q$:
$$
\bar{J} = \frac{\displaystyle\sum_q J_q\, R_q\, \text{JxW}_q}{\displaystyle\sum_q R_q\, \text{JxW}_q}.
$$

3. Scaling factor:
$$
\alpha_q = \left(\frac{\bar{J}}{J_q}\right)^{1/3}.
$$

4. All components of $\mathbf{F}$ are scaled uniformly:
$$
\bar{F}_{rr} = \alpha_q F_{rr}, \quad
\bar{F}_{rz} = \alpha_q F_{rz}, \quad
\bar{F}_{zr} = \alpha_q F_{zr}, \quad
\bar{F}_{zz} = \alpha_q F_{zz}, \quad
\bar{F}_{\theta\theta} = \alpha_q F_{\theta\theta}.
$$

5. $\mathbf{P} = \text{UMAT}(\bar{\mathbf{F}})$.

The weak form and residual assembly are identical to Section 6.2.1 with $\mathbf{P}$ evaluated at $\bar{\mathbf{F}}$.

### 6.4 B-bar for Axisymmetric Small-Strain

The volumetric strain at a quadrature point is

$$
\varepsilon_v = \varepsilon_{rr} + \varepsilon_{\theta\theta} + \varepsilon_{zz}.
$$

The element-average volumetric strain (with physical weight $r\,\text{JxW}$) is

$$
\bar{\varepsilon}_v = \frac{1}{V_e} \int_{\Omega_e} \varepsilon_v\, dV
= \frac{\displaystyle\sum_q \varepsilon_v(q)\, r_q\, \text{JxW}_q}{\displaystyle\sum_q r_q\, \text{JxW}_q}.
$$

The modified strain tensor is

$$
\bar{\boldsymbol{\varepsilon}} = \boldsymbol{\varepsilon} + \frac{1}{3}\!\left(\bar{\varepsilon}_v - \varepsilon_v\right) \mathbf{I},
$$

which replaces the local volumetric part with the element average while leaving the deviatoric part unchanged. The residual is assembled using $\boldsymbol{\sigma}(\bar{\boldsymbol{\varepsilon}})$ in the expressions of Section 6.2.2.

---

## 7. Consistent Tangent via Automatic Differentiation

The Newton–Raphson linearisation requires the consistent tangent stiffness matrix

$$
\mathbf{K} = \frac{d\mathbf{R}}{d\mathbf{u}}.
$$

Rather than deriving and implementing the material and geometric tangent terms by hand, diffsolid computes $\mathbf{K}$ using JAX forward-mode automatic differentiation:

$$
\mathbf{K}\,\Delta\mathbf{u} = \frac{d\mathbf{R}(\mathbf{u} + t\,\Delta\mathbf{u})}{dt}\bigg|_{t=0},
$$

which is evaluated by a single call to `jax.jvp(R, (u,), (du,))`. This approach is exact (to floating-point precision), requires no hand derivation, and automatically handles any user-supplied constitutive model that is differentiable in JAX. The full stiffness matrix is assembled column-by-column by applying `jax.jvp` with unit perturbation vectors, or more efficiently via `jax.linearize` over the full DOF vector.

This design means that any UMAT written as a pure JAX function automatically yields a consistent tangent, including path-dependent models that carry internal state variables (provided the state is treated as fixed during the linearisation step, i.e., the algorithmic tangent is used).

---

## 8. Boundary Conditions

### 8.1 Dirichlet Conditions

Essential boundary conditions are enforced by row elimination. Each Dirichlet constraint is stored as a triplet $(\text{node\_inds},\, \text{vec\_inds},\, \text{vals})$ specifying the constrained nodes, the constrained DOF component, and the prescribed value. During assembly, the corresponding rows of the residual and stiffness matrix are replaced by the constraint equations $u_i^{(a)} = \bar{u}_i^{(a)}$.

Multiple Dirichlet variables (e.g., displacement and temperature in a coupled problem) are supported; each variable maintains its own constraint list.

### 8.2 Neumann (Traction) Conditions

Natural boundary conditions prescribe a traction $\mathbf{t}$ on a portion $\partial\Omega_t$ of the boundary. The traction residual is

$$
R_a^i = \int_{\partial\Omega_t} t^i\, N_a\, dS.
$$

In the finite-strain setting the physical area element $dS$ is related to the reference area element $dA$ via Nanson's formula:

$$
\mathbf{n}\, dS = J\, \mathbf{F}^{-T}\, \mathbf{N}\, dA,
$$

where $\mathbf{N}$ is the outward unit normal in the reference configuration. The implementation computes the Nanson scaling factor on each boundary face as

$$
\text{nanson\_scale} = \left\|\mathbf{n}_\text{ref}\, \mathbf{j}^{-1}\right\| \det(\mathbf{j})\, w_\text{face},
$$

where $\mathbf{j}$ is the surface Jacobian and $w_\text{face}$ is the face quadrature weight. Face quadrature points and weights are generated by `basix` for the appropriate facet type.

---

## 9. Post-Processing

### 9.1 Stress Recovery

**Cauchy stress from PK1:** Given the first Piola–Kirchhoff stress $\mathbf{P}$ and the deformation gradient $\mathbf{F}$, the Cauchy stress is recovered by the push-forward

$$
\boldsymbol{\sigma} = \frac{1}{J}\, \mathbf{P}\, \mathbf{F}^T, \quad \sigma_{ij} = \frac{1}{J}\, P_{iK}\, F_{jK}.
$$

**Logarithmic (Hencky) strain:** The logarithmic strain tensor is computed from the right Cauchy–Green tensor $\mathbf{C} = \mathbf{F}^T \mathbf{F}$ via spectral decomposition:

$$
\boldsymbol{\varepsilon}_\text{log} = \frac{1}{2}\ln(\mathbf{F}^T \mathbf{F}) = \sum_{\alpha=1}^{3} \frac{1}{2}\ln(\lambda_\alpha^2)\, \mathbf{n}_\alpha \otimes \mathbf{n}_\alpha,
$$

where $\lambda_\alpha$ and $\mathbf{n}_\alpha$ are the principal stretches and principal directions obtained by eigendecomposition of $\mathbf{C}$.

**Internal (state) variables** are stored as arrays of shape `(n_cells, n_quads, n_state)` and are updated in-place at each converged load step.

### 9.2 Output Format

Solution fields and post-processed quantities are written to VTK format via the `save_sol` and `save_sol_extended` functions (`io/` module). The VTK files can be visualised in ParaView or similar tools. Nodal displacement fields, element-averaged stresses, logarithmic strains, and user-defined state variables can all be exported. Time-history data (e.g., reaction forces, average stresses) can be written to CSV using the `history_csv` argument of the `solve` function.

---

## 10. Plane-Stress Formulation

### 10.1 Kinematic Constraint

Under the plane-stress assumption the out-of-plane normal stress vanishes:

$$
\sigma_{33} = 0 \quad \text{(small strain)}, \qquad P_{33} = 0 \quad \text{(finite strain)}.
$$

The 2D mesh carries DOFs $(u_x, u_y)$. The out-of-plane strain $\varepsilon_{33}$ (or deformation gradient component $F_{33}$) is not an independent DOF; it is determined locally at each quadrature point by enforcing the plane-stress constraint.

### 10.2 Small-Strain Plane Stress (`PlaneStressMechanicsEquation`)

Given the in-plane displacement gradient $\nabla_\text{2D}\mathbf{u}$, the 3D strain tensor is assembled as

$$
\boldsymbol{\varepsilon}_\text{3D} = \begin{pmatrix} \varepsilon_{11} & \varepsilon_{12} & 0 \\ \varepsilon_{12} & \varepsilon_{22} & 0 \\ 0 & 0 & \varepsilon_{33} \end{pmatrix},
$$

where $\varepsilon_{33}$ is the unknown. At each quadrature point, $\varepsilon_{33}$ is found by solving the scalar nonlinear equation

$$
f(\varepsilon_{33}) \;:=\; \sigma_{33}\!\left(\boldsymbol{\varepsilon}_\text{2D},\, \varepsilon_{33};\, \text{state}\right) = 0,
$$

using a local Newton iteration (maximum 20 iterations, tolerance $10^{-12}$) implemented via `jax.lax.custom_root`. The consistent tangent with respect to the in-plane strains is obtained by the implicit function theorem (IFT):

$$
\frac{\partial \varepsilon_{33}}{\partial \varepsilon_{\alpha\beta}} = -\left(\frac{\partial \sigma_{33}}{\partial \varepsilon_{33}}\right)^{-1} \frac{\partial \sigma_{33}}{\partial \varepsilon_{\alpha\beta}}, \quad \alpha,\beta \in \{1,2\}.
$$

The `tangent_solve` function in `custom_root` implements this as a scalar division: `return y / df` where `df = ∂σ₃₃/∂ε₃₃`. Once $\varepsilon_{33}$ is known, the full 3D stress is evaluated and the in-plane block $\boldsymbol{\sigma}_\text{2D}$ is extracted for the residual assembly.

### 10.3 Finite-Strain Plane Stress (`PlaneStressFiniteStrainEquation`)

The 2D deformation gradient $\mathbf{F}_\text{2D}$ is embedded into a 3D gradient with unknown $F_{33}$:

$$
\mathbf{F}_\text{3D} = \begin{pmatrix} F_{11} & F_{12} & 0 \\ F_{21} & F_{22} & 0 \\ 0 & 0 & F_{33} \end{pmatrix}.
$$

At each quadrature point, $F_{33}$ is found by solving

$$
g(F_{33}) \;:=\; P_{33}\!\left(\mathbf{F}_\text{2D},\, F_{33};\, \text{state}\right) = 0,
$$

again via `jax.lax.custom_root` with IFT tangent. The initial guess is $F_{33} = 1$ (undeformed). The in-plane PK1 block $\mathbf{P}_\text{2D}$ is used in the standard residual formula (Section 4.2).


---

## 11. Periodic Boundary Conditions

### 11.1 Multi-Point Constraint Formulation

Periodic boundary conditions (PBCs) link the displacement on opposite faces of a unit cell. For a pair of master node $m$ and slave node $s$ related by the periodicity vector $\mathbf{L}$, the constraint is

$$
\mathbf{u}_s = \mathbf{u}_m + \bar{\boldsymbol{\varepsilon}}\, \mathbf{L},
$$

where $\bar{\boldsymbol{\varepsilon}}$ is the prescribed macroscopic strain (or $\bar{\mathbf{F}}$ in the finite-strain case). These are linear multi-point constraints (MPCs).

### 11.2 Implementation via Constraint Matrix

The full DOF vector $\mathbf{u}_\text{full} \in \mathbb{R}^{N_\text{full}}$ is expressed in terms of a reduced DOF vector $\mathbf{u}_\text{red} \in \mathbb{R}^{N_\text{red}}$ and a prescribed offset $\mathbf{g}$:

$$
\mathbf{u}_\text{full} = \mathbf{P}\, \mathbf{u}_\text{red} + \mathbf{g},
$$

where $\mathbf{P} \in \mathbb{R}^{N_\text{full} \times N_\text{red}}$ is a sparse Boolean constraint matrix (stored as a `scipy.sparse.csr_matrix`). Each slave DOF row of $\mathbf{P}$ contains a single entry pointing to the corresponding master DOF; master and interior DOF rows are identity rows.

The reduced residual and stiffness are obtained by the congruence transformation:

$$
\mathbf{R}_\text{red} = \mathbf{P}^T \mathbf{R}_\text{full}, \qquad
\mathbf{K}_\text{red} = \mathbf{P}^T \mathbf{K}_\text{full}\, \mathbf{P}.
$$

The Newton iteration is performed on $\mathbf{u}_\text{red}$; the full solution is recovered by $\mathbf{u}_\text{full} = \mathbf{P}\,\mathbf{u}_\text{red} + \mathbf{g}$ after each solve.

### 11.3 Node Pairing

Periodic node pairs are identified geometrically: for each node on the $-x$ face, the matching node on the $+x$ face is found by searching for the node whose coordinates differ by exactly $L_x$ in the $x$-direction and are identical in $y$ and $z$ (within a tolerance). The same procedure is applied to the $y$ and $z$ face pairs. Corner and edge nodes shared by multiple face pairs are handled by assigning them to a single master chain.


---

## 12. Enhanced Assumed Strain (EAS) Method

### 12.1 Motivation

The standard isoparametric displacement formulation exhibits parasitic shear locking in bending-dominated problems and volumetric locking in nearly incompressible problems. The EAS method (Simo & Rifai 1990) enriches the strain field with element-level enhanced modes that are orthogonal to the stress field, eliminating these locking pathologies without introducing additional global DOFs.

### 12.2 Enhanced Strain Field

The total compatible strain (or displacement gradient) at a quadrature point is augmented by an enhanced part:

$$
\tilde{\mathbf{H}}(\boldsymbol{\xi}) = \frac{j_0}{j(\boldsymbol{\xi})}\, \mathbf{M}_\text{ref}(\boldsymbol{\xi})\, \mathbf{J}_0^{-1},
$$

where $j_0 = \det(\mathbf{j})|_{\boldsymbol{\xi}=\mathbf{0}}$ is the Jacobian determinant at the element centroid, $j(\boldsymbol{\xi}) = \det(\mathbf{j}(\boldsymbol{\xi}))$ is the local Jacobian determinant, $\mathbf{M}_\text{ref}(\boldsymbol{\xi})$ is the matrix of reference-coordinate enhanced modes, and $\mathbf{J}_0$ is the Jacobian matrix at the centroid. The factor $j_0/j$ ensures the orthogonality condition $\int_{\Omega_e} \boldsymbol{\sigma} : \tilde{\mathbf{H}}\,dV = 0$ for any stress field $\boldsymbol{\sigma}$ that is constant over the element.

The modified deformation gradient passed to the constitutive model is

$$
\bar{\mathbf{F}} = \mathbf{I} + \nabla\mathbf{u} + \tilde{\mathbf{H}}(\boldsymbol{\xi};\, \boldsymbol{\alpha}),
$$

where $\boldsymbol{\alpha}$ is the vector of element-level enhanced DOFs.

### 12.3 Enhanced Mode Matrices

The reference-coordinate mode matrix $\mathbf{M}_\text{ref}$ is defined per element type:

**QUAD4 — EAS7 (plane strain and axisymmetric):**

$$
\mathbf{M}_\text{ref} = \begin{pmatrix}
\xi & 0 & \eta & 0 & \xi\eta & 0 & 0 \\
0 & \eta & 0 & \xi & 0 & \xi\eta & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & \xi\eta/2
\end{pmatrix}^T \in \mathbb{R}^{7 \times 3},
$$

providing 4 normal-strain modes, 2 shear-strain modes, and 1 mixed mode. The axisymmetric variant sets the $\theta\theta$ enhancement to zero.

**HEX8 — EAS9 (3D):**

Nine enhanced modes corresponding to the $\xi$, $\eta$, $\zeta$, $\xi\eta$, $\eta\zeta$, $\xi\zeta$ terms in each of the three normal-strain components (Table 1 of Simo & Rifai 1990).

The reference coordinates $(\xi, \eta, \zeta)$ are mapped from the basix $[0,1]^d$ reference domain to the standard $[-1,1]^d$ domain to preserve the orthogonality of the mode polynomials.

### 12.4 Static Condensation

The enhanced DOFs $\boldsymbol{\alpha}$ are element-local and are eliminated by static condensation before the global assembly. At each Newton iteration, the element-level system is

$$
\begin{pmatrix} \mathbf{K}_{uu} & \mathbf{K}_{u\alpha} \\ \mathbf{K}_{\alpha u} & \mathbf{K}_{\alpha\alpha} \end{pmatrix}
\begin{pmatrix} \Delta\mathbf{u} \\ \Delta\boldsymbol{\alpha} \end{pmatrix}
= -\begin{pmatrix} \mathbf{f}_u \\ \mathbf{f}_\alpha \end{pmatrix}.
$$

Condensing out $\Delta\boldsymbol{\alpha}$:

$$
\Delta\boldsymbol{\alpha} = -\mathbf{K}_{\alpha\alpha}^{-1}\!\left(\mathbf{f}_\alpha + \mathbf{K}_{\alpha u}\,\Delta\mathbf{u}\right),
$$

$$
\tilde{\mathbf{K}}_{uu} = \mathbf{K}_{uu} - \mathbf{K}_{u\alpha}\,\mathbf{K}_{\alpha\alpha}^{-1}\,\mathbf{K}_{\alpha u}.
$$

The condensed stiffness $\tilde{\mathbf{K}}_{uu}$ and residual $\tilde{\mathbf{f}}_u = \mathbf{f}_u - \mathbf{K}_{u\alpha}\,\mathbf{K}_{\alpha\alpha}^{-1}\,\mathbf{f}_\alpha$ are assembled into the global system. The enhanced DOFs $\boldsymbol{\alpha}$ are updated locally via a `jax.lax.while_loop` Newton iteration (maximum 30 iterations, relative tolerance $10^{-10}$) and stored as element-level state between global Newton steps.


---

## 13. F-bar Patch Method

### 13.1 Motivation

The standard F-bar method (Section 4.3) averages the Jacobian over a single element. For simplex elements (TRI3, TET4) with only one quadrature point, the element-average equals the point value and no averaging occurs — the method degenerates. The F-bar patch method (de Souza Neto et al. 1996) extends the averaging domain to a *patch* of elements sharing a common edge (2D) or face (3D), restoring the anti-locking effect for simplex meshes.

### 13.2 Patch Construction

**2D (TRI3):** Interior edges are identified. Each interior edge is shared by exactly two elements; these two elements form a patch of size 2. Boundary elements that are not part of any interior-edge pair are adopted into the nearest available patch.

**3D (TET4):** Interior faces are identified. Each interior face is shared by exactly two elements; these form a size-2 patch. Boundary elements are adopted similarly.

The patch construction is performed once at problem setup using a greedy matching algorithm. The result is a mapping `patch_id[e]` that assigns each element to a patch, and `patch_elements[p]` that lists the elements in patch $p$.

### 13.3 Patch-Average Jacobian

For patch $p$ containing elements $\{e_1, e_2, \ldots\}$, the patch-average Jacobian is

$$
\bar{J}_p = \frac{\displaystyle\sum_{e \in p}\sum_{q} J_{e,q}\,\text{JxW}_{e,q}}{\displaystyle\sum_{e \in p}\sum_{q} \text{JxW}_{e,q}} = \frac{V_p^\text{def}}{V_p},
$$

where $V_p = \sum_{e \in p} V_e$ is the reference volume of the patch and $V_p^\text{def}$ is the deformed volume. The modified deformation gradient for element $e$ at quadrature point $q$ is

$$
\bar{\mathbf{F}}_{e,q} = \left(\frac{\bar{J}_p}{J_{e,q}}\right)^{1/n} \mathbf{F}_{e,q},
$$

with $n = 2$ for plane strain and $n = 3$ for 3D and axisymmetric problems.

### 13.4 Consistent Tangent

The patch coupling introduces off-diagonal stiffness blocks between elements in the same patch. The consistent tangent is assembled in two parts:

**Local (diagonal) block** — the autodiff tangent of the modified residual with respect to the element's own DOFs. The patch-average Jacobian is split as

$$
\bar{J}_p = c_1^{(e)}\, J_{e,q} + \underbrace{c_2^{(e)}}_{\text{stop\_gradient}},
$$

where $c_1^{(e)} = \text{JxW}_{e,q} / V_p$ is the weight of element $e$'s contribution to $\bar{J}_p$. JAX's autodiff sees $\partial \bar{J}_p / \partial J_{e,q} = c_1^{(e)}$, giving the correct local volumetric stiffness.

**Coupling (off-diagonal) block** — the rank-1 correction (eq. 15 of de Souza Neto et al. 1996):

$$
\mathbf{K}_{es} = \mathbf{g}_e \otimes \mathbf{h}_s, \quad e \neq s \in p,
$$

where

$$
\mathbf{g}_e = \frac{\partial \mathbf{R}_e}{\partial \bar{J}_p} \quad \text{(JAX jacfwd w.r.t. scalar } \bar{J}_p\text{)},
$$

$$
\mathbf{h}_s = c_1^{(s)}\, J_{s,q}\, \left(\nabla N_a \cdot \mathbf{F}_s^{-1}\right)_{a,i}.
$$

The coupling blocks are assembled as sparse COO entries and added to the global stiffness matrix after the standard volume assembly.


---

## 14. Constitutive Models (`materials/`)

All constitutive models implement one of two interfaces defined in `materials/umat.py`:

- **`UserPotential`** — path-independent materials expressible as a scalar strain-energy density $\Psi$. The user implements `upot(x) -> scalar`; stress and tangent are derived automatically via `jax.grad` and `jax.jvp`.
- **`UserMaterial`** — path-dependent materials (plasticity, viscoelasticity, crystal plasticity). The user implements `umat(F_or_eps, state: dict, dt) -> (stress, new_state)` and declares `state_fields` specifying the names and shapes of internal variables. The framework handles packing/unpacking of the flat state array.

### 14.1 Elastic Potentials (`UserPotential`)

#### 14.1.1 Isotropic Linear Elasticity (`UPOT_LinearElasticity`)

Strain-energy density:

$$
\Psi(\boldsymbol{\varepsilon}) = \frac{\lambda}{2}\left[\text{tr}(\boldsymbol{\varepsilon})\right]^2 + \mu\,\|\boldsymbol{\varepsilon}\|^2,
$$

with Lamé constants $\lambda = E\nu/[(1+\nu)(1-2\nu)]$ and $\mu = E/[2(1+\nu)]$. The Cauchy stress is

$$
\boldsymbol{\sigma} = \lambda\,\text{tr}(\boldsymbol{\varepsilon})\,\mathbf{I} + 2\mu\,\boldsymbol{\varepsilon}.
$$

**Parameters:** Young's modulus $E$, Poisson's ratio $\nu$.

#### 14.1.2 Compressible Neo-Hookean (`UPOT_NeoHookean`)

Simo–Hesch decoupled form:

$$
\Psi(\mathbf{F}) = \underbrace{\frac{\mu}{2}\!\left(J^{-2/3} I_1 - 3\right)}_{\text{isochoric}} + \underbrace{\frac{\kappa}{2}\!\left[\tfrac{1}{2}(J^2 - 1) - \ln J\right]}_{\text{volumetric}},
$$

where $I_1 = \text{tr}(\mathbf{F}^T\mathbf{F})$, $J = \det\mathbf{F}$, $\mu = E/[2(1+\nu)]$, $\kappa = E/[3(1-2\nu)]$.

**Parameters:** $E$, $\nu$.

#### 14.1.3 Mooney–Rivlin (`UPOT_MooneyRivlin`)

$$
\Psi = \frac{c_1}{2}(\bar{I}_1 - 3) + \frac{c_2}{2}(\bar{I}_2 - 3) + \kappa\, U(J),
$$

with isochoric invariants $\bar{I}_1 = J^{-2/3} I_1$, $\bar{I}_2 = J^{-4/3} I_2$, and volumetric penalty $U(J) = \beta^{-2}(J^\beta - \beta\ln J - 1)$ (default $\beta = 2$).

**Parameters:** $c_1$, $c_2$, $\kappa$, $\beta$.

#### 14.1.4 Gent–Mooney–Rivlin (`UPOT_GentMooneyRivlin`)

Extends Mooney–Rivlin with a Gent-type extensibility limit $J_m$ on the isochoric first invariant:

$$
\Psi = -\frac{c_1}{2} J_m \ln\!\left(1 - \frac{\bar{I}_1 - 3}{J_m}\right) + \frac{c_2}{2}(\bar{I}_2 - 3) + \kappa\, U(J).
$$

**Parameters:** $c_1$, $c_2$, $J_m$, $\kappa$, $\beta$.

#### 14.1.5 Ogden (`UPOT_Ogden`)

$$
\Psi = \sum_{i=1}^{N} \frac{\mu_i}{\alpha_i}\!\left(\bar{\lambda}_1^{\alpha_i} + \bar{\lambda}_2^{\alpha_i} + \bar{\lambda}_3^{\alpha_i} - 3\right) + \kappa\, U(J),
$$

where $\bar{\lambda}_k = J^{-1/3}\lambda_k$ are the isochoric principal stretches obtained from the eigenvalues of $\mathbf{F}^T\mathbf{F}$.

**Parameters:** arrays $\boldsymbol{\mu}$, $\boldsymbol{\alpha}$; bulk modulus $\kappa$; exponent $\beta$.

#### 14.1.6 St. Venant–Kirchhoff (`UPOT_StVenantKirchhoff`)

$$
\Psi(\mathbf{E}) = \frac{\lambda}{2}\left[\text{tr}(\mathbf{E})\right]^2 + \mu\,\|\mathbf{E}\|^2, \qquad \mathbf{E} = \tfrac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I}).
$$

**Parameters:** $E$, $\nu$.


### 14.2 Small-Strain Plasticity (`UserMaterial`)

#### 14.2.1 J2 Plasticity — Return Mapping (`UMAT_J2SmallStrain`)

Implements the radial-return algorithm of Simo & Hughes (1998) BOX 3.2 for isotropic/kinematic hardening von Mises plasticity.

**Yield function:**

$$
f(\boldsymbol{\xi}, \alpha) = \|\boldsymbol{\xi}\| - \sqrt{\tfrac{2}{3}}\, K(\alpha) \leq 0,
$$

where $\boldsymbol{\xi} = \mathbf{s} - \boldsymbol{\beta}$ is the shifted deviatoric stress, $\mathbf{s}$ is the deviatoric Cauchy stress, $\boldsymbol{\beta}$ is the back stress, and $K(\alpha)$ is the isotropic yield stress as a function of the accumulated equivalent plastic strain $\alpha$.

**Trial elastic state:**

$$
\mathbf{s}^\text{tr} = 2\mu\!\left(\mathbf{e}_{n+1} - \mathbf{e}^p_n\right), \qquad
\boldsymbol{\xi}^\text{tr} = \mathbf{s}^\text{tr} - \boldsymbol{\beta}_n, \qquad
f^\text{tr} = \|\boldsymbol{\xi}^\text{tr}\| - \sqrt{\tfrac{2}{3}}\, K(\alpha_n).
$$

If $f^\text{tr} \leq 0$: elastic step, no update.

**Plastic step** — fixed flow direction $\hat{\mathbf{n}} = \boldsymbol{\xi}^\text{tr}/\|\boldsymbol{\xi}^\text{tr}\|$, scalar consistency equation:

$$
g(\Delta\gamma) = \|\boldsymbol{\xi}^\text{tr}\| - 2\mu\Delta\gamma - \sqrt{\tfrac{2}{3}}\!\left[H(\alpha_{n+1}) - H(\alpha_n)\right] - \sqrt{\tfrac{2}{3}}\, K(\alpha_{n+1}) = 0,
$$

where $\alpha_{n+1} = \alpha_n + \sqrt{2/3}\,\Delta\gamma$ and $H(\alpha)$ is the kinematic hardening potential. This scalar equation is solved via `jax.lax.custom_root` with IFT tangent.

**State update:**

$$
\mathbf{e}^p_{n+1} = \mathbf{e}^p_n + \Delta\gamma\,\hat{\mathbf{n}}, \quad
\boldsymbol{\beta}_{n+1} = \boldsymbol{\beta}_n + \sqrt{\tfrac{2}{3}}\!\left[H(\alpha_{n+1}) - H(\alpha_n)\right]\hat{\mathbf{n}}, \quad
\boldsymbol{\sigma}_{n+1} = \kappa\,\text{tr}(\boldsymbol{\varepsilon}_{n+1})\,\mathbf{I} + \mathbf{s}^\text{tr} - 2\mu\Delta\gamma\,\hat{\mathbf{n}}.
$$

**State fields:** `eps_p` (deviatoric plastic strain), `alpha` (accumulated equivalent plastic strain), `beta` (back stress), `sigma_zz` (out-of-plane stress, plane-strain only).

**Hardening laws** (callable objects $\alpha \mapsto K(\alpha)$):

| Class | Formula |
|---|---|
| `LinearIsotropicHardening` | $K = \sigma_0 + H\alpha$ |
| `VoceHardening` | $K = \sigma_0 + (\sigma_u - \sigma_0)(1 - e^{-b\alpha})$ |
| `NonlinearIsotropicHardening` | $K = \sigma_Y + K_\infty\alpha + (K_\infty - K_0)(1 - e^{-\delta\alpha})$ |
| `MixedHardening` | $K = \beta_\text{mix}\,h(\alpha)$, $H' = (1-\beta_\text{mix})\,h'(\alpha)$, $h = K_\infty - (K_\infty - K_0)e^{-\delta\alpha} + \bar{H}'\alpha$ |

#### 14.2.2 Viscoplastic Armstrong–Frederick (`UMAT_ArmstrongFrederick`)

Combines Norton power-law viscous flow with Voce isotropic hardening and two Armstrong–Frederick nonlinear kinematic backstress tensors.

**Viscoplastic flow rule:**

$$
\dot{p} = \left\langle \frac{f - \sigma_Y}{K_\text{visc}} \right\rangle_+^m,
$$

where $f = \|\boldsymbol{\xi}\|$ is the effective stress, $\sigma_Y(p)$ is the isotropic yield stress, and $K_\text{visc}$, $m$ are the Norton viscosity and exponent.

**Armstrong–Frederick kinematic hardening** (two backstress tensors $\mathbf{X}_1$, $\mathbf{X}_2$):

$$
\dot{\mathbf{X}}_k = \tfrac{2}{3} C_k\,\dot{\boldsymbol{\varepsilon}}^p - \gamma_k\,\mathbf{X}_k\,\dot{p}, \quad k = 1, 2.
$$

The implicit backward-Euler update is solved by a local Newton iteration (20 steps via `jax.lax.scan`).

**State fields:** `eps_p`, `alpha`, `X1`, `X2`.


### 14.3 Finite-Strain Plasticity

#### 14.3.1 Multiplicative $\mathbf{F}^e\mathbf{F}^p$ J2 Plasticity (`UMAT_FeFpJ2Plasticity`)

Implements the multiplicative decomposition $\mathbf{F} = \mathbf{F}^e\mathbf{F}^p$ with the return-mapping algorithm of Simo & Hughes (1998) BOX 9.1.

**Relative deformation gradient:**

$$
\mathbf{f} = \mathbf{F}_{n+1}\mathbf{F}_n^{-1}, \qquad \bar{\mathbf{f}} = (\det\mathbf{f})^{-1/3}\mathbf{f}.
$$

**Trial elastic left Cauchy–Green tensor:**

$$
\bar{\mathbf{b}}^e_\text{tr} = \bar{\mathbf{f}}\,\bar{\mathbf{b}}^e_n\,\bar{\mathbf{f}}^T.
$$

**Trial deviatoric Kirchhoff stress:**

$$
\mathbf{s}^\text{tr} = \mu\,\text{dev}[\bar{\mathbf{b}}^e_\text{tr}], \qquad f^\text{tr} = \|\mathbf{s}^\text{tr}\| - \sqrt{\tfrac{2}{3}}\,k(\alpha_n).
$$

**Return mapping** (if $f^\text{tr} > 0$):

$$
\bar{I}^e = \tfrac{1}{3}\text{tr}[\bar{\mathbf{b}}^e_\text{tr}], \quad \bar{\mu} = \mu\bar{I}^e,
$$

$$
\hat{f}(\Delta\gamma) = \|\mathbf{s}^\text{tr}\| - \sqrt{\tfrac{2}{3}}\,k\!\left(\alpha_n + \sqrt{\tfrac{2}{3}}\Delta\gamma\right) - 2\bar{\mu}\Delta\gamma = 0.
$$

**Kirchhoff and PK1 stresses:**

$$
\boldsymbol{\tau} = \frac{\kappa}{2}(J^2 - 1)\,\mathbf{I} + \mathbf{s}_{n+1}, \qquad
\mathbf{P} = \boldsymbol{\tau}\,\mathbf{F}_{n+1}^{-T}.
$$

**State fields:** `be_bar` (isochoric elastic left Cauchy–Green tensor), `p` (accumulated equivalent plastic strain), `F_old` (deformation gradient at start of step).

### 14.4 Viscoelasticity

#### 14.4.1 Standard Linear Solid (`UMAT_StandardLinearSolid`)

Zener model: an equilibrium elastic branch (stiffness $E_\infty$, $\nu_\infty$) in parallel with a Maxwell branch (spring $E_1$, $\nu_1$; dashpot viscosity $\eta$). The relaxation time is $\tau = \eta/E_1$.

**Exact exponential integrator** (Simo & Hughes 1998):

$$
\boldsymbol{\varepsilon}^v_{n+1} = \boldsymbol{\varepsilon}_{n+1} + e^{-\Delta t/\tau}\!\left(\boldsymbol{\varepsilon}^v_n - \boldsymbol{\varepsilon}_n\right) - e^{-\Delta t/(2\tau)}\Delta\boldsymbol{\varepsilon},
$$

$$
\boldsymbol{\sigma} = \boldsymbol{\sigma}_\infty(\boldsymbol{\varepsilon}_{n+1}) + \boldsymbol{\sigma}_1\!\left(\boldsymbol{\varepsilon}_{n+1} - \boldsymbol{\varepsilon}^v_{n+1}\right).
$$

**State fields:** `epsv` (viscous strain), `eps_old` (total strain at start of step).

#### 14.4.2 Generalized Maxwell (`UMAT_GeneralizedMaxwell`)

Prony series with $N$ Maxwell branches, each with modulus $E_k$ and relaxation time $\tau_k$. The same exact exponential integrator is applied independently to each branch.

**State fields:** `epsv_k` for $k = 1, \ldots, N$ (viscous strain of each branch), `eps_old`.


### 14.5 Crystal Plasticity (`UserMaterial`, `kinematics = "deformation_gradient"`)

All crystal plasticity models use the multiplicative decomposition $\mathbf{F} = \mathbf{F}^e\mathbf{F}^p$ and an implicit S-based (2nd Piola–Kirchhoff) Newton solver with half-step backtracking line search, implemented via `jax.lax.while_loop` and wrapped with `jax.custom_jvp` for the implicit-function-theorem consistent tangent.

#### 14.5.1 FCC Crystal Plasticity (`UMAT_FCCCrystalPlasticity`)

**Reference:** Kalidindi, Bronkhorst & Anand (1992).

**Slip systems:** 12 × $\{111\}\langle110\rangle$.

**Elastic law:** Cubic anisotropic stiffness $\mathbf{C}$ (constants $C_{11}$, $C_{12}$, $C_{44}$), rotated to the crystal frame:

$$
\mathbf{S} = \mathbf{C}_\text{rot} : \mathbf{E}^e, \qquad \mathbf{E}^e = \tfrac{1}{2}(\mathbf{F}^{eT}\mathbf{F}^e - \mathbf{I}).
$$

**Resolved shear stress:**

$$
\tau^\alpha = \mathbf{S} : \mathbf{m}^\alpha, \qquad \mathbf{m}^\alpha = \text{sym}(\mathbf{s}^\alpha \otimes \mathbf{n}^\alpha),
$$

where $\mathbf{s}^\alpha$ and $\mathbf{n}^\alpha$ are the slip direction and slip-plane normal of system $\alpha$.

**Slip rate (power law):**

$$
\dot{\gamma}^\alpha = a_0\,\left|\frac{\tau^\alpha}{g^\alpha}\right|^{1/m}\!\text{sign}(\tau^\alpha).
$$

**Kalidindi hardening:**

$$
\Delta g^\alpha = h\,\sum_\beta Q^{\alpha\beta}\,|\Delta\gamma^\beta|\,\left|1 - \frac{g^\beta}{g_\text{sat}}\right|^{g_a}\!\text{sign}\!\left(1 - \frac{g^\beta}{g_\text{sat}}\right),
$$

where $Q^{\alpha\beta}$ is the latent-hardening matrix (self-hardening $q = 1$, latent-hardening $q = r$ for co-planar systems, $q = 1$ otherwise).

**Plastic velocity gradient:**

$$
\mathbf{L}^p = \sum_\alpha \dot{\gamma}^\alpha\,\mathbf{s}^\alpha \otimes \mathbf{n}^\alpha, \qquad
\mathbf{F}^p_{n+1} = \left(\mathbf{I} + \Delta t\,\mathbf{L}^p\right)\mathbf{F}^p_n.
$$

**State fields:** `Fp_inv` ($\mathbf{F}^{p-1}$, tensor), `slip_resistance` ($g^\alpha$, array of 12), `slip` (accumulated $\gamma^\alpha$, array of 12), `rot_mat` (crystal orientation, tensor). Total: 42 scalars per quadrature point.

**Parameters:** $C_{11}$, $C_{12}$, $C_{44}$; $g_0$ (initial slip resistance); $h$ (hardening modulus); $g_\text{sat}$ (saturation resistance); $g_a$ (hardening exponent); $a_0$ (reference slip rate); $m$ (rate sensitivity).

#### 14.5.2 BCC Crystal Plasticity (`UMAT_BCCCrystalPlasticity`)

Identical algorithm to FCC. Differences:

- **Slip systems:** 12 × $\{110\}\langle111\rangle$.
- **Default parameters:** Tantalum ($C_{11} = 267\,\text{GPa}$, $C_{12} = 161\,\text{GPa}$, $C_{44} = 82.5\,\text{GPa}$; $g_0 = 67.5\,\text{MPa}$).

**State fields:** same layout as FCC (42 scalars).

#### 14.5.3 HCP Crystal Plasticity (`UMAT_HCPCrystalPlasticity`)

**Slip systems:** 24 total — basal $\langle a\rangle$ (3), prismatic $\langle a\rangle$ (3), pyramidal-1st $\langle a\rangle$ (6), pyramidal-1st $\langle c+a\rangle$ (12).

**Elastic law:** Transversely isotropic stiffness (constants $C_{11}$, $C_{12}$, $C_{13}$, $C_{33}$, $C_{44}$; $c$-axis along $z$). Default: Zirconium.

**Thermally activated slip rate:**

$$
\dot{\gamma}^\alpha = \rho_m v_0 b^2 \exp\!\left(-\frac{\Delta F}{k_B T}\right) \sinh\!\left(\frac{\langle|\tau^\alpha| - \tau_c^\alpha\rangle_+ \Delta V}{k_B T}\right) \text{sign}(\tau^\alpha),
$$

where $\rho_m$ is the mobile dislocation density, $v_0$ the attempt frequency, $b$ the Burgers vector magnitude, $\Delta F$ the activation energy, $\Delta V$ the activation volume, and $T$ the temperature.

**SSD evolution (per system $\alpha$):**

$$
\dot{\rho}^\alpha_\text{SSD} = \frac{1}{b}\!\left(\frac{\sqrt{\rho^\alpha_\text{SSD}}}{K^\alpha} - 2 y_c^\alpha \rho^\alpha_\text{SSD}\right)|\dot{\gamma}^\alpha|,
$$

where $K^\alpha$ is the storage coefficient and $y_c^\alpha = y_\text{fac}\,b$ is the annihilation distance.

**Taylor hardening:**

$$
\tau_c^\alpha = \tau_{c0}^\alpha + \lambda_T G b \sqrt{\sum_\beta A^{\alpha\beta} \rho^\beta_\text{SSD}},
$$

where $A^{\alpha\beta}$ is the dislocation interaction matrix (diagonal $A_\text{self}$, off-diagonal $A_\text{latent}$).

**State fields:** `Fp_inv` (tensor), `slip_resistance` (array of 24), `slip` (array of 24), `rot_mat` (tensor), `rho_SSD` (array of 24). Total: 90 scalars per quadrature point.

#### 14.5.4 HCP + GND (`UMAT_HCPCrystalPlasticity_GND`)

Extends `UMAT_HCPCrystalPlasticity` with geometrically necessary dislocation (GND) density following Arsenlis & Parks (1999).

**Nye tensor:**

$$
\boldsymbol{\Lambda} = \text{curl}(\mathbf{F}^p) = -\nabla \times \mathbf{F}^{p-1},
$$

computed from the spatial gradient of $\mathbf{F}^{p-1}$ via an L2 node-projection / interpolation procedure at the mesh level.

**Screw and edge decomposition** (per slip system $\alpha$):

$$
\boldsymbol{\Lambda} = \sum_\alpha \left(\rho^\alpha_\text{screw}\,\mathbf{b}^\alpha \otimes \mathbf{s}^\alpha + \rho^\alpha_\text{edge}\,\mathbf{b}^\alpha \otimes \mathbf{t}^\alpha\right), \quad \mathbf{t}^\alpha = \mathbf{s}^\alpha \times \mathbf{n}^\alpha,
$$

solved as a least-squares system $\mathbf{A}\,\boldsymbol{\rho}_\text{GND} = \text{vec}(\boldsymbol{\Lambda})$ with $\mathbf{A} \in \mathbb{R}^{9 \times 2N_\text{slip}}$.

**Modified Taylor hardening:**

$$
\tau_c^\alpha = \tau_{c0}^\alpha + \lambda_T G b \sqrt{\sum_\beta A^{\alpha\beta}\!\left(\rho^\beta_\text{SSD} + \rho^\beta_\text{GND}\right)}.
$$

The GND update is a mesh-level operation performed via a `post_step` hook registered on the `Problem` object after each converged Newton step. The hook reads $\mathbf{F}^{p-1}$ from the state array, computes the Nye tensor via `jax.ops.segment_sum`-based node projection, and writes the updated $\rho_\text{GND}$ back into the state.

**State fields:** same as HCP plus `rho_GND` (array of 24). Total: 114 scalars per quadrature point.

### 14.6 Phase-Field Fracture Coupling

Phase-field fracture models couple the mechanics problem to a scalar damage field $d \in [0,1]$ through a degradation function $g(d)$. **Full theory** (damage PDEs, staggered coupling, strategy matrix): [Phase-field fracture](phase-field-fracture.md).

**Degradation function:**

$$
g(d) = (1 - d)^2 + \eta, \quad \eta = 10^{-10} \text{ (residual stiffness)}.
$$

**Strain energy split** — the elastic strain energy is decomposed into a tensile (crack-driving) part $\mathcal{H}^+$ and a compressive (crack-resisting) part $\mathcal{H}^-$:

$$
\Psi = g(d)\,\mathcal{H}^+ + \mathcal{H}^-.
$$

Two splits are implemented:

**Spectral split (Miehe et al. 2010):**

$$
\mathcal{H}^\pm = \frac{\lambda}{2}\langle\text{tr}\,\boldsymbol{\varepsilon}\rangle_\pm^2 + \mu\,\|\boldsymbol{\varepsilon}^\pm\|^2,
$$

where $\boldsymbol{\varepsilon}^\pm = \sum_\alpha \langle\varepsilon_\alpha\rangle_\pm\,\mathbf{n}_\alpha \otimes \mathbf{n}_\alpha$ are the positive/negative spectral projections of the strain tensor. The spectral projection uses a custom JVP (Miehe 2001) for correct gradients at repeated eigenvalues.

**Volumetric–deviatoric split (Amor et al. 2009):**

$$
\mathcal{H}^+ = \frac{\kappa}{2}\langle\text{tr}\,\boldsymbol{\varepsilon}\rangle_+^2 + \mu\,\|\text{dev}\,\boldsymbol{\varepsilon}\|^2, \qquad
\mathcal{H}^- = \frac{\kappa}{2}\langle\text{tr}\,\boldsymbol{\varepsilon}\rangle_-^2.
$$

**Phase-field energy models** (`PhasefieldEnergyConfig`):

| Model | $c_w$ | Crack surface density |
|---|---|---|
| AT2 | 2 | $\gamma(d,\nabla d) = \frac{1}{c_w}\!\left(\frac{d^2}{l} + l|\nabla d|^2\right)$ |
| AT1 | 8/3 | $\gamma = \frac{1}{c_w}\!\left(\frac{d}{l} + l|\nabla d|^2\right)$ |
| Cohesive | $\pi$ | $\gamma = \frac{1}{c_w}\!\left(\frac{\alpha(d)}{l} + l|\nabla d|^2\right)$ |


---

## 15. Nonlinear Solvers (`solvers/`)

### 15.1 Newton–Raphson Method

The quasi-static equilibrium problem $\mathbf{R}(\mathbf{u}) = \mathbf{0}$ is solved by the Newton–Raphson iteration:

$$
\mathbf{K}(\mathbf{u}^k)\,\Delta\mathbf{u}^k = -\mathbf{R}(\mathbf{u}^k), \qquad \mathbf{u}^{k+1} = \mathbf{u}^k + s^k\,\Delta\mathbf{u}^k,
$$

where $\mathbf{K} = \partial\mathbf{R}/\partial\mathbf{u}$ is the consistent tangent stiffness and $s^k \in (0,1]$ is the step length determined by the line search.

**Convergence criteria** (both must be satisfied):

$$
\|\mathbf{R}^k\| \leq \varepsilon_\text{abs} \quad \text{and} \quad \frac{\|\mathbf{R}^k\|}{\|\mathbf{R}^0\|} \leq \varepsilon_\text{rel},
$$

with defaults $\varepsilon_\text{abs} = 10^{-5}$ and $\varepsilon_\text{rel} = 10^{-7}$. The maximum number of iterations is 100.

### 15.2 Line Search (Matthies & Strang 1979)

The line search finds a step length $s$ such that the directional derivative of the residual along the Newton direction $\mathbf{d} = \Delta\mathbf{u}$ is approximately zero:

$$
G(s) = \mathbf{d}^T \mathbf{R}(\mathbf{u} + s\,\mathbf{d}) \approx 0.
$$

Note that $G(0) = \mathbf{d}^T\mathbf{R}(\mathbf{u}) < 0$ (descent direction) and the full Newton step $s = 1$ is accepted if $|G(1)| \leq s_\text{tol}\,|G(0)|$ (default $s_\text{tol} = 0.5$).

**Algorithm:**

1. **Bracketing:** Starting from $s = 1$, double $s$ until $G(s)$ changes sign (maximum 16 doublings). This handles cases where the full Newton step overshoots.
2. **Illinois algorithm** (modified regula falsi): Given a bracket $[s_a, s_b]$ with $G(s_a) < 0 < G(s_b)$, iterate:
$$
s_{k+1} = s_a - G(s_a)\,\frac{s_b - s_a}{G(s_b) - G(s_a)},
$$
with the Illinois modification to prevent stagnation (maximum 10 iterations).

### 15.3 Artificial Stabilization

For problems with near-singular tangents (e.g., large plastic flow, snap-through), an optional diagonal stabilization is added:

$$
\tilde{\mathbf{K}} = \mathbf{K} + \alpha_\text{stab}\,\text{diag}(\mathbf{K}),
$$

with default $\alpha_\text{stab} = 2 \times 10^{-4}$. A warning is issued if the stabilization energy exceeds 5% of the elastic strain energy.

### 15.4 Arc-Length Method

The arc-length method (Crisfield 1981) extends Newton–Raphson to trace equilibrium paths through limit points and snap-through/snap-back behaviour by treating the load parameter $\lambda$ as an additional unknown.

**Augmented system:**

$$
\mathbf{R}(\mathbf{u}, \lambda) = \mathbf{F}_\text{int}(\mathbf{u}) - \lambda\,\mathbf{F}_\text{ext} = \mathbf{0},
$$

$$
c(\mathbf{u}, \lambda) = \|\Delta\mathbf{u}\|^2 + \psi^2\,\Delta\lambda^2\,\|\mathbf{q}\|^2 - \Delta l^2 = 0,
$$

where $\Delta l$ is the prescribed arc length, $\psi$ is a scaling parameter, and $\mathbf{q}$ is a reference load vector.

**Linearisation** at iteration $k$:

$$
\begin{pmatrix} \mathbf{K} & -\mathbf{F}_\text{ext} \\ 2\Delta\mathbf{u}^T & 2\psi^2\Delta\lambda\,\|\mathbf{q}\|^2 \end{pmatrix}
\begin{pmatrix} \delta\mathbf{u} \\ \delta\lambda \end{pmatrix}
= -\begin{pmatrix} \mathbf{R} \\ c \end{pmatrix}.
$$

The increment $\delta\lambda$ is found by solving the quadratic equation arising from the arc-length constraint; the sign is chosen to maintain a positive dot product with the previous increment direction (Crisfield's criterion).

Two variants are implemented:

- **Displacement-driven** (`arc_length_solver_disp_driven`): $\mathbf{q} = \mathbf{u}_b$ (reference displacement).
- **Force-driven** (`arc_length_solver_force_driven`): $\mathbf{q} = \mathbf{F}_\text{ext}$ (reference force).


---

## 16. Linear Solvers

### 16.1 Overview

The global tangent stiffness matrix $\mathbf{K}$ is stored in CSR (Compressed Sparse Row) format. Several linear solver backends are supported, selectable via the `solver` argument of the Newton solver:

| Backend | Format | Device | Notes |
|---|---|---|---|
| `jax_gpu_solver` | BCOO | GPU | BiCGSTAB, diagonal preconditioner |
| `jax_solver` | BCOO | CPU | BiCGSTAB, diagonal preconditioner |
| `amgx_solver` | CSR | GPU | NVIDIA AmgX AMG (**preferred** iterative GPU) |
| `cudss_solver` | CSR | GPU | NVIDIA cuDSS direct solver |
| `amgcl_solver` | CSR | CPU/GPU | Algebraic multigrid, Chebyshev smoother |
| `umfpack_solver` | CSR | CPU | Direct sparse LU (`scipy.sparse.linalg.spsolve`) |
| `petsc_solver` | CSR | CPU | PETSc KSP (BCGSL + ILU) |

### 16.2 JAX-Native Iterative Solvers

#### 16.2.1 BiCGSTAB

The Biconjugate Gradient Stabilized method (van der Vorst 1992) solves non-symmetric systems $\mathbf{K}\mathbf{x} = \mathbf{b}$ without forming the transpose. The iteration is implemented via `jax.lax.while_loop` and is therefore fully JAX-traceable (compatible with JIT and `vmap`).

**Convergence criterion:**

$$
\|\mathbf{r}^k\| \leq \varepsilon_\text{rel}\,\|\mathbf{b}\| + \varepsilon_\text{abs},
$$

with defaults $\varepsilon_\text{rel} = \varepsilon_\text{abs} = 10^{-10}$ and maximum 10,000 iterations.

An optional diagonal preconditioner $\mathbf{M}^{-1} = \text{diag}(1/K_{ii})$ is applied to improve conditioning.

#### 16.2.2 Conjugate Gradient (CG)

For symmetric positive-definite systems (linear elasticity, phase-field Laplacian), the standard preconditioned CG method is available with the same convergence and preconditioner options as BiCGSTAB.

### 16.3 NVIDIA AmgX

NVIDIA AmgX provides GPU algebraic multigrid. DiffSolid binds AmgX through
ctypes to `libamgxsh.so` (set `AMGX_LIBRARY` or `AMGX_HOME`). On current
implicit benchmarks (H200), AmgX is the preferred iterative backend for large
elasticity / plasticity systems, typically ahead of AMGCL CUDA and competitive
with or better than cuDSS at large DOF.

### 16.4 AMGCL Algebraic Multigrid

The AMGCL library (Demidov 2019) provides GPU-accelerated algebraic multigrid (AMG) preconditioners (fallback when AmgX is unavailable). Recommended configurations based on empirical benchmarking on this library:

| Problem type | Smoother | Krylov solver |
|---|---|---|
| J2/crystal plasticity | Chebyshev (degree 2) | FGMRES |
| Linear elasticity | Chebyshev (degree 2) | CG |

AMG coarsening uses the classical Ruge–Stüben algorithm. ILU(0), SPAI-0, and block-size variants were found to be slower or less robust on typical finite-element matrices arising from this library and are not recommended.

### 16.5 Solver Selection Guidelines

- **Small problems** ($< 10^5$ DOFs): `CUDSS` (direct, robust).
- **Large problems** on GPU ($10^5$–$10^7$ DOFs): `AMGx()` preferred; `AMGCL(gpu=True, ...)` as fallback when AmgX is unavailable.
- **JAX-only environments** (no C++ extensions / no AmgX): JAX BiCGSTAB/CG backends.

---

## References

1. de Souza Neto, E. A., Perić, D., Dutko, M., & Owen, D. R. J. (1996). Design of simple low order finite elements for large strain analysis of nearly incompressible solids. *International Journal of Solids and Structures*, 33(20–22), 3277–3296.
2. Hughes, T. J. R. (1980). Generalization of selective integration procedures to anisotropic and nonlinear media. *International Journal for Numerical Methods in Engineering*, 15(9), 1413–1418.
3. Simo, J. C., & Hughes, T. J. R. (1998). *Computational Inelasticity*. Springer, New York.
4. Bonet, J., & Wood, R. D. (2008). *Nonlinear Continuum Mechanics for Finite Element Analysis* (2nd ed.). Cambridge University Press.
5. Wriggers, P. (2008). *Nonlinear Finite Element Methods*. Springer, Berlin.
6. Simo, J. C., & Rifai, M. S. (1990). A class of mixed assumed strain methods and the method of incompatible modes. *International Journal for Numerical Methods in Engineering*, 29(8), 1595–1638.
7. Miehe, C., Hofacker, M., & Welschinger, F. (2010). A phase field model for rate-independent crack propagation. *Computer Methods in Applied Mechanics and Engineering*, 199(45–48), 2765–2778.
8. Amor, H., Marigo, J.-J., & Maurini, C. (2009). Regularized formulation of the variational brittle fracture with unilateral contact. *Journal of the Mechanics and Physics of Solids*, 57(8), 1209–1228.
9. Kalidindi, S. R., Bronkhorst, C. A., & Anand, L. (1992). Crystallographic texture evolution in bulk deformation processing of FCC metals. *Journal of the Mechanics and Physics of Solids*, 40(3), 537–569.
10. Arsenlis, A., & Parks, D. M. (1999). Crystallographic aspects of geometrically-necessary and statistically-stored dislocation density. *Acta Materialia*, 47(5), 1597–1611.
11. Matthies, H. G., & Strang, G. (1979). The solution of nonlinear finite element equations. *International Journal for Numerical Methods in Engineering*, 14(11), 1613–1626.
12. Crisfield, M. A. (1981). A fast incremental/iterative solution procedure that handles snap-through. *Computers & Structures*, 13(1–3), 55–62.
13. Flanagan, D. P., & Belytschko, T. (1981). A uniform strain hexahedron and quadrilateral with orthogonal hourglass control. *International Journal for Numerical Methods in Engineering*, 17(5), 679–706.
14. van der Vorst, H. A. (1992). Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems. *SIAM Journal on Scientific and Statistical Computing*, 13(2), 631–644.
15. Demidov, D. (2019). AMGCL: A C++ library for solution of large sparse linear systems. *Software X*, 10, 100285.
16. Miehe, C. (2001). Algorithms for computation of stresses and elasticity moduli in terms of Seth–Hill's family of generalized strain tensors. *Communications in Numerical Methods in Engineering*, 17(5), 337–353.

---

## Appendix: Explicit dynamics integrator routing

See [Dynamic mechanics — §6](dynamic-mechanics.md#6-explicit-material-kernels-and-jax-scan-backends) for context.

| Integrator | Backend | Typical use |
|------------|---------|-------------|
| `auto` | registry pick | Default; stateful J2/CP → `scan_stateful` |
| `scan_energy` | L0 strain-energy gradient | Linear elastic, hourglass |
| `scan_residual` | L1 `compute_residual` scan | Stateless UPOT; MPC+PBC elastic waves |
| `scan_stateful` | L3 `(u,v,state)` scan | J2/CP/GND, B-bar/F-bar/EAS, time-varying PBC |
| `python` | Python Verlet loop | Debug, unsupported paths |

MPC/PBC: time-varying `_mpc_offset` (e.g. `F_bar(t)`) requires `tick_fn` from `Simulation.solve`; both `scan_stateful` and `scan_residual` precompute per-step `(f_ext, g)` sequences.
