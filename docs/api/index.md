# DiffSolid API Reference

**Version:** 0.1.0  
**Scope:** Public Python API accessed via `import diffsolid as ds`

---

## Document Scope

This reference documents **all user-facing functionality** shipped with DiffSolid.
The API is organised in two layers:

| Layer | Import path | Intended use |
|-------|-------------|--------------|
| **Facade (primary)** | `ds.Simulation`, `ds.physics`, `ds.materials`, `ds.solvers`, … | Standard simulations — **start here** |
| **Extended user API** | `ds.post`, `ds.inverse`, `ds.io`, top-level `ds.Mesh`, … | Post-processing, inverse problems, mesh I/O |

Sections 1–16 cover the facade and extended user API. Theory is in [formulations.md](../theory/formulations.md).

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Simulation Manager](#2-simulation-manager)
3. [Step Configuration](#3-step-configuration)
4. [Phase-Field Fracture Strategies (S1–S7)](#4-phase-field-fracture-strategies-s1s7)
5. [Physics Descriptors](#5-physics-descriptors)
6. [Material Descriptors](#6-material-descriptors)
7. [Couplers](#7-couplers)
8. [Solver Descriptors](#8-solver-descriptors)
9. [Amplitudes](#9-amplitudes)
10. [Constraints and Advanced Boundary Conditions](#10-constraints-and-advanced-boundary-conditions)
11. [Output, Results, and Post-Processing](#11-output-results-and-post-processing)
12. [Mesh I/O and Checkpoints](#12-mesh-io-and-checkpoints)
13. [Pure Mechanics Dynamics](#13-pure-mechanics-dynamics)
14. [Phase-Field Dislocation Dynamics (PFDD)](#14-phase-field-dislocation-dynamics-pfdd)
15. [Inverse Problems and Optimisation](#15-inverse-problems-and-optimisation)
16. [Type Aliases and Enumerations](#16-type-aliases-and-enumerations)
17. [Environment Variables](#17-environment-variables)
18. [Appendix — Legacy API Migration](#18-appendix-legacy-api-migration)

---

## 1. Introduction

DiffSolid is a JAX-native finite element framework for nonlinear solid mechanics,
phase-field fracture, GPU linear algebra, explicit/implicit dynamics, crystal
plasticity, and gradient-based inverse problems.

### 1.1 Design Principles

- **Step-scoped configuration.** Boundary conditions, damage evolution, outputs,
  and per-step solver overrides attach to loading steps.
- **Validated fracture strategies.** Coupled mechanics–damage workflows map to
  literature strategy IDs **S1–S7** and are validated at solve time.
- **Material-centric phase-field physics.** Strain energy splits and degradation
  functions live on constitutive models, not on the coupler.
- **Double precision.** `jax_enable_x64` is enabled on import.

### 1.2 Canonical Import

```python
import diffsolid as ds
```

### 1.3 Minimal Workflow

```python
sim = ds.Simulation(name="case", dim=3, ele_type="HEX8")
sim.load_mesh("mesh.msh")
sim.add_physics(ds.physics.SolidMechanics(material=mat))
sim.set_linear_solver(ds.solvers.AMGx())

step = sim.add_step(name="load", duration=1.0, dt=0.01)
step.add_dirichlet_bc(on="x == 0", components=["x", "y", "z"], value=0.0)
result = sim.solve(output_dir="results/")
```

For phase-field fracture, add `PhaseField` physics, a coupler, and step-level
damage/dynamics configuration (Sections 4–5, 7).

---

## 2. Simulation Manager

Class: `ds.Simulation`

### 2.1 Construction and Mesh

#### `Simulation.__init__(name="simulation", dim=3, ele_type="HEX8")`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"simulation"` | Identifier for logs and output |
| `dim` | `int` | `3` | Spatial dimension; updated after mesh load |
| `ele_type` | `str` | `"HEX8"` | Default element type |

#### `sim.load_mesh(path_or_mesh, ele_type=None)`

Loads meshio-compatible files (`.msh`, `.inp`, `.vtu`, …) or a `ds.Mesh` object.
Returns the internal mesh; updates `sim.dim` when inferable.

#### `sim.add_physics(physics_descriptor)`

Registers `SolidMechanics`, `PhaseField`, or (for PFDD) descriptors consumed by
`DislocationSimulation`. Returns the descriptor for chaining.

**Limitation:** staggered **phase-field fracture** currently supports **one**
loading step when both solid mechanics and phase field are present.

#### `sim.add_physical_group(name, indices)`

Registers a named node group for `@name` location selectors in boundary conditions.

### 2.2 Global Solver and Algorithm Configuration

| Method | Description |
|--------|-------------|
| `set_coupler(coupler)` | Multi-physics coupling (`Staggered`, `FractureCoupling`, …) |
| `set_linear_solver(solver)` | Default linear backend for Newton solves |
| `set_dynamics_solver(solver)` | `ExplicitDynamicsSolver` or `ImplicitDynamicsSolver` descriptor |
| `set_pf_solver(solver)` | Phase-field nonlinear solver override |
| `set_pf_linear_solver(solver)` | Linear backend for VI/Newton phase-field solves |
| `set_phase_field_solver(...)` | Combined PF solver + linear backend shorthand |
| `set_newton_options(rtol=..., atol=..., max_iter=..., line_search=...)` | Global Newton tolerances |
| `set_arc_length_solver(psi=..., Delta_l=..., tol=..., mode="displacement")` | Crisfield arc-length for limit-point problems |
| `set_adaptive_stepping(enabled=True, dt_min=..., dt_max=..., ...)` | Quasi-static adaptive load increments |
| `set_stagger_adaptive_stepping(dt_min=..., dt_max=..., ...)` | Adaptive increment for staggered PF |
| `set_log(level="summary")` | `"summary"`, `"solver"`, or `"debug"` |
| `set_performance_report(True, output_dir=...)` | Detailed timing report |

### 2.3 Steps and Analysis Drivers

#### `sim.add_step(name=None, *, duration, dt=None, **solver_overrides)`

Creates a loading interval with step-local time `t_local ∈ [0, duration]`.

| Parameter | Description |
|-----------|-------------|
| `duration` | Physical step length (required) |
| `dt` | Increment size; `None` → CFL-based auto estimation for dynamics |
| `**solver_overrides` | Per-step Newton, adaptive, or dynamics options (Section 3.6) |

Returns a step builder (`_Step`). Supports context manager form:

```python
with sim.add_step(name="hold", duration=2.0, dt=0.1) as step:
    step.add_dirichlet_bc(...)
```

#### `sim.add_reference_point(name, *, coords=None, ndofs=1, labels=None)`

Virtual control node for coupling constraints and concentrated loads.

#### `sim.add_periodic_bc(formulation="auto", axes=None, eps_bar=None, F_bar=None, ...)`

Returns a `_PeriodicBC` object attached via `step.add_periodic_bc(pbc)`.

### 2.4 History Output (Simulation-Level)

```python
sim.add_history_output("F_top", on="@top", value="reaction.z", every=1)
```

Global history keys must be unique. Step-level `add_history_output` scopes recording
to that step's increment window.

### 2.5 `sim.solve(...)`

Runs all registered steps sequentially.

```python
result = sim.solve(
    output_dir="results/",
    save_every=1,
    callback=None,
    output_format="xdmf",       # "xdmf" | "vtu" | "none"
    field_output=None,            # None → ["u"]
    field_save_every=1,
    history_csv="history.csv",    # None to disable CSV
)
```

**Routing (mechanics + phase field):**

1. Merge step damage evolution and fracture dynamics configuration.
2. Validate against S1–S7 (`validate_fracture_combo`).
3. `mechanics_integrator != "quasi_static"` → `_solve_fracture_dynamics`.
4. Else → `_solve_staggered`.

Returns a `Result` object (dict subclass) — Section 11.

### 2.6 Checkpoints

```python
sim.set_checkpoint_options(every=10, path="chk/")
sim.save_checkpoint(path="state.eqx")
sim.load_checkpoint(path="state.eqx")
```

Checkpoints store displacement, internal variables, and phase-field state where
applicable.

---

## 3. Step Configuration

Objects returned by `sim.add_step()`. Configuration is **not inherited** across steps.

### 3.1 Dirichlet and Neumann Conditions

```python
step.add_dirichlet_bc(on="x == 0", components=["x", "y", "z"], value=0.0,
                      amplitude=ds.amplitudes.Ramp(0, 1))
step.add_neumann_bc(on="x == 1", value=traction_vector, amplitude=amp)
```

Unified API:

```python
step.add_bc(field="u", kind="dirichlet", on="@bottom",
            components=["x", "y", "z"], value=0.0)
step.add_bc(field="d", kind="dirichlet", on="y == 0", value=0.0)
step.add_scalar_dirichlet_bc(field="d", on=..., value=...)
step.add_scalar_neumann_bc(field="d", on=..., value=...)
```

**Location selectors:** geometric strings (`"x == 0"`), `@physical_group`, or callables.

**Amplitudes** are evaluated at step-local time (Section 9).

### 3.2 Convenience Mechanical BCs

| Method | Description |
|--------|-------------|
| `add_symmetry_bc(on, plane="x")` | Symmetry about a coordinate plane |
| `add_encastre(on)` | Fully fixed (all translational DOFs) |
| `add_inclined_dirichlet_bc(on, direction=..., value=...)` | Dirichlet in rotated frame |

### 3.3 Damage and Fracture Dynamics

#### `step.set_damage_evolution(field=..., pde=..., integrator=..., eta=..., tau=..., rho_d=..., mu_d=..., lumped=...)`

Configures the damage PDE and time integrator for strategy axis 1–2.

#### `step.set_fracture_dynamics(mechanics_integrator=..., coupling_mode=..., cfl_safety=..., integrator=..., scan_mode=..., drive_field_mode=..., irreversibility_mode=..., damage_subcycles=..., explicit=...)`

Configures mechanics time integration and stagger coupling (strategy axis 3–4).
Setting `mechanics_integrator="explicit_central_difference"` marks the step as explicit dynamics.

**Explicit damage options** (`FractureExplicitOptions`):

| Field | Default | Description |
|-------|---------|-------------|
| `drive_field_mode` | `"current_energy"` | `"current_energy"` or `"history_max"` |
| `irreversibility_mode` | `"macaulay_rate"` | Irreversibility enforcement scheme |
| `damage_subcycles` | `False` | Sub-cycle damage integration |
| `clip_upper` | `True` | Clip damage to `[0, 1]` |
| `pseudo_solver_backend` | `"host_spsolve"` | S7 pseudo-parabolic backend |
| `pure_sobolev_research` | `False` | Research flag for pure Sobolev S7 |

### 3.4 Output Declarations

```python
step.add_history_output("F", on="@top", value="reaction.z", every=1)
step.add_energy_output(name="energy")
step.add_field_output(values=["u", "cauchy.von_mises", "d"], every=5)
step.add_field_snapshot(value="cauchy.zz", every=0.25, deformed=True,
                        output_dir="snaps/", cmap="viridis")
```

### 3.5 Advanced Constraints (Step Methods)

| Method | Underlying type |
|--------|-----------------|
| `add_equation_constraint(terms, rhs=..., amplitude=...)` | `_LinearMPC` |
| `add_periodic_bc(pbc)` | `_PeriodicBC` |
| `add_concentrated_force(target=..., components=..., value=...)` | `_ConcentratedLoad` |
| `add_pressure_bc(on=..., magnitude=..., follower=True)` | `_PressureBC` |
| `add_coupling_constraint(rp=..., on=..., type="kinematic")` | `_CouplingConstraint` |

See Section 10 for parameter semantics.

### 3.6 Per-Step Solver Overrides (`add_step` keyword arguments)

**Quasi-static Newton / adaptive:**

`adaptive`, `dt_min`, `dt_max`, `iter_boost`, `iter_keep`, `iter_cutback`,
`grow_factor`, `cutback_factor`, `max_retries`, `line_search`, `stabilize`,
`rtol`, `atol`, `max_iter`, `verbose`, …

**Explicit / implicit dynamics:**

| Key | Description |
|-----|-------------|
| `dynamics` | `None`, `"explicit"`, or `"implicit"` |
| `integrator` | `"auto"`, `"scan_energy"`, `"scan_residual"`, `"scan_stateful"`, `"python"` |
| `scan_mode` | JAX scan routing (`"auto"` or bool) |
| `cfl_safety` | CFL factor when `dt=None` |
| `wave_speed` | Override elastic wave speed for CFL |
| `mass_scale`, `target_dt`, `auto_mass_scale` | Mass scaling to reach target Δt |
| `use_element_cfl` | Element-wise CFL (default `True`) |
| `alpha_m`, `alpha_k` | Rayleigh damping |
| `beta`, `gamma` | Newmark parameters (implicit) |
| `initial_displacement`, `initial_velocity`, `initial_acceleration` | IC overrides |
| `save_every` | Override simulation-level output cadence |
| `cp_chunk_size` | Crystal plasticity scan chunk size |
| `verbose_dynamics` | Print time-step progress |

---

## 4. Phase-Field Fracture Strategies (S1–S7)

At solve time,
`validate_fracture_combo()` returns the strategy ID or raises `ValueError`.

### 4.1 Valid Combinations

| ID | Mechanics | Damage PDE | Damage integrator | Coupling |
|----|-----------|------------|-------------------|----------|
| **S1** | `quasi_static` | `elliptic` | `implicit` | `stagger_fixed_point` |
| **S2** | `explicit_central_difference` | `elliptic` | `implicit` | `stagger_one_pass` |
| **S3** | `explicit_central_difference` | `parabolic_viscous` | `explicit_euler` | `stagger_one_pass` |
| **S4** | `explicit_central_difference` | `inertial` | `explicit_verlet` | `stagger_one_pass` |
| **S5** | `quasi_static` | `parabolic_viscous` | `implicit` | `stagger_fixed_point` |
| **S6** | `quasi_static` | `pseudo_parabolic` | `implicit` | `stagger_fixed_point` |
| **S7** | `explicit_central_difference` | `pseudo_parabolic` | `explicit_euler_tau` | `stagger_one_pass` |

### 4.2 Configuration Axes

| Axis | Configuration | API |
|------|---------------|-----|
| 1–2 | Damage PDE + integrator | `DamageEvolution`, `step.set_damage_evolution` |
| 3–4 | Mechanics integrator + coupling | `step.set_fracture_dynamics` |
| 5 | Explicit damage options | `FractureExplicitOptions` fields on `set_fracture_dynamics` |

### 4.3 Damage PDE Types

| `pde` | Physical model |
|-------|----------------|
| `elliptic` | Quasi-static Allen–Cahn / AT-type elliptic damage |
| `parabolic_viscous` | Rate-dependent damage with viscosity η |
| `pseudo_parabolic` | Gradient viscosity with parameter τ |
| `inertial` | Second-order damage dynamics with ρ_d, μ_d |

### 4.4 Defaults

If `set_fracture_dynamics()` is omitted:

- Explicit damage integrators → dynamic mechanics + one-pass coupling.
- Implicit damage integrators → quasi-static mechanics + fixed-point coupling.

Worked examples: [quickstart.md](../quickstart.md).

---

## 5. Physics Descriptors

Namespace: `ds.physics`

### 5.1 `SolidMechanics`

```python
solid = ds.physics.SolidMechanics(
    material=mat,
    geometry="3d",              # "3d" | "plane_strain" | "plane_stress" | "axisymmetric"
    formulation="standard",   # "standard" | "bbar" | "fbar" | "fbar_patch" | "eas"
    kinematics=None,            # "strain" | "deformation_gradient"
    integration=None,         # Integration.FULL | REDUCED | int gauss order
    hourglass_alpha=None,       # Stabilisation for reduced integration
    hourglass_mu=None,
    density=None,               # Required for dynamics
    thickness=1.0,              # 2D plane models
    dt=1.0,
)
```

**Factory behaviour:** `geometry="axisymmetric"` redirects to `AxisymmetricSolidMechanics`.

| Method | Description |
|--------|-------------|
| `add_section(cells=..., material=..., name=...)` | Multi-material mesh sections |
| `add_body_force(value, amplitude=None)` | Volume force (summed over calls) |

Boundary conditions are **step-scoped** (Section 3). Physics-level `add_dirichlet_bc`
was removed.

### 5.2 `AxisymmetricSolidMechanics`

Dedicated 2D r–z axisymmetric backend. Prefer the unified entry point
`SolidMechanics(..., geometry="axisymmetric")`.

### 5.3 `PhaseField`

```python
pf = sim.add_physics(ds.physics.PhaseField(
    field="d",
    type=ds.physics.Fracture(
        degradation=degrad,  # AT2_Degradation(Gc=..., l0=...) — l0 is ℓ, not mesh h
        damage=ds.physics.Elliptic(),  # or DamageEvolution(pde="elliptic", integrator="implicit")
        driving="W_plus",
        history="max",
    ),
    integration=None,
))
```

Only `type="fracture"` (or `Fracture(...)`) is implemented. The phase-field length
scale is `l0` on the degradation object; mesh size `h` comes from the mesh.

### 5.4 `Fracture` / `PhaseFieldFracture`

Alias: `ds.physics.Fracture`. Bundles degradation law, default damage evolution,
driving variable name, and history mode.

### 5.5 Damage Descriptor Shortcuts

| Class | Maps to |
|-------|---------|
| `Elliptic()` | S1 elliptic + implicit |
| `Parabolic(eta=..., integrator=...)` | Parabolic viscous |
| `ParabolicViscous(eta=..., integrator="explicit_euler")` | S3 default |
| `PseudoParabolic(eta=..., tau=..., integrator=...)` | S6 / S7 |
| `InertialDamage(rho_d=..., mu_d=...)` | S4 inertial damage |

### 5.6 `DamageEvolution`

Dataclass with fields: `pde`, `integrator`, `eta`, `tau`, `rho_d`, `mu_d`, `lumped`, `dt`.
Also available as `ds.physics.DamageEvolution`.

### 5.7 Phase-Field Regions

```python
pf.add_section(cells=..., degradation=..., name=...)   # PFSection
pf.add_active_zone(cells=..., ...)                     # PFActiveZone
```

Restricts damage evolution or material degradation to mesh subsets.

---

## 6. Material Descriptors

Namespace: `ds.materials`

Facade classes wrap library `UPOT_*` / `UMAT_*` implementations.

### 6.1 Elasticity

| Class / function | Description |
|------------------|-------------|
| `Elasticity(E, nu)` | Linear isotropic elasticity |
| `NeoHookean(E, nu)` | Compressible Neo-Hookean |
| `Elasticity_with_eigenstrain(C11, C12, C44, eps_0=...)` | Cubic elasticity + eigenstrain |

### 6.2 Phase-Field Ready Solids

| Function | Description |
|----------|-------------|
| `FractureElasticity(E, nu, split="spectral", hybrid=False, degradation=...)` | Linear elasticity + PF state (`d`, `W_plus`) |
| `FractureNeoHookean(E, nu, split="vol_dev", hybrid=False)` | Finite-strain Neo-Hookean + PF |

**Strain splits:** `spectral`, `spectral_pff` (2D explicit eigenvalues), `amor`, `vol_dev`.

Pass `degradation=` so mechanics uses the matching `g(d)` from the degradation model.

### 6.3 Degradation Functions

| Class | Model |
|-------|-------|
| `AT1_Degradation(Gc, l0)` | AT1 linear crack geometric function |
| `AT2_Degradation(Gc, l0)` | AT2 quadratic function (standard) |
| `CohesiveDegradation(Gc, l0, sigma_c, E0, cohesive_law="linear")` | Cohesive-zone PF |

### 6.4 Dislocation / PFDD

| Class | Description |
|-------|-------------|
| `SlipEigenstrainElasticity(C11, C12, C44, slip_systems, burgers_over_d)` | Cubic elasticity driven by slip eigenstrain |

### 6.5 Advanced Constitutive Models (Package Root)

Plasticity, viscoelasticity, crystal plasticity, and hyperelastic potentials are
exported at package root for custom `SolidMechanics(material=...)` wiring:

```python
from diffsolid import UMAT_FeFpJ2Plasticity, UMAT_FCCCrystalPlasticity
from diffsolid import UPOT_MooneyRivlin, UPOT_Ogden
```

See Section 17 for the full list. These follow the JAX-UMAT interface
(`UserMaterial`, `UserPotential`).

---

## 7. Couplers

Namespace: `ds.couplers`

### 7.1 `Staggered`

Alternating minimization for quasi-static and one-pass dynamic fracture.

```python
sim.set_coupler(ds.couplers.Staggered(
    max_iter=50,
    tol=1e-4,
    stagger_dd_criterion="positive_increment",  # or "abs_max"
    stagger_damage_accel="off",                   # or "aitken"
    mech_solver=ds.solvers.NewtonSolver(...),
    pf_solver=ds.solvers.VINewtonSolver(preset="elliptic"),
    pf_linear_solver=ds.solvers.AMGx(),
    driving_force_fn=None,
    mech_single_linear=False,
    at2_history_mode=False,
))
```

**Driving force `H`:**

| Configuration | Behaviour |
|---------------|-----------|
| `driving_force_fn=None` | Read `psi_plus` from material internal state |
| Plain callable | `H = fn(u_sol, mech_problem)` each stagger pass |
| `HistoryMax(fn)` | `H = max(H_prev, fn(...))` — irreversible history |

**Deprecated:** `strain_split` on the coupler — specify split on
`FractureElasticity(split=...)` instead.

### 7.2 `FractureCoupling`

Subclass of `Staggered` with explicit `coupling_mode` for literature matrix
compatibility (`stagger_fixed_point` vs `stagger_one_pass`).

### 7.3 `HistoryMax`

```python
ds.couplers.HistoryMax(lambda u_sol, prob: raw_field(u_sol, prob))
```

Wrapper signalling history accumulation of a quadrature-point field
`(n_cells, n_quads)`.

### 7.4 `SlipStaggered` (PFDD)

Staggered coupling for slip Allen–Cahn + mechanics. Used with
`DislocationSimulation` (Section 14).

---

## 8. Solver Descriptors

Namespace: `ds.solvers`

### 8.1 Linear Solvers

| Class | Role |
|-------|------|
| `AMGx(...)` | NVIDIA AmgX AMG — **preferred GPU iterative solver** |
| `CUDSS(reorder="amd")` | GPU direct sparse solver |
| `AMGCL(gpu=True, relaxation="chebyshev", solver_type="fgmres", ...)` | Iterative AMG (CUDA extension) |
| `UMFPACK()` | CPU direct (default fallback) |
| `SciPy()` | CPU SciPy backend |
| `BiCGSTAB(tol=..., maxiter=...)` | JAX iterative (VI inner solves) |
| `CG(tol=..., maxiter=...)` | JAX CG for SPD systems |

**AmgX notes:**

- Preferred for large implicit elasticity / plasticity on NVIDIA GPUs (H100/H200 class).
- Requires AmgX + pyamgx (optional GPU extra); see Install guide.

**AMGCL notes:**

- Use `relaxation="chebyshev"` for 3D elasticity and plasticity; `spai0`/`ilu0`
  may stagnate.
- `cache_slot` (0–3): separate caches for mechanics vs phase-field hierarchies.
- `symmetric_dirichlet=True` required for `solver_type="cg"`.

**Selection guide:**

| Problem | Default |
|---------|---------|
| Large 2D/3D elastic / PF elliptic (GPU) | `AMGx()` |
| 3D plasticity, necking, arc-length | `CUDSS(reorder="amd")` or `AMGx()` |
| GPU without AmgX installed | `AMGCL(gpu=True, relaxation="chebyshev")` |
| 2D moderate DOF | `CUDSS` or `UMFPACK` |
| Explicit dynamics | No per-step linear solve |

### 8.2 Nonlinear Solvers

| Class | Role |
|-------|------|
| `NewtonSolver(linear_solver=..., tol=..., rel_tol=..., max_iter=...)` | Newton–Raphson for mechanics / elliptic residuals |
| `LBFGSSolver(maxiter=..., gtol=..., ftol=...)` | Energy minimisation for PF |
| `VINewtonSolver(preset="elliptic", linear_solver=..., ...)` | VI active-set PF with irreversibility |

**VI presets:** `"elliptic"`, `"parabolic"`, `"hyperbolic"`, `"pff_miehe"`.

Key VI options: `vi_reduced_space`, `vi_fast_jax_linear`, `vi_fullspace_block_elim`,
`line_search`, `vi_lbfgs_fallback`.

Aliases: `Newton = NewtonSolver`, `LBFGS = LBFGSSolver`.

### 8.3 Dynamics Solver Descriptors

| Class | Role |
|-------|------|
| `ExplicitDynamicsSolver(damping=0.0)` | Pure mechanics explicit integration |
| `ImplicitDynamicsSolver(linear_solver=..., beta=0.25, gamma=0.5, ...)` | Newmark-beta implicit |

Set via `sim.set_dynamics_solver(...)` or step `dynamics="explicit"` / `"implicit"`.

---

## 9. Amplitudes

Namespace: `ds.amplitudes`

Time-dependent scale factors for BC values: `prescribed = reference_value × amplitude(t_local)`.

| Class | Description |
|-------|-------------|
| `Ramp(t_start, t_end, v_start=0, v_end=1)` | Linear ramp, clamped outside interval |
| `Smooth(t_start, t_end, v_start=0, v_end=1)` | C² Hermite smoothstep |
| `Step(t_step, v_before=0, v_after=1)` | Heaviside step |
| `Tabular(times, values, extrapolate="clamp")` | Piecewise-linear table |
| `Sine(amplitude, frequency, phase=0, offset=0)` | Harmonic loading |
| `Constant(value=1.0)` | Unit scale |
| `Amplitude` | Abstract base; supports `+`, `*` composition |

---

## 10. Constraints and Advanced Boundary Conditions

Namespace: `ds.constraints` (types); attached via step methods (Section 3.5).

### 10.1 `_LinearMPC`

General linear multi-point constraint: Σ c_k u_{dof_k} = b(t).

### 10.2 `_PeriodicBC`

Periodic boundary conditions on a box RVE.

| Parameter | Description |
|-----------|-------------|
| `formulation` | `"auto"`, `"small_strain"`, `"large_strain"` |
| `axes` | Periodic axes (default: all) |
| `eps_bar` / `F_bar` | Macro-strain or deformation-gradient driver |
| `lattice` | Optional cell dimensions for mod-L pairing |
| `anchor` | Rigid-body anchor node |

Small strain: u(x+) − u(x−) = ε̄ · L.  
Large strain: u(X+) − u(X−) = (F̄ − I) · L.

### 10.3 `_ConcentratedLoad`

Point force at a node or reference point.

### 10.4 `_PressureBC`

Surface pressure; `follower=True` uses deformed outward normal.

### 10.5 `_InclinedDirichlet`

Dirichlet constraint along a rotated direction vector.

### 10.6 `_CouplingConstraint`

Tie slave surface to master reference point (`kinematic` or `distributing`).

---

## 11. Output, Results, and Post-Processing

### 11.1 Field Output Keys

Used in `field_output`, `add_field_output`, and `add_field_snapshot`:

| Key pattern | Quantity |
|-------------|----------|
| `"u"` | Displacement |
| `"d"` | Phase-field damage |
| `"cauchy.von_mises"` | von Mises stress |
| `"cauchy.{xx,yy,zz,xy,yz,zx}"` | Cauchy components |
| `"strain.{...}"` | Strain components |
| `"displacement.{x,y,z,mag}"` | Nodal displacement |
| UMAT state name | e.g. `"alpha"`, `"ep"` |

### 11.2 XDMF / VTU

- `output_format="xdmf"`: accumulates `output_dir/results.xdmf` + `.h5`.
- `output_format="vtu"`: per-step VTU files.
- `output_format="none"`: no disk output (benchmarks).

### 11.3 `Result` Object

Returned by `sim.solve()`. Subclasses `dict` — legacy `result["u"]` still works.

| Attribute | Description |
|-----------|-------------|
| `u`, `d`, `H` | Primary fields |
| `sigma`, `von_mises`, `strain` | Lazy fields via `StepSnapshot` |
| `state`, `state_dict` | Internal variables |
| `history` | User history outputs |
| `diagnostics` | Newton iteration records |
| `performance` | Timing report |
| `converged`, `iterations`, `wall_time` | Summary accessors |
| `at_node(i)`, `at_element(e)` | `NodeView`, `ElementView` |

### 11.4 Post-Processing Namespace

`import diffsolid.post as post` or `from diffsolid import post`

**Stress and strain recovery:**

`gauss_stress`, `gauss_strain`, `nodal_average`, `von_mises`, `hydrostatic`,
`principal_stresses`, `reaction_force_2d`, `reaction_force_3d`, …

**Plotting:**

`post.plot(...)`, `post.contour(...)`, `post.plot_fields(...)`

**Phase-field diagnostics:**

`crack_surface_density`, `dissipated_energy`, `crack_length`,
`phase_field_profile`, `build_pf_frame_data`, `save_pf_xdmf`

---

## 12. Mesh I/O and Checkpoints

Exported at package root (`ds.*`):

| Function | Description |
|----------|-------------|
| `parse_mesh_file(path)` | Parse external mesh file |
| `rectangle_mesh`, `box_mesh`, `box_mesh_gmsh`, `cylinder_mesh_gmsh` | Built-in mesh generators |
| `save_sol`, `save_sol_extended` | Write solution to VTK/XDMF |
| `save_checkpoint`, `load_checkpoint` | JAX/checkpoint I/O |
| `Mesh` | Core mesh object |
| `MeshParser` | Structured mesh parsing helper |

---

## 13. Pure Mechanics Dynamics

For simulations **without** phase field.

### 13.1 Step-Based (Preferred)

```python
step = sim.add_step(name="wave", duration=T, dt=dt, dynamics="explicit",
                    cfl_safety=0.9, integrator="auto")
sim.solve(output_dir="results/")
```

### 13.2 Legacy Entry Points

Still available on `Simulation`:

```python
sim.solve_explicit_dynamics(t_end, dt, output_dir=..., save_every=...,
                            damping=0.0, u0=..., v0=..., callback=...)
sim.solve_implicit_dynamics(t_end, dt, beta=0.25, gamma=0.5, ...)
```

Prefer `add_step` + `sim.solve()` for new workflows.

### 13.3 Utility Functions

```python
from diffsolid import build_lumped_mass, estimate_dt_crit, estimate_dynamics_dt
```

---

## 14. Phase-Field Dislocation Dynamics (PFDD)

Parallel facade for slip-based dislocation modelling.

### 14.1 `DislocationSimulation`

Mirrors `Simulation` with slip fields instead of fracture damage:

```python
sim = ds.DislocationSimulation(name="pfdd", dim=3, ele_type="HEX8")
sim.load_mesh(mesh)
sim.add_physics(ds.physics.SolidMechanics(
    material=ds.materials.SlipEigenstrainElasticity(...)))
sim.add_physics(ds.physics.Dislocation(
    field="eta", slip_systems=slip, crystal_energy=energy, beta=0.02, L=1.0))
sim.set_coupler(ds.couplers.SlipStaggered(max_iter=30, tol=1e-3))
sim.set_initial_slip([eta1_0, eta2_0])
result = sim.solve(...)
eta = result.eta   # list of slip fields per system
```

### 14.2 GPU Baseline Helpers

```python
ds.apply_pfdd_gpu_baseline(sim, mech="cudss", slip="amgcl")
ds.pfdd_amgcl_mech_baseline()
ds.pfdd_amgcl_slip_baseline()
```

Default pairs **mechanics = CUDSS** with **slip = AMGCL(gpu=True)** to avoid GPU
memory contention.

### 14.3 Related Types

- `ds.physics.Dislocation` — slip Allen–Cahn descriptor (in `dislocation` module)
- `ds.couplers.SlipStaggered` — staggered mechanics + slip solve

---

## 15. Inverse Problems and Optimisation

### 15.1 Simulation Facade

```python
params = sim.list_inverse_params(include_mech=True, include_field=True)
problem = sim.inverse(params, observes=obs_list, mode="transient", dt=1.0)
```

Returns a configured `InverseProblem` for gradient-based optimisation.

### 15.2 Inverse Namespace

`import diffsolid.inverse as inv` or top-level exports:

| Symbol | Description |
|--------|-------------|
| `from_simulation(sim, ...)` | Build inverse problem from configured simulation |
| `InverseResult` | Optimisation result container |
| `pack_params` | Flatten parameters for JAX |
| `implicit_vjp`, `ad_wrapper`, `ad_wrapper_transient` | Adjoint drivers |
| `mse_loss`, `stress_strain_mse` | Loss functions |
| `gradient_check` | Finite-difference verification |
| `compliance_objective`, `volume_fraction` | Topology optimisation helpers |
| `simp_penalization`, `heaviside_projection` | SIMP / projection filters |
| `make_topology_opt_objective` | Combined TO objective factory |

---

## 16. Type Aliases and Enumerations

Exported at package root:

### 16.1 `Kinematics`

`"strain"` | `"deformation_gradient"` — kinematic input expected by the material.

### 16.2 `Formulation`

| Value | Method |
|-------|--------|
| `"standard"` | Standard displacement formulation |
| `"bbar"` | B-bar volumetric mean (small strain) |
| `"fbar"` | F-bar (finite strain) |
| `"fbar_patch"` | F-bar patch (TRI3/TET4) |
| `"eas"` | Enhanced Assumed Strain |

Pair `"standard"` + `integration="reduced"` with `hourglass_alpha` for explicit
dynamics on QUAD4/HEX8.

### 16.3 `Geometry`

`"3d"` | `"plane_strain"` | `"plane_stress"` | `"axisymmetric"`

### 16.4 `Integration` (Enum)

`Integration.FULL` | `Integration.REDUCED` — quadrature rules per element type.
Integer gauss orders also accepted (FEniCS-style).

---

## 17. Environment Variables


| Variable | Description |
|----------|-------------|
| `DIFFSOLID_SCAN_CHUNK` | Max steps per JAX `lax.scan` chunk (default 8192) |
| `DIFFSOLID_SCAN_CHUNK_MIN` | Minimum chunk floor |
| `DIFFSOLID_SCAN_CHUNK_LARGE` | Chunk cap for large DOF problems |
| `DIFFSOLID_SCAN_LARGE_NDOF` | DOF threshold for large-problem path |
| `DIFFSOLID_ENABLE_LEGACY_EXPLICIT_FAST_PATHS` | Experimental fast paths (`0` default) |
| `DIFFSOLID_EXPLICIT_PROFILE_LEVEL` | `summary`, `operator`, `memory`, `trace` |
| `DIFFSOLID_STAGGER_DAMAGE_ACCEL` | `off` or `aitken` stagger acceleration |
| `DIFFSOLID_PERF` | Enable performance reporting |

---

## 18. Appendix — Legacy API Migration

Removed keys raise `ValueError` with a migration table.

| Legacy | Replacement |
|--------|-------------|
| `equation="elliptic"` / `model=` | `DamageEvolution(pde="elliptic", integrator="implicit")` |
| `equation="parabolic"` | `pde="parabolic_viscous"` + integrator |
| `equation="hyperbolic"` | **Not** inertial damage — use `parabolic_viscous` + explicit dynamics |
| `time_scheme="implicit"` / `"explicit"` | `integrator="implicit"` / `"explicit_euler"` |
| `sim.solve(time_steps=..., dt=...)` | `add_step(duration=..., dt=...)` then `sim.solve()` |
| Physics-level BC methods | Step-level `add_dirichlet_bc` / `add_bc` |
| `set_field_evolution`, `set_phase_field` | `set_damage_evolution` |
| `Staggered(strain_split=...)` | `FractureElasticity(split=...)` on material |

---

## References

- Theory: [formulations.md](../theory/formulations.md)
- Installation: [install.md](../install.md)
- Quick examples: [quickstart.md](../quickstart.md)
