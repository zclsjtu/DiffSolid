<div class="ds-home" markdown="1">

<section class="ds-home__hero">

<p class="ds-home__kicker">Differentiable Solid Mechanics</p>
<h1 class="ds-home__title">DiffSolid</h1>
<p class="ds-home__lead">
JAX-native finite element framework for nonlinear mechanics, GPU sparse solvers,
and validated phase-field fracture workflows.
</p>

<div class="ds-home__actions">
  <a href="quickstart/">Quick Start</a>
  <a href="install/">Install</a>
  <a href="api/">API</a>
</div>

<ul class="ds-home__tags">
  <li>JAX · AD</li>
  <li>FEM</li>
  <li>Phase-field S1–S7</li>
  <li>AMGCL · cuDSS</li>
</ul>

</section>

<div class="ds-home__grid">

<a class="ds-tile" href="quickstart/">
  <span class="ds-tile__label">Examples</span>
  <span class="ds-tile__title">Quick Start</span>
  <span class="ds-tile__desc">S1 quasi-static and S3 explicit dynamics.</span>
</a>

<a class="ds-tile" href="api/">
  <span class="ds-tile__label">Reference</span>
  <span class="ds-tile__title">API</span>
  <span class="ds-tile__desc">Simulation, steps, materials, solvers.</span>
</a>

<a class="ds-tile" href="install/">
  <span class="ds-tile__label">Setup</span>
  <span class="ds-tile__title">Installation</span>
  <span class="ds-tile__desc">Wheel, JAX GPU, AMGCL, cuDSS.</span>
</a>

<a class="ds-tile" href="theory/formulations/">
  <span class="ds-tile__label">Theory</span>
  <span class="ds-tile__title">Formulations</span>
  <span class="ds-tile__desc">FE weak forms, constitutive models, solvers.</span>
</a>

</div>

</div>

!!! info "Documentation only"
    Public docs and examples. Solver binaries ship via GitHub Releases under a proprietary license.

## Architecture

```mermaid
flowchart LR
  api[API] --> sched[Scheduler]
  sched --> eq[Weak forms]
  sched --> mat[Materials]
  sched --> sol[Solvers]
  sol --> gpu[GPU backends]
```

## Phase-field strategies (S1–S7)

| ID | Mechanics | Damage PDE | Integrator | Coupling |
|----|-----------|------------|------------|----------|
| S1 | quasi_static | elliptic | implicit | stagger |
| S2 | explicit CD | elliptic | implicit | one_pass |
| S3 | explicit CD | parabolic | explicit | one_pass |
| S4 | explicit CD | inertial | verlet | one_pass |
| S5 | quasi_static | parabolic | implicit | stagger |
| S6 | quasi_static | pseudo_par | implicit | stagger |
| S7 | explicit CD | pseudo_par | explicit | one_pass |

Details: [API §4](api/index.md#4-phase-field-fracture-strategies-s1s7).
