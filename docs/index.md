<div class="ds-home" markdown="1">

<section class="ds-home__hero">

<p class="ds-home__kicker">Simulation Platform</p>
<h1 class="ds-home__title">DiffSolid</h1>
<p class="ds-home__lead">
A JAX-native finite element environment for building, running, and differentiating
nonlinear solid mechanics simulations on CPU and GPU.
</p>

<div class="ds-home__actions">
  <a href="quickstart/">Quick Start</a>
  <a href="install/">Install</a>
  <a href="api/">API</a>
</div>

<ul class="ds-home__tags">
  <li>Nonlinear FEM</li>
  <li>Automatic differentiation</li>
  <li>GPU solvers</li>
  <li>Multi-physics workflows</li>
</ul>

</section>

<div class="ds-home__grid">

<a class="ds-tile" href="quickstart/">
  <span class="ds-tile__label">Examples</span>
  <span class="ds-tile__title">Quick Start</span>
  <span class="ds-tile__desc">Minimal scripts to set up and run a simulation.</span>
</a>

<a class="ds-tile" href="api/">
  <span class="ds-tile__label">Reference</span>
  <span class="ds-tile__title">API</span>
  <span class="ds-tile__desc">Simulation setup, materials, solvers, and output.</span>
</a>

<a class="ds-tile" href="install/">
  <span class="ds-tile__label">Setup</span>
  <span class="ds-tile__title">Installation</span>
  <span class="ds-tile__desc">Package install and optional GPU backends.</span>
</a>

<a class="ds-tile" href="theory/formulations/">
  <span class="ds-tile__label">Theory</span>
  <span class="ds-tile__title">Formulations</span>
  <span class="ds-tile__desc">Finite element and constitutive theory reference.</span>
</a>

</div>

</div>

!!! info "Documentation only"
    Public docs and examples. Solver binaries ship via GitHub Releases under a proprietary license.

## What you can do

- **Build FE models** — load meshes, assign elements, boundary conditions, and loading steps through a unified Python API.
- **Run nonlinear mechanics** — quasi-static and dynamic analysis with a library of constitutive models and element formulations.
- **Couple multiple physics** — staggered and monolithic multi-field workflows from a single simulation manager.
- **Scale on GPU** — optional sparse linear algebra backends for large 3D problems.
- **Differentiate and calibrate** — JAX-native assembly supports gradient-based inverse problems and parameter identification.
- **Post-process and export** — VTK output, checkpoints, and built-in post-processing hooks.

## Architecture

```mermaid
flowchart LR
  api[User API] --> sched[Scheduler]
  sched --> eq[Weak forms]
  sched --> mat[Materials]
  sched --> sol[Solvers]
  sol --> gpu[GPU backends]
```

Specific problem setups and advanced physics options are documented in the [API reference](api/index.md) and [theory](theory/formulations.md) sections.
