# DiffSolid Quick Start

Minimal examples for the two most common phase-field fracture workflows.

---

## Prerequisites

Install the package from a [GitHub Release wheel](install.md#installing-from-github-releases), then verify:

```bash
python -c "import diffsolid as ds; print(ds.__name__)"
```

See [install.md](install.md) for GPU setup.

---

## Example 1 — Quasi-static staggered fracture (Strategy S1)

Classical implicit elliptic phase-field with alternating minimization between
mechanics and damage fields.

```python
import diffsolid as ds

sim = ds.Simulation(name="bar_fracture", dim=2, ele_type="QUAD4")
sim.load_mesh("bar.msh")

steel = ds.materials.FractureElasticity(E=210e3, nu=0.3, split="spectral")
solid = sim.add_physics(ds.physics.SolidMechanics(material=steel))

degrad = ds.materials.AT2_Degradation(Gc=2.7e-3, l0=0.01)
pf = sim.add_physics(ds.physics.PhaseField(
    field="d",
    type=ds.physics.Fracture(
        degradation=degrad,
        damage=ds.physics.DamageEvolution(pde="elliptic", integrator="implicit"),
    ),
))

sim.set_coupler(ds.couplers.Staggered(max_iter=50, tol=1e-4))
sim.set_linear_solver(ds.solvers.AMGCL(gpu=True, relaxation="chebyshev"))

step = sim.add_step(name="load", duration=1.0, dt=0.01)
step.add_dirichlet_bc(on="x == 0", components=["x", "y"], value=0.0)
step.add_dirichlet_bc(on="x == 1", components=["x"], value=1e-3,
                      amplitude=ds.amplitudes.Ramp(0.0, 1.0))

sim.solve(output_dir="results/", save_every=5)
```

**Strategy ID:** S1 — `quasi_static` + `elliptic` + `implicit` + `stagger_fixed_point`.

---

## Example 2 — Explicit dynamic fracture (Strategy S3)

Central-difference mechanics coupled with explicit Euler integration of a
parabolic viscous damage equation (one-pass stagger).

```python
import diffsolid as ds

sim = ds.Simulation(name="mersel_bar", dim=3, ele_type="HEX8")
sim.load_mesh("bar.msh")

degrad = ds.materials.AT2_Degradation(Gc=2.7e-3, l0=0.05)
sim.add_physics(ds.physics.SolidMechanics(
    material=ds.materials.FractureElasticity(
        E=210e3, nu=0.3, split="spectral", degradation=degrad,
    ),
    density=8000.0,
))
pf = sim.add_physics(ds.physics.PhaseField(
    field="d",
    type=ds.physics.Fracture(
        degradation=degrad,
        damage=ds.physics.DamageEvolution(
            pde="parabolic_viscous",
            integrator="explicit_euler",
            eta=1e-2,
        ),
    ),
))

step = sim.add_step(name="dynamic", duration=2e-6, dt=1e-6)
step.add_bc(field="u", kind="dirichlet", on="x == 0",
            components=["x", "y", "z"], value=0.0)
step.add_bc(field="u", kind="dirichlet", on="x == 1",
            components=["x"], value=1e-3)

step.set_damage_evolution(field=pf.field, pde="parabolic_viscous",
                          integrator="explicit_euler", eta=1e-2)
step.set_fracture_dynamics(
    mechanics_integrator="explicit_central_difference",
    coupling_mode="stagger_one_pass",
    drive_field_mode="current_energy",
    irreversibility_mode="macaulay_rate",
)

sim.solve(output_dir="results/", output_format="none")
```

**Strategy ID:** S3 — `explicit_central_difference` + `parabolic_viscous` +
`explicit_euler` + `stagger_one_pass`.

---

## Strategy Reference

| ID | Mechanics | Damage PDE | Damage integrator | Coupling |
|----|-----------|------------|-------------------|----------|
| S1 | quasi_static | elliptic | implicit | stagger_fixed_point |
| S2 | explicit_central_difference | elliptic | implicit | stagger_one_pass |
| S3 | explicit_central_difference | parabolic_viscous | explicit_euler | stagger_one_pass |
| S4 | explicit_central_difference | inertial | explicit_verlet | stagger_one_pass |
| S5 | quasi_static | parabolic_viscous | implicit | stagger_fixed_point |
| S6 | quasi_static | pseudo_parabolic | implicit | stagger_fixed_point |
| S7 | explicit_central_difference | pseudo_parabolic | explicit_euler_tau | stagger_one_pass |

Full API details: [api/index.md](api/index.md).

---

## Next Steps

- Solver selection: [Solver descriptors](api/index.md#8-solver-descriptors)
- Step-scoped BCs and history output: [Step configuration](api/index.md#3-step-configuration)
- Theory: [formulations.md](theory/formulations.md)
