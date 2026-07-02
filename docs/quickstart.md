# DiffSolid Quick Start

Worked examples for common simulation setups and custom constitutive models.

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

## Example 3 — Custom UMAT (user-defined material)

Define your own constitutive model by subclassing `UserMaterial` and passing
an instance to `SolidMechanics`. The framework assembles the weak form and
builds the consistent tangent with JAX automatic differentiation.

### Interface

| Item | Meaning |
|------|---------|
| `kinematics` | `"strain"` → input symmetric strain `ε`, return Cauchy stress `σ`; `"deformation_gradient"` → input `F`, return PK1 stress `P` |
| `state_fields` | Persistent Gauss-point variables, e.g. `{"eps_p": "tensor", "alpha": "scalar"}`; use `{}` if stateless |
| `umat(eps_or_F, state, dt)` | Return `(stress, state_new)`; extra dict keys become output fields |

Use **`jnp.*` / `jax.lax.*` only** inside `umat` (no Python `if` on traced arrays).

### Stateless linear elasticity

```python
import jax.numpy as jnp
import diffsolid as ds
from diffsolid import UserMaterial


class LinearElasticUMAT(UserMaterial):
    kinematics = "strain"
    state_fields = {}

    def __init__(self, E: float, nu: float):
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        self.lam, self.mu = lam, mu

    def umat(self, eps, state, dt):
        dim = eps.shape[0]
        sigma = self.lam * jnp.trace(eps) * jnp.eye(dim) + 2 * self.mu * eps
        return sigma, state


sim = ds.Simulation(name="umat_plate", dim=2, ele_type="QUAD4")
sim.load_mesh("plate.msh")

mat = LinearElasticUMAT(E=210e3, nu=0.3)
sim.add_physics(ds.physics.SolidMechanics(material=mat))
sim.set_linear_solver(ds.solvers.AMGCL(gpu=True))

step = sim.add_step(name="stretch", duration=1.0, dt=0.05)
step.add_dirichlet_bc(on="x == 0", components=["x", "y"], value=0.0)
step.add_dirichlet_bc(on="x == 1", components=["x"], value=1e-3)

sim.solve(output_dir="results/")
```

### Stateful UMAT (history variables)

Declare `state_fields`, read/write them in `umat`, and return the updated dict:

```python
class J2PlasticityUMAT(UserMaterial):
    kinematics = "strain"
    state_fields = {"eps_p": "tensor", "alpha": "scalar"}

    def __init__(self, E, nu, sigma_y, H):
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        self.sigma_y, self.H = sigma_y, H

    def umat(self, eps, state, dt):
        eps_p, alpha = state["eps_p"], state["alpha"]
        dim = eps.shape[0]
        eps_e = eps - eps_p
        sigma = self.lam * jnp.trace(eps_e) * jnp.eye(dim) + 2 * self.mu * eps_e
        # ... radial return (use jnp.where, not Python if) ...
        return sigma, {"eps_p": eps_p_new, "alpha": alpha_new}
```

Use the same `Simulation` / `add_step` / `solve` workflow as built-in materials.
Optional: request extra outputs in `step.add_output(...)` using names from
`state_fields` or keys in the returned state dict.

Runnable script: [examples/custom_umat.py](https://github.com/zclsjtu/DiffSolid/blob/main/examples/custom_umat.py).

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
