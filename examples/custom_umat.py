# Requires: pip install from GitHub Release wheel
# See ../docs/install.md

"""Custom UMAT — small-strain linear elasticity via UserMaterial."""

import jax.numpy as jnp
import diffsolid as ds
from diffsolid import UserMaterial


class LinearElasticUMAT(UserMaterial):
    kinematics = "strain"
    state_fields = {}

    def __init__(self, E: float, nu: float) -> None:
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        self.lam, self.mu = lam, mu

    def umat(self, eps, state, dt):
        dim = eps.shape[0]
        sigma = self.lam * jnp.trace(eps) * jnp.eye(dim) + 2 * self.mu * eps
        return sigma, state


def main() -> None:
    sim = ds.Simulation(name="umat_plate", dim=2, ele_type="QUAD4")
    sim.load_mesh("meshes/bar.msh")

    mat = LinearElasticUMAT(E=210e3, nu=0.3)
    sim.add_physics(ds.physics.SolidMechanics(material=mat))
    sim.set_linear_solver(ds.solvers.AMGx())

    step = sim.add_step(name="stretch", duration=1.0, dt=0.05)
    step.add_dirichlet_bc(on="x == 0", components=["x", "y"], value=0.0)
    step.add_dirichlet_bc(on="x == 1", components=["x"], value=1e-3)

    sim.solve(output_dir="results/")
    print("Custom UMAT solve complete.")


if __name__ == "__main__":
    main()
