# Requires: pip install from approved wheel — see ../docs/install.md

"""Explicit dynamics on a linear elastic bar (mechanics only)."""

import diffsolid as ds
from diffsolid import UPOT_LinearElasticity


def main() -> None:
    sim = ds.Simulation(name="elastic_wave", dim=3, ele_type="HEX8")
    sim.load_mesh("meshes/bar.msh")

    mat = UPOT_LinearElasticity(E=210e3, nu=0.3, density=7850.0)
    sim.add_physics(ds.physics.SolidMechanics(material=mat, geometry="3d"))

    step = sim.add_step(
        name="wave",
        duration=2e-5,
        dt=None,
        dynamics="explicit",
        cfl_safety=0.5,
        integrator="scan_energy",
    )
    step.add_dirichlet_bc(on="x == 0", components=["x", "y", "z"], value=0.0)
    step.add_energy_output("energy")

    sim.solve(output_dir="results/", output_format="none")
    print("Explicit dynamics complete.")


if __name__ == "__main__":
    main()
