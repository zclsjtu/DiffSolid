# Requires: pip install preview wheel from email — see ../docs/install.md

"""Quasi-static finite-strain J2 plasticity (Fe–Fp, F-bar HEX8)."""

import diffsolid as ds
from diffsolid import NonlinearIsotropicHardening, UMAT_FeFpJ2Plasticity


def main() -> None:
    sim = ds.Simulation(name="fe_fp_bar", dim=3, ele_type="HEX8")
    sim.load_mesh("meshes/bar.msh")

    mat = UMAT_FeFpJ2Plasticity(
        mu=80.0,
        kappa=164.0,
        yield_stress=NonlinearIsotropicHardening(sig_Y=50.0, K=500.0),
    )
    sim.add_physics(
        ds.physics.SolidMechanics(
            material=mat,
            geometry="3d",
            formulation="fbar",
        )
    )
    sim.set_linear_solver(ds.solvers.AMGx())

    step = sim.add_step(name="tension", duration=1.0, dt=0.02, line_search=True)
    step.add_dirichlet_bc(on="x == 0", components=["x", "y", "z"], value=0.0)
    step.add_dirichlet_bc(
        on="x == 1",
        components=["x"],
        value=0.05,
        amplitude=ds.amplitudes.Ramp(0.0, 1.0),
    )

    sim.solve(output_dir="results/", save_every=10)
    print("Finite-strain Fe–Fp tension complete.")


if __name__ == "__main__":
    main()
