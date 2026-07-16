# Requires: pip install from GitHub Release wheel
# See ../docs/install.md

"""Quasi-static staggered phase-field fracture (Strategy S1)."""

import diffsolid as ds


def main() -> None:
    sim = ds.Simulation(name="bar_fracture", dim=2, ele_type="QUAD4")
    sim.load_mesh("meshes/bar.msh")

    steel = ds.materials.FractureElasticity(E=210e3, nu=0.3, split="spectral")
    sim.add_physics(ds.physics.SolidMechanics(material=steel))

    # l0 = regularisation length ℓ (mesh size h comes from the mesh).
    degrad = ds.materials.AT2_Degradation(Gc=2.7e-3, l0=0.01)
    pf = sim.add_physics(
        ds.physics.PhaseField(
            field="d",
            type=ds.physics.Fracture(
                degradation=degrad,
                damage=ds.physics.Elliptic(),
                driving="W_plus",
                history="max",
            ),
        )
    )

    sim.set_coupler(
        ds.couplers.Staggered(
            max_iter=50,
            tol=1e-4,
            pf_solver=ds.solvers.VINewtonSolver(preset="elliptic"),
        )
    )
    sim.set_linear_solver(ds.solvers.AMGx())

    step = sim.add_step(name="load", duration=1.0, dt=0.01)
    step.add_dirichlet_bc(on="x == 0", components=["x", "y"], value=0.0)
    step.add_dirichlet_bc(
        on="x == 1",
        components=["x"],
        value=1e-3,
        amplitude=ds.amplitudes.Ramp(0.0, 1.0),
    )

    sim.solve(output_dir="results/", save_every=5)
    print(f"Strategy S1 complete; phase field: {pf.field}")


if __name__ == "__main__":
    main()
