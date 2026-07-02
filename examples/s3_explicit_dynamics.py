# Requires: pip install from GitHub Release wheel
# See ../docs/install.md

"""Explicit dynamic fracture with parabolic viscous damage (Strategy S3)."""

import diffsolid as ds


def main() -> None:
    sim = ds.Simulation(name="mersel_bar", dim=3, ele_type="HEX8")
    sim.load_mesh("meshes/bar.msh")

    degrad = ds.materials.AT2_Degradation(Gc=2.7e-3, l0=0.05)
    sim.add_physics(
        ds.physics.SolidMechanics(
            material=ds.materials.FractureElasticity(
                E=210e3,
                nu=0.3,
                split="spectral",
                degradation=degrad,
            ),
            density=8000.0,
        )
    )
    pf = sim.add_physics(
        ds.physics.PhaseField(
            field="d",
            type=ds.physics.Fracture(
                degradation=degrad,
                damage=ds.physics.DamageEvolution(
                    pde="parabolic_viscous",
                    integrator="explicit_euler",
                    eta=1e-2,
                ),
            ),
        )
    )

    step = sim.add_step(name="dynamic", duration=2e-6, dt=1e-6)
    step.add_bc(field="u", kind="dirichlet", on="x == 0", components=["x", "y", "z"], value=0.0)
    step.add_bc(field="u", kind="dirichlet", on="x == 1", components=["x"], value=1e-3)

    step.set_damage_evolution(
        field=pf.field,
        pde="parabolic_viscous",
        integrator="explicit_euler",
        eta=1e-2,
    )
    step.set_fracture_dynamics(
        mechanics_integrator="explicit_central_difference",
        coupling_mode="stagger_one_pass",
        drive_field_mode="current_energy",
        irreversibility_mode="macaulay_rate",
    )

    sim.solve(output_dir="results/", output_format="none")
    print("Strategy S3 complete")


if __name__ == "__main__":
    main()
