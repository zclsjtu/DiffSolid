# DiffSolid

**JAX-native differentiable finite element framework** for nonlinear solid mechanics,
phase-field fracture, GPU-accelerated sparse linear algebra, explicit and implicit
dynamics, and gradient-based inverse problems.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-indigo)](https://zclsjtu.github.io/DiffSolid/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Proprietary-red)]()

```python
import diffsolid as ds

sim = ds.Simulation(name="demo", dim=3, ele_type="HEX8")
sim.load_mesh("mesh.msh")
sim.add_physics(ds.physics.SolidMechanics(material=mat))
result = sim.solve(output_dir="results/")
```

---

## Highlights

- **Differentiable FEM** — JAX-native assembly, adjoint-friendly workflows, double precision by default
- **Phase-field fracture** — Validated strategy matrix **S1–S7** (quasi-static, explicit dynamics, inertial damage)
- **GPU solvers** — AMGCL CUDA algebraic multigrid, NVIDIA cuDSS direct sparse, JAX iterative backends

---

## Documentation

| Resource | Description |
|----------|-------------|
| [Documentation site](https://zclsjtu.github.io/DiffSolid/) | Install, API, theory, quick start |
| [Install](docs/install.md) | Wheel install, GPU setup, AMGCL build |
| [API Reference](docs/api/index.md) | User-facing Python API |
| [Quick Start](docs/quickstart.md) | S1 quasi-static and S3 explicit dynamics |
| [Gallery](gallery/) | Benchmark figures (work in progress) |

---

## Installing the software

This repository contains **documentation, examples, and gallery assets only**.

The solver is distributed as a Python wheel via **GitHub Releases** on the private
source repository (`DiffSolid-core`). See the [installation guide](docs/install.md).

```bash
pip install "https://github.com/zclsjtu/DiffSolid-core/releases/download/v0.1.0-dev.1/diffsolid-0.1.0.dev1-py3-none-any.whl"
```

> Preview wheels may include Python source inside the package. Compiled wheels without
> source will be published when the public API stabilises.

---

## Examples

API-only scripts (require an installed wheel):

```bash
pip install <wheel-url>
python examples/s1_quasi_static.py
python examples/s3_explicit_dynamics.py
```

---

## Software vs. documentation

| In this public repo | In the distributed package |
|---------------------|----------------------------|
| Documentation, examples, gallery | Compiled / packaged solver |
| Architecture overview, API reference | `import diffsolid as ds` |
| Benchmark figures | Numerical kernels and GPU extensions |

The implementation is **not open source**. Use is subject to the [license](docs/legal/license.md).

---

## Citation

If you use DiffSolid in academic work, please cite:

```bibtex
@software{diffsolid2026,
  title  = {DiffSolid: JAX-native differentiable solid mechanics and phase-field fracture},
  author = {DiffSolid developers},
  year   = {2026},
  url    = {https://github.com/zclsjtu/DiffSolid}
}
```

See also [CITATION.cff](CITATION.cff).

---

## Contact

For academic access, collaboration, or commercial licensing, open an issue or contact
the maintainers via GitHub.
