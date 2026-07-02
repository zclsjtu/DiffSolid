# Third-Party Notices

DiffSolid depends on open-source and vendor-licensed components. This page summarises
the main runtime dependencies. The complete notice file ships inside the wheel
(`THIRD_PARTY_NOTICES.md`).

## Python runtime dependencies

| Component | License | Notes |
|-----------|---------|-------|
| NumPy | BSD-3-Clause | Core array stack |
| SciPy | BSD-3-Clause | Sparse helpers |
| JAX / jaxlib | Apache-2.0 | Differentiable FE assembly |
| meshio | MIT | Mesh I/O |
| h5py | BSD-3-Clause | HDF5 checkpoints |
| matplotlib | PSF-based | Plotting (optional workflows) |

## Optional extras

| Component | License | Notes |
|-----------|---------|-------|
| CuPy | MIT | GPU arrays |
| nvmath-python | NVIDIA | cuDSS Python bindings |
| nvidia-cudss-cu12 | NVIDIA proprietary | Direct sparse solver — user installs under NVIDIA EULA |
| PyVista / VTK | BSD-3-Clause | `[viz]` extra |
| gmsh | GPL-2.0+ | `[mesh]` extra — user installs separately |

## Bundled third-party code (wheel)

| Component | License | Notes |
|-----------|---------|-------|
| AMGCL | MIT | Header library used by the CUDA extension; copyright Denis Demidov |

## Not bundled

The following are **not** redistributed inside DiffSolid wheels:

- NVIDIA CUDA toolkit runtime (pulled via user environment)
- cuDSS, cuBLAS, cuSPARSE (NVIDIA EULA)
- PETSc / petsc4py (optional, user-built)
- pyamgx / NVIDIA AmgX (optional)

## AMGCL attribution

AMGCL is copyright Denis Demidov and licensed under the MIT License. The full license
text is included in the distributed package.

---

For questions about compliance, see [license.md](license.md) or contact the maintainers.
