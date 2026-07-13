# DiffSolid Installation Guide

This document describes how to install DiffSolid, configure JAX on NVIDIA GPUs, build
the optional AMGCL CUDA extension, and verify linear solver backends.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Package Layout](#2-package-layout)
3. [Python Environment](#3-python-environment)
4. [Installing DiffSolid](#4-installing-diffsolid)
5. [AMGCL Header Dependency](#5-amgcl-header-dependency)
6. [Building the AMGCL CUDA Extension](#6-building-the-amgcl-cuda-extension)
7. [Optional Backends](#7-optional-backends)
8. [Verification](#8-verification)
9. [Troubleshooting](#9-troubleshooting)
10. [Dependency Reference](#10-dependency-reference)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA Turing (compute capability ≥ 7.5) | Ampere, Ada, or Hopper class |
| GPU memory | 16 GB | 32 GB or more for large 3D problems |
| NVIDIA driver | ≥ 525 (CUDA 12.x capable) | Latest production driver |
| CUDA toolkit | 12.4+ | Match JAX CUDA 12 wheels |
| OS | Ubuntu 20.04+, RHEL 8+ | Ubuntu 22.04 LTS |
| Host compiler | GCC 11–13 for `nvcc` | GCC 12 or 13 |
| Python | 3.10 | 3.10 or 3.11 |

**Note on GCC 14.** CUDA 12.x officially supports GCC ≤ 13. If only GCC 14 is available,
the provided build script passes `-allow-unsupported-compiler`; compilation usually
succeeds but is not guaranteed on all platforms.

---

## 2. Package Layout

```
DiffSolid/
├── diffsolid/                  # Python package
├── ext/diffsolid_amgcl_cuda/   # CUDA extension source (must be rebuilt per machine)
├── third_party/amgcl/          # AMGCL headers (header-only)
├── pyproject.toml
├── requirements.txt
├── INSTALL.md
├── API.md
└── scripts/check_install.py
```

The compiled extension artifact under `ext/diffsolid_amgcl_cuda/build/` is
**machine-specific** and must not be copied between servers. Rebuild on each target
system.

Full conda environment specifications are available to source licensees on request.

---

## 3. Python Environment

### Option A — Minimal pip install (CPU or GPU JAX)

```bash
cd /path/to/DiffSolid-core
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt   # pins JAX, CuPy, cuDSS, etc.
```

### Option B — Conda environment (recommended for GPU HPC)

```bash
cd /path/to/DiffSolid-core-dev
conda env create -f environment.yml
conda activate diffsolid    # or the name defined in environment.yml
cd ../DiffSolid
pip install -e .
```

### JAX GPU support

After installing JAX, confirm device visibility:

```bash
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0), ...]
```

If only CPU devices appear:

```bash
pip install jax-cuda12-plugin jax-cuda12-pjrt
```

---

## 4. Installing DiffSolid

DiffSolid is **not** published on PyPI. Preview wheels are distributed **by email
on request** — not from this documentation site.

### Request a preview wheel

Email **[ChenlongZhao@sjtu.edu.cn](mailto:ChenlongZhao@sjtu.edu.cn)** with:

- Your name and affiliation
- Intended use (research, evaluation, collaboration, …)
- Python version and OS

We review each request and reply when we can provide a wheel. Preview wheels may
**include plain Python source** inside the package; we decide per request whether
to distribute them.

> Compiled wheels without source may be offered later when the API stabilises.
> Do not redistribute wheels you receive.

### Install

After you receive a wheel by email:

```bash
pip install /path/to/diffsolid-*.whl
python -c "import diffsolid as ds; print('DiffSolid OK')"
```

Optional extras (after the core wheel is installed):

```bash
pip install diffsolid[gpu]    # CuPy, nvmath, cuDSS
pip install diffsolid[viz]    # PyVista, VTK
pip install diffsolid[mesh]   # gmsh mesh generation helpers
```

### Developer editable install (source licensees only)

If you have been granted access to the private source repository:

```bash
cd /path/to/DiffSolid-core
pip install -e .
python scripts/check_install.py
```

---

---

## 5. AMGCL Header Dependency

AMGCL is a header-only C++ library used by the CUDA extension. Clone it into
`third_party/amgcl/`:

```bash
git clone --depth 1 https://github.com/ddemidov/amgcl.git \
    /path/to/DiffSolid-core/third_party/amgcl
```

Verify:

```bash
test -f /path/to/DiffSolid-core/third_party/amgcl/amgcl/amg.hpp && echo OK
```

If GitHub is unreachable, copy the `third_party/amgcl/` directory from an existing
installation (~2 MB, platform-independent).

---

## 6. Building the AMGCL CUDA Extension

### Prerequisites

| Dependency | Source | Purpose |
|------------|--------|---------|
| `nvcc` | Conda `cuda-nvcc` or system CUDA | CUDA compilation |
| `libboost-headers` | Conda | AMGCL JSON parameter parsing |
| `pybind11` | pip | Python/C++ bindings |
| `cmake` ≥ 3.18 | pip or conda | Build system |
| `CUDA::cusparse` | CUDA toolkit | Sparse linear algebra |

Install build dependencies:

```bash
conda install -c conda-forge cuda-nvcc libboost-headers cmake -y
pip install pybind11
```

### Build

```bash
conda activate diffsolid
bash /path/to/DiffSolid-core/ext/diffsolid_amgcl_cuda/build_diffsolid_amgcl_cuda.sh
```

Successful output includes a path such as:

```
Built: .../ext/diffsolid_amgcl_cuda/build/diffsolid_amgcl_cuda.cpython-310-x86_64-linux-gnu.so
```

Quick import test:

```bash
PYTHONPATH=/path/to/DiffSolid-core/ext/diffsolid_amgcl_cuda/build \
    python -c "import diffsolid_amgcl_cuda; print('OK')"
```

### CUDA architecture flags

Edit `CMakeLists.txt` if the target GPU is not covered by default architectures:

| GPU family | `sm` version |
|------------|--------------|
| Turing (RTX 20xx, T4) | 75 |
| Ampere (RTX 30xx, A100) | 80, 86 |
| Ada Lovelace (RTX 40xx, RTX 6000 Ada) | 89 |
| Hopper (H100, H200) | 90 |

Example:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "...")
```

### Common build failures

**Unsupported host compiler**

```bash
export CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
bash ext/diffsolid_amgcl_cuda/build_diffsolid_amgcl_cuda.sh
```

**Boost not found**

```bash
conda install -c conda-forge libboost-headers libboost -y
```

**No kernel image available at runtime**

Rebuild with the correct `CMAKE_CUDA_ARCHITECTURES` for your GPU.

---

## 7. Optional Backends

### cuDSS (direct GPU sparse solver)

Required for plasticity, necking, and other problems where a robust direct solve is
preferred:

```bash
pip install cupy-cuda12x nvmath-python nvidia-cudss-cu12
```

Facade usage:

```python
sim.set_linear_solver(ds.solvers.CUDSS(reorder="amd"))
```

### PETSc (optional)

PETSc is not installed via pip. Use conda-forge:

```bash
conda install -c conda-forge petsc=3.23 petsc4py openmpi -y
```

### NVIDIA AmgX / pyamgx (optional)

Only required if you explicitly use the AmgX backend. See the
[pyamgx documentation](https://pyamgx.readthedocs.io/). AMGCL, cuDSS, and JAX iterative
solvers do not depend on AmgX.

---

## 8. Verification

### Package import

```bash
python scripts/check_install.py
```

### Linear solver smoke test

```bash
python - <<'EOF'
import numpy as np
import scipy.sparse
import jax.numpy as jnp

A = scipy.sparse.csr_matrix(np.array([
    [4., -1.,  0.,  0.],
    [-1., 4., -1.,  0.],
    [0., -1.,  4., -1.],
    [0.,  0., -1.,  4.],
], dtype=np.float64))
b = jnp.array(A @ np.ones(4))

def check(name, x):
    err = float(np.linalg.norm(np.array(x) - 1.0))
    print(f"  [{'OK' if err < 1e-6 else 'FAIL'}] {name}: ||x-1||={err:.2e}")

print("=== Linear solver verification ===")

try:
    from diffsolid.solvers.linear.amgcl_gpu import amgcl_solve, AMGCL_CUDA_AVAILABLE
    if AMGCL_CUDA_AVAILABLE:
        check("AMGCL CUDA", amgcl_solve(A, b, tol=1e-10))
    else:
        print("  [SKIP] AMGCL CUDA extension not built")
except Exception as e:
    print(f"  [ERROR] AMGCL: {e}")

try:
    from diffsolid.solvers.linear.cudss import cudss_solve
    check("cuDSS", cudss_solve(A, b))
except Exception as e:
    print(f"  [ERROR] cuDSS: {e}")

print("=== Done ===")
EOF
```

### Regression tests (development workspace)

```bash
cd ../DiffSolid-dev
pytest tests/test_fracture_dynamics_s3_scan.py -q
```

---

## 9. Troubleshooting

### `AMGCL_CUDA_AVAILABLE = False`

The extension `.so` is missing or not on `PYTHONPATH`. Rebuild Section 6 and confirm:

```bash
ls ext/diffsolid_amgcl_cuda/build/diffsolid_amgcl_cuda*.so
```

### CuPy import failure

Match CuPy to your CUDA major version:

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
```

### JAX sees only CPU

```bash
pip install jax-cuda12-plugin jax-cuda12-pjrt
python -c "import jax; print(jax.devices())"
```

### cuDSS / nvmath errors

```bash
pip install nvmath-python nvidia-cudss-cu12
```

### Double-precision

DiffSolid enables `jax_enable_x64` on import. Ensure your JAX build supports 64-bit
floating point on GPU.

---

## 10. Dependency Reference

### Core Python packages (pip)

| Package | Typical version | Role |
|---------|-----------------|------|
| `jax`, `jaxlib` | 0.6.x | Core AD and JIT |
| `jax-cuda12-plugin` | 0.6.x | GPU backend |
| `numpy` | 2.2.x | Host numerics |
| `scipy` | 1.15.x | Sparse linear algebra |
| `meshio` | 5.3.x | Mesh I/O |
| `gmsh` | 4.14.x | Mesh generation |
| `h5py` | 3.15.x | HDF5 / XDMF |
| `matplotlib` | 3.10.x | Plotting |

### GPU solver packages

| Package | Role |
|---------|------|
| `cupy-cuda12x` | GPU arrays; AMGCL zero-copy path |
| `nvmath-python`, `nvidia-cudss-cu12` | cuDSS direct solver |
| `pybind11` | AMGCL CUDA extension build |

### Third-party source

| Component | Location |
|-----------|----------|
| AMGCL headers | `third_party/amgcl/` |
| CUDA extension | `ext/diffsolid_amgcl_cuda/` |

For pinned versions used in production, see `requirements.txt` and
`../DiffSolid-dev/environment.yml`.
