# DiffSolid Installation Guide

DiffSolid is **not on PyPI**. Preview wheels are sent **by email on request**.
This page covers install, JAX GPU setup, and the linear-solver backends you will
actually use.

**Recommended path for most users**

1. Request and install a preview wheel ([§3](#3-get-diffsolid)).
2. Confirm JAX sees a CUDA device ([§4](#4-python-and-jax-gpu)).
3. Install **NVIDIA AmgX** (preferred GPU iterative solver) and **cuDSS** (direct) ([§5](#5-gpu-linear-solvers)).
4. Run the smoke test ([§7](#7-verification)).

AMGCL CUDA is an optional fallback and usually requires a **source** tree to build
([§6](#6-source-licensees-amgcl-cuda-extension)).

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [What You Need](#2-what-you-need)
3. [Get DiffSolid](#3-get-diffsolid)
4. [Python and JAX GPU](#4-python-and-jax-gpu)
5. [GPU Linear Solvers](#5-gpu-linear-solvers)
6. [Source Licensees: AMGCL CUDA Extension](#6-source-licensees-amgcl-cuda-extension)
7. [Verification](#7-verification)
8. [Troubleshooting](#8-troubleshooting)
9. [Dependency Reference](#9-dependency-reference)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA Turing (compute capability ≥ 7.5) | Ampere, Ada, or Hopper (H100/H200) |
| GPU memory | 16 GB | 32 GB+ for large 3D problems |
| NVIDIA driver | ≥ 525 (CUDA 12.x capable) | Latest production driver |
| CUDA toolkit | 12.4+ (needed to **build** AmgX / AMGCL) | Match JAX CUDA 12 wheels |
| OS | Ubuntu 20.04+, RHEL 8+ | Ubuntu 22.04 LTS |
| Host compiler | GCC 11–13 for `nvcc` | GCC 12 or 13 |
| Python | 3.10 | 3.10 or 3.11 |

**Note on GCC 14.** CUDA 12.x officially supports GCC ≤ 13. Building AmgX or the
AMGCL extension with GCC 14 may require `-allow-unsupported-compiler` and is not
guaranteed.

CPU-only installs work for small examples, but large implicit / GPU solver paths
expect an NVIDIA GPU.

---

## 2. What You Need

| Audience | What you receive | Typical next steps |
|----------|------------------|--------------------|
| Preview users | Email wheel (`.whl`) | `pip install` → JAX GPU → AmgX + cuDSS |
| Source licensees | Private source tree | Editable install → optional AMGCL CUDA build → AmgX + cuDSS |

| Backend | Role | How you get it |
|---------|------|----------------|
| **NVIDIA AmgX** | **Preferred** GPU iterative (AMG) for large elasticity / plasticity | Build/install [NVIDIA AmgX](https://github.com/NVIDIA/AMGX); set `AMGX_LIBRARY` — **no pyamgx** |
| **cuDSS** | GPU **direct** sparse solver | `pip install diffsolid[gpu]` or CuPy + nvmath + nvidia-cudss |
| **AMGCL CUDA** | GPU iterative **fallback** when AmgX is unavailable | Prebuilt in some wheels, or build from source ([§6](#6-source-licensees-amgcl-cuda-extension)) |
| UMFPACK / SciPy / JAX Krylov | CPU or JAX-only fallbacks | Bundled with DiffSolid / SciPy / JAX |

---

## 3. Get DiffSolid

### Request a preview wheel {#request-preview-wheel}

Email **[ChenlongZhao@sjtu.edu.cn](mailto:ChenlongZhao@sjtu.edu.cn)** with:

- Your name and affiliation
- Intended use (research, evaluation, collaboration, …)
- Python version and OS

We review each request and reply when we can provide a wheel. Preview wheels may
**include plain Python source** inside the package; we decide per request whether
to distribute them. See also [Download](download.md).

!!! warning "Distribution terms"
    Preview wheels are for your use only. Do not redistribute them.
    DiffSolid is not published on PyPI and is not downloadable from this site.

### Install the wheel

```bash
pip install /path/to/diffsolid-*.whl
python -c "import diffsolid as ds; print('DiffSolid OK', ds.__version__)"
```

Optional extras (after the core wheel is installed):

```bash
pip install "diffsolid[gpu]"    # CuPy, nvmath, cuDSS  — not AmgX
pip install "diffsolid[viz]"    # PyVista, VTK
pip install "diffsolid[mesh]"   # gmsh helpers
```

`[gpu]` does **not** install AmgX. AmgX is a native library you install separately
([§5.1](#51-nvidia-amgx-preferred-gpu-iterative)).

### Developer editable install (source licensees only)

```bash
cd /path/to/DiffSolid-core
pip install -e .
pip install -e ".[gpu]"
python scripts/check_install.py
```

---

## 4. Python and JAX GPU

Use a dedicated virtualenv or conda env. After DiffSolid (and JAX) are installed:

```bash
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0), ...]
```

If only CPU devices appear:

```bash
pip install jax-cuda12-plugin jax-cuda12-pjrt
python -c "import jax; print(jax.devices())"
```

DiffSolid enables `jax_enable_x64` on import. Keep a CUDA-12-compatible JAX stack
(see [§9](#9-dependency-reference)).

---

## 5. GPU Linear Solvers

Facade usage (after `import diffsolid as ds` and building a `Simulation`):

```python
sim.set_linear_solver(ds.solvers.AMGx())                          # preferred iterative
sim.set_linear_solver(ds.solvers.CUDSS(reorder="amd"))            # direct
sim.set_linear_solver(ds.solvers.AMGCL(gpu=True, relaxation="chebyshev"))  # fallback
```

### 5.1 NVIDIA AmgX (preferred GPU iterative)

Recommended for large implicit elasticity and plasticity on NVIDIA GPUs
(H100/H200 class). DiffSolid loads AmgX with **ctypes** from `libamgxsh.so`.
There is **no pyamgx** dependency.

#### Build and install AmgX

Requires a CUDA toolkit and a supported host compiler (GCC ≤ 13 recommended).

```bash
git clone --recursive https://github.com/NVIDIA/AMGX.git
cd AMGX && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/amgx-install"
cmake --build . -j"$(nproc)"
cmake --install .
```

If CMake cannot find CUDA, set `CUDA_TOOLKIT_ROOT_DIR` / `CMAKE_CUDA_COMPILER`
to your toolkit (or use the `nvcc` from `conda install -c conda-forge cuda-nvcc`).

#### Point DiffSolid at the library

Put these in your shell profile (or job script) before running DiffSolid:

```bash
export AMGX_LIBRARY="$HOME/opt/amgx-install/lib/libamgxsh.so"
# equivalent: export AMGX_HOME="$HOME/opt/amgx-install"
export LD_LIBRARY_PATH="$HOME/opt/amgx-install/lib:${LD_LIBRARY_PATH}"
```

Verify the file exists:

```bash
test -f "$AMGX_LIBRARY" && echo "AmgX library OK"
```

#### Use AmgX

```python
sim.set_linear_solver(ds.solvers.AMGx())
```

### 5.2 cuDSS (GPU direct)

Use when you want a robust **direct** sparse solve (plasticity, necking, stiff
systems). Installed via pip — no separate native build beyond the NVIDIA wheels:

```bash
pip install "diffsolid[gpu]"
# or: pip install cupy-cuda12x nvmath-python nvidia-cudss-cu12
```

```python
sim.set_linear_solver(ds.solvers.CUDSS(reorder="amd"))
```

### 5.3 PETSc (optional)

Not required for the default GPU path. Via conda-forge:

```bash
conda install -c conda-forge petsc=3.23 petsc4py openmpi -y
```

### 5.4 AMGCL CUDA (fallback)

Use when AmgX is not installed. Prefer Chebyshev relaxation for 3D elasticity /
plasticity:

```python
sim.set_linear_solver(
    ds.solvers.AMGCL(gpu=True, relaxation="chebyshev", solver_type="fgmres")
)
```

Wheel recipients: try the smoke test first. If `AMGCL_CUDA_AVAILABLE` is false,
either use AmgX/cuDSS or obtain a source build ([§6](#6-source-licensees-amgcl-cuda-extension)).

---

## 6. Source Licensees: AMGCL CUDA Extension

Skip this section if you only have a preview wheel and AmgX/cuDSS already work.

### Package layout (source tree)

```
DiffSolid-core/
├── diffsolid/                  # Python package
├── ext/diffsolid_amgcl_cuda/   # CUDA extension (rebuild per machine)
├── third_party/amgcl/          # AMGCL headers (header-only)
├── pyproject.toml
├── requirements.txt
├── INSTALL.md
└── scripts/check_install.py
```

The compiled `.so` under `ext/diffsolid_amgcl_cuda/build/` is **machine-specific** —
do not copy it between servers.

### AMGCL headers

```bash
git clone --depth 1 https://github.com/ddemidov/amgcl.git \
    /path/to/DiffSolid-core/third_party/amgcl
test -f /path/to/DiffSolid-core/third_party/amgcl/amgcl/amg.hpp && echo OK
```

### Build prerequisites

| Dependency | Source | Purpose |
|------------|--------|---------|
| `nvcc` | Conda `cuda-nvcc` or system CUDA | CUDA compilation |
| `libboost-headers` | Conda | AMGCL JSON parameter parsing |
| `pybind11` | pip | Python/C++ bindings |
| `cmake` ≥ 3.18 | pip or conda | Build system |
| `CUDA::cusparse` | CUDA toolkit | Sparse linear algebra |

```bash
conda install -c conda-forge cuda-nvcc libboost-headers cmake -y
pip install pybind11
```

### Build

```bash
conda activate diffsolid   # or your env name
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

Edit `CMakeLists.txt` if your GPU is not covered by the default architectures:

| GPU family | `sm` version |
|------------|--------------|
| Turing (RTX 20xx, T4) | 75 |
| Ampere (RTX 30xx, A100) | 80, 86 |
| Ada Lovelace (RTX 40xx, RTX 6000 Ada) | 89 |
| Hopper (H100, H200) | 90 |

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

## 7. Verification

### Import check

```bash
python -c "import diffsolid as ds; print('DiffSolid OK', ds.__version__)"
# source trees: python scripts/check_install.py
```

### Linear solver smoke test

Requires `AMGX_LIBRARY` (and `LD_LIBRARY_PATH`) for AmgX, and `[gpu]` extras for cuDSS.

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
    from diffsolid.solvers.linear.amgx import amgx_solve
    check("AmgX", amgx_solve(A, b, {"tolerance": 1e-10}))
except Exception as e:
    print(f"  [SKIP/ERROR] AmgX (set AMGX_LIBRARY): {e}")

try:
    from diffsolid.solvers.linear.cudss import cudss_solve
    check("cuDSS", cudss_solve(A, b))
except Exception as e:
    print(f"  [SKIP/ERROR] cuDSS (pip install diffsolid[gpu]): {e}")

try:
    from diffsolid.solvers.linear.amgcl_gpu import amgcl_solve, AMGCL_CUDA_AVAILABLE
    if AMGCL_CUDA_AVAILABLE:
        check("AMGCL CUDA", amgcl_solve(A, b, tol=1e-10))
    else:
        print("  [SKIP] AMGCL CUDA extension not built / not on PYTHONPATH")
except Exception as e:
    print(f"  [SKIP/ERROR] AMGCL: {e}")

print("=== Done ===")
EOF
```

A healthy GPU install should show **AmgX OK** and ideally **cuDSS OK**. AMGCL may
remain SKIPPED without a source build — that is fine if AmgX works.

---

## 8. Troubleshooting

### AmgX: `ImportError` / cannot find `libamgxsh.so`

```bash
# Confirm the shared library
ls -l "$AMGX_LIBRARY"
# or
ls -l "$AMGX_HOME/lib/libamgxsh.so"

# Ensure the dynamic linker can resolve dependencies
export LD_LIBRARY_PATH="$(dirname "$AMGX_LIBRARY"):${LD_LIBRARY_PATH}"
```

Rebuild AmgX if the library is missing. DiffSolid does **not** use pyamgx.

### AmgX: CUDA / driver mismatch at load time

Match the AmgX build CUDA toolkit to a driver that supports that toolkit.
Re-run the smoke test after fixing `LD_LIBRARY_PATH`.

### `AMGCL_CUDA_AVAILABLE = False`

The extension `.so` is missing or not importable. Prefer AmgX, or rebuild
[§6](#6-source-licensees-amgcl-cuda-extension) and confirm:

```bash
ls ext/diffsolid_amgcl_cuda/build/diffsolid_amgcl_cuda*.so
```

### CuPy import failure

Match CuPy to your CUDA major version:

```bash
pip install cupy-cuda12x   # CUDA 12.x
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

DiffSolid enables `jax_enable_x64` on import. Ensure your JAX build supports
64-bit floating point on GPU.

---

## 9. Dependency Reference

### Core Python packages (pip)

| Package | Typical version | Role |
|---------|-----------------|------|
| `jax`, `jaxlib` | 0.6.x | Core AD and JIT |
| `jax-cuda12-plugin`, `jax-cuda12-pjrt` | 0.6.x | GPU backend |
| `numpy` | 2.2.x | Host numerics |
| `scipy` | 1.15.x | Sparse linear algebra |
| `meshio` | 5.3.x | Mesh I/O |
| `gmsh` | 4.14.x | Mesh generation (`[mesh]`) |
| `h5py` | 3.15.x | HDF5 / XDMF |
| `matplotlib` | 3.10.x | Plotting |

### GPU solver packages / libraries

| Component | Role | Install |
|-----------|------|---------|
| **NVIDIA AmgX** (`libamgxsh.so`) | Preferred GPU iterative AMG | Build from [NVIDIA/AMGX](https://github.com/NVIDIA/AMGX); set `AMGX_LIBRARY` |
| `cupy-cuda12x` | GPU arrays; cuDSS / AMGCL support | `diffsolid[gpu]` |
| `nvmath-python`, `nvidia-cudss-cu12` | cuDSS direct solver | `diffsolid[gpu]` |
| `pybind11` + CUDA toolkit | AMGCL CUDA extension build | Source licensees only |

### Third-party source (source trees / some wheels)

| Component | Location / notes |
|-----------|------------------|
| AMGCL headers | `third_party/amgcl/` |
| AMGCL CUDA extension | `ext/diffsolid_amgcl_cuda/` |
| NVIDIA AmgX | **Not** redistributed — user-installed native library |

Pinned versions for a given release are listed in the wheel metadata /
`requirements.txt` of the package you received.
