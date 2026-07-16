# DiffSolid Preview v0.1.0

**DiffSolid** is a JAX-native differentiable finite-element framework for
nonlinear solid mechanics, phase-field fracture, GPU computing, and
gradient-based inverse design.

## What's in this preview

- Quasi-static and explicit dynamics on JAX (CPU/GPU)
- Phase-field fracture (AT1/AT2, cohesive; staggered S1–S7)
- GPU sparse solvers (**NVIDIA AmgX** preferred; cuDSS, AMGCL CUDA)
- Gradient-based inverse / calibration workflows
- Public docs, Quick Start, theory chapters, and gallery benchmarks

## Links

- Documentation: https://zclsjtu.github.io/DiffSolid/
- Repository: https://github.com/zclsjtu/DiffSolid
- Quick Start: https://zclsjtu.github.io/DiffSolid/quickstart/
- Gallery: https://zclsjtu.github.io/DiffSolid/gallery/

## Distribution

This repository publishes **documentation and examples only**. The numerical
kernel is proprietary and **not on PyPI**. Request a preview wheel by email:

**ChenlongZhao@sjtu.edu.cn**

Preview wheels may include plain Python source; redistribution is not permitted.

## Citation

```bibtex
@software{zhao_diffsolid_2026,
  author  = {Zhao, Chenlong},
  title   = {DiffSolid: JAX-native differentiable finite elements
             for solid mechanics and phase-field fracture},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/zclsjtu/DiffSolid}
}
```
