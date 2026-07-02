# GPU solvers

Linear solver scaling and explicit-dynamics GPU efficiency from benchmark suites in `tests/benchmarks` and large phase-field runs.

<div class="ds-gallery" markdown="1">

<figure class="ds-gallery__item ds-gallery__item--wide">
  <a href="../assets/solvers/gpu_solver_scaling_2d.png" target="_blank" rel="noopener">
    <img src="../assets/solvers/gpu_solver_scaling_2d.png" alt="2D implicit solver scaling across backends" loading="lazy" />
  </a>
  <figcaption>2D implicit benchmarks — solver scaling (all backends)</figcaption>
</figure>

<figure class="ds-gallery__item">
  <a href="../assets/solvers/solver_scaling_amgcl_cudss.png" target="_blank" rel="noopener">
    <img src="../assets/solvers/solver_scaling_amgcl_cudss.png" alt="AMGCL versus cuDSS solver scaling" loading="lazy" />
  </a>
  <figcaption>Solver scaling — AMGCL vs. cuDSS (test suite)</figcaption>
</figure>

<figure class="ds-gallery__item">
  <a href="../assets/solvers/kalthoff_gpu_efficiency.png" target="_blank" rel="noopener">
    <img src="../assets/solvers/kalthoff_gpu_efficiency.png" alt="Kalthoff explicit dynamics GPU efficiency" loading="lazy" />
  </a>
  <figcaption>Kalthoff explicit dynamics — GPU efficiency (compact)</figcaption>
</figure>

<figure class="ds-gallery__item">
  <a href="../assets/solvers/kalthoff_strict_performance.png" target="_blank" rel="noopener">
    <img src="../assets/solvers/kalthoff_strict_performance.png" alt="Strict Kalthoff benchmark performance comparison" loading="lazy" />
  </a>
  <figcaption>Kalthoff strict benchmark — performance summary</figcaption>
</figure>

</div>

[← Gallery overview](index.md) · [Quick Start](../quickstart.md)
