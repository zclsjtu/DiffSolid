# Solver efficiency

Linear-solver scaling on **2D implicit** benchmarks — **DiffSolid on NVIDIA H200**
(AmgX leading) vs. **FEniCSx on 64-core CPU** — plus **GPU throughput** on large
explicit Kalthoff-type phase-field runs.

<div class="ds-gallery ds-gallery--story" markdown="1">

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">1</span>
  <a href="../assets/stories/solver-efficiency/01_implicit_2d_all_solvers.png" target="_blank" rel="noopener">
    <img src="../assets/stories/solver-efficiency/01_implicit_2d_all_solvers.png" alt="2D implicit elastic scaling — DiffSolid AmgX on H200 vs FEniCSx Hypre on 64 CPU ranks" loading="lazy" />
  </a>
  <figcaption><strong>Implicit 2D scaling.</strong> Linear elasticity with full operator update — <strong>DiffSolid/AmgX</strong> on <strong>NVIDIA H200</strong> leads DiffSolid/cuDSS and DiffSolid/AMGCL; matched N=50 timing vs. <strong>FEniCSx/Hypre</strong> on <strong>64 CPU ranks</strong>.</figcaption>
</figure>

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">2</span>
  <a href="../assets/stories/solver-efficiency/02_gpu_efficiency.png" target="_blank" rel="noopener">
    <img src="../assets/stories/solver-efficiency/02_gpu_efficiency.png" alt="Explicit GPU efficiency for Kalthoff benchmark cases n100k through n2000k" loading="lazy" />
  </a>
  <figcaption><strong>Explicit GPU efficiency.</strong> Kalthoff strict benchmark — throughput, normalized GPU efficiency, step time, and peak VRAM vs. problem size (parabolic vs. inertial phase-field; runtime excludes JIT).</figcaption>
</figure>

</div>

---

**Setup.** 2D implicit study: plane-strain elasticity; DiffSolid GPU runs on
**NVIDIA H200** (AmgX preferred; cuDSS and AMGCL also shown), FEniCSx baselines on
**MPI×64 CPU**. GPU panel: explicit Kalthoff dynamics on H200; cases `n100k … n2000k`.

[← Gallery overview](index.md)
