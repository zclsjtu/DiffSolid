# Solver efficiency

Linear-solver scaling on **2D implicit** benchmarks — **DiffSolid on NVIDIA H200** vs.
**FEniCSx on 64-core CPU** — plus **GPU throughput** on large explicit Kalthoff-type
phase-field runs.

<div class="ds-gallery ds-gallery--story" markdown="1">

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">1</span>
  <a href="../assets/stories/solver-efficiency/01_implicit_2d_all_solvers.png" target="_blank" rel="noopener">
    <img src="../assets/stories/solver-efficiency/01_implicit_2d_all_solvers.png" alt="2D implicit benchmark — DiffSolid H200 GPU vs FEniCSx 64-core CPU" loading="lazy" />
  </a>
  <figcaption><strong>Implicit 2D scaling.</strong> 2D QUAD4 plane-strain linear solve (step 2) — <strong>DiffSolid</strong> on <strong>NVIDIA H200</strong> (AMGCL-CG, cuDSS) vs. <strong>FEniCSx</strong> on <strong>64-core CPU</strong> (hypre BoomerAMG, MUMPS direct).</figcaption>
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

**Setup.** 2D implicit study: staggered nonlinear solves with configurable sparse linear backends.
GPU panel: explicit dynamics on triangular meshes from benchmark cases <code>n100k … n2000k</code>.

[← Gallery overview](index.md)
