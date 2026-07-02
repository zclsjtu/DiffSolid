# Dynamic fracture

A end-to-end case study on the same phase-field stack: quasi-static branching on an elliptic
plate, dynamic Kalthoff-type impact with dense mesh closure, staggered **S3** coupling
sensitivity, and explicit-dynamics **GPU efficiency** on the strict Kalthoff benchmark.

<div class="ds-gallery ds-gallery--story" markdown="1">

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">1</span>
  <a href="../assets/stories/kalthoff-borden-gpu/01_damage_evolution.png" target="_blank" rel="noopener">
    <img src="../assets/stories/kalthoff-borden-gpu/01_damage_evolution.png" alt="Horizontal damage evolution on Borden elliptic plate, Miehe–Amor benchmark" loading="lazy" />
  </a>
  <figcaption><strong>Phase-field branching.</strong> Miehe–Amor elliptic plate (Borden-type benchmark) — real damage evolution in a horizontal panel (<code>n ≈ 1000</code>).</figcaption>
</figure>

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">2</span>
  <a href="../assets/stories/kalthoff-borden-gpu/02_crack_morphology.png" target="_blank" rel="noopener">
    <img src="../assets/stories/kalthoff-borden-gpu/02_crack_morphology.png" alt="Selected Kalthoff crack morphologies on dense mesh" loading="lazy" />
  </a>
  <figcaption><strong>Kalthoff morphology.</strong> Selected crack paths from the dense mesh paper-closure study — morphology comparison across cases.</figcaption>
</figure>

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">3</span>
  <a href="../assets/stories/kalthoff-borden-gpu/03_crack_velocity_s3.png" target="_blank" rel="noopener">
    <img src="../assets/stories/kalthoff-borden-gpu/03_crack_velocity_s3.png" alt="Crack tip velocity versus S3 staggered coupling parameters" loading="lazy" />
  </a>
  <figcaption><strong>S3 staggered coupling.</strong> Crack-tip velocity response to staggered strategy <strong>S3</strong> parameters on the dense Kalthoff closure mesh.</figcaption>
</figure>

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">4</span>
  <a href="../assets/stories/kalthoff-borden-gpu/04_gpu_efficiency.png" target="_blank" rel="noopener">
    <img src="../assets/stories/kalthoff-borden-gpu/04_gpu_efficiency.png" alt="Explicit GPU efficiency for Kalthoff strict benchmark across case indices n100 to n2000" loading="lazy" />
  </a>
  <figcaption><strong>GPU scaling.</strong> Explicit dynamics GPU efficiency on the strict Kalthoff benchmark — benchmark cases <code>n100 … n2000</code>.</figcaption>
</figure>

</div>

---

**Setup.** Elliptic phase-field fracture with cuDSS linear solves; dynamic Kalthoff-type impact on
refined meshes; staggered S1/S3/S4 strategy matrix documented in
[Theory: phase-field fracture](../theory/phase-field-fracture.md).

[← Gallery overview](index.md)
