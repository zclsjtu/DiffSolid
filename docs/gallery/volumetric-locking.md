# Volumetric locking

Near-incompressible benchmarks where standard low-order elements lock, and **F-bar**,
**F-bar-Patch**, and **EAS** recover the expected response.

<div class="ds-gallery ds-gallery--story" markdown="1">

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">1</span>
  <a href="../assets/stories/volumetric-locking/01_cook_convergence.png" target="_blank" rel="noopener">
    <img src="../assets/stories/volumetric-locking/01_cook_convergence.png" alt="Cook membrane mesh convergence — standard vs F-bar vs EAS" loading="lazy" />
  </a>
  <figcaption><strong>Cook's membrane.</strong> Mesh convergence at the top-right corner — standard TRI3/QUAD4 lock (low displacement); F-bar, F-bar-Patch, and EAS track the reference trend.</figcaption>
</figure>

<figure class="ds-gallery__item ds-gallery__item--wide ds-gallery__item--step">
  <span class="ds-gallery__step">2</span>
  <a href="../assets/stories/volumetric-locking/02_axi_necking_elements.png" target="_blank" rel="noopener">
    <img src="../assets/stories/volumetric-locking/02_axi_necking_elements.png" alt="Axisymmetric finite-strain necking — element comparison with deformed fields and load curves" loading="lazy" />
  </a>
  <figcaption><strong>Axisymmetric necking.</strong> Finite-strain J<sub>2</sub> plasticity — deformed <code>|u|</code> for QUAD4 F-bar, TRI3 F-bar-Patch, and QUAD4 EAS (top), with reaction–displacement curves on one panel (bottom).</figcaption>
</figure>

</div>

---

**Setup.** Cook's membrane: linear elasticity, μ = 40, κ = 500/3 (de Souza Neto et al., 2005).
Axisymmetric necking: Fe–F<sub>p</sub> J<sub>2</sub> finite-strain plasticity on a cylindrical bar.

[← Gallery overview](index.md) · [Theory: quasi-static mechanics](../theory/formulations.md)
