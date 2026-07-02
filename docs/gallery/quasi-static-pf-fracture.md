# Quasi-static phase-field fracture

Cohesive phase-field fracture under **quasi-static** loading: an **L-panel** with Miehe
spectral–deviatoric split and JPFF degradation (standard vs. **hybrid** linear preconditioner),
then **periodic** unit cells with one or two holes — mechanics and phase field both on **PBC**.

<div class="ds-gallery ds-gallery--story" markdown="1">

<div class="ds-gallery__pair ds-gallery__item--step">
  <span class="ds-gallery__step">1</span>
  <figure class="ds-gallery__item">
    <a href="../assets/stories/quasi-static-pf-fracture/01_lpanel_standard_damage.png" target="_blank" rel="noopener">
      <img src="../assets/stories/quasi-static-pf-fracture/01_lpanel_standard_damage.png" alt="L-panel final damage, cohesive Miehe split, standard solver" loading="lazy" />
    </a>
    <figcaption><strong>L-panel · standard.</strong> Cohesive crack with Miehe decomposition — AMGCL, no hybrid preconditioner.</figcaption>
  </figure>
  <figure class="ds-gallery__item">
    <a href="../assets/stories/quasi-static-pf-fracture/02_lpanel_hybrid_damage.png" target="_blank" rel="noopener">
      <img src="../assets/stories/quasi-static-pf-fracture/02_lpanel_hybrid_damage.png" alt="L-panel final damage, cohesive Miehe split, hybrid AMGCL-CG SPAI0" loading="lazy" />
    </a>
    <figcaption><strong>L-panel · hybrid.</strong> Same setup with hybrid AMGCL + CG + SPAI0 (<code>tol = 10⁻¹²</code>).</figcaption>
  </figure>
</div>

<div class="ds-gallery__pair ds-gallery__item--step">
  <span class="ds-gallery__step">2</span>
  <figure class="ds-gallery__item">
    <a href="../assets/stories/quasi-static-pf-fracture/03_pbc_one_hole.png" target="_blank" rel="noopener">
      <img src="../assets/stories/quasi-static-pf-fracture/03_pbc_one_hole.png" alt="Periodic unit cell with one hole, PBC on mechanics and phase field" loading="lazy" />
    </a>
    <figcaption><strong>PBC · one hole.</strong> Periodic cell — quasi-static mechanics and phase field on matching PBC.</figcaption>
  </figure>
  <figure class="ds-gallery__item">
    <a href="../assets/stories/quasi-static-pf-fracture/04_pbc_two_holes.png" target="_blank" rel="noopener">
      <img src="../assets/stories/quasi-static-pf-fracture/04_pbc_two_holes.png" alt="Periodic unit cell with two holes, PBC on mechanics and phase field" loading="lazy" />
    </a>
    <figcaption><strong>PBC · two holes.</strong> Same PBC coupling on a two-hole periodic unit cell.</figcaption>
  </figure>
</div>

</div>

---

**Setup.** JPFF cohesive phase field with Miehe strain split; implicit staggered solves (AMGCL /
hybrid CG–SPAI0). Periodic examples use consistent PBC on displacement and damage fields.

[← Gallery overview](index.md) · [Theory: phase-field fracture](../theory/phase-field-fracture.md)
