# Quasi-static phase-field fracture

Cohesive phase-field fracture under **quasi-static** loading: an **L-panel** with Miehe
spectral–deviatoric split vs. **hybrid** stress decomposition (same cohesive degradation,
**VI-Newton** solver), then **periodic** unit cells with one or two holes — mechanics and
phase field both on **PBC**.

<div class="ds-gallery ds-gallery--story" markdown="1">

<div class="ds-gallery__pair ds-gallery__item--step">
  <span class="ds-gallery__step">1</span>
  <figure class="ds-gallery__item">
    <a href="../assets/stories/quasi-static-pf-fracture/01_lpanel_standard_damage.png" target="_blank" rel="noopener">
      <img src="../assets/stories/quasi-static-pf-fracture/01_lpanel_standard_damage.png" alt="L-panel final damage, Miehe stress decomposition, VI-Newton" loading="lazy" />
    </a>
    <figcaption><strong>L-panel · Miehe.</strong> Cohesive degradation with Miehe spectral–deviatoric split — staggered <strong>VI-Newton</strong>.</figcaption>
  </figure>
  <figure class="ds-gallery__item">
    <a href="../assets/stories/quasi-static-pf-fracture/02_lpanel_hybrid_damage.png" target="_blank" rel="noopener">
      <img src="../assets/stories/quasi-static-pf-fracture/02_lpanel_hybrid_damage.png" alt="L-panel final damage, hybrid stress decomposition, VI-Newton" loading="lazy" />
    </a>
    <figcaption><strong>L-panel · hybrid.</strong> Same cohesive setup and VI-Newton solver — <strong>hybrid</strong> stress decomposition instead of Miehe.</figcaption>
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

**Setup.** Cohesive phase-field degradation with Miehe strain split; implicit staggered solves (AMGCL /
hybrid CG–SPAI0). Periodic examples use consistent PBC on displacement and damage fields.

[← Gallery overview](index.md) · [Theory: phase-field fracture](../theory/phase-field-fracture.md)
