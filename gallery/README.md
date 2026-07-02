# DiffSolid Gallery

Curated benchmark figures are published on the **documentation site**:

**https://zclsjtu.github.io/DiffSolid/gallery/**

## Cases

- [Dynamic fracture](../docs/gallery/dynamic-fracture.md)
- [Quasi-static phase-field fracture](../docs/gallery/quasi-static-pf-fracture.md)
- [Volumetric locking](../docs/gallery/volumetric-locking.md)
- [Solver efficiency](../docs/gallery/solver-efficiency.md)

## Assets

Source PNGs live under `docs/gallery/assets/`. To refresh from local benchmark runs:

```bash
python ../DiffSolid-dev/scripts/sync_gallery_assets.py
python ../DiffSolid-dev/scripts/build_gallery_locking_story.py
```

## Note

Gallery assets are for **visualisation and portfolio purposes** only. They do not
include solver source code or full simulation result archives.
