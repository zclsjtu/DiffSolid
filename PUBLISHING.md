# Publishing DiffSolid

This directory is the **public** repository (`DiffSolid`). It contains documentation,
examples, and gallery placeholders only — no solver source code.

## One-time setup

1. Edit [`site.yaml`](site.yaml) if your GitHub username is not `zcl`.
2. From the workspace root:

```bash
bash DiffSolid-dev/scripts/publish_public_repo.sh
```

3. Initialize and push (first time):

```bash
cd DiffSolid-public
git init
git branch -M main
git add .
git commit -m "Initial public documentation site"
git remote add origin https://github.com/zclsjtu/DiffSolid.git
git push -u origin main
```

4. Enable GitHub Pages: repository **Settings → Pages → Build and deployment → GitHub Actions**.

## Updating documentation

After editing user docs in the private `DiffSolid/` package:

```bash
bash DiffSolid-dev/scripts/publish_public_repo.sh
cd DiffSolid-public
git add docs/ README.md mkdocs.yml
git commit -m "docs: sync from release package"
git push
```

## Preview wheel (private repo)

Preview wheels are built from `DiffSolid-core` (private), not this repository.

```bash
bash DiffSolid-dev/scripts/release_preview_wheel.sh v0.1.0-dev.1
```

Tag and push on the **private** repo to trigger GitHub Actions release:

```bash
cd ../DiffSolid
git tag v0.1.0-dev.1
git remote add origin https://github.com/zclsjtu/DiffSolid-core.git   # once
git push -u origin main
git push origin v0.1.0-dev.1
```

> Preview wheels contain plain Python source inside the package. Restrict distribution
> to trusted testers until compiled wheels are published.

## Repository map

| Local path | GitHub | Visibility |
|------------|--------|------------|
| `DiffSolid-public/` | `DiffSolid` | Public |
| `DiffSolid/` | `DiffSolid-core` | Private |
| `DiffSolid-dev/` | (same private remote or submodule) | Private |
