# Enable GitHub Pages (one-time, ~30 seconds)

Documentation has been deployed to the **`gh-pages`** branch automatically.

1. Open: https://github.com/zclsjtu/DiffSolid/settings/pages
2. Under **Build and deployment → Source**, choose **Deploy from a branch**
3. **Branch:** `gh-pages` · **Folder:** `/ (root)` · click **Save**

After 1–2 minutes the site will be live at:

**https://zclsjtu.github.io/DiffSolid/**

---

## Future updates

```bash
bash /home/zcl/DiffSolid-dev/scripts/publish_public_repo.sh
cd /home/zcl/DiffSolid-public
mkdocs gh-deploy --force
```

Or push to `main` and use the GitHub Actions workflow after switching Source to **GitHub Actions**.
