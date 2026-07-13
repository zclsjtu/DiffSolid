#!/usr/bin/env python3
"""Inject site.yaml values into generated docs (install / download sections)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_YAML = ROOT / "site.yaml"
DOCS = ROOT / "docs"


def _parse_block(key: str) -> dict[str, str]:
    if not SITE_YAML.is_file():
        return {}
    lines = SITE_YAML.read_text(encoding="utf-8").splitlines()
    in_block = False
    out: dict[str, str] = {}
    for line in lines:
        if line.strip() == f"{key}:":
            in_block = True
            continue
        if in_block:
            if line and not line.startswith(" "):
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            k, _, v = stripped.partition(":")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _contact_email(dist: dict[str, str]) -> str:
    return dist.get("contact_email", "ChenlongZhao@sjtu.edu.cn")


def render_download_md(dist: dict[str, str]) -> str:
    contact = _contact_email(dist)

    return f"""---
title: Download wheel
---

# Download wheel

Preview wheels are **not on PyPI** and **not publicly downloadable** from this site.

<div class="ds-download__portal ds-download__portal--pending" markdown="1">

<p class="ds-download__pending">Email <a href="mailto:{contact}">{contact}</a> to request a preview wheel. We reply by email when distribution is appropriate — there is no automatic download here.</p>

</div>

## How it works

1. **Email** **[{contact}](mailto:{contact})** with your name, affiliation, intended use, Python version, and OS.
2. **We review** each request and decide whether to send a preview wheel (preview builds may include plain Python source).
3. **Install** from the wheel we attach or link in our reply:

```bash
pip install /path/to/diffsolid-*.whl
python -c "import diffsolid as ds; print('DiffSolid OK')"
```

See [Install guide](install.md) for GPU setup and optional extras.

!!! warning "Distribution terms"
    Preview wheels are for your use only. Do not redistribute wheels or share them publicly.

[← Install guide](install.md#request-preview-wheel)
"""


def patch_install_wheel_section(install_text: str, dist: dict[str, str]) -> str:
    contact = _contact_email(dist)
    form_url = dist.get("request_form_url", "").strip()
    form_section = f"\nOr use the [request form]({form_url}).\n" if form_url else ""

    replacement = f"""## 4. Installing DiffSolid

DiffSolid is **not** published on PyPI. Preview wheels are distributed **by email
on request** — not from this documentation site.

### Request a preview wheel

Email **[{contact}](mailto:{contact})** with:

- Your name and affiliation
- Intended use (research, evaluation, collaboration, …)
- Python version and OS

We review each request and reply when we can provide a wheel. Preview wheels may
**include plain Python source** inside the package; we decide per request whether
to distribute them.

> Compiled wheels without source may be offered later when the API stabilises.
> Do not redistribute wheels you receive.
{form_section}
### Install

After you receive a wheel by email:

```bash
pip install /path/to/diffsolid-*.whl
python -c "import diffsolid as ds; print('DiffSolid OK')"
```

Optional extras (after the core wheel is installed):

```bash
pip install diffsolid[gpu]    # CuPy, nvmath, cuDSS
pip install diffsolid[viz]    # PyVista, VTK
pip install diffsolid[mesh]   # gmsh mesh generation helpers
```

### Developer editable install (source licensees only)

If you have been granted access to the private source repository:

```bash
cd /path/to/DiffSolid-core
pip install -e .
python scripts/check_install.py
```
"""

    return re.sub(
        r"## 4\. Installing DiffSolid\n\n.*?(?=\n---\n\n## 5\.)",
        replacement + "\n---\n",
        install_text,
        count=1,
        flags=re.DOTALL,
    )


def main() -> None:
    dist = _parse_block("wheel_distribution")
    (DOCS / "download.md").write_text(render_download_md(dist), encoding="utf-8")
    install_path = DOCS / "install.md"
    install_path.write_text(
        patch_install_wheel_section(install_path.read_text(encoding="utf-8"), dist),
        encoding="utf-8",
    )
    print(f"Applied site.yaml → download.md, install.md (contact: {_contact_email(dist)})")


if __name__ == "__main__":
    main()
