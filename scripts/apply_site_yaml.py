#!/usr/bin/env python3
"""Inject site.yaml values into generated docs (download portal, install section)."""

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
    portal = dist.get("download_portal_url", "").strip()

    if portal:
        portal_block = f"""
<div class="ds-download__portal" markdown="1">

<form class="ds-download__form" method="post" action="{portal.rstrip('/')}/download">
  <label class="ds-download__label" for="wheel-password">Access password</label>
  <input class="ds-download__input" id="wheel-password" name="password" type="password" autocomplete="current-password" required />
  <button class="ds-download__submit" type="submit">Download preview wheel</button>
</form>

<p class="ds-download__hint">The form opens the password-protected download service. Use the password from your approval email only.</p>

<p class="ds-download__alt">Prefer the hosted portal? <a href="{portal}" target="_blank" rel="noopener">Open download page ↗</a></p>

</div>
"""
    else:
        portal_block = """
<div class="ds-download__portal ds-download__portal--pending" markdown="1">

<p class="ds-download__pending">The password download portal is enabled after your access request is approved. You will receive the portal URL and password by email.</p>

</div>
"""

    return f"""---
title: Download wheel
---

# Download wheel

Preview wheels are **not on PyPI**. Approved users install via a **password-protected download** — credentials are sent by email after [access approval](install.md#request-access).

{portal_block}

## Before you download

1. Email **[{contact}](mailto:{contact})** with your name, affiliation, intended use, Python version, and OS.
2. After approval, use the password (and portal link, if provided) from our reply.
3. Install locally:

```bash
pip install /path/to/diffsolid-*.whl
python -c "import diffsolid as ds; print('DiffSolid OK')"
```

See [Install guide](install.md) for GPU setup and optional extras.

!!! warning "Distribution terms"
    Do not share passwords, portal links, or wheel files. Preview wheels may contain plain Python source; redistribution is not permitted.

[← Request access](install.md#request-access)
"""


def patch_install_portal(install_text: str, dist: dict[str, str]) -> str:
    contact = _contact_email(dist)
    portal = dist.get("download_portal_url", "").strip()

    portal_section = ""
    if portal:
        portal_section = f"""
### Download wheel (password)

Approved users can download the latest preview wheel on the **[Download wheel](download.md)** page or directly at [{portal}]({portal}).

Enter the access password from your approval email. Do not share the password or wheel file.
"""
    else:
        portal_section = f"""
### Download wheel (password)

After approval, you will receive a **portal link and access password** by email. The [Download wheel](download.md) page describes the install flow.

"""

    replacement = f"""## 4. Installing DiffSolid

DiffSolid is **not** published on PyPI. Preview wheels are distributed on request
only — they are not publicly downloadable.

### Request access

Send an email to **[{contact}](mailto:{contact})** with:
- Your name and affiliation
- Intended use (research, evaluation, collaboration, …)
- Python version and OS

We typically respond within a few business days. Approved users receive a **download password**
(and portal link when enabled) by email.

> **Preview wheels** may contain plain Python source inside the package. Do not
> redistribute wheels or credentials. Compiled wheels without source will follow
> when the API stabilises.

{portal_section}
### Install after approval

Download the wheel via the portal, then install:

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
        patch_install_portal(install_path.read_text(encoding="utf-8"), dist),
        encoding="utf-8",
    )
    portal = dist.get("download_portal_url", "").strip() or "(email only — set download_portal_url in site.yaml)"
    print(f"Applied site.yaml → download.md, install.md (portal: {portal})")


if __name__ == "__main__":
    main()
