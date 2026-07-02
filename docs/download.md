---
title: Download wheel
---

# Download wheel

Preview wheels are **not on PyPI**. Approved users install via a **password-protected download** — credentials are sent by email after [access approval](install.md#request-access).


<div class="ds-download__portal ds-download__portal--pending" markdown="1">

<p class="ds-download__pending">The password download portal is enabled after your access request is approved. You will receive the portal URL and password by email.</p>

</div>


## Before you download

1. Email **[ChenlongZhao@sjtu.edu.cn](mailto:ChenlongZhao@sjtu.edu.cn)** with your name, affiliation, intended use, Python version, and OS.
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
