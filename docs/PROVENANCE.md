# Provenance

The publication archive records the exact source state used for the thesis:

- Git bundles for the public project and upstream REALM checkout.
- A patch containing the uncommitted REALM and CUDA rasterizer modifications.
- A filtered REALM source snapshot with those modifications already applied.
- A snapshot of all authored HandAffordSplat scripts.
- Commit and remote records for REALM, OpenAD, 3DAffordSplat, and SAMURAI.
- Exact environment exports and system version reports.

This is important because the local REALM checkout used for the final pipeline
contains project-specific Python and rasterizer changes that are not represented
by the upstream commit alone.
