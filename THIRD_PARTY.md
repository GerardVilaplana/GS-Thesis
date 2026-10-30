# Third-party components

HandAffordSplat builds on the following projects. Their source code, models, and
data remain subject to their respective licenses.

| Component | Upstream | Thesis source commit |
|---|---|---|
| HANDAL | https://github.com/NVlabs/HANDAL | Dataset source; see upstream release |
| REALM | Upstream recorded in the preservation archive | b6f2498ca4c6d0082105140c48cd54c658296279 |
| OpenAD | https://github.com/Fsoft-AIC/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds | b082265ed085455a3fdc80b0d8d95dc51351e7b4 |
| 3DAffordSplat | https://github.com/HCPLab-SYSU/3DAffordSplat | afdab93acadef2322bd795f360a4313fef48b545 |
| SAMURAI | Upstream recorded in the preservation archive | 76ba195984892b0d1e3db5d9c9f90bb62175680a |

The final language-guided pipeline also uses Qwen2.5-VL, SAM2, DINOv2, COLMAP,
and 3D Gaussian Splatting. Exact package versions, repository remotes, local
patches, and environment exports are stored in the separate provenance archive.

The local REALM working tree used for the thesis includes project-specific
Python and CUDA rasterizer changes. Reproducing the pipeline requires the saved
source snapshot or applying the saved patch to the recorded upstream commit.
