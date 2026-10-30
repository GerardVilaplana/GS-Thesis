# Dataset package

The curated benchmark contains 535 object scenes from 17 HANDAL categories.
Ten unreliable reconstructions were excluded before model development. Splits
are defined at the physical object-instance level, so scenes of the same object
do not cross training, validation, and test partitions.

## External archive layout

The dataset is not committed to Git. The preservation package uses this layout:

```text
HandAffordSplat-publication/
  benchmark/
    supervised_npz/          # 535 refined object scenes and lifted labels
    dinov2_object_crop/      # 535 matching multi-view DINOv2 feature files
    visualization_ply/       # 535 refined Gaussian PLY files
    splits/
      seen_split_manifest.csv
      seen_split_manifest.json
      excluded_scenes.txt
      loco/                   # 17 held-out-category manifests
  models/
    seen/                     # final seen checkpoint and records
    loco/                     # one final checkpoint per held-out category
  paper_reproduction/         # ablations, baselines, figures, and metrics
  private_material/           # own-scene inputs; not for public release
  provenance/                 # source snapshots, patches, and environments
  manifests/                  # inventories, checksums, and validation report
```

## Supervised NPZ schema

The portable model expects these arrays:

- `xyz`: Gaussian centers, shape `(N, 3)`.
- `scale`: Gaussian scales, shape `(N, 3)`.
- `rotation`: Gaussian rotations, shape `(N, 4)`.
- `opacity`: Gaussian opacity, shape `(N,)` or `(N, 1)`.
- `color`: RGB appearance, shape `(N, 3)`.
- `handle_labels_thr0_25`: binary lifted handle labels, shape `(N,)`.

Metadata fields include category, scene ID, instance ID, source split, and source
PLY. Metadata may be stored as scalar NumPy strings.

## DINOv2 NPZ schema

- `dino_features`: per-Gaussian DINOv2 embeddings, shape `(N, 384)`.
- `valid_dino`: whether an embedding was assigned, shape `(N,)`.
- `dino_weight_sum`: accumulated assignment weight, shape `(N,)`.
- `dino_visible_views`: number of contributing views, shape `(N,)`.

The Gaussian and DINOv2 files must contain the same number and ordering of
Gaussians. Invalid DINOv2 rows are set to zero before feature concatenation.

## Distribution note

HANDAL is an upstream dataset distributed under CC BY-NC-SA 4.0. The derived
Gaussian representations, lifted labels, released model weights, and associated
benchmark artifacts distributed by HandAffordSplat use the same CC BY-NC-SA
4.0 terms. See [`DATA_LICENSE.md`](../DATA_LICENSE.md) for the release policy.
The original personal scene captures under `private_material` remain private
and are excluded from the public benchmark.
