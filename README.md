# HandAffordSplat

HandAffordSplat predicts handle regions directly on object-level 3D Gaussian
Splatting representations. The selected model combines Gaussian parameters,
multi-view DINOv2 features, and object-level context through a PointNet-style
network with mean pooling.

The project also contains the language-guided pipeline used to obtain queried
object representations with Qwen2.5-VL, SAM2, and REALM. Large datasets,
checkpoints, reconstructions, and paper artifacts are intentionally kept out of
Git and are described in [docs/DATASET.md](docs/DATASET.md).

## Main results

| Evaluation | mIoU | F1 |
|---|---:|---:|
| Supervised HANDAL, seen test | 0.855 | 0.917 |
| Supervised HANDAL, LOCO test | 0.672 | 0.763 |
| Language-guided, seen test | 0.831 | 0.891 |
| Language-guided, LOCO test | 0.653 | 0.733 |

LOCO results are first averaged over scenes within each held-out category and
then averaged equally across the 17 categories.

## Installation

```bash
conda env create -f environment.yml
conda activate handaffordsplat
pip install -e .
```

The exact historical environments and third-party source states used for the
thesis are preserved in the separate publication archive. The environment here
is a compact inference and evaluation environment.

## Inference

Each prediction requires the refined object NPZ, its matching object-crop
DINOv2 NPZ, and a trained checkpoint:

```bash
handaffordsplat-predict \
  --object-npz path/to/category__scene.npz \
  --dino-npz path/to/category__scene_object_crop_center_avg_features.npz \
  --checkpoint path/to/model.pt \
  --output prediction.npz
```

To export a thesis-style blue/green PLY overlay, also pass the source object PLY:

```bash
handaffordsplat-predict \
  --object-npz object.npz \
  --dino-npz dino_features.npz \
  --checkpoint model.pt \
  --output prediction.npz \
  --source-ply object.ply \
  --output-ply prediction_overlay.ply
```

The checkpoint stores the training normalization statistics and selected
validation threshold. Inference therefore does not refit any normalization or
threshold on the target scene.

## Repository layout

- `src/handaffordsplat/`: portable feature, model, metric, checkpoint, and PLY utilities.
- `scripts/`: user-facing prediction and archive-validation entry points.
- `Affordances/scripts/`: selected data-construction, training, ablation, and
  evaluation workflows. See its README for the active script set and the
  `thesis-submission-snapshot` tag for the complete historical tree.
- `REALM-Code/`: project-side REALM integration files.
- `Supervision/scripts/`: mask-generation utilities.
- `docs/`: dataset, results, and reproducibility documentation.
- `tests/`: focused tests for the portable implementation.

## Reproducibility

Start with [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The separate
publication archive contains the 535-scene benchmark, DINOv2 features, 18 final
checkpoints, split manifests, structured experiment records, exact source
snapshots, environment exports, and paper assets.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

Original HandAffordSplat source code is released under the [MIT
License](LICENSE). The HANDAL-derived benchmark, released model weights, and
associated research artifacts are released under CC BY-NC-SA 4.0; see
[DATA_LICENSE.md](DATA_LICENSE.md). Third-party components retain their own
licenses, as documented in [THIRD_PARTY.md](THIRD_PARTY.md). Personal scene
captures under `private_material` are not part of the public release.
