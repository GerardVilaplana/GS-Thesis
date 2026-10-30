# Reproducibility guide

## Reproducing final model inference

1. Install the compact environment from `environment.yml`.
2. Obtain one refined object NPZ and its matching object-crop DINOv2 NPZ.
3. Select the seen checkpoint or the checkpoint for the held-out LOCO category.
4. Run `handaffordsplat-predict` as shown in the main README.
5. Use the threshold stored in the checkpoint. Do not tune it on test data.

## Reproducing model development

The original thesis scripts are retained under `Affordances/scripts/`. The key
final scripts are:

- `build_handal_exp40_15k_gt_features.py`: supervised object construction.
- `extract_handal_dinov2_gaussian_embeddings.py`: multi-view DINOv2 assignment.
- `train_handal_exp41_architecture_baselines.py`: architecture comparison.
- `train_handal_exp44_pointnet_mean_feature_sweep.py`: Gaussian and DINOv2 feature experiments.
- `run_exp49_e2e_auto_realm_handle_eval.py`: language-guided evaluation.
- `recompute_external_baselines_seen_canonical.py`: external baseline metrics.
- `recompute_graph_ablation_validation.py`: graph appendix metrics.

These scripts preserve the exact thesis procedure but some retain historical
absolute paths. The separate provenance archive contains the complete authored
source snapshot and the exact working-tree patch. Prefer the portable package
for inference and use the research scripts when reproducing an experiment.

## Selection protocol

- Training scenes fit model parameters and feature standardization.
- Validation scenes select checkpoints, thresholds, architectures, and features.
- Test scenes are used after selecting the final configuration.
- Seen metrics are scene averages.
- For LOCO, scenes are averaged within each held-out category and the 17 category
  values are then averaged equally.

## Exact environment preservation

The publication archive contains `conda env export`, `--from-history`, and
`pip freeze` records for the REALM, Qwen2.5-VL, SAM2/SAMURAI, and COLMAP helper
environments, together with CUDA, compiler, and repository commit information.
Those records are the authoritative historical environment description.

## Language-guided pipeline

The language-guided pipeline uses query-based detection with Qwen2.5-VL,
SAM2 mask extraction and propagation, REALM Gaussian segmentation, center and
ellipsoid reprojection refinement, object-crop DINOv2 assignment, and the same
supervised HandAffordSplat checkpoint. The model is not retrained on
language-guided representations.
