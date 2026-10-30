# Research workflows

This directory contains the scripts needed to reproduce the final thesis data,
model-selection experiments, and evaluations. It intentionally excludes older
exploratory and scene-specific one-offs from the active branch.

The complete script tree used during thesis development remains available in
the `thesis-submission-snapshot` Git tag and in the publication archive. No
historical implementation was discarded from the project record.

## Data construction

- `prepare_handal_3dgs_scenes.py` and `run_handal_exp40_3dgs.py`: prepare and
  reconstruct HANDAL scenes.
- `build_handal_exp40_15k_gt_features.py`: construct the supervised object
  representations and lifted labels.
- `extract_handal_dinov2_gaussian_embeddings.py`: assign multi-view DINOv2
  features to Gaussians.
- `make_handal_exp40_split.py`: build the instance-level data partitions.

## Language-guided object representations

- `prepare_handal_qwen_anchor_frames.py`: prepare query anchor frames.
- `build_handal_qwen_sam_anchor_realm_scenes.py` and
  `build_handal_sam2_multimask_realm_scenes.py`: generate language-guided masks
  and REALM inputs.
- `cleanup_reprojection_qwen_sam_scenes.py`: apply reprojection refinement.
- `run_exp49_e2e_auto_realm_handle_eval.py`: evaluate handle prediction on the
  resulting object representations.

## Model development

- `train_handal_exp41_architecture_baselines.py`: architecture comparison.
- `train_handal_exp44_pointnet_mean_feature_sweep.py`: Gaussian and DINOv2
  feature ablations.
- `train_exp48_gnn_embeddings_pointnet_mean.py` and
  `train_exp50_gnn_variations.py`: graph-based ablations.
- `recompute_graph_ablation_validation.py`: validation-protocol graph results.

## Final evaluation

- `recompute_external_baselines_seen_canonical.py`: canonical external-baseline
  comparison.
- `make_pointnet_category_seen_loco_test_plot.py`: final category analysis.
- `export_thesis_seen_loco_comparison_plys.py`: qualitative seen/LOCO assets.

Several retained entry points import helpers from other files in this directory.
Those helper files remain alongside the documented commands to preserve the
exact thesis behavior. Some paths are historical and must be overridden for a
new installation; use the portable `handaffordsplat-predict` command for normal
inference.
