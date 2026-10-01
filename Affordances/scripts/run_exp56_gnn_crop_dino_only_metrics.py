#!/usr/bin/env python3
from pathlib import Path

import train_exp50_gnn_variations as exp50

BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
DINO_BASE = BASE_ROOT / "outputs" / "03_handle_generalization" / "46_exp40_dino_assignment_all_features_v1"

exp50.OUT_ROOT = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "56_exp46_crop_dino_only_gnn_pointnet_mean_v1"
)

exp50.CONFIGS["object_crop_dino_only"] = {
    "dino_method": "object_crop_center_avg",
    "feature_variant": "dino_only",
    "dino_root": DINO_BASE / "object_crop_center_avg_patch896x672_views48",
}


if __name__ == "__main__":
    exp50.main()
