#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import train_handal_exp44_pointnet_mean_feature_sweep as exp44  # noqa: E402


def main() -> None:
    exp44.OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "55_exp46_crop_dino_only_pointnet_mean_v1"
    exp44.ACTIVE_DINO_ROOT = (
        BASE_ROOT
        / "outputs"
        / "03_handle_generalization"
        / "46_exp40_dino_assignment_all_features_v1"
        / "object_crop_center_avg_patch896x672_views48"
    )
    exp44.export_qa_plys = lambda *args, **kwargs: None

    if "--out_root" not in sys.argv:
        sys.argv.extend(["--out_root", str(exp44.OUT_ROOT)])
    if "--dino_root" not in sys.argv:
        sys.argv.extend(["--dino_root", str(exp44.ACTIVE_DINO_ROOT)])
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "pointnet_mean"])
    if "--feature_variant" not in sys.argv:
        sys.argv.extend(["--feature_variant", "dino_only"])

    exp44.main()


if __name__ == "__main__":
    main()
