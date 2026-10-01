#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
TRAIN_SCRIPT = BASE_ROOT / "scripts" / "train_handal_exp53_pointnet_mean_sh_ablation.py"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "54_exp40_pointnet_mean_sh_loco_missing_v1"

FEATURES = [
    "xyz_sh",
    "xyz_scale_opacity_sh",
    "geometry_sh_scene_norm",
]

CATEGORIES = [
    "adjustable_wrenches",
    "combinational_wrenches",
    "fixed_joint_pliers",
    "hammers",
    "ladles",
    "locking_pliers",
    "measuring_cups",
    "mugs",
    "pots_pans",
    "power_drills",
    "ratchets",
    "screwdrivers",
    "slip_joint_pliers",
    "spatulas",
    "strainers",
    "utensils",
    "whisks",
]


def cleanup_heavy_outputs(out_root: Path) -> None:
    for pattern in ("**/model.pt", "**/test_predictions.npz", "**/loss_curve.png", "**/accuracy_curve.png"):
        for path in out_root.glob(pattern):
            path.unlink(missing_ok=True)


def run_feature(feature: str, categories: list[str], device: str) -> None:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        "pointnet_mean",
        "--feature_variant",
        feature,
        "--modes",
        "loco",
        "--heldout_categories",
        *categories,
        "--out_root",
        str(OUT_ROOT),
        "--device",
        device,
        "--skip_existing",
    ]
    print(f"[launch] feature={feature} categories={len(categories)} device={device}", flush=True)
    subprocess.run(cmd, check=True)
    cleanup_heavy_outputs(OUT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", choices=FEATURES, required=True)
    parser.add_argument("--category_shard", type=int, default=0)
    parser.add_argument("--num_category_shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    categories = CATEGORIES[args.category_shard :: args.num_category_shards]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for feature in args.features:
        run_feature(feature, categories, args.device)


if __name__ == "__main__":
    main()
