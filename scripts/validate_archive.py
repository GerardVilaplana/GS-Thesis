#!/usr/bin/env python3
"""Validate a HandAffordSplat publication archive."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from handaffordsplat.checkpoint import load_predictor
from handaffordsplat.features import build_final_features

DINO_SUFFIX = "_object_crop_center_avg_features.npz"
PLY_SUFFIX = "_object_center_ell20s75.ply"


def keys_with_suffix(root: Path, suffix: str) -> Set[str]:
    return {
        path.name[: -len(suffix)]
        for path in root.glob("*" + suffix)
        if path.name.endswith(suffix)
    }


def require_equal(name: str, expected: Set[str], actual: Set[str], errors: List[str]) -> None:
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        errors.append(
            "{} mismatch: missing={} extra={}".format(name, missing[:10], extra[:10])
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--deep", action="store_true", help="Load and assemble all 535 feature pairs.")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.archive.resolve()
    errors: List[str] = []
    warnings: List[str] = []

    scene_file = root / "manifests" / "final_scene_keys.txt"
    expected = {
        line.strip() for line in scene_file.read_text().splitlines() if line.strip()
    }
    if len(expected) != 535:
        errors.append("Expected 535 unique final scene keys, found {}".format(len(expected)))

    npz_root = root / "benchmark" / "supervised_npz"
    dino_root = root / "benchmark" / "dinov2_object_crop" / "features"
    ply_root = root / "benchmark" / "visualization_ply"

    object_keys = {path.stem for path in npz_root.glob("*.npz")}
    dino_keys = keys_with_suffix(dino_root, DINO_SUFFIX)
    ply_keys = keys_with_suffix(ply_root, PLY_SUFFIX)
    require_equal("supervised NPZ", expected, object_keys, errors)
    require_equal("DINOv2 features", expected, dino_keys, errors)
    require_equal("visualization PLY", expected, ply_keys, errors)

    loco_manifests = sorted((root / "benchmark" / "splits" / "loco").glob("*.json"))
    if len(loco_manifests) != 17:
        errors.append("Expected 17 LOCO split manifests, found {}".format(len(loco_manifests)))

    checkpoints = sorted((root / "models").glob("**/model.pt"))
    if len(checkpoints) != 18:
        errors.append("Expected 18 final checkpoints, found {}".format(len(checkpoints)))
    checkpoint_rows: List[Dict[str, object]] = []
    for path in checkpoints:
        try:
            predictor = load_predictor(path)
            checkpoint_rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "input_dim": int(predictor.mean.shape[1]),
                    "threshold": predictor.threshold,
                    "latent_dim": int(predictor.model.encoder[3].out_features),
                }
            )
        except Exception as exc:
            errors.append("Checkpoint {}: {}".format(path, exc))

    checked_scenes = 0
    if args.deep:
        for key in sorted(expected):
            object_path = npz_root / (key + ".npz")
            dino_path = dino_root / (key + DINO_SUFFIX)
            try:
                bundle = build_final_features(object_path, dino_path)
                if bundle.labels is None:
                    errors.append("{} has no lifted handle labels".format(key))
                if len(bundle.features) == 0:
                    errors.append("{} contains no Gaussians".format(key))
                checked_scenes += 1
            except Exception as exc:
                errors.append("Feature pair {}: {}".format(key, exc))

    broken_links = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_symlink() and not path.exists()
    ]
    if broken_links:
        errors.append("Broken symbolic links: {}".format(broken_links[:10]))

    compressed_masks = sorted(
        (root / "private_material" / "own_scene_inputs").glob("*.tar.gz")
    )
    for path in compressed_masks:
        try:
            with tarfile.open(path, "r:gz") as archive:
                if not archive.getmembers():
                    errors.append("{} is empty".format(path.name))
        except Exception as exc:
            errors.append("Compressed mask archive {}: {}".format(path, exc))

    if not (root / "provenance" / "thesis" / "MISSING_OVERLEAF_SOURCE.md").exists():
        warnings.append("Missing explicit Overleaf source reminder.")

    report = {
        "archive": str(root),
        "status": "ok" if not errors else "failed",
        "expected_scenes": len(expected),
        "object_npz": len(object_keys),
        "dino_feature_files": len(dino_keys),
        "visualization_ply": len(ply_keys),
        "loco_manifests": len(loco_manifests),
        "checkpoints": checkpoint_rows,
        "deep_checked_scenes": checked_scenes,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = args.report or root / "manifests" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
