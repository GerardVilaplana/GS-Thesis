#!/usr/bin/env python3
"""Create scene_specs CSV for HANDAL test_seen + val_seen scenes with usable assets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DATA = Path("/home/gvilaplana/GS-Thesis/Affordances/data")
ROOTS = [
    (
        DATA / "handal_handle_generalization_v1" / "manifests" / "manifest.csv",
        DATA / "handal_handle_generalization_features",
    ),
    (
        DATA / "handal_new_categories_delta_v1" / "manifests" / "manifest.csv",
        DATA / "handal_new_categories_delta_v1_features",
    ),
]
QUERY_BY_CATEGORY = {
    "adjustable_wrenches": "adjustable wrench",
    "combinational_wrenches": "combination wrench",
    "fixed_joint_pliers": "joint plier",
    "hammers": "hammer",
    "ladles": "ladle",
    "locking_pliers": "locking plier",
    "measuring_cups": "measuring cup",
    "mugs": "mug",
    "pots_pans": "pot",
    "power_drills": "power drill",
    "ratchets": "ratchet",
    "screwdrivers": "screwdriver",
    "slip_joint_pliers": "slip joint plier",
    "spatulas": "spatula",
    "strainers": "strainer",
    "utensils": "utensil",
    "whisks": "whisk",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test_seen", "val_seen"])
    args = parser.parse_args()

    rows = []
    skipped = []
    seen = set()
    for manifest, feature_root in ROOTS:
        with manifest.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] not in set(args.splits):
                    continue
                category = row["category"]
                scene_id = row["scene_id"]
                scene_key = f"{category}__{scene_id}"
                key = (scene_key, str(feature_root))
                if key in seen:
                    continue
                seen.add(key)
                object_ply = feature_root / "work" / "object_ply" / f"{scene_key}_object_pruned_thr0.60.ply"
                cameras = feature_root / "work" / "3dgs_models" / scene_key / "cameras.json"
                scene_root = feature_root / "work" / "3dgs_scenes" / scene_key
                query = QUERY_BY_CATEGORY.get(category)
                if query and object_ply.exists() and cameras.exists() and scene_root.exists():
                    rows.append(
                        {
                            "category": category,
                            "scene_id": scene_id,
                            "scene_key": scene_key,
                            "query": query,
                            "feature_root": str(feature_root),
                            "split": row["split"],
                        }
                    )
                else:
                    skipped.append(
                        {
                            "split": row["split"],
                            "category": category,
                            "scene_id": scene_id,
                            "scene_key": scene_key,
                            "has_query": bool(query),
                            "has_object_ply": object_ply.exists(),
                            "has_cameras": cameras.exists(),
                            "has_scene_root": scene_root.exists(),
                        }
                    )

    rows.sort(key=lambda r: (r["split"], r["category"], r["scene_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "scene_id", "scene_key", "query", "feature_root", "split"])
        writer.writeheader()
        writer.writerows(rows)
    skipped_path = args.output.with_name(args.output.stem + "_skipped.csv")
    with skipped_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "category", "scene_id", "scene_key", "has_query", "has_object_ply", "has_cameras", "has_scene_root"],
        )
        writer.writeheader()
        writer.writerows(skipped)
    print(f"wrote {len(rows)} scene specs to {args.output}")
    print(f"wrote {len(skipped)} skipped rows to {skipped_path}")


if __name__ == "__main__":
    main()
