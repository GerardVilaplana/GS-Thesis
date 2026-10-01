#!/usr/bin/env python3
"""Create exp39 scene specs from the 17-category PointNet seen-instance val/test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "scene_id", "scene_key", "query", "feature_root", "split", "num_gaussians"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=Path(
            "/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/"
            "08_17category_mlp_baselines_v1/01_seen_instance_17cat/split_manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--num_shards", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text())["splits"]
    rows = []
    skipped = []
    for split in args.splits:
        for item in manifest[split]:
            scene_key = item["scene_key"]
            category = item["category"]
            query = QUERY_BY_CATEGORY.get(category)
            npz_path = Path(item["npz"])
            feature_root = npz_path.parent.parent
            scene_root = feature_root / "work" / "3dgs_scenes" / scene_key
            model_cameras = feature_root / "work" / "3dgs_models" / scene_key / "cameras.json"
            object_ply = feature_root / "work" / "object_ply" / f"{scene_key}_object_pruned_thr0.60.ply"
            ok = bool(query) and scene_root.exists() and model_cameras.exists() and object_ply.exists()
            record = {
                "category": category,
                "scene_id": item["scene_id"],
                "scene_key": scene_key,
                "query": query or "",
                "feature_root": str(feature_root),
                "split": split,
                "num_gaussians": int(item.get("num_gaussians", 0)),
            }
            if ok:
                rows.append(record)
            else:
                skipped.append(
                    {
                        **record,
                        "has_query": bool(query),
                        "has_scene_root": scene_root.exists(),
                        "has_model_cameras": model_cameras.exists(),
                        "has_object_ply": object_ply.exists(),
                    }
                )

    rows.sort(key=lambda r: (r["split"], r["category"], r["scene_key"]))
    write_csv(args.output, rows)
    skipped_path = args.output.with_name(args.output.stem + "_skipped.csv")
    if skipped:
        with skipped_path.open("w", newline="") as f:
            fields = list(skipped[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(skipped)
    else:
        skipped_path.write_text("")

    shard_rows = [[] for _ in range(args.num_shards)]
    shard_loads = [0 for _ in range(args.num_shards)]
    for row in sorted(rows, key=lambda r: int(r["num_gaussians"]), reverse=True):
        idx = min(range(args.num_shards), key=lambda i: shard_loads[i])
        shard_rows[idx].append(row)
        shard_loads[idx] += int(row["num_gaussians"])
    for idx, shard in enumerate(shard_rows):
        shard.sort(key=lambda r: (r["split"], r["category"], r["scene_key"]))
        write_csv(args.output.with_name(f"{args.output.stem}_shard{idx}.csv"), shard)

    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote {len(skipped)} skipped rows to {skipped_path}")
    for idx, shard in enumerate(shard_rows):
        print(f"shard{idx}: scenes={len(shard)} estimated_gaussians={shard_loads[idx]}")


if __name__ == "__main__":
    main()
