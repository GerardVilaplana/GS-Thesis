#!/usr/bin/env python3
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
FEATURE_ROOTS = [
    BASE_ROOT / "data" / "handal_handle_generalization_features",
    BASE_ROOT / "data" / "handal_new_categories_delta_v1_features",
]
OUT_ROOT = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "18_handal_category_truecolor_inspection_v1"
)


def scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def read_item(path, feature_root):
    category, scene_id = path.stem.rsplit("__", 1)
    with np.load(path, allow_pickle=False) as z:
        split = str(scalar(z["split"])) if "split" in z.files else "unknown"
        instance_id = str(scalar(z["instance_id"])) if "instance_id" in z.files else scene_id[:3]
        object_ply = Path(str(scalar(z["source_object_ply"])))
        labels = z["handle_labels_thr0_25"].astype(bool)
    raw_scene_ply = feature_root / "work" / "raw_3dgs_ply" / f"{path.stem}_iter3000.ply"
    return {
        "scene_key": path.stem,
        "category": category,
        "scene_id": scene_id,
        "instance_id": instance_id,
        "source_split": split,
        "npz": path,
        "raw_scene_ply": raw_scene_ply,
        "object_ply": object_ply,
        "num_handle": int(labels.sum()),
        "handle_ratio": float(labels.mean()),
    }


def evenly_sample(items, n):
    items = sorted(items, key=lambda x: x["scene_key"])
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[int(i)] for i in idx]


def move_existing_object_examples(out_root):
    object_dir = out_root / "handal_objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for ply in sorted(out_root.glob("*.ply")):
        dst = object_dir / ply.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(ply), str(dst))
        moved.append(dst)

    old_manifest = out_root / "handal_nonmug_truecolor_examples_manifest.csv"
    if old_manifest.exists():
        dst_manifest = object_dir / old_manifest.name
        if dst_manifest.exists():
            dst_manifest.unlink()
        shutil.move(str(old_manifest), str(dst_manifest))
    return moved


def export_scene_examples(out_root, num_per_category):
    scene_dir = out_root / "handal_scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)

    by_category = defaultdict(list)
    seen = set()
    for root in FEATURE_ROOTS:
        for path in sorted((root / "npz").glob("*.npz")):
            if path.stem in seen:
                raise ValueError(f"Duplicate scene key: {path.stem}")
            seen.add(path.stem)
            item = read_item(path, root)
            if item["raw_scene_ply"].exists():
                by_category[item["category"]].append(item)

    rows = []
    for category in sorted(by_category):
        selected = evenly_sample(by_category[category], num_per_category)
        for item in selected:
            out_ply = scene_dir / f"{item['scene_key']}_full_scene_truecolor.ply"
            shutil.copy2(item["raw_scene_ply"], out_ply)
            rows.append(
                {
                    "category": category,
                    "scene_key": item["scene_key"],
                    "scene_id": item["scene_id"],
                    "instance_id": item["instance_id"],
                    "source_split": item["source_split"],
                    "handle_ratio_in_object_npz": item["handle_ratio"],
                    "source_raw_scene_ply": str(item["raw_scene_ply"]),
                    "output_ply": str(out_ply),
                }
            )
            print(f"wrote {out_ply}", flush=True)

    manifest = scene_dir / "handal_full_scene_truecolor_examples_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest} rows={len(rows)} categories={len(by_category)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--num_scenes_per_category", type=int, default=3)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    moved = move_existing_object_examples(args.out_root)
    print(f"moved object-pruned PLYs to handal_objects: {len(moved)}", flush=True)
    export_scene_examples(args.out_root, args.num_scenes_per_category)


if __name__ == "__main__":
    main()
