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
OUT_DIR = (
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


def read_item(path):
    category, scene_id = path.stem.rsplit("__", 1)
    with np.load(path, allow_pickle=False) as z:
        source_ply = Path(str(scalar(z["source_object_ply"])))
        split = str(scalar(z["split"])) if "split" in z.files else "unknown"
        instance_id = str(scalar(z["instance_id"])) if "instance_id" in z.files else scene_id[:3]
        labels = z["handle_labels_thr0_25"].astype(bool)
        num_gaussians = int(len(labels))
        num_handle = int(labels.sum())
    return {
        "scene_key": path.stem,
        "category": category,
        "scene_id": scene_id,
        "instance_id": instance_id,
        "split": split,
        "npz": path,
        "source_ply": source_ply,
        "num_gaussians": num_gaussians,
        "num_handle": num_handle,
        "handle_ratio": float(num_handle / max(num_gaussians, 1)),
    }


def evenly_sample(items, n):
    items = sorted(items, key=lambda x: x["scene_key"])
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[int(i)] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--num_per_category", type=int, default=2)
    parser.add_argument("--include_mugs", action="store_true")
    args = parser.parse_args()

    items = []
    seen = set()
    for root in FEATURE_ROOTS:
        for path in sorted((root / "npz").glob("*.npz")):
            if path.stem in seen:
                raise ValueError(f"Duplicate scene key: {path.stem}")
            seen.add(path.stem)
            item = read_item(path)
            if item["source_ply"].exists():
                items.append(item)

    by_category = defaultdict(list)
    for item in items:
        if item["category"] == "mugs" and not args.include_mugs:
            continue
        by_category[item["category"]].append(item)

    rows = []
    for category in sorted(by_category):
        selected = evenly_sample(by_category[category], args.num_per_category)
        category_dir = args.out_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for item in selected:
            out_ply = category_dir / f"{item['scene_key']}_truecolor.ply"
            shutil.copy2(item["source_ply"], out_ply)
            rows.append(
                {
                    "category": category,
                    "scene_key": item["scene_key"],
                    "scene_id": item["scene_id"],
                    "instance_id": item["instance_id"],
                    "source_split": item["split"],
                    "num_gaussians": item["num_gaussians"],
                    "num_handle": item["num_handle"],
                    "handle_ratio": item["handle_ratio"],
                    "source_ply": str(item["source_ply"]),
                    "output_ply": str(out_ply),
                }
            )
            print(f"wrote {out_ply}", flush=True)

    manifest = args.out_dir / "handal_nonmug_truecolor_examples_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest} rows={len(rows)} categories={len(by_category)}", flush=True)


if __name__ == "__main__":
    main()
