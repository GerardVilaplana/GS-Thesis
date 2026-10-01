#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from plyfile import PlyData


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
FEATURE_ROOTS = [
    BASE_ROOT / "data" / "handal_handle_generalization_features",
    BASE_ROOT / "data" / "handal_new_categories_delta_v1_features",
]
OUT_DIR = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "21_handal_opacity_threshold_visual_check_v1"
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
        opacity = np.asarray(z["opacity"], dtype=np.float32).reshape(-1)
        labels = z["handle_labels_thr0_25"].astype(bool)
    return {
        "scene_key": path.stem,
        "category": category,
        "scene_id": scene_id,
        "instance_id": instance_id,
        "split": split,
        "npz": path,
        "source_ply": source_ply,
        "opacity": opacity,
        "labels": labels,
    }


def choose_instances(items, num_instances):
    by_instance = defaultdict(list)
    for item in sorted(items, key=lambda x: x["scene_key"]):
        by_instance[item["instance_id"]].append(item)

    instance_ids = sorted(by_instance)
    if len(instance_ids) <= num_instances:
        selected_instance_ids = instance_ids
    else:
        idx = np.linspace(0, len(instance_ids) - 1, num_instances).round().astype(int)
        selected_instance_ids = [instance_ids[int(i)] for i in idx]

    selected = []
    for instance_id in selected_instance_ids:
        scenes = by_instance[instance_id]
        selected.append(scenes[len(scenes) // 2])
    return selected


def write_filtered_ply(source_ply, opacity, threshold, out_ply):
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(opacity):
        raise ValueError(
            f"Opacity/Ply count mismatch for {source_ply}: "
            f"{len(opacity)} opacity values vs {len(vertices)} vertices"
        )
    keep = opacity > threshold
    ply["vertex"].data = vertices[keep]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_ply))
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--num_instances", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.15, 0.30])
    args = parser.parse_args()

    items = []
    seen = set()
    for root in FEATURE_ROOTS:
        npz_dir = root / "npz"
        for path in sorted(npz_dir.glob("*.npz")):
            if path.stem in seen:
                raise ValueError(f"Duplicate scene key across feature roots: {path.stem}")
            seen.add(path.stem)
            item = read_item(path)
            if item["source_ply"].exists():
                items.append(item)

    by_category = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)

    rows = []
    for category in sorted(by_category):
        selected = choose_instances(by_category[category], args.num_instances)
        for item in selected:
            for threshold in args.thresholds:
                threshold_tag = str(threshold).replace(".", "p")
                out_ply = (
                    args.out_dir
                    / category
                    / f"thr{threshold_tag}"
                    / f"{item['scene_key']}_opacity_gt_{threshold_tag}.ply"
                )
                keep = write_filtered_ply(item["source_ply"], item["opacity"], threshold, out_ply)
                labels = item["labels"]
                kept_labels = labels[keep]
                rows.append(
                    {
                        "category": category,
                        "scene_key": item["scene_key"],
                        "scene_id": item["scene_id"],
                        "instance_id": item["instance_id"],
                        "source_split": item["split"],
                        "threshold": threshold,
                        "num_gaussians_before": int(len(keep)),
                        "num_gaussians_after": int(keep.sum()),
                        "keep_ratio": float(keep.mean()),
                        "handle_gaussians_before": int(labels.sum()),
                        "handle_gaussians_after": int(kept_labels.sum()),
                        "handle_keep_ratio": float(kept_labels.sum() / max(labels.sum(), 1)),
                        "source_ply": str(item["source_ply"]),
                        "output_ply": str(out_ply),
                    }
                )
                print(
                    f"wrote {out_ply} kept={int(keep.sum())}/{len(keep)} "
                    f"handle_kept={int(kept_labels.sum())}/{int(labels.sum())}",
                    flush=True,
                )

    manifest = args.out_dir / "opacity_threshold_visual_check_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = args.out_dir / "opacity_threshold_visual_check_summary.csv"
    summary_rows = []
    for category in sorted(by_category):
        for threshold in args.thresholds:
            subset = [r for r in rows if r["category"] == category and r["threshold"] == threshold]
            summary_rows.append(
                {
                    "category": category,
                    "threshold": threshold,
                    "num_examples": len(subset),
                    "mean_keep_ratio": float(np.mean([r["keep_ratio"] for r in subset])),
                    "mean_handle_keep_ratio": float(np.mean([r["handle_keep_ratio"] for r in subset])),
                    "mean_gaussians_after": float(np.mean([r["num_gaussians_after"] for r in subset])),
                }
            )
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"done categories={len(by_category)} files={len(rows)} "
        f"manifest={manifest} summary={summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
