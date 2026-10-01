#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))
import train_handal_17cat_mlp_baselines as base  # noqa: E402
import train_mug_synthetic_domain_gap as prev  # noqa: E402
import train_mug_stage13_targeted_domain_alignment as stage13  # noqa: E402
import train_mug_stage14_uniform_mixed_split as stage14  # noqa: E402


C0 = 0.28209479177387814
STAGE14_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "14_mug_uniform_mixed_60_10_30_v1"
OUT_ROOT = STAGE14_ROOT / "diagnostic_plys"
HANDAL_OBJECT_PLY = BASE_ROOT / "data" / "handal_handle_generalization_features" / "work" / "object_ply"

RUNS = [
    "stage14_control_clean_mixed__geometry_color_scene_norm__label-original__data-clean",
    "stage14_best2_all_combined__geometry_color_scene_norm__label-dilate_small__data-combined",
    "stage14_best1_balanced_combined__geometry_color_scene_norm__label-dilate_small__data-combined",
]


def scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.tolist()).strip("[]'\"")


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def set_rgb(vertices, rgb):
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    return vertices


def red_blue_from_bool(mask):
    mask = np.asarray(mask).astype(bool)
    rgb = np.full((len(mask), 3), np.array([0.05, 0.18, 1.0], dtype=np.float32))
    rgb[mask] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    return rgb


def write_colored_ply(source_ply, out_ply, rgb):
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(rgb):
        raise ValueError(f"Length mismatch for {source_ply}: ply={len(vertices)} rgb={len(rgb)}")
    ply["vertex"].data = set_rgb(vertices, rgb)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_ply))


def write_transformed_visual_ply(source_ply, out_ply, labels, rng, floater_source=None):
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(labels):
        raise ValueError(f"Length mismatch for {source_ply}: ply={len(vertices)} labels={len(labels)}")

    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    extent = np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 1e-6)
    jitter = rng.normal(0.0, 0.025 * float(np.linalg.norm(extent)), size=xyz.shape).astype(np.float32)
    vertices["x"] = xyz[:, 0] + jitter[:, 0]
    vertices["y"] = xyz[:, 1] + jitter[:, 1]
    vertices["z"] = xyz[:, 2] + jitter[:, 2]

    if "opacity" in vertices.dtype.names:
        lo, hi = np.percentile(vertices["opacity"].astype(np.float32), [5, 95])
        op = vertices["opacity"].astype(np.float32)
        op = (op - lo) / max(float(hi - lo), 1e-6)
        vertices["opacity"] = np.clip(op * 0.45 + rng.normal(0, 0.05, size=len(op)), 0.0, 1.0)

    vertices = set_rgb(vertices, red_blue_from_bool(labels))

    if floater_source is not None and floater_source.exists():
        floater_ply = PlyData.read(str(floater_source))
        floater_vertices = np.array(floater_ply["vertex"].data, copy=True)
        common = [name for name in vertices.dtype.names if name in floater_vertices.dtype.names]
        if len(floater_vertices) and common == list(vertices.dtype.names):
            n = max(1, int(round(0.20 * len(vertices))))
            idx = rng.choice(len(floater_vertices), size=n, replace=len(floater_vertices) < n)
            extra = floater_vertices[idx].copy()
            extra = set_rgb(extra, red_blue_from_bool(np.zeros(n, dtype=bool)))
            vertices = np.concatenate([vertices, extra])

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=ply.text, byte_order=ply.byte_order).write(str(out_ply))


def load_threshold(model_dir):
    with open(model_dir / "overall_metrics.json", "r") as f:
        return float(json.load(f)["selected_threshold"])


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_handal_predictions(stage14_root, out_root):
    rows = []
    for run_name in RUNS:
        model_name = "pointnet_geometry_color_scene_norm"
        model_dir = stage14_root / run_name / model_name
        threshold = load_threshold(model_dir)
        pred = np.load(model_dir / "test_predictions.npz", allow_pickle=False)
        scene_keys = [str(x) for x in pred["scene_keys"]]
        offsets = pred["offsets"]
        scores = pred["scores"].astype(np.float32)
        labels = pred["labels"].astype(np.uint8)
        for scene_key, (start, end) in zip(scene_keys, offsets):
            if scene_key.startswith("affordsplat_"):
                continue
            start, end = int(start), int(end)
            source_ply = HANDAL_OBJECT_PLY / f"{scene_key}_object_pruned_thr0.60.ply"
            scene_scores = scores[start:end]
            scene_labels = labels[start:end]
            pred_mask = scene_scores >= threshold
            pred_ply = out_root / "handal_test_predictions" / run_name / f"{scene_key}_{run_name}_pred_red_blue.ply"
            gt_ply = out_root / "handal_test_ground_truth" / f"{scene_key}_gt_handle_red_blue.ply"
            write_colored_ply(source_ply, pred_ply, red_blue_from_bool(pred_mask))
            if not gt_ply.exists():
                write_colored_ply(source_ply, gt_ply, red_blue_from_bool(scene_labels))
            rows.append(
                {
                    "kind": "handal_prediction",
                    "run_name": run_name,
                    "scene_key": scene_key,
                    "threshold": threshold,
                    "source_ply": str(source_ply),
                    "output_ply": str(pred_ply),
                    "gt_ply": str(gt_ply),
                    "num_gaussians": len(scene_scores),
                    "pred_positive_ratio": float(pred_mask.mean()),
                    "gt_positive_ratio": float(scene_labels.mean()),
                }
            )
    return rows


def export_synthetic_transform_examples(syn_items, handal_items, out_root, seed, num_examples):
    rng = random.Random(seed)
    candidates = [x for x in syn_items if x["source_split"] == "train"]
    candidates = sorted(candidates, key=lambda x: base.stable_key(x["scene_key"], seed))
    chosen = candidates[:num_examples]

    handal_ply_candidates = [
        HANDAL_OBJECT_PLY / f"{item['scene_key']}_object_pruned_thr0.60.ply"
        for item in sorted(handal_items, key=lambda x: base.stable_key(x["scene_key"], seed + 99))
    ]
    floater_source = next((p for p in handal_ply_candidates if p.exists()), None)

    rows = []
    for item in chosen:
        with np.load(item["npz"], allow_pickle=False) as z:
            source_ply = Path(scalar(z["source_ply"]))
            labels = z["handle_labels_thr0_25"].astype(np.float32)
        x, y = stage13.load_feature_arrays(item, "geometry_color_scene_norm")
        dilated = stage13.transform_labels(x, y, "dilate_small").astype(bool)

        original_ply = out_root / "affordsplat_transform_sanity" / item["scene_key"] / f"{item['scene_key']}_original_label_red_blue.ply"
        dilated_ply = out_root / "affordsplat_transform_sanity" / item["scene_key"] / f"{item['scene_key']}_dilate_small_label_red_blue.ply"
        combined_ply = out_root / "affordsplat_transform_sanity" / item["scene_key"] / f"{item['scene_key']}_combined_visual_approx_dilate_small_red_blue.ply"

        write_colored_ply(source_ply, original_ply, red_blue_from_bool(labels))
        write_colored_ply(source_ply, dilated_ply, red_blue_from_bool(dilated))
        np_rng = np.random.default_rng(seed + base.stable_key(item["scene_key"], seed) % 1_000_000)
        write_transformed_visual_ply(source_ply, combined_ply, dilated, np_rng, floater_source=floater_source)

        for name, out_ply, lab in [
            ("original_label", original_ply, labels.astype(bool)),
            ("dilate_small_label", dilated_ply, dilated),
            ("combined_visual_approx_dilate_small", combined_ply, dilated),
        ]:
            rows.append(
                {
                    "kind": "affordsplat_transform_sanity",
                    "view": name,
                    "scene_key": item["scene_key"],
                    "source_ply": str(source_ply),
                    "output_ply": str(out_ply),
                    "num_gaussians": int(len(lab)),
                    "positive_ratio": float(np.mean(lab)),
                    "note": "combined_visual_approx is a visual approximation; training transform is applied in feature space",
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage14_root", type=Path, default=STAGE14_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--syn_root", type=Path, default=stage14.SYN_ROOT)
    parser.add_argument("--handal_root", type=Path, default=stage14.HANDAL_ROOT)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--num_synthetic_examples", type=int, default=5)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    syn_items, handal_items = prev.load_items(args.syn_root, args.handal_root)

    rows = []
    rows.extend(export_handal_predictions(args.stage14_root, args.out_root))
    rows.extend(export_synthetic_transform_examples(syn_items, handal_items, args.out_root, args.seed, args.num_synthetic_examples))
    write_csv(args.out_root / "stage14_diagnostic_plys_manifest.csv", rows)
    print(f"wrote {len(rows)} diagnostic PLY entries")
    print(args.out_root)


if __name__ == "__main__":
    main()
