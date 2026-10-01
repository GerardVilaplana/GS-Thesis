#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))

import train_mug_synthetic_domain_gap as prev  # noqa: E402
import train_mug_stage13_targeted_domain_alignment as stage13  # noqa: E402
import train_mug_stage17_b0_b6_full_mugs as stage17  # noqa: E402


C0 = 0.28209479177387814
FEATURE = "geometry_color_scene_norm"


def logit(x, eps=1e-6):
    x = np.clip(np.asarray(x, dtype=np.float32), eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def evenly_sample(items, n):
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[int(i)] for i in idx]


def write_b3_ply(item, target_stats, out_path, seed):
    scene = stage13.make_scene(
        item,
        FEATURE,
        "original",
        "opacity_match",
        seed,
        target_stats,
        apply_to_source=True,
    )
    x = scene["x"]
    labels = scene["y"].astype(bool)

    with np.load(item["npz"], allow_pickle=False) as z:
        source_ply = Path(str(np.asarray(z["source_ply"]).item()))
        original_opacity = z["opacity"].astype(np.float32)

    if len(x) != len(original_opacity):
        raise ValueError(f"Unexpected Gaussian count change for {item['scene_key']}")

    matched_opacity_feature = x[:, 10].astype(np.float32)

    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    vertices["opacity"] = logit(matched_opacity_feature)

    rgb = np.zeros((len(vertices), 3), dtype=np.float32)
    rgb[~labels] = np.array([0.05, 0.18, 1.0], dtype=np.float32)
    rgb[labels] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    for name in vertices.dtype.names:
        if name.startswith("f_rest_"):
            vertices[name] = 0.0

    ply["vertex"].data = vertices
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_path))

    return {
        "scene_key": item["scene_key"],
        "scene_id": item["scene_id"],
        "split": item["source_split"],
        "num_gaussians": int(len(vertices)),
        "positive_ratio": float(labels.mean()),
        "original_opacity_mean": float(original_opacity.mean()),
        "b3_matched_opacity_feature_mean": float(matched_opacity_feature.mean()),
        "original_opacity_min": float(original_opacity.min()),
        "original_opacity_max": float(original_opacity.max()),
        "b3_matched_opacity_feature_min": float(matched_opacity_feature.min()),
        "b3_matched_opacity_feature_max": float(matched_opacity_feature.max()),
        "source_ply": str(source_ply),
        "output_ply": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_root", type=Path, default=stage17.SYN_ROOT)
    parser.add_argument("--handal_root", type=Path, default=stage17.HANDAL_ROOT)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=stage17.OUT_ROOT / "b3_transformed_synthetic_mug_plys",
    )
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    syn_items, handal_items = prev.load_items(args.syn_root, args.handal_root)
    handal_split = stage17.grouped_ratio_split(handal_items, args.seed, train_ratio=0.60, val_ratio=0.10)
    syn_split = prev.split_synthetic(syn_items)
    target_stats = stage13.build_target_stats(handal_split["train"], [FEATURE])

    selected = evenly_sample(sorted(syn_split["train"], key=lambda x: x["scene_key"]), args.num_examples)
    rows = []
    for item in selected:
        out_ply = args.out_dir / f"{item['scene_key']}_B3_opacity_match_labels_red_blue.ply"
        rows.append(write_b3_ply(item, target_stats, out_ply, args.seed))
        print(f"wrote {out_ply}", flush=True)

    manifest = args.out_dir / "b3_transformed_synthetic_mug_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
