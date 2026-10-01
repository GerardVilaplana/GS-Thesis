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


FEATURE = "geometry_color_scene_norm"


def logit(x, eps=1e-6):
    x = np.clip(np.asarray(x, dtype=np.float32), eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def evenly_sample(items, n):
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[int(i)] for i in idx]


def scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def export_synthetic_b3_truecolor(item, target_stats, out_path, seed):
    scene = stage13.make_scene(
        item,
        FEATURE,
        "original",
        "opacity_match",
        seed,
        target_stats,
        apply_to_source=True,
    )
    matched_opacity_feature = scene["x"][:, 10].astype(np.float32)
    labels = scene["y"].astype(bool)

    with np.load(item["npz"], allow_pickle=False) as z:
        source_ply = Path(str(scalar(z["source_ply"])))
        original_opacity = z["opacity"].astype(np.float32)

    if len(matched_opacity_feature) != len(original_opacity):
        raise ValueError(f"Unexpected Gaussian count change for {item['scene_key']}")

    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    vertices["opacity"] = logit(matched_opacity_feature)
    ply["vertex"].data = vertices

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_path))
    return {
        "domain": "affordsplat_b3_opacity_match",
        "scene_key": item["scene_key"],
        "scene_id": item["scene_id"],
        "split": item["source_split"],
        "num_gaussians": int(len(vertices)),
        "positive_ratio": float(labels.mean()),
        "original_opacity_mean": float(original_opacity.mean()),
        "export_opacity_feature_mean": float(matched_opacity_feature.mean()),
        "source_ply": str(source_ply),
        "output_ply": str(out_path),
    }


def export_handal_truecolor(item, out_path):
    with np.load(item["npz"], allow_pickle=False) as z:
        source_ply = Path(str(scalar(z["source_object_ply"])))
        labels = z["handle_labels_thr0_25"].astype(bool)
        opacity = np.asarray(z["opacity"], dtype=np.float32).reshape(-1)

    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ply["vertex"].data = vertices
    ply.write(str(out_path))
    return {
        "domain": "handal_train_truecolor",
        "scene_key": item["scene_key"],
        "scene_id": item["scene_id"],
        "split": item["source_split"],
        "num_gaussians": int(len(vertices)),
        "positive_ratio": float(labels.mean()),
        "original_opacity_mean": float(opacity.mean()),
        "export_opacity_feature_mean": "",
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
        default=stage17.OUT_ROOT / "truecolor_domain_comparison_plys",
    )
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    syn_items, handal_items = prev.load_items(args.syn_root, args.handal_root)
    handal_split = stage17.grouped_ratio_split(handal_items, args.seed, train_ratio=0.60, val_ratio=0.10)
    syn_split = prev.split_synthetic(syn_items)
    target_stats = stage13.build_target_stats(handal_split["train"], [FEATURE])

    rows = []
    syn_selected = evenly_sample(sorted(syn_split["train"], key=lambda x: x["scene_key"]), args.num_examples)
    handal_selected = evenly_sample(sorted(handal_split["train"], key=lambda x: x["scene_key"]), args.num_examples)

    for item in syn_selected:
        out_ply = args.out_dir / "affordsplat_b3_truecolor" / f"{item['scene_key']}_B3_opacity_match_truecolor.ply"
        rows.append(export_synthetic_b3_truecolor(item, target_stats, out_ply, args.seed))
        print(f"wrote {out_ply}", flush=True)

    for item in handal_selected:
        out_ply = args.out_dir / "handal_train_truecolor" / f"{item['scene_key']}_HANDAL_train_truecolor.ply"
        rows.append(export_handal_truecolor(item, out_ply))
        print(f"wrote {out_ply}", flush=True)

    manifest = args.out_dir / "truecolor_domain_comparison_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
