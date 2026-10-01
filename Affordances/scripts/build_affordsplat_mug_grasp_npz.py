#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import cKDTree


C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sh_to_rgb(f_dc):
    return np.clip(0.5 + C0 * f_dc, 0.0, 1.0).astype(np.float32)


def normalize_xyz(xyz):
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    return (centered / radius).astype(np.float32)


def read_gaussian_ply(path):
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    xyz_norm = normalize_xyz(xyz)
    scale = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]).astype(np.float32)
    rot = np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]).astype(np.float32)
    opacity = sigmoid(np.asarray(v["opacity"], dtype=np.float32))[:, None]
    color = sh_to_rgb(
        np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float32)
    )
    geometry = np.concatenate([xyz_norm, scale, rot, opacity], axis=1).astype(np.float32)
    return {
        "xyz": xyz,
        "geometry_features": geometry,
        "scale": scale,
        "rotation": rot,
        "opacity": opacity[:, 0],
        "color": color,
    }


def labels_from_annotation(base_xyz, anno_ply, tolerance=1e-8):
    anno_xyz = read_gaussian_ply(anno_ply)["xyz"]
    labels = np.zeros(len(base_xyz), dtype=np.uint8)
    if len(anno_xyz) == 0:
        return labels, 0.0
    tree = cKDTree(base_xyz.astype(np.float64))
    dist, idx = tree.query(anno_xyz.astype(np.float64), k=1)
    max_dist = float(dist.max()) if len(dist) else 0.0
    if max_dist > tolerance:
        raise ValueError(f"Annotation mismatch for {anno_ply}: max_dist={max_dist:.6g}")
    labels[idx] = 1
    return labels, max_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/gvilaplana/GS-Thesis/Affordances/data/AffordSplat_GS_grasp_wrap_subset"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("/home/gvilaplana/GS-Thesis/Affordances/data/affordsplat_mug_grasp_features_v1"),
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    npz_root = args.out_root / "npz"
    npz_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.dataset_root / "manifest.csv")
    manifest = manifest[
        (manifest["category"] == "mug") & (manifest["affordance"] == "grasp")
    ].copy()
    manifest = manifest.sort_values(["split", "gs_ply"]).reset_index(drop=True)

    rows = []
    for _, row in manifest.iterrows():
        split = str(row["split"])
        gs_rel = str(row["gs_ply"])
        anno_rel = str(row["anno_ply"])
        gs_path = args.dataset_root / gs_rel
        anno_path = args.dataset_root / anno_rel
        scene_id = Path(gs_rel).stem.replace("GS_", "")
        scene_key = f"affordsplat_mug_{split}_{scene_id}"

        data = read_gaussian_ply(gs_path)
        labels, max_dist = labels_from_annotation(data["xyz"], anno_path)
        out_path = npz_root / f"{scene_key}.npz"
        np.savez_compressed(
            out_path,
            xyz=data["xyz"].astype(np.float32),
            geometry_features=data["geometry_features"].astype(np.float32),
            scale=data["scale"].astype(np.float32),
            rotation=data["rotation"].astype(np.float32),
            opacity=data["opacity"].astype(np.float32),
            color=data["color"].astype(np.float32),
            handle_labels_thr0_25=labels.astype(np.uint8),
            labels_grasp=labels.astype(np.uint8),
            category=np.asarray("mugs"),
            source_domain=np.asarray("affordsplat"),
            scene_key=np.asarray(scene_key),
            scene_id=np.asarray(scene_id),
            split=np.asarray(split),
            instance_id=np.asarray(scene_id),
            source_ply=np.asarray(str(gs_path)),
            source_anno_ply=np.asarray(str(anno_path)),
            affordance=np.asarray("grasp"),
        )
        rows.append(
            {
                "scene_key": scene_key,
                "split": split,
                "scene_id": scene_id,
                "npz": str(out_path),
                "num_gaussians": int(len(labels)),
                "num_grasp": int(labels.sum()),
                "grasp_ratio": float(labels.mean()),
                "max_match_dist": max_dist,
                "source_ply": str(gs_path),
                "source_anno_ply": str(anno_path),
            }
        )
        print(f"wrote {out_path}")

    with (args.out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_root / 'summary.csv'} scenes={len(rows)}")


if __name__ == "__main__":
    main()
