#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


C0 = 0.28209479177387814


def read_xyz(path):
    vertices = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    return xyz


def labels_from_annotation(base_ply, anno_ply, tolerance=1e-8):
    base_xyz = read_xyz(base_ply)
    anno_xyz = read_xyz(anno_ply)
    labels = np.zeros(len(base_xyz), dtype=bool)
    if len(anno_xyz) == 0:
        return labels, 0.0
    tree = cKDTree(base_xyz.astype(np.float64))
    dist, idx = tree.query(anno_xyz.astype(np.float64), k=1)
    max_dist = float(dist.max()) if len(dist) else 0.0
    if max_dist > tolerance:
        raise ValueError(f"Annotation does not match base PLY: {anno_ply} max_dist={max_dist:.6g}")
    labels[idx] = True
    return labels, max_dist


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def write_label_ply(base_ply, out_ply, grasp, wrap):
    ply = PlyData.read(str(base_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(grasp) or len(vertices) != len(wrap):
        raise ValueError(f"Label length mismatch for {base_ply}")

    rgb = np.zeros((len(vertices), 3), dtype=np.float32)
    non = ~grasp & ~wrap
    grasp_only = grasp & ~wrap
    wrap_only = ~grasp & wrap
    both = grasp & wrap

    rgb[non] = np.array([0.05, 0.18, 1.0], dtype=np.float32)       # blue
    rgb[grasp_only] = np.array([1.0, 0.02, 0.02], dtype=np.float32) # red
    rgb[wrap_only] = np.array([1.0, 0.92, 0.02], dtype=np.float32)  # yellow
    rgb[both] = np.array([1.0, 0.45, 0.0], dtype=np.float32)        # orange

    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_ply))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/gvilaplana/GS-Thesis/Affordances/data/AffordSplat_GS_grasp_wrap_subset"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "/home/gvilaplana/GS-Thesis/Affordances/outputs/04_3daffordsplat_baseline/"
            "mug_grasp_wrap_label_overlap_ply"
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--scene_ids", nargs="+", default=["0001", "0002", "0003", "0004"])
    args = parser.parse_args()

    root = args.dataset_root / "Seen" / args.split / "mug"
    rows = []
    for scene_id in args.scene_ids:
        base_ply = root / "Gaussian" / f"GS_{scene_id}.ply"
        grasp_ply = root / "grasp" / f"GS_anno_{scene_id}.ply"
        wrap_ply = root / "wrap_grasp" / f"GS_anno_{scene_id}.ply"
        if not base_ply.exists():
            raise FileNotFoundError(base_ply)
        if not grasp_ply.exists():
            raise FileNotFoundError(grasp_ply)
        if not wrap_ply.exists():
            raise FileNotFoundError(wrap_ply)

        grasp, grasp_max_dist = labels_from_annotation(base_ply, grasp_ply)
        wrap, wrap_max_dist = labels_from_annotation(base_ply, wrap_ply)
        both = grasp & wrap
        out_ply = args.out_dir / f"mug_{args.split}_GS_{scene_id}_grasp_red_wrap_yellow_both_orange.ply"
        write_label_ply(base_ply, out_ply, grasp, wrap)
        n = len(grasp)
        rows.append(
            {
                "split": args.split,
                "category": "mug",
                "scene_id": scene_id,
                "num_gaussians": n,
                "grasp_count": int(grasp.sum()),
                "wrap_grasp_count": int(wrap.sum()),
                "both_count": int(both.sum()),
                "grasp_only_count": int((grasp & ~wrap).sum()),
                "wrap_only_count": int((~grasp & wrap).sum()),
                "non_labeled_count": int((~grasp & ~wrap).sum()),
                "grasp_ratio": float(grasp.mean()),
                "wrap_grasp_ratio": float(wrap.mean()),
                "both_ratio": float(both.mean()),
                "grasp_max_match_dist": grasp_max_dist,
                "wrap_max_match_dist": wrap_max_dist,
                "output_ply": str(out_ply),
            }
        )
        print(f"wrote {out_ply}")

    summary = args.out_dir / "mug_grasp_wrap_overlap_summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
