#!/usr/bin/env python3
"""Export REALM per-Gaussian predicted object IDs as colored PLY files."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def id2rgb(idx: int, max_num_obj: int = 256) -> np.ndarray:
    rgb = np.zeros((3,), dtype=np.uint8)
    if idx == 0:
        return rgb
    golden_ratio = 1.6180339887
    h = (idx * golden_ratio) % 1
    s = 0.5 + (idx % 2) * 0.5
    r, g, b = colorsys.hls_to_rgb(h, 0.5, s)
    rgb[:] = int(r * 255), int(g * 255), int(b * 255)
    return rgb


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    return (rgb - 0.5) / C0


def predict_ids(vertex_data: np.ndarray, classifier_path: Path, num_classes: int) -> np.ndarray:
    obj = np.stack([vertex_data[f"obj_dc_{i}"] for i in range(16)], axis=0).astype(np.float32)
    classifier = torch.nn.Conv2d(16, num_classes, kernel_size=1)
    classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.from_numpy(obj)[None, :, :, None])
        return logits.argmax(dim=1).squeeze(0).squeeze(-1).cpu().numpy().astype(np.int32)


def write_3dgs_colored(ply_data: PlyData, pred_ids: np.ndarray, output_path: Path, num_classes: int) -> None:
    vertex = ply_data["vertex"].data.copy()
    for class_id in np.unique(pred_ids):
        mask = pred_ids == class_id
        sh = rgb_to_sh(id2rgb(int(class_id), num_classes))
        vertex["f_dc_0"][mask] = sh[0]
        vertex["f_dc_1"][mask] = sh[1]
        vertex["f_dc_2"][mask] = sh[2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply_data.text).write(output_path)


def write_points_colored(vertex_data: np.ndarray, pred_ids: np.ndarray, output_path: Path, num_classes: int) -> None:
    xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)
    colors = np.stack([id2rgb(int(i), num_classes) for i in pred_ids], axis=0)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    out = np.empty(len(xyz), dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["red"], out["green"], out["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(out, "vertex")], text=True).write(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point_cloud", required=True, type=Path)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_classes", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ply_data = PlyData.read(args.point_cloud)
    vertex = ply_data["vertex"].data
    pred_ids = predict_ids(vertex, args.classifier, args.num_classes)

    full_path = args.output_dir / "scene_1_realm_pred_ids_3dgs_colored.ply"
    points_path = args.output_dir / "scene_1_realm_pred_ids_points_colored.ply"
    meta_path = args.output_dir / "scene_1_realm_pred_ids_metadata.json"

    write_3dgs_colored(ply_data, pred_ids, full_path, args.num_classes)
    write_points_colored(vertex, pred_ids, points_path, args.num_classes)

    ids, counts = np.unique(pred_ids, return_counts=True)
    meta = {
        "point_cloud": str(args.point_cloud),
        "classifier": str(args.classifier),
        "num_gaussians": int(len(pred_ids)),
        "num_unique_ids": int(len(ids)),
        "id_counts": {str(int(i)): int(c) for i, c in zip(ids, counts)},
        "outputs": {
            "colored_3dgs_ply": str(full_path),
            "colored_points_ply": str(points_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
