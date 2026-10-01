#!/usr/bin/env python3
"""Export REALM query-selected Gaussian IDs as true-color and ID-color PLY variants."""

from __future__ import annotations

import argparse
import colorsys
import json
import re
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def id2rgb(idx: int, max_num_obj: int = 256) -> np.ndarray:
    rgb = np.zeros((3,), dtype=np.uint8)
    if idx == 0:
        return rgb
    h = (idx * 1.6180339887) % 1
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


def set_vertex_color_by_id(vertex: np.ndarray, ids: np.ndarray, num_classes: int) -> None:
    for class_id in np.unique(ids):
        mask = ids == class_id
        sh = rgb_to_sh(id2rgb(int(class_id), num_classes))
        vertex["f_dc_0"][mask] = sh[0]
        vertex["f_dc_1"][mask] = sh[1]
        vertex["f_dc_2"][mask] = sh[2]


def set_vertex_gray(vertex: np.ndarray, gray: int = 145) -> None:
    sh = rgb_to_sh(np.array([gray, gray, gray], dtype=np.uint8))
    vertex["f_dc_0"][:] = sh[0]
    vertex["f_dc_1"][:] = sh[1]
    vertex["f_dc_2"][:] = sh[2]


def write_ply(vertex: np.ndarray, text: bool, path: Path) -> None:
    PlyData([PlyElement.describe(vertex, "vertex")], text=text).write(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point_cloud", required=True, type=Path)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--query_report", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--num_classes", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.query_report.read_text())
    ply_data = PlyData.read(args.point_cloud)
    vertex = ply_data["vertex"].data
    pred_ids = predict_ids(vertex, args.classifier, args.num_classes)

    summary = {}
    for query in args.queries:
        q = report["queries"].get(query)
        if q is None:
            summary[query] = {"error": "query_missing_from_report"}
            continue
        selected_ids = [int(i) for i in q.get("selected_class_ids", [])]
        selected = np.isin(pred_ids, selected_ids)
        slug = slugify(query)

        true_vertex = vertex[selected].copy()
        true_path = args.output_dir / f"{slug}_object_truecolor_3dgs.ply"
        write_ply(true_vertex, ply_data.text, true_path)

        id_vertex = vertex[selected].copy()
        set_vertex_color_by_id(id_vertex, pred_ids[selected], args.num_classes)
        id_path = args.output_dir / f"{slug}_object_idcolor_3dgs.ply"
        write_ply(id_vertex, ply_data.text, id_path)

        scene_vertex = vertex.copy()
        set_vertex_gray(scene_vertex)
        for class_id in selected_ids:
            mask = pred_ids == class_id
            sh = rgb_to_sh(id2rgb(class_id, args.num_classes))
            scene_vertex["f_dc_0"][mask] = sh[0]
            scene_vertex["f_dc_1"][mask] = sh[1]
            scene_vertex["f_dc_2"][mask] = sh[2]
        scene_path = args.output_dir / f"{slug}_scene_gray_selected_idcolor_3dgs.ply"
        write_ply(scene_vertex, ply_data.text, scene_path)

        summary[query] = {
            "selected_ids": selected_ids,
            "num_selected_gaussians": int(selected.sum()),
            "selected_fraction": float(selected.mean()),
            "truecolor": str(true_path),
            "object_idcolor": str(id_path),
            "scene_gray_selected": str(scene_path),
        }

    out_summary = args.output_dir / "selected_id_ply_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
