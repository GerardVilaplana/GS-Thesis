#!/usr/bin/env python3
"""Export one true-color Gaussian PLY per selected REALM object ID."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def predict_ids(vertex_data: np.ndarray, classifier_path: Path, num_classes: int) -> np.ndarray:
    obj = np.stack([vertex_data[f"obj_dc_{i}"] for i in range(16)], axis=0).astype(np.float32)
    classifier = torch.nn.Conv2d(16, num_classes, kernel_size=1)
    classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.from_numpy(obj)[None, :, :, None])
        return logits.argmax(dim=1).squeeze(0).squeeze(-1).cpu().numpy().astype(np.int32)


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
        slug = slugify(query)
        selected_ids = [int(i) for i in q.get("selected_class_ids", [])]
        query_out = args.output_dir / slug
        query_out.mkdir(parents=True, exist_ok=True)

        outputs = []
        for class_id in selected_ids:
            mask = pred_ids == class_id
            out_vertex = vertex[mask].copy()
            out_path = query_out / f"{slug}_realm_id{class_id:03d}_truecolor_3dgs.ply"
            PlyData([PlyElement.describe(out_vertex, "vertex")], text=ply_data.text).write(out_path)
            outputs.append(
                {
                    "class_id": class_id,
                    "num_gaussians": int(mask.sum()),
                    "path": str(out_path),
                }
            )
        summary[query] = {
            "selected_ids": selected_ids,
            "num_parts": len(outputs),
            "parts": outputs,
        }

    summary_path = args.output_dir / "per_id_truecolor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
