#!/usr/bin/env python3
"""Export selected reprojection strategy PLYs for one automatic REALM scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from segment_anything import SamPredictor, sam_model_registry

REALM_ROOT = Path("/home/gvilaplana/GS-Thesis/REALM-Code")
AFF = Path("/home/gvilaplana/GS-Thesis/Affordances")
if str(REALM_ROOT) not in sys.path:
    sys.path.insert(0, str(REALM_ROOT))
if str(AFF / "scripts") not in sys.path:
    sys.path.insert(0, str(AFF / "scripts"))

from evaluate_exp38_reprojection_strategy_ablation import (  # noqa: E402
    STRATEGIES,
    parse_frame_number,
    pred_ply_path,
    qwen_boxes,
    qwen_rows,
    sample_points,
    source_image_for_frame,
    strategy_keep,
)
from scripts.make_qwen_sam_realm_stage_videos import sam_masks_for_boxes  # noqa: E402


def load_rgb(path: Path, max_width: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_width > 0 and im.width > max_width:
        new_h = int(round(im.height * max_width / im.width))
        im = im.resize((max_width, new_h), Image.BICUBIC)
    return np.array(im)


def make_sam_mask(predictor: SamPredictor, image_np: np.ndarray, boxes: list[list[float]], device: torch.device) -> np.ndarray:
    union = np.zeros(image_np.shape[:2], dtype=bool)
    for mask in sam_masks_for_boxes(predictor, image_np, boxes, device):
        union |= mask
    return union


def write_subset(ply: PlyData, keep: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.array(ply["vertex"].data, copy=True)[keep]
    PlyData([PlyElement.describe(data, "vertex")], text=ply.text).write(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--frame_dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", default=["center_strict", "ellipsoid20_strict75"])
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--sam_checkpoint", default=str(REALM_ROOT / "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"))
    args = parser.parse_args()

    cameras = sorted(json.loads((args.scene_dir / "realm_model" / "cameras.json").read_text()), key=lambda c: int(c["id"]))
    rows = qwen_rows(args.scene_dir, args.query)
    if not rows:
        raise RuntimeError(f"No usable Qwen rows for {args.scene_dir.name}/{args.query}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to(device).eval()
    predictor = SamPredictor(sam)

    frames = []
    with torch.no_grad():
        for row in rows:
            image_path = source_image_for_frame(args.scene_dir, args.frame_dir, cameras, row["frame_name"], args.max_width)
            image_np = load_rgb(image_path, args.max_width)
            boxes = qwen_boxes(row, image_np.shape[1], image_np.shape[0])
            mask = make_sam_mask(predictor, image_np, boxes, device)
            frames.append({"frame_name": row["frame_name"], "camera": cameras[parse_frame_number(row["frame_name"]) - 1], "mask": mask})

    raw_path = pred_ply_path(args.scene_dir, args.query)
    ply = PlyData.read(raw_path)
    vertex = ply["vertex"].data
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scene_key": args.scene_dir.name,
        "query": args.query,
        "raw_ply": str(raw_path),
        "qwen_frames_used": len(rows),
        "strategies": {},
    }
    sample_cache = {}
    for name in args.strategies:
        strategy = STRATEGIES[name]
        sample_kind = strategy["sample"]
        if sample_kind not in sample_cache:
            sample_cache[sample_kind] = sample_points(vertex, xyz, sample_kind)
        keep, keep_summary, _ = strategy_keep(xyz, sample_cache[sample_kind], frames, strategy)
        out = args.output_dir / f"object_{name}_truecolor_15k.ply"
        write_subset(ply, keep, out)
        summary["strategies"][name] = {"output": str(out), **keep_summary}
        print(json.dumps({"strategy": name, "output": str(out), **keep_summary}, indent=2), flush=True)

    (args.output_dir / "reprojection_strategy_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
