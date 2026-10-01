#!/usr/bin/env python3
"""Create SAM-vs-REALM diagnostic videos for HANDAL query pruning checks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from segment_anything import SamPredictor, sam_model_registry

REALM_ROOT = Path("/home/gvilaplana/GS-Thesis/REALM-Code")
if str(REALM_ROOT) not in sys.path:
    sys.path.insert(0, str(REALM_ROOT))

from scripts.make_qwen_sam_realm_stage_videos import box_items_to_xyxy, sam_masks_for_boxes


FRAME_RE = re.compile(r"frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_frame_number(name: str) -> int:
    match = FRAME_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected frame name: {name}")
    return int(match.group(1))


def load_rgb(path: Path, max_width: int, interpolation=Image.BICUBIC) -> tuple[np.ndarray, float]:
    arr = np.array(Image.open(path).convert("RGB"))
    if max_width <= 0 or arr.shape[1] <= max_width:
        return arr, 1.0
    scale = max_width / arr.shape[1]
    h = int(round(arr.shape[0] * scale))
    arr = np.array(Image.fromarray(arr).resize((max_width, h), interpolation))
    return arr, scale


def add_label(arr: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
    draw.text((7, 7), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    if mask.any():
        out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_three(rgb: np.ndarray, sam: np.ndarray, realm: np.ndarray) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.55).copy()
    sam_only = sam & ~realm
    realm_only = realm & ~sam
    inter = sam & realm
    out[sam_only] = (255, 0, 0)
    out[realm_only] = (255, 255, 0)
    out[inter] = (255, 145, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def process_scene(row: dict[str, str], predictor: SamPredictor, device: torch.device, args: argparse.Namespace) -> dict:
    scene_key = row["scene_key"]
    query = row["query"]
    qwen_summary = json.loads(Path(row["qwen_summary"]).read_text())
    rows = qwen_summary["queries"][query]
    image_dir = Path(row["frame_dir"])
    realm_root = Path(row["model_root"]) / "train" / "ours_3000"
    pred_dir = realm_root / "objects_pred"
    out_dir = args.output_dir / scene_key
    out_dir.mkdir(parents=True, exist_ok=True)

    first_rgb, _ = load_rgb(image_dir / rows[0]["frame_name"], args.max_width)
    h, w = first_rgb.shape[:2]
    side_writer = make_writer(out_dir / f"{scene_key}_{query.replace(' ', '_')}_sam_vs_realm_side_by_side.mp4", args.fps, (w * 2, h))
    overlay_writer = make_writer(out_dir / f"{scene_key}_{query.replace(' ', '_')}_sam_realm_intersection_overlay.mp4", args.fps, (w, h))

    diagnostics = []
    with torch.no_grad():
        for item in rows:
            frame_name = item["frame_name"]
            frame_number = parse_frame_number(frame_name)
            render_index = frame_number - 1
            pred_path = pred_dir / f"{render_index:05d}.png"
            if not pred_path.exists():
                diagnostics.append({"frame_name": frame_name, "missing_pred": True})
                continue

            rgb, scale = load_rgb(image_dir / frame_name, args.max_width, Image.BICUBIC)
            pred_rgb, _ = load_rgb(pred_path, args.max_width, Image.NEAREST)
            if pred_rgb.shape[:2] != rgb.shape[:2]:
                pred_rgb = np.array(Image.fromarray(pred_rgb).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))

            boxes = box_items_to_xyxy(item.get("parsed", []), scale, rgb.shape[1], rgb.shape[0])
            sam_masks = sam_masks_for_boxes(predictor, rgb, boxes, device)
            sam_union = np.zeros(rgb.shape[:2], dtype=bool)
            for mask in sam_masks:
                sam_union |= mask

            realm_mask = pred_rgb.sum(axis=-1) > 0
            inter = sam_union & realm_mask
            union = sam_union | realm_mask
            iou = float(inter.sum() / max(union.sum(), 1))

            left = add_label(overlay_mask(rgb, sam_union, (255, 0, 0)), f"SAM mask | {query} | {frame_name}")
            right = add_label(overlay_mask(rgb, realm_mask, (255, 255, 0)), f"REALM foreground | IoU {iou:.3f}")
            side_writer.write(np.hstack([left, right])[:, :, ::-1])

            overlay = add_label(overlay_three(rgb, sam_union, realm_mask), "red=SAM yellow=REALM orange=intersection")
            overlay_writer.write(overlay[:, :, ::-1])

            diagnostics.append({
                "frame_name": frame_name,
                "render_index": int(render_index),
                "num_boxes": int(len(boxes)),
                "sam_area": int(sam_union.sum()),
                "realm_area": int(realm_mask.sum()),
                "intersection_area": int(inter.sum()),
                "iou": iou,
            })

    side_writer.release()
    overlay_writer.release()

    return {
        "scene_key": scene_key,
        "query": query,
        "num_frames": len(diagnostics),
        "mean_iou": float(np.mean([d["iou"] for d in diagnostics if "iou" in d])) if diagnostics else 0.0,
        "side_by_side": str(out_dir / f"{scene_key}_{query.replace(' ', '_')}_sam_vs_realm_side_by_side.mp4"),
        "overlay": str(out_dir / f"{scene_key}_{query.replace(' ', '_')}_sam_realm_intersection_overlay.mp4"),
        "frames": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/"
        "24_handal_full_realm_query_pruning_5scenes_v1"
    ))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--sam_checkpoint", type=Path, default=Path(
        "/home/gvilaplana/GS-Thesis/REALM-Code/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"
    ))
    parser.add_argument("--max_width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = args.root / "run_manifest.csv"
    if args.output_dir is None:
        args.output_dir = args.root / "diagnostic_videos"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint)).to(device)
    predictor = SamPredictor(sam)

    reports = []
    with args.manifest.open() as f:
        for row in csv.DictReader(f):
            print(f"Processing {row['scene_key']} / {row['query']}", flush=True)
            reports.append(process_scene(row, predictor, device, args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sam_realm_mismatch_video_report.json").write_text(json.dumps(reports, indent=2))
    print(args.output_dir / "sam_realm_mismatch_video_report.json")


if __name__ == "__main__":
    main()
