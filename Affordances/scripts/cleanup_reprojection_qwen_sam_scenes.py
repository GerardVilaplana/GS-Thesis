#!/usr/bin/env python3
"""Scene-by-scene Qwen+SAM reprojection cleanup and metrics aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from segment_anything import SamPredictor, sam_model_registry

REALM_ROOT = Path("/home/gvilaplana/GS-Thesis/REALM-Code")
if str(REALM_ROOT) not in sys.path:
    sys.path.insert(0, str(REALM_ROOT))

from scripts.make_qwen_sam_realm_stage_videos import sam_masks_for_boxes  # noqa: E402


def parse_frame_number(frame_name: str) -> int:
    match = re.search(r"frame_(\d+)", frame_name)
    if not match:
        raise ValueError(f"Unexpected frame name: {frame_name}")
    return int(match.group(1))


def scene_slug(scene_key: str) -> str:
    return scene_key.replace("__", "_").replace("/", "_")


def read_xyz(path: Path) -> tuple[PlyData, np.ndarray]:
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    return ply, xyz


def write_subset(ply: PlyData, keep: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.array(ply["vertex"].data, copy=True)[keep]
    PlyData([PlyElement.describe(data, "vertex")], text=ply.text).write(out)


def load_specs(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pred_ply_path(scene_dir: Path, query: str) -> Path:
    summary = json.loads((scene_dir / "final_ply" / "selected_id_ply_summary.json").read_text())
    return Path(summary[query]["truecolor"])


def qwen_rows(scene_dir: Path, query: str) -> list[dict]:
    summary = json.loads((scene_dir / "qwen" / "qwen_vl_box_summary.json").read_text())
    rows = summary.get("queries", {}).get(query, [])
    return [r for r in rows if int(r.get("num_boxes", 0)) > 0 and not r.get("parse_error")]


def qwen_boxes(row: dict, width: int, height: int) -> list[list[float]]:
    boxes = []
    for item in row.get("parsed", []):
        box = item.get("bbox_2d", item.get("box", []))
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1 = max(0.0, min(width - 1.0, x1))
        x2 = max(0.0, min(width - 1.0, x2))
        y1 = max(0.0, min(height - 1.0, y1))
        y2 = max(0.0, min(height - 1.0, y2))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def load_resized_rgb(path: Path, max_width: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_width > 0 and im.width > max_width:
        new_h = int(round(im.height * max_width / im.width))
        im = im.resize((max_width, new_h), Image.BICUBIC)
    return np.array(im)


def source_image_for_frame(exp_root: Path, scene_dir: Path, cameras: list[dict], frame_name: str) -> Path:
    saved = exp_root / "frame_images" / scene_dir.name / frame_name
    if saved.exists():
        return saved
    meta = json.loads((scene_dir / "metadata.json").read_text())
    idx = parse_frame_number(frame_name) - 1
    stem = cameras[idx]["img_name"]
    rgb_dir = Path(meta["source_scene"]) / "rgb"
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = rgb_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing source RGB for {scene_dir.name} frame {frame_name} stem {stem}")


def project_points(points: np.ndarray, cam: dict, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot_c2w = np.asarray(cam["rotation"], dtype=np.float64)
    center = np.asarray(cam["position"], dtype=np.float64)
    cam_xyz = (points - center) @ rot_c2w
    z = cam_xyz[:, 2]
    sx = out_w / float(cam["width"])
    sy = out_h / float(cam["height"])
    x = float(cam["fx"]) * sx * (cam_xyz[:, 0] / z) + out_w * 0.5
    y = float(cam["fy"]) * sy * (cam_xyz[:, 1] / z) + out_h * 0.5
    valid = (z > 1e-8) & (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
    return x, y, valid


def reference_tolerance(gt_xyz: np.ndarray, factor: float) -> float:
    if len(gt_xyz) < 2:
        return 1e-6
    d, _ = cKDTree(gt_xyz).query(gt_xyz, k=2)
    return max(float(np.median(d[:, 1])) * factor, 1e-6)


def metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray, tol: float) -> dict:
    pred_n = len(pred_xyz)
    gt_n = len(gt_xyz)
    if pred_n == 0 or gt_n == 0:
        inter_p = inter_g = 0
    else:
        gt_tree = cKDTree(gt_xyz)
        pred_tree = cKDTree(pred_xyz)
        dp, _ = gt_tree.query(pred_xyz, k=1)
        dg, _ = pred_tree.query(gt_xyz, k=1)
        inter_p = int((dp <= tol).sum())
        inter_g = int((dg <= tol).sum())
    precision = inter_p / pred_n if pred_n else 0.0
    recall = inter_g / gt_n if gt_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = pred_n + gt_n - min(inter_p, inter_g)
    iou = min(inter_p, inter_g) / union if union else 0.0
    return {
        "pred_count": pred_n,
        "gt_count": gt_n,
        "pred_near_gt": inter_p,
        "gt_covered_by_pred": inter_g,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def cleanup_one(spec: dict, args: argparse.Namespace, predictor: SamPredictor, device: torch.device) -> None:
    scene_key = spec["scene_key"]
    query = spec["query"]
    scene_dir = args.exp_root / scene_key
    raw_ply_path = pred_ply_path(scene_dir, query)
    raw_ply, raw_xyz = read_xyz(raw_ply_path)
    cameras = sorted(json.loads((scene_dir / "realm_model" / "cameras.json").read_text()), key=lambda c: int(c["id"]))
    rows = qwen_rows(scene_dir, query)
    if not rows:
        raise RuntimeError(f"No usable Qwen rows for {scene_key}/{query}")

    visible = np.zeros(len(raw_xyz), dtype=np.int32)
    inside = np.zeros(len(raw_xyz), dtype=np.int32)
    frame_rows = []
    with torch.no_grad():
        for row in rows:
            cam = cameras[parse_frame_number(row["frame_name"]) - 1]
            image_np = load_resized_rgb(source_image_for_frame(args.exp_root, scene_dir, cameras, row["frame_name"]), args.max_width)
            boxes = qwen_boxes(row, image_np.shape[1], image_np.shape[0])
            sam_union = np.zeros(image_np.shape[:2], dtype=bool)
            for mask in sam_masks_for_boxes(predictor, image_np, boxes, device):
                sam_union |= mask
            px, py, valid = project_points(raw_xyz, cam, image_np.shape[1], image_np.shape[0])
            visible[valid] += 1
            if valid.any() and sam_union.any():
                xi = np.floor(px[valid]).astype(np.int64)
                yi = np.floor(py[valid]).astype(np.int64)
                in_mask_valid = sam_union[yi, xi]
                valid_idx = np.flatnonzero(valid)
                inside[valid_idx[in_mask_valid]] += 1
                frame_inside = int(in_mask_valid.sum())
            else:
                frame_inside = 0
            frame_rows.append(
                {
                    "frame_name": row["frame_name"],
                    "num_boxes": len(boxes),
                    "sam_area": int(sam_union.sum()),
                    "projected_visible": int(valid.sum()),
                    "projected_inside": frame_inside,
                }
            )

    ratio = np.divide(inside, np.maximum(visible, 1), dtype=np.float64)
    keep = ((visible >= args.min_visible) & (ratio >= args.inside_fraction)) | ((visible < args.min_visible) & (inside >= 1))
    clean_ply = args.out_dir / "plys" / f"{scene_slug(scene_key)}_reproj_clean.ply"
    write_subset(raw_ply, keep, clean_ply)

    gt_ply = Path(spec["feature_root"]) / "work" / "object_ply" / f"{scene_key}_object_pruned_thr0.60.ply"
    _, gt_xyz = read_xyz(gt_ply)
    tol = reference_tolerance(gt_xyz, args.nn_tol_factor)
    metric_rows = []
    for stage, xyz, ply_path in [
        ("raw", raw_xyz, raw_ply_path),
        ("reproj_clean", raw_xyz[keep], clean_ply),
    ]:
        metric_rows.append(
            {
                "scene_key": scene_key,
                "category": spec["category"],
                "scene_id": spec["scene_id"],
                "split": spec.get("split", ""),
                "query": query,
                "stage": stage,
                "nn_tolerance": tol,
                "pred_ply": str(ply_path),
                "gt_ply": str(gt_ply),
                **metrics(xyz, gt_xyz, tol),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("frame_summaries", "scene_metrics", "scene_summaries"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "frame_summaries" / f"{scene_slug(scene_key)}_frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_name", "num_boxes", "sam_area", "projected_visible", "projected_inside"])
        writer.writeheader()
        writer.writerows(frame_rows)
    pd.DataFrame(metric_rows).to_csv(args.out_dir / "scene_metrics" / f"{scene_slug(scene_key)}_metrics.csv", index=False)
    summary = {
        "scene_key": scene_key,
        "category": spec["category"],
        "scene_id": spec["scene_id"],
        "split": spec.get("split", ""),
        "query": query,
        "qwen_frames_used": len(rows),
        "points_before": int(len(raw_xyz)),
        "points_after": int(keep.sum()),
        "points_removed": int((~keep).sum()),
        "keep_ratio": float(keep.mean()) if len(keep) else 0.0,
        "mean_visible_frames": float(visible.mean()) if len(visible) else 0.0,
        "mean_inside_frames": float(inside.mean()) if len(inside) else 0.0,
        "median_visible_frames": float(np.median(visible)) if len(visible) else 0.0,
        "median_inside_fraction": float(np.median(ratio)) if len(ratio) else 0.0,
        "cleaned_ply": str(clean_ply),
    }
    (args.out_dir / "scene_summaries" / f"{scene_slug(scene_key)}_cleanup.json").write_text(json.dumps(summary, indent=2))
    print(f"{scene_key}: reproj kept {summary['points_after']}/{summary['points_before']}")


def aggregate(out_dir: Path) -> None:
    metric_files = sorted((out_dir / "scene_metrics").glob("*_metrics.csv")) if (out_dir / "scene_metrics").exists() else []
    summary_files = sorted((out_dir / "scene_summaries").glob("*_cleanup.json")) if (out_dir / "scene_summaries").exists() else []
    if not metric_files:
        print("no scene metrics to aggregate")
        return
    per_scene = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True)
    per_scene.to_csv(out_dir / "per_scene_metrics.csv", index=False)
    cleanup = pd.DataFrame([json.loads(p.read_text()) for p in summary_files])
    cleanup.to_csv(out_dir / "cleanup_summary.csv", index=False)
    agg = per_scene.groupby(["stage"], as_index=False).agg(
        mean_iou=("iou", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_pred_count=("pred_count", "mean"),
        scenes=("scene_key", "count"),
    )
    agg.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    for group in ("split", "category"):
        per_scene.groupby([group, "stage"], as_index=False).agg(
            mean_iou=("iou", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_pred_count=("pred_count", "mean"),
            scenes=("scene_key", "count"),
        ).to_csv(out_dir / f"aggregate_by_{group}.csv", index=False)
    md = ["# Reprojection Cleanup Metrics", ""]
    for metric in ["mean_iou", "mean_precision", "mean_recall", "mean_f1", "mean_pred_count"]:
        md += [f"## {metric}", "", "```", agg.set_index("stage")[[metric]].round(4).to_string(), "```", ""]
    (out_dir / "summary_tables.md").write_text("\n".join(md))
    print(f"aggregated {len(metric_files)} scenes into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--scene_specs_csv", type=Path, required=True)
    parser.add_argument("--only_scene", default=None)
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--sam_checkpoint", default=str(REALM_ROOT / "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"))
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--min_visible", type=int, default=3)
    parser.add_argument("--inside_fraction", type=float, default=0.30)
    parser.add_argument("--nn_tol_factor", type=float, default=2.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        specs = load_specs(args.scene_specs_csv)
        if args.only_scene:
            specs = [s for s in specs if s["scene_key"] == args.only_scene]
        if not specs:
            raise RuntimeError("No scenes selected")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to(device).eval()
        predictor = SamPredictor(sam)
        for spec in specs:
            cleanup_one(spec, args, predictor, device)
    aggregate(args.out_dir)


if __name__ == "__main__":
    main()
