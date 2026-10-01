#!/usr/bin/env python3
"""Clean exp30 selected-object PLYs by reprojection into Qwen+SAM 2D masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
OUT_BASE = BASE / "outputs" / "03_handle_generalization"
EXP30_ROOT = OUT_BASE / "30_handal_qwen_sam_best3conf_10cand_plus_auto_realm_query_5test_v1"
EXP32_ROOT = OUT_BASE / "32_exp30_reprojection_cleanup_v1"
FEATURE_ROOTS = [
    BASE / "data" / "handal_handle_generalization_features",
    BASE / "data" / "handal_new_categories_delta_v1_features",
]
SCENE_ABBR = {
    "mugs__024004": "mug",
    "screwdrivers__010000": "scrw",
    "hammers__031010": "hamm",
    "combinational_wrenches__002002": "comb",
    "strainers__032005": "strn",
}


def parse_frame_number(frame_name: str) -> int:
    stem = Path(frame_name).stem
    if not stem.startswith("frame_"):
        raise ValueError(f"Unexpected frame name: {frame_name}")
    return int(stem.split("_", 1)[1])


def read_xyz(path: Path) -> tuple[PlyData, np.ndarray]:
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    return ply, xyz


def write_subset(ply: PlyData, keep: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.array(ply["vertex"].data, copy=True)[keep]
    PlyData([PlyElement.describe(data, "vertex")], text=ply.text).write(out)


def feature_root_for(scene_key: str) -> Path:
    for root in FEATURE_ROOTS:
        if (root / "work" / "object_ply" / f"{scene_key}_object_pruned_thr0.60.ply").exists():
            return root
    raise FileNotFoundError(f"No HANDAL reference object PLY for {scene_key}")


def scene_query(scene_dir: Path) -> str:
    return json.loads((scene_dir / "metadata.json").read_text())["query"]


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


def source_image_for_frame(scene_dir: Path, cameras: list[dict], frame_name: str) -> Path:
    meta = json.loads((scene_dir / "metadata.json").read_text())
    frame_dir = EXP30_ROOT / "frame_images" / scene_dir.name
    saved = frame_dir / frame_name
    if saved.exists():
        return saved

    idx = parse_frame_number(frame_name) - 1
    stem = cameras[idx]["img_name"]
    rgb_dir = Path(meta["source_scene"]) / "rgb"
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = rgb_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing source RGB for {scene_dir.name} frame {frame_name} stem {stem}")


def make_sam_mask(predictor: SamPredictor, image_np: np.ndarray, boxes: list[list[float]], device: torch.device) -> np.ndarray:
    union = np.zeros(image_np.shape[:2], dtype=bool)
    for mask in sam_masks_for_boxes(predictor, image_np, boxes, device):
        union |= mask
    return union


def project_points(points: np.ndarray, cam: dict, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot_c2w = np.asarray(cam["rotation"], dtype=np.float64)
    center = np.asarray(cam["position"], dtype=np.float64)
    cam_xyz = (points - center) @ rot_c2w
    z = cam_xyz[:, 2]
    sx = out_w / float(cam["width"])
    sy = out_h / float(cam["height"])
    fx = float(cam["fx"]) * sx
    fy = float(cam["fy"]) * sy
    cx = out_w * 0.5
    cy = out_h * 0.5
    x = fx * (cam_xyz[:, 0] / z) + cx
    y = fy * (cam_xyz[:, 1] / z) + cy
    inside = (z > 1e-8) & (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
    return x, y, inside


def cleanup_keep_mask(
    xyz: np.ndarray,
    scene_dir: Path,
    query: str,
    predictor: SamPredictor,
    device: torch.device,
    min_visible: int,
    inside_fraction: float,
    max_width: int,
) -> tuple[np.ndarray, dict, list[dict]]:
    cameras = sorted(json.loads((scene_dir / "realm_model" / "cameras.json").read_text()), key=lambda c: int(c["id"]))
    rows = qwen_rows(scene_dir, query)
    if not rows:
        raise RuntimeError(f"No usable Qwen rows for {scene_dir.name}/{query}")

    visible = np.zeros(len(xyz), dtype=np.int32)
    inside = np.zeros(len(xyz), dtype=np.int32)
    frame_rows = []

    with torch.no_grad():
        for row in rows:
            frame_num = parse_frame_number(row["frame_name"])
            cam = cameras[frame_num - 1]
            image_np = load_resized_rgb(source_image_for_frame(scene_dir, cameras, row["frame_name"]), max_width)
            boxes = qwen_boxes(row, image_np.shape[1], image_np.shape[0])
            sam_union = make_sam_mask(predictor, image_np, boxes, device)
            px, py, valid = project_points(xyz, cam, image_np.shape[1], image_np.shape[0])
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
            frame_rows.append({
                "frame_name": row["frame_name"],
                "num_boxes": len(boxes),
                "sam_area": int(sam_union.sum()),
                "projected_visible": int(valid.sum()),
                "projected_inside": frame_inside,
            })

    ratio = np.divide(inside, np.maximum(visible, 1), dtype=np.float64)
    keep = ((visible >= min_visible) & (ratio >= inside_fraction)) | ((visible < min_visible) & (inside >= 1))
    summary = {
        "qwen_frames_used": len(rows),
        "points_before": int(len(xyz)),
        "points_after": int(keep.sum()),
        "points_removed": int((~keep).sum()),
        "keep_ratio": float(keep.mean()) if len(keep) else 0.0,
        "mean_visible_frames": float(visible.mean()) if len(visible) else 0.0,
        "mean_inside_frames": float(inside.mean()) if len(inside) else 0.0,
        "median_visible_frames": float(np.median(visible)) if len(visible) else 0.0,
        "median_inside_fraction": float(np.median(ratio)) if len(ratio) else 0.0,
    }
    return keep, summary, frame_rows


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


def write_summary_tables(out_dir: Path, per_scene: pd.DataFrame, agg: pd.DataFrame) -> None:
    md = [
        "# Exp30 Reprojection Cleanup",
        "",
        "Cleanup is applied after REALM object pruning. Each Gaussian center is projected into saved Qwen+SAM query masks.",
        "Metrics use the same nearest-neighbor spatial overlap as exp31.",
        "",
    ]
    for metric in ["mean_iou", "mean_precision", "mean_recall", "mean_f1", "mean_pred_count"]:
        table = agg.pivot(index="variant", columns="stage", values=metric).round(4)
        md += [f"## {metric}", "", "```", table.to_string(), "```", ""]
    md += ["## per_scene", "", "```", per_scene[["scene", "stage", "iou", "precision", "recall", "f1", "pred_count"]].round(4).to_string(index=False), "```", ""]
    (out_dir / "summary_tables.md").write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp30_root", type=Path, default=EXP30_ROOT)
    parser.add_argument("--out_dir", type=Path, default=EXP32_ROOT)
    parser.add_argument("--sam_checkpoint", default=str(REALM_ROOT / "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"))
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--min_visible", type=int, default=3)
    parser.add_argument("--inside_fraction", type=float, default=0.30)
    parser.add_argument("--nn_tol_factor", type=float, default=2.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = args.out_dir / "plys"
    ply_dir.mkdir(parents=True, exist_ok=True)
    frame_summary_dir = args.out_dir / "frame_summaries"
    frame_summary_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to(device).eval()
    predictor = SamPredictor(sam)

    per_scene_rows = []
    cleanup_rows = []
    for scene_key in sorted(SCENE_ABBR):
        scene_dir = args.exp30_root / scene_key
        query = scene_query(scene_dir)
        raw_ply_path = pred_ply_path(scene_dir, query)
        raw_ply, raw_xyz = read_xyz(raw_ply_path)
        keep, cleanup_summary, frame_rows = cleanup_keep_mask(
            raw_xyz,
            scene_dir,
            query,
            predictor,
            device,
            args.min_visible,
            args.inside_fraction,
            args.max_width,
        )
        short = f"{SCENE_ABBR[scene_key]}_b3_reproj.ply"
        clean_ply_path = ply_dir / short
        write_subset(raw_ply, keep, clean_ply_path)

        feature_root = feature_root_for(scene_key)
        gt_ply = feature_root / "work" / "object_ply" / f"{scene_key}_object_pruned_thr0.60.ply"
        _, gt_xyz = read_xyz(gt_ply)
        tol = reference_tolerance(gt_xyz, args.nn_tol_factor)
        variants = [
            ("raw_exp30", raw_xyz, raw_ply_path),
            ("reproj_clean", raw_xyz[keep], clean_ply_path),
        ]
        for stage, xyz, ply_path in variants:
            per_scene_rows.append({
                "variant": "30_best3",
                "stage": stage,
                "scene_key": scene_key,
                "scene": SCENE_ABBR[scene_key],
                "query": query,
                "nn_tolerance": tol,
                "pred_ply": str(ply_path),
                "gt_ply": str(gt_ply),
                **metrics(xyz, gt_xyz, tol),
            })

        cleanup_rows.append({
            "scene_key": scene_key,
            "scene": SCENE_ABBR[scene_key],
            "query": query,
            "cleaned_ply": str(clean_ply_path),
            **cleanup_summary,
        })
        with (frame_summary_dir / f"{SCENE_ABBR[scene_key]}_frames.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_name", "num_boxes", "sam_area", "projected_visible", "projected_inside"])
            writer.writeheader()
            writer.writerows(frame_rows)
        print(f"{scene_key}: kept {cleanup_summary['points_after']}/{cleanup_summary['points_before']} -> {clean_ply_path}", flush=True)

    per_scene = pd.DataFrame(per_scene_rows)
    cleanup = pd.DataFrame(cleanup_rows)
    agg = per_scene.groupby(["variant", "stage"], as_index=False).agg(
        mean_iou=("iou", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_pred_count=("pred_count", "mean"),
        scenes=("scene_key", "count"),
    )
    per_scene.to_csv(args.out_dir / "per_scene_metrics.csv", index=False)
    cleanup.to_csv(args.out_dir / "cleanup_summary.csv", index=False)
    agg.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    write_summary_tables(args.out_dir, per_scene, agg)
    print(f"wrote {args.out_dir}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
