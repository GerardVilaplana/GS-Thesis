#!/usr/bin/env python3
"""Ablate HANDAL reprojection cleanup strategies on exp35 raw object PLYs."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
OUT_BASE = BASE / "outputs" / "03_handle_generalization"
EXP35_ROOT = OUT_BASE / "35_exp30_15k_realm_reproj_5test_v1"
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

STRATEGIES = {
    "center_base": {"sample": "center", "point_rule": "frac", "point_frac": 1.0, "min_visible": 3, "frame_frac": 0.30},
    "center_strict": {"sample": "center", "point_rule": "frac", "point_frac": 1.0, "min_visible": 5, "frame_frac": 0.50},
    "axis6_any": {"sample": "axis6", "point_rule": "any", "point_frac": 0.0, "min_visible": 3, "frame_frac": 0.30},
    "axis6_vote50": {"sample": "axis6", "point_rule": "frac", "point_frac": 0.50, "min_visible": 3, "frame_frac": 0.30},
    "axis6_center_or_vote50": {"sample": "axis6", "point_rule": "center_or_frac", "point_frac": 0.50, "min_visible": 3, "frame_frac": 0.30},
    "ellipsoid12_any": {"sample": "ellipsoid12", "point_rule": "any", "point_frac": 0.0, "min_visible": 3, "frame_frac": 0.30},
    "ellipsoid12_vote75": {"sample": "ellipsoid12", "point_rule": "frac", "point_frac": 0.75, "min_visible": 3, "frame_frac": 0.30},
    "ellipsoid20_vote50": {"sample": "ellipsoid20", "point_rule": "frac", "point_frac": 0.50, "min_visible": 3, "frame_frac": 0.30},
    "ellipsoid20_vote75": {"sample": "ellipsoid20", "point_rule": "frac", "point_frac": 0.75, "min_visible": 3, "frame_frac": 0.30},
    "ellipsoid20_strict75": {"sample": "ellipsoid20", "point_rule": "frac", "point_frac": 0.75, "min_visible": 5, "frame_frac": 0.50},
}


def parse_frame_number(frame_name: str) -> int:
    return int(Path(frame_name).stem.split("_", 1)[1])


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


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
    raise FileNotFoundError(scene_key)


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


def load_rgb(path: Path, max_width: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_width > 0 and im.width > max_width:
        new_h = int(round(im.height * max_width / im.width))
        im = im.resize((max_width, new_h), Image.BICUBIC)
    return np.array(im)


def source_image_for_frame(scene_dir: Path, frame_dir: Path, cameras: list[dict], frame_name: str, max_width: int) -> Path:
    saved = frame_dir / frame_name
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
    raise FileNotFoundError(f"{scene_dir.name}/{frame_name}")


def make_sam_mask(predictor: SamPredictor, image_np: np.ndarray, boxes: list[list[float]], device: torch.device) -> np.ndarray:
    union = np.zeros(image_np.shape[:2], dtype=bool)
    for mask in sam_masks_for_boxes(predictor, image_np, boxes, device):
        union |= mask
    return union


def build_rotation_np(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def fibonacci_dirs(n: int) -> np.ndarray:
    dirs = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        dirs.append([math.cos(theta) * radius, y, math.sin(theta) * radius])
    return np.asarray(dirs, dtype=np.float64)


def sample_points(vertex: np.ndarray, xyz: np.ndarray, sample_kind: str) -> np.ndarray:
    center = xyz[:, None, :]
    if sample_kind == "center":
        return center
    scales = np.exp(np.vstack([vertex[f"scale_{i}"] for i in range(3)]).T.astype(np.float64))
    rots = np.vstack([vertex[f"rot_{i}"] for i in range(4)]).T.astype(np.float64)
    R = build_rotation_np(rots)
    if sample_kind == "axis6":
        dirs = np.asarray(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float64,
        )
    elif sample_kind == "ellipsoid12":
        dirs = fibonacci_dirs(12)
    elif sample_kind == "ellipsoid20":
        dirs = fibonacci_dirs(20)
    else:
        raise ValueError(sample_kind)
    local = dirs[None, :, :] * scales[:, None, :]
    offsets = np.einsum("nij,nkj->nki", R, local)
    return np.concatenate([center, xyz[:, None, :] + offsets], axis=1)


def project_points(points: np.ndarray, cam: dict, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = points.reshape(-1, 3)
    rot_c2w = np.asarray(cam["rotation"], dtype=np.float64)
    center = np.asarray(cam["position"], dtype=np.float64)
    cam_xyz = (flat - center) @ rot_c2w
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
    shape = points.shape[:2]
    return x.reshape(shape), y.reshape(shape), inside.reshape(shape)


def frame_hit(inside_pts: np.ndarray, valid_pts: np.ndarray, point_rule: str, point_frac: float) -> np.ndarray:
    center_inside = inside_pts[:, 0]
    if point_rule == "any":
        return inside_pts.any(axis=1)
    denom = np.maximum(valid_pts.sum(axis=1), 1)
    frac = inside_pts.sum(axis=1) / denom
    vote = frac >= point_frac
    if point_rule == "frac":
        return vote
    if point_rule == "center_or_frac":
        return center_inside | vote
    raise ValueError(point_rule)


def strategy_keep(xyz: np.ndarray, samples: np.ndarray, frames: list[dict], strategy: dict) -> tuple[np.ndarray, dict]:
    visible = np.zeros(len(xyz), dtype=np.int32)
    inside = np.zeros(len(xyz), dtype=np.int32)
    frame_records = []
    for frame in frames:
        px, py, valid_pts = project_points(samples, frame["camera"], frame["mask"].shape[1], frame["mask"].shape[0])
        center_valid = valid_pts[:, 0]
        visible[center_valid] += 1
        hit = np.zeros(len(xyz), dtype=bool)
        if valid_pts.any() and frame["mask"].any():
            yi = np.floor(np.clip(py, 0, frame["mask"].shape[0] - 1)).astype(np.int64)
            xi = np.floor(np.clip(px, 0, frame["mask"].shape[1] - 1)).astype(np.int64)
            inside_pts = valid_pts & frame["mask"][yi, xi]
            hit = frame_hit(inside_pts, valid_pts, strategy["point_rule"], strategy["point_frac"])
        inside[center_valid & hit] += 1
        frame_records.append(
            {
                "frame_name": frame["frame_name"],
                "visible_centers": int(center_valid.sum()),
                "hit_centers": int((center_valid & hit).sum()),
                "sam_area": int(frame["mask"].sum()),
            }
        )
    ratio = inside / np.maximum(visible, 1)
    min_visible = int(strategy["min_visible"])
    frame_frac = float(strategy["frame_frac"])
    keep = ((visible >= min_visible) & (ratio >= frame_frac)) | ((visible < min_visible) & (inside >= 1))
    summary = {
        "points_before": int(len(xyz)),
        "points_after": int(keep.sum()),
        "points_removed": int((~keep).sum()),
        "keep_ratio": float(keep.mean()) if len(keep) else 0.0,
        "mean_visible_frames": float(visible.mean()) if len(visible) else 0.0,
        "mean_inside_frames": float(inside.mean()) if len(inside) else 0.0,
        "median_visible_frames": float(np.median(visible)) if len(visible) else 0.0,
        "median_inside_fraction": float(np.median(ratio)) if len(ratio) else 0.0,
    }
    return keep, summary, frame_records


def reference_tolerance(gt_xyz: np.ndarray, factor: float) -> float:
    if len(gt_xyz) < 2:
        return 1e-6
    d, _ = cKDTree(gt_xyz).query(gt_xyz, k=2)
    return max(float(np.median(d[:, 1])) * factor, 1e-6)


def metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray, tol: float) -> dict:
    pred_n, gt_n = len(pred_xyz), len(gt_xyz)
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
    return {
        "pred_count": pred_n,
        "gt_count": gt_n,
        "pred_near_gt": inter_p,
        "gt_covered_by_pred": inter_g,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": min(inter_p, inter_g) / union if union else 0.0,
    }


def read_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_metrics_md(out_dir: Path, aggregate: pd.DataFrame, per_scene: pd.DataFrame, cleanup: pd.DataFrame) -> None:
    md = [
        "# Exp38 Reprojection Strategy Ablation",
        "",
        "Input is exp35 15k raw object pruning, before reprojection. Qwen summaries are reused from exp35; SAM masks are regenerated once per scene in memory and not saved as image masks.",
        "",
        "## Aggregate Metrics",
        "",
        aggregate.round(4).to_markdown(index=False),
        "",
        "## Best Per Metric",
        "",
    ]
    for metric in ["mean_f1", "mean_iou", "mean_precision", "mean_recall"]:
        row = aggregate.sort_values(metric, ascending=False).iloc[0]
        md.append(f"- `{metric}`: `{row['strategy']}` = `{row[metric]:.4f}`")
    md += [
        "",
        "## Per Scene Metrics",
        "",
        per_scene[["scene", "strategy", "pred_count", "precision", "recall", "f1", "iou"]]
        .round(4)
        .to_markdown(index=False),
        "",
        "## Cleanup Counts",
        "",
        cleanup[["scene", "strategy", "points_before", "points_after", "points_removed", "keep_ratio"]]
        .round(4)
        .to_markdown(index=False),
        "",
    ]
    (out_dir / "metrics.md").write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", type=Path, default=EXP35_ROOT)
    parser.add_argument("--out_dir", type=Path, default=OUT_BASE / "38_exp35_reprojection_strategy_ablation_v1")
    parser.add_argument("--sam_checkpoint", default=str(REALM_ROOT / "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"))
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--nn_tol_factor", type=float, default=2.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plys_dir = args.out_dir / "plys"
    frame_summary_dir = args.out_dir / "frame_summaries"
    plys_dir.mkdir(parents=True, exist_ok=True)
    frame_summary_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to(device).eval()
    predictor = SamPredictor(sam)

    per_scene_rows = []
    cleanup_rows = []
    mask_summary_rows = []
    manifest_rows = [r for r in read_manifest(args.base_root / "run_manifest.csv") if r.get("status") == "ok"]
    for item in manifest_rows:
        scene_key = item["scene_key"]
        query = item["query"]
        scene_dir = args.base_root / scene_key
        cameras = sorted(json.loads((scene_dir / "realm_model" / "cameras.json").read_text()), key=lambda c: int(c["id"]))
        frame_dir = Path(item["frame_dir"])
        rows = qwen_rows(scene_dir, query)

        frames = []
        with torch.no_grad():
            for row in rows:
                image_path = source_image_for_frame(scene_dir, frame_dir, cameras, row["frame_name"], args.max_width)
                image_np = load_rgb(image_path, args.max_width)
                boxes = qwen_boxes(row, image_np.shape[1], image_np.shape[0])
                mask = make_sam_mask(predictor, image_np, boxes, device)
                frames.append({"frame_name": row["frame_name"], "camera": cameras[parse_frame_number(row["frame_name"]) - 1], "mask": mask})
                mask_summary_rows.append(
                    {"scene_key": scene_key, "scene": SCENE_ABBR[scene_key], "frame_name": row["frame_name"], "num_boxes": len(boxes), "sam_area": int(mask.sum())}
                )

        raw_path = pred_ply_path(scene_dir, query)
        raw_ply, raw_xyz = read_xyz(raw_path)
        vertex = raw_ply["vertex"].data
        gt_path = feature_root_for(scene_key) / "work" / "object_ply" / f"{scene_key}_object_pruned_thr0.60.ply"
        _, gt_xyz = read_xyz(gt_path)
        tol = reference_tolerance(gt_xyz, args.nn_tol_factor)

        sample_cache = {}
        for strategy_name, strategy in STRATEGIES.items():
            sample_kind = strategy["sample"]
            if sample_kind not in sample_cache:
                sample_cache[sample_kind] = sample_points(vertex, raw_xyz, sample_kind)
            keep, summary, frame_records = strategy_keep(raw_xyz, sample_cache[sample_kind], frames, strategy)
            out_ply = plys_dir / f"{SCENE_ABBR[scene_key]}_{strategy_name}.ply"
            write_subset(raw_ply, keep, out_ply)
            per_scene_rows.append(
                {
                    "scene_key": scene_key,
                    "scene": SCENE_ABBR[scene_key],
                    "query": query,
                    "strategy": strategy_name,
                    "pred_ply": str(out_ply),
                    "gt_ply": str(gt_path),
                    "nn_tolerance": tol,
                    **metrics(raw_xyz[keep], gt_xyz, tol),
                }
            )
            cleanup_rows.append({"scene_key": scene_key, "scene": SCENE_ABBR[scene_key], "query": query, "strategy": strategy_name, **summary})
            with (frame_summary_dir / f"{SCENE_ABBR[scene_key]}_{strategy_name}_frames.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["frame_name", "visible_centers", "hit_centers", "sam_area"])
                writer.writeheader()
                writer.writerows(frame_records)
        print(f"{scene_key}: strategies={len(STRATEGIES)} raw_points={len(raw_xyz)} qwen_frames={len(frames)}", flush=True)

    per_scene = pd.DataFrame(per_scene_rows)
    cleanup = pd.DataFrame(cleanup_rows)
    mask_summary = pd.DataFrame(mask_summary_rows)
    aggregate = per_scene.groupby("strategy", as_index=False).agg(
        mean_iou=("iou", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_pred_count=("pred_count", "mean"),
        scenes=("scene_key", "count"),
    )
    aggregate = aggregate.sort_values(["mean_f1", "mean_iou"], ascending=False)
    per_scene.to_csv(args.out_dir / "per_scene_metrics.csv", index=False)
    cleanup.to_csv(args.out_dir / "cleanup_summary.csv", index=False)
    mask_summary.to_csv(args.out_dir / "sam_mask_summary.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    write_metrics_md(args.out_dir, aggregate, per_scene, cleanup)
    print(f"wrote {args.out_dir}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
