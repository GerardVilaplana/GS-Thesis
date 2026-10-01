#!/usr/bin/env python3
"""Clean own-scene1 per-REALM-ID object PLYs with Qwen+SAM reprojection masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from segment_anything import SamPredictor, sam_model_registry

BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
REALM_ROOT = Path("/home/gvilaplana/GS-Thesis/REALM-Code")
if str(REALM_ROOT) not in sys.path:
    sys.path.insert(0, str(REALM_ROOT))
if str(BASE / "scripts") not in sys.path:
    sys.path.insert(0, str(BASE / "scripts"))

from evaluate_exp38_reprojection_strategy_ablation import (  # noqa: E402
    STRATEGIES,
    project_points,
    sample_points,
    strategy_keep,
)
from scripts.export_multi_id_query_ply import pack_rgb, resize_rgb  # noqa: E402
from scripts.export_realm_query_ply import id2rgb  # noqa: E402
from scripts.make_qwen_sam_realm_stage_videos import (  # noqa: E402
    box_items_to_xyxy,
    parse_frame_number,
    sam_masks_for_boxes,
)

DEFAULT_OBJECT_ROOT = BASE / "outputs/06_own_scenes/scene_1_15k_realm_query_pruning"
DEFAULT_MODEL_ROOT = REALM_ROOT / "output/own_scenes/scene_1_realm_1600_objpred_15k"
DEFAULT_IMAGE_DIR = BASE / "data/own_scenes/scene_1_realm_1600/images_registered_frame_names_15k"


def read_xyz(path: Path) -> tuple[PlyData, np.ndarray]:
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    return ply, xyz


def write_subset(ply: PlyData, keep: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.array(ply["vertex"].data, copy=True)[keep]
    PlyData([PlyElement.describe(data, "vertex")], text=ply.text).write(out)


def rel_to_abs(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else BASE / path


def selected_id_records(per_id_summary: dict) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for query, block in per_id_summary.items():
        for part in block.get("parts", []):
            class_id = int(part["class_id"])
            rec = records.setdefault(
                class_id,
                {
                    "class_id": class_id,
                    "queries": [],
                    "raw_paths": [],
                    "num_gaussians": int(part.get("num_gaussians", 0)),
                },
            )
            rec["queries"].append(query)
            rec["raw_paths"].append(rel_to_abs(part["path"]))
    return records


def build_qwen_row_index(qwen_summary: dict) -> dict[str, dict[str, dict]]:
    return {
        query: {row["frame_name"]: row for row in rows}
        for query, rows in qwen_summary.get("queries", {}).items()
    }


def class_mask(pred_rgb: np.ndarray, class_id: int, num_classes: int) -> np.ndarray:
    color = id2rgb(class_id, num_classes).astype(np.uint8)
    code = int((int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2]))
    return pack_rgb(pred_rgb) == code


def best_sam_mask_for_id(
    predictor: SamPredictor,
    row: dict,
    image_path: Path,
    pred_path: Path,
    class_id: int,
    num_classes: int,
    max_width: int,
    device: torch.device,
) -> dict | None:
    image_np, scale = resize_rgb(image_path, max_width, Image.BICUBIC)
    pred_rgb, _ = resize_rgb(pred_path, max_width, Image.NEAREST)
    realm_mask = class_mask(pred_rgb, class_id, num_classes)
    realm_area = int(realm_mask.sum())
    if realm_area == 0:
        return None
    boxes = box_items_to_xyxy(row.get("parsed", []), scale, image_np.shape[1], image_np.shape[0])
    masks = sam_masks_for_boxes(predictor, image_np, boxes, device)
    best = None
    for box_idx, mask in enumerate(masks):
        sam_area = int(mask.sum())
        if sam_area == 0:
            continue
        inter = int((realm_mask & mask).sum())
        union = realm_area + sam_area - inter
        iou = inter / max(union, 1)
        if best is None or iou > best["iou"]:
            best = {"mask": mask, "iou": float(iou), "box_idx": int(box_idx), "sam_area": sam_area, "realm_area": realm_area}
    return best


def collect_frames_for_id(
    class_id: int,
    queries: list[str],
    query_report: dict,
    row_index: dict[str, dict[str, dict]],
    cameras: list[dict],
    predictor: SamPredictor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    by_frame: dict[str, dict] = {}
    diagnostics = []
    for query in sorted(set(queries)):
        q_block = query_report.get("queries", {}).get(query, {})
        for frame in q_block.get("frames", []):
            frame_name = frame["frame_name"]
            row = row_index.get(query, {}).get(frame_name)
            if row is None or int(row.get("num_boxes", 0)) <= 0:
                continue
            render_idx = parse_frame_number(frame_name) - 1
            pred_path = args.model_root / f"train/ours_{args.iteration}/objects_pred" / f"{render_idx:05d}.png"
            image_path = args.image_dir / frame_name
            if not pred_path.exists() or not image_path.exists():
                continue
            best = best_sam_mask_for_id(
                predictor,
                row,
                image_path,
                pred_path,
                class_id,
                args.num_classes,
                args.max_width,
                device,
            )
            if best is None:
                continue
            diagnostics.append(
                {
                    "class_id": class_id,
                    "query": query,
                    "frame_name": frame_name,
                    "box_idx": best["box_idx"],
                    "best_sam_realm_iou": best["iou"],
                    "sam_area": best["sam_area"],
                    "realm_area": best["realm_area"],
                }
            )
            if best["iou"] < args.min_sam_realm_iou:
                continue
            old = by_frame.get(frame_name)
            if old is None or best["iou"] > old["sam_realm_iou"]:
                by_frame[frame_name] = {
                    "frame_name": frame_name,
                    "camera": cameras[render_idx],
                    "mask": best["mask"],
                    "query": query,
                    "sam_realm_iou": best["iou"],
                }
    frames = [by_frame[k] for k in sorted(by_frame, key=parse_frame_number)]
    return frames, diagnostics


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--iteration", type=int, default=15000)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--strategy", default="ellipsoid20_strict75", choices=sorted(STRATEGIES))
    parser.add_argument("--sam_checkpoint", type=Path, default=REALM_ROOT / "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth")
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--num_classes", type=int, default=256)
    parser.add_argument("--min_sam_realm_iou", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.object_root / "final_ply/per_id_reprojection_ellipsoid20_strict75"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_id_summary = json.loads((args.object_root / "final_ply/per_id_truecolor/per_id_truecolor_summary.json").read_text())
    query_report = json.loads((args.object_root / "exported_ply/multi_id_query_report.json").read_text())
    qwen_summary = json.loads((args.object_root / "qwen/qwen_vl_box_summary.json").read_text())
    row_index = build_qwen_row_index(qwen_summary)
    cameras = sorted(json.loads((args.model_root / "cameras.json").read_text()), key=lambda c: int(c["id"]))

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint)).to(device).eval()
    predictor = SamPredictor(sam)
    strategy = STRATEGIES[args.strategy]

    records = selected_id_records(per_id_summary)
    summary_rows = []
    frame_rows = []
    detailed_summary = {"settings": vars(args), "objects": {}}
    for class_id, rec in sorted(records.items()):
        raw_path = sorted(set(rec["raw_paths"]), key=lambda p: (len(str(p)), str(p)))[0]
        raw_ply, raw_xyz = read_xyz(raw_path)
        frames, diagnostics = collect_frames_for_id(
            class_id,
            rec["queries"],
            query_report,
            row_index,
            cameras,
            predictor,
            args,
            device,
        )
        samples = sample_points(raw_ply["vertex"].data, raw_xyz, strategy["sample"])
        keep, cleanup, cleanup_frame_records = strategy_keep(raw_xyz, samples, frames, strategy)
        alias = "_".join(sorted({q.replace(" ", "_") for q in rec["queries"]}))
        out_ply = args.out_dir / f"realm_id{class_id:03d}_{alias}_{args.strategy}_truecolor_3dgs.ply"
        write_subset(raw_ply, keep, out_ply)

        row = {
            "class_id": class_id,
            "queries": ";".join(sorted(set(rec["queries"]))),
            "raw_ply": str(raw_path),
            "clean_ply": str(out_ply),
            "matched_frames_used": len(frames),
            "mean_sam_realm_iou": float(np.mean([f["sam_realm_iou"] for f in frames])) if frames else 0.0,
            **cleanup,
        }
        summary_rows.append(row)
        for fr in cleanup_frame_records:
            frame_rows.append({"class_id": class_id, **fr})
        detailed_summary["objects"][str(class_id)] = {**row, "diagnostics": diagnostics}
        print(
            f"realm_id{class_id:03d}: {cleanup['points_before']} -> {cleanup['points_after']} "
            f"frames={len(frames)}",
            flush=True,
        )

    save_csv(args.out_dir / "cleanup_summary.csv", summary_rows)
    save_csv(args.out_dir / "frame_summary.csv", frame_rows)
    (args.out_dir / "per_id_reprojection_summary.json").write_text(json.dumps(detailed_summary, indent=2, default=str))
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
