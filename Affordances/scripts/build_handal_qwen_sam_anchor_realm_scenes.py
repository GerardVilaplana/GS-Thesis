#!/usr/bin/env python3
"""Build annotation-free Qwen+SAM2 anchor supervision for HANDAL REALM scenes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from build_handal_sam2_multimask_realm_scenes import (
    PALETTE,
    SceneSpec,
    bbox_from_mask,
    compressed_rle,
    load_json,
    load_scene_specs,
    normalize_obj_ids,
    ordered_scene_frames,
    prepare_scene_dirs,
    tensor_masks_to_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--scene_specs_csv", type=Path, required=True)
    parser.add_argument("--only_scene", type=str, default=None)
    parser.add_argument("--qwen_summary", type=Path, required=True)
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--sam2_checkpoint",
        default="/home/gvilaplana/GS-Thesis/Supervision/external/samurai/sam2/checkpoints/sam2.1_hiera_large.pt",
    )
    parser.add_argument("--target_width", type=int, default=960)
    parser.add_argument("--total_mask_ids", type=int, default=4)
    parser.add_argument("--min_mask_area", type=int, default=500)
    parser.add_argument("--max_target_overlap_for_auto", type=float, default=0.10)
    parser.add_argument("--max_auto_overlap", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def overlay_label(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colors = PALETTE[labels % len(PALETTE)]
    out = rgb.astype(np.float32)
    mask = labels > 0
    out[mask] = out[mask] * (1.0 - alpha) + colors[mask].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_video(scene_out: Path, frames_dir: Path, label_dir: Path, stems: list[str], fps: int) -> None:
    first = np.array(Image.open(frames_dir / f"{stems[0]}.jpg").convert("RGB"))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(scene_out / "sam2_multimask_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open overlay video writer")
    try:
        for stem in stems:
            rgb = np.array(Image.open(frames_dir / f"{stem}.jpg").convert("RGB"))
            labels = np.array(Image.open(label_dir / f"{stem}.png"))
            writer.write(cv2.cvtColor(overlay_label(rgb, labels), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def qwen_box_to_xyxy(item: dict, width: int, height: int) -> list[float] | None:
    box = item.get("bbox_2d", item.get("box", []))
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0.0, min(width - 1.0, x1))
    x2 = max(0.0, min(width - 1.0, x2))
    y1 = max(0.0, min(height - 1.0, y1))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_qwen_rows(summary_path: Path, query: str) -> list[dict]:
    summary = json.loads(summary_path.read_text())
    rows = summary.get("queries", {}).get(query, [])
    if not rows:
        raise RuntimeError(f"No Qwen rows for query {query!r} in {summary_path}")
    return rows


def choose_qwen_anchor_rows(rows: list[dict]) -> tuple[str, list[dict]]:
    with_boxes = [r for r in rows if int(r.get("num_boxes", 0)) > 0]
    if not with_boxes:
        raise RuntimeError("Qwen produced zero boxes on all anchor frames")
    counts = [int(r.get("num_boxes", 0)) for r in with_boxes]
    if max(counts) > 1:
        chosen = max(with_boxes, key=lambda r: int(r.get("num_boxes", 0)))
        return "single_multi_box_anchor", [chosen]
    return "multi_anchor_same_object_id", with_boxes


def sam2_masks_for_boxes(predictor: SAM2ImagePredictor, image_np: np.ndarray, boxes_xyxy: list[list[float]]) -> tuple[list[np.ndarray], list[float]]:
    if not boxes_xyxy:
        return [], []
    predictor.set_image(image_np)
    masks, ious, _ = predictor.predict(
        box=np.asarray(boxes_xyxy, dtype=np.float32),
        multimask_output=False,
        normalize_coords=True,
    )
    masks = masks[:, 0] if masks.ndim == 4 else masks
    ious = ious[:, 0] if ious.ndim == 2 else ious
    predictor.reset_predictor()
    return [m.astype(bool) for m in masks], [float(v) for v in np.asarray(ious).reshape(-1)]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / max(union, 1)


def select_auto_masks(auto_masks: list[dict], target_masks: list[np.ndarray], needed: int, args: argparse.Namespace) -> list[dict]:
    if needed <= 0:
        return []
    target_union = np.zeros_like(target_masks[0], dtype=bool) if target_masks else None
    for mask in target_masks:
        target_union |= mask
    selected: list[dict] = []
    for ann in sorted(auto_masks, key=lambda a: (a.get("stability_score", 0.0), a.get("predicted_iou", 0.0), a.get("area", 0)), reverse=True):
        mask = ann["segmentation"].astype(bool)
        if int(mask.sum()) < args.min_mask_area:
            continue
        if target_union is not None and (mask & target_union).sum() / max(int(mask.sum()), 1) > args.max_target_overlap_for_auto:
            continue
        if any(mask_iou(mask, prev["segmentation"]) > args.max_auto_overlap for prev in selected):
            continue
        item = dict(ann)
        item["segmentation"] = mask
        item["source"] = "sam2_auto_distractor"
        selected.append(item)
        if len(selected) >= needed:
            break
    return selected


def propagate_scene(video_predictor, tmp_video_dir: Path, frames: list[tuple[str, Path]], objects: list[dict], label_dir: Path, rle_path: Path) -> tuple[int, tuple[int, int]]:
    labels_by_idx: dict[int, np.ndarray] = {}
    prompt_indices = [p["frame_idx"] for obj in objects for p in obj["prompts"]]
    start_forward = min(prompt_indices)
    start_backward = max(prompt_indices)
    mask_shape = None
    with rle_path.open("w") as rle_f:
        for reverse, start_idx in ((False, start_forward), (True, start_backward)):
            state = video_predictor.init_state(video_path=str(tmp_video_dir))
            for obj in objects:
                for prompt in obj["prompts"]:
                    video_predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=int(prompt["frame_idx"]),
                        obj_id=int(obj["obj_id"]),
                        mask=prompt["mask"],
                    )
            gen = video_predictor.propagate_in_video(state, start_frame_idx=start_idx, reverse=reverse)
            expected = start_idx + 1 if reverse else len(frames) - start_idx
            for frame_idx, obj_ids, video_res_masks in tqdm(gen, total=expected, desc="SAM2 backward" if reverse else "SAM2 forward", unit="frame"):
                frame_idx = int(frame_idx)
                ids = normalize_obj_ids(obj_ids)
                masks = tensor_masks_to_numpy(video_res_masks)
                if mask_shape is None:
                    mask_shape = masks.shape[1:]
                labels = labels_by_idx.setdefault(frame_idx, np.zeros(mask_shape, dtype=np.uint8))
                stem = frames[frame_idx][0]
                for pos, obj_id in enumerate(ids):
                    mask = masks[pos].astype(bool)
                    labels[mask] = int(obj_id)
                    rle_f.write(json.dumps({
                        "frame": stem,
                        "frame_index": frame_idx,
                        "mask_id": int(obj_id),
                        "height": int(mask.shape[0]),
                        "width": int(mask.shape[1]),
                        "area": int(mask.sum()),
                        "bbox_xyxy": bbox_from_mask(mask),
                        "rle_order": "C",
                        "rle_zlib_hex": compressed_rle(mask),
                    }, separators=(",", ":")) + "\n")
            if hasattr(video_predictor, "reset_state"):
                try:
                    video_predictor.reset_state(state)
                except Exception:
                    pass
            del state
            torch.cuda.empty_cache()
    if mask_shape is None:
        raise RuntimeError("SAM2 propagation did not return any masks")
    for idx, (stem, _) in enumerate(frames):
        Image.fromarray(labels_by_idx.get(idx, np.zeros(mask_shape, dtype=np.uint8))).save(label_dir / f"{stem}.png")
    return len(objects), mask_shape


def process_scene(spec: SceneSpec, args: argparse.Namespace, image_predictor, auto_generator, video_predictor) -> dict:
    src_scene_root = spec.feature_root / "work" / "3dgs_scenes" / spec.scene_key
    src_model_root = spec.feature_root / "work" / "3dgs_models" / spec.scene_key
    source_info = load_json(src_scene_root / "handal_source.json")
    frames = ordered_scene_frames(src_model_root, src_scene_root)
    scene_out = args.output_root / spec.scene_key
    if scene_out.exists() and args.overwrite:
        shutil.rmtree(scene_out)
    scene_out.mkdir(parents=True, exist_ok=True)
    realm_scene, label_dir, tmp_video_dir = prepare_scene_dirs(scene_out, src_scene_root, frames, args.target_width)

    qwen_rows = load_qwen_rows(args.qwen_summary, spec.query)
    qwen_mode, selected_rows = choose_qwen_anchor_rows(qwen_rows)
    objects: list[dict] = []
    target_masks: list[np.ndarray] = []
    qwen_details = []
    next_obj_id = 1

    if qwen_mode == "multi_anchor_same_object_id":
        obj = {"obj_id": next_obj_id, "source": "qwen_sam_target", "prompts": []}
        for row in selected_rows:
            frame_idx = int(row["frame_name"].split("_")[1].split(".")[0]) - 1
            rgb = np.array(Image.open(tmp_video_dir / f"{frame_idx:08d}.jpg").convert("RGB"))
            boxes = [b for item in row.get("parsed", []) if (b := qwen_box_to_xyxy(item, rgb.shape[1], rgb.shape[0])) is not None]
            masks, ious = sam2_masks_for_boxes(image_predictor, rgb, boxes[:1])
            if masks:
                obj["prompts"].append({"frame_idx": frame_idx, "mask": masks[0]})
                target_masks.append(masks[0])
                qwen_details.append({"frame_name": row["frame_name"], "frame_index": frame_idx, "num_boxes": len(boxes), "used_box": boxes[0], "sam2_predicted_iou": ious[0]})
        if not obj["prompts"]:
            raise RuntimeError("No usable Qwen+SAM target prompts")
        objects.append(obj)
        next_obj_id += 1
    else:
        row = selected_rows[0]
        frame_idx = int(row["frame_name"].split("_")[1].split(".")[0]) - 1
        rgb = np.array(Image.open(tmp_video_dir / f"{frame_idx:08d}.jpg").convert("RGB"))
        boxes = [b for item in row.get("parsed", []) if (b := qwen_box_to_xyxy(item, rgb.shape[1], rgb.shape[0])) is not None]
        masks, ious = sam2_masks_for_boxes(image_predictor, rgb, boxes[: args.total_mask_ids])
        for mask, box, iou in zip(masks, boxes, ious):
            objects.append({"obj_id": next_obj_id, "source": "qwen_sam_target_ambiguous_single_anchor", "prompts": [{"frame_idx": frame_idx, "mask": mask}]})
            target_masks.append(mask)
            qwen_details.append({"frame_name": row["frame_name"], "frame_index": frame_idx, "num_boxes": len(boxes), "used_box": box, "sam2_predicted_iou": iou})
            next_obj_id += 1
            if next_obj_id > args.total_mask_ids:
                break

    auto_needed = max(0, args.total_mask_ids - len(objects))
    auto_anchor_idx = qwen_details[0]["frame_index"]
    auto_rgb = np.array(Image.open(tmp_video_dir / f"{auto_anchor_idx:08d}.jpg").convert("RGB"))
    auto_masks = auto_generator.generate(auto_rgb)
    for ann in select_auto_masks(auto_masks, target_masks, auto_needed, args):
        objects.append({"obj_id": next_obj_id, "source": "sam2_auto_distractor", "prompts": [{"frame_idx": auto_anchor_idx, "mask": ann["segmentation"]}], "auto_stats": ann})
        next_obj_id += 1

    if not objects:
        raise RuntimeError("No masks selected for propagation")
    rle_path = scene_out / "sam2_multimask_rle.jsonl"
    num_ids, mask_shape = propagate_scene(video_predictor, tmp_video_dir, frames, objects, label_dir, rle_path)
    make_video(scene_out, realm_scene / "images", label_dir, [s for s, _ in frames], args.fps)

    cfg = {
        "densify_until_iter": 1000,
        "num_classes": int(num_ids + 1),
        "reg3d_interval": 5,
        "reg3d_k": 5,
        "reg3d_lambda_val": 2,
        "reg3d_max_points": 200000,
        "reg3d_sample_size": 1000,
        "object_loss_start_iter": 1000,
    }
    config_path = scene_out / "realm_multimask_config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    metadata = {
        "scene_key": spec.scene_key,
        "query": spec.query,
        "feature_root": str(spec.feature_root),
        "source_scene": source_info.get("source_scene"),
        "source_model_root": str(src_model_root),
        "realm_scene": str(realm_scene),
        "config_path": str(config_path),
        "mask_source": "qwen_sam_3anchor_plus_sam2_auto",
        "qwen_summary": str(args.qwen_summary),
        "qwen_anchor_mode": qwen_mode,
        "num_frames": len(frames),
        "num_qwen_target_ids": sum(1 for obj in objects if str(obj["source"]).startswith("qwen_sam_target")),
        "num_auto_distractor_ids": sum(1 for obj in objects if obj["source"] == "sam2_auto_distractor"),
        "num_generated_auto_masks": len(auto_masks),
        "num_selected_mask_ids": num_ids,
        "mask_shape_hw": [int(mask_shape[0]), int(mask_shape[1])],
        "qwen_sam_prompts": qwen_details,
        "selected_masks": [
            {
                "mask_id": int(obj["obj_id"]),
                "source": obj["source"],
                "prompt_frames": [int(p["frame_idx"]) for p in obj["prompts"]],
                "areas": [int(p["mask"].sum()) for p in obj["prompts"]],
                "bboxes_xyxy": [bbox_from_mask(p["mask"]) for p in obj["prompts"]],
            }
            for obj in objects
        ],
    }
    (scene_out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"{spec.scene_key}: qwen_mode={qwen_mode} ids={num_ids} classes={num_ids + 1}", flush=True)
    return metadata


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device, apply_postprocessing=False)
    image_predictor = SAM2ImagePredictor(sam2_model)
    auto_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=16,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.82,
        box_nms_thresh=0.50,
        crop_n_layers=0,
        multimask_output=False,
        min_mask_region_area=250,
        output_mode="binary_mask",
    )
    try:
        video_predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, device=device, vos_optimized=False, apply_postprocessing=False)
    except TypeError:
        video_predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, device=device, vos_optimized=False)

    specs = load_scene_specs(args.scene_specs_csv)
    if args.only_scene:
        specs = [s for s in specs if s.scene_key == args.only_scene]
        if not specs:
            raise ValueError(f"Scene {args.only_scene!r} not found in {args.scene_specs_csv}")
    manifest = [process_scene(spec, args, image_predictor, auto_generator, video_predictor) for spec in specs]
    (args.output_root / "qwen_sam_anchor_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
