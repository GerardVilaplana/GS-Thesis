#!/usr/bin/env python3
"""Build SAM2 automatic multi-ID masks for a small HANDAL REALM pilot.

Outputs per scene:
  - resized RGB images and multi-ID object_mask PNGs ready for REALM
  - an RLE JSONL archive of propagated masks
  - quick overlay videos for visual QA
  - metadata.json with selected anchor/mask stats

Run with the samurai environment and SAM2 on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor


@dataclass(frozen=True)
class SceneSpec:
    scene_key: str
    query: str
    feature_root: Path


SCENES = [
    SceneSpec(
        "mugs__024004",
        "mug",
        Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features"),
    ),
    SceneSpec(
        "screwdrivers__010000",
        "screwdriver",
        Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features"),
    ),
    SceneSpec(
        "fixed_joint_pliers__007005",
        "joint plier",
        Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_new_categories_delta_v1_features"),
    ),
    SceneSpec(
        "locking_pliers__003001",
        "locking plier",
        Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_new_categories_delta_v1_features"),
    ),
    SceneSpec(
        "hammers__009009",
        "hammer",
        Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features"),
    ),
]


PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 23, 68],
        [0, 229, 255],
        [124, 77, 255],
        [0, 200, 83],
        [255, 179, 0],
        [255, 87, 34],
        [0, 188, 212],
        [139, 195, 74],
        [240, 98, 146],
        [255, 152, 0],
        [63, 81, 181],
        [38, 166, 154],
        [205, 220, 57],
        [233, 30, 99],
        [0, 150, 136],
        [255, 112, 67],
        [76, 175, 80],
        [103, 58, 183],
        [255, 235, 59],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(
            "/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/"
            "25_handal_sam2_auto_multimask_realm_query_5scenes_v1"
        ),
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=(
            "/home/gvilaplana/GS-Thesis/Supervision/external/samurai/sam2/"
            "checkpoints/sam2.1_hiera_large.pt"
        ),
    )
    parser.add_argument("--target_width", type=int, default=480)
    parser.add_argument("--max_masks", type=int, default=12)
    parser.add_argument("--min_mask_area", type=int, default=500)
    parser.add_argument("--min_object_overlap", type=float, default=0.08)
    parser.add_argument("--min_selected_object_coverage", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--scene_specs_csv", type=Path, default=None)
    parser.add_argument("--only_scene", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_scene_specs(path: Path | None) -> list[SceneSpec]:
    if path is None:
        return list(SCENES)
    specs: list[SceneSpec] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            specs.append(
                SceneSpec(
                    row["scene_key"],
                    row["query"],
                    Path(row["feature_root"]),
                )
            )
    return specs


def stem_number(stem: str) -> int:
    match = re.search(r"(\d+)", stem)
    return int(match.group(1)) if match else 0


def resize_rgb(image: Image.Image, target_width: int) -> Image.Image:
    image = image.convert("RGB")
    if image.width == target_width:
        return image
    target_height = round(image.height * target_width / image.width)
    return image.resize((target_width, target_height), Image.BICUBIC)


def resize_mask(mask: Image.Image, target_width: int) -> Image.Image:
    mask = mask.convert("L")
    if mask.width == target_width:
        return mask
    target_height = round(mask.height * target_width / mask.width)
    return mask.resize((target_width, target_height), Image.NEAREST)


def mask_path_for_source(source_scene: Path, stem: str) -> Path | None:
    mask_dir = source_scene / "mask"
    for name in (f"{stem}_000000.png", f"{int(stem_number(stem)):06d}_000000.png"):
        candidate = mask_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(mask_dir.glob(f"{stem}*.png"))
    return matches[0] if matches else None


def source_rgb_for_stem(source_scene: Path, stem: str) -> Path | None:
    rgb_dir = source_scene / "rgb"
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = rgb_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def ordered_scene_frames(model_root: Path, scene_root: Path) -> list[tuple[str, Path]]:
    cameras = sorted(load_json(model_root / "cameras.json"), key=lambda c: int(c["id"]))
    frames: list[tuple[str, Path]] = []
    for cam in cameras:
        stem = cam["img_name"]
        src = None
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = scene_root / "images" / f"{stem}{suffix}"
            if candidate.exists():
                src = candidate
                break
        if src is None:
            raise FileNotFoundError(f"Missing image for {stem} in {scene_root / 'images'}")
        frames.append((stem, src))
    return frames


def choose_anchor(frames: list[tuple[str, Path]], source_scene: Path, target_width: int) -> tuple[int, np.ndarray]:
    best_idx = len(frames) // 2
    best_area = -1
    best_mask = None
    for idx, (stem, _) in enumerate(frames):
        mask_path = mask_path_for_source(source_scene, stem)
        if mask_path is None:
            continue
        arr = np.array(resize_mask(Image.open(mask_path), target_width)) > 0
        area = int(arr.sum())
        if area > best_area:
            best_idx = idx
            best_area = area
            best_mask = arr
    if best_mask is None:
        stem, _ = frames[best_idx]
        raise FileNotFoundError(f"No HANDAL object mask found for anchor selection near {source_scene} ({stem})")
    return best_idx, best_mask


def rle_counts(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    counts: list[int] = []
    prev = 0
    run = 0
    for value in flat:
        value = int(value > 0)
        if value == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = value
    counts.append(run)
    return counts


def compressed_rle(mask: np.ndarray) -> str:
    payload = json.dumps(rle_counts(mask), separators=(",", ":")).encode("utf-8")
    return zlib.compress(payload, level=6).hex()


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


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
    for stem in stems:
        rgb = np.array(Image.open(frames_dir / f"{stem}.jpg").convert("RGB"))
        labels = np.array(Image.open(label_dir / f"{stem}.png"))
        overlay = overlay_label(rgb, labels)
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    writer.release()


def filter_anchor_masks(masks: list[dict], object_mask: np.ndarray, args: argparse.Namespace) -> list[dict]:
    kept = []
    object_area = max(int(object_mask.sum()), 1)
    for ann in masks:
        mask = ann["segmentation"].astype(bool)
        area = int(mask.sum())
        if area < args.min_mask_area:
            continue
        inter = int((mask & object_mask).sum())
        if inter == 0:
            continue
        overlap_mask = inter / max(area, 1)
        overlap_object = inter / object_area
        if overlap_mask < args.min_object_overlap and overlap_object < 0.005:
            continue
        ann = dict(ann)
        ann["segmentation"] = mask
        ann["object_overlap_mask"] = float(overlap_mask)
        ann["object_overlap_object"] = float(overlap_object)
        ann["intersection_with_handal_object"] = inter
        kept.append(ann)
    kept.sort(
        key=lambda a: (
            a["intersection_with_handal_object"],
            a.get("predicted_iou", 0.0),
            a.get("stability_score", 0.0),
        ),
        reverse=True,
    )
    return kept[: args.max_masks]


def selected_object_coverage(selected: list[dict], object_mask: np.ndarray) -> float:
    if not selected:
        return 0.0
    union = np.zeros_like(object_mask, dtype=bool)
    for ann in selected:
        union |= ann["segmentation"].astype(bool)
    return float((union & object_mask).sum() / max(int(object_mask.sum()), 1))


def fallback_handal_mask(object_mask: np.ndarray) -> dict:
    return {
        "segmentation": object_mask.astype(bool),
        "predicted_iou": 1.0,
        "stability_score": 1.0,
        "object_overlap_mask": 1.0,
        "object_overlap_object": 1.0,
        "intersection_with_handal_object": int(object_mask.sum()),
        "fallback_source": "handal_object_mask",
    }


def tensor_masks_to_numpy(video_res_masks: torch.Tensor | np.ndarray) -> np.ndarray:
    if hasattr(video_res_masks, "detach"):
        arr = video_res_masks.detach()
        if arr.ndim == 4:
            arr = arr[:, 0]
        return arr.gt(0.0).to(dtype=torch.uint8).cpu().numpy()
    arr = np.asarray(video_res_masks)
    if arr.ndim == 4:
        arr = arr[:, 0]
    return (arr > 0.0).astype(np.uint8)


def normalize_obj_ids(obj_ids) -> list[int]:
    if hasattr(obj_ids, "detach"):
        obj_ids = obj_ids.detach().cpu().numpy()
    return [int(x) for x in np.asarray(obj_ids).reshape(-1)]


def propagate_scene(
    video_predictor,
    tmp_video_dir: Path,
    frames: list[tuple[str, Path]],
    selected_masks: list[dict],
    anchor_idx: int,
    label_dir: Path,
    rle_path: Path,
) -> tuple[int, tuple[int, int]]:
    labels_by_idx: dict[int, np.ndarray] = {}
    mask_shape = None
    rle_f = rle_path.open("w")
    try:
        for reverse in (False, True):
            state = video_predictor.init_state(video_path=str(tmp_video_dir))
            for obj_id, ann in enumerate(selected_masks, start=1):
                video_predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=anchor_idx,
                    obj_id=obj_id,
                    mask=ann["segmentation"],
                )
            gen = video_predictor.propagate_in_video(
                state,
                start_frame_idx=anchor_idx,
                reverse=reverse,
            )
            direction = "backward" if reverse else "forward"
            expected = anchor_idx + 1 if reverse else len(frames) - anchor_idx
            for frame_idx, obj_ids, video_res_masks in tqdm(
                gen,
                total=expected,
                desc=f"SAM2 {direction}",
                unit="frame",
            ):
                frame_idx = int(frame_idx)
                ids = normalize_obj_ids(obj_ids)
                masks = tensor_masks_to_numpy(video_res_masks)
                if mask_shape is None:
                    mask_shape = masks.shape[1:]
                labels = labels_by_idx.get(frame_idx)
                if labels is None:
                    labels = np.zeros(mask_shape, dtype=np.uint8)
                    labels_by_idx[frame_idx] = labels
                stem = frames[frame_idx][0]
                for pos, obj_id in enumerate(ids):
                    mask = masks[pos].astype(bool)
                    labels[mask] = int(obj_id)
                    record = {
                        "frame": stem,
                        "frame_index": frame_idx,
                        "mask_id": int(obj_id),
                        "height": int(mask.shape[0]),
                        "width": int(mask.shape[1]),
                        "area": int(mask.sum()),
                        "bbox_xyxy": bbox_from_mask(mask),
                        "rle_order": "C",
                        "rle_zlib_hex": compressed_rle(mask),
                    }
                    rle_f.write(json.dumps(record, separators=(",", ":")) + "\n")
            if hasattr(video_predictor, "reset_state"):
                try:
                    video_predictor.reset_state(state)
                except Exception:
                    pass
            del state
            torch.cuda.empty_cache()
    finally:
        rle_f.close()

    if mask_shape is None:
        raise RuntimeError("SAM2 propagation did not return any masks")
    for idx, (stem, _) in enumerate(frames):
        labels = labels_by_idx.get(idx)
        if labels is None:
            labels = np.zeros(mask_shape, dtype=np.uint8)
        Image.fromarray(labels).save(label_dir / f"{stem}.png")
    return len(selected_masks), mask_shape


def prepare_scene_dirs(scene_out: Path, src_scene_root: Path, frames: list[tuple[str, Path]], target_width: int) -> tuple[Path, Path, Path]:
    realm_scene = scene_out / "realm_scene"
    image_dir = realm_scene / "images"
    label_dir = realm_scene / "object_mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    sparse_dst = realm_scene / "sparse"
    if sparse_dst.exists() or sparse_dst.is_symlink():
        sparse_dst.unlink() if sparse_dst.is_symlink() else shutil.rmtree(sparse_dst)
    os.symlink(os.path.relpath(src_scene_root / "sparse", realm_scene), sparse_dst)

    tmp_video_dir = scene_out / "sam2_video_frames"
    if tmp_video_dir.exists():
        shutil.rmtree(tmp_video_dir)
    tmp_video_dir.mkdir(parents=True)

    for idx, (stem, frame_path) in enumerate(frames):
        rgb = resize_rgb(Image.open(frame_path), target_width)
        rgb.save(image_dir / f"{stem}.jpg", quality=95, subsampling=0)
        rgb.save(tmp_video_dir / f"{idx:08d}.jpg", quality=95, subsampling=0)
    return realm_scene, label_dir, tmp_video_dir


def process_scene(spec: SceneSpec, args: argparse.Namespace, auto_generator, video_predictor) -> dict:
    src_scene_root = spec.feature_root / "work" / "3dgs_scenes" / spec.scene_key
    src_model_root = spec.feature_root / "work" / "3dgs_models" / spec.scene_key
    source_info = load_json(src_scene_root / "handal_source.json")
    source_scene = Path(source_info["source_scene"])
    frames = ordered_scene_frames(src_model_root, src_scene_root)

    scene_out = args.output_root / spec.scene_key
    if scene_out.exists():
        shutil.rmtree(scene_out)
    scene_out.mkdir(parents=True, exist_ok=True)

    realm_scene, label_dir, tmp_video_dir = prepare_scene_dirs(
        scene_out, src_scene_root, frames, args.target_width
    )
    anchor_idx, anchor_object_mask = choose_anchor(frames, source_scene, args.target_width)
    anchor_stem = frames[anchor_idx][0]
    anchor_rgb = np.array(Image.open(tmp_video_dir / f"{anchor_idx:08d}.jpg").convert("RGB"))

    print(f"\n[{spec.scene_key}] anchor={anchor_stem} index={anchor_idx}")
    auto_masks = auto_generator.generate(anchor_rgb)
    selected = filter_anchor_masks(auto_masks, anchor_object_mask, args)
    fallback_used = False
    auto_selected_object_coverage = selected_object_coverage(selected, anchor_object_mask)
    if not selected or auto_selected_object_coverage < args.min_selected_object_coverage:
        fallback_used = True
        selected = [fallback_handal_mask(anchor_object_mask)]
    print(
        f"[{spec.scene_key}] SAM2 masks generated={len(auto_masks)} "
        f"selected={len(selected)} fallback={fallback_used} "
        f"auto_object_coverage={auto_selected_object_coverage:.4f}"
    )

    rle_path = scene_out / "sam2_multimask_rle.jsonl"
    num_ids, mask_shape = propagate_scene(
        video_predictor,
        tmp_video_dir,
        frames,
        selected,
        anchor_idx,
        label_dir,
        rle_path,
    )
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
        "source_scene": str(source_scene),
        "source_model_root": str(src_model_root),
        "realm_scene": str(realm_scene),
        "config_path": str(config_path),
        "anchor_frame": anchor_stem,
        "anchor_index": int(anchor_idx),
        "num_frames": len(frames),
        "num_generated_auto_masks": len(auto_masks),
        "num_selected_mask_ids": num_ids,
        "fallback_used": fallback_used,
        "auto_selected_object_coverage": auto_selected_object_coverage,
        "min_selected_object_coverage": args.min_selected_object_coverage,
        "mask_shape_hw": [int(mask_shape[0]), int(mask_shape[1])],
        "selected_masks": [
            {
                "mask_id": idx,
                "area": int(ann["segmentation"].sum()),
                "bbox_xyxy": bbox_from_mask(ann["segmentation"]),
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "stability_score": float(ann.get("stability_score", 0.0)),
                "object_overlap_mask": float(ann["object_overlap_mask"]),
                "object_overlap_object": float(ann["object_overlap_object"]),
                "fallback_source": ann.get("fallback_source"),
            }
            for idx, ann in enumerate(selected, start=1)
        ],
    }
    (scene_out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    sam2_model = build_sam2(
        args.sam2_config,
        args.sam2_checkpoint,
        device=device,
        apply_postprocessing=False,
    )
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
        video_predictor = build_sam2_video_predictor(
            args.sam2_config,
            args.sam2_checkpoint,
            device=device,
            vos_optimized=False,
            apply_postprocessing=False,
        )
    except TypeError:
        video_predictor = build_sam2_video_predictor(
            args.sam2_config,
            args.sam2_checkpoint,
            device=device,
            vos_optimized=False,
        )

    specs = load_scene_specs(args.scene_specs_csv)
    if args.only_scene:
        specs = [spec for spec in specs if spec.scene_key == args.only_scene]
        if not specs:
            raise ValueError(f"Scene {args.only_scene!r} not found in {args.scene_specs_csv}")

    manifest = []
    for spec in specs:
        manifest.append(process_scene(spec, args, auto_generator, video_predictor))
        torch.cuda.empty_cache()
    (args.output_root / "sam2_multimask_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Outputs: {args.output_root}")


if __name__ == "__main__":
    main()
