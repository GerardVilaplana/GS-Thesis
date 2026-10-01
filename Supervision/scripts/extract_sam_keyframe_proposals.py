import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


FRAME_RE = re.compile(r"(?:frame|test)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def sorted_image_paths(image_dir):
    paths = [p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    def key(path):
        match = FRAME_RE.match(path.name)
        if match:
            group = 0 if path.name.lower().startswith("frame_") else 1
            return (group, int(match.group(1)), path.name)
        return (2, path.name)

    return sorted(paths, key=key)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def containment(smaller, larger):
    inter = np.logical_and(smaller, larger).sum()
    return float(inter / max(smaller.sum(), 1))


def filter_and_rank_masks(raw_masks, image_area, args):
    min_area = int(args.min_area_frac * image_area)
    max_area = int(args.max_area_frac * image_area)
    candidates = []

    for idx, item in enumerate(raw_masks):
        area = int(item.get("area", int(item["segmentation"].sum())))
        if area < min_area or area > max_area:
            continue
        score = float(item.get("predicted_iou", 0.0)) * float(item.get("stability_score", 0.0))
        candidates.append({
            "raw_index": idx,
            "segmentation": item["segmentation"].astype(bool),
            "area": area,
            "bbox": [int(v) for v in item.get("bbox", [0, 0, 0, 0])],
            "predicted_iou": float(item.get("predicted_iou", 0.0)),
            "stability_score": float(item.get("stability_score", 0.0)),
            "score": score,
        })

    candidates.sort(key=lambda x: (x["score"], x["area"]), reverse=True)
    kept = []
    for cand in candidates:
        duplicate = False
        for prev in kept:
            iou = mask_iou(cand["segmentation"], prev["segmentation"])
            if iou >= args.nms_iou:
                duplicate = True
                break

            cand_area = cand["area"]
            prev_area = prev["area"]
            if cand_area <= prev_area:
                contained = containment(cand["segmentation"], prev["segmentation"])
                if contained >= args.containment:
                    duplicate = True
                    break
            else:
                contained = containment(prev["segmentation"], cand["segmentation"])
                if contained >= args.containment and cand["score"] <= prev["score"] * args.containment_replace_margin:
                    duplicate = True
                    break

        if duplicate:
            continue
        kept.append(cand)
        if len(kept) >= args.max_masks_per_keyframe:
            break

    return kept, {
        "raw_masks": len(raw_masks),
        "after_area_filter": len(candidates),
        "kept": len(kept),
        "min_area_px": min_area,
        "max_area_px": max_area,
    }


def color_for_index(index):
    rng = np.random.default_rng(index + 12345)
    return rng.integers(40, 255, size=3, dtype=np.uint8)


def save_overlay(image_rgb, masks, output_path, alpha=0.45):
    overlay = image_rgb.copy().astype(np.float32)
    contours_canvas = image_rgb.copy()
    for idx, item in enumerate(masks):
        mask = item["segmentation"]
        color = color_for_index(idx).astype(np.float32)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_canvas, contours, -1, tuple(int(c) for c in color[::-1]), 2)

    out = np.concatenate([overlay.clip(0, 255).astype(np.uint8), contours_canvas], axis=1)
    Image.fromarray(out).save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Extract conservative SAM automatic-mask proposals on sparse keyframes.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--model_type", default="vit_h")
    parser.add_argument("--keyframe_stride", type=int, default=10)
    parser.add_argument("--max_keyframes", type=int, default=0)
    parser.add_argument("--max_masks_per_keyframe", type=int, default=30)
    parser.add_argument("--min_area_frac", type=float, default=0.0005)
    parser.add_argument("--max_area_frac", type=float, default=0.50)
    parser.add_argument("--nms_iou", type=float, default=0.80)
    parser.add_argument("--containment", type=float, default=0.90)
    parser.add_argument("--containment_replace_margin", type=float, default=1.05)
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.0)
    parser.add_argument("--stability_score_thresh", type=float, default=0.0)
    parser.add_argument("--crop_n_layers", type=int, default=1)
    args = parser.parse_args()

    image_paths = sorted_image_paths(args.image_dir)
    keyframes = image_paths[:: max(args.keyframe_stride, 1)]
    if args.max_keyframes > 0:
        keyframes = keyframes[: args.max_keyframes]
    if not keyframes:
        raise RuntimeError(f"No keyframes selected from {args.image_dir}")

    out = Path(args.output_dir)
    masks_dir = out / "masks_npz"
    meta_dir = out / "metadata"
    overlay_dir = out / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).to(device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
    )

    summary = {
        "settings": vars(args),
        "device": device,
        "num_images_total": len(image_paths),
        "num_keyframes": len(keyframes),
        "keyframes": [],
    }

    for key_idx, image_path in enumerate(keyframes):
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        raw_masks = generator.generate(image_rgb)
        kept, stats = filter_and_rank_masks(raw_masks, image_rgb.shape[0] * image_rgb.shape[1], args)

        masks = np.stack([m["segmentation"] for m in kept], axis=0) if kept else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
        stem = image_path.stem
        npz_path = masks_dir / f"{stem}_sam_proposals.npz"
        meta_path = meta_dir / f"{stem}_sam_proposals.json"
        overlay_path = overlay_dir / f"{stem}_sam_proposals_overlay.png"

        np.savez_compressed(npz_path, masks=masks.astype(np.uint8))
        mask_meta = []
        for new_id, item in enumerate(kept, start=1):
            mask_meta.append({
                "proposal_id": new_id,
                "raw_index": item["raw_index"],
                "area": item["area"],
                "bbox_xywh": item["bbox"],
                "predicted_iou": item["predicted_iou"],
                "stability_score": item["stability_score"],
                "rank_score": item["score"],
            })
        meta = {
            "frame_index_in_selected_keyframes": key_idx,
            "frame_name": image_path.name,
            "image_shape": list(image_rgb.shape),
            "npz": str(npz_path),
            "overlay": str(overlay_path),
            "stats": stats,
            "masks": mask_meta,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        save_overlay(image_rgb, kept, overlay_path)
        summary["keyframes"].append(meta)
        print(json.dumps({
            "frame": image_path.name,
            "raw_masks": stats["raw_masks"],
            "after_area_filter": stats["after_area_filter"],
            "kept": stats["kept"],
        }), flush=True)

    (out / "sam_keyframe_proposals_summary.json").write_text(json.dumps(summary, indent=2))
    print(out / "sam_keyframe_proposals_summary.json", flush=True)


if __name__ == "__main__":
    main()
