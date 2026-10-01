import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.utils.amg import (
    MaskData,
    batched_mask_to_box,
    calculate_stability_score,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    uncrop_masks,
)


FRAME_RE = re.compile(r"(?:frame|test)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


class OneMultimaskCandidateSAM2AutomaticMaskGenerator(SAM2AutomaticMaskGenerator):
    """SAM2 AMG variant that keeps one candidate mask for each grid point."""

    selection_strategy = "largest"

    def _process_batch(self, points, im_size, crop_box, orig_size, normalize=False):
        orig_h, orig_w = orig_size

        points = torch.as_tensor(points, dtype=torch.float32, device=self.predictor.device)
        in_points = self.predictor._transforms.transform_coords(points, normalize=normalize, orig_hw=im_size)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # For each grid point, SAM2 returns several valid granularities. Selecting
        # one candidate per point lets us test if oversegmentation comes from
        # preserving multiple multimask alternatives.
        if self.selection_strategy == "highest_iou":
            selected_idx = iou_preds.argmax(dim=1)
        elif self.selection_strategy == "largest":
            binary_for_area = masks > self.mask_threshold
            areas = binary_for_area.flatten(2).sum(dim=-1)
            selected_idx = areas.argmax(dim=1)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

        batch_idx = torch.arange(masks.shape[0], device=masks.device)
        masks = masks[batch_idx, selected_idx]
        iou_preds = iou_preds[batch_idx, selected_idx]
        low_res_masks = low_res_masks[batch_idx, selected_idx]

        data = MaskData(
            masks=masks,
            iou_preds=iou_preds,
            points=points,
            low_res_masks=low_res_masks,
        )
        del masks

        if not self.use_m2m:
            if self.pred_iou_thresh > 0.0:
                data.filter(data["iou_preds"] > self.pred_iou_thresh)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                data.filter(data["stability_score"] >= self.stability_score_thresh)
        else:
            in_points = self.predictor._transforms.transform_coords(
                data["points"], normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                data.filter(data["iou_preds"] > self.pred_iou_thresh)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                data.filter(data["stability_score"] >= self.stability_score_thresh)

        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data


class LargestMultimaskSAM2AutomaticMaskGenerator(OneMultimaskCandidateSAM2AutomaticMaskGenerator):
    """SAM2 AMG variant that keeps the largest candidate mask for each grid point."""

    selection_strategy = "largest"


class HighestIouMultimaskSAM2AutomaticMaskGenerator(OneMultimaskCandidateSAM2AutomaticMaskGenerator):
    """SAM2 AMG variant that keeps the highest predicted-IoU mask for each grid point."""

    selection_strategy = "highest_iou"


def sorted_image_paths(image_dir):
    paths = [p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    def key(path):
        match = FRAME_RE.match(path.name)
        if match:
            group = 0 if path.name.lower().startswith("frame_") else 1
            return (group, int(match.group(1)), path.name)
        return (2, path.name)

    return sorted(paths, key=key)


def color_for_index(index):
    rng = np.random.default_rng(index + 271828)
    return rng.integers(30, 255, size=3, dtype=np.uint8)


def draw_overlay(image_rgb, masks, output_path, max_masks=0, alpha=0.42):
    order = list(range(len(masks)))
    if max_masks > 0:
        order = order[:max_masks]

    overlay = image_rgb.astype(np.float32).copy()
    contour_canvas = image_rgb.copy()
    for mask_idx in order:
        mask = masks[mask_idx]["segmentation"].astype(bool)
        color = color_for_index(mask_idx).astype(np.float32)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_canvas, contours, -1, tuple(int(c) for c in color[::-1]), 2)

    out = np.concatenate([overlay.clip(0, 255).astype(np.uint8), contour_canvas], axis=1)
    Image.fromarray(out).save(output_path)


def draw_contact_sheet(image_rgb, masks, output_path, max_masks=120, tile_size=180):
    font = ImageFont.load_default()
    ordered = sorted(range(len(masks)), key=lambda i: int(masks[i].get("area", 0)), reverse=True)
    ordered = ordered[:max_masks]
    if not ordered:
        Image.fromarray(image_rgb).save(output_path)
        return

    cols = 8
    rows = int(np.ceil(len(ordered) / cols))
    sheet = Image.new("RGB", (cols * tile_size, rows * tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for tile_idx, mask_idx in enumerate(ordered):
        item = masks[mask_idx]
        mask = item["segmentation"].astype(bool)
        x, y, w, h = [int(v) for v in item["bbox"]]
        crop = image_rgb[y : y + h, x : x + w].copy()
        crop_mask = mask[y : y + h, x : x + w]
        if crop.size == 0 or crop_mask.size == 0:
            continue

        color = color_for_index(mask_idx)
        crop_overlay = crop.astype(np.float32)
        crop_overlay[crop_mask] = 0.35 * crop_overlay[crop_mask] + 0.65 * color.astype(np.float32)
        crop_overlay = crop_overlay.clip(0, 255).astype(np.uint8)

        tile = Image.fromarray(crop_overlay)
        tile.thumbnail((tile_size, tile_size - 24), Image.BICUBIC)
        tile_x = (tile_idx % cols) * tile_size
        tile_y = (tile_idx // cols) * tile_size
        paste_x = tile_x + (tile_size - tile.width) // 2
        paste_y = tile_y + 22 + (tile_size - 24 - tile.height) // 2
        sheet.paste(tile, (paste_x, paste_y))

        label = f"raw {mask_idx} | a={int(item.get('area', 0))}"
        draw.rectangle([tile_x, tile_y, tile_x + tile_size, tile_y + 20], fill=(0, 0, 0))
        draw.text((tile_x + 4, tile_y + 5), label, fill=(255, 255, 255), font=font)

    sheet.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Save raw whole-frame SAM2 AutomaticMaskGenerator outputs without custom filtering.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--keyframe_stride", type=int, default=24)
    parser.add_argument("--max_keyframes", type=int, default=6)
    parser.add_argument("--points_per_side", type=int, default=16)
    parser.add_argument("--points_per_batch", type=int, default=64)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--box_nms_thresh", type=float, default=0.7)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--crop_nms_thresh", type=float, default=0.7)
    parser.add_argument("--min_mask_region_area", type=int, default=0)
    parser.add_argument("--use_m2m", action="store_true")
    parser.add_argument("--single_mask_per_point", action="store_true")
    parser.add_argument("--largest_multimask_per_point", action="store_true")
    parser.add_argument("--highest_iou_multimask_per_point", action="store_true")
    parser.add_argument("--max_masks_overlay", type=int, default=0, help="0 means draw all returned masks.")
    parser.add_argument("--max_masks_contact_sheet", type=int, default=120)
    args = parser.parse_args()

    image_paths = sorted_image_paths(args.image_dir)
    selected = image_paths[:: max(args.keyframe_stride, 1)]
    if args.max_keyframes > 0:
        selected = selected[: args.max_keyframes]
    if not selected:
        raise RuntimeError(f"No images selected from {args.image_dir}")

    out = Path(args.output_dir)
    masks_dir = out / "raw_masks_npz"
    meta_dir = out / "metadata"
    overlay_dir = out / "overlays"
    sheet_dir = out / "mask_contact_sheets"
    for d in [masks_dir, meta_dir, overlay_dir, sheet_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.config, args.checkpoint, device=device)
    if args.largest_multimask_per_point and args.highest_iou_multimask_per_point:
        raise ValueError("Choose only one of --largest_multimask_per_point and --highest_iou_multimask_per_point")
    if args.largest_multimask_per_point:
        generator_cls = LargestMultimaskSAM2AutomaticMaskGenerator
    elif args.highest_iou_multimask_per_point:
        generator_cls = HighestIouMultimaskSAM2AutomaticMaskGenerator
    else:
        generator_cls = SAM2AutomaticMaskGenerator
    generator = generator_cls(
        model=model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
        use_m2m=args.use_m2m,
        multimask_output=not args.single_mask_per_point,
    )

    summary = {
        "settings": vars(args),
        "note": "Raw means no custom filtering after SAM2. SAM2 AutomaticMaskGenerator still applies its configured quality and NMS gates.",
        "device": device,
        "num_images_total": len(image_paths),
        "num_keyframes": len(selected),
        "frames": [],
    }

    for selected_idx, image_path in enumerate(selected):
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        raw_masks = generator.generate(image_rgb)
        raw_masks = sorted(raw_masks, key=lambda m: int(m.get("area", 0)), reverse=True)

        stem = image_path.stem
        npz_path = masks_dir / f"{stem}_raw_sam2_masks.npz"
        meta_path = meta_dir / f"{stem}_raw_sam2_masks.json"
        overlay_path = overlay_dir / f"{stem}_raw_sam2_overlay.png"
        sheet_path = sheet_dir / f"{stem}_raw_sam2_contact_sheet.jpg"

        masks = np.stack([m["segmentation"].astype(np.uint8) for m in raw_masks], axis=0) if raw_masks else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        bboxes = np.array([m.get("bbox", [0, 0, 0, 0]) for m in raw_masks], dtype=np.int32)
        areas = np.array([int(m.get("area", 0)) for m in raw_masks], dtype=np.int64)
        pred_ious = np.array([float(m.get("predicted_iou", 0.0)) for m in raw_masks], dtype=np.float32)
        stability = np.array([float(m.get("stability_score", 0.0)) for m in raw_masks], dtype=np.float32)
        np.savez_compressed(npz_path, masks=masks, bboxes_xywh=bboxes, areas=areas, predicted_ious=pred_ious, stability_scores=stability)

        draw_overlay(image_rgb, raw_masks, overlay_path, max_masks=args.max_masks_overlay)
        draw_contact_sheet(image_rgb, raw_masks, sheet_path, max_masks=args.max_masks_contact_sheet)

        frame_meta = {
            "selected_index": selected_idx,
            "frame_name": image_path.name,
            "image_shape": list(image_rgb.shape),
            "num_raw_masks": len(raw_masks),
            "npz": str(npz_path),
            "overlay": str(overlay_path),
            "contact_sheet": str(sheet_path),
            "mask_stats": {
                "area_min": int(areas.min()) if len(areas) else 0,
                "area_median": float(np.median(areas)) if len(areas) else 0.0,
                "area_max": int(areas.max()) if len(areas) else 0,
                "predicted_iou_min": float(pred_ious.min()) if len(pred_ious) else 0.0,
                "predicted_iou_median": float(np.median(pred_ious)) if len(pred_ious) else 0.0,
                "predicted_iou_max": float(pred_ious.max()) if len(pred_ious) else 0.0,
                "stability_min": float(stability.min()) if len(stability) else 0.0,
                "stability_median": float(np.median(stability)) if len(stability) else 0.0,
                "stability_max": float(stability.max()) if len(stability) else 0.0,
            },
            "masks": [
                {
                    "raw_sorted_index": i,
                    "area": int(areas[i]),
                    "bbox_xywh": [int(v) for v in bboxes[i].tolist()],
                    "predicted_iou": float(pred_ious[i]),
                    "stability_score": float(stability[i]),
                }
                for i in range(len(raw_masks))
            ],
        }
        meta_path.write_text(json.dumps(frame_meta, indent=2))
        summary["frames"].append(frame_meta)
        print(json.dumps({
            "frame": image_path.name,
            "num_raw_masks": len(raw_masks),
            "area_median": frame_meta["mask_stats"]["area_median"],
            "area_max": frame_meta["mask_stats"]["area_max"],
        }), flush=True)

    summary_path = out / "sam2_raw_automatic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
