import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO


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


def color_for_index(index):
    rng = np.random.default_rng(index + 424242)
    return tuple(int(v) for v in rng.integers(30, 255, size=3))


def draw_boxes(image_rgb, detections, output_path):
    im = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for i, det in enumerate(detections):
        color = color_for_index(i)
        x1, y1, x2, y2 = det["box_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        bbox = draw.textbbox((x1 + 3, max(0, y1 - 16)), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x1 + 3, max(0, y1 - 16)), label, fill=color, font=font)
    im.save(output_path)


def draw_masks(image_rgb, masks, detections, output_path, alpha=0.45):
    overlay = image_rgb.astype(np.float32).copy()
    contour_canvas = image_rgb.copy()
    im_h, im_w = image_rgb.shape[:2]
    font = ImageFont.load_default()

    for i, (mask, det) in enumerate(zip(masks, detections)):
        color = np.array(color_for_index(i), dtype=np.float32)
        mask = mask.astype(bool)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_canvas, contours, -1, tuple(int(c) for c in color[::-1]), 2)

        x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
        x1 = max(0, min(im_w - 1, x1))
        y1 = max(0, min(im_h - 1, y1))
        label = f"{i}: {det['class_name']} {det['confidence']:.2f}"
        pil_tmp = Image.fromarray(contour_canvas)
        draw = ImageDraw.Draw(pil_tmp)
        bbox = draw.textbbox((x1 + 3, max(0, y1 - 16)), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x1 + 3, max(0, y1 - 16)), label, fill=tuple(int(c) for c in color), font=font)
        contour_canvas = np.array(pil_tmp)

    out = np.concatenate([overlay.clip(0, 255).astype(np.uint8), contour_canvas], axis=1)
    Image.fromarray(out).save(output_path)


def draw_contact_sheet(image_rgb, masks, detections, output_path, tile_size=180):
    if not detections:
        Image.fromarray(image_rgb).save(output_path)
        return
    font = ImageFont.load_default()
    cols = 6
    rows = int(np.ceil(len(detections) / cols))
    sheet = Image.new("RGB", (cols * tile_size, rows * tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (mask, det) in enumerate(zip(masks, detections)):
        x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
        x1 = max(0, min(image_rgb.shape[1] - 1, x1))
        x2 = max(0, min(image_rgb.shape[1], x2))
        y1 = max(0, min(image_rgb.shape[0] - 1, y1))
        y2 = max(0, min(image_rgb.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_rgb[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2].astype(bool)
        color = np.array(color_for_index(i), dtype=np.float32)
        crop_overlay = crop.astype(np.float32)
        crop_overlay[crop_mask] = 0.35 * crop_overlay[crop_mask] + 0.65 * color
        tile = Image.fromarray(crop_overlay.clip(0, 255).astype(np.uint8))
        tile.thumbnail((tile_size, tile_size - 30), Image.BICUBIC)
        tile_x = (i % cols) * tile_size
        tile_y = (i // cols) * tile_size
        sheet.paste(tile, (tile_x + (tile_size - tile.width) // 2, tile_y + 28 + (tile_size - 30 - tile.height) // 2))
        label = f"{i} {det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([tile_x, tile_y, tile_x + tile_size, tile_y + 24], fill=(0, 0, 0))
        draw.text((tile_x + 4, tile_y + 7), label, fill=(255, 255, 255), font=font)
    sheet.save(output_path)


def build_overview(output_dir, summary):
    overlay_dir = output_dir / "sam2_from_yolo_box_overlays"
    out = output_dir / "overview"
    out.mkdir(exist_ok=True)
    font = ImageFont.load_default()
    tiles = []
    for frame in summary["frames"]:
        stem = Path(frame["frame_name"]).stem
        path = overlay_dir / f"{stem}_yolo_box_sam2_masks.png"
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        im = im.crop((0, 0, im.width // 2, im.height))
        im.thumbnail((320, 238), Image.BICUBIC)
        tile = Image.new("RGB", (330, 268), (255, 255, 255))
        tile.paste(im, ((330 - im.width) // 2, 28 + (238 - im.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, 330, 24], fill=(0, 0, 0))
        draw.text((8, 7), f"{stem} | yolo={frame['num_yolo_boxes']}", fill=(255, 255, 255), font=font)
        tiles.append(tile)

    if not tiles:
        return None
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 330, rows * 268), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 330, (i // cols) * 268))
    path = out / "yolo_box_sam2_overlay_overview.jpg"
    sheet.save(path, quality=92)
    return path


def load_yolo_model(model_names):
    errors = []
    for name in model_names:
        try:
            model = YOLO(name)
            return name, model
        except Exception as exc:
            errors.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
    raise RuntimeError(f"Could not load any YOLO model: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Use YOLO instance proposals as SAM2 box prompts.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sam2_config", required=True)
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--yolo_models", nargs="+", default=["yolo11x-seg.pt", "yolo11l-seg.pt", "yolo11m-seg.pt"])
    parser.add_argument("--keyframe_stride", type=int, default=12)
    parser.add_argument("--max_keyframes", type=int, default=13)
    parser.add_argument("--yolo_imgsz", type=int, default=1280)
    parser.add_argument("--yolo_conf", type=float, default=0.15)
    parser.add_argument("--yolo_iou", type=float, default=0.7)
    parser.add_argument("--max_boxes_per_frame", type=int, default=80)
    parser.add_argument("--sam2_multimask", action="store_true")
    args = parser.parse_args()

    image_paths = sorted_image_paths(args.image_dir)
    selected = image_paths[:: max(args.keyframe_stride, 1)]
    if args.max_keyframes > 0:
        selected = selected[: args.max_keyframes]
    if not selected:
        raise RuntimeError(f"No images selected from {args.image_dir}")

    output_dir = Path(args.output_dir)
    yolo_box_dir = output_dir / "yolo_box_overlays"
    sam_overlay_dir = output_dir / "sam2_from_yolo_box_overlays"
    contact_dir = output_dir / "sam2_mask_contact_sheets"
    mask_dir = output_dir / "sam2_masks_npz"
    meta_dir = output_dir / "metadata"
    for d in [yolo_box_dir, sam_overlay_dir, contact_dir, mask_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    yolo_name, yolo = load_yolo_model(args.yolo_models)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    summary = {
        "settings": vars(args),
        "loaded_yolo_model": yolo_name,
        "device": device,
        "frames": [],
    }

    class_names = yolo.names
    for selected_idx, image_path in enumerate(selected):
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        result = yolo.predict(
            source=str(image_path),
            imgsz=args.yolo_imgsz,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            device=0 if device == "cuda" else "cpu",
            verbose=False,
        )[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            clss = result.boxes.cls.detach().cpu().numpy().astype(int)
            order = np.argsort(-confs)[: args.max_boxes_per_frame]
            for idx in order:
                x1, y1, x2, y2 = [float(v) for v in boxes[idx].tolist()]
                detections.append({
                    "box_xyxy": [x1, y1, x2, y2],
                    "confidence": float(confs[idx]),
                    "class_id": int(clss[idx]),
                    "class_name": str(class_names.get(int(clss[idx]), int(clss[idx])) if isinstance(class_names, dict) else class_names[int(clss[idx])]),
                })

        stem = image_path.stem
        draw_boxes(image_rgb, detections, yolo_box_dir / f"{stem}_yolo_boxes.png")

        sam_masks = []
        sam_ious = []
        if detections:
            predictor.set_image(image_rgb)
            box_np = np.array([d["box_xyxy"] for d in detections], dtype=np.float32)
            masks, ious, _ = predictor.predict(
                box=box_np,
                multimask_output=args.sam2_multimask,
                normalize_coords=True,
            )
            if args.sam2_multimask:
                best_idx = ious.argmax(axis=1)
                masks = masks[np.arange(masks.shape[0]), best_idx]
                ious = ious[np.arange(ious.shape[0]), best_idx]
            else:
                masks = masks[:, 0] if masks.ndim == 4 else masks
                ious = ious[:, 0] if ious.ndim == 2 else ious
            sam_masks = [m.astype(bool) for m in masks]
            sam_ious = [float(v) for v in np.asarray(ious).reshape(-1)]
            predictor.reset_predictor()

        for det, sam_iou, mask in zip(detections, sam_ious, sam_masks):
            det["sam2_predicted_iou"] = sam_iou
            det["sam2_mask_area"] = int(mask.sum())

        draw_masks(image_rgb, sam_masks, detections, sam_overlay_dir / f"{stem}_yolo_box_sam2_masks.png")
        draw_contact_sheet(image_rgb, sam_masks, detections, contact_dir / f"{stem}_yolo_box_sam2_contact_sheet.jpg")
        np.savez_compressed(
            mask_dir / f"{stem}_yolo_box_sam2_masks.npz",
            masks=np.stack([m.astype(np.uint8) for m in sam_masks], axis=0) if sam_masks else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8),
            boxes_xyxy=np.array([d["box_xyxy"] for d in detections], dtype=np.float32) if detections else np.zeros((0, 4), dtype=np.float32),
            confidences=np.array([d["confidence"] for d in detections], dtype=np.float32) if detections else np.zeros((0,), dtype=np.float32),
            class_ids=np.array([d["class_id"] for d in detections], dtype=np.int32) if detections else np.zeros((0,), dtype=np.int32),
            sam2_predicted_ious=np.array(sam_ious, dtype=np.float32),
        )

        frame_meta = {
            "selected_index": selected_idx,
            "frame_name": image_path.name,
            "image_shape": list(image_rgb.shape),
            "num_yolo_boxes": len(detections),
            "yolo_box_overlay": str(yolo_box_dir / f"{stem}_yolo_boxes.png"),
            "sam2_overlay": str(sam_overlay_dir / f"{stem}_yolo_box_sam2_masks.png"),
            "contact_sheet": str(contact_dir / f"{stem}_yolo_box_sam2_contact_sheet.jpg"),
            "npz": str(mask_dir / f"{stem}_yolo_box_sam2_masks.npz"),
            "detections": detections,
        }
        meta_path = meta_dir / f"{stem}_yolo_box_sam2.json"
        meta_path.write_text(json.dumps(frame_meta, indent=2))
        summary["frames"].append(frame_meta)
        print(json.dumps({
            "frame": image_path.name,
            "yolo_boxes": len(detections),
            "labels": [d["class_name"] for d in detections[:10]],
        }), flush=True)

    overview = build_overview(output_dir, summary)
    if overview is not None:
        summary["overview"] = str(overview)
    summary_path = output_dir / "yolo_sam2_box_prompt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
