import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from segment_anything import SamPredictor, sam_model_registry

FRAME_RE = re.compile(r"frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def resize_np(image_np, max_width):
    if max_width <= 0 or image_np.shape[1] <= max_width:
        return image_np, 1.0
    scale = max_width / image_np.shape[1]
    new_h = int(round(image_np.shape[0] * scale))
    resized = np.array(Image.fromarray(image_np).resize((max_width, new_h), Image.BICUBIC))
    return resized, scale


def add_label(image_np, text):
    im = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def overlay_binary_mask(image_np, mask, color=(255, 80, 80), alpha=0.48):
    out = image_np.astype(np.float32).copy()
    if mask.any():
        out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def overlay_color_mask(image_np, color_mask, selected_mask, alpha=0.62):
    out = image_np.astype(np.float32).copy()
    if selected_mask.any():
        out[selected_mask] = (1 - alpha) * out[selected_mask] + alpha * color_mask[selected_mask].astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def make_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def parse_frame_number(frame_name):
    match = FRAME_RE.match(frame_name)
    if not match:
        raise ValueError(f"Unexpected frame filename: {frame_name}")
    return int(match.group(1))


def box_items_to_xyxy(box_items, scale, width, height):
    boxes = []
    for item in box_items:
        box = item.get("bbox_2d", item.get("box", []))
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) * scale for v in box]
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


def sam_masks_for_boxes(predictor, image_np, boxes_xyxy, device):
    if len(boxes_xyxy) == 0:
        return []
    predictor.set_image(image_np)
    boxes = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device)
    transformed = predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2]).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=False,
    )
    return [masks[i, 0].detach().cpu().numpy().astype(bool) for i in range(masks.shape[0])]


def pack_rgb(rgb):
    arr = rgb.astype(np.uint32)
    return (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]


def unpack_color(code):
    return (int((code >> 16) & 255), int((code >> 8) & 255), int(code & 255))


def selected_realm_colors_fast(objects_pred_rgb, sam_union, ioa_threshold):
    if not sam_union.any():
        return [], []
    packed = pack_rgb(objects_pred_rgb)
    sam_codes = packed[sam_union]
    if sam_codes.size == 0:
        return [], []
    candidate_codes = np.unique(sam_codes)
    selected = []
    scored = []
    sam_area = max(int(sam_union.sum()), 1)
    for code in candidate_codes:
        if int(code) == 0:
            continue
        obj_mask = packed == code
        obj_area = int(obj_mask.sum())
        if obj_area == 0:
            continue
        inter = int((obj_mask & sam_union).sum())
        ioa = inter / max(obj_area, 1)
        iom = inter / sam_area
        color = unpack_color(int(code))
        scored.append((ioa, iom, color, obj_area, inter))
        if ioa >= ioa_threshold:
            selected.append(color)
    if not selected and scored:
        scored.sort(reverse=True)
        selected.append(scored[0][2])
    scored.sort(reverse=True)
    return selected, scored


def mask_from_colors(objects_pred_rgb, colors):
    if not colors:
        return np.zeros(objects_pred_rgb.shape[:2], dtype=bool)
    packed = pack_rgb(objects_pred_rgb)
    codes = np.array([(c[0] << 16) | (c[1] << 8) | c[2] for c in colors], dtype=np.uint32)
    return np.isin(packed, codes)


def load_rgb(path, max_width):
    arr = np.array(Image.open(path).convert("RGB"))
    return resize_np(arr, max_width)


def filter_rows_with_realm_outputs(rows, render_dir, pred_dir):
    kept = []
    skipped = []
    for row in rows:
        frame_number = parse_frame_number(row["frame_name"])
        render_index = frame_number - 1
        render_path = render_dir / f"{render_index:05d}.png"
        pred_path = pred_dir / f"{render_index:05d}.png"
        if render_path.exists() and pred_path.exists():
            kept.append(row)
        else:
            skipped.append({
                "frame_name": row["frame_name"],
                "render_index": int(render_index),
                "missing_render": not render_path.exists(),
                "missing_objects_pred": not pred_path.exists(),
            })
    return kept, skipped


def main():
    parser = argparse.ArgumentParser(description="SAM and REALM-ID stage videos from saved Qwen2.5-VL boxes.")
    parser.add_argument("--summary_json", default="output/lerf/figurines/qwen_vl_box_videos_stride3/qwen_vl_box_summary.json")
    parser.add_argument("--image_dir", default="/home/gvilaplana/GS-Thesis/data/lerf/figurines/images")
    parser.add_argument("--realm_root", default="output/lerf/figurines/train/ours_30000")
    parser.add_argument("--output_dir", default="output/lerf/figurines/qwen_sam_realm_stage_videos_stride3")
    parser.add_argument("--sam_checkpoint", default="Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth")
    parser.add_argument("--queries", nargs="+", default=["yellow rubber duck", "red apple", "green apple", "camera", "chairs"])
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--ioa_threshold", type=float, default=0.20)
    parser.add_argument("--top_global_ids", type=int, default=1)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text())
    image_dir = Path(args.image_dir)
    render_dir = Path(args.realm_root) / "renders"
    pred_dir = Path(args.realm_root) / "objects_pred"
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    diagnostics = {"settings": vars(args), "queries": {}}

    for query in args.queries:
        if query not in summary["queries"]:
            print(f"Skipping missing query in Qwen summary: {query}", flush=True)
            continue
        slug = query.lower().replace(" ", "_").replace("/", "_")
        rows, skipped_rows = filter_rows_with_realm_outputs(summary["queries"][query], render_dir, pred_dir)
        if not rows:
            continue
        first_frame = image_dir / rows[0]["frame_name"]
        first_np, _ = load_rgb(first_frame, args.max_width)
        h, w = first_np.shape[:2]
        print(f"Processing query: {query} ({len(rows)} sampled frames)", flush=True)

        color_votes = Counter()
        color_best_scores = defaultdict(float)
        per_frame_summary = []

        sam_writer = make_writer(out_root / f"{slug}_sam_masks.mp4", args.fps, (w, h))
        local_writer = make_writer(out_root / f"{slug}_realm_local_id_match.mp4", args.fps, (w, h))

        try:
            with torch.no_grad():
                for i, row in enumerate(rows):
                    frame_name = row["frame_name"]
                    frame_number = parse_frame_number(frame_name)
                    render_index = frame_number - 1
                    image_np, scale = load_rgb(image_dir / frame_name, args.max_width)
                    render_np, _ = load_rgb(render_dir / f"{render_index:05d}.png", args.max_width)
                    pred_rgb, _ = load_rgb(pred_dir / f"{render_index:05d}.png", args.max_width)
                    boxes = box_items_to_xyxy(row.get("parsed", []), scale, image_np.shape[1], image_np.shape[0])

                    masks = sam_masks_for_boxes(predictor, image_np, boxes, device)
                    sam_union = np.zeros(image_np.shape[:2], dtype=bool)
                    for m in masks:
                        sam_union |= m

                    selected_colors, scored = selected_realm_colors_fast(pred_rgb, sam_union, args.ioa_threshold)
                    for c in selected_colors:
                        color_votes[c] += 1
                    for ioa, iom, c, area, inter in scored[:10]:
                        color_best_scores[c] = max(color_best_scores[c], ioa)

                    local_mask = mask_from_colors(pred_rgb, selected_colors)
                    sam_overlay = add_label(
                        overlay_binary_mask(image_np, sam_union),
                        f"Qwen->SAM | {query} | {frame_name} | boxes={len(boxes)} masks={len(masks)}",
                    )
                    local_overlay = add_label(
                        overlay_color_mask(render_np, pred_rgb, local_mask),
                        f"Qwen->REALM local ID match | colors={len(selected_colors)} | {frame_name}",
                    )
                    sam_writer.write(cv2.cvtColor(sam_overlay, cv2.COLOR_RGB2BGR))
                    local_writer.write(cv2.cvtColor(local_overlay, cv2.COLOR_RGB2BGR))

                    per_frame_summary.append({
                        "frame_name": frame_name,
                        "render_index": int(render_index),
                        "num_boxes": int(len(boxes)),
                        "num_sam_masks": int(len(masks)),
                        "selected_colors_rgb": [list(c) for c in selected_colors],
                        "top_scored_colors": [
                            {"ioa": float(ioa), "iom": float(iom), "rgb": list(c), "obj_area": int(area), "intersection": int(inter)}
                            for ioa, iom, c, area, inter in scored[:10]
                        ],
                    })
                    if (i + 1) % 25 == 0 or i == len(rows) - 1:
                        print(f"  SAM/local {i + 1}/{len(rows)}", flush=True)
        finally:
            sam_writer.release()
            local_writer.release()

        global_colors = [c for c, _ in color_votes.most_common(args.top_global_ids)]
        global_writer = make_writer(out_root / f"{slug}_realm_global_voted_mask.mp4", args.fps, (w, h))
        try:
            for row in rows:
                frame_number = parse_frame_number(row["frame_name"])
                render_index = frame_number - 1
                render_np, _ = load_rgb(render_dir / f"{render_index:05d}.png", args.max_width)
                pred_rgb, _ = load_rgb(pred_dir / f"{render_index:05d}.png", args.max_width)
                global_mask = mask_from_colors(pred_rgb, global_colors)
                global_overlay = add_label(
                    overlay_color_mask(render_np, pred_rgb, global_mask),
                    f"Qwen->REALM global voted mask | ids={len(global_colors)} | {row['frame_name']}",
                )
                global_writer.write(cv2.cvtColor(global_overlay, cv2.COLOR_RGB2BGR))
        finally:
            global_writer.release()

        diagnostics["queries"][query] = {
            "global_selected_colors_rgb": [list(c) for c in global_colors],
            "skipped_rows_without_realm_output": skipped_rows,
            "color_votes_top20": [
                {"rgb": list(c), "votes": int(v), "best_ioa": float(color_best_scores[c])}
                for c, v in color_votes.most_common(20)
            ],
            "per_frame": per_frame_summary,
        }
        (out_root / "qwen_sam_realm_stage_summary.json").write_text(json.dumps(diagnostics, indent=2))
        print(f"Saved videos for query: {query}", flush=True)

    print(f"Saved all outputs to {out_root}", flush=True)


if __name__ == "__main__":
    main()
