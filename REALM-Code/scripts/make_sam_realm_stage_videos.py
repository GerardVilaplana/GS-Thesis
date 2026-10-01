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


def sorted_frame_paths(image_dir):
    paths = [p for p in Path(image_dir).iterdir() if p.is_file() and FRAME_RE.match(p.name)]
    return sorted(paths, key=lambda p: int(FRAME_RE.match(p.name).group(1)))


def sorted_pngs(path):
    return sorted(Path(path).glob("*.png"))


def resize_np(image_np, max_width):
    if max_width <= 0 or image_np.shape[1] <= max_width:
        return image_np
    scale = max_width / image_np.shape[1]
    new_h = int(round(image_np.shape[0] * scale))
    return np.array(Image.fromarray(image_np).resize((max_width, new_h), Image.BICUBIC))


def add_label(image_np, text):
    im = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def overlay_binary_mask(image_np, mask, color=(255, 80, 80), alpha=0.45):
    base = image_np.astype(np.float32)
    out = base.copy()
    if mask.any():
        color_arr = np.array(color, dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return out.clip(0, 255).astype(np.uint8)


def overlay_color_mask(image_np, color_mask, selected_mask, alpha=0.55):
    base = image_np.astype(np.float32)
    out = base.copy()
    if selected_mask.any():
        cm = color_mask.astype(np.float32)
        out[selected_mask] = (1.0 - alpha) * out[selected_mask] + alpha * cm[selected_mask]
    return out.clip(0, 255).astype(np.uint8)


def make_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def sam_masks_for_boxes(predictor, image_np, boxes_xyxy):
    if len(boxes_xyxy) == 0:
        return []
    predictor.set_image(image_np)
    boxes = torch.tensor(boxes_xyxy, dtype=torch.float32, device="cuda")
    transformed = predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2]).to("cuda")
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=False,
    )
    return [masks[i, 0].detach().cpu().numpy().astype(bool) for i in range(masks.shape[0])]


def selected_realm_colors(objects_pred_rgb, sam_union, ioa_threshold):
    if not sam_union.any():
        return [], []
    pixels = objects_pred_rgb.reshape(-1, 3)
    mask_flat = sam_union.reshape(-1)
    unique_colors = np.unique(pixels[mask_flat], axis=0)
    selected = []
    scored = []
    for color in unique_colors:
        if np.all(color == 0):
            continue
        obj_mask = np.all(objects_pred_rgb == color, axis=-1)
        obj_area = int(obj_mask.sum())
        if obj_area == 0:
            continue
        inter = int((obj_mask & sam_union).sum())
        ioa = inter / max(obj_area, 1)
        iom = inter / max(int(sam_union.sum()), 1)
        scored.append((ioa, iom, tuple(int(x) for x in color.tolist()), obj_area, inter))
        if ioa >= ioa_threshold:
            selected.append(tuple(int(x) for x in color.tolist()))
    if not selected and scored:
        scored.sort(reverse=True)
        selected.append(scored[0][2])
    scored.sort(reverse=True)
    return selected, scored


def mask_from_colors(objects_pred_rgb, colors):
    mask = np.zeros(objects_pred_rgb.shape[:2], dtype=bool)
    for color in colors:
        c = np.array(color, dtype=np.uint8)
        mask |= np.all(objects_pred_rgb == c, axis=-1)
    return mask


def main():
    parser = argparse.ArgumentParser(description="Create SAM and REALM-ID stage videos from saved GroundingDINO detections.")
    parser.add_argument("--summary_json", default="output/lerf/figurines/groundingdino_detection_videos/groundingdino_detection_summary.json")
    parser.add_argument("--image_dir", default="/home/gvilaplana/GS-Thesis/data/lerf/figurines/images")
    parser.add_argument("--realm_root", default="output/lerf/figurines/train/ours_30000")
    parser.add_argument("--output_dir", default="output/lerf/figurines/sam_realm_stage_videos")
    parser.add_argument("--sam_checkpoint", default="Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth")
    parser.add_argument("--queries", nargs="+", default=["yellow rubber duck", "red apple", "green apple", "camera", "chairs"])
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--ioa_threshold", type=float, default=0.20)
    parser.add_argument("--top_global_ids", type=int, default=1)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text())
    frame_paths = sorted_frame_paths(args.image_dir)
    render_paths = sorted_pngs(Path(args.realm_root) / "renders")
    pred_paths = sorted_pngs(Path(args.realm_root) / "objects_pred")
    n = min(len(frame_paths), len(render_paths), len(pred_paths))
    if n == 0:
        raise RuntimeError("No aligned frames/renders/objects_pred images found")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).cuda()
    predictor = SamPredictor(sam)

    first = resize_np(np.array(Image.open(frame_paths[0]).convert("RGB")), args.max_width)
    h, w = first.shape[:2]

    diagnostics = {"settings": vars(args), "num_frames": n, "queries": {}}

    for query in args.queries:
        if query not in summary["queries"]:
            print(f"Skipping missing query in summary: {query}")
            continue
        slug = query.lower().replace(" ", "_").replace("/", "_")
        rows = summary["queries"][query][:n]

        per_frame = []
        color_votes = Counter()
        color_best_scores = defaultdict(float)

        print(f"First pass SAM+REALM matching for query: {query}", flush=True)
        with torch.no_grad():
            for i in range(n):
                image_np = resize_np(np.array(Image.open(frame_paths[i]).convert("RGB")), args.max_width)
                pred_rgb = resize_np(np.array(Image.open(pred_paths[i]).convert("RGB")), args.max_width)
                boxes = rows[i]["boxes_xyxy"]
                masks = sam_masks_for_boxes(predictor, image_np, boxes)
                sam_union = np.zeros(image_np.shape[:2], dtype=bool)
                for m in masks:
                    sam_union |= m
                selected_colors, scored = selected_realm_colors(pred_rgb, sam_union, args.ioa_threshold)
                for c in selected_colors:
                    color_votes[c] += 1
                for ioa, iom, c, obj_area, inter in scored:
                    color_best_scores[c] = max(color_best_scores[c], ioa)
                per_frame.append({
                    "sam_union": sam_union,
                    "selected_colors": selected_colors,
                    "num_boxes": len(boxes),
                    "num_sam_masks": len(masks),
                    "top_scored_colors": scored[:10],
                })
                if (i + 1) % 50 == 0 or i == n - 1:
                    print(f"  processed {i + 1}/{n}", flush=True)

        global_colors = [c for c, _ in color_votes.most_common(args.top_global_ids)]
        diagnostics["queries"][query] = {
            "global_selected_colors_rgb": [list(c) for c in global_colors],
            "color_votes_top20": [
                {"rgb": list(c), "votes": int(v), "best_ioa": float(color_best_scores[c])}
                for c, v in color_votes.most_common(20)
            ],
            "per_frame": [
                {
                    "frame": i,
                    "num_boxes": int(item["num_boxes"]),
                    "num_sam_masks": int(item["num_sam_masks"]),
                    "selected_colors_rgb": [list(c) for c in item["selected_colors"]],
                    "top_scored_colors": [
                        {"ioa": float(ioa), "iom": float(iom), "rgb": list(c), "obj_area": int(area), "intersection": int(inter)}
                        for ioa, iom, c, area, inter in item["top_scored_colors"]
                    ],
                }
                for i, item in enumerate(per_frame)
            ],
        }

        sam_writer = make_writer(out_root / f"{slug}_sam_masks.mp4", args.fps, (w, h))
        local_writer = make_writer(out_root / f"{slug}_realm_local_id_match.mp4", args.fps, (w, h))
        global_writer = make_writer(out_root / f"{slug}_realm_global_voted_mask.mp4", args.fps, (w, h))
        side_writer = make_writer(out_root / f"{slug}_sam_local_global_triptych.mp4", args.fps, (w * 3, h))

        try:
            for i, item in enumerate(per_frame):
                image_np = resize_np(np.array(Image.open(frame_paths[i]).convert("RGB")), args.max_width)
                render_np = resize_np(np.array(Image.open(render_paths[i]).convert("RGB")), args.max_width)
                pred_rgb = resize_np(np.array(Image.open(pred_paths[i]).convert("RGB")), args.max_width)

                sam_overlay = add_label(
                    overlay_binary_mask(image_np, item["sam_union"], color=(255, 60, 60), alpha=0.48),
                    f"SAM masks | {query} | frame {i:05d} | boxes={item['num_boxes']}",
                )
                local_mask = mask_from_colors(pred_rgb, item["selected_colors"])
                local_overlay = add_label(
                    overlay_color_mask(render_np, pred_rgb, local_mask, alpha=0.62),
                    f"REALM local ID match | colors={len(item['selected_colors'])} | frame {i:05d}",
                )
                global_mask = mask_from_colors(pred_rgb, global_colors)
                global_overlay = add_label(
                    overlay_color_mask(render_np, pred_rgb, global_mask, alpha=0.62),
                    f"REALM global voted mask | ids={len(global_colors)} | frame {i:05d}",
                )

                sam_writer.write(cv2.cvtColor(sam_overlay, cv2.COLOR_RGB2BGR))
                local_writer.write(cv2.cvtColor(local_overlay, cv2.COLOR_RGB2BGR))
                global_writer.write(cv2.cvtColor(global_overlay, cv2.COLOR_RGB2BGR))
                triptych = np.concatenate([sam_overlay, local_overlay, global_overlay], axis=1)
                side_writer.write(cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))
        finally:
            sam_writer.release()
            local_writer.release()
            global_writer.release()
            side_writer.release()

        print(f"Saved videos for query: {query}", flush=True)

    (out_root / "sam_realm_stage_summary.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"Saved all outputs to {out_root}")


if __name__ == "__main__":
    main()
