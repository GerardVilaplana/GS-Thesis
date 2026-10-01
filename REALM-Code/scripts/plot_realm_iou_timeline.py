import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from scripts.export_realm_query_ply import id2rgb
from scripts.make_qwen_sam_realm_stage_videos import (
    box_items_to_xyxy,
    filter_rows_with_realm_outputs,
    parse_frame_number,
    sam_masks_for_boxes,
)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def resize_rgb(path, max_width, interpolation):
    arr = np.array(Image.open(path).convert("RGB"))
    if max_width <= 0 or arr.shape[1] <= max_width:
        return arr, 1.0
    scale = max_width / arr.shape[1]
    new_h = int(round(arr.shape[0] * scale))
    arr = np.array(Image.fromarray(arr).resize((max_width, new_h), interpolation))
    return arr, scale


def pack_rgb(rgb):
    arr = rgb.astype(np.uint32)
    return (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]


def unpack_color(code):
    return (int((code >> 16) & 255), int((code >> 8) & 255), int(code & 255))


def rgb_to_class_id(rgb, num_classes):
    rgb = np.asarray(rgb, dtype=np.int32)
    palette = np.stack([id2rgb(i, num_classes).astype(np.int32) for i in range(num_classes)], axis=0)
    d = ((palette - rgb) ** 2).sum(axis=1).astype(np.float64)
    d[0] = 1e18
    idx = int(d.argmin())
    return idx, float(np.sqrt(d[idx]))


def compute_timeline(args):
    qwen_summary = json.loads(Path(args.qwen_summary).read_text())
    if args.query not in qwen_summary["queries"]:
        raise KeyError(f"Query {args.query!r} not found in {args.qwen_summary}")

    image_dir = Path(args.image_dir)
    realm_root = Path(args.realm_root)
    render_dir = realm_root / "renders"
    pred_dir = realm_root / "objects_pred"
    rows, skipped = filter_rows_with_realm_outputs(qwen_summary["queries"][args.query], render_dir, pred_dir)
    if not rows:
        raise RuntimeError("No rows have matching REALM render/objects_pred outputs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    # code -> frame_idx -> {visible, area, iou, num_sam_masks}
    records = defaultdict(dict)
    frame_names = []
    per_frame = []

    with torch.no_grad():
        for frame_idx, row in enumerate(rows):
            frame_name = row["frame_name"]
            frame_names.append(frame_name)
            frame_number = parse_frame_number(frame_name)
            render_index = frame_number - 1

            image_np, scale = resize_rgb(image_dir / frame_name, args.max_width, Image.BICUBIC)
            pred_rgb, _ = resize_rgb(pred_dir / f"{render_index:05d}.png", args.max_width, Image.NEAREST)
            h, w = pred_rgb.shape[:2]
            image_area = h * w
            min_realm_area = max(int(args.min_realm_area_frac * image_area), args.min_realm_area_px)
            min_sam_area = max(int(args.min_sam_area_frac * image_area), args.min_sam_area_px)

            boxes = box_items_to_xyxy(row.get("parsed", []), scale, w, h)
            masks = sam_masks_for_boxes(predictor, image_np, boxes, device)
            masks = [m for m in masks if int(m.sum()) >= min_sam_area]
            sam_areas = [int(m.sum()) for m in masks]

            packed = pack_rgb(pred_rgb)
            frame_visible = 0
            frame_matched = 0
            for code, area in zip(*np.unique(packed, return_counts=True)):
                code = int(code)
                area = int(area)
                if code == 0 or area < min_realm_area:
                    continue
                frame_visible += 1
                realm_mask = packed == code
                best_iou = 0.0
                best_sam_index = None
                for sam_idx, sam_mask in enumerate(masks):
                    inter = int((realm_mask & sam_mask).sum())
                    if inter <= 0:
                        continue
                    union = area + sam_areas[sam_idx] - inter
                    iou = inter / max(union, 1)
                    if iou > best_iou:
                        best_iou = iou
                        best_sam_index = sam_idx
                if best_iou >= args.include_iou_threshold:
                    frame_matched += 1
                records[code][frame_idx] = {
                    "visible": True,
                    "area_px": area,
                    "iou": float(best_iou),
                    "best_sam_index": best_sam_index,
                    "num_sam_masks": len(masks),
                }

            per_frame.append({
                "frame_name": frame_name,
                "render_index": int(render_index),
                "num_qwen_boxes": int(len(boxes)),
                "num_valid_sam_masks": int(len(masks)),
                "num_visible_realm_ids": int(frame_visible),
                "num_ids_with_iou_ge_include_threshold": int(frame_matched),
            })
            if (frame_idx + 1) % 10 == 0 or frame_idx == len(rows) - 1:
                print(f"Processed {frame_idx + 1}/{len(rows)}", flush=True)

    selected_codes = []
    row_stats = []
    for code, by_frame in records.items():
        ious = [v["iou"] for v in by_frame.values()]
        if not any(iou >= args.include_iou_threshold for iou in ious):
            continue
        first_visible = min(by_frame.keys())
        first_match = min((idx for idx, v in by_frame.items() if v["iou"] >= args.include_iou_threshold), default=first_visible)
        class_id, palette_distance = rgb_to_class_id(unpack_color(code), args.num_classes)
        visible_count = len(by_frame)
        matched_count = sum(1 for v in by_frame.values() if v["iou"] >= args.include_iou_threshold)
        row_stats.append({
            "code": int(code),
            "rgb": list(unpack_color(code)),
            "class_id": int(class_id),
            "palette_distance": palette_distance,
            "first_visible_frame_index": int(first_visible),
            "first_match_frame_index": int(first_match),
            "visible_frames": int(visible_count),
            "frames_with_iou_ge_include_threshold": int(matched_count),
            "max_iou": float(max(ious)),
            "mean_iou_visible_frames": float(np.mean(ious)),
        })
        selected_codes.append(code)

    row_stats.sort(key=lambda x: (x["first_visible_frame_index"], x["first_match_frame_index"], x["class_id"]))
    selected_codes = [r["code"] for r in row_stats]
    return records, selected_codes, row_stats, frame_names, per_frame, skipped


def render_plot(args, records, selected_codes, row_stats, frame_names, per_frame, skipped):
    n_rows = len(selected_codes)
    n_frames = len(frame_names)
    if n_rows == 0:
        raise RuntimeError("No object IDs had an IoU match above include_iou_threshold")

    cmap = LinearSegmentedColormap.from_list("iou_red_green", ["#b2182b", "#ffffbf", "#1a9850"])
    norm = Normalize(vmin=0.0, vmax=1.0)
    white = np.array([1.0, 1.0, 1.0, 1.0])
    purple = np.array([0.58, 0.36, 0.95, 1.0])

    rgba = np.zeros((n_rows, n_frames, 4), dtype=np.float32)
    for r, code in enumerate(selected_codes):
        for c in range(n_frames):
            rec = records[code].get(c)
            if rec is None:
                rgba[r, c] = white
            elif rec["iou"] <= args.no_detection_iou_threshold:
                rgba[r, c] = purple
            else:
                rgba[r, c] = cmap(norm(rec["iou"]))

    fig_w = max(12, min(28, n_frames * 0.18))
    fig_h = max(5, min(40, n_rows * 0.28 + 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    ax.set_title(f"REALM ID / Qwen-SAM IoU timeline | query: {args.query}")
    ax.set_xlabel("Frames (chronological sampled order)")
    ax.set_ylabel("REALM object ID")

    labels = [str(r["class_id"]) for r in row_stats]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(labels, fontsize=8)

    tick_step = max(1, n_frames // 12)
    xticks = np.arange(0, n_frames, tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], rotation=0, fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("IoU with best Qwen-SAM instance mask")

    legend = [
        Patch(facecolor="white", edgecolor="black", label="REALM ID not visible"),
        Patch(facecolor=purple, edgecolor="black", label="REALM ID visible, no Qwen-SAM overlap"),
    ]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=2, fontsize=8)
    fig.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.query)
    png_path = out_dir / f"{slug}_realm_iou_timeline.png"
    json_path = out_dir / f"{slug}_realm_iou_timeline.json"
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)

    report = {
        "query": args.query,
        "qwen_summary": args.qwen_summary,
        "realm_root": args.realm_root,
        "settings": vars(args),
        "num_frames": n_frames,
        "num_object_rows": n_rows,
        "frame_names": frame_names,
        "skipped_rows_without_realm_output": skipped,
        "per_frame": per_frame,
        "rows": row_stats,
    }
    json_path.write_text(json.dumps(report, indent=2))
    return png_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Plot per-REALM-ID IoU timeline against Qwen-box SAM masks.")
    parser.add_argument("--qwen_summary", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--image_dir", default="/home/gvilaplana/GS-Thesis/data/lerf/figurines/images")
    parser.add_argument("--realm_root", default="output/lerf/figurines_clean145/train/ours_30000")
    parser.add_argument("--output_dir", default="output/lerf/figurines_clean145/iou_timeline_plots")
    parser.add_argument("--sam_checkpoint", default="Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth")
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--num_classes", type=int, default=256)
    parser.add_argument("--min_realm_area_frac", type=float, default=0.001)
    parser.add_argument("--min_realm_area_px", type=int, default=50)
    parser.add_argument("--min_sam_area_frac", type=float, default=0.001)
    parser.add_argument("--min_sam_area_px", type=int, default=50)
    parser.add_argument("--include_iou_threshold", type=float, default=0.01,
                        help="Keep rows with at least one frame at or above this IoU.")
    parser.add_argument("--no_detection_iou_threshold", type=float, default=0.0,
                        help="Visible IDs at or below this IoU are colored purple.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    records, selected_codes, row_stats, frame_names, per_frame, skipped = compute_timeline(args)
    png_path, json_path = render_plot(args, records, selected_codes, row_stats, frame_names, per_frame, skipped)
    print(json.dumps({"png": str(png_path), "json": str(json_path), "rows": len(row_stats), "frames": len(frame_names)}, indent=2))


if __name__ == "__main__":
    main()
