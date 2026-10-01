import argparse
import colorsys
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from segment_anything import SamPredictor, sam_model_registry

from scripts.make_qwen_sam_realm_stage_videos import (
    box_items_to_xyxy,
    filter_rows_with_realm_outputs,
    parse_frame_number,
    sam_masks_for_boxes,
)
from scripts.export_realm_query_ply import C0, id2rgb, predict_gaussian_ids


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def resize_rgb(path, max_width, interpolation=Image.BICUBIC):
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
    d = ((palette - rgb) ** 2).sum(axis=1)
    d[0] = 10**12
    return int(d.argmin()), float(np.sqrt(d.min()))


def highlight_rgb_for_index(index):
    palette = [
        (255, 40, 40),
        (40, 130, 255),
        (30, 220, 110),
        (255, 210, 40),
        (220, 70, 255),
        (255, 130, 40),
        (40, 230, 230),
        (180, 180, 255),
    ]
    if index < len(palette):
        return np.array(palette[index], dtype=np.uint8)
    h = (index * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def rgb_to_sh(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    return (rgb - 0.5) / C0


def write_multi_highlight_ply(ply_data, pred_ids, selected_ids, output_path, dim_factor):
    vertex = ply_data["vertex"].data.copy()
    selected_any = np.isin(pred_ids, selected_ids)
    if dim_factor < 1.0:
        vertex["f_dc_0"][~selected_any] *= dim_factor
        vertex["f_dc_1"][~selected_any] *= dim_factor
        vertex["f_dc_2"][~selected_any] *= dim_factor
    for idx, class_id in enumerate(selected_ids):
        mask = pred_ids == class_id
        sh = rgb_to_sh(highlight_rgb_for_index(idx))
        vertex["f_dc_0"][mask] = sh[0]
        vertex["f_dc_1"][mask] = sh[1]
        vertex["f_dc_2"][mask] = sh[2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply_data.text).write(output_path)


def apply_anchor_filter(selected, visible_frames_by_code, matched_frames_by_code, frame_records, args):
    if not args.use_anchor_filter or len(selected) <= 1:
        return selected, {"enabled": False, "reason": "disabled_or_single_selection"}

    avg_boxes_per_frame = sum(r["num_boxes"] for r in frame_records) / max(len(frame_records), 1)
    if avg_boxes_per_frame > args.anchor_max_avg_boxes_per_frame:
        return selected, {
            "enabled": False,
            "reason": "query_behaves_multi_object",
            "avg_boxes_per_frame": float(avg_boxes_per_frame),
            "anchor_max_avg_boxes_per_frame": args.anchor_max_avg_boxes_per_frame,
        }

    anchor = selected[0]
    anchor_code = anchor["color_code"]
    anchor_visible = visible_frames_by_code.get(anchor_code, set())
    if len(anchor_visible) < args.anchor_min_visible_frames:
        return selected, {
            "enabled": False,
            "reason": "anchor_not_visible_enough",
            "anchor_class_id": anchor["class_id"],
            "anchor_visible_frames": len(anchor_visible),
        }

    kept = [anchor]
    rejected = []
    for candidate in selected[1:]:
        code = candidate["color_code"]
        visible = visible_frames_by_code.get(code, set())
        matched = matched_frames_by_code.get(code, set())

        matched_on_anchor = len(matched & anchor_visible)
        matched_outside_anchor = len(matched - anchor_visible)
        visible_on_anchor = len(visible & anchor_visible)
        co_match_fraction = matched_on_anchor / max(visible_on_anchor, 1)
        fallback_match_fraction = matched_outside_anchor / max(len(matched), 1)

        candidate["anchor_filter"] = {
            "anchor_class_id": anchor["class_id"],
            "avg_boxes_per_frame": float(avg_boxes_per_frame),
            "matched_on_anchor_visible_frames": int(matched_on_anchor),
            "matched_outside_anchor_visible_frames": int(matched_outside_anchor),
            "visible_on_anchor_visible_frames": int(visible_on_anchor),
            "co_match_fraction": float(co_match_fraction),
            "fallback_match_fraction": float(fallback_match_fraction),
        }

        reject_as_fallback = (
            candidate["match_fraction"] < args.anchor_keep_min_match_fraction
            and fallback_match_fraction >= args.anchor_fallback_match_fraction
            and co_match_fraction <= args.anchor_max_co_match_fraction
        )
        if reject_as_fallback:
            candidate["anchor_filter"]["decision"] = "rejected_fallback_substitute"
            rejected.append(candidate)
        else:
            candidate["anchor_filter"]["decision"] = "kept"
            kept.append(candidate)

    return kept, {
        "enabled": True,
        "anchor_class_id": anchor["class_id"],
        "avg_boxes_per_frame": float(avg_boxes_per_frame),
        "rejected_class_ids": [c["class_id"] for c in rejected],
        "rejected": rejected,
    }


def evaluate_query(query, rows, image_dir, pred_dir, predictor, device, args):
    visible = defaultdict(int)
    matched = defaultdict(int)
    iou_sum = defaultdict(float)
    best_iou_global = defaultdict(float)
    visible_frames_by_code = defaultdict(set)
    matched_frames_by_code = defaultdict(set)
    frame_records = []

    for frame_idx, row in enumerate(rows):
        frame_name = row["frame_name"]
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

        packed = pack_rgb(pred_rgb)
        visible_codes = []
        for code, area in zip(*np.unique(packed, return_counts=True)):
            code = int(code)
            if code == 0 or int(area) < min_realm_area:
                continue
            visible_codes.append((code, int(area)))
            visible[code] += 1
            visible_frames_by_code[code].add(frame_idx)

        matched_codes = []
        for code, realm_area in visible_codes:
            realm_mask = packed == code
            best_iou = 0.0
            for sam_mask in masks:
                inter = int((realm_mask & sam_mask).sum())
                if inter == 0:
                    continue
                union = realm_area + int(sam_mask.sum()) - inter
                iou = inter / max(union, 1)
                best_iou = max(best_iou, iou)
            best_iou_global[code] = max(best_iou_global[code], best_iou)
            if best_iou >= args.iou_threshold:
                matched[code] += 1
                matched_frames_by_code[code].add(frame_idx)
                iou_sum[code] += best_iou
                matched_codes.append({"rgb": list(unpack_color(code)), "iou": best_iou})

        frame_records.append({
            "frame_name": frame_name,
            "render_index": int(render_index),
            "num_boxes": int(len(boxes)),
            "num_valid_sam_masks": int(len(masks)),
            "num_visible_realm_ids": int(len(visible_codes)),
            "matched_ids": matched_codes,
        })

    candidates = []
    for code, v_count in visible.items():
        m_count = matched.get(code, 0)
        frac = m_count / max(v_count, 1)
        class_id, palette_distance = rgb_to_class_id(unpack_color(code), args.num_classes)
        candidates.append({
            "color_code": int(code),
            "rgb": list(unpack_color(code)),
            "class_id": int(class_id),
            "palette_distance": palette_distance,
            "visible_frames": int(v_count),
            "matched_frames": int(m_count),
            "match_fraction": float(frac),
            "mean_iou_on_matched": float(iou_sum[code] / m_count) if m_count else 0.0,
            "best_iou": float(best_iou_global[code]),
        })
    candidates.sort(key=lambda x: (x["match_fraction"], x["matched_frames"], x["visible_frames"], x["mean_iou_on_matched"]), reverse=True)

    user_selected = [c for c in candidates if c["match_fraction"] >= args.min_match_fraction]
    if args.use_relaxed_rule:
        relaxed_selected = [
            c for c in candidates
            if (
                c["matched_frames"] >= max(
                    args.relaxed_min_matched_frames,
                    int(math.ceil(args.relaxed_min_visible_fraction * c["visible_frames"])),
                )
                and c["mean_iou_on_matched"] >= args.relaxed_min_mean_iou
                and c["best_iou"] >= args.relaxed_min_best_iou
            )
        ]
        selected_by_id = {c["class_id"]: c for c in user_selected}
        for candidate in relaxed_selected:
            selected_by_id.setdefault(candidate["class_id"], candidate)
        user_selected = list(selected_by_id.values())
        user_selected.sort(
            key=lambda x: (
                x["match_fraction"],
                x["matched_frames"],
                x["mean_iou_on_matched"],
                x["best_iou"],
            ),
            reverse=True,
        )
    user_selected, user_anchor_filter = apply_anchor_filter(
        user_selected, visible_frames_by_code, matched_frames_by_code, frame_records, args
    )
    corrected_selected = [
        c for c in user_selected
        if c["visible_frames"] >= args.min_visible_frames and c["matched_frames"] >= args.min_matched_frames
    ]
    corrected_selected, corrected_anchor_filter = apply_anchor_filter(
        corrected_selected, visible_frames_by_code, matched_frames_by_code, frame_records, args
    )

    return {
        "query": query,
        "settings": {
            "iou_threshold": args.iou_threshold,
            "min_match_fraction": args.min_match_fraction,
            "min_realm_area_frac": args.min_realm_area_frac,
            "min_realm_area_px": args.min_realm_area_px,
            "min_sam_area_frac": args.min_sam_area_frac,
            "min_sam_area_px": args.min_sam_area_px,
            "min_visible_frames": args.min_visible_frames,
            "min_matched_frames": args.min_matched_frames,
            "use_relaxed_rule": args.use_relaxed_rule,
            "relaxed_min_matched_frames": args.relaxed_min_matched_frames,
            "relaxed_min_visible_fraction": args.relaxed_min_visible_fraction,
            "relaxed_min_mean_iou": args.relaxed_min_mean_iou,
            "relaxed_min_best_iou": args.relaxed_min_best_iou,
            "use_anchor_filter": args.use_anchor_filter,
            "anchor_max_avg_boxes_per_frame": args.anchor_max_avg_boxes_per_frame,
            "anchor_keep_min_match_fraction": args.anchor_keep_min_match_fraction,
            "anchor_fallback_match_fraction": args.anchor_fallback_match_fraction,
            "anchor_max_co_match_fraction": args.anchor_max_co_match_fraction,
            "anchor_min_visible_frames": args.anchor_min_visible_frames,
        },
        "num_frames": len(rows),
        "candidates": candidates,
        "user_selected": user_selected,
        "corrected_selected": corrected_selected,
        "user_anchor_filter": user_anchor_filter,
        "corrected_anchor_filter": corrected_anchor_filter,
        "frames": frame_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Export multi-ID REALM query Gaussian PLYs using per-ID trajectory validation.")
    parser.add_argument("--qwen_summary", required=True)
    parser.add_argument("--image_dir", default="/home/gvilaplana/GS-Thesis/data/lerf/figurines/images")
    parser.add_argument("--realm_root", default="output/lerf/figurines_clean145/train/ours_30000")
    parser.add_argument("--point_cloud", default="output/lerf/figurines_clean145/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--classifier", default="output/lerf/figurines_clean145/point_cloud/iteration_30000/classifier.pth")
    parser.add_argument("--output_dir", default="output/lerf/figurines_clean145/query_ply_multi")
    parser.add_argument("--sam_checkpoint", default="Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth")
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_match_fraction", type=float, default=0.5)
    parser.add_argument("--min_visible_frames", type=int, default=5)
    parser.add_argument("--min_matched_frames", type=int, default=3)
    parser.add_argument("--use_relaxed_rule", action="store_true",
                        help="Also accept IDs with strong repeated evidence even if match_fraction is below min_match_fraction.")
    parser.add_argument("--relaxed_min_matched_frames", type=int, default=8)
    parser.add_argument("--relaxed_min_visible_fraction", type=float, default=0.15,
                        help="Relaxed rule also requires this fraction of visible frames to be matched.")
    parser.add_argument("--relaxed_min_mean_iou", type=float, default=0.6)
    parser.add_argument("--relaxed_min_best_iou", type=float, default=0.8)
    parser.add_argument("--use_anchor_filter", action="store_true",
                        help="Reject fallback-like secondary IDs for specific/single-object query behavior.")
    parser.add_argument("--anchor_max_avg_boxes_per_frame", type=float, default=1.25,
                        help="Only apply anchor filtering when Qwen behaves like a single-object detector.")
    parser.add_argument("--anchor_keep_min_match_fraction", type=float, default=0.5,
                        help="Never reject candidates that already satisfy this visible-trajectory fraction.")
    parser.add_argument("--anchor_fallback_match_fraction", type=float, default=0.6,
                        help="Reject if at least this fraction of matches happen when the anchor is not visible.")
    parser.add_argument("--anchor_max_co_match_fraction", type=float, default=0.25,
                        help="Reject if candidate rarely matches while the anchor is visible.")
    parser.add_argument("--anchor_min_visible_frames", type=int, default=10)
    parser.add_argument("--min_realm_area_frac", type=float, default=0.001)
    parser.add_argument("--min_realm_area_px", type=int, default=50)
    parser.add_argument("--min_sam_area_frac", type=float, default=0.001)
    parser.add_argument("--min_sam_area_px", type=int, default=50)
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--num_classes", type=int, default=256)
    parser.add_argument("--dim_factor", type=float, default=0.25)
    args = parser.parse_args()

    qwen_summary = json.loads(Path(args.qwen_summary).read_text())
    image_dir = Path(args.image_dir)
    pred_dir = Path(args.realm_root) / "objects_pred"
    render_dir = Path(args.realm_root) / "renders"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    ply_data = PlyData.read(args.point_cloud)
    pred_gaussian_ids = predict_gaussian_ids(ply_data["vertex"].data, args.classifier, args.num_classes)

    report = {"settings": vars(args), "queries": {}}
    with torch.no_grad():
        for query in args.queries:
            if query not in qwen_summary["queries"]:
                print(f"Skipping missing query: {query}", flush=True)
                continue
            rows, skipped = filter_rows_with_realm_outputs(qwen_summary["queries"][query], render_dir, pred_dir)
            result = evaluate_query(query, rows, image_dir, pred_dir, predictor, device, args)
            result["skipped_rows_without_realm_output"] = skipped

            slug = slugify(query)
            selected_ids = [c["class_id"] for c in result["user_selected"]]
            ply_path = out / f"{slug}_iou{int(args.iou_threshold * 100):03d}_multi_id_3dgs.ply"
            write_multi_highlight_ply(ply_data, pred_gaussian_ids, selected_ids, ply_path, args.dim_factor)
            result["output"] = str(ply_path)
            result["selected_class_ids"] = selected_ids
            print(json.dumps({
                "query": query,
                "selected_class_ids": selected_ids,
                "num_selected_ids": len(selected_ids),
                "output": str(ply_path),
            }), flush=True)

            report["queries"][query] = result
            (out / "multi_id_query_report.json").write_text(json.dumps(report, indent=2))

    print(out / "multi_id_query_report.json", flush=True)


if __name__ == "__main__":
    main()
