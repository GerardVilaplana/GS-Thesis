import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np

from plot_realm_iou_timeline import compute_timeline, slugify


def infer_x_axis_label(frame_names):
    frame_numbers = []
    for name in frame_names:
        try:
            frame_numbers.append(int(Path(name).stem.split("_")[-1]))
        except ValueError:
            frame_numbers = []
            break
    if len(frame_numbers) >= 2:
        diffs = np.diff(frame_numbers)
        if np.all(diffs == diffs[0]):
            stride = int(diffs[0])
            first = int(frame_numbers[0])
            if stride == 1:
                return f"Clean frame index (original frame = {first} + x)"
            return f"Sampled frame index (stride-{stride}: original frame = {first} + {stride}*x)"
    return "Frame index"


def render_area_height_plot(args, records, selected_codes, row_stats, frame_names, per_frame, skipped):
    n_rows = len(selected_codes)
    n_frames = len(frame_names)
    if n_rows == 0:
        raise RuntimeError("No object IDs had an IoU match above include_iou_threshold")

    cmap = LinearSegmentedColormap.from_list("iou_red_green", ["#b2182b", "#ffffbf", "#1a9850"])
    norm = Normalize(vmin=0.0, vmax=1.0)
    purple = np.array([0.58, 0.36, 0.95, 1.0])

    fig_w = max(12, min(28, n_frames * 0.18))
    fig_h = max(5, min(40, n_rows * 0.32 + 2.8))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")

    max_area_by_code = {
        code: max(rec["area_px"] for rec in records[code].values())
        for code in selected_codes
    }

    cell_report = []
    for row_idx, code in enumerate(selected_codes):
        max_area = max(max_area_by_code[code], 1)
        row_cells = []
        for frame_idx in range(n_frames):
            rec = records[code].get(frame_idx)
            if rec is None:
                row_cells.append({"frame_index": frame_idx, "visible": False})
                continue
            area_frac = rec["area_px"] / max_area
            height = max(args.min_area_height, min(1.0, area_frac))
            y0 = row_idx - height / 2.0
            color = purple if rec["iou"] <= args.no_detection_iou_threshold else cmap(norm(rec["iou"]))
            ax.add_patch(Rectangle((frame_idx - 0.5, y0), 1.0, height, facecolor=color, edgecolor="none"))
            row_cells.append({
                "frame_index": frame_idx,
                "visible": True,
                "area_px": int(rec["area_px"]),
                "area_fraction_of_id_max": float(area_frac),
                "iou": float(rec["iou"]),
            })
        cell_report.append({
            "class_id": row_stats[row_idx]["class_id"],
            "rgb": row_stats[row_idx]["rgb"],
            "max_area_px": int(max_area),
            "cells": row_cells,
        })

    ax.set_xlim(-0.5, n_frames - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_title(f"REALM ID / Qwen-SAM IoU timeline | area-height | query: {args.query}")
    ax.set_xlabel(infer_x_axis_label(frame_names))
    ax.set_ylabel("REALM object ID")

    labels = [str(r["class_id"]) for r in row_stats]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(labels, fontsize=8)

    tick_step = max(1, n_frames // 12)
    xticks = np.arange(0, n_frames, tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], fontsize=8)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("IoU with best Qwen-SAM instance mask")

    legend = [
        Patch(facecolor="white", edgecolor="black", label="REALM ID not visible"),
        Patch(facecolor=purple, edgecolor="black", label="Visible, no Qwen-SAM overlap"),
        Patch(facecolor="gray", edgecolor="black", label="Cell height = area / max area for that ID"),
    ]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=2, fontsize=8)
    fig.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.query)
    png_path = out_dir / f"{slug}_realm_iou_timeline_area_height.png"
    json_path = out_dir / f"{slug}_realm_iou_timeline_area_height.json"
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
        "area_height_cells": cell_report,
    }
    json_path.write_text(json.dumps(report, indent=2))
    return png_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Plot REALM/Qwen-SAM IoU timeline with cell height encoding visible ID area.")
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
    parser.add_argument("--include_iou_threshold", type=float, default=0.01)
    parser.add_argument("--no_detection_iou_threshold", type=float, default=0.0)
    parser.add_argument("--min_area_height", type=float, default=0.03)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    records, selected_codes, row_stats, frame_names, per_frame, skipped = compute_timeline(args)
    png_path, json_path = render_area_height_plot(args, records, selected_codes, row_stats, frame_names, per_frame, skipped)
    print(json.dumps({"png": str(png_path), "json": str(json_path), "rows": len(row_stats), "frames": len(frame_names)}, indent=2))


if __name__ == "__main__":
    main()
