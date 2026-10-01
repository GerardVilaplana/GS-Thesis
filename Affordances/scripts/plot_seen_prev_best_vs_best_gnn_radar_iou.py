#!/usr/bin/env python3
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization")
SUMMARY_DIR = OUT_ROOT / "summary_plots"
PN_PATH = (
    OUT_ROOT
    / "47_exp46_dino_pointnet_mean_feature_sweep_v1/object_crop_center_avg"
    / "01_seen_instance_17cat/dino_geometry_color_quality/pointnet_mean/per_category_metrics.csv"
)
GNN_PATH = (
    OUT_ROOT
    / "48_exp46_dino_gnn_embeddings_pointnet_mean_v1/object_crop_best"
    / "01_seen_instance_17cat/object_crop_best/gnn_embeddings_pointnet_mean/per_category_metrics.csv"
)


def display_category(category: str) -> str:
    text = category.replace("_", " ")
    return "\n".join(textwrap.wrap(text, width=13, break_long_words=False))


def closed(values: list[float]) -> list[float]:
    return values + values[:1]


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    pn = pd.read_csv(PN_PATH)[["category", "macro_scene_iou", "macro_scene_f1", "macro_scene_auprc"]]
    gnn = pd.read_csv(GNN_PATH)[["category", "macro_scene_iou", "macro_scene_f1", "macro_scene_auprc"]]
    df = pn.merge(gnn, on="category", suffixes=("_prev_best", "_gnn"))
    df = df.rename(
        columns={
            "macro_scene_iou_prev_best": "prev_best_iou",
            "macro_scene_iou_gnn": "gnn_iou",
            "macro_scene_f1_prev_best": "prev_best_f1",
            "macro_scene_f1_gnn": "gnn_f1",
            "macro_scene_auprc_prev_best": "prev_best_auprc",
            "macro_scene_auprc_gnn": "gnn_auprc",
        }
    )
    df["delta_iou"] = df["gnn_iou"] - df["prev_best_iou"]
    df["delta_f1"] = df["gnn_f1"] - df["prev_best_f1"]
    df["delta_auprc"] = df["gnn_auprc"] - df["prev_best_auprc"]
    df = df.sort_values("gnn_iou", ascending=False).reset_index(drop=True)

    csv_path = SUMMARY_DIR / "seen_prev_best_vs_best_gnn_by_category.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    labels = [display_category(c) for c in df["category"]]
    pn_values = (df["prev_best_iou"] * 100.0).tolist()
    gnn_values = (df["gnn_iou"] * 100.0).tolist()
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = closed(angles)

    tomato = "#ff6347"
    cyan = "#00bcd4"
    fig = plt.figure(figsize=(14.5, 11.2), dpi=180)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles_closed, closed(pn_values), color=tomato, linewidth=2.8, label="Previous best: PointNet Mean crop DINO+geom+q")
    ax.fill(angles_closed, closed(pn_values), color=tomato, alpha=0.16)
    ax.plot(angles_closed, closed(gnn_values), color=cyan, linewidth=2.8, label="Best seen: crop DINO direct GNN")
    ax.fill(angles_closed, closed(gnn_values), color=cyan, alpha=0.16)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#8a8a8a", fontsize=12)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold", color="#565656")
    ax.tick_params(axis="x", pad=18)
    ax.grid(color="#d6d6d6", linewidth=1.15, alpha=0.85)
    ax.spines["polar"].set_color("#d0d0d0")
    ax.spines["polar"].set_linewidth(1.5)

    fig.suptitle("Seen Per-Category mIoU Comparison", fontsize=27, fontweight="bold", color="#555555", y=0.965)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=2.6,
    )
    fig.subplots_adjust(top=0.81, bottom=0.05, left=0.04, right=0.96)

    png = SUMMARY_DIR / "seen_prev_best_vs_best_gnn_radar_iou.png"
    pdf = SUMMARY_DIR / "seen_prev_best_vs_best_gnn_radar_iou.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"png={png}")
    print(f"pdf={pdf}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
