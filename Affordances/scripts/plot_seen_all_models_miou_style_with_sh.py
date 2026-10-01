#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization")


def add(rows: list[tuple[str, float, str]], label: str, group: str, path: Path) -> None:
    if not path.exists():
        return
    with path.open() as handle:
        data = json.load(handle)
    rows.append((label, float(data["macro_scene_average"]["iou"]), group))


def main() -> None:
    rows: list[tuple[str, float, str]] = []

    root41 = OUT_ROOT / "41_exp40_architecture_baselines_v1" / "01_seen_instance_17cat"
    add(rows, "MLP\ngeom+color", "mlp", root41 / "mlp" / "overall_metrics.json")
    add(rows, "PN Max\ngeom+color", "pointnet_type", root41 / "pointnet_max" / "overall_metrics.json")
    add(rows, "PN Mean\ngeom+color", "pointnet_type", root41 / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN MaxMean\ngeom+color", "pointnet_type", root41 / "pointnet_max_mean" / "overall_metrics.json")

    root44 = OUT_ROOT / "44_exp40_pointnet_mean_feature_sweep_v1" / "01_seen_instance_17cat"
    add(rows, "PN Mean\nxyz+color", "features", root44 / "xyz_color" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\nxyz+scale+opac+color", "features", root44 / "xyz_scale_opacity_color" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\nold DINO+geom", "dino", root44 / "dino_geometry_color" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\nold DINO+geom+q", "dino", root44 / "dino_geometry_color_quality" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\nDINO only", "dino", root44 / "dino_only" / "pointnet_mean" / "overall_metrics.json")

    root47 = OUT_ROOT / "47_exp46_dino_pointnet_mean_feature_sweep_v1"
    add(rows, "PN Mean\ncrop DINO+geom", "dino", root47 / "object_crop_center_avg" / "01_seen_instance_17cat" / "dino_geometry_color" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\ncrop DINO+geom+q", "dino", root47 / "object_crop_center_avg" / "01_seen_instance_17cat" / "dino_geometry_color_quality" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\nfull DINO+geom", "dino", root47 / "full_center_avg" / "01_seen_instance_17cat" / "dino_geometry_color" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\nfull DINO+geom+q", "dino", root47 / "full_center_avg" / "01_seen_instance_17cat" / "dino_geometry_color_quality" / "pointnet_mean" / "overall_metrics.json")

    root53 = OUT_ROOT / "53_exp40_pointnet_mean_sh_ablation_seen_v1" / "01_seen_instance_17cat"
    add(rows, "PN Mean\nxyz+SH", "sh", root53 / "xyz_sh" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\nxyz+scale+opac+SH", "sh", root53 / "xyz_scale_opacity_sh" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\ngeom+SH", "sh", root53 / "geometry_sh_scene_norm" / "pointnet_mean" / "overall_metrics.json")
    add(rows, "PN Mean\ncrop DINO+geom+SH+q", "sh", root53 / "dino_geometry_sh_quality" / "pointnet_mean" / "overall_metrics.json")

    rows = sorted(rows, key=lambda item: item[1], reverse=True)
    out_dir = OUT_ROOT / "summary_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    palettes = {
        "dino": ["#6f94b8", "#7fa3c4", "#8eb2ce", "#9dc0d8", "#adcde1", "#bed9e9", "#cfe5f0"],
        "pointnet_type": ["#d9c275", "#dfca86", "#e5d299", "#ebdbab"],
        "features": ["#9dbb8f", "#adc9a2", "#bfd6b5"],
        "mlp": ["#c69abb", "#d3abc8"],
        "sh": ["#cf7d7d", "#d98f8f", "#e2a3a3", "#ebb6b6"],
    }
    seen_by_group = {key: 0 for key in palettes}
    colors = []
    for _, _, group in rows:
        palette = palettes[group]
        colors.append(palette[seen_by_group[group] % len(palette)])
        seen_by_group[group] += 1

    fig, ax = plt.subplots(figsize=(17, 6), dpi=180)
    bars = ax.bar(
        range(len(rows)),
        [value for _, value, _ in rows],
        color=colors,
        edgecolor="#555555",
        linewidth=0.8,
    )
    for bar, (_, value, _) in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title("Seen model comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Seen mIoU", fontsize=10, fontweight="bold")
    ax.set_xlabel("Experiment", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([label for label, _, _ in rows], rotation=35, ha="right", fontsize=8, fontweight="bold")
    ax.set_ylim(max(0.45, min(value for _, value, _ in rows) - 0.04), min(1.0, max(value for _, value, _ in rows) + 0.06))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=palettes["dino"][0], label="DINO"),
            plt.Rectangle((0, 0), 1, 1, color=palettes["pointnet_type"][0], label="PointNet type"),
            plt.Rectangle((0, 0), 1, 1, color=palettes["features"][0], label="Feature variants"),
            plt.Rectangle((0, 0), 1, 1, color=palettes["mlp"][0], label="MLP"),
            plt.Rectangle((0, 0), 1, 1, color=palettes["sh"][0], label="SH variants"),
        ],
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()

    png = out_dir / "seen_all_models_miou_barplot_exp_family4_palette_with_sh.png"
    pdf = out_dir / "seen_all_models_miou_barplot_exp_family4_palette_with_sh.pdf"
    csv = out_dir / "seen_all_models_miou_barplot_exp_family4_palette_with_sh.csv"
    fig.savefig(png)
    fig.savefig(pdf)
    with csv.open("w") as handle:
        handle.write("experiment,seen_miou,group\n")
        for label, value, group in rows:
            handle.write(f'"{label.replace(chr(10), " ")}",{value:.8f},{group}\n')

    print(f"png={png}")
    print(f"pdf={pdf}")
    print(f"csv={csv}")
    for label, value, group in rows:
        print(f"{label.replace(chr(10), ' ')}: {value:.4f} ({group})")


if __name__ == "__main__":
    main()
