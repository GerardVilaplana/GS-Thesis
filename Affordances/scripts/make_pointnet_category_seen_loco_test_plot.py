#!/usr/bin/env python3
"""Plot final PointNet per-category IoU using test scenes only."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
RUN_ROOT = (
    ROOT
    / "outputs/03_handle_generalization"
    / "47_exp46_dino_pointnet_mean_feature_sweep_v1"
    / "object_crop_center_avg"
)
CONFIG = Path("dino_geometry_color_quality/pointnet_mean")
SEEN_CSV = RUN_ROOT / "01_seen_instance_17cat" / CONFIG / "per_category_metrics.csv"
LOCO_ROOT = RUN_ROOT / "02_leave_one_category_out_17cat"
OUT_DIR = ROOT / "outputs/thesis_figures/results_plots/categories"

CATEGORY_LABELS = {
    "adjustable_wrenches": "Adjustable wrenches",
    "combinational_wrenches": "Combination wrenches",
    "fixed_joint_pliers": "Fixed joint pliers",
    "hammers": "Hammers",
    "ladles": "Ladles",
    "locking_pliers": "Locking pliers",
    "measuring_cups": "Measuring cups",
    "mugs": "Mugs",
    "pots_pans": "Pots/pans",
    "power_drills": "Power drills",
    "ratchets": "Ratchets",
    "screwdrivers": "Screwdrivers",
    "slip_joint_pliers": "Slip joint pliers",
    "spatulas": "Spatulas",
    "strainers": "Strainers",
    "utensils": "Utensils",
    "whisks": "Whisks",
}


def load_seen() -> tuple[dict[str, float], dict[str, int]]:
    values: dict[str, float] = {}
    counts: dict[str, int] = {}
    with SEEN_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            category = row["category"]
            values[category] = float(row["macro_scene_iou"])
            counts[category] = int(row["num_scenes"])
    return values, counts


def load_loco_test() -> tuple[dict[str, float], dict[str, int]]:
    values: dict[str, float] = {}
    counts: dict[str, int] = {}
    for category in CATEGORY_LABELS:
        path = LOCO_ROOT / category / CONFIG / "per_scene_metrics.csv"
        with path.open(newline="") as handle:
            test_ious = [
                float(row["iou"])
                for row in csv.DictReader(handle)
                if row["source_split"] == "test"
            ]
        if not test_ious:
            raise RuntimeError(f"No test scenes found for {category}: {path}")
        values[category] = float(np.mean(test_ious))
        counts[category] = len(test_ious)
    return values, counts


def main() -> None:
    seen, seen_counts = load_seen()
    loco, loco_counts = load_loco_test()

    expected = set(CATEGORY_LABELS)
    if set(seen) != expected or set(loco) != expected:
        raise RuntimeError("Seen and LOCO category sets do not match the 17 expected categories")
    if sum(seen_counts.values()) != 89 or sum(loco_counts.values()) != 89:
        raise RuntimeError(
            f"Expected 89 test scenes, found seen={sum(seen_counts.values())}, "
            f"LOCO={sum(loco_counts.values())}"
        )
    if seen_counts != loco_counts:
        raise RuntimeError("Seen and LOCO evaluations do not contain the same test scenes")

    # Keep the original figure convention: categories are sorted by LOCO IoU.
    order = sorted(CATEGORY_LABELS, key=loco.get, reverse=True)
    labels = [CATEGORY_LABELS[category] for category in order]
    seen_values = np.array([seen[category] for category in order])
    loco_values = np.array([loco[category] for category in order])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "selected_pointnet_category_values.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", "category", "num_scenes", "iou"])
        writer.writeheader()
        for setting, values, counts in (
            ("seen", seen, seen_counts),
            ("loco", loco, loco_counts),
        ):
            for category in order:
                writer.writerow(
                    {
                        "setting": setting,
                        "category": CATEGORY_LABELS[category],
                        "num_scenes": counts[category],
                        "iou": values[category],
                    }
                )

    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    y = np.arange(len(order))
    height = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 8.0), dpi=180)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4E4E4", linewidth=0.8)
    ax.barh(
        y - height / 2,
        seen_values,
        height,
        label="Seen",
        color="#6F9EC7",
        edgecolor="#666666",
        linewidth=0.5,
    )
    ax.barh(
        y + height / 2,
        loco_values,
        height,
        label="LOCO",
        color="#C9796C",
        edgecolor="#666666",
        linewidth=0.5,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("IoU")
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()

    png_path = OUT_DIR / "pointnet_category_seen_loco_comparison.png"
    pdf_path = OUT_DIR / "pointnet_category_seen_loco_comparison.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"seen_test_scenes={sum(seen_counts.values())}")
    print(f"loco_test_scenes={sum(loco_counts.values())}")
    print(f"seen_category_macro_iou={np.mean(list(seen.values())):.6f}")
    print(f"loco_category_macro_iou={np.mean(list(loco.values())):.6f}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
