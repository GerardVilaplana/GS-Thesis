#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
EXP_ROOT = ROOT / "outputs" / "03_handle_generalization" / "41_exp40_architecture_baselines_v1"
OUT_DIR = EXP_ROOT / "summary_plots"

MODEL_ORDER = ["mlp", "pointnet_max", "pointnet_mean", "pointnet_max_mean"]
MODEL_LABELS = {
    "mlp": "MLP",
    "pointnet_max": "PointNet Max",
    "pointnet_mean": "PointNet Mean",
    "pointnet_max_mean": "PointNet MaxMean",
}


def load_metric(path: Path) -> dict:
    with path.open() as handle:
        data = json.load(handle)
    split_group = path.relative_to(EXP_ROOT).parts[0]
    return {
        "path": str(path),
        "run_name": data["run_name"],
        "split_group": split_group,
        "model": data["model"],
        "miou": float(data["macro_scene_average"]["iou"]),
    }


def main() -> None:
    rows = [load_metric(path) for path in EXP_ROOT.rglob("overall_metrics.json")]
    seen = [row for row in rows if row["split_group"] == "01_seen_instance_17cat"]
    unseen = [row for row in rows if row["split_group"] == "02_leave_one_category_out_17cat"]

    summary_rows = []
    for model in MODEL_ORDER:
        model_seen = [row["miou"] for row in seen if row["model"] == model]
        model_unseen = [row["miou"] for row in unseen if row["model"] == model]
        if len(model_seen) != 1 or len(model_unseen) != 17:
            raise RuntimeError(f"Unexpected counts for {model}: seen={len(model_seen)} unseen={len(model_unseen)}")
        summary_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seen_miou": model_seen[0],
                "unseen_loco_mean_miou": float(np.mean(model_unseen)),
                "unseen_loco_std_miou": float(np.std(model_unseen, ddof=0)),
                "unseen_loco_categories": len(model_unseen),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "exp41_seen_unseen_miou_by_model.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    x = np.arange(2)
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]
    for i, row in enumerate(summary_rows):
        values = [row["seen_miou"], row["unseen_loco_mean_miou"]]
        offset = (i - (len(summary_rows) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=row["model_label"], color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(["Seen", "Unseen LOCO"])
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.0)
    ax.set_title("Exp41 Handle Prediction mIoU")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    png_path = OUT_DIR / "exp41_seen_unseen_miou_by_model.png"
    pdf_path = OUT_DIR / "exp41_seen_unseen_miou_by_model.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)

    print(f"csv={csv_path}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")
    for row in summary_rows:
        print(
            f"{row['model_label']}: seen={row['seen_miou']:.4f}, "
            f"unseen={row['unseen_loco_mean_miou']:.4f}"
        )


if __name__ == "__main__":
    main()
