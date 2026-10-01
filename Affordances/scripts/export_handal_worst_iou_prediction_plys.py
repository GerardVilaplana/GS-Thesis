import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_handal_exp40_mlp import label_path, write_prediction_ply


DEFAULT_SUP_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_full_supervision")
DEFAULT_MLP_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_full_mlp")
BASELINES = ("dino_only", "geometry_only", "dino_geometry")


def read_per_scene(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key != "scene":
                row[key] = float(value)
    return rows


def load_prediction(npz_path, scene):
    data = np.load(npz_path)
    return data[f"{scene}_valid_score"], data[f"{scene}_valid_mask"].astype(bool)


def export_gt_ply(scene, sup_root, out_ply):
    labels = np.load(label_path(scene, sup_root))["is_handle"].astype(np.float32)
    valid = np.ones(len(labels), dtype=bool)
    write_prediction_ply(scene, sup_root, labels, valid, out_ply, threshold=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_root", type=Path, default=DEFAULT_MLP_ROOT)
    parser.add_argument("--sup_root", type=Path, default=DEFAULT_SUP_ROOT)
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument("--num_worst", type=int, default=5)
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES), choices=BASELINES)
    args = parser.parse_args()

    out_root = args.out_root or args.mlp_root / "worst_iou_prediction_ply"
    out_root.mkdir(parents=True, exist_ok=True)

    selected_by_baseline = {}
    selected_scenes = []
    for baseline in args.baselines:
        rows = read_per_scene(args.mlp_root / baseline / "per_scene_metrics.csv")
        worst = sorted(rows, key=lambda row: row["iou"])[: args.num_worst]
        selected_by_baseline[baseline] = worst
        for row in worst:
            if row["scene"] not in selected_scenes:
                selected_scenes.append(row["scene"])

    summary_rows = []
    for scene in selected_scenes:
        scene_dir = out_root / f"scene_{scene}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        export_gt_ply(scene, args.sup_root, scene_dir / f"handal_mug_scene_{scene}_pseudo_gt_handle_label.ply")

        for baseline in args.baselines:
            metric_rows = read_per_scene(args.mlp_root / baseline / "per_scene_metrics.csv")
            metrics = next(row for row in metric_rows if row["scene"] == scene)
            score, valid = load_prediction(args.mlp_root / baseline / "test_predictions.npz", scene)
            out_ply = scene_dir / f"handal_mug_scene_{scene}_{baseline}_prediction.ply"
            write_prediction_ply(scene, args.sup_root, score, valid, out_ply, threshold=0.5)
            summary_rows.append(
                {
                    "scene": scene,
                    "baseline": baseline,
                    "iou": metrics["iou"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "gt_handle_ratio": metrics["gt_handle_ratio"],
                    "pred_handle_ratio": metrics["pred_handle_ratio"],
                    "correct_percent": metrics["correct_percent"],
                    "prediction_ply": str(out_ply),
                }
            )

    with open(out_root / "worst_iou_selected_scenes.json", "w") as f:
        json.dump(selected_by_baseline, f, indent=2)
    with open(out_root / "worst_iou_comparison_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Exported {len(selected_scenes)} unique scenes to {out_root}")
    for baseline, rows in selected_by_baseline.items():
        print(f"{baseline} worst scenes: " + ", ".join(f"{row['scene']} ({row['iou']:.3f})" for row in rows))


if __name__ == "__main__":
    main()
