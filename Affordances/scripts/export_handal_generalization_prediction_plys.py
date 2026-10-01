import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


C0 = 0.28209479177387814
FEATURE_ROOT = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features"
)
MLP_ROOT = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/"
    "01_per_gaussian_mlp_baseline"
)


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def color_prediction(scores, threshold):
    scores = np.asarray(scores, dtype=np.float32)
    rgb = np.zeros((len(scores), 3), dtype=np.float32)
    rgb[:, 0] = np.maximum(scores, 0.08)
    rgb[:, 1] = 0.08 * (1.0 - scores)
    rgb[:, 2] = 1.0 - scores
    uncertain = np.abs(scores - threshold) < 0.05
    rgb[uncertain] = np.array([1.0, 0.85, 0.05], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def color_ground_truth(labels):
    labels = np.asarray(labels).astype(bool)
    rgb = np.full((len(labels), 3), np.array([0.12, 0.18, 0.85], dtype=np.float32))
    rgb[labels] = np.array([1.0, 0.03, 0.02], dtype=np.float32)
    return rgb


def write_colored_ply(source_ply, out_ply, rgb):
    ply = PlyData.read(source_ply)
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(rgb):
        raise ValueError(f"Length mismatch for {source_ply}: ply={len(vertices)} rgb={len(rgb)}")
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(out_ply)


def load_threshold(overall_path):
    with open(overall_path, "r") as f:
        return float(json.load(f)["selected_threshold"])


def export_baseline(experiment_dir, baseline_dir, object_ply_root, out_root):
    pred_npz = np.load(baseline_dir / "test_predictions.npz", allow_pickle=False)
    scene_keys = [str(x) for x in pred_npz["scene_keys"]]
    offsets = pred_npz["offsets"]
    scores = pred_npz["scores"]
    labels = pred_npz["labels"]
    threshold = load_threshold(baseline_dir / "overall_metrics.json")
    baseline = baseline_dir.name
    experiment = experiment_dir.name

    pred_out = out_root / experiment / baseline
    gt_out = out_root / experiment / "ground_truth_handle_red"
    rows = []
    for scene_key, (start, end) in zip(scene_keys, offsets):
        source_ply = object_ply_root / f"{scene_key}_object_pruned_thr0.60.ply"
        if not source_ply.exists():
            raise FileNotFoundError(source_ply)
        scene_scores = scores[start:end]
        scene_labels = labels[start:end]
        pred_ply = pred_out / f"{scene_key}_{baseline}_prediction.ply"
        gt_ply = gt_out / f"{scene_key}_gt_handle_red.ply"
        write_colored_ply(source_ply, pred_ply, color_prediction(scene_scores, threshold))
        if not gt_ply.exists():
            write_colored_ply(source_ply, gt_ply, color_ground_truth(scene_labels))
        rows.append(
            {
                "scene_key": scene_key,
                "baseline": baseline,
                "prediction_ply": str(pred_ply),
                "ground_truth_ply": str(gt_ply),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--mlp_root", type=Path, default=MLP_ROOT)
    parser.add_argument("--out_dir", type=Path, default=MLP_ROOT / "prediction_ply")
    args = parser.parse_args()

    object_ply_root = args.feature_root / "work" / "object_ply"
    all_rows = []
    for experiment_dir in sorted(args.mlp_root.glob("exp*")):
        if not experiment_dir.is_dir():
            continue
        for baseline_dir in sorted(experiment_dir.iterdir()):
            if not baseline_dir.is_dir():
                continue
            if not (baseline_dir / "test_predictions.npz").exists():
                continue
            rows = export_baseline(experiment_dir, baseline_dir, object_ply_root, args.out_dir)
            all_rows.extend(rows)
            print(f"exported {len(rows)} PLYs for {experiment_dir.name}/{baseline_dir.name}")

    manifest = args.out_dir / "prediction_ply_manifest.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"wrote manifest: {manifest}")
    print(f"total prediction entries: {len(all_rows)}")


if __name__ == "__main__":
    main()
