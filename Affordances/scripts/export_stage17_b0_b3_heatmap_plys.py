#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
STAGE17 = BASE / "outputs" / "03_handle_generalization" / "17_mug_b0_b6_full_mugs_v1"
HANDAL_OBJECT_PLY = BASE / "data" / "handal_handle_generalization_features" / "work" / "object_ply"
OUT_ROOT = STAGE17 / "prediction_heatmap_plys" / "b0_vs_b3_requested_scenes"

C0 = 0.28209479177387814
SCENES = ["mugs__007009", "mugs__007008", "mugs__007007", "mugs__007006", "mugs__002003"]
RUNS = {
    "B0_HANDAL_only": STAGE17 / "B0_handal_only__orig_clean" / "pointnet_max_geometry_color_scene_norm",
    "B3_HANDAL_plus_opacity_match": STAGE17 / "B3_handal_plus_synth__orig_opacity_match" / "pointnet_max_geometry_color_scene_norm",
}


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def score_to_blue_red(scores):
    scores = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
    blue = np.array([0.02, 0.16, 1.0], dtype=np.float32)
    cyan = np.array([0.0, 0.85, 1.0], dtype=np.float32)
    yellow = np.array([1.0, 0.92, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.02, 0.02], dtype=np.float32)

    rgb = np.empty((len(scores), 3), dtype=np.float32)
    low = scores < 0.5
    mid = (scores >= 0.5) & (scores < 0.75)
    high = scores >= 0.75
    if low.any():
        t = (scores[low] / 0.5)[:, None]
        rgb[low] = blue * (1.0 - t) + cyan * t
    if mid.any():
        t = ((scores[mid] - 0.5) / 0.25)[:, None]
        rgb[mid] = cyan * (1.0 - t) + yellow * t
    if high.any():
        t = ((scores[high] - 0.75) / 0.25)[:, None]
        rgb[high] = yellow * (1.0 - t) + red * t
    return rgb


def load_overall(model_dir):
    with open(model_dir / "overall_metrics.json") as f:
        return json.load(f)


def load_predictions(model_dir):
    pack = np.load(model_dir / "test_predictions.npz", allow_pickle=False)
    scene_keys = [str(x) for x in pack["scene_keys"]]
    offsets = pack["offsets"]
    scores = pack["scores"].astype(np.float32)
    labels = pack["labels"].astype(np.uint8)
    out = {}
    for scene_key, (start, end) in zip(scene_keys, offsets):
        start, end = int(start), int(end)
        out[scene_key] = {
            "scores": scores[start:end],
            "labels": labels[start:end],
        }
    return out


def write_heatmap_ply(source_ply, out_ply, scores):
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(scores):
        raise ValueError(f"Length mismatch for {source_ply}: ply={len(vertices)} scores={len(scores)}")
    rgb = score_to_blue_red(scores)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_ply))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []
    for run_label, model_dir in RUNS.items():
        overall = load_overall(model_dir)
        threshold = float(overall["selected_threshold"])
        preds = load_predictions(model_dir)
        for scene_key in SCENES:
            if scene_key not in preds:
                raise KeyError(f"{scene_key} not found in {model_dir / 'test_predictions.npz'}")
            source_ply = HANDAL_OBJECT_PLY / f"{scene_key}_object_pruned_thr0.60.ply"
            scores = preds[scene_key]["scores"]
            labels = preds[scene_key]["labels"]
            out_ply = OUT_ROOT / run_label / f"{scene_key}_{run_label}_score_heatmap_blue0_red1.ply"
            write_heatmap_ply(source_ply, out_ply, scores)
            rows.append(
                {
                    "run": run_label,
                    "scene_key": scene_key,
                    "threshold": threshold,
                    "num_gaussians": int(len(scores)),
                    "score_min": float(scores.min()),
                    "score_mean": float(scores.mean()),
                    "score_max": float(scores.max()),
                    "gt_positive_ratio": float(labels.mean()),
                    "pred_positive_ratio_at_threshold": float((scores >= threshold).mean()),
                    "source_ply": str(source_ply),
                    "output_ply": str(out_ply),
                }
            )
            print(f"wrote {out_ply}")
    write_csv(OUT_ROOT / "b0_b3_requested_heatmap_manifest.csv", rows)
    print(f"wrote {OUT_ROOT / 'b0_b3_requested_heatmap_manifest.csv'}")


if __name__ == "__main__":
    main()
