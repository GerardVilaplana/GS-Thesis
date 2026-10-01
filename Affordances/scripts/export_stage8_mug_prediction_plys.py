#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
STAGE7_ROOT = BASE / "outputs" / "03_handle_generalization" / "11_mug_stage7_domain_normalization_v1"
HANDAL_OBJECT_PLY = BASE / "data" / "handal_handle_generalization_features" / "work" / "object_ply"
OUT_ROOT = STAGE7_ROOT / "stage8_prediction_plys"

C0 = 0.28209479177387814

EXPORTS = [
    ("stage7_synth_to_handal_synthval", "pointnet_xyzrgb_scene_color"),
    ("stage7_synth_to_handal_handalval", "pointnet_xyzrgb"),
    ("stage7_synth_to_handal_handalval", "pointnet_geometry_color_scene_norm"),
    ("stage7_handal_only", "pointnet_xyzrgb_scene_color"),
    ("stage7_handal_plus_synthetic", "pointnet_geometry_color_scene_norm"),
]


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def write_prediction_ply(source_ply, out_ply, scores, threshold):
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    scores = np.asarray(scores, dtype=np.float32)
    if len(vertices) != len(scores):
        raise ValueError(f"Length mismatch: {source_ply} has {len(vertices)} vertices, scores has {len(scores)}")

    pred = scores >= threshold
    rgb = np.empty((len(scores), 3), dtype=np.float32)
    rgb[~pred] = np.array([0.05, 0.18, 1.0], dtype=np.float32)
    rgb[pred] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_ply))
    return int(pred.sum()), float(pred.mean())


def load_threshold(model_dir):
    with open(model_dir / "overall_metrics.json", "r") as f:
        return float(json.load(f)["selected_threshold"])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []
    for run_name, model_name in EXPORTS:
        model_dir = STAGE7_ROOT / run_name / model_name
        pred_path = model_dir / "test_predictions.npz"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        threshold = load_threshold(model_dir)
        pack = np.load(pred_path, allow_pickle=False)
        scene_keys = [str(x) for x in pack["scene_keys"]]
        offsets = pack["offsets"]
        scores = pack["scores"].astype(np.float32)
        for scene_key, (start, end) in zip(scene_keys, offsets):
            source_ply = HANDAL_OBJECT_PLY / f"{scene_key}_object_pruned_thr0.60.ply"
            if not source_ply.exists():
                raise FileNotFoundError(source_ply)
            scene_scores = scores[int(start) : int(end)]
            out_ply = OUT_ROOT / run_name / model_name / f"{scene_key}_{run_name}_{model_name}_pred_red_blue.ply"
            pred_count, pred_ratio = write_prediction_ply(source_ply, out_ply, scene_scores, threshold)
            row = {
                "run_name": run_name,
                "model": model_name,
                "scene_key": scene_key,
                "threshold": threshold,
                "source_ply": str(source_ply),
                "output_ply": str(out_ply),
                "num_gaussians": int(end - start),
                "pred_positive_count": pred_count,
                "pred_positive_ratio": pred_ratio,
            }
            rows.append(row)
            print(f"wrote {out_ply}")
    write_csv(OUT_ROOT / "stage8_prediction_plys_manifest.csv", rows)
    print(f"wrote manifest with {len(rows)} rows: {OUT_ROOT / 'stage8_prediction_plys_manifest.csv'}")


if __name__ == "__main__":
    main()
