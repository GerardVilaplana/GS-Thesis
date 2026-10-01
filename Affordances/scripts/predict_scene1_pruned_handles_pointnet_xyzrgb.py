#!/usr/bin/env python3
"""Run the HANDAL PointNet handle predictor on scene_1 REALM-pruned objects."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from train_handal_17cat_mlp_baselines import normalize_xyz, sigmoid_np  # noqa: E402
from train_handal_17cat_pointnet_global import PointNetGlobal  # noqa: E402


C0 = 0.28209479177387814
MODEL_PATH = (
    BASE
    / "outputs"
    / "03_handle_generalization"
    / "12_17cat_stage9_gaussian_native_pointnet_v1"
    / "01_seen_instance_17cat"
    / "pointnet_xyzrgb"
    / "model.pt"
)
IN_ROOT = (
    BASE
    / "outputs"
    / "06_own_scenes"
    / "scene_1_realm_query_pruning"
    / "final_ply"
)
OUT_ROOT = (
    BASE
    / "outputs"
    / "06_own_scenes"
    / "scene_1_handle_predictions_pointnet_xyzrgb"
)


MAIN_OBJECTS = [
    ("hammer", IN_ROOT / "hammer_object_truecolor_3dgs.ply"),
    ("screwdriver", IN_ROOT / "screwdriver_object_truecolor_3dgs.ply"),
    ("joint_plier", IN_ROOT / "joint_plier_object_truecolor_3dgs.ply"),
    ("cutting_plier", IN_ROOT / "cutting_plier_object_truecolor_3dgs.ply"),
]


def collect_per_id_objects() -> list[tuple[str, Path]]:
    per_id_root = IN_ROOT / "per_id_truecolor"
    if not per_id_root.exists():
        return []
    return [(p.stem.replace("_truecolor_3dgs", ""), p) for p in sorted(per_id_root.rglob("*_truecolor_3dgs.ply"))]


def rgb_to_sh(rgb_01: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb_01, dtype=np.float32) - 0.5) / C0


def base_color(vertices: np.ndarray) -> np.ndarray:
    names = vertices.dtype.names
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        color = np.vstack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).T
        return np.clip(color.astype(np.float32) * C0 + 0.5, 0.0, 1.0)
    if {"red", "green", "blue"}.issubset(names):
        return np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0
    return np.full((len(vertices), 3), 0.5, dtype=np.float32)


def score_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    blue = np.array([0.02, 0.16, 1.0], dtype=np.float32)
    cyan = np.array([0.0, 0.85, 1.0], dtype=np.float32)
    yellow = np.array([1.0, 0.92, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    rgb = np.empty((len(values), 3), dtype=np.float32)

    low = values < 0.5
    mid = (values >= 0.5) & (values < 0.75)
    high = values >= 0.75
    if low.any():
        t = (values[low] / 0.5)[:, None]
        rgb[low] = blue * (1.0 - t) + cyan * t
    if mid.any():
        t = ((values[mid] - 0.5) / 0.25)[:, None]
        rgb[mid] = cyan * (1.0 - t) + yellow * t
    if high.any():
        t = ((values[high] - 0.75) / 0.25)[:, None]
        rgb[high] = yellow * (1.0 - t) + red * t
    return np.clip(rgb, 0.0, 1.0)


def logit_heatmap(logits: np.ndarray, threshold_logit: float) -> tuple[np.ndarray, dict]:
    logits = np.asarray(logits, dtype=np.float32)
    lo, hi = np.percentile(logits, [2, 98])
    lo = float(min(lo, threshold_logit))
    hi = float(max(hi, threshold_logit))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    scaled = np.clip((logits - lo) / (hi - lo), 0.0, 1.0)
    return score_heatmap(scaled), {"logit_color_min": lo, "logit_color_max": hi}


def red_blue(scores: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.asarray(scores) >= threshold
    rgb = np.full((len(scores), 3), np.array([0.02, 0.16, 1.0], dtype=np.float32))
    rgb[labels] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    return rgb


def write_colored_ply(ply: PlyData, vertices: np.ndarray, out_path: Path, rgb_01: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_vertices = np.array(vertices, copy=True)
    sh = rgb_to_sh(rgb_01)
    out_vertices["f_dc_0"] = sh[:, 0]
    out_vertices["f_dc_1"] = sh[:, 1]
    out_vertices["f_dc_2"] = sh[:, 2]
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(out_vertices, "vertex"))
        else:
            elements.append(element)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(out_path))


def load_model(device: torch.device):
    pack = torch.load(MODEL_PATH, map_location=device)
    model = PointNetGlobal(int(pack["input_dim"]), latent_dim=int(pack["latent_dim"])).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    return (
        model,
        np.asarray(pack["mean"], dtype=np.float32),
        np.asarray(pack["std"], dtype=np.float32),
        float(pack["threshold"]),
        pack,
    )


@torch.no_grad()
def predict_logits_scores(model, vertices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device):
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    color = base_color(vertices).astype(np.float32)
    x = np.concatenate([normalize_xyz(xyz), color], axis=1)
    x = ((x - mean) / std).astype(np.float32)
    logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().astype(np.float32)
    scores = sigmoid_np(logits).astype(np.float32)
    return logits, scores


def process_one(group: str, name: str, source_ply: Path, model, mean, std, threshold, device):
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    logits, scores = predict_logits_scores(model, vertices, mean, std, device)
    threshold_logit = float(np.log(threshold / (1.0 - threshold)))

    logit_rgb, logit_meta = logit_heatmap(logits, threshold_logit)
    logit_path = OUT_ROOT / group / "logit_heatmap_blue_low_red_high" / f"{name}_handle_logit_heatmap.ply"
    pred_path = OUT_ROOT / group / "threshold_red_blue" / f"{name}_handle_pred_thr{threshold:.2f}.ply"
    prob_path = OUT_ROOT / group / "probability_heatmap_blue0_red1" / f"{name}_handle_probability_heatmap.ply"

    write_colored_ply(ply, vertices, logit_path, logit_rgb)
    write_colored_ply(ply, vertices, pred_path, red_blue(scores, threshold))
    write_colored_ply(ply, vertices, prob_path, score_heatmap(scores))

    return {
        "group": group,
        "object": name,
        "source_ply": str(source_ply),
        "num_gaussians": int(len(vertices)),
        "threshold": float(threshold),
        "threshold_logit": threshold_logit,
        "pred_positive_ratio": float((scores >= threshold).mean()),
        "logit_min": float(logits.min()),
        "logit_mean": float(logits.mean()),
        "logit_max": float(logits.max()),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        **logit_meta,
        "logit_heatmap_ply": str(logit_path),
        "probability_heatmap_ply": str(prob_path),
        "threshold_red_blue_ply": str(pred_path),
    }


def save_manifest(rows, model_pack):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    csv_path = OUT_ROOT / "prediction_manifest.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT_ROOT / "prediction_manifest.json"
    json_path.write_text(
        json.dumps(
            {
                "model_path": str(MODEL_PATH),
                "feature_variant": model_pack.get("feature_variant"),
                "architecture": model_pack.get("overall", {}).get("architecture"),
                "threshold": float(model_pack["threshold"]),
                "note": "Main objects use combined REALM-pruned query PLYs. per_id uses each selected REALM ID as its own diagnostic object.",
                "objects": rows,
            },
            indent=2,
        )
    )
    return csv_path, json_path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, threshold, model_pack = load_model(device)
    rows = []

    for name, path in MAIN_OBJECTS:
        if path.exists():
            row = process_one("combined_objects", name, path, model, mean, std, threshold, device)
            rows.append(row)
            print(f"[combined] {name}: N={row['num_gaussians']} pred_ratio={row['pred_positive_ratio']:.3f}")

    for name, path in collect_per_id_objects():
        row = process_one("per_id_objects", name, path, model, mean, std, threshold, device)
        rows.append(row)
        print(f"[per_id] {name}: N={row['num_gaussians']} pred_ratio={row['pred_positive_ratio']:.3f}")

    csv_path, json_path = save_manifest(rows, model_pack)
    print(f"[done] {csv_path}")
    print(f"[done] {json_path}")


if __name__ == "__main__":
    main()
