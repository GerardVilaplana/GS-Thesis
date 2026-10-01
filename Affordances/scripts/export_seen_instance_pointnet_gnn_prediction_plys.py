#!/usr/bin/env python3
"""Export HANDAL seen-instance prediction PLYs for PointNet and GNN runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
FEATURE_ROOTS = [
    BASE / "data" / "handal_handle_generalization_features" / "npz",
    BASE / "data" / "handal_new_categories_delta_v1_features" / "npz",
]
OUT_ROOT = BASE / "outputs" / "03_handle_generalization" / "22_seen_instance_pointnet_gnn_prediction_plys_v1"

POINTNET_DIR = (
    BASE
    / "outputs"
    / "03_handle_generalization"
    / "12_17cat_stage9_gaussian_native_pointnet_v1"
    / "01_seen_instance_17cat"
    / "pointnet_xyzrgb"
)
GNN_DIR = (
    BASE
    / "outputs"
    / "03_handle_generalization"
    / "15_graph_attention_gaussian_v1"
    / "01_seen_instance_17cat"
    / "graph_attention_geometry_color_scene_norm"
)

TARGET_CATEGORIES = [
    "hammers",
    "screwdrivers",
    "fixed_joint_pliers",
    "slip_joint_pliers",
    "locking_pliers",
    "mugs",
]
C0 = 0.28209479177387814


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_threshold(run_dir: Path) -> float:
    with (run_dir / "overall_metrics.json").open() as f:
        data = json.load(f)
    return float(data.get("selected_threshold", data.get("threshold", data.get("overall", {}).get("selected_threshold"))))


def load_predictions(run_dir: Path) -> dict[str, np.ndarray]:
    z = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
    out = {}
    for key, (lo, hi) in zip(z["scene_keys"], z["offsets"]):
        out[str(key)] = z["scores"][int(lo) : int(hi)].astype(np.float32, copy=False)
    return out


def select_scene_keys(pointnet_rows: list[dict], per_category: int = 3) -> list[dict]:
    selected = []
    for category in TARGET_CATEGORIES:
        rows = [r for r in pointnet_rows if r["category"] == category]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: float(r["iou"]))
        if len(rows) <= per_category:
            picks = [(f"rank{i}", r) for i, r in enumerate(rows)]
        else:
            picks = [("worst", rows[0]), ("median", rows[len(rows) // 2]), ("best", rows[-1])]
        for role, row in picks:
            row = dict(row)
            row["selection_role"] = role
            selected.append(row)
    return selected


def rgb_to_sh(rgb_01: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb_01, dtype=np.float32) - 0.5) / C0


def red_blue(scores: np.ndarray, threshold: float) -> np.ndarray:
    positive = scores >= threshold
    rgb = np.full((len(scores), 3), np.array([0.02, 0.16, 1.0], dtype=np.float32))
    rgb[positive] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    return rgb


def write_colored_ply(source_ply: Path, out_path: Path, rgb_01: np.ndarray) -> None:
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(rgb_01):
        raise ValueError(f"Length mismatch for {source_ply}: ply={len(vertices)} scores={len(rgb_01)}")
    sh = rgb_to_sh(rgb_01)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertices, "vertex"))
        else:
            elements.append(element)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(out_path))


def source_object_ply(scene_key: str) -> Path:
    npz_path = None
    for root in FEATURE_ROOTS:
        candidate = root / f"{scene_key}.npz"
        if candidate.exists():
            npz_path = candidate
            break
    if npz_path is None:
        raise FileNotFoundError(f"{scene_key}.npz not found in {[str(root) for root in FEATURE_ROOTS]}")
    with np.load(npz_path, allow_pickle=False) as z:
        return Path(str(z["source_object_ply"][0]))


def export_model(model_name: str, run_dir: Path, selected: list[dict]) -> list[dict]:
    threshold = load_threshold(run_dir)
    predictions = load_predictions(run_dir)
    rows = []
    for item in selected:
        scene_key = item["scene_key"]
        if scene_key not in predictions:
            continue
        scores = predictions[scene_key]
        source_ply = source_object_ply(scene_key)
        out_path = (
            OUT_ROOT
            / model_name
            / item["category"]
            / f"{item['selection_role']}__{scene_key}__{model_name}_pred_thr{threshold:.2f}.ply"
        )
        write_colored_ply(source_ply, out_path, red_blue(scores, threshold))
        rows.append(
            {
                "model": model_name,
                "run_dir": str(run_dir),
                "category": item["category"],
                "scene_key": scene_key,
                "selection_role": item["selection_role"],
                "source_object_ply": str(source_ply),
                "output_ply": str(out_path),
                "threshold": threshold,
                "num_gaussians": int(len(scores)),
                "pred_positive_ratio": float((scores >= threshold).mean()),
                "pointnet_selection_iou": float(item["iou"]),
            }
        )
        print(f"[{model_name}] {item['category']} {item['selection_role']} {scene_key} -> {out_path.name}")
    return rows


def save_manifest(rows: list[dict], selected: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for row in rows for k in row.keys()})
        with (OUT_ROOT / "prediction_ply_manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    (OUT_ROOT / "selection_manifest.json").write_text(
        json.dumps(
            {
                "output_root": str(OUT_ROOT),
                "selection_rule": "For each requested category, export worst/median/best test scenes by PointNet scene IoU.",
                "category_note": "'joint pliers' is represented by both fixed_joint_pliers and slip_joint_pliers; locking_pliers is separate.",
                "target_categories": TARGET_CATEGORIES,
                "models": {
                    "pointnet_xyzrgb_maxmean_seen17": str(POINTNET_DIR),
                    "graph_attention_geometry_color_scene_norm_seen17": str(GNN_DIR),
                },
                "selected_scenes": selected,
                "exports": rows,
            },
            indent=2,
        )
    )


def main() -> None:
    pointnet_rows = load_rows(POINTNET_DIR / "per_scene_metrics.csv")
    selected = select_scene_keys(pointnet_rows)
    rows = []
    rows.extend(export_model("pointnet_xyzrgb_maxmean_seen17", POINTNET_DIR, selected))
    rows.extend(export_model("graph_attention_geometry_color_scene_norm_seen17", GNN_DIR, selected))
    save_manifest(rows, selected)
    print(f"[done] {OUT_ROOT}")
    print(f"[done] exported {len(rows)} PLY files")


if __name__ == "__main__":
    main()
