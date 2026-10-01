#!/usr/bin/env python3
"""Export red/blue handle predictions for scene_1 REALM per-ID object PLYs."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from train_handal_17cat_mlp_baselines import normalize_xyz, sigmoid_np  # noqa: E402
from train_handal_17cat_pointnet_global import PointNetGlobal  # noqa: E402
from train_handal_stage16_context_ablation import PointNetPooling  # noqa: E402


C0 = 0.28209479177387814
IN_ROOT = (
    BASE
    / "outputs"
    / "06_own_scenes"
    / "scene_1_realm_query_pruning"
    / "final_ply"
    / "per_id_truecolor"
)
OUT_ROOT = (
    BASE
    / "outputs"
    / "06_own_scenes"
    / "scene_1_handle_predictions_nondino_model_comparison"
)
STAGE12 = BASE / "outputs" / "03_handle_generalization" / "12_17cat_stage9_gaussian_native_pointnet_v1"
STAGE16 = BASE / "outputs" / "03_handle_generalization" / "16_context_denoise_ablation_v1"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    architecture: str
    feature_variant: str
    pooling: str = "max_mean"


MODEL_SPECS = [
    ModelSpec(
        name="pointnet_xyzrgb_maxmean_seen17",
        checkpoint=STAGE12 / "01_seen_instance_17cat" / "pointnet_xyzrgb" / "model.pt",
        architecture="stage12_pointnet_global",
        feature_variant="xyzrgb",
        pooling="max_mean",
    ),
    ModelSpec(
        name="pointnet_current_geometry_color_maxmean_seen17",
        checkpoint=STAGE12 / "01_seen_instance_17cat" / "pointnet_current_geometry_color" / "model.pt",
        architecture="stage12_pointnet_global",
        feature_variant="current_geometry_color",
        pooling="max_mean",
    ),
    ModelSpec(
        name="pointnet_no_scale_opacity_maxmean_seen17",
        checkpoint=STAGE12 / "01_seen_instance_17cat" / "pointnet_geometry_color_no_scale_opacity" / "model.pt",
        architecture="stage12_pointnet_global",
        feature_variant="geometry_color_no_scale_opacity",
        pooling="max_mean",
    ),
    ModelSpec(
        name="pointnet_gcnorm_mean_seen17_stage16",
        checkpoint=STAGE16
        / "01_seen_instance_17cat"
        / "pointnet__geometry_color_scene_norm__prep-none__pool-mean"
        / "model.pt",
        architecture="stage16_pointnet_pooling",
        feature_variant="geometry_color_scene_norm",
        pooling="mean",
    ),
    ModelSpec(
        name="pointnet_gcnorm_max_seen17_stage16",
        checkpoint=STAGE16
        / "01_seen_instance_17cat"
        / "pointnet__geometry_color_scene_norm__prep-none__pool-max"
        / "model.pt",
        architecture="stage16_pointnet_pooling",
        feature_variant="geometry_color_scene_norm",
        pooling="max",
    ),
    ModelSpec(
        name="pointnet_gcnorm_maxmean_seen17_stage16",
        checkpoint=STAGE16
        / "01_seen_instance_17cat"
        / "pointnet__geometry_color_scene_norm__prep-none__pool-max_mean"
        / "model.pt",
        architecture="stage16_pointnet_pooling",
        feature_variant="geometry_color_scene_norm",
        pooling="max_mean",
    ),
]


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


def scene_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return ((x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), eps)).astype(np.float32)


def scene_minmax_unit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    return ((x - lo) / np.maximum(hi - lo, eps)).astype(np.float32)


def field(vertices: np.ndarray, names: list[str], default: float = 0.0) -> np.ndarray:
    dtype_names = set(vertices.dtype.names or [])
    cols = []
    for name in names:
        if name in dtype_names:
            cols.append(vertices[name].astype(np.float32))
        else:
            cols.append(np.full(len(vertices), default, dtype=np.float32))
    return np.vstack(cols).T.astype(np.float32, copy=False)


def feature_matrix(vertices: np.ndarray, variant: str) -> np.ndarray:
    xyz = field(vertices, ["x", "y", "z"])
    xyz_norm = normalize_xyz(xyz)
    color = base_color(vertices).astype(np.float32)
    scale = field(vertices, ["scale_0", "scale_1", "scale_2"])
    rotation = field(vertices, ["rot_0", "rot_1", "rot_2", "rot_3"])
    opacity = field(vertices, ["opacity"])

    if variant == "xyzrgb":
        x = np.concatenate([xyz_norm, color], axis=1)
    elif variant == "current_geometry_color":
        x = np.concatenate([xyz_norm, scale, rotation, opacity, color], axis=1)
    elif variant == "geometry_color_no_scale_opacity":
        x = np.concatenate([xyz_norm, rotation, color], axis=1)
    elif variant == "geometry_color_scene_norm":
        x = np.concatenate(
            [
                xyz_norm,
                scene_zscore(scale),
                rotation,
                scene_minmax_unit(opacity),
                scene_zscore(color),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported feature variant: {variant}")
    return x.astype(np.float32, copy=False)


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


def load_model(spec: ModelSpec, device: torch.device):
    pack = torch.load(spec.checkpoint, map_location=device)
    input_dim = int(pack["input_dim"])
    latent_dim = int(pack["latent_dim"])
    if spec.architecture == "stage16_pointnet_pooling":
        model = PointNetPooling(input_dim, latent_dim=latent_dim, pooling=spec.pooling).to(device)
    else:
        model = PointNetGlobal(input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    return model, np.asarray(pack["mean"], dtype=np.float32), np.asarray(pack["std"], dtype=np.float32), float(pack["threshold"]), pack


@torch.no_grad()
def predict(model, vertices: np.ndarray, mean: np.ndarray, std: np.ndarray, variant: str, device: torch.device):
    x = feature_matrix(vertices, variant)
    x = ((x - mean) / std).astype(np.float32, copy=False)
    logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().astype(np.float32)
    scores = sigmoid_np(logits).astype(np.float32)
    return logits, scores


def collect_inputs() -> list[Path]:
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"Missing per-ID object folder: {IN_ROOT}")
    return sorted(IN_ROOT.glob("*_truecolor_3dgs.ply"))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    objects = collect_inputs()
    for spec in MODEL_SPECS:
        model, mean, std, threshold, pack = load_model(spec, device)
        model_out = OUT_ROOT / spec.name / "threshold_red_blue"
        for source_ply in objects:
            obj_name = source_ply.stem.replace("_truecolor_3dgs", "")
            ply = PlyData.read(str(source_ply))
            vertices = np.array(ply["vertex"].data, copy=True)
            logits, scores = predict(model, vertices, mean, std, spec.feature_variant, device)
            out_path = model_out / f"{obj_name}_{spec.name}_pred_thr{threshold:.2f}.ply"
            write_colored_ply(ply, vertices, out_path, red_blue(scores, threshold))
            row = {
                "model": spec.name,
                "source_ply": str(source_ply),
                "object_id": obj_name,
                "output_ply": str(out_path),
                "checkpoint": str(spec.checkpoint),
                "architecture": spec.architecture,
                "feature_variant": spec.feature_variant,
                "pooling": spec.pooling,
                "threshold": threshold,
                "num_gaussians": int(len(vertices)),
                "pred_positive_ratio": float((scores >= threshold).mean()),
                "score_min": float(scores.min()),
                "score_mean": float(scores.mean()),
                "score_max": float(scores.max()),
                "logit_min": float(logits.min()),
                "logit_mean": float(logits.mean()),
                "logit_max": float(logits.max()),
            }
            rows.append(row)
            print(f"[{spec.name}] {obj_name}: N={len(vertices)} pred_ratio={row['pred_positive_ratio']:.3f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    csv_path = OUT_ROOT / "prediction_manifest.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT_ROOT / "prediction_manifest.json"
    json_path.write_text(
        json.dumps(
            {
                "input_root": str(IN_ROOT),
                "output_root": str(OUT_ROOT),
                "note": "Only independent REALM-ID object PLYs are evaluated. No DINO-feature checkpoints are used.",
                "models": [{"name": spec.name, "checkpoint": str(spec.checkpoint), "architecture": spec.architecture, "feature_variant": spec.feature_variant, "pooling": spec.pooling} for spec in MODEL_SPECS],
                "predictions": rows,
            },
            indent=2,
        )
    )
    print(f"[done] {csv_path}")
    print(f"[done] {json_path}")


if __name__ == "__main__":
    main()
