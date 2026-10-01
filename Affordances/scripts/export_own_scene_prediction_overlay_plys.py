#!/usr/bin/env python3
"""Export alpha-blended own-scene prediction PLYs for thesis visualization."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
OUT_DIR = BASE / "outputs/06_own_scenes/objects_plys_overlaymask"
C0 = 0.28209479177387814
ALPHA = 0.70


def sh_to_rgb(sh: np.ndarray) -> np.ndarray:
    return np.clip(sh * C0 + 0.5, 0.0, 1.0).astype(np.float32)


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def vertex_rgb(vertex: np.ndarray) -> np.ndarray:
    sh = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
    return sh_to_rgb(sh)


def write_overlay(truecolor_ply: Path, maskcolor_ply: Path, out_ply: Path) -> dict:
    true_ply = PlyData.read(str(truecolor_ply))
    mask_ply = PlyData.read(str(maskcolor_ply))
    true_vertices = np.array(true_ply["vertex"].data, copy=True)
    mask_vertices = np.array(mask_ply["vertex"].data, copy=False)
    if len(true_vertices) != len(mask_vertices):
        raise ValueError(
            f"Length mismatch: {truecolor_ply} has {len(true_vertices)}, "
            f"{maskcolor_ply} has {len(mask_vertices)}"
        )

    base_rgb = vertex_rgb(true_vertices)
    mask_rgb = vertex_rgb(mask_vertices)
    overlay_rgb = (1.0 - ALPHA) * base_rgb + ALPHA * mask_rgb
    overlay_sh = rgb_to_sh(overlay_rgb)
    true_vertices["f_dc_0"] = overlay_sh[:, 0]
    true_vertices["f_dc_1"] = overlay_sh[:, 1]
    true_vertices["f_dc_2"] = overlay_sh[:, 2]

    elements = [
        PlyElement.describe(true_vertices, "vertex") if elem.name == "vertex" else elem
        for elem in true_ply.elements
    ]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=true_ply.text, byte_order=true_ply.byte_order).write(str(out_ply))

    # Hard prediction colors are blended, but still useful to report the handle ratio.
    green = np.array([0.02, 0.82, 0.12], dtype=np.float32)
    blue = np.array([0.02, 0.16, 1.0], dtype=np.float32)
    pred_handle = np.linalg.norm(mask_rgb - green, axis=1) < np.linalg.norm(mask_rgb - blue, axis=1)
    return {
        "truecolor_ply": str(truecolor_ply),
        "maskcolor_ply": str(maskcolor_ply),
        "output_ply": str(out_ply),
        "num_gaussians": len(true_vertices),
        "pred_handle_ratio": float(pred_handle.mean()) if len(pred_handle) else 0.0,
    }


def scene_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["scene_key"]: row
        for row in rows
        if row.get("model") == "pointnet_mean_seen"
    }


def main() -> None:
    scene1 = scene_manifest(
        BASE
        / "outputs/06_own_scenes/scene_1_15k_per_id_reproj_best_dino_handle_predictions"
        / "prediction_manifest.csv"
    )
    scene5 = scene_manifest(
        BASE
        / "outputs/06_own_scenes/scene_5_15k_per_id_reproj_best_dino_handle_predictions"
        / "prediction_manifest.csv"
    )

    specs: list[tuple[str, Path, Path]] = []

    for scene_key in sorted(scene1):
        row = scene1[scene_key]
        specs.append(
            (
                f"scene1_{scene_key.replace('scene1_', '')}_overlay_a0.70.ply",
                Path(row["source_object_ply"]),
                Path(row["threshold_ply"]),
            )
        )

    for scene_key, name in [
        ("scene5_realm_id002", "scene5_spoon_realm_id002_overlay_a0.70.ply"),
        ("scene5_realm_id003", "scene5_whisk_realm_id003_overlay_a0.70.ply"),
    ]:
        row = scene5[scene_key]
        specs.append((name, Path(row["source_object_ply"]), Path(row["threshold_ply"])))

    aligned = (
        BASE
        / "outputs/06_own_scenes/scene_5_15k_strainer_corrected_clean_aligned_to_original_scene"
    )
    specs.append(
        (
            "scene5_strainer_corrected_clean_aligned_overlay_a0.70.ply",
            aligned / "corrected_clean_object_aligned_to_original_scene5.ply",
            aligned / "corrected_clean_prediction_thr058_aligned_to_original_scene5.ply",
        )
    )

    rows = []
    for out_name, truecolor_ply, maskcolor_ply in specs:
        if not truecolor_ply.exists():
            raise FileNotFoundError(truecolor_ply)
        if not maskcolor_ply.exists():
            raise FileNotFoundError(maskcolor_ply)
        rows.append(write_overlay(truecolor_ply, maskcolor_ply, OUT_DIR / out_name))

    manifest = OUT_DIR / "overlay_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} overlay PLYs to {OUT_DIR}")
    print(manifest)


if __name__ == "__main__":
    main()
