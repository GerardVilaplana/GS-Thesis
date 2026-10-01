#!/usr/bin/env python3
"""Align corrected scene-5 strainer PLYs to the original scene-5 coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


ORIG_CAMERAS = Path(
    "/home/gvilaplana/GS-Thesis/REALM-Code/output/own_scenes/"
    "scene_5_realm_1600_objpred_15k/cameras.json"
)
CORR_CAMERAS = Path(
    "/home/gvilaplana/GS-Thesis/REALM-Code/output/own_scenes/"
    "scene_5_strainer_corrected_realm_1600_objpred_15k/cameras.json"
)

CORR_PRED_ROOT = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/outputs/06_own_scenes/"
    "scene_5_strainer_corrected_15k_correctmask_best_dino_handle_predictions"
)
CORR_OBJ_ROOT = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/outputs/06_own_scenes/"
    "scene_5_strainer_corrected_15k_realm_query_pruning_correct_mask_refinement"
)
OUT_ROOT = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/outputs/06_own_scenes/"
    "scene_5_15k_strainer_corrected_clean_aligned_to_original_scene"
)


def load_camera_centers(path: Path) -> dict[str, np.ndarray]:
    cameras = json.loads(path.read_text())
    return {cam["img_name"]: np.asarray(cam["position"], dtype=np.float64) for cam in cameras}


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping src to dst."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = dst_centered.T @ src_centered / len(src)
    u, svals, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rot = u @ correction @ vt
    var_src = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.sum(svals * np.diag(correction)) / var_src)
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / s,
                    0.25 * s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    0.25 * s,
                    (rot[1, 2] + rot[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [
                    (rot[1, 0] - rot[0, 1]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                    (rot[1, 2] + rot[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    return quat / max(np.linalg.norm(quat), 1e-12)


def transform_ply(src: Path, dst: Path, scale: float, rot: np.ndarray, trans: np.ndarray) -> None:
    ply = PlyData.read(str(src))
    vertices = np.array(ply["vertex"].data, copy=True)
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    xyz_new = scale * (xyz @ rot.T) + trans
    vertices["x"] = xyz_new[:, 0]
    vertices["y"] = xyz_new[:, 1]
    vertices["z"] = xyz_new[:, 2]

    if all(f"scale_{i}" in vertices.dtype.names for i in range(3)):
        log_scale = np.log(max(scale, 1e-12))
        for i in range(3):
            vertices[f"scale_{i}"] = vertices[f"scale_{i}"] + log_scale

    if all(f"rot_{i}" in vertices.dtype.names for i in range(4)):
        for i in range(len(vertices)):
            q_old = np.array([vertices[f"rot_{j}"][i] for j in range(4)], dtype=np.float64)
            r_new = rot @ quat_to_rot(q_old)
            q_new = rot_to_quat(r_new)
            for j in range(4):
                vertices[f"rot_{j}"][i] = q_new[j]

    elements = [
        PlyElement.describe(vertices, "vertex") if elem.name == "vertex" else elem
        for elem in ply.elements
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(dst))


def main() -> None:
    original = load_camera_centers(ORIG_CAMERAS)
    corrected = load_camera_centers(CORR_CAMERAS)
    names = sorted(set(original) & set(corrected))
    src = np.stack([corrected[name] for name in names])
    dst = np.stack([original[name] for name in names])
    scale, rot, trans = umeyama_similarity(src, dst)

    aligned = scale * (src @ rot.T) + trans
    err = np.linalg.norm(aligned - dst, axis=1)
    report = {
        "source_cameras": str(CORR_CAMERAS),
        "target_cameras": str(ORIG_CAMERAS),
        "num_common_cameras": len(names),
        "scale": scale,
        "rotation": rot.tolist(),
        "translation": trans.tolist(),
        "camera_alignment_rmse": float(np.sqrt(np.mean(err**2))),
        "camera_alignment_median_error": float(np.median(err)),
        "camera_alignment_max_error": float(np.max(err)),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "alignment_report.json").write_text(json.dumps(report, indent=2))

    files = {
        "corrected_clean_object_aligned_to_original_scene5.ply": CORR_OBJ_ROOT
        / "final_ply/per_id_reprojection_ellipsoid20_strict75"
        / "realm_id001_strainer_correctedmask_ellipsoid20_strict75_truecolor_3dgs.ply",
        "corrected_clean_prediction_thr058_aligned_to_original_scene5.ply": CORR_PRED_ROOT
        / "prediction_plys/pointnet_mean_seen/threshold_green_blue"
        / "realm_id001_pointnet_mean_seen_thr0.58.ply",
        "corrected_clean_prediction_probability_aligned_to_original_scene5.ply": CORR_PRED_ROOT
        / "prediction_plys/pointnet_mean_seen/probability_heatmap_blue0_red1"
        / "realm_id001_pointnet_mean_seen_prob.ply",
    }
    for out_name, in_path in files.items():
        transform_ply(in_path, OUT_ROOT / out_name, scale, rot, trans)
        print(OUT_ROOT / out_name)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
