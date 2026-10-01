#!/usr/bin/env python3
"""Export eight supervised HANDAL objects as one labeled 2x4 Gaussian grid."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
NPZ_ROOT = ROOT / "data/handal_exp40_15k_gt_features_v1/npz/B_center_ellipsoid20_strict75"
PLY_ROOT = ROOT / "data/handal_exp40_15k_gt_features_v1/ply/B_center_ellipsoid20_strict75"
OUTPUT_DIR = ROOT / "outputs/thesis_figures/handle_lifted_8_categories_grid"

SCENES = [
    ("adjustable_wrenches__033007", "Adjustable wrench"),
    ("hammers__003009", "Hammer"),
    ("measuring_cups__014010", "Measuring cup"),
    ("pots_pans__044004", "Pot/Pan"),
    ("mugs__023008", "Mug"),
    ("power_drills__006006", "Power drill"),
    ("slip_joint_pliers__006005", "Slip-joint pliers"),
    ("strainers__032006", "Strainer"),
]

BLUE = np.array([3.0, 40.0, 255.0]) / 255.0
GREEN = np.array([5.0, 215.0, 35.0]) / 255.0
OVERLAY_ALPHA = 0.70
SH_C0 = 0.28209479177387814

COL_CENTERS = np.array([-3.75, -1.25, 1.25, 3.75])
ROW_CENTERS = np.array([1.35, -1.35])
TARGET_WIDTH = 1.90
TARGET_HEIGHT = 1.70


def robust_range(values: np.ndarray) -> float:
    lo, hi = np.quantile(values, [0.005, 0.995])
    return max(float(hi - lo), 1e-8)


def choose_top_view(xyz: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Choose a right-handed basis that exposes the handle in the XY top view."""
    center = np.median(xyz, axis=0)
    centered = xyz - center
    radius = np.linalg.norm(centered, axis=1)
    core = centered[radius <= np.quantile(radius, 0.995)]
    covariance = np.cov(core, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    pca = eigenvectors[:, ::-1]

    handle = centered[labels]
    object_ranges = [robust_range(centered @ pca[:, axis]) for axis in range(3)]
    handle_ranges = [robust_range(handle @ pca[:, axis]) for axis in range(3)]

    candidates = []
    for view_axis in range(3):
        plane = [axis for axis in range(3) if axis != view_axis]
        handle_area = handle_ranges[plane[0]] * handle_ranges[plane[1]]
        object_area = object_ranges[plane[0]] * object_ranges[plane[1]]
        candidates.append((handle_area + 0.05 * object_area, plane))
    _, plane = max(candidates, key=lambda item: item[0])

    if object_ranges[plane[1]] > object_ranges[plane[0]]:
        plane = [plane[1], plane[0]]
    ex = pca[:, plane[0]].copy()
    ey = pca[:, plane[1]].copy()
    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)

    basis = np.column_stack([ex, ey, ez])
    local = centered @ basis

    # Keep the handle predominantly on the right for an easier top-view reading.
    if np.median(local[labels, 0]) < np.median(local[~labels, 0]):
        basis[:, 0] *= -1.0
        basis[:, 2] *= -1.0
        local = centered @ basis

    # Look from +Z with the handle-facing side closest to the viewer when possible.
    if np.median(local[labels, 2]) < np.median(local[~labels, 2]):
        basis[:, 1] *= -1.0
        basis[:, 2] *= -1.0

    assert np.linalg.det(basis) > 0.999
    return basis


def transform_rotations(vertices: np.ndarray, global_rotation: np.ndarray) -> None:
    q_wxyz = np.column_stack([vertices[f"rot_{i}"] for i in range(4)]).astype(np.float64)
    q_norm = np.linalg.norm(q_wxyz, axis=1, keepdims=True)
    q_wxyz /= np.maximum(q_norm, 1e-12)
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    old_rotations = Rotation.from_quat(q_xyzw).as_matrix()
    new_rotations = global_rotation[None, :, :] @ old_rotations
    new_xyzw = Rotation.from_matrix(new_rotations).as_quat()
    new_wxyz = new_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    for i in range(4):
        vertices[f"rot_{i}"] = new_wxyz[:, i]


def apply_label_overlay(vertices: np.ndarray, labels: np.ndarray) -> None:
    sh = np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]).astype(np.float64)
    rgb = np.clip(sh * SH_C0 + 0.5, 0.0, 1.0)
    overlay = np.where(labels[:, None], GREEN, BLUE)
    rgb = (1.0 - OVERLAY_ALPHA) * rgb + OVERLAY_ALPHA * overlay
    sh = ((rgb - 0.5) / SH_C0).astype(np.float32)
    for i in range(3):
        vertices[f"f_dc_{i}"] = sh[:, i]


def load_and_place(scene_key: str, row: int, col: int):
    npz_path = NPZ_ROOT / f"{scene_key}.npz"
    ply_path = PLY_ROOT / f"{scene_key}_object_center_ell20s75.ply"
    data = np.load(npz_path)
    labels = data["handle_labels_thr0_25"].astype(bool)
    source = PlyData.read(ply_path)
    vertices = source["vertex"].data.copy()
    if len(vertices) != len(labels):
        raise ValueError(f"PLY/label mismatch for {scene_key}: {len(vertices)} vs {len(labels)}")

    xyz = np.column_stack([vertices[axis] for axis in "xyz"]).astype(np.float64)
    center = np.median(xyz, axis=0)
    basis = choose_top_view(xyz, labels)
    local = (xyz - center) @ basis

    # Include a conservative 3-sigma Gaussian radius when fitting each cell.
    max_radius = 3.0 * float(np.max(np.exp(np.column_stack([
        vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]
    ]))))
    width = float(np.ptp(local[:, 0]) + 2.0 * max_radius)
    height = float(np.ptp(local[:, 1]) + 2.0 * max_radius)
    scale = min(TARGET_WIDTH / width, TARGET_HEIGHT / height)

    local *= scale
    local[:, 0] += COL_CENTERS[col]
    local[:, 1] += ROW_CENTERS[row]
    local[:, 2] -= np.median(local[:, 2])
    for index, axis in enumerate("xyz"):
        vertices[axis] = local[:, index].astype(np.float32)

    global_rotation = basis.T
    transform_rotations(vertices, global_rotation)
    log_scale = math.log(scale)
    for field in ("scale_0", "scale_1", "scale_2"):
        vertices[field] = (vertices[field].astype(np.float64) + log_scale).astype(np.float32)
    apply_label_overlay(vertices, labels)

    radius_after = max_radius * scale
    bounds = {
        "xmin": float(local[:, 0].min() - radius_after),
        "xmax": float(local[:, 0].max() + radius_after),
        "ymin": float(local[:, 1].min() - radius_after),
        "ymax": float(local[:, 1].max() + radius_after),
    }
    record = {
        "scene_key": scene_key,
        "row": row + 1,
        "column": col + 1,
        "num_gaussians": len(vertices),
        "num_handle_gaussians": int(labels.sum()),
        "handle_ratio": float(labels.mean()),
        "uniform_scale": scale,
        "source_ply": str(ply_path),
        "source_npz": str(npz_path),
        **bounds,
    }
    return vertices, labels, record, source.text, source.byte_order


def verify_no_overlap(records: list[dict]) -> None:
    for i, first in enumerate(records):
        for second in records[i + 1:]:
            overlap_x = first["xmin"] < second["xmax"] and second["xmin"] < first["xmax"]
            overlap_y = first["ymin"] < second["ymax"] and second["ymin"] < first["ymax"]
            if overlap_x and overlap_y:
                raise RuntimeError(f"Grid overlap: {first['scene_key']} and {second['scene_key']}")


def write_preview(chunks: list[np.ndarray], labels_list: list[np.ndarray], records: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(12, 6.2), dpi=220)
    for vertices, labels, record in zip(chunks, labels_list, records):
        x = vertices["x"]
        y = vertices["y"]
        stride = max(1, len(vertices) // 18000)
        idx = np.arange(0, len(vertices), stride)
        colors = np.where(labels[idx, None], GREEN, BLUE)
        axis.scatter(x[idx], y[idx], c=colors, s=0.5, linewidths=0, alpha=0.65)
        axis.text(
            COL_CENTERS[record["column"] - 1],
            ROW_CENTERS[record["row"] - 1] - 1.02,
            record["display_name"],
            ha="center",
            va="top",
            fontsize=9,
        )
    axis.set_aspect("equal")
    axis.set_xlim(-5.0, 5.0)
    axis.set_ylim(-2.55, 2.45)
    axis.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUTPUT_DIR / "handle_lifted_8_categories_2x4_topview_preview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []
    labels_list = []
    records = []
    text = True
    byte_order = "="

    for index, (scene_key, display_name) in enumerate(SCENES):
        row, col = divmod(index, 4)
        vertices, labels, record, text, byte_order = load_and_place(scene_key, row, col)
        record["display_name"] = display_name
        chunks.append(vertices)
        labels_list.append(labels)
        records.append(record)

    verify_no_overlap(records)
    combined = np.concatenate(chunks)
    comments = [
        "Eight ellipsoid-refined supervised HANDAL objects arranged in a 2x4 grid.",
        "Green: lifted handle label; blue: lifted non-handle label; overlay alpha: 0.70.",
        "Top view is along the positive Z axis toward the XY plane.",
    ]
    output_path = OUTPUT_DIR / "handle_lifted_8_categories_2x4_topview.ply"
    result = PlyData(
        [PlyElement.describe(combined, "vertex")],
        text=text,
        byte_order=byte_order,
        comments=comments,
    )
    result.write(output_path)

    fieldnames = list(records[0].keys())
    with (OUTPUT_DIR / "handle_lifted_8_categories_2x4_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with (OUTPUT_DIR / "handle_lifted_8_categories_2x4_manifest.json").open("w") as handle:
        json.dump(records, handle, indent=2)

    write_preview(chunks, labels_list, records)
    print(f"Wrote {output_path}")
    print(f"Gaussians: {len(combined):,}")
    for record in records:
        print(
            f"r{record['row']}c{record['column']} {record['display_name']}: "
            f"{record['scene_key']} ({record['num_gaussians']:,} Gaussians)"
        )


if __name__ == "__main__":
    main()
