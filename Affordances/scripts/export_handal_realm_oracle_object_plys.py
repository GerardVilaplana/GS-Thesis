#!/usr/bin/env python3
"""Export HANDAL objects pruned by REALM IDs selected from HANDAL object-mask evidence.

This is a diagnostic: HANDAL object masks are used only to choose which REALM
Gaussian object IDs correspond to the object. The exported object itself is the
set of full-scene Gaussians predicted with those REALM IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814


OLD_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features")
NEW_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_new_categories_delta_v1_features")


def rgb_to_sh(rgb: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return (arr - 0.5) / C0


def id_color(idx: int) -> tuple[int, int, int]:
    if idx == 0:
        return (0, 0, 0)
    palette = [
        (230, 40, 40),
        (40, 120, 245),
        (35, 200, 100),
        (250, 200, 40),
        (200, 70, 240),
        (245, 120, 35),
        (40, 210, 220),
        (180, 180, 255),
    ]
    return palette[idx % len(palette)]


def scene_key(category: str, scene_id: str) -> str:
    return f"{category}__{scene_id}"


def root_for_scene(key: str) -> Path:
    if (OLD_ROOT / "npz" / f"{key}.npz").exists():
        return OLD_ROOT
    if (NEW_ROOT / "npz" / f"{key}.npz").exists():
        return NEW_ROOT
    raise FileNotFoundError(f"No feature npz found for {key}")


def load_manifest_scenes(manifest_path: Path, limit_per_category: int) -> list[dict[str, str]]:
    data = json.loads(manifest_path.read_text())
    rows = data["selected_scenes"]
    counts: dict[str, int] = {}
    out = []
    for row in rows:
        cat = row["category"]
        if counts.get(cat, 0) >= limit_per_category:
            continue
        out.append({"category": cat, "scene_id": row["scene_id"], "scene_key": row["scene_key"]})
        counts[cat] = counts.get(cat, 0) + 1
    return out


def predict_realm_ids(vertex: np.ndarray, classifier_path: Path, num_classes: int) -> np.ndarray:
    obj = np.stack([vertex[f"obj_dc_{i}"] for i in range(16)], axis=0).astype(np.float32)
    state = torch.load(classifier_path, map_location="cpu")
    inferred_classes = int(state["weight"].shape[0])
    if num_classes <= 0:
        num_classes = inferred_classes
    elif num_classes != inferred_classes:
        print(
            f"warning: requested num_classes={num_classes}, but classifier has {inferred_classes}; using {inferred_classes}",
            file=sys.stderr,
            flush=True,
        )
        num_classes = inferred_classes
    classifier = torch.nn.Conv2d(16, num_classes, kernel_size=1)
    classifier.load_state_dict(state)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.from_numpy(obj)[None, :, :, None])
    return logits.argmax(dim=1).squeeze(0).squeeze(-1).cpu().numpy().astype(np.int32)


def select_ids_from_object_indices(
    pred_ids: np.ndarray,
    object_indices: np.ndarray,
    coverage_target: float,
    min_id_fraction: float,
    max_ids: int,
) -> tuple[list[int], list[dict[str, float]]]:
    object_ids = pred_ids[object_indices]
    ids, counts = np.unique(object_ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = int(len(object_indices))
    selected: list[int] = []
    table = []
    covered = 0
    for pos in order:
        class_id = int(ids[pos])
        count = int(counts[pos])
        frac = count / max(total, 1)
        table.append({"class_id": class_id, "object_overlap_count": count, "object_overlap_fraction": frac})
        if class_id == 0 and len(ids) > 1:
            continue
        if frac >= min_id_fraction or covered / max(total, 1) < coverage_target:
            selected.append(class_id)
            covered += count
        if len(selected) >= max_ids or covered / max(total, 1) >= coverage_target:
            break
    if not selected and len(table):
        selected = [int(table[0]["class_id"])]
    return selected, table


def write_subset_ply(ply_data: PlyData, mask: np.ndarray, path: Path) -> int:
    vertex = ply_data["vertex"].data.copy()[mask]
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply_data.text).write(path)
    return int(len(vertex))


def write_idcolor_subset_ply(ply_data: PlyData, pred_ids: np.ndarray, mask: np.ndarray, path: Path) -> int:
    vertex = ply_data["vertex"].data.copy()[mask]
    ids = pred_ids[mask]
    for class_id in np.unique(ids):
        color_sh = rgb_to_sh(id_color(int(class_id)))
        class_mask = ids == class_id
        vertex["f_dc_0"][class_mask] = color_sh[0]
        vertex["f_dc_1"][class_mask] = color_sh[1]
        vertex["f_dc_2"][class_mask] = color_sh[2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply_data.text).write(path)
    return int(len(vertex))


def export_scene(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    key = row["scene_key"]
    root = root_for_scene(key)
    npz_path = root / "npz" / f"{key}.npz"
    model_root = root / "work" / "3dgs_models" / key
    point_cloud = model_root / "point_cloud" / "iteration_3000" / "point_cloud.ply"
    classifier = model_root / "point_cloud" / "iteration_3000" / "classifier.pth"
    if not point_cloud.exists() or not classifier.exists():
        raise FileNotFoundError(f"Missing REALM point cloud/classifier for {key}")

    data = np.load(npz_path, allow_pickle=True)
    object_indices = data["gaussian_indices"].astype(np.int64)
    ply_data = PlyData.read(point_cloud)
    pred_ids = predict_realm_ids(ply_data["vertex"].data, classifier, args.num_classes)

    selected_ids, id_table = select_ids_from_object_indices(
        pred_ids,
        object_indices,
        args.coverage_target,
        args.min_id_fraction,
        args.max_ids,
    )
    selected_mask = np.isin(pred_ids, selected_ids)
    reference_mask = np.zeros(len(pred_ids), dtype=bool)
    reference_mask[object_indices] = True

    out_dir = args.output_dir / row["category"]
    out_dir.mkdir(parents=True, exist_ok=True)
    true_path = out_dir / f"{key}_realm_oracle_ids_truecolor.ply"
    id_path = out_dir / f"{key}_realm_oracle_ids_idcolor.ply"
    ref_path = out_dir / f"{key}_mask_uplift_reference_truecolor.ply"

    selected_count = write_subset_ply(ply_data, selected_mask, true_path)
    write_idcolor_subset_ply(ply_data, pred_ids, selected_mask, id_path)
    reference_count = write_subset_ply(ply_data, reference_mask, ref_path)

    return {
        "scene_key": key,
        "category": row["category"],
        "scene_id": row["scene_id"],
        "feature_root": str(root),
        "point_cloud": str(point_cloud),
        "classifier": str(classifier),
        "npz": str(npz_path),
        "selected_realm_ids": selected_ids,
        "id_overlap_table": id_table[: args.max_report_ids],
        "full_gaussians": int(len(pred_ids)),
        "mask_reference_gaussians": int(reference_count),
        "realm_pruned_gaussians": int(selected_count),
        "realm_to_mask_size_ratio": float(selected_count / max(reference_count, 1)),
        "outputs": {
            "realm_truecolor": str(true_path),
            "realm_idcolor": str(id_path),
            "mask_uplift_reference_truecolor": str(ref_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(
        "/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/"
        "22_seen_instance_pointnet_gnn_prediction_plys_v1/selection_manifest.json"
    ))
    parser.add_argument("--limit_per_category", type=int, default=2)
    parser.add_argument("--coverage_target", type=float, default=0.85)
    parser.add_argument("--min_id_fraction", type=float, default=0.05)
    parser.add_argument("--max_ids", type=int, default=6)
    parser.add_argument("--max_report_ids", type=int, default=12)
    parser.add_argument("--num_classes", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = load_manifest_scenes(args.manifest, args.limit_per_category)
    results = []
    for row in scenes:
        print(f"exporting {row['scene_key']}", flush=True)
        results.append(export_scene(row, args))

    report = {
        "note": (
            "Diagnostic REALM object pruning. HANDAL object-mask lifted Gaussian indices are used only "
            "to select the REALM IDs that correspond to the object; exported truecolor PLYs are the full-scene "
            "Gaussians whose REALM predicted ID is selected."
        ),
        "settings": dict(vars(args), output_dir=str(args.output_dir), manifest=str(args.manifest)),
        "results": results,
    }
    (args.output_dir / "realm_oracle_object_pruning_report.json").write_text(json.dumps(report, indent=2))

    with (args.output_dir / "realm_oracle_object_pruning_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_key",
                "category",
                "scene_id",
                "selected_realm_ids",
                "full_gaussians",
                "mask_reference_gaussians",
                "realm_pruned_gaussians",
                "realm_to_mask_size_ratio",
                "realm_truecolor",
                "realm_idcolor",
                "mask_uplift_reference_truecolor",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({
                "scene_key": row["scene_key"],
                "category": row["category"],
                "scene_id": row["scene_id"],
                "selected_realm_ids": " ".join(map(str, row["selected_realm_ids"])),
                "full_gaussians": row["full_gaussians"],
                "mask_reference_gaussians": row["mask_reference_gaussians"],
                "realm_pruned_gaussians": row["realm_pruned_gaussians"],
                "realm_to_mask_size_ratio": row["realm_to_mask_size_ratio"],
                "realm_truecolor": row["outputs"]["realm_truecolor"],
                "realm_idcolor": row["outputs"]["realm_idcolor"],
                "mask_uplift_reference_truecolor": row["outputs"]["mask_uplift_reference_truecolor"],
            })
    print(args.output_dir / "realm_oracle_object_pruning_report.json")


if __name__ == "__main__":
    main()
