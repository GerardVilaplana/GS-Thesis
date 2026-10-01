from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
RUN_ROOT = (
    BASE
    / "outputs/03_handle_generalization/47_exp46_dino_pointnet_mean_feature_sweep_v1/object_crop_center_avg"
)
SEEN_DIR = RUN_ROOT / "01_seen_instance_17cat/dino_geometry_color_quality/pointnet_mean"
LOCO_ROOT = RUN_ROOT / "02_leave_one_category_out_17cat"
OUT_DIR = BASE / "outputs/thesis_figures/best_supervisedOE_model_plys/comparison"
SOURCE_PLY_DIR = BASE / "data/handal_exp40_15k_gt_features_v1/ply/B_center_ellipsoid20_strict75"

C0 = 0.28209479177387814
CATEGORIES = [
    "measuring_cups",
    "adjustable_wrenches",
    "hammers",
    "pots_pans",
    "mugs",
    "power_drills",
]
CANDIDATE_COUNTS = {
    "adjustable_wrenches": 4,
    "pots_pans": 4,
}
DROP_CATEGORIES = {"mugs", "power_drills"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_threshold(model_dir: Path) -> float:
    with (model_dir / "overall_metrics.json").open() as f:
        return float(json.load(f)["selected_threshold"])


def load_prediction_slices(model_dir: Path) -> dict[str, tuple[int, int, np.ndarray, np.ndarray]]:
    z = np.load(model_dir / "test_predictions.npz", allow_pickle=False)
    scene_keys = z["scene_keys"]
    offsets = z["offsets"]
    scores = z["scores"]
    labels = z["labels"]
    return {
        str(scene_key): (int(start), int(end), scores[int(start) : int(end)], labels[int(start) : int(end)])
        for scene_key, (start, end) in zip(scene_keys, offsets)
    }


def rgb_to_sh(rgb_01: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb_01, dtype=np.float32) - 0.5) / C0


def qa_colors(labels: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    labels = labels.astype(bool)
    pred = scores >= threshold
    rgb = np.zeros((len(labels), 3), dtype=np.float32)
    rgb[np.logical_and(~labels, ~pred)] = np.array([0.02, 0.16, 1.0], dtype=np.float32)
    rgb[np.logical_and(labels, pred)] = np.array([0.02, 0.85, 0.12], dtype=np.float32)
    rgb[np.logical_and(labels, ~pred)] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    rgb[np.logical_and(~labels, pred)] = np.array([1.0, 0.48, 0.02], dtype=np.float32)
    return rgb


def write_colored_ply(source_ply: Path, out_path: Path, rgb_01: np.ndarray) -> None:
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(rgb_01):
        raise ValueError(f"PLY/color length mismatch for {source_ply}: {len(vertices)} vs {len(rgb_01)}")
    sh = rgb_to_sh(rgb_01)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    elements = [
        PlyElement.describe(vertices, "vertex") if element.name == "vertex" else element
        for element in ply.elements
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(out_path))


def safe_name(scene_key: str) -> str:
    return scene_key.replace("/", "_")


def source_ply_for(scene_key: str, row: dict[str, str]) -> Path:
    from_row = row.get("source_object_ply", "")
    if from_row:
        return Path(from_row)
    return SOURCE_PLY_DIR / f"{scene_key}_object_center_ell20s75.ply"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_rows = {row["scene_key"]: row for row in read_csv(SEEN_DIR / "per_scene_metrics.csv")}
    seen_slices = load_prediction_slices(SEEN_DIR)
    seen_threshold = load_threshold(SEEN_DIR)

    manifest = []
    for category in CATEGORIES:
        loco_dir = LOCO_ROOT / category / "dino_geometry_color_quality/pointnet_mean"
        loco_rows = {row["scene_key"]: row for row in read_csv(loco_dir / "per_scene_metrics.csv")}
        loco_slices = load_prediction_slices(loco_dir)
        loco_threshold = load_threshold(loco_dir)

        common = []
        for scene_key, seen_row in seen_rows.items():
            if seen_row["category"] != category or scene_key not in loco_rows:
                continue
            common.append(
                {
                    "scene_key": scene_key,
                    "category": category,
                    "seen_iou": float(seen_row["iou"]),
                    "loco_iou": float(loco_rows[scene_key]["iou"]),
                    "seen_f1": float(seen_row["f1"]),
                    "loco_f1": float(loco_rows[scene_key]["f1"]),
                    "source_object_ply": str(source_ply_for(scene_key, seen_row)),
                }
            )
        if len(common) < 2:
            raise RuntimeError(f"Need at least two common seen/LOCO scenes for {category}, found {len(common)}")

        if category in DROP_CATEGORIES:
            selected = sorted(common, key=lambda r: (r["seen_iou"] - r["loco_iou"], r["seen_iou"]), reverse=True)[: CANDIDATE_COUNTS.get(category, 2)]
            strategy = "largest_seen_to_loco_drop"
        else:
            selected = sorted(common, key=lambda r: (r["loco_iou"], r["seen_iou"]), reverse=True)[: CANDIDATE_COUNTS.get(category, 2)]
            strategy = "highest_loco_iou"

        for rank, row in enumerate(selected, start=1):
            scene_key = row["scene_key"]
            source_ply = Path(row["source_object_ply"])
            for setting, model_dir, threshold, slices in [
                ("seen", SEEN_DIR, seen_threshold, seen_slices),
                ("loco", loco_dir, loco_threshold, loco_slices),
            ]:
                _, _, scores, labels = slices[scene_key]
                iou = row["seen_iou"] if setting == "seen" else row["loco_iou"]
                out_name = (
                    f"{category}__candidate{rank:02d}__{safe_name(scene_key)}"
                    f"__seen{row['seen_iou']:.3f}__loco{row['loco_iou']:.3f}__{setting}.ply"
                )
                out_path = OUT_DIR / out_name
                write_colored_ply(source_ply, out_path, qa_colors(labels, scores, threshold))
                manifest.append(
                    {
                        "category": category,
                        "candidate": rank,
                        "scene_key": scene_key,
                        "setting": setting,
                        "selection_strategy": strategy,
                        "seen_iou": f"{row['seen_iou']:.6f}",
                        "loco_iou": f"{row['loco_iou']:.6f}",
                        "seen_f1": f"{row['seen_f1']:.6f}",
                        "loco_f1": f"{row['loco_f1']:.6f}",
                        "threshold": f"{threshold:.6f}",
                        "model_dir": str(model_dir),
                        "source_object_ply": str(source_ply),
                        "output_ply": str(out_path),
                    }
                )

    write_csv(OUT_DIR / "comparison_manifest.csv", manifest)
    print(f"Wrote {len(manifest)} PLYs to {OUT_DIR}")
    print(f"Manifest: {OUT_DIR / 'comparison_manifest.csv'}")


if __name__ == "__main__":
    main()
