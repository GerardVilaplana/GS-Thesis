#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from affordsplat_paper_prompts import make_paper_grasp_prompt  # noqa: E402
from run_3daffordsplat_handal_eval import C0, load_model, load_question_table, predict_scene  # noqa: E402


DEFAULT_SCENE_ROOTS = {
    "scene1": BASE_ROOT
    / "outputs/06_own_scenes/scene_1_15k_realm_query_pruning/final_ply/per_id_reprojection_ellipsoid20_strict75",
    "scene2": BASE_ROOT
    / "outputs/06_own_scenes/scene_2_15k_realm_query_pruning/final_ply/per_id_reprojection_ellipsoid20_strict75",
}
DEFAULT_OUT = BASE_ROOT / "outputs/04_3daffordsplat_baseline/own_scenes_scene1_scene2_grasp_v1"


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) - 0.5) / C0


def write_score_ply(source_ply: Path, out_ply: Path, scores: np.ndarray) -> None:
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    scores = np.asarray(scores, dtype=np.float32)
    if len(vertices) != len(scores):
        raise ValueError(f"PLY/score length mismatch: {source_ply} {len(vertices)} vs {len(scores)}")
    clipped = np.clip(scores, 0.0, 1.0)
    rgb = np.column_stack(
        [
            np.maximum(clipped, 0.05),
            np.full_like(clipped, 0.04),
            1.0 - clipped,
        ]
    ).astype(np.float32)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(out_ply))


def write_threshold_ply(source_ply: Path, out_ply: Path, scores: np.ndarray, threshold: float) -> None:
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    scores = np.asarray(scores, dtype=np.float32)
    if len(vertices) != len(scores):
        raise ValueError(f"PLY/score length mismatch: {source_ply} {len(vertices)} vs {len(scores)}")
    pred = scores >= threshold
    rgb = np.zeros((len(scores), 3), dtype=np.float32)
    rgb[~pred] = np.array([0.05, 0.18, 0.95], dtype=np.float32)
    rgb[pred] = np.array([0.0, 0.95, 0.2], dtype=np.float32)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(out_ply))


def object_name_from_ply(path: Path) -> tuple[str, str]:
    match = re.match(r"realm_id(\d+)_(.*)_ellipsoid20_strict75_truecolor_3dgs$", path.stem)
    if not match:
        return "unknown", path.stem.replace("_", " ")
    realm_id, slug = match.groups()
    return f"realm_id{realm_id}", slug.replace("_", " ")


def make_grasp_prompt(object_name: str) -> tuple[str, str]:
    return f"If you want to grasp the {object_name}, which points should your hand or fingers touch?", "<Aff>"


def collect_plys(scene_roots: dict[str, Path]) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for scene, root in scene_roots.items():
        if not root.exists():
            raise FileNotFoundError(root)
        for ply in sorted(root.glob("*.ply")):
            items.append((scene, ply))
    if not items:
        raise FileNotFoundError(f"No PLY files found in {scene_roots}")
    return items


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene1_root", type=Path, default=DEFAULT_SCENE_ROOTS["scene1"])
    parser.add_argument("--scene2_root", type=Path, default=DEFAULT_SCENE_ROOTS["scene2"])
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--paper_style_prompts", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_model(device)
    question_table = load_question_table()
    rows = []
    score_dir = args.out_dir / "scores"
    plys = collect_plys({"scene1": args.scene1_root, "scene2": args.scene2_root})
    if args.limit:
        plys = plys[: args.limit]

    for scene, ply_path in plys:
        realm_id, object_name = object_name_from_ply(ply_path)
        prompt_source = "minimal_aff_token_answer"
        prompt_object = object_name
        if args.paper_style_prompts:
            question, answer, prompt_source, prompt_object = make_paper_grasp_prompt(object_name, question_table)
        else:
            question, answer = make_grasp_prompt(object_name)
        scores, predicted_text = predict_scene(model, ply_path, question, answer, device)
        key = f"{scene}_{ply_path.stem}"
        score_path = score_dir / f"{key}_scores.npz"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            scene=np.array([scene]),
            realm_id=np.array([realm_id]),
            object_name=np.array([object_name]),
            source_ply=np.array([str(ply_path)]),
            question=np.array([question]),
            answer_prompt=np.array([answer]),
            prompt_source=np.array([prompt_source]),
            prompt_object=np.array([prompt_object]),
            predicted_text=np.array([str(predicted_text)]),
            scores=scores.astype(np.float32),
        )

        heatmap_ply = args.out_dir / "prediction_ply_raw_score" / f"{key}_affordsplatnet_grasp_score.ply"
        threshold_ply = (
            args.out_dir
            / "prediction_ply_threshold_green_blue"
            / f"{key}_affordsplatnet_grasp_thr{args.threshold:.2f}.ply"
        )
        write_score_ply(ply_path, heatmap_ply, scores)
        write_threshold_ply(ply_path, threshold_ply, scores, args.threshold)
        row = {
            "scene": scene,
            "realm_id": realm_id,
            "object_name": object_name,
            "num_gaussians": int(len(scores)),
            "score_min": float(np.min(scores)) if len(scores) else 0.0,
            "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "score_max": float(np.max(scores)) if len(scores) else 0.0,
            "pred_grasp_ratio": float(np.mean(scores >= args.threshold)) if len(scores) else 0.0,
            "threshold": float(args.threshold),
            "question": question,
            "answer_prompt": answer,
            "prompt_source": prompt_source,
            "prompt_object": prompt_object,
            "predicted_text": str(predicted_text),
            "source_ply": str(ply_path),
            "score_npz": str(score_path),
            "score_ply": str(heatmap_ply),
            "threshold_ply": str(threshold_ply),
        }
        rows.append(row)
        print(
            f"{scene}/{realm_id} {object_name}: n={row['num_gaussians']} "
            f"mean={row['score_mean']:.3f} max={row['score_max']:.3f} "
            f"pred_ratio={row['pred_grasp_ratio']:.3f}",
            flush=True,
        )

    save_csv(args.out_dir / "prediction_manifest.csv", rows)
    print(f"Wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
