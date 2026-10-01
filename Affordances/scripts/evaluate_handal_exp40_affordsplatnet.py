#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import train_handal_17cat_graph_attention as g15  # noqa: E402
from affordsplat_paper_prompts import make_paper_grasp_prompt  # noqa: E402
from run_3daffordsplat_handal_eval import CHECKPOINT, load_model, load_question_table, make_prompt, predict_scene  # noqa: E402


DATA_ROOT = BASE_ROOT / "data/handal_exp40_15k_gt_features_v1"
NPZ_ROOT = DATA_ROOT / "npz/B_center_ellipsoid20_strict75"
OUT_ROOT = BASE_ROOT / "outputs/04_3daffordsplat_baseline/handal_exp40_b_gt_pruned_grasp_eval_v1"

BAD_SCENES = {
    "spatulas__032005",
    "slip_joint_pliers__090006",
    "slip_joint_pliers__092005",
    "locking_pliers__040003",
    "locking_pliers__091004",
    "utensils__022007",
    "slip_joint_pliers__093004",
    "locking_pliers__090004",
    "utensils__026007",
    "locking_pliers__041005",
}


def scalar_str(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    value = z[key]
    return str(value[0] if getattr(value, "shape", ()) else value)


def load_items(npz_root: Path, include_bad_scenes: bool = False) -> list[dict]:
    items = []
    for path in sorted(npz_root.glob("*.npz")):
        category, scene_id = path.stem.rsplit("__", 1)
        with np.load(path, allow_pickle=False) as z:
            y = z["handle_labels_thr0_25"].astype(np.uint8)
            items.append(
                {
                    "scene_key": path.stem,
                    "category": scalar_str(z, "category", category),
                    "scene_id": scalar_str(z, "scene_id", scene_id),
                    "instance_id": scalar_str(z, "instance_id", scene_id),
                    "source_split": scalar_str(z, "model_split", "unknown"),
                    "raw_split": scalar_str(z, "raw_split", "unknown"),
                    "source_object_ply": scalar_str(z, "source_object_ply", ""),
                    "npz": str(path),
                    "num_gaussians": int(len(y)),
                    "num_handle": int(y.sum()),
                }
            )
    if not items:
        raise FileNotFoundError(f"No NPZ files found in {npz_root}")
    if include_bad_scenes:
        return items
    before = len(items)
    items = [item for item in items if item["scene_key"] not in BAD_SCENES]
    print(f"[dataset] excluded {before - len(items)} bad scenes; remaining={len(items)}", flush=True)
    return items


def load_labels(item: dict) -> np.ndarray:
    with np.load(item["npz"], allow_pickle=False) as z:
        return z["handle_labels_thr0_25"].astype(np.uint8)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_seen(items: list[dict]) -> dict[str, list[dict]]:
    out = {"train": [], "val": [], "test": []}
    for item in items:
        if item["source_split"] in out:
            out[item["source_split"]].append(item)
    return {k: sorted(v, key=lambda x: x["scene_key"]) for k, v in out.items()}


def split_loco(items: list[dict], heldout_category: str) -> dict[str, list[dict]]:
    return {
        "val": sorted(
            [x for x in items if x["category"] != heldout_category and x["source_split"] == "val"],
            key=lambda x: x["scene_key"],
        ),
        "test": sorted([x for x in items if x["category"] == heldout_category], key=lambda x: x["scene_key"]),
    }


def metric_slices(items: list[dict], labels: list[np.ndarray], scores: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list]:
    y_all = []
    s_all = []
    slices = []
    start = 0
    for item, y, s in zip(items, labels, scores):
        if len(y) != len(s):
            raise ValueError(f"{item['scene_key']}: labels={len(y)} scores={len(s)}")
        y_all.append(y.astype(np.uint8))
        s_all.append(s.astype(np.float32))
        end = start + len(y)
        slices.append((item, start, end))
        start = end
    if not y_all:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float32), []
    return np.concatenate(y_all), np.concatenate(s_all), slices


def score_path(score_dir: Path, scene_key: str, affordance: str) -> Path:
    return score_dir / f"{scene_key}_{affordance}_scores.npz"


def compute_or_load_scores(
    item: dict,
    affordance: str,
    model,
    question_table,
    device: torch.device,
    score_dir: Path,
    prompt_mode: str,
    paper_style_prompts: bool = False,
) -> np.ndarray:
    path = score_path(score_dir, item["scene_key"], affordance)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z["scores"].astype(np.float32)
    ply_path = Path(item["source_object_ply"])
    if not ply_path.exists():
        raise FileNotFoundError(f"{item['scene_key']}: missing source_object_ply={ply_path}")
    prompt_source = "legacy_make_prompt"
    prompt_object = item["category"]
    if paper_style_prompts and affordance == "grasp":
        question, answer, prompt_source, prompt_object = make_paper_grasp_prompt(item["category"], question_table)
    else:
        question, answer = make_prompt(item["category"], affordance, prompt_mode, question_table)
    scores, predicted_text = predict_scene(model, ply_path, question, answer, device)
    labels = load_labels(item)
    if len(scores) != len(labels):
        raise ValueError(f"{item['scene_key']}: scores={len(scores)} labels={len(labels)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scene_key=np.array([item["scene_key"]]),
        category=np.array([item["category"]]),
        affordance=np.array([affordance]),
        source_split=np.array([item["source_split"]]),
        source_object_ply=np.array([str(ply_path)]),
        question=np.array([question]),
        answer_prompt=np.array([answer]),
        prompt_source=np.array([prompt_source]),
        prompt_object=np.array([prompt_object]),
        scores=scores.astype(np.float32),
        labels=labels.astype(np.uint8),
        predicted_text=np.array([str(predicted_text)]),
    )
    return scores.astype(np.float32)


def cache_scores(
    items: list[dict],
    affordance: str,
    model,
    question_table,
    device: torch.device,
    score_dir: Path,
    prompt_mode: str,
    paper_style_prompts: bool = False,
) -> dict[str, np.ndarray]:
    scores = {}
    for idx, item in enumerate(items, start=1):
        scores[item["scene_key"]] = compute_or_load_scores(
            item, affordance, model, question_table, device, score_dir, prompt_mode, paper_style_prompts
        )
        if idx % 25 == 0 or idx == len(items):
            print(f"[scores] {idx}/{len(items)} cached", flush=True)
    return scores


def evaluate_phase(
    phase_dir: Path,
    phase_name: str,
    val_items: list[dict],
    test_items: list[dict],
    scores_by_scene: dict[str, np.ndarray],
    selected_threshold: float | None = None,
) -> dict:
    val_labels = [load_labels(item) for item in val_items]
    val_scores = [scores_by_scene[item["scene_key"]] for item in val_items]
    y_val, s_val, val_slices = metric_slices(val_items, val_labels, val_scores)
    if selected_threshold is None:
        th_best, threshold_curve = g15.choose_threshold(y_val, s_val, val_slices)
        threshold = float(th_best["threshold"])
    else:
        threshold = float(selected_threshold)
        threshold_curve = []

    test_labels = [load_labels(item) for item in test_items]
    test_scores = [scores_by_scene[item["scene_key"]] for item in test_items]
    y_test, s_test, test_slices = metric_slices(test_items, test_labels, test_scores)
    per_scene = g15.per_scene_metrics(test_slices, y_test, s_test, threshold)
    per_category = g15.per_category_metrics(per_scene)
    micro = g15.metrics_from_scores(y_test, s_test, threshold)
    macro = {name: g15.macro_scene_score(per_scene, name) for name in g15.METRIC_NAMES}

    save_csv(phase_dir / "per_scene_metrics.csv", per_scene)
    save_csv(phase_dir / "per_category_metrics.csv", per_category)
    save_csv(phase_dir / "threshold_curve.csv", threshold_curve)
    summary = {
        "phase": phase_name,
        "checkpoint": str(CHECKPOINT),
        "selected_threshold": threshold,
        "num_val_scenes": len(val_items),
        "num_test_scenes": len(test_items),
        "macro_scene": macro,
        "micro": micro,
        "metrics": g15.METRIC_NAMES,
        "note": "AffordSplatNet external seen checkpoint; no HANDAL retraining. Fixed threshold if selected_threshold is provided; otherwise threshold selected on validation.",
    }
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_dataset_summary(items: list[dict], out_dir: Path) -> None:
    rows = []
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)
    for category, group in sorted(by_category.items()):
        rows.append(
            {
                "category": category,
                "num_scenes": len(group),
                "num_train": sum(x["source_split"] == "train" for x in group),
                "num_val": sum(x["source_split"] == "val" for x in group),
                "num_test": sum(x["source_split"] == "test" for x in group),
                "num_gaussians": sum(x["num_gaussians"] for x in group),
                "num_handle": sum(x["num_handle"] for x in group),
            }
        )
    save_csv(out_dir / "dataset_summary_by_category.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=Path, default=NPZ_ROOT)
    parser.add_argument("--out_dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--affordance", default="grasp")
    parser.add_argument("--prompt_mode", default="single", choices=["single", "official_all"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include_bad_scenes", action="store_true")
    parser.add_argument("--paper_style_prompts", action="store_true")
    parser.add_argument("--fixed_threshold", type=float, default=None)
    args = parser.parse_args()

    items = load_items(args.npz_root, include_bad_scenes=args.include_bad_scenes)
    if args.limit:
        items = items[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_summary(items, args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_model(device)
    question_table = load_question_table()
    score_dir = args.out_dir / "scores" / args.affordance
    scores_by_scene = cache_scores(items, args.affordance, model, question_table, device, score_dir, args.prompt_mode, args.paper_style_prompts)

    summaries = []
    seen = split_seen(items)
    summaries.append(
        evaluate_phase(
            args.out_dir / "01_seen_instance_17cat" / args.affordance,
            "seen_instance_17cat",
            seen["val"],
            seen["test"],
            scores_by_scene,
            selected_threshold=args.fixed_threshold,
        )
    )

    categories = sorted({item["category"] for item in items})
    loco_rows = []
    for category in categories:
        loco = split_loco(items, category)
        summary = evaluate_phase(
            args.out_dir / "02_leave_one_category_out_17cat" / category / args.affordance,
            f"loco_{category}",
            loco["val"],
            loco["test"],
            scores_by_scene,
            selected_threshold=args.fixed_threshold,
        )
        row = {
            "heldout_category": category,
            "selected_threshold": summary["selected_threshold"],
            "num_test_scenes": summary["num_test_scenes"],
        }
        for name, value in summary["macro_scene"].items():
            row[f"macro_{name}"] = value
        for name, value in summary["micro"].items():
            row[f"micro_{name}"] = value
        loco_rows.append(row)
    save_csv(args.out_dir / "02_leave_one_category_out_17cat" / args.affordance / "loco_summary_by_category.csv", loco_rows)

    overview = {
        "checkpoint": str(CHECKPOINT),
        "npz_root": str(args.npz_root),
        "affordance": args.affordance,
        "prompt_mode": args.prompt_mode,
        "paper_style_prompts": bool(args.paper_style_prompts),
        "fixed_threshold": args.fixed_threshold,
        "num_scenes": len(items),
        "seen": summaries[0],
        "loco_macro_over_categories": {
            name: float(np.nanmean([row[f"macro_{name}"] for row in loco_rows])) for name in g15.METRIC_NAMES
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(overview, indent=2))
    print(json.dumps(overview, indent=2), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
