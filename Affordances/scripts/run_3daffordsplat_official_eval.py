#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from run_3daffordsplat_handal_eval import (
    C0,
    load_model,
    load_question_table,
    make_prompt,
    metric_row,
    predict_scene,
)


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
DEFAULT_DATASET = BASE_ROOT / "data" / "AffordSplat_GS_grasp_wrap_subset"
DEFAULT_OUT = BASE_ROOT / "outputs" / "04_3daffordsplat_baseline" / "official_affordsplat_test"


def read_xyz(path):
    vertices = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    return xyz, vertices


def labels_from_annotation(base_ply, anno_ply, tolerance=1e-8):
    base_xyz, _ = read_xyz(base_ply)
    anno_xyz, _ = read_xyz(anno_ply)
    labels = np.zeros(len(base_xyz), dtype=np.uint8)
    if len(anno_xyz) == 0:
        return labels, {"anno_vertices": 0, "unique_matches": 0, "max_match_dist": 0.0}
    tree = cKDTree(base_xyz.astype(np.float64))
    dist, idx = tree.query(anno_xyz.astype(np.float64), k=1)
    if float(dist.max()) > tolerance:
        raise ValueError(
            f"Annotation vertices do not match base PLY within tolerance: "
            f"{anno_ply} max_dist={float(dist.max()):.6g}"
        )
    labels[idx] = 1
    return labels, {
        "anno_vertices": int(len(anno_xyz)),
        "unique_matches": int(len(np.unique(idx))),
        "max_match_dist": float(dist.max()),
    }


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def write_score_ply(source_ply, out_ply, scores):
    ply = PlyData.read(str(source_ply))
    vertices = ply["vertex"].data.copy()
    scores = np.asarray(scores, dtype=np.float32)
    if len(vertices) != len(scores):
        raise ValueError(f"PLY/score length mismatch: {source_ply} {len(vertices)} vs {len(scores)}")
    rgb = np.zeros((len(scores), 3), dtype=np.float32)
    rgb[:, 0] = np.maximum(np.clip(scores, 0.0, 1.0), 0.05)
    rgb[:, 1] = 0.04
    rgb[:, 2] = 1.0 - np.clip(scores, 0.0, 1.0)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(out_ply))


def write_overlap_ply(source_ply, out_ply, scores, labels, threshold):
    ply = PlyData.read(str(source_ply))
    vertices = ply["vertex"].data.copy()
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels).astype(bool)
    pred = scores >= threshold
    rgb = np.zeros((len(scores), 3), dtype=np.float32)
    rgb[~pred & ~labels] = np.array([0.05, 0.18, 0.95], dtype=np.float32)
    rgb[pred & ~labels] = np.array([1.0, 0.05, 0.05], dtype=np.float32)
    rgb[~pred & labels] = np.array([0.0, 0.95, 0.2], dtype=np.float32)
    rgb[pred & labels] = np.array([1.0, 0.85, 0.0], dtype=np.float32)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(out_ply))


def load_manifest(dataset_root, split, affordances):
    manifest = pd.read_csv(dataset_root / "manifest.csv")
    manifest = manifest[manifest["split"] == split].copy()
    if affordances:
        manifest = manifest[manifest["affordance"].isin(affordances)].copy()
    manifest = manifest.sort_values(["affordance", "category", "gs_ply", "anno_ply"]).reset_index(drop=True)
    return manifest


def aggregate(rows, group_keys):
    out = []
    df = pd.DataFrame(rows)
    for key, group in df.groupby(group_keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        entry = {name: value for name, value in zip(group_keys, key)}
        tp, fp, fn, tn = (int(group[col].sum()) for col in ["tp", "fp", "fn", "tn"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        iou = tp / max(tp + fp + fn, 1)
        entry.update(
            num_scene_queries=int(len(group)),
            num_gaussians=int(group["num_gaussians"].sum()),
            macro_iou=float(group["iou"].mean()),
            macro_f1=float(group["f1"].mean()),
            macro_precision=float(group["precision"].mean()),
            macro_recall=float(group["recall"].mean()),
            macro_auc=float(group["auc"].mean()),
            macro_gt_ratio=float(group["gt_handle_ratio"].mean()),
            macro_pred_ratio=float(group["pred_handle_ratio"].mean()),
            micro_iou=float(iou),
            micro_f1=float(f1),
            micro_precision=float(precision),
            micro_recall=float(recall),
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
        )
        out.append(entry)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def threshold_sweep(rows, score_dir, thresholds):
    by_aff = {}
    for row in rows:
        by_aff.setdefault(row["affordance"], []).append(row)
    sweep_rows = []
    for aff, aff_rows in by_aff.items():
        labels_all = []
        scores_all = []
        scene_cache = []
        for row in aff_rows:
            pack = np.load(score_dir / f"{row['query_key']}_scores.npz")
            labels = pack["labels"].astype(np.uint8)
            scores = pack["scores"].astype(np.float32)
            labels_all.append(labels)
            scores_all.append(scores)
            scene_cache.append((labels, scores))
        labels_cat = np.concatenate(labels_all)
        scores_cat = np.concatenate(scores_all)
        for threshold in thresholds:
            per_scene = [metric_row(labels, scores, threshold) for labels, scores in scene_cache]
            micro = metric_row(labels_cat, scores_cat, threshold)
            sweep_rows.append(
                {
                    "affordance": aff,
                    "threshold": float(threshold),
                    "num_scene_queries": len(per_scene),
                    "macro_iou": float(np.mean([m["iou"] for m in per_scene])),
                    "macro_f1": float(np.mean([m["f1"] for m in per_scene])),
                    "macro_precision": float(np.mean([m["precision"] for m in per_scene])),
                    "macro_recall": float(np.mean([m["recall"] for m in per_scene])),
                    "macro_pred_ratio": float(np.mean([m["pred_handle_ratio"] for m in per_scene])),
                    "micro_iou": micro["iou"],
                    "micro_f1": micro["f1"],
                    "micro_precision": micro["precision"],
                    "micro_recall": micro["recall"],
                    "micro_pred_ratio": micro["pred_handle_ratio"],
                }
            )
    return sweep_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--affordances", nargs="*", default=["grasp", "wrap_grasp"])
    parser.add_argument("--prompt_mode", default="official_all", choices=["single", "official_all"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export_plys_per_group", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_model(device)
    question_table = load_question_table()
    manifest = load_manifest(args.dataset_root, args.split, args.affordances)
    if args.limit:
        manifest = manifest.head(args.limit).copy()

    rows = []
    export_counts = {}
    score_dir = args.out_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    for _, item in manifest.iterrows():
        category = str(item["category"])
        affordance = str(item["affordance"])
        gs_ply = args.dataset_root / str(item["gs_ply"])
        anno_ply = args.dataset_root / str(item["anno_ply"])
        scene_id = Path(str(item["gs_ply"])).stem.replace("GS_", "")
        query_key = f"{args.split}_{category}_{affordance}_{scene_id}"

        labels, label_info = labels_from_annotation(gs_ply, anno_ply)
        question, answer = make_prompt(category, affordance, args.prompt_mode, question_table)
        scores, predicted_text = predict_scene(model, gs_ply, question, answer, device)
        if len(scores) != len(labels):
            raise ValueError(f"{query_key}: scores={len(scores)} labels={len(labels)}")

        np.savez_compressed(
            score_dir / f"{query_key}_scores.npz",
            scores=scores.astype(np.float32),
            labels=labels.astype(np.uint8),
            category=np.array([category]),
            affordance=np.array([affordance]),
            split=np.array([args.split]),
            gs_ply=np.array([str(gs_ply)]),
            anno_ply=np.array([str(anno_ply)]),
        )

        row = {
            "query_key": query_key,
            "split": args.split,
            "category": category,
            "affordance": affordance,
            "scene_id": scene_id,
            "gs_ply": str(gs_ply),
            "anno_ply": str(anno_ply),
            "question": question,
            "answer_prompt": answer,
            "predicted_text": predicted_text,
            **label_info,
        }
        row.update(metric_row(labels, scores, args.threshold))
        rows.append(row)

        group = (category, affordance)
        if export_counts.get(group, 0) < args.export_plys_per_group:
            score_ply = args.out_dir / "prediction_ply_raw_score" / affordance / category / f"{query_key}_raw_score.ply"
            overlap_ply = args.out_dir / "prediction_ply_overlap" / affordance / category / f"{query_key}_overlap_thr{args.threshold:.2f}.ply"
            write_score_ply(gs_ply, score_ply, scores)
            write_overlap_ply(gs_ply, overlap_ply, scores, labels, args.threshold)
            row["raw_score_ply"] = str(score_ply)
            row["overlap_ply"] = str(overlap_ply)
            export_counts[group] = export_counts.get(group, 0) + 1

        print(
            f"{query_key}: IoU={row['iou']:.3f} F1={row['f1']:.3f} "
            f"pred={row['pred_handle_ratio']:.3f} gt={row['gt_handle_ratio']:.3f}"
        )

    write_csv(args.out_dir / "per_scene_metrics.csv", rows)
    write_csv(args.out_dir / "per_category_affordance_metrics.csv", aggregate(rows, ["affordance", "category"]))
    write_csv(args.out_dir / "per_affordance_metrics.csv", aggregate(rows, ["affordance"]))
    write_csv(args.out_dir / "per_category_metrics.csv", aggregate(rows, ["category"]))

    thresholds = np.linspace(0.0, 1.0, 101)
    sweep_rows = threshold_sweep(rows, score_dir, thresholds)
    write_csv(args.out_dir / "threshold_sweep.csv", sweep_rows)
    best_rows = []
    for aff, group in pd.DataFrame(sweep_rows).groupby("affordance"):
        for metric in ["macro_iou", "macro_f1", "micro_iou", "micro_f1"]:
            best = group.sort_values(metric, ascending=False).iloc[0].to_dict()
            best_rows.append({"affordance": aff, "selection": f"best_{metric}", **best})
    write_csv(args.out_dir / "best_threshold_summary.csv", best_rows)

    summary = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "affordances": args.affordances,
        "threshold": args.threshold,
        "prompt_mode": args.prompt_mode,
        "num_scene_queries": len(rows),
        "per_scene_metrics": str(args.out_dir / "per_scene_metrics.csv"),
        "per_category_affordance_metrics": str(args.out_dir / "per_category_affordance_metrics.csv"),
        "per_affordance_metrics": str(args.out_dir / "per_affordance_metrics.csv"),
        "threshold_sweep": str(args.out_dir / "threshold_sweep.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
