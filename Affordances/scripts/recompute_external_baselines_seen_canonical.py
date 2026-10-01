#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
NPZ_ROOT = BASE / "data/handal_exp40_15k_gt_features_v1/npz/B_center_ellipsoid20_strict75"
OUT_ROOT = BASE / "outputs/04_3daffordsplat_baseline/external_seen_canonical_v1"
OPENAD_ROOT = BASE / "external/OpenAD"
OPENAD_CKPT = OPENAD_ROOT / "checkpoints/best_model_openad_pn2_estimation.t7"

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


def load_items(npz_root: Path) -> list[dict]:
    items = []
    for path in sorted(npz_root.glob("*.npz")):
        if path.stem in BAD_SCENES:
            continue
        category, scene_id = path.stem.rsplit("__", 1)
        with np.load(path, allow_pickle=False) as z:
            split = scalar_str(z, "model_split", "unknown")
            if split not in {"val", "test"}:
                continue
            labels = z["handle_labels_thr0_25"].astype(np.uint8)
            items.append(
                {
                    "scene_key": path.stem,
                    "category": scalar_str(z, "category", category),
                    "scene_id": scalar_str(z, "scene_id", scene_id),
                    "model_split": split,
                    "npz": str(path),
                    "source_object_ply": scalar_str(z, "source_object_ply", ""),
                    "num_gaussians": int(len(labels)),
                    "num_handle": int(labels.sum()),
                }
            )
    if not items:
        raise FileNotFoundError(f"No validation/test NPZ files found in {npz_root}")
    return items


def load_labels(item: dict) -> np.ndarray:
    with np.load(item["npz"], allow_pickle=False) as z:
        return z["handle_labels_thr0_25"].astype(np.uint8)


def pc_normalize(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz.astype(np.float32, copy=True)
    xyz -= xyz.mean(axis=0, keepdims=True)
    radius = np.sqrt((xyz * xyz).sum(axis=1)).max()
    if radius > 1e-12:
        xyz /= radius
    return xyz


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_slices(items: list[dict], labels: list[np.ndarray], scores: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list]:
    y_all, s_all, slices = [], [], []
    start = 0
    for item, y, s in zip(items, labels, scores):
        if len(y) != len(s):
            raise ValueError(f"{item['scene_key']}: labels={len(y)} scores={len(s)}")
        y_all.append(y.astype(np.uint8, copy=False))
        s_all.append(s.astype(np.float32, copy=False))
        end = start + len(y)
        slices.append((item, start, end))
        start = end
    return np.concatenate(y_all), np.concatenate(s_all), slices


def aggregate_macro(g15, per_scene: list[dict]) -> dict:
    return {name: g15.macro_scene_score(per_scene, name) for name in g15.METRIC_NAMES}


def threshold_curve(g15, y: np.ndarray, scores: np.ndarray, slices: list, thresholds: np.ndarray) -> list[dict]:
    rows = []
    for threshold in thresholds:
        scene_rows = []
        for item, start, end in slices:
            row = {"scene_key": item["scene_key"], "category": item["category"]}
            row.update(g15.metrics_from_scores(y[start:end], scores[start:end], float(threshold)))
            scene_rows.append(row)
        macro = aggregate_macro(g15, scene_rows)
        rows.append({"threshold": float(threshold), **{f"macro_{k}": v for k, v in macro.items()}})
    return rows


def choose_threshold(curve: list[dict], objective: str = "iou") -> dict:
    key = f"macro_{objective}"
    return max(curve, key=lambda row: (float(row[key]), float(row["macro_f1"]), float(row["macro_balanced_accuracy"])))


def per_scene_rows(g15, items: list[dict], labels: list[np.ndarray], scores: list[np.ndarray], threshold: float) -> list[dict]:
    rows = []
    for item, y, s in zip(items, labels, scores):
        row = {
            "scene_key": item["scene_key"],
            "category": item["category"],
            "model_split": item["model_split"],
            "threshold": float(threshold),
        }
        row.update(g15.metrics_from_scores(y, s, float(threshold)))
        rows.append(row)
    return rows


def save_scores(out_dir: Path, method: str, items: list[dict], labels: list[np.ndarray], scores: list[np.ndarray]) -> None:
    score_dir = out_dir / method / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for item, y, s in zip(items, labels, scores):
        np.savez_compressed(
            score_dir / f"{item['scene_key']}.npz",
            scene_key=np.array([item["scene_key"]]),
            category=np.array([item["category"]]),
            model_split=np.array([item["model_split"]]),
            labels=y.astype(np.uint8),
            scores=s.astype(np.float32),
        )


def load_or_compute_openad(items: list[dict], args) -> tuple[list[np.ndarray], list[np.ndarray]]:
    score_dir = args.out_root / "openad" / "scores"
    cached = [score_dir / f"{item['scene_key']}.npz" for item in items]
    if all(path.exists() for path in cached):
        labels, scores = [], []
        for path in cached:
            with np.load(path, allow_pickle=False) as z:
                labels.append(z["labels"].astype(np.uint8))
                scores.append(z["scores"].astype(np.float32))
        return labels, scores

    sys.path.insert(0, str(OPENAD_ROOT))
    from models.openad_pn2 import OpenAD_PN2  # noqa: WPS433

    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f"cuda:{args.gpu}")
    model = OpenAD_PN2(args=None, num_classes=2, normal_channel=False).to(device).eval()
    model.load_state_dict(torch.load(OPENAD_CKPT, map_location="cpu"), strict=True)

    labels, scores = [], []
    with torch.no_grad():
        for idx, item in enumerate(items, start=1):
            with np.load(item["npz"], allow_pickle=False) as z:
                xyz = z["xyz"].astype(np.float32)
                y = z["handle_labels_thr0_25"].astype(np.uint8)
            x = torch.from_numpy(pc_normalize(xyz).T[None]).float().to(device)
            logp = model(x, ["grasp", "none"])
            prob_grasp = torch.exp(logp)[0, 0].detach().cpu().numpy().astype(np.float32)
            labels.append(y)
            scores.append(prob_grasp)
            if idx == 1 or idx % 25 == 0 or idx == len(items):
                print(f"[OpenAD] {idx}/{len(items)} {item['scene_key']}", flush=True)
            del x, logp
    save_scores(args.out_root, "openad", items, labels, scores)
    return labels, scores


def load_or_compute_affordsplatnet(items: list[dict], args) -> tuple[list[np.ndarray], list[np.ndarray]]:
    score_dir = args.out_root / "affordsplatnet" / "scores"
    cached = [score_dir / f"{item['scene_key']}.npz" for item in items]
    if all(path.exists() for path in cached):
        labels, scores = [], []
        for path in cached:
            with np.load(path, allow_pickle=False) as z:
                labels.append(z["labels"].astype(np.uint8))
                scores.append(z["scores"].astype(np.float32))
        return labels, scores

    sys.path.insert(0, str(BASE / "scripts"))
    from affordsplat_paper_prompts import make_paper_grasp_prompt  # noqa: WPS433
    from run_3daffordsplat_handal_eval import CHECKPOINT, load_model, load_question_table, predict_scene  # noqa: WPS433

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    question_table = load_question_table()

    labels, scores = [], []
    for idx, item in enumerate(items, start=1):
        question, answer, prompt_source, prompt_object = make_paper_grasp_prompt(item["category"], question_table)
        ply_path = Path(item["source_object_ply"])
        s, predicted_text = predict_scene(model, ply_path, question, answer, device)
        y = load_labels(item)
        if len(s) != len(y):
            raise ValueError(f"{item['scene_key']}: labels={len(y)} scores={len(s)}")
        labels.append(y)
        scores.append(s.astype(np.float32))
        if idx == 1 or idx % 10 == 0 or idx == len(items):
            print(f"[AffordSplatNet] {idx}/{len(items)} {item['scene_key']} prompt={prompt_source}:{prompt_object}", flush=True)
    save_scores(args.out_root, "affordsplatnet", items, labels, scores)
    return labels, scores


def evaluate_method(method: str, items: list[dict], labels: list[np.ndarray], scores: list[np.ndarray], args) -> dict:
    sys.path.insert(0, str(BASE / "scripts"))
    import train_handal_17cat_graph_attention as g15  # noqa: WPS433

    out_dir = args.out_root / method / "metrics"
    val_idx = [idx for idx, item in enumerate(items) if item["model_split"] == "val"]
    test_idx = [idx for idx, item in enumerate(items) if item["model_split"] == "test"]
    val_items = [items[idx] for idx in val_idx]
    test_items = [items[idx] for idx in test_idx]
    val_labels = [labels[idx] for idx in val_idx]
    test_labels = [labels[idx] for idx in test_idx]
    val_scores = [scores[idx] for idx in val_idx]
    test_scores = [scores[idx] for idx in test_idx]

    y_val, s_val, val_slices = metric_slices(val_items, val_labels, val_scores)
    thresholds = np.linspace(0.0, 1.0, args.num_thresholds, dtype=np.float32)
    curve = threshold_curve(g15, y_val, s_val, val_slices, thresholds)
    best = choose_threshold(curve, args.objective)
    val_threshold = float(best["threshold"])

    rows = []
    for setting, threshold in [("fixed_0.5", 0.5), ("val_threshold", val_threshold)]:
        scene_rows = per_scene_rows(g15, test_items, test_labels, test_scores, threshold)
        macro = aggregate_macro(g15, scene_rows)
        micro_y, micro_s, _ = metric_slices(test_items, test_labels, test_scores)
        micro = g15.metrics_from_scores(micro_y, micro_s, threshold)
        save_csv(out_dir / f"per_scene_{setting}.csv", scene_rows)
        row = {
            "method": method,
            "setting": setting,
            "threshold": threshold,
            "num_val_scenes": len(val_items),
            "num_test_scenes": len(test_items),
            **{f"macro_{k}": v for k, v in macro.items()},
            **{f"micro_{k}": v for k, v in micro.items()},
        }
        rows.append(row)
    save_csv(out_dir / "threshold_curve.csv", curve)
    save_csv(out_dir / "comparison_rows.csv", rows)

    ranking = {
        "method": method,
        "num_test_scenes": len(test_items),
        "macro_roc_auc": rows[0]["macro_roc_auc"],
        "macro_auprc": rows[0]["macro_auprc"],
        "micro_roc_auc": rows[0]["micro_roc_auc"],
        "micro_auprc": rows[0]["micro_auprc"],
    }
    summary = {
        "method": method,
        "npz_root": str(args.npz_root),
        "num_val_scenes": len(val_items),
        "num_test_scenes": len(test_items),
        "objective": args.objective,
        "fixed_threshold": 0.5,
        "selected_threshold": val_threshold,
        "ranking_metrics_from_same_test_scores": ranking,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{method}] selected_threshold={val_threshold:.4f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["openad", "affordsplatnet"], required=True)
    parser.add_argument("--npz_root", type=Path, default=NPZ_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--objective", default="iou", choices=["iou", "f1", "balanced_accuracy"])
    parser.add_argument("--num_thresholds", type=int, default=1001)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    items = load_items(args.npz_root)
    print(f"[dataset] val/test scenes={len(items)} val={sum(i['model_split']=='val' for i in items)} test={sum(i['model_split']=='test' for i in items)}", flush=True)

    if args.method == "openad":
        labels, scores = load_or_compute_openad(items, args)
    else:
        labels, scores = load_or_compute_affordsplatnet(items, args)
    evaluate_method(args.method, items, labels, scores, args)


if __name__ == "__main__":
    main()
