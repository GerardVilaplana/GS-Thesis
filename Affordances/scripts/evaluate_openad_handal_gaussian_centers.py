#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
OPENAD_ROOT = BASE / "external" / "OpenAD"
NPZ_ROOT = BASE / "data" / "handal_exp40_15k_gt_features_v1" / "npz" / "B_center_ellipsoid20_strict75"
OUT_ROOT = BASE / "outputs" / "03_handle_generalization" / "51_openad_pointcloud_baseline_v1"
CKPT = OPENAD_ROOT / "checkpoints" / "best_model_openad_pn2_estimation.t7"

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

COLORS = {
    "tp": (34, 197, 94),
    "tn": (37, 99, 235),
    "fp": (245, 158, 11),
    "fn": (220, 38, 38),
}
C0 = 0.28209479177387814


def scalar_str(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    value = z[key]
    return str(value[0] if getattr(value, "shape", ()) else value)


def pc_normalize(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz.astype(np.float32, copy=True)
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz -= centroid
    radius = np.sqrt((xyz * xyz).sum(axis=1)).max()
    if radius > 1e-12:
        xyz /= radius
    return xyz


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> dict:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.logical_and(y_true, y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())
    n = int(len(y_true))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(n, 1)
    try:
        bal_acc = float(balanced_accuracy_score(y_true.astype(np.uint8), y_pred.astype(np.uint8)))
    except ValueError:
        bal_acc = math.nan
    try:
        roc_auc = float(roc_auc_score(y_true.astype(np.uint8), prob))
    except ValueError:
        roc_auc = math.nan
    try:
        auprc = float(average_precision_score(y_true.astype(np.uint8), prob))
    except ValueError:
        auprc = math.nan
    return {
        "num_points": n,
        "num_gt_handle": int(y_true.sum()),
        "num_pred_handle": int(y_pred.sum()),
        "gt_handle_ratio": float(y_true.mean()) if n else 0.0,
        "pred_handle_ratio": float(y_pred.mean()) if n else 0.0,
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "auprc": auprc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp_percent": tp / max(n, 1),
        "tn_percent": tn / max(n, 1),
        "fp_percent": fp / max(n, 1),
        "fn_percent": fn / max(n, 1),
        "correct_percent": (tp + tn) / max(n, 1),
        "incorrect_percent": (fp + fn) / max(n, 1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def rgb_to_sh(rgb_01: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb_01, dtype=np.float32) - 0.5) / C0


def confusion_rgb(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    rgb = np.zeros((len(y_true), 3), dtype=np.float32)
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    rgb[np.logical_and(y_true, y_pred)] = np.asarray(COLORS["tp"], dtype=np.float32) / 255.0
    rgb[np.logical_and(~y_true, ~y_pred)] = np.asarray(COLORS["tn"], dtype=np.float32) / 255.0
    rgb[np.logical_and(~y_true, y_pred)] = np.asarray(COLORS["fp"], dtype=np.float32) / 255.0
    rgb[np.logical_and(y_true, ~y_pred)] = np.asarray(COLORS["fn"], dtype=np.float32) / 255.0
    return rgb


def write_confusion_ply(path: Path, source_ply: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(y_true):
        raise ValueError(f"PLY/label length mismatch for {source_ply}: {len(vertices)} vs {len(y_true)}")
    sh = rgb_to_sh(confusion_rgb(y_true, y_pred))
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    elements = []
    for element in ply.elements:
        elements.append(PlyElement.describe(vertices, "vertex") if element.name == "vertex" else element)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(path))


def load_items(npz_root: Path, limit: int | None) -> list[dict]:
    items = []
    for path in sorted(npz_root.glob("*.npz")):
        if path.stem in BAD_SCENES:
            continue
        category, scene_id = path.stem.rsplit("__", 1)
        with np.load(path, allow_pickle=False) as z:
            items.append(
                {
                    "scene_key": path.stem,
                    "category": scalar_str(z, "category", category),
                    "scene_id": scalar_str(z, "scene_id", scene_id),
                    "instance_id": scalar_str(z, "instance_id", scene_id),
                    "model_split": scalar_str(z, "model_split", "unknown"),
                    "raw_split": scalar_str(z, "raw_split", "unknown"),
                    "npz": str(path),
                    "source_object_ply": scalar_str(z, "source_object_ply", ""),
                }
            )
    return items[:limit] if limit else items


def aggregate(rows: list[dict], group_key: str) -> list[dict]:
    out = []
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    for group, items in sorted(groups.items()):
        agg = {"group_by": group_key, group_key: group, "num_scenes": len(items)}
        for key in [
            "num_points",
            "num_gt_handle",
            "num_pred_handle",
            "tp",
            "tn",
            "fp",
            "fn",
        ]:
            agg[key] = int(sum(int(x[key]) for x in items))
        for key in [
            "iou",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "auprc",
            "gt_handle_ratio",
            "pred_handle_ratio",
        ]:
            vals = [float(x[key]) for x in items if not math.isnan(float(x[key]))]
            agg[f"mean_{key}"] = float(np.mean(vals)) if vals else math.nan
        out.append(agg)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=Path, default=NPZ_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--openad_root", type=Path, default=OPENAD_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--affordances", nargs="+", default=["grasp", "none"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_plys", action="store_true")
    parser.add_argument("--summary_every", type=int, default=25)
    args = parser.parse_args()

    if args.affordances[:2] != ["grasp", "none"]:
        raise ValueError("This baseline currently assumes class 0 is grasp and class 1 is none.")

    sys.path.insert(0, str(args.openad_root))
    from models.openad_pn2 import OpenAD_PN2  # noqa: WPS433

    torch.cuda.set_device(int(args.gpu))
    model = OpenAD_PN2(args=None, num_classes=len(args.affordances), normal_channel=False).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)

    items = load_items(args.npz_root, args.limit)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    all_counter = Counter()

    with torch.no_grad():
        for idx, item in enumerate(items, start=1):
            with np.load(item["npz"], allow_pickle=False) as z:
                xyz = z["xyz"].astype(np.float32)
                y = z["handle_labels_thr0_25"].astype(np.uint8)
            xyz_norm = pc_normalize(xyz)
            x = torch.from_numpy(xyz_norm.T[None]).float().cuda()
            logp = model(x, args.affordances)
            p = torch.exp(logp)[0].permute(1, 0).detach().cpu().numpy()
            pred_class = np.argmax(p, axis=1)
            prob_grasp = p[:, 0]
            pred = (pred_class == 0).astype(np.uint8)
            row = {
                **item,
                "checkpoint": str(args.checkpoint),
                "affordances": "|".join(args.affordances),
                **metrics(y, pred, prob_grasp),
            }
            rows.append(row)
            if args.save_plys:
                ply_path = args.out_root / "confusion_plys" / item["category"] / f"{item['scene_key']}_openad_grasp_none_confusion.ply"
                write_confusion_ply(ply_path, Path(item["source_object_ply"]), y, pred)
                row["confusion_ply"] = str(ply_path)
            all_counter.update([item["model_split"]])
            if idx == 1 or idx == len(items) or idx % args.summary_every == 0:
                print(f"[{idx}/{len(items)}] {item['scene_key']} IoU={row['iou']:.4f} pred={row['pred_handle_ratio']:.3f}", flush=True)

    write_csv(args.out_root / "metrics" / "per_scene_metrics.csv", rows)
    write_csv(args.out_root / "metrics" / "per_category_metrics.csv", aggregate(rows, "category"))
    write_csv(args.out_root / "metrics" / "per_model_split_metrics.csv", aggregate(rows, "model_split"))
    write_csv(args.out_root / "metrics" / "per_raw_split_metrics.csv", aggregate(rows, "raw_split"))

    overall = metrics(
        np.concatenate([np.load(r["npz"], allow_pickle=False)["handle_labels_thr0_25"].astype(np.uint8) for r in rows]),
        np.concatenate([
            np.array([], dtype=np.uint8)
            for _ in []
        ]),
        np.array([], dtype=np.float32),
    ) if False else {}
    totals = {
        "num_scenes": len(rows),
        "splits": dict(all_counter),
        "mean_scene_iou": float(np.mean([r["iou"] for r in rows])) if rows else math.nan,
        "mean_scene_f1": float(np.mean([r["f1"] for r in rows])) if rows else math.nan,
        "mean_scene_precision": float(np.mean([r["precision"] for r in rows])) if rows else math.nan,
        "mean_scene_recall": float(np.mean([r["recall"] for r in rows])) if rows else math.nan,
        "mean_scene_accuracy": float(np.mean([r["accuracy"] for r in rows])) if rows else math.nan,
        "mean_scene_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in rows])) if rows else math.nan,
        "mean_scene_roc_auc": float(np.nanmean([r["roc_auc"] for r in rows])) if rows else math.nan,
        "mean_scene_auprc": float(np.nanmean([r["auprc"] for r in rows])) if rows else math.nan,
        "total_points": int(sum(r["num_points"] for r in rows)),
        "total_gt_handle": int(sum(r["num_gt_handle"] for r in rows)),
        "total_pred_handle": int(sum(r["num_pred_handle"] for r in rows)),
        "bad_scenes_excluded": sorted(BAD_SCENES),
        "note": "OpenAD PN2 baseline on Gaussian centers only; prompt classes are grasp and none.",
    }
    (args.out_root / "metrics").mkdir(parents=True, exist_ok=True)
    (args.out_root / "metrics" / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    md = [
        "# OpenAD Gaussian-Center Point-Cloud Baseline",
        "",
        f"- scenes: {totals['num_scenes']}",
        f"- affordances: {' | '.join(args.affordances)}",
        f"- checkpoint: `{args.checkpoint}`",
        f"- mean scene IoU: {totals['mean_scene_iou']:.4f}",
        f"- mean scene F1: {totals['mean_scene_f1']:.4f}",
        f"- total points: {totals['total_points']}",
        "",
        "Input points are the centers of the cleaned object-pruned HANDAL Gaussians. Gaussian color, opacity, scale, rotation, and DINO features are not used by this baseline.",
    ]
    (args.out_root / "metrics" / "metrics.md").write_text("\n".join(md) + "\n")
    print(f"[done] wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
