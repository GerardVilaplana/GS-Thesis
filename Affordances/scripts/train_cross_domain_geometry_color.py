#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


C0 = 0.28209479177387814
BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
HANDAL_FEATURE_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
HANDAL_SPLIT = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "01_per_gaussian_mlp_baseline"
    / "exp1a_seen_instance"
    / "split_manifest.json"
)
OFFICIAL_ROOT = BASE_ROOT / "data" / "AffordSplat_GS_grasp_wrap_subset"
OUT_ROOT = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "05_cross_domain_geometry_color_v1"
)


def sigmoid_np(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_xyz(xyz):
    xyz = xyz.astype(np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    return centered / radius


def base_color_from_vertices(vertices):
    names = vertices.dtype.names
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        color = (
            np.vstack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).T.astype(np.float32)
            * C0
            + 0.5
        )
        return np.clip(color, 0.0, 1.0).astype(np.float32)
    if {"red", "green", "blue"}.issubset(names):
        return (
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32)
            / 255.0
        ).astype(np.float32)
    return np.full((len(vertices), 3), 0.5, dtype=np.float32)


def geometry_from_vertices(vertices):
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    scale = np.exp(
        np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T.astype(np.float32)
    )
    rotation = np.vstack(
        [vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]
    ).T.astype(np.float32)
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True).clip(min=1e-12)
    opacity = sigmoid_np(np.asarray(vertices["opacity"], dtype=np.float32))[:, None].astype(np.float32)
    color = base_color_from_vertices(vertices)
    geometry = np.concatenate([xyz, scale, rotation, opacity], axis=1).astype(np.float32)
    return xyz, geometry, color


def feature_matrix(geometry, xyz, color):
    geom = geometry.astype(np.float32).copy()
    geom[:, :3] = normalize_xyz(xyz)
    return np.concatenate([geom, color.astype(np.float32)], axis=1).astype(np.float32)


def labels_from_annotations(base_xyz, anno_paths, tolerance):
    labels = np.zeros(len(base_xyz), dtype=np.uint8)
    if not anno_paths:
        return labels
    tree = cKDTree(base_xyz.astype(np.float64))
    for anno_path in anno_paths:
        anno_vertices = PlyData.read(str(anno_path))["vertex"].data
        anno_xyz = np.vstack([anno_vertices["x"], anno_vertices["y"], anno_vertices["z"]]).T.astype(np.float32)
        if len(anno_xyz) == 0:
            continue
        dist, idx = tree.query(anno_xyz.astype(np.float64), k=1)
        if float(dist.max()) > tolerance:
            raise ValueError(f"Annotation mismatch for {anno_path}: max_dist={float(dist.max()):.6g}")
        labels[idx] = 1
    return labels


def scene_key_from_npz(path):
    category, scene_id = path.stem.rsplit("__", 1)
    return category, scene_id


def read_handal_item(raw):
    path = Path(raw["npz"])
    category, scene_id = scene_key_from_npz(path)
    return {
        "domain": "handal",
        "scene_key": path.stem,
        "category": category,
        "scene_id": scene_id,
        "instance_id": scene_id[:3],
        "npz": str(path),
        "source": str(path),
    }


def load_handal_split(split_path):
    with split_path.open("r") as f:
        payload = json.load(f)
    split = {}
    for split_name, items in payload["splits"].items():
        split[split_name] = [read_handal_item(item) for item in items]
    return split


def load_official_split(dataset_root, target_affordances):
    manifest = pd.read_csv(dataset_root / "manifest.csv")
    manifest = manifest[manifest["affordance"].isin(target_affordances)].copy()
    split = {"train": [], "val": [], "test": []}
    for (split_name, category, gs_ply), group in manifest.groupby(["split", "category", "gs_ply"], sort=True):
        gs_rel = str(gs_ply)
        scene_id = Path(gs_rel).stem.replace("GS_", "")
        affordances = sorted(group["affordance"].unique().tolist())
        anno_paths = [str(dataset_root / str(x)) for x in group["anno_ply"].tolist()]
        item = {
            "domain": "official",
            "scene_key": f"official_{split_name}_{category}_{scene_id}",
            "category": category,
            "scene_id": scene_id,
            "instance_id": scene_id,
            "split": split_name,
            "affordances": "+".join(affordances),
            "gs_ply": str(dataset_root / gs_rel),
            "anno_plys": anno_paths,
            "source": str(dataset_root / gs_rel),
        }
        split[str(split_name)].append(item)
    for name in split:
        split[name] = sorted(split[name], key=lambda x: (x["category"], x["scene_id"], x.get("affordances", "")))
    return split


def load_scene(item, tolerance):
    if item["domain"] == "handal":
        with np.load(item["npz"], allow_pickle=False) as z:
            x = feature_matrix(z["geometry_features"], z["xyz"], z["color"])
            y = z["handle_labels_thr0_25"].astype(np.float32)
            xyz_norm = normalize_xyz(z["xyz"].astype(np.float32))
        return x, y, xyz_norm

    ply = PlyData.read(item["gs_ply"])
    vertices = ply["vertex"].data
    xyz, geometry, color = geometry_from_vertices(vertices)
    y = labels_from_annotations(xyz, [Path(p) for p in item["anno_plys"]], tolerance).astype(np.float32)
    x = feature_matrix(geometry, xyz, color)
    xyz_norm = normalize_xyz(xyz)
    return x, y, xyz_norm


def local_mean_features(x, xyz_norm, k):
    n = len(x)
    if n <= 1:
        return x.copy()
    nn = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm="auto").fit(xyz_norm)
    indices = nbrs.kneighbors(xyz_norm, return_distance=False)
    if indices.shape[1] > 1:
        indices = indices[:, 1:]
    return x[indices].mean(axis=1).astype(np.float32)


def augment_scene(x, xyz_norm, model_variant, k_neighbors):
    if model_variant == "mlp_geometry_color":
        return x
    if model_variant == "knn_geometry_color":
        global_mean = np.repeat(x.mean(axis=0, keepdims=True), len(x), axis=0)
        local_mean = local_mean_features(x, xyz_norm, k_neighbors)
        return np.concatenate([x, global_mean, local_mean], axis=1).astype(np.float32)
    raise ValueError(f"Unknown model variant: {model_variant}")


def load_arrays(items, args):
    xs, ys, slices = [], [], []
    offset = 0
    for item in items:
        x, y, xyz_norm = load_scene(item, args.match_tolerance)
        x = augment_scene(x, xyz_norm, args.model_variant, args.k_neighbors)
        xs.append(x)
        ys.append(y)
        slices.append((item, offset, offset + len(y)))
        offset += len(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), slices


def standardize_fit(x):
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(192, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict_scores(model, x, batch_size, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            logits.append(model(xb).detach().cpu().numpy())
    return sigmoid_np(np.concatenate(logits, axis=0))


def metrics_from_scores(y_true, score, threshold):
    y_true = y_true.astype(bool)
    y_pred = score >= threshold
    tp = int(np.logical_and(y_pred, y_true).sum())
    tn = int(np.logical_and(~y_pred, ~y_true).sum())
    fp = int(np.logical_and(y_pred, ~y_true).sum())
    fn = int(np.logical_and(~y_pred, y_true).sum())
    n = max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "num_gaussians": int(n),
        "gt_handle_ratio": float(y_true.mean()) if n else 0.0,
        "pred_handle_ratio": float(y_pred.mean()) if n else 0.0,
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def per_scene_metrics(slices, y_all, score_all, threshold):
    rows = []
    for item, start, end in slices:
        row = {
            "domain": item["domain"],
            "scene_key": item["scene_key"],
            "category": item["category"],
            "scene_id": item["scene_id"],
            "source": item["source"],
            "affordances": item.get("affordances", "handle"),
        }
        row.update(metrics_from_scores(y_all[start:end], score_all[start:end], threshold))
        rows.append(row)
    return rows


def macro_scene_score(rows, metric):
    return float(np.nanmean([row[metric] for row in rows]))


def choose_threshold(y_val, score_val, val_slices):
    best = None
    curve = []
    for threshold in np.linspace(0.05, 0.95, 91):
        rows = per_scene_metrics(val_slices, y_val, score_val, float(threshold))
        entry = {
            "threshold": float(threshold),
            "macro_scene_iou": macro_scene_score(rows, "iou"),
            "macro_scene_f1": macro_scene_score(rows, "f1"),
        }
        curve.append(entry)
        if best is None or (entry["macro_scene_iou"], entry["macro_scene_f1"]) > (
            best["macro_scene_iou"],
            best["macro_scene_f1"],
        ):
            best = entry
    return best, curve


def aggregate_scene_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        entry = {name: value for name, value in zip(keys, key)}
        for metric in [
            "iou",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
            "gt_handle_ratio",
            "pred_handle_ratio",
        ]:
            entry[f"macro_scene_{metric}"] = float(np.nanmean([r[metric] for r in group]))
        entry["num_scenes"] = len(group)
        entry["num_gaussians"] = int(sum(r["num_gaussians"] for r in group))
        out.append(entry)
    return out


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def domain_split_counts(split, name):
    rows = []
    for split_name, items in split.items():
        by_category = defaultdict(list)
        for item in items:
            by_category[item["category"]].append(item)
        for category, group in sorted(by_category.items()):
            rows.append(
                {
                    "dataset": name,
                    "split": split_name,
                    "category": category,
                    "num_scenes": len(group),
                }
            )
    return rows


def make_experiment_splits(handal, official):
    return {
        "train_handal": {
            "train": handal["train"],
            "val": handal["val"],
            "eval": {"handal": handal["test"], "official": official["test"]},
        },
        "train_official": {
            "train": official["train"],
            "val": official["val"],
            "eval": {"handal": handal["test"], "official": official["test"]},
        },
        "train_mixed": {
            "train": handal["train"] + official["train"],
            "val": handal["val"] + official["val"],
            "eval": {"handal": handal["test"], "official": official["test"]},
        },
    }


def train_one(name, split, args, device):
    out_dir = args.out_root / name / args.model_variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}/{args.model_variant}] loading train/val arrays", flush=True)
    x_train, y_train, _ = load_arrays(split["train"], args)
    x_val, y_val, val_slices = load_arrays(split["val"], args)
    mean, std = standardize_fit(x_train)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    model = MLP(x_train.shape[1]).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    best = None
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_score = predict_scores(model, x_val, args.batch_size, device)
        th_best, _ = choose_threshold(y_val, val_score, val_slices)
        val_rows = per_scene_metrics(val_slices, y_val, val_score, th_best["threshold"])
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_scene_iou": macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": macro_scene_score(val_rows, "f1"),
            "val_threshold": th_best["threshold"],
        }
        history.append(record)
        score_tuple = (record["val_macro_scene_iou"], record["val_macro_scene_f1"])
        if best is None or score_tuple > (best["val_macro_scene_iou"], best["val_macro_scene_f1"]):
            best = dict(record)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[{name}/{args.model_variant}] epoch {epoch:03d}/{args.epochs} "
                f"loss={record['loss']:.4f} val_iou={record['val_macro_scene_iou']:.3f} "
                f"val_f1={record['val_macro_scene_f1']:.3f} th={record['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"[{name}/{args.model_variant}] early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    val_score = predict_scores(model, x_val, args.batch_size, device)
    th_best, threshold_curve = choose_threshold(y_val, val_score, val_slices)
    threshold = float(th_best["threshold"])
    save_csv(out_dir / "history.csv", history)
    save_csv(out_dir / "threshold_curve.csv", threshold_curve)

    eval_summaries = []
    all_scene_rows = []
    for eval_domain, eval_items in split["eval"].items():
        print(f"[{name}/{args.model_variant}] evaluating on {eval_domain}", flush=True)
        x_test, y_test, test_slices = load_arrays(eval_items, args)
        x_test = (x_test - mean) / std
        test_score = predict_scores(model, x_test, args.batch_size, device)
        scene_rows = per_scene_metrics(test_slices, y_test, test_score, threshold)
        for row in scene_rows:
            row["train_setup"] = name
            row["model_variant"] = args.model_variant
            row["selected_threshold"] = threshold
        save_csv(out_dir / f"per_scene_metrics_eval_{eval_domain}.csv", scene_rows)
        save_csv(out_dir / f"per_category_metrics_eval_{eval_domain}.csv", aggregate_scene_rows(scene_rows, ["category"]))
        micro = metrics_from_scores(y_test, test_score, threshold)
        macro = {
            metric: macro_scene_score(scene_rows, metric)
            for metric in [
                "iou",
                "f1",
                "precision",
                "recall",
                "balanced_accuracy",
                "gt_handle_ratio",
                "pred_handle_ratio",
            ]
        }
        summary = {
            "train_setup": name,
            "eval_domain": eval_domain,
            "model_variant": args.model_variant,
            "selected_epoch": int(best["epoch"]),
            "selected_threshold": threshold,
            "train_scenes": len(split["train"]),
            "val_scenes": len(split["val"]),
            "test_scenes": len(eval_items),
            "train_gaussians": int(len(y_train)),
            "val_gaussians": int(len(y_val)),
            "test_gaussians": int(len(y_test)),
            "train_positive_ratio": float(y_train.mean()),
            "val_positive_ratio": float(y_val.mean()),
            "test_positive_ratio": float(y_test.mean()),
            "macro_scene": macro,
            "micro": micro,
        }
        eval_summaries.append(summary)
        all_scene_rows.extend(scene_rows)
        print(
            f"[{name}/{args.model_variant}] {eval_domain}: "
            f"macro IoU={macro['iou']:.3f} F1={macro['f1']:.3f} "
            f"pred={macro['pred_handle_ratio']:.3f} gt={macro['gt_handle_ratio']:.3f}",
            flush=True,
        )

    with (out_dir / "overall_metrics.json").open("w") as f:
        json.dump(eval_summaries, f, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "model_variant": args.model_variant,
            "threshold": threshold,
            "train_setup": name,
            "eval_summaries": eval_summaries,
        },
        out_dir / "model.pt",
    )
    return eval_summaries, all_scene_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handal_split", type=Path, default=HANDAL_SPLIT)
    parser.add_argument("--official_root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--target_affordances", nargs="+", default=["grasp", "wrap_grasp"])
    parser.add_argument("--model_variant", choices=["mlp_geometry_color", "knn_geometry_color"], default="knn_geometry_color")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument("--match_tolerance", type=float, default=1e-8)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    handal = load_handal_split(args.handal_split)
    official = load_official_split(args.official_root, args.target_affordances)
    splits = make_experiment_splits(handal, official)

    split_rows = domain_split_counts(handal, "handal") + domain_split_counts(official, "official")
    save_csv(args.out_root / "split_counts.csv", split_rows)
    with (args.out_root / "experiment_definition.json").open("w") as f:
        json.dump(
            {
                "target": "binary grasp-like/contact region",
                "handal_label": "handle_labels_thr0_25",
                "official_label": "union of target affordance annotation PLYs per base Gaussian object",
                "target_affordances": args.target_affordances,
                "model_variant": args.model_variant,
                "feature_layout": "geometry_features[x,y,z,scale0,scale1,scale2,rot0,rot1,rot2,rot3,opacity] + color[RGB]; xyz normalized per object",
                "train_setups": list(splits.keys()),
            },
            f,
            indent=2,
        )

    all_summaries = []
    all_scene_rows = []
    for name, split in splits.items():
        summaries, scene_rows = train_one(name, split, args, device)
        all_summaries.extend(summaries)
        all_scene_rows.extend(scene_rows)

    flat_summary = []
    for item in all_summaries:
        row = {
            "train_setup": item["train_setup"],
            "eval_domain": item["eval_domain"],
            "model_variant": item["model_variant"],
            "selected_epoch": item["selected_epoch"],
            "selected_threshold": item["selected_threshold"],
            "train_scenes": item["train_scenes"],
            "val_scenes": item["val_scenes"],
            "test_scenes": item["test_scenes"],
            "train_gaussians": item["train_gaussians"],
            "val_gaussians": item["val_gaussians"],
            "test_gaussians": item["test_gaussians"],
            "train_positive_ratio": item["train_positive_ratio"],
            "val_positive_ratio": item["val_positive_ratio"],
            "test_positive_ratio": item["test_positive_ratio"],
        }
        for key, value in item["macro_scene"].items():
            row[f"macro_scene_{key}"] = value
        for key, value in item["micro"].items():
            if isinstance(value, (int, float)):
                row[f"micro_{key}"] = value
        flat_summary.append(row)
    save_csv(args.out_root / "cross_domain_summary.csv", flat_summary)
    save_csv(args.out_root / "all_per_scene_metrics.csv", all_scene_rows)
    save_csv(args.out_root / "all_per_category_metrics.csv", aggregate_scene_rows(all_scene_rows, ["train_setup", "domain", "category"]))
    print(f"wrote {args.out_root / 'cross_domain_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
