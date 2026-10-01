#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import train_handal_17cat_mlp_baselines as base  # noqa: E402


BASELINE_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "08_17category_mlp_baselines_v1"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "15_graph_attention_gaussian_v1"

FEATURE_VARIANTS = ["geometry_color_scene_norm", "dino_geometry_color"]


def scene_zscore(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    return ((x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), eps)).astype(np.float32)


def scene_minmax_unit(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    return ((x - lo) / np.maximum(hi - lo, eps)).astype(np.float32)


def load_feature_arrays(item, feature_variant):
    with np.load(item["npz"], allow_pickle=False) as z:
        xyz = z["xyz"].astype(np.float32)
        xyz_norm = base.normalize_xyz(xyz)
        scale = z["scale"].astype(np.float32)
        rotation = z["rotation"].astype(np.float32)
        opacity = np.asarray(z["opacity"], dtype=np.float32).reshape(-1, 1)
        color = z["color"].astype(np.float32)
        y = z["handle_labels_thr0_25"].astype(np.float32)
        if feature_variant == "geometry_color_scene_norm":
            x = np.concatenate(
                [
                    xyz_norm,
                    scene_zscore(scale),
                    rotation,
                    scene_minmax_unit(opacity),
                    scene_zscore(color),
                ],
                axis=1,
            )
        elif feature_variant == "dino_geometry_color":
            geom = z["geometry_features"].astype(np.float32).copy()
            geom[:, :3] = xyz_norm
            x = np.concatenate([z["dino_features"].astype(np.float32), geom, color], axis=1)
        else:
            raise ValueError(f"Unknown feature variant: {feature_variant}")
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False), xyz_norm.astype(np.float32, copy=False)


def build_knn(xyz_norm, max_k):
    n = len(xyz_norm)
    if n == 0:
        return np.empty((0, max_k), dtype=np.int64)
    if n == 1:
        return np.zeros((1, max_k), dtype=np.int64)
    query_k = min(max_k + 1, n)
    _, idx = cKDTree(xyz_norm.astype(np.float32, copy=False)).query(
        xyz_norm.astype(np.float32, copy=False), k=query_k, workers=-1
    )
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[:, None]
    cleaned = []
    arange = np.arange(n, dtype=np.int64)
    for row, self_idx in zip(idx, arange):
        row = row[row != self_idx]
        if len(row) == 0:
            row = np.asarray([self_idx], dtype=np.int64)
        if len(row) < max_k:
            row = np.pad(row, (0, max_k - len(row)), mode="edge")
        cleaned.append(row[:max_k])
    return np.stack(cleaned, axis=0).astype(np.int64, copy=False)


def fit_standardizer(items, feature_variant):
    total = 0
    sum_x = None
    sumsq_x = None
    pos = 0
    for item in items:
        x, y, _ = load_feature_arrays(item, feature_variant)
        x64 = x.astype(np.float64, copy=False)
        sum_x = x64.sum(axis=0) if sum_x is None else sum_x + x64.sum(axis=0)
        sumsq_x = np.square(x64).sum(axis=0) if sumsq_x is None else sumsq_x + np.square(x64).sum(axis=0)
        total += len(x)
        pos += int(y.sum())
    mean64 = sum_x / max(total, 1)
    var64 = (sumsq_x / max(total, 1)) - np.square(mean64)
    mean = mean64.astype(np.float32)[None, :]
    std = np.sqrt(np.maximum(var64, 1e-12)).astype(np.float32)[None, :]
    return mean, np.maximum(std, 1e-6), total, pos


def load_scene_list(items, feature_variant, mean, std, max_k):
    scenes = []
    for item in items:
        x, y, xyz_norm = load_feature_arrays(item, feature_variant)
        scenes.append(
            {
                "item": item,
                "x": ((x - mean) / std).astype(np.float32, copy=False),
                "y": y.astype(np.float32, copy=False),
                "knn": build_knn(xyz_norm, max_k),
            }
        )
    return scenes


def average_precision_score_np(y_true, score):
    y_true = y_true.astype(bool)
    if y_true.sum() == 0 or (~y_true).sum() == 0:
        return np.nan
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision[y]).sum() / max(y_true.sum(), 1))


def roc_auc_score_np(y_true, score):
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = ranks[y_true].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc)


def sim_score_np(y_true, score):
    y = y_true.astype(np.float64)
    s = np.clip(score.astype(np.float64), 0.0, 1.0)
    if y.sum() <= 0 or s.sum() <= 0:
        return np.nan
    y = y / y.sum()
    s = s / s.sum()
    return float(np.minimum(y, s).sum())


def metrics_from_scores(y_true, score, threshold):
    out = base.metrics_from_scores(y_true, score, threshold)
    out["auprc"] = average_precision_score_np(y_true, score)
    out["roc_auc"] = roc_auc_score_np(y_true, score)
    out["sim"] = sim_score_np(y_true, score)
    out["mae"] = float(np.mean(np.abs(score.astype(np.float32) - y_true.astype(np.float32))))
    return out


def per_scene_metrics(slices, y_all, score_all, threshold):
    rows = []
    for item, start, end in slices:
        row = {
            "scene_key": item["scene_key"],
            "category": item["category"],
            "scene_id": item["scene_id"],
            "instance_id": item["instance_id"],
            "source_split": item["source_split"],
        }
        row.update(metrics_from_scores(y_all[start:end], score_all[start:end], threshold))
        rows.append(row)
    return rows


METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "iou",
    "auprc",
    "sim",
    "mae",
    "roc_auc",
    "gt_handle_ratio",
    "pred_handle_ratio",
    "correct_percent",
    "fp_percent",
    "fn_percent",
]


def macro_scene_score(scene_rows, metric):
    return float(np.nanmean([row[metric] for row in scene_rows]))


def per_category_metrics(scene_rows):
    by_category = defaultdict(list)
    for row in scene_rows:
        by_category[row["category"]].append(row)
    out = []
    for category, rows in sorted(by_category.items()):
        merged = {
            "category": category,
            "num_scenes": len(rows),
            "num_gaussians": sum(r["num_gaussians"] for r in rows),
        }
        for name in METRIC_NAMES:
            merged[f"macro_scene_{name}"] = macro_scene_score(rows, name)
        out.append(merged)
    return out


def choose_threshold(y_val, score_val, val_slices):
    # Threshold search only needs binary overlap metrics. Ranking metrics such as
    # AUPRC/ROC-AUC are threshold-independent and are computed once for final rows.
    best = None
    curve = []
    for threshold in np.linspace(0.05, 0.95, 91):
        rows = []
        for item, start, end in val_slices:
            row = base.metrics_from_scores(y_val[start:end], score_val[start:end], float(threshold))
            rows.append(row)
        entry = {
            "threshold": float(threshold),
            "macro_scene_iou": float(np.nanmean([row["iou"] for row in rows])),
            "macro_scene_f1": float(np.nanmean([row["f1"] for row in rows])),
        }
        curve.append(entry)
        if best is None or (entry["macro_scene_iou"], entry["macro_scene_f1"]) > (
            best["macro_scene_iou"],
            best["macro_scene_f1"],
        ):
            best = entry
    return best, curve


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = dim**-0.5

    def forward(self, h, knn_idx):
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        neigh_k = k[knn_idx]
        neigh_v = v[knn_idx]
        attn = torch.softmax((q[:, None, :] * neigh_k).sum(dim=-1) * self.scale, dim=1)
        msg = (attn[..., None] * neigh_v).sum(dim=1)
        return self.norm(h + self.dropout(self.out(msg)))


class GraphAttentionGaussianNet(nn.Module):
    def __init__(self, in_dim, latent_dim=192, k_layers=(16, 32), dropout=0.1):
        super().__init__()
        hidden = 384 if in_dim >= 256 else 192
        self.k_layers = tuple(k_layers)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(latent_dim),
        )
        self.gat_layers = nn.ModuleList([GraphAttentionLayer(latent_dim, dropout=dropout) for _ in self.k_layers])
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, knn_idx):
        h = self.encoder(x)
        for k, layer in zip(self.k_layers, self.gat_layers):
            h = layer(h, knn_idx[:, :k])
        global_max = h.max(dim=0, keepdim=True).values
        global_mean = h.mean(dim=0, keepdim=True)
        global_feat = torch.cat([global_max, global_mean], dim=1).expand(len(h), -1)
        return self.head(torch.cat([h, global_feat], dim=1)).squeeze(-1)


def predict_scene_scores(model, scenes, device):
    model.eval()
    scores = []
    slices = []
    y_all = []
    offset = 0
    with torch.no_grad():
        for scene in scenes:
            x = torch.from_numpy(scene["x"]).to(device)
            knn = torch.from_numpy(scene["knn"]).long().to(device)
            logits = model(x, knn).detach().cpu().numpy()
            score = base.sigmoid_np(logits)
            scores.append(score.astype(np.float32, copy=False))
            y_all.append(scene["y"].astype(np.float32, copy=False))
            end = offset + len(score)
            slices.append((scene["item"], offset, end))
            offset = end
    return np.concatenate(y_all), np.concatenate(scores), slices


def train_epoch(model, scenes, loss_fn, opt, device, seed, grad_clip):
    model.train()
    order = list(range(len(scenes)))
    random.Random(seed).shuffle(order)
    losses = []
    for idx in order:
        scene = scenes[idx]
        x = torch.from_numpy(scene["x"]).to(device)
        y = torch.from_numpy(scene["y"]).to(device)
        knn = torch.from_numpy(scene["knn"]).long().to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, knn), y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_one(run_name, split, feature_variant, out_dir, args, device):
    model_name = f"graph_attention_{feature_variant}"
    model_dir = out_dir / model_name
    done_path = model_dir / "overall_metrics.json"
    if args.skip_existing and done_path.exists():
        print(f"[{run_name}/{model_name}] skip existing {done_path}", flush=True)
        with open(done_path, "r") as f:
            return json.load(f)

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{run_name}/{model_name}] fitting standardizer", flush=True)
    mean, std, train_n, train_pos = fit_standardizer(split["train"], feature_variant)
    pos_weight = (train_n - train_pos) / max(float(train_pos), 1.0)
    print(
        f"[{run_name}/{model_name}] loading scenes train={len(split['train'])} "
        f"val={len(split['val'])} test={len(split['test'])} train_gaussians={train_n} "
        f"pos_weight={pos_weight:.3f} max_k={args.max_k}",
        flush=True,
    )
    train_scenes = load_scene_list(split["train"], feature_variant, mean, std, args.max_k)
    val_scenes = load_scene_list(split["val"], feature_variant, mean, std, args.max_k)
    test_scenes = load_scene_list(split["test"], feature_variant, mean, std, args.max_k)

    in_dim = int(train_scenes[0]["x"].shape[1])
    model = GraphAttentionGaussianNet(
        in_dim,
        latent_dim=args.latent_dim,
        k_layers=tuple(args.k_layers),
        dropout=args.dropout,
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_scenes, loss_fn, opt, device, args.seed + epoch, args.grad_clip)
        y_val, val_score, val_slices = predict_scene_scores(model, val_scenes, device)
        th_best, _ = choose_threshold(y_val, val_score, val_slices)
        val_rows = per_scene_metrics(val_slices, y_val, val_score, th_best["threshold"])
        record = {
            "epoch": epoch,
            "loss": loss,
            "val_macro_scene_iou": macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": macro_scene_score(val_rows, "f1"),
            "val_macro_scene_auprc": macro_scene_score(val_rows, "auprc"),
            "val_threshold": th_best["threshold"],
        }
        history.append(record)
        score_tuple = (record["val_macro_scene_iou"], record["val_macro_scene_f1"])
        if best is None or score_tuple > (best["val_macro_scene_iou"], best["val_macro_scene_f1"]):
            best = dict(record)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[{run_name}/{model_name}] epoch {epoch:03d}/{args.epochs} "
                f"loss={record['loss']:.4f} val_iou={record['val_macro_scene_iou']:.3f} "
                f"val_f1={record['val_macro_scene_f1']:.3f} val_auprc={record['val_macro_scene_auprc']:.3f} "
                f"th={record['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{run_name}/{model_name}] early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    y_val, val_score, val_slices = predict_scene_scores(model, val_scenes, device)
    th_best, threshold_curve = choose_threshold(y_val, val_score, val_slices)
    threshold = th_best["threshold"]
    y_test, test_score, test_slices = predict_scene_scores(model, test_scenes, device)
    test_rows = per_scene_metrics(test_slices, y_test, test_score, threshold)
    cat_rows = per_category_metrics(test_rows)
    micro = metrics_from_scores(y_test, test_score, threshold)
    macro = {key: macro_scene_score(test_rows, key) for key in METRIC_NAMES}
    overall = {
        "run_name": run_name,
        "model": model_name,
        "architecture": "per-Gaussian encoder; graph attention k=16 then k=32; global max+mean pooling; local+global classifier",
        "feature_variant": feature_variant,
        "input_dim": in_dim,
        "latent_dim": args.latent_dim,
        "k_layers": list(args.k_layers),
        "train_scenes": len(split["train"]),
        "val_scenes": len(split["val"]),
        "test_scenes": len(split["test"]),
        "train_gaussians": int(sum(len(s["y"]) for s in train_scenes)),
        "val_gaussians": int(sum(len(s["y"]) for s in val_scenes)),
        "test_gaussians": int(len(y_test)),
        "train_handle_ratio": float(sum(float(s["y"].sum()) for s in train_scenes) / max(sum(len(s["y"]) for s in train_scenes), 1)),
        "val_handle_ratio": float(sum(float(s["y"].sum()) for s in val_scenes) / max(sum(len(s["y"]) for s in val_scenes), 1)),
        "test_handle_ratio": float(y_test.mean()),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(threshold),
        "selection_metric": "validation macro scene IoU, tie-broken by validation macro scene F1",
        "loss": "weighted BCEWithLogitsLoss",
        "positive_label": "handle_labels_thr0_25",
        "micro": micro,
        "macro_scene_average": macro,
        "best_scene_iou": max(test_rows, key=lambda r: r["iou"])["scene_key"],
        "worst_scene_iou": min(test_rows, key=lambda r: r["iou"])["scene_key"],
    }

    save_csv(model_dir / "history.csv", history)
    save_csv(model_dir / "threshold_curve.csv", threshold_curve)
    save_csv(model_dir / "per_scene_metrics.csv", test_rows)
    save_csv(model_dir / "per_category_metrics.csv", cat_rows)
    with open(model_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "feature_variant": feature_variant,
            "input_dim": in_dim,
            "latent_dim": args.latent_dim,
            "k_layers": list(args.k_layers),
            "threshold": float(threshold),
            "overall": overall,
        },
        model_dir / "model.pt",
    )
    offsets = np.asarray([[start, end] for _, start, end in test_slices], dtype=np.int64)
    scene_keys = np.asarray([item["scene_key"] for item, _, _ in test_slices])
    np.savez_compressed(
        model_dir / "test_predictions.npz",
        scene_keys=scene_keys,
        offsets=offsets,
        scores=test_score.astype(np.float32),
        labels=y_test.astype(np.uint8),
    )
    print(
        f"[{run_name}/{model_name}] TEST macro IoU={macro['iou']:.3f} "
        f"macro F1={macro['f1']:.3f} AUPRC={macro['auprc']:.3f} "
        f"threshold={threshold:.2f}",
        flush=True,
    )
    return overall


def load_split_manifest(path):
    with open(path, "r") as f:
        return json.load(f)["splits"]


def load_seen_split():
    return load_split_manifest(BASELINE_ROOT / "01_seen_instance_17cat" / "split_manifest.json")


def load_loco_split(category):
    return load_split_manifest(BASELINE_ROOT / "02_leave_one_category_out_17cat" / category / "split_manifest.json")


def write_flat_summary(path, results):
    rows = []
    for result in results:
        macro = result["macro_scene_average"]
        micro = result["micro"]
        rows.append(
            {
                "run_name": result["run_name"],
                "model": result["model"],
                "feature_variant": result["feature_variant"],
                "input_dim": result["input_dim"],
                "latent_dim": result["latent_dim"],
                "k_layers": json.dumps(result["k_layers"]),
                "train_scenes": result["train_scenes"],
                "val_scenes": result["val_scenes"],
                "test_scenes": result["test_scenes"],
                "selected_epoch": result["selected_epoch"],
                "selected_threshold": result["selected_threshold"],
                "macro_iou": macro["iou"],
                "macro_f1": macro["f1"],
                "macro_precision": macro["precision"],
                "macro_recall": macro["recall"],
                "macro_auprc": macro["auprc"],
                "macro_sim": macro["sim"],
                "macro_mae": macro["mae"],
                "macro_roc_auc": macro["roc_auc"],
                "macro_gt_handle_ratio": macro["gt_handle_ratio"],
                "macro_pred_handle_ratio": macro["pred_handle_ratio"],
                "micro_iou": micro["iou"],
                "micro_f1": micro["f1"],
                "micro_auprc": micro["auprc"],
                "micro_sim": micro["sim"],
                "micro_mae": micro["mae"],
                "micro_roc_auc": micro["roc_auc"],
                "micro_correct_percent": micro["correct_percent"],
                "worst_scene_iou": result["worst_scene_iou"],
                "best_scene_iou": result["best_scene_iou"],
            }
        )
    save_csv(path, rows)


def aggregate_results(out_root):
    results = []
    for path in sorted(out_root.glob("**/overall_metrics.json")):
        with open(path, "r") as f:
            results.append(json.load(f))
    write_flat_summary(out_root / "all_finished_models_summary.csv", results)
    with open(out_root / "all_finished_models_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[aggregate] collected {len(results)} finished model results", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--mode", choices=["seen", "loco", "aggregate"], required=True)
    parser.add_argument("--feature_variants", nargs="+", default=FEATURE_VARIANTS, choices=FEATURE_VARIANTS)
    parser.add_argument("--heldout_categories", nargs="+", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--k_layers", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--max_k", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    args.max_k = max(args.max_k, max(args.k_layers))
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "aggregate":
        aggregate_results(args.out_root)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    with open(BASELINE_ROOT / "feature_roots.json", "r") as f:
        feature_meta = json.load(f)
    categories = sorted(feature_meta["categories"])
    with open(args.out_root / "source_baseline_root.json", "w") as f:
        json.dump(
            {
                "baseline_root": str(BASELINE_ROOT),
                "feature_roots": feature_meta["feature_roots"],
                "categories": categories,
                "note": "Graph Attention runs reuse the exact 17-category MLP split manifests.",
                "k_layers": args.k_layers,
                "feature_variants": args.feature_variants,
            },
            f,
            indent=2,
        )

    results = []
    if args.mode == "seen":
        split = load_seen_split()
        out_dir = args.out_root / "01_seen_instance_17cat"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[seen] split scenes: train={len(split['train'])} "
            f"val={len(split['val'])} test={len(split['test'])}",
            flush=True,
        )
        for feature_variant in args.feature_variants:
            results.append(train_one("01_seen_instance_17cat", split, feature_variant, out_dir, args, device))
            write_flat_summary(out_dir / "seen_instance_summary.csv", results)
    else:
        heldout = args.heldout_categories or categories
        heldout = [c for c in heldout if c in categories]
        heldout = [c for i, c in enumerate(sorted(heldout)) if i % args.num_shards == args.shard]
        print(f"[loco shard {args.shard}/{args.num_shards}] heldout={heldout}", flush=True)
        for category in heldout:
            split = load_loco_split(category)
            out_dir = args.out_root / "02_leave_one_category_out_17cat" / category
            out_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{category}] split scenes: train={len(split['train'])} "
                f"val={len(split['val'])} test={len(split['test'])}",
                flush=True,
            )
            category_results = []
            for feature_variant in args.feature_variants:
                result = train_one(category, split, feature_variant, out_dir, args, device)
                results.append(result)
                category_results.append(result)
                write_flat_summary(out_dir / "category_summary.csv", category_results)
                write_flat_summary(args.out_root / f"loco_summary_shard_{args.shard}_of_{args.num_shards}.csv", results)
        with open(args.out_root / f"loco_summary_shard_{args.shard}_of_{args.num_shards}.json", "w") as f:
            json.dump(results, f, indent=2)

    aggregate_results(args.out_root)


if __name__ == "__main__":
    main()
