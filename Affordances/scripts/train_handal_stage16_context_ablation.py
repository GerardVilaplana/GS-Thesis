#!/usr/bin/env python3
import argparse
import csv
import json
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

import train_handal_17cat_graph_attention as g15  # noqa: E402
import train_handal_17cat_mlp_baselines as base  # noqa: E402


BASELINE_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "08_17category_mlp_baselines_v1"
STAGE15_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "15_graph_attention_gaussian_v1"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "16_context_denoise_ablation_v1"

CRITICAL_LOCO = ["measuring_cups", "mugs", "power_drills", "screwdrivers", "utensils", "whisks"]
FEATURE_VARIANT = "geometry_color_scene_norm"


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_split_manifest(path):
    with open(path, "r") as f:
        return json.load(f)["splits"]


def load_seen_split():
    return load_split_manifest(BASELINE_ROOT / "01_seen_instance_17cat" / "split_manifest.json")


def load_loco_split(category):
    return load_split_manifest(BASELINE_ROOT / "02_leave_one_category_out_17cat" / category / "split_manifest.json")


def union_find_components(n, edges):
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    unique, counts = np.unique(roots, return_counts=True)
    return roots == unique[np.argmax(counts)]


def largest_component_mask(xyz, args):
    n = len(xyz)
    if n < args.component_k + 2:
        return np.ones(n, dtype=bool)
    tree = cKDTree(xyz.astype(np.float32, copy=False))
    d, idx = tree.query(xyz, k=min(args.component_k + 1, n), workers=-1)
    d = np.asarray(d)
    idx = np.asarray(idx)
    kth = d[:, -1]
    radius = float(np.quantile(kth[np.isfinite(kth)], args.component_radius_quantile) * args.component_radius_factor)
    if not np.isfinite(radius) or radius <= 0:
        return np.ones(n, dtype=bool)
    edges = []
    for i in range(n):
        neigh = idx[i, 1:] if idx.ndim == 2 else []
        dist = d[i, 1:] if d.ndim == 2 else []
        for j, dij in zip(neigh, dist):
            if dij <= radius:
                edges.append((i, int(j)))
    if not edges:
        return np.ones(n, dtype=bool)
    keep = union_find_components(n, edges)
    if keep.mean() < args.min_keep_ratio:
        return np.ones(n, dtype=bool)
    return keep


def density_trim_mask(xyz, args):
    n = len(xyz)
    if n < args.density_k + 2:
        return np.ones(n, dtype=bool)
    tree = cKDTree(xyz.astype(np.float32, copy=False))
    d, _ = tree.query(xyz, k=min(args.density_k + 1, n), workers=-1)
    d = np.asarray(d)
    kth = d[:, -1]
    center = np.median(xyz, axis=0, keepdims=True)
    radius = np.linalg.norm(xyz - center, axis=1)
    keep = (kth <= np.quantile(kth, args.density_quantile)) & (
        radius <= np.quantile(radius, args.radius_quantile)
    )
    if keep.mean() < args.min_keep_ratio:
        return np.ones(n, dtype=bool)
    return keep


def preprocess_mask(xyz, args):
    if args.preprocess == "none":
        return np.ones(len(xyz), dtype=bool)
    if args.preprocess == "largest_component":
        return largest_component_mask(xyz, args)
    if args.preprocess == "density_trim":
        return density_trim_mask(xyz, args)
    raise ValueError(f"Unknown preprocess: {args.preprocess}")


def scene_zscore(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    return ((x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), eps)).astype(np.float32)


def scene_minmax_unit(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    return ((x - lo) / np.maximum(hi - lo, eps)).astype(np.float32)


def load_feature_arrays(item, args):
    with np.load(item["npz"], allow_pickle=False) as z:
        xyz_raw = z["xyz"].astype(np.float32)
        keep = preprocess_mask(xyz_raw, args)
        xyz = xyz_raw[keep]
        scale = z["scale"].astype(np.float32)[keep]
        rotation = z["rotation"].astype(np.float32)[keep]
        opacity = np.asarray(z["opacity"], dtype=np.float32).reshape(-1, 1)[keep]
        color = z["color"].astype(np.float32)[keep]
        y = z["handle_labels_thr0_25"].astype(np.float32)[keep]
        xyz_norm = base.normalize_xyz(xyz)
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
    stats = {
        "raw_gaussians": int(len(xyz_raw)),
        "kept_gaussians": int(len(xyz)),
        "kept_ratio": float(len(xyz) / max(len(xyz_raw), 1)),
    }
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False), xyz_norm.astype(np.float32), stats


def build_knn(xyz_norm, max_k):
    return g15.build_knn(xyz_norm, max_k)


def fit_standardizer(items, args):
    total = 0
    sum_x = None
    sumsq_x = None
    pos = 0
    kept = 0
    raw = 0
    for item in items:
        x, y, _, stats = load_feature_arrays(item, args)
        x64 = x.astype(np.float64, copy=False)
        sum_x = x64.sum(axis=0) if sum_x is None else sum_x + x64.sum(axis=0)
        sumsq_x = np.square(x64).sum(axis=0) if sumsq_x is None else sumsq_x + np.square(x64).sum(axis=0)
        total += len(x)
        pos += int(y.sum())
        kept += stats["kept_gaussians"]
        raw += stats["raw_gaussians"]
    mean64 = sum_x / max(total, 1)
    var64 = (sumsq_x / max(total, 1)) - np.square(mean64)
    mean = mean64.astype(np.float32)[None, :]
    std = np.sqrt(np.maximum(var64, 1e-12)).astype(np.float32)[None, :]
    return mean, np.maximum(std, 1e-6), total, pos, float(kept / max(raw, 1))


def load_scene_list(items, mean, std, args):
    scenes = []
    raw = 0
    kept = 0
    for item in items:
        x, y, xyz_norm, stats = load_feature_arrays(item, args)
        scenes.append(
            {
                "item": item,
                "x": ((x - mean) / std).astype(np.float32, copy=False),
                "y": y.astype(np.float32, copy=False),
                "xyz_norm": xyz_norm.astype(np.float32, copy=False),
                "knn": build_knn(xyz_norm, args.max_k),
                "stats": stats,
            }
        )
        raw += stats["raw_gaussians"]
        kept += stats["kept_gaussians"]
    return scenes, float(kept / max(raw, 1))


class EdgeGraphAttentionLayer(nn.Module):
    def __init__(self, dim, edge_dim=4, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.edge_score = nn.Sequential(nn.Linear(edge_dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.edge_msg = nn.Sequential(nn.Linear(edge_dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = dim**-0.5

    def forward(self, h, knn_idx, xyz):
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        neigh_xyz = xyz[knn_idx]
        rel = neigh_xyz - xyz[:, None, :]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        edge = torch.cat([rel, dist], dim=-1)
        neigh_k = k[knn_idx] + self.edge_score(edge)
        neigh_v = v[knn_idx] + self.edge_msg(edge)
        attn = torch.softmax((q[:, None, :] * neigh_k).sum(dim=-1) * self.scale, dim=1)
        msg = (attn[..., None] * neigh_v).sum(dim=1)
        return self.norm(h + self.dropout(self.out(msg)))


class EdgeGraphAttentionGaussianNet(nn.Module):
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
        self.gat_layers = nn.ModuleList([EdgeGraphAttentionLayer(latent_dim, dropout=dropout) for _ in self.k_layers])
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, knn_idx, xyz):
        h = self.encoder(x)
        for k, layer in zip(self.k_layers, self.gat_layers):
            h = layer(h, knn_idx[:, :k], xyz)
        global_max = h.max(dim=0, keepdim=True).values
        global_mean = h.mean(dim=0, keepdim=True)
        global_feat = torch.cat([global_max, global_mean], dim=1).expand(len(h), -1)
        return self.head(torch.cat([h, global_feat], dim=1)).squeeze(-1)


class PointNetPooling(nn.Module):
    def __init__(self, in_dim, latent_dim=192, pooling="max_mean", dropout=0.1):
        super().__init__()
        hidden = 384 if in_dim >= 256 else 192
        self.pooling = pooling
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(latent_dim),
        )
        pool_mult = 2 if pooling == "max_mean" else 1
        self.head = nn.Sequential(
            nn.Linear(latent_dim * (1 + pool_mult), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.encoder(x)
        if self.pooling == "max":
            global_feat = h.max(dim=0, keepdim=True).values
        elif self.pooling == "mean":
            global_feat = h.mean(dim=0, keepdim=True)
        elif self.pooling == "max_mean":
            global_feat = torch.cat([h.max(dim=0, keepdim=True).values, h.mean(dim=0, keepdim=True)], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.head(torch.cat([h, global_feat.expand(len(h), -1)], dim=1)).squeeze(-1)


def build_model(in_dim, args):
    if args.model_type == "graph":
        return g15.GraphAttentionGaussianNet(
            in_dim, latent_dim=args.latent_dim, k_layers=tuple(args.k_layers), dropout=args.dropout
        )
    if args.model_type == "edge_graph":
        return EdgeGraphAttentionGaussianNet(
            in_dim, latent_dim=args.latent_dim, k_layers=tuple(args.k_layers), dropout=args.dropout
        )
    if args.model_type == "pointnet":
        return PointNetPooling(in_dim, latent_dim=args.latent_dim, pooling=args.pooling, dropout=args.dropout)
    raise ValueError(f"Unknown model_type: {args.model_type}")


def model_name(args):
    bits = [args.model_type, FEATURE_VARIANT, f"prep-{args.preprocess}"]
    if args.model_type in {"graph", "edge_graph"}:
        bits.append("k-" + "-".join(str(k) for k in args.k_layers))
    if args.model_type == "pointnet":
        bits.append(f"pool-{args.pooling}")
    return "__".join(bits)


def forward_scene(model, scene, args, device):
    x = torch.from_numpy(scene["x"]).to(device)
    if args.model_type == "pointnet":
        return model(x)
    knn = torch.from_numpy(scene["knn"]).long().to(device)
    if args.model_type == "edge_graph":
        xyz = torch.from_numpy(scene["xyz_norm"]).to(device)
        return model(x, knn, xyz)
    return model(x, knn)


def predict_scene_scores(model, scenes, args, device):
    model.eval()
    scores = []
    slices = []
    y_all = []
    offset = 0
    with torch.no_grad():
        for scene in scenes:
            logits = forward_scene(model, scene, args, device).detach().cpu().numpy()
            score = base.sigmoid_np(logits)
            scores.append(score.astype(np.float32, copy=False))
            y_all.append(scene["y"].astype(np.float32, copy=False))
            end = offset + len(score)
            slices.append((scene["item"], offset, end))
            offset = end
    return np.concatenate(y_all), np.concatenate(scores), slices


def train_epoch(model, scenes, loss_fn, opt, args, device, seed):
    model.train()
    order = list(range(len(scenes)))
    random.Random(seed).shuffle(order)
    losses = []
    for idx in order:
        scene = scenes[idx]
        y = torch.from_numpy(scene["y"]).to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(forward_scene(model, scene, args, device), y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_one(run_name, split, out_dir, args, device):
    name = model_name(args)
    model_dir = out_dir / name
    done_path = model_dir / "overall_metrics.json"
    if args.skip_existing and done_path.exists():
        print(f"[{run_name}/{name}] skip existing {done_path}", flush=True)
        with open(done_path, "r") as f:
            return json.load(f)

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{run_name}/{name}] fitting standardizer", flush=True)
    mean, std, train_n, train_pos, train_keep_ratio = fit_standardizer(split["train"], args)
    pos_weight = (train_n - train_pos) / max(float(train_pos), 1.0)
    print(
        f"[{run_name}/{name}] loading scenes train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])} train_gaussians={train_n} pos_weight={pos_weight:.3f} "
        f"train_keep_ratio={train_keep_ratio:.3f}",
        flush=True,
    )
    train_scenes, train_keep_ratio = load_scene_list(split["train"], mean, std, args)
    val_scenes, val_keep_ratio = load_scene_list(split["val"], mean, std, args)
    test_scenes, test_keep_ratio = load_scene_list(split["test"], mean, std, args)

    in_dim = int(train_scenes[0]["x"].shape[1])
    model = build_model(in_dim, args).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_scenes, loss_fn, opt, args, device, args.seed + epoch)
        y_val, val_score, val_slices = predict_scene_scores(model, val_scenes, args, device)
        th_best, _ = g15.choose_threshold(y_val, val_score, val_slices)
        val_rows = g15.per_scene_metrics(val_slices, y_val, val_score, th_best["threshold"])
        record = {
            "epoch": epoch,
            "loss": loss,
            "val_macro_scene_iou": g15.macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": g15.macro_scene_score(val_rows, "f1"),
            "val_macro_scene_auprc": g15.macro_scene_score(val_rows, "auprc"),
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
                f"[{run_name}/{name}] epoch {epoch:03d}/{args.epochs} loss={loss:.4f} "
                f"val_iou={record['val_macro_scene_iou']:.3f} val_f1={record['val_macro_scene_f1']:.3f} "
                f"val_auprc={record['val_macro_scene_auprc']:.3f} th={record['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{run_name}/{name}] early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    y_val, val_score, val_slices = predict_scene_scores(model, val_scenes, args, device)
    th_best, threshold_curve = g15.choose_threshold(y_val, val_score, val_slices)
    threshold = th_best["threshold"]
    y_test, test_score, test_slices = predict_scene_scores(model, test_scenes, args, device)
    test_rows = g15.per_scene_metrics(test_slices, y_test, test_score, threshold)
    cat_rows = g15.per_category_metrics(test_rows)
    micro = g15.metrics_from_scores(y_test, test_score, threshold)
    macro = {key: g15.macro_scene_score(test_rows, key) for key in g15.METRIC_NAMES}
    overall = {
        "stage": "16_context_denoise_ablation_v1",
        "run_name": run_name,
        "model": name,
        "model_type": args.model_type,
        "feature_variant": FEATURE_VARIANT,
        "preprocess": args.preprocess,
        "pooling": args.pooling if args.model_type == "pointnet" else "",
        "input_dim": in_dim,
        "latent_dim": args.latent_dim,
        "k_layers": list(args.k_layers) if args.model_type in {"graph", "edge_graph"} else [],
        "train_scenes": len(split["train"]),
        "val_scenes": len(split["val"]),
        "test_scenes": len(split["test"]),
        "train_gaussians": int(sum(len(s["y"]) for s in train_scenes)),
        "val_gaussians": int(sum(len(s["y"]) for s in val_scenes)),
        "test_gaussians": int(len(y_test)),
        "train_keep_ratio": train_keep_ratio,
        "val_keep_ratio": val_keep_ratio,
        "test_keep_ratio": test_keep_ratio,
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
            "feature_variant": FEATURE_VARIANT,
            "input_dim": in_dim,
            "latent_dim": args.latent_dim,
            "k_layers": list(args.k_layers),
            "threshold": float(threshold),
            "overall": overall,
        },
        model_dir / "model.pt",
    )
    np.savez_compressed(
        model_dir / "test_predictions.npz",
        scene_keys=np.asarray([item["scene_key"] for item, _, _ in test_slices]),
        offsets=np.asarray([[start, end] for _, start, end in test_slices], dtype=np.int64),
        scores=test_score.astype(np.float32),
        labels=y_test.astype(np.uint8),
    )
    print(
        f"[{run_name}/{name}] TEST macro IoU={macro['iou']:.3f} macro F1={macro['f1']:.3f} "
        f"AUPRC={macro['auprc']:.3f} threshold={threshold:.2f}",
        flush=True,
    )
    return overall


def flat_row(result):
    macro = result["macro_scene_average"]
    micro = result["micro"]
    return {
        "run_name": result["run_name"],
        "model": result["model"],
        "model_type": result["model_type"],
        "feature_variant": result["feature_variant"],
        "preprocess": result["preprocess"],
        "pooling": result["pooling"],
        "k_layers": json.dumps(result["k_layers"]),
        "input_dim": result["input_dim"],
        "latent_dim": result["latent_dim"],
        "train_scenes": result["train_scenes"],
        "val_scenes": result["val_scenes"],
        "test_scenes": result["test_scenes"],
        "train_gaussians": result["train_gaussians"],
        "test_gaussians": result["test_gaussians"],
        "train_keep_ratio": result["train_keep_ratio"],
        "val_keep_ratio": result["val_keep_ratio"],
        "test_keep_ratio": result["test_keep_ratio"],
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


def write_flat_summary(path, results):
    save_csv(path, [flat_row(r) for r in results])


def aggregate_results(out_root):
    results = []
    for path in sorted(out_root.glob("**/overall_metrics.json")):
        with open(path, "r") as f:
            results.append(json.load(f))
    write_flat_summary(out_root / "all_finished_models_summary.csv", results)
    with open(out_root / "all_finished_models_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[aggregate] collected {len(results)} finished model results", flush=True)


def build_run_specs(args):
    specs = []
    if args.mode in {"seen", "both"}:
        specs.append(("01_seen_instance_17cat", load_seen_split(), args.out_root / "01_seen_instance_17cat"))
    if args.mode in {"loco", "both"}:
        heldout = args.heldout_categories or CRITICAL_LOCO
        heldout = [c for i, c in enumerate(sorted(heldout)) if i % args.num_shards == args.shard]
        for category in heldout:
            specs.append((category, load_loco_split(category), args.out_root / "02_leave_one_category_out_critical" / category))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--mode", choices=["seen", "loco", "both", "aggregate"], required=True)
    parser.add_argument("--model_type", choices=["graph", "edge_graph", "pointnet"], default="graph")
    parser.add_argument("--preprocess", choices=["none", "largest_component", "density_trim"], default="none")
    parser.add_argument("--pooling", choices=["max", "mean", "max_mean"], default="max_mean")
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
    parser.add_argument("--component_k", type=int, default=8)
    parser.add_argument("--component_radius_quantile", type=float, default=0.80)
    parser.add_argument("--component_radius_factor", type=float, default=2.5)
    parser.add_argument("--density_k", type=int, default=8)
    parser.add_argument("--density_quantile", type=float, default=0.95)
    parser.add_argument("--radius_quantile", type=float, default=0.98)
    parser.add_argument("--min_keep_ratio", type=float, default=0.55)
    args = parser.parse_args()

    args.max_k = max(args.max_k, max(args.k_layers) if args.k_layers else 1)
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
    with open(args.out_root / "stage16_config_note.json", "w") as f:
        json.dump(
            {
                "baseline_root": str(BASELINE_ROOT),
                "stage15_root": str(STAGE15_ROOT),
                "feature_roots": feature_meta["feature_roots"],
                "categories": feature_meta["categories"],
                "feature_variant": FEATURE_VARIANT,
                "critical_loco": CRITICAL_LOCO,
                "note": "Stage 16 reuses the exact 17-category MLP split manifests; synthetic data is not used.",
            },
            f,
            indent=2,
        )

    results = []
    specs = build_run_specs(args)
    print(
        f"[stage16] model={model_name(args)} mode={args.mode} runs={len(specs)} device={device}",
        flush=True,
    )
    for run_name, split, out_dir in specs:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{run_name}] split scenes: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}",
            flush=True,
        )
        result = train_one(run_name, split, out_dir, args, device)
        results.append(result)
        write_flat_summary(out_dir / "summary.csv", [result])
        write_flat_summary(args.out_root / f"partial_{args.model_type}_{args.preprocess}_{args.shard}_of_{args.num_shards}.csv", results)

    aggregate_results(args.out_root)


if __name__ == "__main__":
    main()
