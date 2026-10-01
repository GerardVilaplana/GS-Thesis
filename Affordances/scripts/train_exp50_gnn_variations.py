#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

BASE_ROOT = Path('/home/gvilaplana/GS-Thesis/Affordances')
SCRIPT_DIR = BASE_ROOT / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))

import train_handal_exp44_pointnet_mean_feature_sweep as e44  # noqa: E402
import train_handal_17cat_graph_attention as g15  # noqa: E402

NPZ_ROOT = BASE_ROOT / 'data' / 'handal_exp40_15k_gt_features_v1' / 'npz' / 'B_center_ellipsoid20_strict75'
OUT_ROOT = BASE_ROOT / 'outputs' / '03_handle_generalization' / '50_exp46_gnn_variations_v1'
DINO_BASE = BASE_ROOT / 'outputs' / '03_handle_generalization' / '46_exp40_dino_assignment_all_features_v1'

CONFIGS = {
    'object_crop_best': {
        'dino_method': 'object_crop_center_avg',
        'feature_variant': 'dino_geometry_color_quality',
        'dino_root': DINO_BASE / 'object_crop_center_avg_patch896x672_views48',
    },
    'full_center_best': {
        'dino_method': 'full_center_avg',
        'feature_variant': 'dino_geometry_color',
        'dino_root': DINO_BASE / 'full_center_avg_patch896x672_views48',
    },
}


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_variant_arrays(item: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e44.ACTIVE_FEATURE_VARIANT = cfg['feature_variant']
    e44.ACTIVE_DINO_ROOT = Path(cfg['dino_root'])
    x, y = e44.load_feature_arrays_variant(item)
    with np.load(item['npz'], allow_pickle=False) as z:
        xyz_norm = g15.base.normalize_xyz(z['xyz'].astype(np.float32))
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False), xyz_norm.astype(np.float32, copy=False)


def fit_standardizer(items: list[dict], cfg: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    total = 0
    pos = 0
    sum_x = None
    sumsq_x = None
    for item in items:
        x, y, _ = load_variant_arrays(item, cfg)
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


def load_graph_scenes(items: list[dict], cfg: dict, mean: np.ndarray, std: np.ndarray, max_k: int) -> list[dict]:
    scenes = []
    for item in items:
        x, y, xyz_norm = load_variant_arrays(item, cfg)
        scenes.append({
            'item': item,
            'x': ((x - mean) / std).astype(np.float32, copy=False),
            'y': y.astype(np.float32, copy=False),
            'xyz': xyz_norm.astype(np.float32, copy=False),
            'knn': g15.build_knn(xyz_norm, max_k),
        })
    return scenes


class GraphEmbeddingNet(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 192, k_layers=(16, 32), dropout: float = 0.1):
        super().__init__()
        hidden = 384 if in_dim >= 256 else 192
        self.k_layers = tuple(k_layers)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim), nn.ReLU(inplace=True), nn.LayerNorm(latent_dim),
        )
        self.gat_layers = nn.ModuleList([g15.GraphAttentionLayer(latent_dim, dropout=dropout) for _ in self.k_layers])
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 3, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def embed(self, x: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for k, layer in zip(self.k_layers, self.gat_layers):
            h = layer(h, knn_idx[:, :k])
        return h

    def forward(self, x: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        h = self.embed(x, knn_idx)
        global_max = h.max(dim=0, keepdim=True).values
        global_mean = h.mean(dim=0, keepdim=True)
        global_feat = torch.cat([global_max, global_mean], dim=1).expand(len(h), -1)
        return self.head(torch.cat([h, global_feat], dim=1)).squeeze(-1)


class EdgeAwareGraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.edge = nn.Sequential(nn.Linear(4, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = dim**-0.5

    def forward(self, h, knn_idx, xyz):
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        rel = xyz[knn_idx] - xyz[:, None, :]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        edge_feat = self.edge(torch.cat([rel, dist], dim=-1))
        neigh_k = k[knn_idx] + edge_feat
        neigh_v = v[knn_idx] + edge_feat
        attn = torch.softmax((q[:, None, :] * neigh_k).sum(dim=-1) * self.scale, dim=1)
        msg = (attn[..., None] * neigh_v).sum(dim=1)
        return self.norm(h + self.dropout(self.out(msg)))


class EdgeAwareGraphEmbeddingNet(nn.Module):
    uses_xyz = True

    def __init__(self, in_dim: int, latent_dim: int = 192, k_layers=(16, 32), dropout: float = 0.1):
        super().__init__()
        hidden = 384 if in_dim >= 256 else 192
        self.k_layers = tuple(k_layers)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim), nn.ReLU(inplace=True), nn.LayerNorm(latent_dim),
        )
        self.gat_layers = nn.ModuleList([EdgeAwareGraphAttentionLayer(latent_dim, dropout=dropout) for _ in self.k_layers])
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 3, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def embed(self, x: torch.Tensor, knn_idx: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for k, layer in zip(self.k_layers, self.gat_layers):
            h = layer(h, knn_idx[:, :k], xyz)
        return h

    def forward(self, x: torch.Tensor, knn_idx: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        h = self.embed(x, knn_idx, xyz)
        global_max = h.max(dim=0, keepdim=True).values
        global_mean = h.mean(dim=0, keepdim=True)
        global_feat = torch.cat([global_max, global_mean], dim=1).expand(len(h), -1)
        return self.head(torch.cat([h, global_feat], dim=1)).squeeze(-1)


def make_gnn_model(in_dim: int, args) -> nn.Module:
    if args.gnn_variant == 'edge_aware_xyz_delta':
        return EdgeAwareGraphEmbeddingNet(in_dim, args.gnn_latent_dim, tuple(args.k_layers), args.dropout)
    return GraphEmbeddingNet(in_dim, args.gnn_latent_dim, tuple(args.k_layers), args.dropout)


def gnn_logits(model, scene, device):
    x = torch.from_numpy(scene['x']).to(device)
    knn = torch.from_numpy(scene['knn']).long().to(device)
    if getattr(model, 'uses_xyz', False):
        xyz = torch.from_numpy(scene['xyz']).to(device)
        return model(x, knn, xyz)
    return model(x, knn)


def gnn_embedding(model, scene, device):
    x = torch.from_numpy(scene['x']).to(device)
    knn = torch.from_numpy(scene['knn']).long().to(device)
    if getattr(model, 'uses_xyz', False):
        xyz = torch.from_numpy(scene['xyz']).to(device)
        return model.embed(x, knn, xyz)
    return model.embed(x, knn)


class PointNetMean(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, latent_dim), nn.ReLU(inplace=True), nn.LayerNorm(latent_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.encoder(x)
        global_mean = local.mean(dim=0, keepdim=True).expand(len(local), -1)
        return self.head(torch.cat([local, global_mean], dim=1)).squeeze(-1)


def train_gnn_epoch(model, scenes, loss_fn, opt, device, seed, grad_clip):
    model.train()
    order = list(range(len(scenes)))
    random.Random(seed).shuffle(order)
    losses = []
    for idx in order:
        scene = scenes[idx]
        y = torch.from_numpy(scene['y']).to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(gnn_logits(model, scene, device), y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def predict_gnn(model, scenes, device):
    model.eval()
    labels, scores, slices = [], [], []
    offset = 0
    with torch.no_grad():
        for scene in scenes:
            logits = gnn_logits(model, scene, device).detach().cpu().numpy()
            score = g15.base.sigmoid_np(logits).astype(np.float32)
            y = scene['y'].astype(np.float32, copy=False)
            labels.append(y); scores.append(score)
            slices.append((scene['item'], offset, offset + len(y)))
            offset += len(y)
    return np.concatenate(labels), np.concatenate(scores), slices


def extract_embedding_scenes(model, scenes, device) -> list[dict]:
    model.eval()
    out = []
    with torch.no_grad():
        for scene in scenes:
            emb = gnn_embedding(model, scene, device).detach().cpu().numpy().astype(np.float32)
            out.append({'item': scene['item'], 'x': emb, 'y': scene['y'].astype(np.float32, copy=False)})
    return out


def fit_scene_standardizer(scenes: list[dict]) -> tuple[np.ndarray, np.ndarray, int, int]:
    total = 0; pos = 0; sum_x = None; sumsq_x = None
    for scene in scenes:
        x = scene['x'].astype(np.float64, copy=False)
        sum_x = x.sum(axis=0) if sum_x is None else sum_x + x.sum(axis=0)
        sumsq_x = np.square(x).sum(axis=0) if sumsq_x is None else sumsq_x + np.square(x).sum(axis=0)
        total += len(x); pos += int(scene['y'].sum())
    mean64 = sum_x / max(total, 1)
    var64 = (sumsq_x / max(total, 1)) - np.square(mean64)
    mean = mean64.astype(np.float32)[None, :]
    std = np.sqrt(np.maximum(var64, 1e-12)).astype(np.float32)[None, :]
    return mean, np.maximum(std, 1e-6), total, pos


def apply_scene_standardizer(scenes: list[dict], mean: np.ndarray, std: np.ndarray) -> list[dict]:
    return [{'item': s['item'], 'x': ((s['x'] - mean) / std).astype(np.float32), 'y': s['y']} for s in scenes]


def train_pointnet_epoch(model, scenes, loss_fn, opt, device, seed, grad_clip):
    model.train()
    order = list(range(len(scenes)))
    random.Random(seed).shuffle(order)
    losses = []
    for idx in order:
        scene = scenes[idx]
        x = torch.from_numpy(scene['x']).to(device)
        y = torch.from_numpy(scene['y']).to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def predict_pointnet(model, scenes, device):
    model.eval()
    labels, scores, slices = [], [], []
    offset = 0
    with torch.no_grad():
        for scene in scenes:
            x = torch.from_numpy(scene['x']).to(device)
            logits = model(x).detach().cpu().numpy()
            score = g15.base.sigmoid_np(logits).astype(np.float32)
            y = scene['y'].astype(np.float32, copy=False)
            labels.append(y); scores.append(score)
            slices.append((scene['item'], offset, offset + len(y)))
            offset += len(y)
    return np.concatenate(labels), np.concatenate(scores), slices


def macro_rows(y, score, slices, threshold):
    rows = g15.per_scene_metrics(slices, y, score, threshold)
    micro = g15.metrics_from_scores(y, score, threshold)
    macro = {key: g15.macro_scene_score(rows, key) for key in g15.METRIC_NAMES}
    return rows, micro, macro


def train_one_pipeline(run_name: str, split: dict, config_name: str, cfg: dict, out_dir: Path, args, device):
    model_dir = out_dir / config_name / 'gnn_embeddings_pointnet_mean'
    done_path = model_dir / 'overall_metrics.json'
    if args.skip_existing and done_path.exists():
        print(f'[{run_name}/{config_name}] skip existing', flush=True)
        return json.load(open(done_path))
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f'[{run_name}/{config_name}] fit raw feature standardizer', flush=True)
    mean, std, train_n, train_pos = fit_standardizer(split['train'], cfg)
    pos_weight = (train_n - train_pos) / max(float(train_pos), 1.0)
    print(f'[{run_name}/{config_name}] load graph scenes train={len(split["train"])} val={len(split["val"])} test={len(split["test"])} pos_weight={pos_weight:.3f}', flush=True)
    train_graph = load_graph_scenes(split['train'], cfg, mean, std, args.max_k)
    val_graph = load_graph_scenes(split['val'], cfg, mean, std, args.max_k)
    test_graph = load_graph_scenes(split['test'], cfg, mean, std, args.max_k)

    gnn = make_gnn_model(train_graph[0]['x'].shape[1], args).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(gnn.parameters(), lr=args.gnn_lr, weight_decay=args.weight_decay)
    best = None; best_state = None; stale = 0; gnn_history = []
    for epoch in range(1, args.gnn_epochs + 1):
        loss = train_gnn_epoch(gnn, train_graph, loss_fn, opt, device, args.seed + epoch, args.grad_clip)
        y_val, val_score, val_slices = predict_gnn(gnn, val_graph, device)
        th_best, _ = g15.choose_threshold(y_val, val_score, val_slices)
        val_rows = g15.per_scene_metrics(val_slices, y_val, val_score, th_best['threshold'])
        rec = {'epoch': epoch, 'loss': loss, 'val_macro_scene_iou': g15.macro_scene_score(val_rows, 'iou'), 'val_macro_scene_f1': g15.macro_scene_score(val_rows, 'f1'), 'val_macro_scene_auprc': g15.macro_scene_score(val_rows, 'auprc'), 'val_threshold': th_best['threshold']}
        gnn_history.append(rec)
        score_tuple = (rec['val_macro_scene_iou'], rec['val_macro_scene_f1'])
        if best is None or score_tuple > (best['val_macro_scene_iou'], best['val_macro_scene_f1']):
            best = dict(rec); best_state = {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}; stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.gnn_epochs:
            print(f'[{run_name}/{config_name}/gnn] epoch {epoch:03d}/{args.gnn_epochs} loss={loss:.4f} val_iou={rec["val_macro_scene_iou"]:.3f} val_f1={rec["val_macro_scene_f1"]:.3f} th={rec["val_threshold"]:.2f}', flush=True)
        if args.patience > 0 and stale >= args.patience:
            print(f'[{run_name}/{config_name}/gnn] early stopping at epoch {epoch}', flush=True)
            break
    gnn.load_state_dict(best_state)
    y_val, val_score, val_slices = predict_gnn(gnn, val_graph, device)
    gnn_th, gnn_curve = g15.choose_threshold(y_val, val_score, val_slices)
    y_test, test_score, test_slices = predict_gnn(gnn, test_graph, device)
    gnn_rows, gnn_micro, gnn_macro = macro_rows(y_test, test_score, test_slices, gnn_th['threshold'])

    print(f'[{run_name}/{config_name}] extract GNN embeddings in memory', flush=True)
    train_emb = extract_embedding_scenes(gnn, train_graph, device)
    val_emb = extract_embedding_scenes(gnn, val_graph, device)
    test_emb = extract_embedding_scenes(gnn, test_graph, device)
    del train_graph, val_graph, test_graph

    emb_mean, emb_std, emb_train_n, emb_train_pos = fit_scene_standardizer(train_emb)
    train_emb = apply_scene_standardizer(train_emb, emb_mean, emb_std)
    val_emb = apply_scene_standardizer(val_emb, emb_mean, emb_std)
    test_emb = apply_scene_standardizer(test_emb, emb_mean, emb_std)

    pn_pos_weight = (emb_train_n - emb_train_pos) / max(float(emb_train_pos), 1.0)
    pn = PointNetMean(train_emb[0]['x'].shape[1], args.pointnet_latent_dim, args.dropout).to(device)
    pn_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pn_pos_weight], device=device))
    pn_opt = torch.optim.AdamW(pn.parameters(), lr=args.pointnet_lr, weight_decay=args.weight_decay)
    best = None; best_state = None; stale = 0; pn_history = []
    for epoch in range(1, args.pointnet_epochs + 1):
        loss = train_pointnet_epoch(pn, train_emb, pn_loss, pn_opt, device, args.seed + 1000 + epoch, args.grad_clip)
        y_val, val_score, val_slices = predict_pointnet(pn, val_emb, device)
        th_best, _ = g15.choose_threshold(y_val, val_score, val_slices)
        val_rows = g15.per_scene_metrics(val_slices, y_val, val_score, th_best['threshold'])
        rec = {'epoch': epoch, 'loss': loss, 'val_macro_scene_iou': g15.macro_scene_score(val_rows, 'iou'), 'val_macro_scene_f1': g15.macro_scene_score(val_rows, 'f1'), 'val_macro_scene_auprc': g15.macro_scene_score(val_rows, 'auprc'), 'val_threshold': th_best['threshold']}
        pn_history.append(rec)
        score_tuple = (rec['val_macro_scene_iou'], rec['val_macro_scene_f1'])
        if best is None or score_tuple > (best['val_macro_scene_iou'], best['val_macro_scene_f1']):
            best = dict(rec); best_state = {k: v.detach().cpu().clone() for k, v in pn.state_dict().items()}; stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.pointnet_epochs:
            print(f'[{run_name}/{config_name}/pointnet] epoch {epoch:03d}/{args.pointnet_epochs} loss={loss:.4f} val_iou={rec["val_macro_scene_iou"]:.3f} val_f1={rec["val_macro_scene_f1"]:.3f} th={rec["val_threshold"]:.2f}', flush=True)
        if args.patience > 0 and stale >= args.patience:
            print(f'[{run_name}/{config_name}/pointnet] early stopping at epoch {epoch}', flush=True)
            break
    pn.load_state_dict(best_state)
    y_val, val_score, val_slices = predict_pointnet(pn, val_emb, device)
    pn_th, pn_curve = g15.choose_threshold(y_val, val_score, val_slices)
    y_test, test_score, test_slices = predict_pointnet(pn, test_emb, device)
    pn_rows, pn_micro, pn_macro = macro_rows(y_test, test_score, test_slices, pn_th['threshold'])

    save_csv(model_dir / 'gnn_history.csv', gnn_history)
    save_csv(model_dir / 'gnn_threshold_curve.csv', gnn_curve)
    save_csv(model_dir / 'gnn_per_scene_metrics.csv', gnn_rows)
    save_csv(model_dir / 'pointnet_history.csv', pn_history)
    save_csv(model_dir / 'pointnet_threshold_curve.csv', pn_curve)
    save_csv(model_dir / 'per_scene_metrics.csv', pn_rows)
    save_csv(model_dir / 'per_category_metrics.csv', g15.per_category_metrics(pn_rows))

    overall = {
        'run_name': run_name,
        'model': f'{args.gnn_variant}_embeddings_pointnet_mean',
        'gnn_variant': args.gnn_variant,
        'dino_method': cfg['dino_method'],
        'source_feature_variant': cfg['feature_variant'],
        'embedding_dim': int(train_emb[0]['x'].shape[1]),
        'train_scenes': len(split['train']), 'val_scenes': len(split['val']), 'test_scenes': len(split['test']),
        'selected_gnn_epoch': int(max(gnn_history, key=lambda r: (r['val_macro_scene_iou'], r['val_macro_scene_f1']))['epoch']),
        'selected_pointnet_epoch': int(max(pn_history, key=lambda r: (r['val_macro_scene_iou'], r['val_macro_scene_f1']))['epoch']),
        'selected_threshold': float(pn_th['threshold']),
        'gnn_direct_macro_scene_average': gnn_macro,
        'gnn_direct_micro': gnn_micro,
        'micro': pn_micro,
        'macro_scene_average': pn_macro,
        'best_scene_iou': max(pn_rows, key=lambda r: r['iou'])['scene_key'],
        'worst_scene_iou': min(pn_rows, key=lambda r: r['iou'])['scene_key'],
    }
    with (model_dir / 'overall_metrics.json').open('w') as f:
        json.dump(overall, f, indent=2)
    torch.save({'model': gnn.state_dict(), 'mean': mean, 'std': std, 'config': cfg, 'overall': overall, 'latent_dim': args.gnn_latent_dim, 'k_layers': args.k_layers, 'gnn_variant': args.gnn_variant}, model_dir / 'gnn_model.pt')
    torch.save({'model': pn.state_dict(), 'embedding_mean': emb_mean, 'embedding_std': emb_std, 'overall': overall, 'threshold': float(pn_th['threshold']), 'latent_dim': args.pointnet_latent_dim}, model_dir / 'pointnet_model.pt')
    offsets = np.asarray([[start, end] for _, start, end in test_slices], dtype=np.int64)
    scene_keys = np.asarray([item['scene_key'] for item, _, _ in test_slices])
    np.savez_compressed(model_dir / 'test_predictions.npz', scene_keys=scene_keys, offsets=offsets, scores=test_score.astype(np.float32), labels=y_test.astype(np.uint8))
    print(f'[{run_name}/{config_name}] POINTNET TEST macro IoU={pn_macro["iou"]:.3f} F1={pn_macro["f1"]:.3f} AUPRC={pn_macro["auprc"]:.3f}; GNN direct IoU={gnn_macro["iou"]:.3f}', flush=True)
    return overall


def write_flat_summary(path: Path, results: list[dict]) -> None:
    rows = []
    for r in results:
        macro = r['macro_scene_average']; gmacro = r['gnn_direct_macro_scene_average']
        rows.append({'run_name': r['run_name'], 'model': r['model'], 'gnn_variant': r.get('gnn_variant', ''), 'dino_method': r['dino_method'], 'source_feature_variant': r['source_feature_variant'], 'macro_iou': macro['iou'], 'macro_f1': macro['f1'], 'macro_precision': macro['precision'], 'macro_recall': macro['recall'], 'macro_auprc': macro['auprc'], 'macro_sim': macro['sim'], 'macro_mae': macro['mae'], 'macro_roc_auc': macro['roc_auc'], 'gnn_direct_macro_iou': gmacro['iou'], 'selected_threshold': r['selected_threshold'], 'best_scene_iou': r['best_scene_iou'], 'worst_scene_iou': r['worst_scene_iou']})
    save_csv(path, rows)


def limit_split_for_smoke(split: dict, args) -> dict:
    limits = {
        'train': args.limit_train_scenes,
        'val': args.limit_val_scenes,
        'test': args.limit_test_scenes,
    }
    if all(v is None for v in limits.values()):
        return split
    return {name: scenes[:limits[name]] if limits[name] is not None else scenes for name, scenes in split.items()}


def aggregate(out_root: Path):
    results = []
    for p in sorted(out_root.glob('**/overall_metrics.json')):
        results.append(json.load(open(p)))
    write_flat_summary(out_root / 'all_finished_models_summary.csv', results)
    with (out_root / 'all_finished_models_summary.json').open('w') as f:
        json.dump(results, f, indent=2)
    print(f'[aggregate] collected {len(results)} results', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_root', type=Path, default=NPZ_ROOT)
    parser.add_argument('--out_root', type=Path, default=OUT_ROOT)
    parser.add_argument('--config', choices=sorted(CONFIGS), default='object_crop_best')
    parser.add_argument('--mode', choices=['seen', 'loco', 'aggregate'], required=True)
    parser.add_argument('--heldout_categories', nargs='+', default=None)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260810)
    parser.add_argument('--gnn_epochs', type=int, default=60)
    parser.add_argument('--pointnet_epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gnn_lr', type=float, default=5e-4)
    parser.add_argument('--pointnet_lr', type=float, default=7e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gnn_latent_dim', type=int, default=192)
    parser.add_argument('--pointnet_latent_dim', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--k_layers', nargs='+', type=int, default=None)
    parser.add_argument('--gnn_variant', choices=['multiscale_k8_16_32_64', 'edge_aware_xyz_delta'], required=True)
    parser.add_argument('--max_k', type=int, default=32)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--limit_train_scenes', type=int, default=None)
    parser.add_argument('--limit_val_scenes', type=int, default=None)
    parser.add_argument('--limit_test_scenes', type=int, default=None)
    args = parser.parse_args()
    if args.k_layers is None:
        args.k_layers = [8, 16, 32, 64] if args.gnn_variant == 'multiscale_k8_16_32_64' else [16, 32]
    args.max_k = max(args.max_k, max(args.k_layers))
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'aggregate':
        aggregate(args.out_root); return
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    items = e44.load_items(args.npz_root)
    cfg = dict(CONFIGS[args.config])
    cfg['dino_root'] = str(cfg['dino_root'])
    with (args.out_root / 'experiment_config.json').open('w') as f:
        json.dump({'experiment': 'Exp50 GNN variations: multiscale and edge-aware embeddings, then PointNet Mean', 'gnn_variant': args.gnn_variant, 'k_layers': args.k_layers, 'configs': {k: {kk: str(vv) for kk, vv in val.items()} for k, val in CONFIGS.items()}, 'no_saved_embeddings': True}, f, indent=2)
    results = []
    if args.mode == 'seen':
        split = limit_split_for_smoke(e44.split_seen(items), args)
        out_dir = args.out_root / args.gnn_variant / args.config / '01_seen_instance_17cat'
        result = train_one_pipeline('01_seen_instance_17cat', split, args.config, cfg, out_dir, args, device)
        results.append(result)
        write_flat_summary(out_dir / 'seen_summary.csv', results)
    else:
        cats = sorted({x['category'] for x in items})
        if args.heldout_categories:
            cats = [c for c in cats if c in set(args.heldout_categories)]
        cats = [c for i, c in enumerate(cats) if i % args.num_shards == args.shard]
        print(f'[loco {args.config} shard {args.shard}/{args.num_shards}] heldout={cats}', flush=True)
        for cat in cats:
            split = limit_split_for_smoke(e44.split_loco(items, cat), args)
            out_dir = args.out_root / args.gnn_variant / args.config / '02_leave_one_category_out_17cat' / cat
            result = train_one_pipeline(cat, split, args.config, cfg, out_dir, args, device)
            results.append(result)
            write_flat_summary(args.out_root / args.gnn_variant / args.config / f'loco_summary_shard_{args.shard}_of_{args.num_shards}.csv', results)
    aggregate(args.out_root)


if __name__ == '__main__':
    main()
