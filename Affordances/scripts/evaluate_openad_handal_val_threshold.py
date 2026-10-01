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
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

BASE = Path('/home/gvilaplana/GS-Thesis/Affordances')
OPENAD_ROOT = BASE / 'external' / 'OpenAD'
NPZ_ROOT = BASE / 'data' / 'handal_exp40_15k_gt_features_v1' / 'npz' / 'B_center_ellipsoid20_strict75'
CKPT = OPENAD_ROOT / 'checkpoints' / 'best_model_openad_pn2_estimation.t7'
OUT_ROOT = BASE / 'outputs' / '03_handle_generalization' / '52_openad_val_threshold_calibration_v1'

BAD_SCENES = {
    'spatulas__032005',
    'slip_joint_pliers__090006',
    'slip_joint_pliers__092005',
    'locking_pliers__040003',
    'locking_pliers__091004',
    'utensils__022007',
    'slip_joint_pliers__093004',
    'locking_pliers__090004',
    'utensils__026007',
    'locking_pliers__041005',
}


def scalar_str(z: np.lib.npyio.NpzFile, key: str, default: str = '') -> str:
    if key not in z.files:
        return default
    value = z[key]
    return str(value[0] if getattr(value, 'shape', ()) else value)


def pc_normalize(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz.astype(np.float32, copy=True)
    xyz -= xyz.mean(axis=0, keepdims=True)
    radius = np.sqrt((xyz * xyz).sum(axis=1)).max()
    if radius > 1e-12:
        xyz /= radius
    return xyz


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> dict:
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
        balanced_accuracy = float(balanced_accuracy_score(y_true.astype(np.uint8), y_pred.astype(np.uint8)))
    except ValueError:
        balanced_accuracy = math.nan
    try:
        roc_auc = float(roc_auc_score(y_true.astype(np.uint8), prob))
    except ValueError:
        roc_auc = math.nan
    try:
        auprc = float(average_precision_score(y_true.astype(np.uint8), prob))
    except ValueError:
        auprc = math.nan
    return {
        'num_points': n,
        'num_gt_handle': int(y_true.sum()),
        'num_pred_handle': int(y_pred.sum()),
        'gt_handle_ratio': float(y_true.mean()) if n else 0.0,
        'pred_handle_ratio': float(y_pred.mean()) if n else 0.0,
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'roc_auc': roc_auc,
        'auprc': auprc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp_percent': tp / max(n, 1),
        'tn_percent': tn / max(n, 1),
        'fp_percent': fp / max(n, 1),
        'fn_percent': fn / max(n, 1),
        'correct_percent': (tp + tn) / max(n, 1),
        'incorrect_percent': (fp + fn) / max(n, 1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_items(npz_root: Path, limit_per_split: int | None = None) -> list[dict]:
    items = []
    split_counts = defaultdict(int)
    for path in sorted(npz_root.glob('*.npz')):
        if path.stem in BAD_SCENES:
            continue
        category, scene_id = path.stem.rsplit('__', 1)
        with np.load(path, allow_pickle=False) as z:
            split = scalar_str(z, 'model_split', 'unknown')
            if limit_per_split is not None and split_counts[split] >= limit_per_split:
                continue
            split_counts[split] += 1
            items.append({
                'scene_key': path.stem,
                'category': scalar_str(z, 'category', category),
                'scene_id': scalar_str(z, 'scene_id', scene_id),
                'instance_id': scalar_str(z, 'instance_id', scene_id),
                'model_split': split,
                'raw_split': scalar_str(z, 'raw_split', 'unknown'),
                'npz': str(path),
            })
    return items


def aggregate(rows: list[dict], group_key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    out = []
    for group, items in sorted(groups.items()):
        agg = {'group_by': group_key, group_key: group, 'num_scenes': len(items)}
        for key in ['num_points', 'num_gt_handle', 'num_pred_handle', 'tp', 'tn', 'fp', 'fn']:
            agg[key] = int(sum(int(x[key]) for x in items))
        for key in ['iou', 'f1', 'precision', 'recall', 'accuracy', 'balanced_accuracy', 'roc_auc', 'auprc', 'gt_handle_ratio', 'pred_handle_ratio']:
            vals = [float(x[key]) for x in items if not math.isnan(float(x[key]))]
            agg[f'mean_{key}'] = float(np.mean(vals)) if vals else math.nan
        out.append(agg)
    return out


def threshold_curve(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    rows = []
    for threshold in thresholds:
        pred = prob >= threshold
        rows.append({'threshold': float(threshold), **metric_dict(y, pred, prob)})
    return rows


def choose_threshold(curve: list[dict], objective: str) -> dict:
    if objective not in curve[0]:
        raise ValueError(f'Unknown objective {objective}')
    return max(curve, key=lambda row: (float(row[objective]), float(row['f1']), float(row['balanced_accuracy'])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_root', type=Path, default=NPZ_ROOT)
    parser.add_argument('--out_root', type=Path, default=OUT_ROOT)
    parser.add_argument('--openad_root', type=Path, default=OPENAD_ROOT)
    parser.add_argument('--checkpoint', type=Path, default=CKPT)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--affordances', nargs='+', default=['grasp', 'none'])
    parser.add_argument('--positive_class', default='grasp')
    parser.add_argument('--objective', default='iou', choices=['iou', 'f1', 'balanced_accuracy'])
    parser.add_argument('--num_thresholds', type=int, default=1001)
    parser.add_argument('--limit_per_split', type=int, default=None)
    parser.add_argument('--summary_every', type=int, default=50)
    args = parser.parse_args()

    if args.positive_class not in args.affordances:
        raise ValueError(f'positive_class {args.positive_class} not found in affordances {args.affordances}')
    pos_idx = args.affordances.index(args.positive_class)

    sys.path.insert(0, str(args.openad_root))
    from models.openad_pn2 import OpenAD_PN2  # noqa: WPS433

    torch.cuda.set_device(int(args.gpu))
    model = OpenAD_PN2(args=None, num_classes=len(args.affordances), normal_channel=False).cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)

    items = load_items(args.npz_root, args.limit_per_split)
    args.out_root.mkdir(parents=True, exist_ok=True)
    scene_cache = []

    with torch.no_grad():
        for idx, item in enumerate(items, start=1):
            with np.load(item['npz'], allow_pickle=False) as z:
                xyz = z['xyz'].astype(np.float32)
                y = z['handle_labels_thr0_25'].astype(np.uint8)
            x = torch.from_numpy(pc_normalize(xyz).T[None]).float().cuda()
            logp = model(x, args.affordances)
            prob = torch.exp(logp)[0, pos_idx].detach().cpu().numpy().astype(np.float32)
            scene_cache.append({**item, 'y': y, 'prob': prob})
            del x, logp
            torch.cuda.empty_cache()
            if idx == 1 or idx == len(items) or idx % args.summary_every == 0:
                print(f'[{idx}/{len(items)}] {item["scene_key"]}', flush=True)

    val_items = [x for x in scene_cache if x['model_split'] == 'val']
    if not val_items:
        raise RuntimeError('No validation scenes found for threshold calibration.')
    val_y = np.concatenate([x['y'] for x in val_items])
    val_prob = np.concatenate([x['prob'] for x in val_items])
    thresholds = np.linspace(0.0, 1.0, args.num_thresholds, dtype=np.float32)
    curve = threshold_curve(val_y, val_prob, thresholds)
    best = choose_threshold(curve, args.objective)
    threshold = float(best['threshold'])

    rows = []
    for item in scene_cache:
        pred = item['prob'] >= threshold
        rows.append({
            'scene_key': item['scene_key'],
            'category': item['category'],
            'scene_id': item['scene_id'],
            'instance_id': item['instance_id'],
            'model_split': item['model_split'],
            'raw_split': item['raw_split'],
            'checkpoint': str(args.checkpoint),
            'affordances': '|'.join(args.affordances),
            'positive_class': args.positive_class,
            'threshold': threshold,
            **metric_dict(item['y'], pred, item['prob']),
        })

    metrics_dir = args.out_root / 'metrics'
    write_csv(metrics_dir / 'val_threshold_curve.csv', curve)
    write_csv(metrics_dir / 'per_scene_metrics.csv', rows)
    write_csv(metrics_dir / 'per_category_metrics.csv', aggregate(rows, 'category'))
    write_csv(metrics_dir / 'per_model_split_metrics.csv', aggregate(rows, 'model_split'))
    write_csv(metrics_dir / 'per_raw_split_metrics.csv', aggregate(rows, 'raw_split'))
    split_rows = aggregate(rows, 'model_split')
    split_iou = {row['model_split']: row.get('mean_iou', math.nan) for row in split_rows}
    summary = {
        'num_scenes': len(rows),
        'affordances': args.affordances,
        'positive_class': args.positive_class,
        'checkpoint': str(args.checkpoint),
        'calibration_split': 'val',
        'objective': args.objective,
        'threshold': threshold,
        'val_objective_at_threshold': float(best[args.objective]),
        'split_mean_iou': split_iou,
        'out_root': str(args.out_root),
    }
    (metrics_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    md = [
        '# OpenAD Validation-Calibrated Threshold',
        '',
        f'- scenes: {len(rows)}',
        f'- affordances: {" | ".join(args.affordances)}',
        f'- positive class: {args.positive_class}',
        '- calibration split: val',
        f'- objective: {args.objective}',
        f'- selected threshold: {threshold:.4f}',
        f'- val {args.objective} at threshold: {float(best[args.objective]):.4f}',
        f'- train mean IoU: {split_iou.get("train", math.nan):.4f}',
        f'- val mean IoU: {split_iou.get("val", math.nan):.4f}',
        f'- test mean IoU: {split_iou.get("test", math.nan):.4f}',
        '',
        'No PLY files are saved in this experiment.',
    ]
    (metrics_dir / 'metrics.md').write_text('\n'.join(md) + '\n')
    print(f'[done] threshold={threshold:.4f} wrote {args.out_root}', flush=True)


if __name__ == '__main__':
    main()
