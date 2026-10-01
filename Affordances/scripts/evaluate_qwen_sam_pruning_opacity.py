#!/usr/bin/env python3
"""Evaluate Qwen+SAM/REALM object pruning against HANDAL object-pruned references."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

BASE = Path('/home/gvilaplana/GS-Thesis/Affordances')
OUT_BASE = BASE / 'outputs' / '03_handle_generalization'
FEATURE_ROOTS = [
    BASE / 'data' / 'handal_handle_generalization_features',
    BASE / 'data' / 'handal_new_categories_delta_v1_features',
]
EXPERIMENTS = {
    '27_3anch': OUT_BASE / '27_handal_qwen_sam_3anchor_plus_auto_realm_query_5test_v1',
    '28_1anch': OUT_BASE / '28_handal_qwen_sam_1anchor_plus_auto_realm_query_5test_v1',
    '29_best1': OUT_BASE / '29_handal_qwen_sam_bestconf_1anchor_plus_auto_realm_query_5test_v1',
    '30_best3': OUT_BASE / '30_handal_qwen_sam_best3conf_10cand_plus_auto_realm_query_5test_v1',
}
SCENE_ABBR = {
    'mugs__024004': 'mug',
    'screwdrivers__010000': 'scrw',
    'hammers__031010': 'hamm',
    'combinational_wrenches__002002': 'comb',
    'strainers__032005': 'strn',
}


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def read_xyz(path: Path) -> tuple[PlyData, np.ndarray]:
    ply = PlyData.read(path)
    v = ply['vertex'].data
    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float64)
    return ply, xyz


def write_subset(ply: PlyData, keep: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.array(ply['vertex'].data, copy=True)[keep]
    PlyData([PlyElement.describe(data, 'vertex')], text=ply.text).write(out)


def slug(query: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in query.lower()).strip('_')


def feature_root_for(scene_key: str) -> Path:
    for root in FEATURE_ROOTS:
        if (root / 'work' / 'object_ply' / f'{scene_key}_object_pruned_thr0.60.ply').exists():
            return root
    raise FileNotFoundError(f'No HANDAL reference object PLY for {scene_key}')


def scene_query(exp_root: Path, scene_key: str) -> str:
    meta = json.loads((exp_root / scene_key / 'metadata.json').read_text())
    return meta['query']


def pred_ply_path(exp_root: Path, scene_key: str, query: str) -> Path:
    summary = json.loads((exp_root / scene_key / 'final_ply' / 'selected_id_ply_summary.json').read_text())
    q = summary[query]
    return Path(q['truecolor'])


def reference_tolerance(gt_xyz: np.ndarray, factor: float) -> float:
    if len(gt_xyz) < 2:
        return 1e-6
    d, _ = cKDTree(gt_xyz).query(gt_xyz, k=2)
    nn = d[:, 1]
    med = float(np.median(nn[np.isfinite(nn)]))
    return max(med * factor, 1e-6)


def metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray, tol: float) -> dict:
    pred_n = len(pred_xyz)
    gt_n = len(gt_xyz)
    if pred_n == 0 or gt_n == 0:
        inter_p = 0
        inter_g = 0
    else:
        gt_tree = cKDTree(gt_xyz)
        pred_tree = cKDTree(pred_xyz)
        dp, _ = gt_tree.query(pred_xyz, k=1)
        dg, _ = pred_tree.query(gt_xyz, k=1)
        inter_p = int((dp <= tol).sum())
        inter_g = int((dg <= tol).sum())
    precision = inter_p / pred_n if pred_n else 0.0
    recall = inter_g / gt_n if gt_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # Symmetric set-style approximation: TP from pred side, FN from GT uncovered.
    union = pred_n + gt_n - min(inter_p, inter_g)
    iou = min(inter_p, inter_g) / union if union else 0.0
    return {
        'pred_count': pred_n,
        'gt_count': gt_n,
        'pred_near_gt': inter_p,
        'gt_covered_by_pred': inter_g,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=Path, default=OUT_BASE / '31_qwen_sam_pruning_opacity_eval_v1')
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.15, 0.30])
    ap.add_argument('--nn_tol_factor', type=float, default=2.5)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = args.out_dir / 'plys'
    ply_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    scenes = sorted(SCENE_ABBR)

    for scene_key in scenes:
        root = feature_root_for(scene_key)
        gt_ply = root / 'work' / 'object_ply' / f'{scene_key}_object_pruned_thr0.60.ply'
        gt_plydata, gt_xyz = read_xyz(gt_ply)
        tol = reference_tolerance(gt_xyz, args.nn_tol_factor)
        shutil.copy2(gt_ply, ply_dir / f'{SCENE_ABBR[scene_key]}_gt.ply')
        for exp_name, exp_root in EXPERIMENTS.items():
            query = scene_query(exp_root, scene_key)
            pred_path = pred_ply_path(exp_root, scene_key, query)
            pred_ply, pred_xyz_all = read_xyz(pred_path)
            v = pred_ply['vertex'].data
            op = sigmoid(v['opacity']) if 'opacity' in v.dtype.names else np.ones(len(v), dtype=np.float64)
            thresholds = [('none', None)] + [(f'op{str(t).replace(".", "p")}', t) for t in args.thresholds]
            for tag, thr in thresholds:
                keep = np.ones(len(v), dtype=bool) if thr is None else (op > float(thr))
                pred_xyz = pred_xyz_all[keep]
                short = f'{SCENE_ABBR[scene_key]}_{exp_name.split("_",1)[1]}_{tag}.ply'
                out_ply = ply_dir / short
                write_subset(pred_ply, keep, out_ply)
                m = metrics(pred_xyz, gt_xyz, tol)
                rows.append({
                    'experiment': exp_name,
                    'scene_key': scene_key,
                    'scene': SCENE_ABBR[scene_key],
                    'query': query,
                    'opacity_filter': tag,
                    'opacity_threshold': '' if thr is None else thr,
                    'nn_tolerance': tol,
                    'pred_ply': str(pred_path),
                    'filtered_ply': str(out_ply),
                    'gt_ply': str(gt_ply),
                    **m,
                })

    per_scene = pd.DataFrame(rows)
    per_scene.to_csv(args.out_dir / 'per_scene_metrics.csv', index=False)
    agg = per_scene.groupby(['experiment', 'opacity_filter'], as_index=False).agg(
        mean_iou=('iou', 'mean'),
        mean_precision=('precision', 'mean'),
        mean_recall=('recall', 'mean'),
        mean_f1=('f1', 'mean'),
        mean_pred_count=('pred_count', 'mean'),
        total_pred_count=('pred_count', 'sum'),
        scenes=('scene_key', 'count'),
    )
    agg.to_csv(args.out_dir / 'aggregate_metrics.csv', index=False)

    for metric in ['iou', 'precision', 'recall', 'f1', 'pred_count']:
        table = agg.pivot(index='experiment', columns='opacity_filter', values=f'mean_{metric}' if metric != 'pred_count' else 'mean_pred_count')
        table.to_csv(args.out_dir / f'{metric}_table.csv')

    md = ['# Qwen+SAM Pruning Opacity Eval', '', 'Opacity is applied only to predicted pruned PLYs. HANDAL object-pruned PLY is unfiltered GT. Metrics use nearest-neighbor spatial overlap because REALM and HANDAL point clouds have different Gaussian counts.', '']
    for metric in ['mean_iou', 'mean_precision', 'mean_recall', 'mean_f1', 'mean_pred_count']:
        tab = agg.pivot(index='experiment', columns='opacity_filter', values=metric).round(4)
        md += [f'## {metric}', '', '```', tab.to_string(), '```', '']
    (args.out_dir / 'summary_tables.md').write_text('\n'.join(md))
    print(f'wrote {args.out_dir}')
    print(agg.sort_values(['experiment', 'opacity_filter']).to_string(index=False))


if __name__ == '__main__':
    main()
