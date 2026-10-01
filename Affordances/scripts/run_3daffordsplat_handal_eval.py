#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from plyfile import PlyData, PlyElement
from sklearn.metrics import roc_auc_score

BASE_ROOT = Path('/home/gvilaplana/GS-Thesis/Affordances')
REPO_ROOT = BASE_ROOT / 'external' / '3DAffordSplat'
FEATURE_ROOT = BASE_ROOT / 'data' / 'handal_handle_generalization_features'
NPZ_ROOT = FEATURE_ROOT / 'npz'
DEFAULT_SPLIT = BASE_ROOT / 'outputs' / '03_handle_generalization' / '01_per_gaussian_mlp_baseline' / 'exp1a_seen_instance' / 'split_manifest.json'
DEFAULT_OUT = BASE_ROOT / 'outputs' / '04_3daffordsplat_baseline'
CHECKPOINT = REPO_ROOT / 'checkpoints' / 'finetune_Seen_best.pth'
QUESTION_CSV = REPO_ROOT / 'AffordSplat' / 'Affordance-Question.csv'

# Compatibility shims for unused imports in the upstream repo under torch 1.12.
xpu = types.ModuleType('torch.xpu')
xpu.device = None
sys.modules.setdefault('torch.xpu', xpu)
torchsummary = types.ModuleType('torchsummary')
torchsummary.summary = lambda *args, **kwargs: None
sys.modules.setdefault('torchsummary', torchsummary)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'models'))

from models.Affordsplat_net import Affordsplat_net  # noqa: E402

C0 = 0.28209479177387814

CATEGORY_ALIASES = {
    'adjustable_wrenches': 'adjustable wrench',
    'measuring_cups': 'measuring cup',
    'mugs': 'mug',
    'pots_pans': 'pot or pan',
    'power_drills': 'power drill',
    'screwdrivers': 'screwdriver',
    'spatulas': 'spatula',
    'whisks': 'whisk',
    'hammers': 'hammer',
}

MUG_GRASP_QUESTION = 'if you grab this mug, which points on the mug handle will your palm touch?'
MUG_WRAP_QUESTION = 'when holding this mug without using the handle, where would your palm wrap around the mug?'


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def mean_normalize(points):
    points = points.astype(np.float32)
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    dist = np.sqrt((centered ** 2).sum(axis=1)).max()
    dist = max(float(dist), 1e-8)
    return centered / dist


def read_gs_features(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
    xyz_norm = mean_normalize(xyz)
    scale = np.column_stack([v['scale_0'], v['scale_1'], v['scale_2']]).astype(np.float32)
    rot = np.column_stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']]).astype(np.float32)
    return np.column_stack([xyz_norm, scale, rot]).astype(np.float32), xyz


def color_scores_on_ply(source_ply, out_ply, scores, threshold=0.5, labels=None):
    ply = PlyData.read(source_ply)
    vertices = np.array(ply['vertex'].data, copy=True)
    scores = np.asarray(scores, dtype=np.float32)
    if len(vertices) != len(scores):
        raise ValueError(f'PLY/score length mismatch: {source_ply} {len(vertices)} vs {len(scores)}')
    rgb = np.zeros((len(scores), 3), dtype=np.float32)
    rgb[:, 0] = np.clip(scores, 0.05, 1.0)
    rgb[:, 1] = 0.04
    rgb[:, 2] = np.clip(1.0 - scores, 0.0, 1.0)
    if labels is not None:
        # Make false negatives magenta-ish and true positives bright red for quick visual debugging.
        labels = np.asarray(labels).astype(bool)
        pred = scores >= threshold
        rgb[np.logical_and(labels, pred)] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
        rgb[np.logical_and(labels, ~pred)] = np.array([1.0, 0.0, 0.8], dtype=np.float32)
    sh = (rgb - 0.5) / C0
    vertices['f_dc_0'] = sh[:, 0]
    vertices['f_dc_1'] = sh[:, 1]
    vertices['f_dc_2'] = sh[:, 2]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(out_ply)


def metric_row(labels, scores, threshold):
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores).astype(np.float32)
    p = s >= threshold
    tp = int(np.logical_and(p, y).sum())
    tn = int(np.logical_and(~p, ~y).sum())
    fp = int(np.logical_and(p, ~y).sum())
    fn = int(np.logical_and(~p, y).sum())
    n = max(len(y), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    try:
        auc = roc_auc_score(y.astype(np.uint8), s) if y.any() and (~y).any() else float('nan')
    except Exception:
        auc = float('nan')
    return {
        'num_gaussians': int(len(y)),
        'gt_handle_ratio': float(y.mean()) if n else 0.0,
        'pred_handle_ratio': float(p.mean()) if n else 0.0,
        'accuracy': float((tp + tn) / n),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'auc': float(auc),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'fp_percent': float(fp / n),
        'fn_percent': float(fn / n),
        'correct_percent': float((tp + tn) / n),
    }


def load_question_table():
    if not QUESTION_CSV.exists():
        return None
    return pd.read_csv(QUESTION_CSV)


def make_prompt(category, affordance, prompt_mode, question_table=None):
    obj = CATEGORY_ALIASES.get(category, category.replace('_', ' '))
    official_obj = 'mug' if category == 'mugs' else obj.replace(' ', '')
    if question_table is not None:
        row = question_table[(question_table['Object'] == official_obj) & (question_table['Affordance'] == affordance)]
        if len(row) > 0:
            if prompt_mode == 'official_all':
                question = str(row[[f'Question{i}' for i in range(15)]].values[0])
            else:
                question = str(row['Question0'].values[0])
            answer = str(row['Answer2'].values[0])
            return question, answer
    if category == 'mugs' and affordance == 'grasp':
        return MUG_GRASP_QUESTION, '<Aff>'
    if category == 'mugs' and affordance == 'wrap_grasp':
        return MUG_WRAP_QUESTION, '<Aff>'
    if affordance == 'wrap_grasp':
        return f'When holding the {obj} by wrapping your palm around it, which points should the hand contact?', '<Aff>'
    return f'If you want to grasp the {obj}, which points should your hand or fingers touch?', '<Aff>'


def load_model(device):
    old_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        model_args = read_yaml(REPO_ROOT / 'config' / 'model_config.yaml')
        model = Affordsplat_net(model_args).to(device)
        ckpt = torch.load(CHECKPOINT, map_location='cpu')
        state = ckpt['model_state_dict']
        if 'mmfm.pos_embed.weight' in state:
            del state['mmfm.pos_embed.weight']
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'Loaded checkpoint: {CHECKPOINT}')
        print(f'missing_keys={len(missing)} unexpected_keys={len(unexpected)}')
        if missing:
            print('missing:', missing[:10])
        if unexpected:
            print('unexpected:', unexpected[:10])
        model.eval()
        return model
    finally:
        os.chdir(old_cwd)


def predict_scene(model, ply_path, question, answer, device):
    old_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        features, _ = read_gs_features(ply_path)
        x = torch.from_numpy(features[None]).to(device)
        mask = torch.ones((1, x.shape[1]), dtype=x.dtype, device=device)
        pc_mean_all = torch.randn((1, 1, 2048, 3), dtype=x.dtype, device=device)
        pc_aff_map_all = torch.ones((1, 1, 2048, 3), dtype=x.dtype, device=device)
        with torch.no_grad():
            _, pred, _, predicted_text = model(
                x, x, mask, pc_mean_all, pc_aff_map_all,
                [question], [answer], device=device, use_csa=False
            )
        return pred.flatten().detach().cpu().numpy().astype(np.float32), predicted_text[0]
    finally:
        os.chdir(old_cwd)


def load_split_items(split_path, split_name, categories=None, limit=None):
    manifest = json.loads(Path(split_path).read_text())
    items = manifest['splits'][split_name]
    if categories:
        cats = set(categories)
        items = [x for x in items if x['category'] in cats]
    if limit:
        items = items[:limit]
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_path', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--categories', nargs='*', default=['mugs'])
    parser.add_argument('--affordances', nargs='*', default=['grasp'])
    parser.add_argument('--out_dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prompt_mode', default='single', choices=['single', 'official_all'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--export_plys', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(device)
    question_table = load_question_table()

    items = load_split_items(args.split_path, args.split, args.categories, args.limit)
    rows = []
    all_labels = []
    all_scores = []
    export_count = 0
    for item in items:
        scene_key = item['scene_key']
        npz_path = NPZ_ROOT / f'{scene_key}.npz'
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        with np.load(npz_path, allow_pickle=False) as z:
            labels = z['handle_labels_thr0_25'].astype(np.uint8)
            ply_path = Path(str(z['source_object_ply'][0]))
            xyz_npz = z['xyz'].astype(np.float32)
        if not ply_path.exists():
            ply_path = FEATURE_ROOT / 'work' / 'object_ply' / f'{scene_key}_object_pruned_thr0.60.ply'
        _, xyz_ply = read_gs_features(ply_path)
        if len(labels) != len(xyz_ply):
            raise ValueError(f'{scene_key}: labels={len(labels)} ply={len(xyz_ply)}')
        xyz_delta = float(np.max(np.abs(xyz_npz - xyz_ply))) if len(labels) else 0.0
        if xyz_delta > 1e-4:
            print(f'WARN xyz mismatch {scene_key}: max_abs_delta={xyz_delta:.6g}')

        for aff in args.affordances:
            question, answer = make_prompt(item['category'], aff, args.prompt_mode, question_table)
            scores, predicted_text = predict_scene(model, ply_path, question, answer, device)
            if len(scores) != len(labels):
                raise ValueError(f'{scene_key}/{aff}: scores={len(scores)} labels={len(labels)}')
            row = {
                'scene_key': scene_key,
                'category': item['category'],
                'scene_id': item['scene_id'],
                'instance_id': item['instance_id'],
                'affordance_query': aff,
                'question': question,
                'answer_prompt': answer,
                'prompt_mode': args.prompt_mode,
                'predicted_text': predicted_text,
                'xyz_max_abs_delta': xyz_delta,
            }
            row.update(metric_row(labels, scores, args.threshold))
            rows.append(row)
            all_labels.append(labels)
            all_scores.append(scores)
            np.savez_compressed(
                args.out_dir / f'{scene_key}_{aff}_scores.npz',
                scene_key=np.array([scene_key]),
                category=np.array([item['category']]),
                affordance_query=np.array([aff]),
                scores=scores.astype(np.float32),
                labels=labels.astype(np.uint8),
            )
            if export_count < args.export_plys:
                out_ply = args.out_dir / 'prediction_ply' / aff / f'{scene_key}_{aff}_3daffordsplat_prediction.ply'
                color_scores_on_ply(ply_path, out_ply, scores, args.threshold, labels=labels)
                row['prediction_ply'] = str(out_ply)
                export_count += 1
            print(f"{scene_key}/{aff}: IoU={row['iou']:.3f} F1={row['f1']:.3f} pred_ratio={row['pred_handle_ratio']:.3f} gt_ratio={row['gt_handle_ratio']:.3f}")

    per_scene_csv = args.out_dir / 'per_scene_metrics.csv'
    with per_scene_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    macro = {k: float(np.nanmean([r[k] for r in rows])) for k in ['accuracy', 'precision', 'recall', 'f1', 'iou', 'auc', 'gt_handle_ratio', 'pred_handle_ratio', 'correct_percent', 'fp_percent', 'fn_percent']}
    y_all = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.uint8)
    s_all = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    micro = metric_row(y_all, s_all, args.threshold) if len(y_all) else {}
    summary = {
        'split': args.split,
        'categories': args.categories,
        'affordances': args.affordances,
        'threshold': args.threshold,
        'prompt_mode': args.prompt_mode,
        'num_scene_queries': len(rows),
        'macro_scene_average': macro,
        'micro': micro,
        'checkpoint': str(CHECKPOINT),
        'mode': '3DAffordSplat Seen checkpoint, zero-shot on HANDAL object-pruned Gaussians, direct npz label evaluation',
    }
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Wrote {per_scene_csv}')


if __name__ == '__main__':
    main()
