import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from build_handal_generalization_feature_pilot import (
    DATASET_ROOT,
    OUT_ROOT,
    dino_embeddings,
    handle_labels,
    load_dinov2,
    load_manifest,
    local_scene_path,
    object_prune,
    save_compact_npz,
    scene_key,
    train_3dgs,
    write_debug_ply,
    write_json,
)


def select_all_rows(rows, dataset_root):
    valid = []
    for row in rows:
        if local_scene_path(dataset_root, row).exists():
            valid.append(row)
    valid.sort(key=lambda row: (row['split'], row['category'], row['instance_id'], row['scene_id']))
    return valid


def compact_npz_path(args, row):
    return args.npz_root / f"{scene_key(row['category'], row['scene_id'])}.npz"


def debug_ply_path(args, row):
    return args.ply_root / f"{scene_key(row['category'], row['scene_id'])}_object_handle_red.ply"


def choose_debug_keys(rows, args):
    selected = set()
    counts = {}
    if args.debug_extra_per_category <= 0:
        return selected
    for row in rows:
        category = row['category']
        if counts.get(category, 0) >= args.debug_extra_per_category:
            continue
        if debug_ply_path(args, row).exists():
            continue
        selected.add(scene_key(category, row['scene_id']))
        counts[category] = counts.get(category, 0) + 1
    return selected


def load_summary_from_npz(row, args):
    path = compact_npz_path(args, row)
    data = np.load(path)
    labels = data['handle_labels_thr0_25']
    valid_dino = data['valid_dino']
    input_gaussians = int(data['input_gaussians'][0]) if 'input_gaussians' in data.files else -1
    object_gaussians = int(labels.shape[0])
    handle_gaussians = int(labels.sum())
    debug_path = debug_ply_path(args, row)
    return {
        'scene_key': scene_key(row['category'], row['scene_id']),
        'split': row['split'],
        'category': row['category'],
        'scene_id': row['scene_id'],
        'input_gaussians': input_gaussians,
        'object_gaussians': object_gaussians,
        'handle_gaussians': handle_gaussians,
        'handle_ratio': float(handle_gaussians / max(object_gaussians, 1)),
        'valid_dino': int(valid_dino.sum()),
        'valid_dino_ratio': float(valid_dino.sum() / max(object_gaussians, 1)),
        'npz': str(path),
        'debug_ply': str(debug_path) if debug_path.exists() else '',
        'object_ply': '',
        'reused': True,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def category_summary(summaries):
    grouped = {}
    for item in summaries:
        grouped.setdefault((item['split'], item['category']), []).append(item)
    rows = []
    for (split, category), items in sorted(grouped.items()):
        rows.append({
            'split': split,
            'category': category,
            'num_scenes': len(items),
            'total_input_gaussians': int(sum(max(item['input_gaussians'], 0) for item in items)),
            'total_object_gaussians': int(sum(item['object_gaussians'] for item in items)),
            'total_handle_gaussians': int(sum(item['handle_gaussians'] for item in items)),
            'mean_object_gaussians': float(np.mean([item['object_gaussians'] for item in items])),
            'mean_handle_ratio': float(np.mean([item['handle_ratio'] for item in items])),
            'mean_valid_dino_ratio': float(np.mean([item['valid_dino_ratio'] for item in items])),
        })
    return rows


def write_summaries(args, summaries, mode):
    summaries = sorted(summaries, key=lambda item: (item['split'], item['category'], item['scene_id']))
    payload = {
        'mode': mode,
        'stored_gaussian_set': 'object_pruned_only',
        'dino_dtype': args.dino_dtype,
        'geometry_dtype': 'float32',
        'object_threshold': args.object_threshold,
        'handle_threshold': args.handle_threshold,
        'npz_root': str(args.npz_root),
        'ply_root': str(args.ply_root),
        'num_scenes': len(summaries),
        'scenes': summaries,
    }
    summary_path = args.out_root / ('full_summary.json' if mode == 'full' else 'pilot_summary.json')
    write_json(summary_path, payload)
    scene_fields = [
        'scene_key', 'split', 'category', 'scene_id', 'input_gaussians',
        'object_gaussians', 'handle_gaussians', 'handle_ratio',
        'valid_dino', 'valid_dino_ratio', 'npz', 'debug_ply', 'object_ply', 'reused',
    ]
    write_csv(args.out_root / 'per_scene_counts.csv', summaries, scene_fields)
    cat_rows = category_summary(summaries)
    write_csv(
        args.out_root / 'per_category_summary.csv',
        cat_rows,
        ['split', 'category', 'num_scenes', 'total_input_gaussians', 'total_object_gaussians',
         'total_handle_gaussians', 'mean_object_gaussians', 'mean_handle_ratio', 'mean_valid_dino_ratio'],
    )
    return summary_path


def build_full_features(rows, args, debug_keys):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model, patch_size, hidden_size = load_dinov2(args.model_name, device, args.local_files_only)
    if args.resize_width % patch_size != 0 or args.resize_height % patch_size != 0:
        raise ValueError(f'resize dimensions must be divisible by patch size {patch_size}')

    summaries = []
    start = time.time()
    for idx, row in enumerate(rows, start=1):
        key = scene_key(row['category'], row['scene_id'])
        npz_path = compact_npz_path(args, row)
        if npz_path.exists() and not args.force_features:
            summary = load_summary_from_npz(row, args)
            summaries.append(summary)
            print(f"[{idx}/{len(rows)}] reuse NPZ {key}: object {summary['object_gaussians']}, handle {summary['handle_ratio']:.1%}", flush=True)
            continue

        print(f'[{idx}/{len(rows)}] building compact features for {key}', flush=True)
        object_ply, object_scores, gaussian_indices, input_gaussians = object_prune(row, args)
        handle_scores, handle_labels_arr, handle_visible = handle_labels(row, args, object_ply)
        dino_pack = dino_embeddings(row, args, object_ply, model, patch_size, hidden_size, device)
        out_path = save_compact_npz(
            row,
            args,
            object_ply,
            object_scores,
            gaussian_indices,
            input_gaussians,
            handle_scores,
            handle_labels_arr,
            handle_visible,
            dino_pack,
        )
        debug_ply = ''
        if key in debug_keys:
            debug_ply = debug_ply_path(args, row)
            write_debug_ply(object_ply, handle_labels_arr, debug_ply)

        object_gaussians = int(len(object_scores))
        handle_gaussians = int(handle_labels_arr.sum())
        summary = {
            'scene_key': key,
            'split': row['split'],
            'category': row['category'],
            'scene_id': row['scene_id'],
            'input_gaussians': int(input_gaussians),
            'object_gaussians': object_gaussians,
            'handle_gaussians': handle_gaussians,
            'handle_ratio': float(handle_gaussians / max(object_gaussians, 1)),
            'valid_dino': int(dino_pack[1].sum()),
            'valid_dino_ratio': float(dino_pack[1].sum() / max(object_gaussians, 1)),
            'npz': str(out_path),
            'debug_ply': str(debug_ply),
            'object_ply': str(object_ply),
            'reused': False,
        }
        summaries.append(summary)
        elapsed = (time.time() - start) / 60.0
        avg = elapsed / max(idx, 1)
        eta = avg * (len(rows) - idx)
        print(
            f"[{idx}/{len(rows)}] {key}: object {object_gaussians}/{input_gaussians}, "
            f"handle {handle_gaussians} ({summary['handle_ratio']:.1%}), "
            f"valid DINO {summary['valid_dino_ratio']:.1%}; ETA {eta:.1f} min",
            flush=True,
        )
        if idx % args.summary_every == 0:
            write_summaries(args, summaries, args.mode)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pilot', 'full'], default='full')
    parser.add_argument('--dataset_root', type=Path, default=DATASET_ROOT)
    parser.add_argument('--out_root', type=Path, default=OUT_ROOT)
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--resolution', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=96)
    parser.add_argument('--init_points', type=int, default=30000)
    parser.add_argument('--force_3dgs', action='store_true')
    parser.add_argument('--force_features', action='store_true')
    parser.add_argument('--skip_3dgs', action='store_true')
    parser.add_argument('--model_name', default='facebook/dinov2-small')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--local_files_only', action='store_true')
    parser.add_argument('--dino_dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--resize_width', type=int, default=448)
    parser.add_argument('--resize_height', type=int, default=336)
    parser.add_argument('--max_dino_views', type=int, default=None)
    parser.add_argument('--object_threshold', type=float, default=0.60)
    parser.add_argument('--object_min_visible', type=int, default=20)
    parser.add_argument('--handle_threshold', type=float, default=0.25)
    parser.add_argument('--handle_min_visible', type=int, default=10)
    parser.add_argument('--sigma_extent', type=float, default=3.0)
    parser.add_argument('--max_radius', type=int, default=24)
    parser.add_argument('--min_var_px', type=float, default=0.25)
    parser.add_argument('--max_patch_radius', type=int, default=4)
    parser.add_argument('--min_patch_radius', type=int, default=1)
    parser.add_argument('--min_var_patch', type=float, default=0.25)
    parser.add_argument('--min_alpha', type=float, default=1.0 / 255.0)
    parser.add_argument('--alpha_clip', type=float, default=0.99)
    parser.add_argument('--min_total_weight', type=float, default=1e-4)
    parser.add_argument('--min_view_weight', type=float, default=1e-5)
    parser.add_argument('--debug_extra_per_category', type=int, default=1)
    parser.add_argument('--summary_every', type=int, default=5)
    args = parser.parse_args()

    args.work_root = args.out_root / 'work'
    args.model_root = args.work_root / '3dgs_models'
    args.log_root = args.work_root / '3dgs_logs'
    args.npz_root = args.out_root / 'npz'
    args.ply_root = args.out_root / 'ply'
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.npz_root.mkdir(parents=True, exist_ok=True)
    args.ply_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.dataset_root / 'manifests' / 'manifest.csv'
    all_rows = select_all_rows(load_manifest(manifest_path), args.dataset_root)
    rows = all_rows if args.mode == 'full' else all_rows[:9]
    write_json(
        args.out_root / ('full_selection.json' if args.mode == 'full' else 'pilot_selection_full_wrapper.json'),
        [
            {
                'scene_key': scene_key(row['category'], row['scene_id']),
                'split': row['split'],
                'category': row['category'],
                'scene_id': row['scene_id'],
                'instance_id': row['instance_id'],
                'num_rgb_images': row['num_rgb_images'],
            }
            for row in rows
        ],
    )
    print(f'Selected {len(rows)} scenes for mode={args.mode}', flush=True)

    missing_feature_rows = [row for row in rows if args.force_features or not compact_npz_path(args, row).exists()]
    print(f'Existing NPZ scenes: {len(rows) - len(missing_feature_rows)}; missing NPZ scenes: {len(missing_feature_rows)}', flush=True)
    if not args.skip_3dgs:
        train_3dgs(missing_feature_rows, args)

    debug_keys = choose_debug_keys(rows, args)
    print(f'Will write {len(debug_keys)} additional debug PLYs', flush=True)
    summaries = build_full_features(rows, args, debug_keys)
    summary_path = write_summaries(args, summaries, args.mode)
    print(f'Summary -> {summary_path}', flush=True)


if __name__ == '__main__':
    main()
