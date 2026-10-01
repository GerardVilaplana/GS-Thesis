#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

BASE = Path('/home/gvilaplana/GS-Thesis/Affordances')
SCRIPTS = BASE / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import build_handal_generalization_feature_pilot as pilot  # noqa: E402
import train_handal_exp44_pointnet_mean_feature_sweep as e44  # noqa: E402
import train_handal_17cat_graph_attention as g15  # noqa: E402
from extract_handal_dinov2_gaussian_embeddings import load_dinov2, load_json, normalize_rows, project_means_and_covariances  # noqa: E402
import extract_exp45_dino_assignment_visual as d45  # noqa: E402
import train_exp48_gnn_embeddings_pointnet_mean as e48  # noqa: E402

NPZ_GT = BASE / 'data/handal_exp40_15k_gt_features_v1/npz/B_center_ellipsoid20_strict75'
AUTO_ROOT = BASE / 'outputs/03_handle_generalization/39_pointnet_valtest_auto_qwen_sam_realm15k_v1'
OUT_ROOT = BASE / 'outputs/03_handle_generalization/49_exp39_auto_realm_e2e_handle_eval_v1'
PN_ROOT = BASE / 'outputs/03_handle_generalization/47_exp46_dino_pointnet_mean_feature_sweep_v1/object_crop_center_avg'
GNN_ROOT = BASE / 'outputs/03_handle_generalization/48_exp46_dino_gnn_embeddings_pointnet_mean_v1/object_crop_best'
AUTO_NPZ = OUT_ROOT / 'auto_npz/object_ellipsoid20_strict75'
DINO_DIR = OUT_ROOT / 'dino/object_crop_center_avg_patch896x672_views48'
C0 = 0.28209479177387814


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def scalar_str(z, key, default=''):
    if key not in z.files:
        return default
    v = z[key]
    return str(v[0] if getattr(v, 'shape', ()) else v)


def load_items() -> list[dict]:
    base_items = {x['scene_key']: x for x in e44.load_items(NPZ_GT)}
    out = []
    for key, item in base_items.items():
        if item['source_split'] not in {'val', 'test'}:
            continue
        with np.load(item['npz'], allow_pickle=False) as z:
            item = dict(item)
            item['source_scene_path'] = scalar_str(z, 'source_scene_path')
            item['source_exp40_npz'] = item['npz']
            item['auto_object_ply'] = str(AUTO_ROOT / key / 'final_ply/object_ellipsoid20_strict75_truecolor_15k.ply')
            item['auto_raw_object_ply'] = str(AUTO_ROOT / key / 'final_ply/object_raw_truecolor_15k.ply')
            item['auto_npz'] = str(AUTO_NPZ / f'{key}.npz')
        out.append(item)
    return sorted(out, key=lambda x: x['scene_key'])


def camera_size(cam: dict, raw_scene: Path, frame_id: int) -> tuple[int, int]:
    return d45.camera_size(cam, raw_scene, frame_id)


def selected_frame_ids(raw_scene: Path, max_images: int) -> list[int]:
    return d45.selected_frame_ids(raw_scene, max_images)


def handle_labels_for_ply(item: dict, object_ply: Path, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_scene = Path(item['source_scene_path'])
    scene_camera = pilot.load_json(raw_scene / 'scene_camera.json')
    scene_gt = pilot.load_json(raw_scene / 'scene_gt.json')
    ply = PlyData.read(str(object_ply))
    vertices = ply['vertex'].data
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float64)
    cov_world = pilot.gaussian_covariances(vertices)
    weighted_hits = np.zeros(len(points), dtype=np.float64)
    weighted_total = np.zeros(len(points), dtype=np.float64)
    visible = np.zeros(len(points), dtype=np.uint16)
    for frame_id in pilot.selected_frame_ids(raw_scene, args.max_images):
        mask_path = raw_scene / 'mask_parts' / f'{frame_id:06d}_000000_handle.png'
        if mask_path.exists():
            mask = pilot.load_mask(mask_path)
        else:
            cam0 = scene_camera[str(frame_id)]
            width0, height0 = pilot.camera_size(cam0, raw_scene, frame_id)
            mask = np.zeros((height0, width0), dtype=bool)
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt['cam_t_m2c'], dtype=np.float64) * 0.001
        cam_k = np.array(cam['cam_K'], dtype=np.float64)
        u, v, z, cov2d, valid = pilot.project_means_and_covariances(points, cov_world, rot, trans, cam_k, *pilot.camera_size(cam, raw_scene, frame_id), args.min_var_px)
        width, _ = pilot.camera_size(cam, raw_scene, frame_id)
        for idx in pilot.center_zbuffer_filter(u, v, z, valid, width):
            inside, total = pilot.footprint_overlap(mask, u[idx], v[idx], cov2d[idx], args.sigma_extent, args.max_radius)
            if total <= 0:
                continue
            visible[idx] += 1
            weighted_hits[idx] += inside
            weighted_total[idx] += total
    score = np.divide(weighted_hits, weighted_total, out=np.zeros(len(points), dtype=np.float64), where=weighted_total > 0).astype(np.float32)
    labels = ((visible >= args.handle_min_visible) & (score >= args.handle_threshold)).astype(np.uint8)
    return score, labels, visible


def prepare_auto_npz(item: dict, args) -> dict:
    out = Path(item['auto_npz'])
    if out.exists() and not args.force:
        with np.load(out, allow_pickle=False) as z:
            return {'scene_key': item['scene_key'], 'category': item['category'], 'split': item['source_split'], 'gaussians': int(z['xyz'].shape[0]), 'num_handle': int(z['handle_labels_thr0_25'].sum()), 'npz': str(out), 'reused_npz': True}
    object_ply = Path(item['auto_object_ply'])
    if not object_ply.exists():
        raise FileNotFoundError(object_ply)
    vertices = PlyData.read(str(object_ply))['vertex'].data
    xyz, scale, rotation, opacity, color, geometry = pilot.geometry_arrays(vertices)
    handle_scores, handle_labels, handle_visible = handle_labels_for_ply(item, object_ply, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        geometry_features=geometry.astype(np.float32),
        xyz=xyz.astype(np.float32),
        scale=scale.astype(np.float32),
        rotation=rotation.astype(np.float32),
        opacity=opacity.astype(np.float32),
        color=color.astype(np.float32),
        handle_scores=handle_scores.astype(np.float32),
        handle_labels_thr0_25=handle_labels.astype(np.uint8),
        handle_visible_views=handle_visible,
        scene_key=np.array([item['scene_key']]),
        scene_id=np.array([item['scene_id']]),
        category=np.array([item['category']]),
        model_split=np.array([item['source_split']]),
        raw_split=np.array([item.get('raw_split', '')]),
        instance_id=np.array([item.get('instance_id', '')]),
        source_scene_path=np.array([item['source_scene_path']]),
        source_exp40_npz=np.array([item['source_exp40_npz']]),
        source_object_ply=np.array([str(object_ply)]),
        source_auto_raw_object_ply=np.array([item['auto_raw_object_ply']]),
        handle_threshold=np.array([args.handle_threshold], dtype=np.float32),
        handle_min_visible=np.array([args.handle_min_visible], dtype=np.int32),
        variant=np.array(['exp39_auto_realm_ellipsoid20_strict75']),
    )
    return {'scene_key': item['scene_key'], 'category': item['category'], 'split': item['source_split'], 'gaussians': int(len(xyz)), 'num_handle': int(handle_labels.sum()), 'npz': str(out), 'reused_npz': False}


@torch.no_grad()
def extract_auto_dino(item: dict, args, model, patch_size: int, hidden_size: int, device) -> dict:
    scene_key = item['scene_key']
    out_path = DINO_DIR / 'features' / f'{scene_key}_object_crop_center_avg_features.npz'
    if out_path.exists() and not args.force:
        with np.load(out_path, allow_pickle=False) as z:
            return {'scene_key': scene_key, 'category': item['category'], 'split': item['source_split'], 'gaussians': int(z['dino_features'].shape[0]), 'valid_dino': int(z['valid_dino'].sum()), 'valid_dino_ratio': float(z['valid_dino'].mean()), 'features': str(out_path), 'reused_dino': True}
    raw_scene = Path(item['source_scene_path'])
    object_ply = Path(item['auto_object_ply'])
    scene_camera = load_json(raw_scene / 'scene_camera.json')
    scene_gt = load_json(raw_scene / 'scene_gt.json')
    frame_ids = selected_frame_ids(raw_scene, args.max_images)
    if args.max_dino_views is not None:
        frame_ids = frame_ids[:args.max_dino_views]
    ply = PlyData.read(str(object_ply))
    vertices = ply['vertex'].data
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float64)
    cov_dummy = np.zeros((len(points), 3, 3), dtype=np.float64)
    feature_sum = np.zeros((len(points), hidden_size), dtype=np.float32)
    weight_sum = np.zeros(len(points), dtype=np.float32)
    visible_views = np.zeros(len(points), dtype=np.uint16)
    assigned_centers = np.zeros(len(points), dtype=np.uint32)
    crop_rows = []
    from PIL import Image
    for view_idx, frame_id in enumerate(frame_ids, start=1):
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt['cam_t_m2c'], dtype=np.float64) * 0.001
        cam_k = np.array(cam['cam_K'], dtype=np.float64)
        width, height = camera_size(cam, raw_scene, frame_id)
        u, v, z, _, valid = project_means_and_covariances(points, cov_dummy, rot, trans, cam_k, width, height, args.min_var_px)
        valid = valid & (z > 1e-5)
        crop = d45.object_crop_from_valid(u, v, valid, width, height, args.crop_margin_frac)
        if crop is None:
            continue
        image = Image.open(raw_scene / 'rgb' / f'{frame_id:06d}.jpg').convert('RGB')
        patch_features = d45.extract_patch_features_pil(model, image.crop(crop), args.resize_width, args.resize_height, patch_size, device, args.normalize_patch_features)
        before = weight_sum.copy()
        added = d45.add_center_patch_votes(feature_sum, weight_sum, visible_views, patch_features, valid, u, v, width, height, args.resize_width, args.resize_height, patch_size, crop)
        assigned_centers[(weight_sum - before) > 0] += 1
        crop_rows.append({'frame_id': frame_id, 'x0': crop[0], 'y0': crop[1], 'x1': crop[2], 'y1': crop[3], 'assigned': added})
        if view_idx == 1 or view_idx == len(frame_ids) or view_idx % args.log_view_every == 0:
            print(f'{scene_key} dino view {view_idx:03d}/{len(frame_ids)} frame {frame_id:06d}: assigned={added}', flush=True)
    emb = np.divide(feature_sum, weight_sum[:, None], out=np.zeros_like(feature_sum), where=weight_sum[:, None] > 0)
    valid_dino = weight_sum >= args.min_total_weight
    if args.normalize_output_features and np.any(valid_dino):
        emb[valid_dino] = normalize_rows(emb[valid_dino]).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.dino_dtype == 'float16' else np.float32
    np.savez_compressed(
        out_path,
        dino_features=emb.astype(dtype),
        valid_dino=valid_dino.astype(bool),
        dino_weight_sum=weight_sum.astype(np.float32),
        dino_visible_views=visible_views,
        center_assignments=assigned_centers,
        scene_key=np.array([scene_key]),
        category=np.array([item['category']]),
        source_auto_npz=np.array([item['auto_npz']]),
        source_object_ply=np.array([item['auto_object_ply']]),
        source_scene_path=np.array([item['source_scene_path']]),
        method=np.array(['object_crop_center_avg']),
        resize_width=np.array([args.resize_width], dtype=np.int32),
        resize_height=np.array([args.resize_height], dtype=np.int32),
        patch_size=np.array([patch_size], dtype=np.int32),
        views=np.array([len(frame_ids)], dtype=np.int32),
    )
    if crop_rows:
        save_csv(DINO_DIR / 'crop_boxes' / f'{scene_key}_crop_boxes.csv', crop_rows)
    return {'scene_key': scene_key, 'category': item['category'], 'split': item['source_split'], 'gaussians': int(len(points)), 'valid_dino': int(valid_dino.sum()), 'valid_dino_ratio': float(valid_dino.mean()), 'features': str(out_path), 'reused_dino': False}


def prepare(args) -> None:
    items = load_items()
    if args.limit_scenes:
        items = items[:args.limit_scenes]
    if args.shard_count > 1:
        items = items[args.shard_index::args.shard_count]
    DINO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    model, patch_size, hidden_size = load_dinov2(args.model_name, device, args.local_files_only)
    for idx, item in enumerate(items, start=1):
        print(f'[prepare {idx:03d}/{len(items):03d}] {item["scene_key"]}', flush=True)
        npz_row = prepare_auto_npz(item, args)
        item['auto_npz'] = npz_row['npz']
        dino_row = extract_auto_dino(item, args, model, patch_size, hidden_size, device)
        rows.append({**npz_row, **{f'dino_{k}': v for k, v in dino_row.items() if k not in {'scene_key','category','split','gaussians'}}})
        tag = f'_shard{args.shard_index:02d}of{args.shard_count:02d}' if args.shard_count > 1 else ''
        save_csv(OUT_ROOT / 'prepare_summary' / f'prepare_summary{tag}.csv', rows)
    print(f'[prepare done] scenes={len(rows)}', flush=True)


def auto_items_for_eval() -> list[dict]:
    items = load_items()
    out = []
    for item in items:
        npz = AUTO_NPZ / f"{item['scene_key']}.npz"
        dino = DINO_DIR / 'features' / f"{item['scene_key']}_object_crop_center_avg_features.npz"
        if item['source_split'] in {'val','test'}:
            if not npz.exists() or not dino.exists():
                raise FileNotFoundError(f'missing prepared auto features for {item["scene_key"]}: {npz.exists()} {dino.exists()}')
            with np.load(npz, allow_pickle=False) as z:
                y = z['handle_labels_thr0_25']
            it = dict(item)
            it['npz'] = str(npz)
            it['source_object_ply'] = str(AUTO_ROOT / item['scene_key'] / 'final_ply/object_ellipsoid20_strict75_truecolor_15k.ply')
            it['num_gaussians'] = int(len(y)); it['num_handle'] = int(y.sum())
            out.append(it)
    return sorted(out, key=lambda x: x['scene_key'])


def set_auto_feature_context():
    e44.ACTIVE_FEATURE_VARIANT = 'dino_geometry_color_quality'
    e44.ACTIVE_DINO_ROOT = DINO_DIR


def load_features(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_auto_feature_context()
    x, y = e44.load_feature_arrays_variant(item)
    with np.load(item['npz'], allow_pickle=False) as z:
        xyz_norm = g15.base.normalize_xyz(z['xyz'].astype(np.float32))
    return x.astype(np.float32), y.astype(np.float32), xyz_norm.astype(np.float32)


def scenes_from_items(items: list[dict], mean: np.ndarray, std: np.ndarray, graph=False, max_k=32) -> list[dict]:
    scenes = []
    for item in items:
        x, y, xyz_norm = load_features(item)
        scene = {'item': item, 'x': ((x - mean) / std).astype(np.float32), 'y': y}
        if graph:
            scene['knn'] = g15.build_knn(xyz_norm, max_k)
        scenes.append(scene)
    return scenes


def predict_pn(ckpt_path: Path, items: list[dict], device) -> tuple[np.ndarray, np.ndarray, list]:
    ck = torch.load(ckpt_path, map_location='cpu')
    model = e44.PointNet(int(ck['input_dim']), 'mean', int(ck['latent_dim']), 0.0).to(device)
    model.load_state_dict(ck['model']); model.eval()
    scenes = scenes_from_items(items, ck['mean'].astype(np.float32), ck['std'].astype(np.float32), graph=False)
    return e44.predict_scenes(model, scenes, device)


def predict_gnn(ckpt_path: Path, items: list[dict], device) -> tuple[np.ndarray, np.ndarray, list]:
    ck = torch.load(ckpt_path, map_location='cpu')
    in_dim = int(ck['mean'].shape[1])
    model = e48.GraphEmbeddingNet(in_dim, int(ck.get('latent_dim', 192)), tuple(ck.get('k_layers', [16, 32])), 0.0).to(device)
    model.load_state_dict(ck['model']); model.eval()
    max_k = max(int(k) for k in ck.get('k_layers', [16, 32]))
    scenes = scenes_from_items(items, ck['mean'].astype(np.float32), ck['std'].astype(np.float32), graph=True, max_k=max_k)
    return e48.predict_gnn(model, scenes, device)


def choose_auto_threshold(y_val, val_score, val_slices):
    best, curve = g15.choose_threshold(y_val, val_score, val_slices)
    return float(best['threshold']), curve


def metric_bundle(y, score, slices, threshold) -> tuple[list[dict], dict, dict]:
    rows = g15.per_scene_metrics(slices, y, score, threshold)
    micro = g15.metrics_from_scores(y, score, threshold)
    macro = {k: g15.macro_scene_score(rows, k) for k in g15.METRIC_NAMES}
    return rows, micro, macro


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def qa_colors(labels, scores, threshold):
    labels = labels.astype(bool); pred = scores >= threshold
    rgb = np.zeros((len(labels), 3), dtype=np.float32)
    rgb[(~labels) & (~pred)] = np.array([0.02, 0.16, 1.0], dtype=np.float32)
    rgb[labels & pred] = np.array([0.02, 0.85, 0.12], dtype=np.float32)
    rgb[labels & (~pred)] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    rgb[(~labels) & pred] = np.array([1.0, 0.48, 0.02], dtype=np.float32)
    return rgb


def write_colored_ply(src: Path, out: Path, rgb: np.ndarray) -> None:
    ply = PlyData.read(str(src)); vertices = np.array(ply['vertex'].data, copy=True)
    if len(vertices) != len(rgb):
        raise ValueError(f'length mismatch {src}: {len(vertices)} vs {len(rgb)}')
    sh = rgb_to_sh(rgb)
    vertices['f_dc_0'] = sh[:, 0]; vertices['f_dc_1'] = sh[:, 1]; vertices['f_dc_2'] = sh[:, 2]
    elems = [PlyElement.describe(vertices, 'vertex') if e.name == 'vertex' else e for e in ply.elements]
    out.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elems, text=ply.text, byte_order=ply.byte_order).write(str(out))


def selected_visual_keys(items: list[dict], per_cat: int) -> set[str]:
    by = defaultdict(list)
    for it in sorted([x for x in items if x['source_split'] == 'test'], key=lambda x: x['scene_key']):
        by[it['category']].append(it)
    keys = set()
    for cat, group in sorted(by.items()):
        for it in group[:per_cat]:
            keys.add(it['scene_key'])
    return keys


def export_visuals(run_dir: Path, run_label: str, eval_scope: str, rows: list[dict], slices: list, y: np.ndarray, scores: np.ndarray, threshold: float, visual_keys: set[str]) -> list[dict]:
    by_key = {item['scene_key']: (item, start, end) for item, start, end in slices}
    row_by_key = {r['scene_key']: r for r in rows}
    manifest = []
    for key in sorted(visual_keys & set(by_key)):
        item, start, end = by_key[key]
        row = row_by_key.get(key, {})
        out = OUT_ROOT / 'visual_prediction_plys' / key / f'{run_label}__{eval_scope}.ply'
        write_colored_ply(Path(item['source_object_ply']), out, qa_colors(y[start:end], scores[start:end], threshold))
        manifest.append({'scene_key': key, 'category': item['category'], 'run_label': run_label, 'eval_scope': eval_scope, 'iou': row.get('iou', ''), 'f1': row.get('f1', ''), 'threshold': threshold, 'output_ply': str(out), 'source_object_ply': item['source_object_ply']})
    return manifest


def checkpoint_path(model_kind: str, scope: str, category: str | None) -> Path:
    if model_kind == 'pn':
        if scope == 'seen':
            return PN_ROOT / '01_seen_instance_17cat/dino_geometry_color_quality/pointnet_mean/model.pt'
        return PN_ROOT / '02_leave_one_category_out_17cat' / category / 'dino_geometry_color_quality/pointnet_mean/model.pt'
    if scope == 'seen':
        return GNN_ROOT / '01_seen_instance_17cat/object_crop_best/gnn_embeddings_pointnet_mean/gnn_model.pt'
    return GNN_ROOT / '02_leave_one_category_out_17cat' / category / 'object_crop_best/gnn_embeddings_pointnet_mean/gnn_model.pt'


def gt_threshold(ckpt_path: Path, model_kind: str) -> float:
    ck = torch.load(ckpt_path, map_location='cpu')
    if model_kind == 'pn':
        return float(ck['threshold'])
    return float(ck['overall']['selected_threshold'])


def run_model_eval(model_kind: str, threshold_mode: str, scope: str, items: list[dict], visual_keys: set[str], device) -> dict:
    model_name = 'pointnet_mean_crop_dino_geom_quality' if model_kind == 'pn' else 'direct_gnn_crop_dino_geom_quality'
    run_label = f'{model_kind}_{threshold_mode}_{scope}'
    all_rows = []
    all_cat_rows = []
    all_visuals = []
    overall_runs = []
    if scope == 'seen':
        ckpt = checkpoint_path(model_kind, 'seen', None)
        val_items = [x for x in items if x['source_split'] == 'val']
        test_items = [x for x in items if x['source_split'] == 'test']
        predictor = predict_pn if model_kind == 'pn' else predict_gnn
        if threshold_mode == 'auto_val':
            yv, sv, slv = predictor(ckpt, val_items, device)
            threshold, curve = choose_auto_threshold(yv, sv, slv)
        else:
            threshold = gt_threshold(ckpt, model_kind); curve = []
        y, s, sl = predictor(ckpt, test_items, device)
        rows, micro, macro = metric_bundle(y, s, sl, threshold)
        for r in rows:
            r.update({'model_kind': model_kind, 'model_name': model_name, 'threshold_mode': threshold_mode, 'scope': scope, 'heldout_category': ''})
        cat_rows = g15.per_category_metrics(rows)
        for r in cat_rows:
            r.update({'model_kind': model_kind, 'model_name': model_name, 'threshold_mode': threshold_mode, 'scope': scope, 'heldout_category': ''})
        run_dir = OUT_ROOT / 'metrics' / run_label
        save_csv(run_dir / 'per_scene_metrics.csv', rows)
        save_csv(run_dir / 'per_category_metrics.csv', cat_rows)
        save_csv(run_dir / 'threshold_curve.csv', curve)
        all_visuals += export_visuals(run_dir, f'{model_kind}_{threshold_mode}', 'seen', rows, sl, y, s, threshold, visual_keys)
        overall = {'run_label': run_label, 'model_kind': model_kind, 'model_name': model_name, 'threshold_mode': threshold_mode, 'scope': scope, 'threshold': threshold, 'test_scenes': len(test_items), 'micro': micro, 'macro_scene_average': macro, 'checkpoint': str(ckpt)}
        (run_dir / 'overall_metrics.json').write_text(json.dumps(overall, indent=2))
        all_rows += rows; all_cat_rows += cat_rows; overall_runs.append(overall)
    else:
        cats = sorted({x['category'] for x in items})
        predictor = predict_pn if model_kind == 'pn' else predict_gnn
        run_dir = OUT_ROOT / 'metrics' / run_label
        for cat in cats:
            ckpt = checkpoint_path(model_kind, 'loco', cat)
            val_items = [x for x in items if x['source_split'] == 'val' and x['category'] != cat]
            test_items = [x for x in items if x['source_split'] == 'test' and x['category'] == cat]
            if threshold_mode == 'auto_val':
                yv, sv, slv = predictor(ckpt, val_items, device)
                threshold, curve = choose_auto_threshold(yv, sv, slv)
            else:
                threshold = gt_threshold(ckpt, model_kind); curve = []
            y, s, sl = predictor(ckpt, test_items, device)
            rows, micro, macro = metric_bundle(y, s, sl, threshold)
            for r in rows:
                r.update({'model_kind': model_kind, 'model_name': model_name, 'threshold_mode': threshold_mode, 'scope': scope, 'heldout_category': cat})
            all_rows += rows
            all_visuals += export_visuals(run_dir, f'{model_kind}_{threshold_mode}', 'loco', rows, sl, y, s, threshold, visual_keys)
            overall = {'run_label': run_label, 'model_kind': model_kind, 'model_name': model_name, 'threshold_mode': threshold_mode, 'scope': scope, 'heldout_category': cat, 'threshold': threshold, 'test_scenes': len(test_items), 'micro': micro, 'macro_scene_average': macro, 'checkpoint': str(ckpt)}
            (run_dir / cat / 'overall_metrics.json').parent.mkdir(parents=True, exist_ok=True)
            (run_dir / cat / 'overall_metrics.json').write_text(json.dumps(overall, indent=2))
            save_csv(run_dir / cat / 'threshold_curve.csv', curve)
            overall_runs.append(overall)
        cat_rows = g15.per_category_metrics(all_rows)
        for r in cat_rows:
            r.update({'model_kind': model_kind, 'model_name': model_name, 'threshold_mode': threshold_mode, 'scope': scope})
        all_cat_rows = cat_rows
        save_csv(run_dir / 'per_scene_metrics.csv', all_rows)
        save_csv(run_dir / 'per_category_metrics.csv', all_cat_rows)
        macro_all = {k: g15.macro_scene_score(all_rows, k) for k in g15.METRIC_NAMES}
        overall_all = {'run_label': run_label, 'model_kind': model_kind, 'model_name': model_name, 'threshold_mode': threshold_mode, 'scope': scope, 'test_scenes': len({r['scene_key'] for r in all_rows}), 'macro_scene_average': macro_all, 'category_runs': overall_runs}
        (run_dir / 'overall_metrics.json').write_text(json.dumps(overall_all, indent=2))
    save_csv(OUT_ROOT / 'visual_prediction_plys' / f'manifest_{run_label}.csv', all_visuals)
    return {'run_label': run_label, 'model_kind': model_kind, 'model_name': model_name, 'threshold_mode': threshold_mode, 'scope': scope, 'num_scene_rows': len(all_rows), 'macro_iou': g15.macro_scene_score(all_rows, 'iou'), 'macro_f1': g15.macro_scene_score(all_rows, 'f1'), 'macro_auprc': g15.macro_scene_score(all_rows, 'auprc'), 'macro_roc_auc': g15.macro_scene_score(all_rows, 'roc_auc')}


def evaluate(args) -> None:
    items = auto_items_for_eval()
    if args.limit_scenes:
        keep = {x['scene_key'] for x in items[:args.limit_scenes]}
        items = [x for x in items if x['scene_key'] in keep]
    visual_keys = selected_visual_keys(items, args.visual_per_category)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    jobs = []
    model_kinds = ['pn', 'gnn'] if args.model_kind == 'all' else [args.model_kind]
    threshold_modes = ['gt_val', 'auto_val'] if args.threshold_mode == 'all' else [args.threshold_mode]
    scopes = ['seen', 'loco'] if args.scope == 'all' else [args.scope]
    for mk in model_kinds:
        for tm in threshold_modes:
            for sc in scopes:
                if args.shard_count > 1:
                    flat = [(m,t,s) for m in model_kinds for t in threshold_modes for s in scopes]
                    if flat.index((mk,tm,sc)) % args.shard_count != args.shard_index:
                        continue
                print(f'[eval] model={mk} threshold={tm} scope={sc}', flush=True)
                jobs.append(run_model_eval(mk, tm, sc, items, visual_keys, device))
                save_csv(OUT_ROOT / 'metrics' / f'eval_summary_shard{args.shard_index:02d}of{args.shard_count:02d}.csv', jobs)
    all_rows = []
    for p in sorted((OUT_ROOT / 'metrics').glob('**/overall_metrics.json')):
        try:
            d = json.loads(p.read_text())
            if 'category_runs' in d or d.get('scope') == 'seen':
                all_rows.append({'run_label': d.get('run_label',''), 'model_kind': d.get('model_kind',''), 'threshold_mode': d.get('threshold_mode',''), 'scope': d.get('scope',''), 'test_scenes': d.get('test_scenes',''), 'macro_iou': d['macro_scene_average']['iou'], 'macro_f1': d['macro_scene_average']['f1'], 'macro_auprc': d['macro_scene_average']['auprc'], 'macro_roc_auc': d['macro_scene_average']['roc_auc']})
        except Exception:
            pass
    save_csv(OUT_ROOT / 'metrics' / 'all_finished_eval_summary.csv', all_rows)


def coverage(args) -> None:
    items = load_items()
    rows = []
    for item in items:
        npz = AUTO_NPZ / f"{item['scene_key']}.npz"
        dino = DINO_DIR / 'features' / f"{item['scene_key']}_object_crop_center_avg_features.npz"
        rows.append({'scene_key': item['scene_key'], 'category': item['category'], 'split': item['source_split'], 'auto_object_ply': Path(item['auto_object_ply']).exists(), 'auto_npz': npz.exists(), 'dino_features': dino.exists()})
    save_csv(OUT_ROOT / 'prepare_summary/coverage.csv', rows)
    print(json.dumps({'rows': len(rows), 'auto_object_ply': sum(r['auto_object_ply'] for r in rows), 'auto_npz': sum(r['auto_npz'] for r in rows), 'dino_features': sum(r['dino_features'] for r in rows)}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['prepare','eval','coverage'], required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--shard_count', type=int, default=1)
    ap.add_argument('--shard_index', type=int, default=0)
    ap.add_argument('--limit_scenes', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--model_name', default='facebook/dinov2-small')
    ap.add_argument('--local_files_only', action='store_true')
    ap.add_argument('--resize_width', type=int, default=896)
    ap.add_argument('--resize_height', type=int, default=672)
    ap.add_argument('--max_images', type=int, default=96)
    ap.add_argument('--max_dino_views', type=int, default=48)
    ap.add_argument('--min_var_px', type=float, default=0.25)
    ap.add_argument('--crop_margin_frac', type=float, default=0.20)
    ap.add_argument('--min_total_weight', type=float, default=1.0)
    ap.add_argument('--dino_dtype', choices=['float16','float32'], default='float16')
    ap.add_argument('--normalize_patch_features', action='store_true', default=True)
    ap.add_argument('--normalize_output_features', action='store_true', default=True)
    ap.add_argument('--log_view_every', type=int, default=12)
    ap.add_argument('--handle_threshold', type=float, default=0.25)
    ap.add_argument('--handle_min_visible', type=int, default=10)
    ap.add_argument('--sigma_extent', type=float, default=3.0)
    ap.add_argument('--max_radius', type=int, default=24)
    ap.add_argument('--model_kind', choices=['pn','gnn','all'], default='all')
    ap.add_argument('--threshold_mode', choices=['gt_val','auto_val','all'], default='all')
    ap.add_argument('--scope', choices=['seen','loco','all'], default='all')
    ap.add_argument('--visual_per_category', type=int, default=2)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'eval':
        evaluate(args)
    else:
        coverage(args)


if __name__ == '__main__':
    main()
