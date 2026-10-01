#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from extract_handal_dinov2_gaussian_embeddings import (  # noqa: E402
    load_dinov2,
    load_json,
    normalize_rows,
    project_means_and_covariances,
    rgb_to_sh,
)
from extract_exp40_b_dino_pilot import BAD_SCENES, NPZ_ROOT, camera_size, read_item, selected_frame_ids  # noqa: E402


OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "45_exp40_dino_assignment_visual_1percat_v1"


def stable_key(text: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def select_items(npz_root: Path, per_category: int, seed: int) -> list[dict]:
    items = [read_item(path) for path in sorted(npz_root.glob("*.npz"))]
    items = [item for item in items if item["scene_key"] not in BAD_SCENES]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)
    selected = []
    for category, group in sorted(by_category.items()):
        group = sorted(group, key=lambda item: stable_key(item["scene_key"], seed))
        selected.extend(group if per_category <= 0 else group[:per_category])
    return selected


def preprocess_pil(image: Image.Image, resize_width: int, resize_height: int, device: torch.device) -> torch.Tensor:
    image = image.convert("RGB").resize((resize_width, resize_height), Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def extract_patch_features_pil(model, image: Image.Image, resize_width: int, resize_height: int, patch_size: int, device, normalize: bool):
    pixel_values = preprocess_pil(image, resize_width, resize_height, device)
    output = model(pixel_values=pixel_values)
    tokens = output.last_hidden_state[:, 1:, :]
    feat_h = resize_height // patch_size
    feat_w = resize_width // patch_size
    features = tokens.reshape(1, feat_h, feat_w, -1).squeeze(0)
    if normalize:
        features = F.normalize(features, dim=-1)
    return features.float().cpu().numpy()


def add_center_patch_votes(
    feature_sum,
    weight_sum,
    visible_views,
    patch_features,
    valid,
    u,
    v,
    src_width,
    src_height,
    resize_width,
    resize_height,
    patch_size,
    crop_xyxy=None,
):
    feat_h, feat_w = patch_features.shape[:2]
    if crop_xyxy is not None:
        x0, y0, x1, y1 = crop_xyxy
        src_width = x1 - x0
        src_height = y1 - y0
        u = u - x0
        v = v - y0
        valid = valid & (u >= 0) & (u < src_width) & (v >= 0) & (v < src_height)

    px = np.floor(u * (resize_width / src_width) / patch_size).astype(np.int64)
    py = np.floor(v * (resize_height / src_height) / patch_size).astype(np.int64)
    keep = valid & (px >= 0) & (px < feat_w) & (py >= 0) & (py < feat_h)
    ids = np.flatnonzero(keep)
    if len(ids) == 0:
        return 0
    feature_sum[ids] += patch_features[py[ids], px[ids]]
    weight_sum[ids] += 1.0
    visible_views[ids] += 1
    return int(len(ids))


def object_crop_from_valid(u, v, valid, width, height, margin_frac):
    ids = np.flatnonzero(valid)
    if len(ids) < 8:
        return None
    x0 = float(np.percentile(u[ids], 1.0))
    x1 = float(np.percentile(u[ids], 99.0))
    y0 = float(np.percentile(v[ids], 1.0))
    y1 = float(np.percentile(v[ids], 99.0))
    margin = margin_frac * max(x1 - x0, y1 - y0, 1.0)
    x0 = max(0, int(np.floor(x0 - margin)))
    y0 = max(0, int(np.floor(y0 - margin)))
    x1 = min(width, int(np.ceil(x1 + margin)))
    y1 = min(height, int(np.ceil(y1 + margin)))
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return None
    return x0, y0, x1, y1


def extract_item(item, args, method_dir, model, patch_size, hidden_size, device):
    scene_key = item["scene_key"]
    out_path = method_dir / "features" / f"{scene_key}_{args.method}_features.npz"
    if out_path.exists() and not args.force:
        with np.load(out_path, allow_pickle=False) as z:
            return {
                "scene_key": scene_key,
                "category": item["category"],
                "gaussians": int(z["dino_features"].shape[0]),
                "valid_dino": int(z["valid_dino"].sum()),
                "valid_dino_ratio": float(z["valid_dino"].sum() / max(z["dino_features"].shape[0], 1)),
                "views": int(z["views"][0]),
                "features": str(out_path),
                "reused": True,
            }

    raw_scene = Path(item["source_scene_path"])
    object_ply = Path(item["source_object_ply"])
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")
    frame_ids = selected_frame_ids(raw_scene, args.max_images)
    if args.max_dino_views is not None:
        frame_ids = frame_ids[: args.max_dino_views]

    ply = PlyData.read(str(object_ply))
    vertices = ply["vertex"].data
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    cov_dummy = np.zeros((len(points), 3, 3), dtype=np.float64)

    feature_sum = np.zeros((len(points), hidden_size), dtype=np.float32)
    weight_sum = np.zeros(len(points), dtype=np.float32)
    visible_views = np.zeros(len(points), dtype=np.uint16)
    assigned_centers = np.zeros(len(points), dtype=np.uint32)
    crop_rows = []

    for view_idx, frame_id in enumerate(frame_ids, start=1):
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * 0.001
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        width, height = camera_size(cam, raw_scene, frame_id)
        u, v, z, _, valid = project_means_and_covariances(points, cov_dummy, rot, trans, cam_k, width, height, args.min_var_px)
        valid = valid & (z > 1e-5)

        image_path = raw_scene / "rgb" / f"{frame_id:06d}.jpg"
        image = Image.open(image_path).convert("RGB")
        crop = None
        source_for_dino = image
        if args.method == "object_crop_center_avg":
            crop = object_crop_from_valid(u, v, valid, width, height, args.crop_margin_frac)
            if crop is None:
                continue
            source_for_dino = image.crop(crop)

        patch_features = extract_patch_features_pil(
            model,
            source_for_dino,
            args.resize_width,
            args.resize_height,
            patch_size,
            device,
            args.normalize_patch_features,
        )
        before = weight_sum.copy()
        added = add_center_patch_votes(
            feature_sum,
            weight_sum,
            visible_views,
            patch_features,
            valid,
            u,
            v,
            width,
            height,
            args.resize_width,
            args.resize_height,
            patch_size,
            crop,
        )
        assigned_centers[(weight_sum - before) > 0] += 1
        if crop is not None:
            crop_rows.append({"frame_id": frame_id, "x0": crop[0], "y0": crop[1], "x1": crop[2], "y1": crop[3], "assigned": added})
        if view_idx == 1 or view_idx == len(frame_ids) or view_idx % args.log_view_every == 0:
            print(f"{scene_key} {args.method} view {view_idx:03d}/{len(frame_ids)} frame {frame_id:06d}: assigned={added}", flush=True)

    embeddings = np.divide(feature_sum, weight_sum[:, None], out=np.zeros_like(feature_sum), where=weight_sum[:, None] > 0)
    valid_dino = weight_sum >= args.min_total_weight
    if args.normalize_output_features and np.any(valid_dino):
        embeddings[valid_dino] = normalize_rows(embeddings[valid_dino]).astype(np.float32)
    dino_dtype = np.float16 if args.dino_dtype == "float16" else np.float32

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        dino_features=embeddings.astype(dino_dtype),
        valid_dino=valid_dino.astype(bool),
        dino_weight_sum=weight_sum.astype(np.float32),
        dino_visible_views=visible_views,
        center_assignments=assigned_centers,
        scene_key=np.array([scene_key]),
        category=np.array([item["category"]]),
        source_exp40_npz=np.array([item["npz"]]),
        source_object_ply=np.array([item["source_object_ply"]]),
        source_scene_path=np.array([item["source_scene_path"]]),
        method=np.array([args.method]),
        dino_model=np.array([args.model_name]),
        resize_width=np.array([args.resize_width], dtype=np.int32),
        resize_height=np.array([args.resize_height], dtype=np.int32),
        patch_size=np.array([patch_size], dtype=np.int32),
        views=np.array([len(frame_ids)], dtype=np.int32),
    )
    if crop_rows:
        save_csv(method_dir / "crop_boxes" / f"{scene_key}_crop_boxes.csv", crop_rows)
    return {
        "scene_key": scene_key,
        "category": item["category"],
        "gaussians": int(len(points)),
        "valid_dino": int(valid_dino.sum()),
        "valid_dino_ratio": float(valid_dino.sum() / max(len(points), 1)),
        "views": int(len(frame_ids)),
        "features": str(out_path),
        "reused": False,
    }


def fit_pca(feature_paths, max_samples):
    samples = []
    for path in feature_paths:
        with np.load(path, allow_pickle=False) as z:
            emb = z["dino_features"].astype(np.float32)
            valid = z["valid_dino"]
            if valid.any():
                samples.append(emb[valid])
    x = np.concatenate(samples, axis=0)
    if len(x) > max_samples:
        rng = np.random.default_rng(7)
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    mean = x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    components = vt[:3].astype(np.float32)
    projected = []
    for path in feature_paths:
        with np.load(path, allow_pickle=False) as z:
            emb = z["dino_features"].astype(np.float32)
            valid = z["valid_dino"]
            if valid.any():
                projected.append((emb[valid] - mean) @ components.T)
    projected = np.concatenate(projected, axis=0)
    lo = np.percentile(projected, 1.0, axis=0)
    hi = np.percentile(projected, 99.0, axis=0)
    return mean.astype(np.float32), components, lo.astype(np.float32), np.maximum(hi - lo, 1e-6).astype(np.float32)


def write_pca_ply(feature_path, out_ply, mean, components, lo, scale):
    with np.load(feature_path, allow_pickle=False) as z:
        emb = z["dino_features"].astype(np.float32)
        valid = z["valid_dino"]
        source_ply = Path(str(z["source_object_ply"][0]))
    rgb = np.clip(((emb - mean) @ components.T - lo) / scale, 0.0, 1.0).astype(np.float32)
    rgb[~valid] = np.array([0.08, 0.08, 0.08], dtype=np.float32)
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    elements = [PlyElement.describe(vertices, "vertex") if e.name == "vertex" else e for e in ply.elements]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(out_ply))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=Path, default=NPZ_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--method", choices=["full_center_avg", "object_crop_center_avg"], required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--per_category", type=int, default=1)
    parser.add_argument("--limit_scenes", type=int, default=None)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--summary_tag", default="")
    parser.add_argument("--model_name", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dino_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--resize_width", type=int, default=896)
    parser.add_argument("--resize_height", type=int, default=672)
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--max_dino_views", type=int, default=48)
    parser.add_argument("--min_var_px", type=float, default=0.25)
    parser.add_argument("--min_total_weight", type=float, default=1.0)
    parser.add_argument("--crop_margin_frac", type=float, default=0.20)
    parser.add_argument("--max_pca_samples", type=int, default=50000)
    parser.add_argument("--normalize_patch_features", action="store_true", default=True)
    parser.add_argument("--normalize_output_features", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--log_view_every", type=int, default=12)
    args = parser.parse_args()

    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")

    method_dir = args.out_root / f"{args.method}_patch{args.resize_width}x{args.resize_height}_views{args.max_dino_views}"
    method_dir.mkdir(parents=True, exist_ok=True)
    items = select_items(args.npz_root, args.per_category, args.seed)
    if args.limit_scenes is not None:
        items = items[: args.limit_scenes]
    if args.shard_count > 1:
        items = items[args.shard_index :: args.shard_count]
    summary_tag = args.summary_tag or (f"_shard{args.shard_index:02d}of{args.shard_count:02d}" if args.shard_count > 1 else "")
    save_csv(method_dir / "summary" / f"selected_scenes{summary_tag}.csv", items)
    print(
        f"[select] method={args.method} scenes={len(items)} categories={len({x['category'] for x in items})} "
        f"shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, patch_size, hidden_size = load_dinov2(args.model_name, device, args.local_files_only)
    rows = []
    for idx, item in enumerate(items, start=1):
        print(f"[{idx:03d}/{len(items):03d}] {args.method} extracting {item['scene_key']} ({item['category']})", flush=True)
        row = extract_item(item, args, method_dir, model, patch_size, hidden_size, device)
        rows.append(row)
        save_csv(method_dir / "summary" / f"summary{summary_tag}.csv", rows)
        print(f"{item['scene_key']}: valid {row['valid_dino']}/{row['gaussians']} ({row['valid_dino_ratio']:.1%})", flush=True)

    if args.no_pca:
        with (method_dir / "summary" / f"summary{summary_tag}.json").open("w") as handle:
            json.dump({"method": args.method, "scenes": rows, "args": vars(args)}, handle, indent=2, default=str)
        print(f"[done] {args.method} features_only={len(rows)}", flush=True)
        return

    feature_paths = [Path(row["features"]) for row in rows]
    mean, components, lo, scale = fit_pca(feature_paths, args.max_pca_samples)
    np.savez_compressed(method_dir / "summary" / "global_pca_projection.npz", mean=mean, components=components, color_low=lo, color_scale=scale)
    for row in rows:
        out_ply = method_dir / "pca_ply" / f"{row['scene_key']}_{args.method}_dino_pca.ply"
        write_pca_ply(Path(row["features"]), out_ply, mean, components, lo, scale)
        row["pca_ply"] = str(out_ply)
    save_csv(method_dir / "summary" / f"summary{summary_tag}.csv", rows)
    with (method_dir / "summary" / f"summary{summary_tag}.json").open("w") as handle:
        json.dump({"method": args.method, "scenes": rows, "args": vars(args)}, handle, indent=2, default=str)
    print(f"[done] {args.method} pca_dir={method_dir / 'pca_ply'}", flush=True)


if __name__ == "__main__":
    main()
