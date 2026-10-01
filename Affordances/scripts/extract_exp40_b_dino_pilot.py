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
from plyfile import PlyData, PlyElement


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from extract_handal_dinov2_gaussian_embeddings import (  # noqa: E402
    C0,
    accumulate_weighted_features,
    extract_patch_features,
    gaussian_covariances,
    load_dinov2,
    load_json,
    make_view_contributions,
    normalize_rows,
    project_means_and_covariances,
    rgb_to_sh,
    safe_name,
    sigmoid,
)


NPZ_ROOT = BASE_ROOT / "data" / "handal_exp40_15k_gt_features_v1" / "npz" / "B_center_ellipsoid20_strict75"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "42_exp40_b_dino_pilot_2percat_v1"

BAD_SCENES = {
    "spatulas__032005",
    "slip_joint_pliers__090006",
    "slip_joint_pliers__092005",
    "locking_pliers__040003",
    "locking_pliers__091004",
    "utensils__022007",
    "slip_joint_pliers__093004",
    "locking_pliers__090004",
    "utensils__026007",
    "locking_pliers__041005",
}


def stable_key(text: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def scalar_str(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    value = z[key]
    return str(value[0] if getattr(value, "shape", ()) else value)


def read_item(path: Path) -> dict:
    category, scene_id = path.stem.rsplit("__", 1)
    with np.load(path, allow_pickle=False) as z:
        return {
            "scene_key": scalar_str(z, "scene_key", path.stem),
            "category": scalar_str(z, "category", category),
            "scene_id": scalar_str(z, "scene_id", scene_id),
            "instance_id": scalar_str(z, "instance_id", scene_id),
            "model_split": scalar_str(z, "model_split", "unknown"),
            "raw_split": scalar_str(z, "raw_split", "unknown"),
            "npz": str(path),
            "source_scene_path": scalar_str(z, "source_scene_path"),
            "source_object_ply": scalar_str(z, "source_object_ply"),
            "num_gaussians": int(z["xyz"].shape[0]),
            "num_handle": int(z["handle_labels_thr0_25"].sum()),
        }


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


def selected_frame_ids(raw_scene: Path, max_images: int | None) -> list[int]:
    rgb_dir = raw_scene / "rgb"
    frames = sorted(int(path.stem) for path in rgb_dir.glob("*.jpg"))
    if max_images is not None:
        frames = frames[:max_images]
    return frames


def camera_size(cam: dict, raw_scene: Path, frame_id: int) -> tuple[int, int]:
    if "width" in cam and "height" in cam:
        return int(cam["width"]), int(cam["height"])
    from PIL import Image

    with Image.open(raw_scene / "rgb" / f"{frame_id:06d}.jpg") as im:
        return im.size


@torch.no_grad()
def extract_dino_for_item(item: dict, args, method_dir: Path, model, patch_size: int, hidden_size: int, device):
    scene_key = item["scene_key"]
    feature_path = method_dir / "features" / f"{scene_key}_dinov2_render_contrib_features.npz"
    if feature_path.exists() and not args.force:
        with np.load(feature_path, allow_pickle=False) as z:
            return {
                "scene_key": scene_key,
                "category": item["category"],
                "model_split": item["model_split"],
                "raw_split": item["raw_split"],
                "gaussians": int(z["dino_features"].shape[0]),
                "valid_dino": int(z["valid_dino"].sum()),
                "valid_dino_ratio": float(z["valid_dino"].sum() / max(z["dino_features"].shape[0], 1)),
                "views": int(z["views"][0]),
                "features": str(feature_path),
                "source_object_ply": item["source_object_ply"],
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
    cov_world = gaussian_covariances(vertices)
    opacities = sigmoid(np.asarray(vertices["opacity"], dtype=np.float64))

    feature_sum = np.zeros((len(points), hidden_size), dtype=np.float32)
    weight_sum = np.zeros(len(points), dtype=np.float32)
    contribution_count = np.zeros(len(points), dtype=np.uint32)
    visible_views = np.zeros(len(points), dtype=np.uint16)

    for view_idx, frame_id in enumerate(frame_ids, start=1):
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * 0.001
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        width, height = camera_size(cam, raw_scene, frame_id)
        image_path = raw_scene / "rgb" / f"{frame_id:06d}.jpg"

        patch_features, original_width, original_height = extract_patch_features(
            model,
            image_path,
            args.resize_width,
            args.resize_height,
            patch_size,
            device,
            args.normalize_patch_features,
        )
        feat_h, feat_w = patch_features.shape[:2]
        u, v, z, cov2d, valid = project_means_and_covariances(
            points, cov_world, rot, trans, cam_k, width, height, args.min_var_px
        )
        contribs = make_view_contributions(
            u,
            v,
            z,
            cov2d,
            valid,
            opacities,
            original_width,
            original_height,
            feat_w,
            feat_h,
            args.resize_width,
            args.resize_height,
            patch_size,
            args.sigma_extent,
            args.max_patch_radius,
            args.min_patch_radius,
            args.min_alpha,
            args.alpha_clip,
            args.min_var_patch,
        )
        if contribs is None:
            print(f"{scene_key} DINO view {view_idx:03d}/{len(frame_ids)} frame {frame_id:06d}: no contributions", flush=True)
            continue
        before = weight_sum.copy()
        patch_ids, gaussian_ids, weights = contribs
        accumulate_weighted_features(feature_sum, weight_sum, contribution_count, patch_features, patch_ids, gaussian_ids, weights)
        visible_views[(weight_sum - before) > args.min_view_weight] += 1
        if args.verbose_views or view_idx == 1 or view_idx == len(frame_ids) or view_idx % args.log_view_every == 0:
            print(
                f"{scene_key} DINO view {view_idx:03d}/{len(frame_ids)} frame {frame_id:06d}: "
                f"{len(weights)} weighted contributions",
                flush=True,
            )

    embeddings = np.divide(feature_sum, weight_sum[:, None], out=np.zeros_like(feature_sum), where=weight_sum[:, None] > 0)
    valid_dino = weight_sum >= args.min_total_weight
    if args.normalize_output_features and np.any(valid_dino):
        embeddings[valid_dino] = normalize_rows(embeddings[valid_dino]).astype(np.float32)
    dino_dtype = np.float16 if args.dino_dtype == "float16" else np.float32

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        dino_features=embeddings.astype(dino_dtype),
        valid_dino=valid_dino.astype(bool),
        dino_weight_sum=weight_sum.astype(np.float32),
        dino_contribution_count=contribution_count,
        dino_visible_views=visible_views,
        scene_key=np.array([scene_key]),
        category=np.array([item["category"]]),
        scene_id=np.array([item["scene_id"]]),
        instance_id=np.array([item["instance_id"]]),
        model_split=np.array([item["model_split"]]),
        raw_split=np.array([item["raw_split"]]),
        source_exp40_npz=np.array([item["npz"]]),
        source_scene_path=np.array([item["source_scene_path"]]),
        source_object_ply=np.array([item["source_object_ply"]]),
        dino_model=np.array([args.model_name]),
        dino_dtype=np.array([args.dino_dtype]),
        resize_width=np.array([args.resize_width], dtype=np.int32),
        resize_height=np.array([args.resize_height], dtype=np.int32),
        patch_size=np.array([patch_size], dtype=np.int32),
        views=np.array([len(frame_ids)], dtype=np.int32),
        sigma_extent=np.array([args.sigma_extent], dtype=np.float32),
        max_patch_radius=np.array([args.max_patch_radius], dtype=np.int32),
        min_patch_radius=np.array([args.min_patch_radius], dtype=np.int32),
        min_total_weight=np.array([args.min_total_weight], dtype=np.float32),
    )
    return {
        "scene_key": scene_key,
        "category": item["category"],
        "model_split": item["model_split"],
        "raw_split": item["raw_split"],
        "gaussians": int(len(points)),
        "valid_dino": int(valid_dino.sum()),
        "valid_dino_ratio": float(valid_dino.sum() / max(len(points), 1)),
        "views": int(len(frame_ids)),
        "features": str(feature_path),
        "source_object_ply": item["source_object_ply"],
        "reused": False,
    }


def fit_pca(feature_paths: list[Path], max_samples: int):
    samples = []
    for path in feature_paths:
        with np.load(path, allow_pickle=False) as z:
            emb = z["dino_features"].astype(np.float32)
            valid = z["valid_dino"]
            if valid.any():
                samples.append(emb[valid])
    if not samples:
        raise RuntimeError("No valid DINO features for PCA.")
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
    scale = np.maximum(hi - lo, 1e-6)
    return mean.astype(np.float32), components, lo.astype(np.float32), scale.astype(np.float32)


def write_pca_ply(feature_path: Path, out_ply: Path, mean, components, lo, scale) -> None:
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
    elements = []
    for element in ply.elements:
        elements.append(PlyElement.describe(vertices, "vertex") if element.name == "vertex" else element)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(out_ply))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=Path, default=NPZ_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--per_category", type=int, default=2)
    parser.add_argument("--limit_scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--model_name", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dino_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--resize_width", type=int, default=448)
    parser.add_argument("--resize_height", type=int, default=336)
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--max_dino_views", type=int, default=None)
    parser.add_argument("--sigma_extent", type=float, default=3.0)
    parser.add_argument("--max_patch_radius", type=int, default=4)
    parser.add_argument("--min_patch_radius", type=int, default=1)
    parser.add_argument("--min_var_px", type=float, default=0.25)
    parser.add_argument("--min_var_patch", type=float, default=0.25)
    parser.add_argument("--min_alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--alpha_clip", type=float, default=0.99)
    parser.add_argument("--min_total_weight", type=float, default=1e-4)
    parser.add_argument("--min_view_weight", type=float, default=1e-5)
    parser.add_argument("--max_pca_samples", type=int, default=50000)
    parser.add_argument("--normalize_patch_features", action="store_true", default=True)
    parser.add_argument("--no_normalize_patch_features", dest="normalize_patch_features", action="store_false")
    parser.add_argument("--normalize_output_features", action="store_true", default=True)
    parser.add_argument("--no_normalize_output_features", dest="normalize_output_features", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--summary_tag", default="")
    parser.add_argument("--verbose_views", action="store_true")
    parser.add_argument("--log_view_every", type=int, default=24)
    args = parser.parse_args()

    if args.resize_width % 14 != 0 or args.resize_height % 14 != 0:
        raise ValueError("resize dimensions should be divisible by 14 for DINOv2-small.")
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count.")

    method_name = f"{safe_name(args.model_name)}_render_contrib_patch{args.resize_width}x{args.resize_height}"
    method_dir = args.out_root / method_name
    (method_dir / "features").mkdir(parents=True, exist_ok=True)
    (method_dir / "pca_ply").mkdir(parents=True, exist_ok=True)
    (method_dir / "summary").mkdir(parents=True, exist_ok=True)

    items = select_items(args.npz_root, args.per_category, args.seed)
    if args.limit_scenes is not None:
        items = items[: args.limit_scenes]
    if args.shard_count > 1:
        items = items[args.shard_index :: args.shard_count]
    summary_tag = args.summary_tag or (f"_shard{args.shard_index:02d}of{args.shard_count:02d}" if args.shard_count > 1 else "")
    selected_rows = [
        {
            "scene_key": item["scene_key"],
            "category": item["category"],
            "model_split": item["model_split"],
            "raw_split": item["raw_split"],
            "num_gaussians": item["num_gaussians"],
            "num_handle": item["num_handle"],
            "source_exp40_npz": item["npz"],
            "source_object_ply": item["source_object_ply"],
        }
        for item in items
    ]
    save_csv(method_dir / "summary" / f"selected_scenes{summary_tag}.csv", selected_rows)
    print(f"[select] scenes={len(items)} categories={len({x['category'] for x in items})} shard={args.shard_index}/{args.shard_count}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, patch_size, hidden_size = load_dinov2(args.model_name, device, args.local_files_only)
    if args.resize_width % patch_size != 0 or args.resize_height % patch_size != 0:
        raise ValueError(f"resize dimensions must be divisible by patch size {patch_size}.")

    summaries = []
    for idx, item in enumerate(items, start=1):
        print(f"[{idx:03d}/{len(items):03d}] extracting {item['scene_key']} ({item['category']})", flush=True)
        summary = extract_dino_for_item(item, args, method_dir, model, patch_size, hidden_size, device)
        summaries.append(summary)
        print(
            f"{item['scene_key']}: valid DINO {summary['valid_dino']}/{summary['gaussians']} "
            f"({summary['valid_dino_ratio']:.1%}) -> {summary['features']}",
            flush=True,
        )
        save_csv(method_dir / "summary" / f"dinov2_pilot_summary{summary_tag}.csv", summaries)

    pca_path = None
    if not args.no_pca:
        feature_paths = [Path(item["features"]) for item in summaries]
        mean, components, lo, scale = fit_pca(feature_paths, args.max_pca_samples)
        pca_path = method_dir / "summary" / "global_pca_projection.npz"
        np.savez_compressed(pca_path, mean=mean, components=components, color_low=lo, color_scale=scale)
        for item in summaries:
            out_ply = method_dir / "pca_ply" / f"{item['scene_key']}_dinov2_render_contrib_pca.ply"
            write_pca_ply(Path(item["features"]), out_ply, mean, components, lo, scale)
            item["pca_ply"] = str(out_ply)
            print(f"{item['scene_key']}: PCA PLY -> {out_ply}", flush=True)
        save_csv(method_dir / "summary" / f"dinov2_pilot_summary{summary_tag}.csv", summaries)

    with (method_dir / "summary" / "dinov2_pilot_summary.json").open("w") as f:
        json.dump(
            {
                "method": "Exp40 B object-pruned Gaussian DINOv2 features using projected Gaussian footprint and front-to-back T*alpha render-contribution weights",
                "npz_root": str(args.npz_root),
                "bad_scenes_excluded": sorted(BAD_SCENES),
                "model_name": args.model_name,
                "resize_width": args.resize_width,
                "resize_height": args.resize_height,
                "patch_size": patch_size,
                "dino_dtype": args.dino_dtype,
                "per_category": args.per_category,
                "pca_projection": str(pca_path) if pca_path else None,
                "scenes": summaries,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
