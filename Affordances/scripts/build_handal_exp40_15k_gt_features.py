#!/usr/bin/env python3
"""Build compact 15k HANDAL GT object/handle feature data.

This rebuild uses HANDAL object masks for object pruning and HANDAL handle masks
for handle labels. It intentionally does not extract DINO features.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
from plyfile import PlyData


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
REALM_ROOT = Path("/home/gvilaplana/GS-Thesis/REALM-Code")
PYTHON = Path("/home/gvilaplana/miniconda3/envs/realm/bin/python")
SCALE = 0.001

sys.path.insert(0, str(BASE_ROOT / "scripts"))

import build_handal_generalization_feature_pilot as pilot  # noqa: E402


RAW_ROOT_BY_FEATURE_ROOT = {
    "handal_handle_generalization_features": BASE_ROOT / "data" / "handal_handle_generalization_v1",
    "handal_new_categories_delta_v1_features": BASE_ROOT / "data" / "handal_new_categories_delta_v1",
    "handal_mugs_b_quality_features_v1": BASE_ROOT / "data" / "handal_handle_generalization_v1",
}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def scene_key(category: str, scene_id: str) -> str:
    return f"{category}__{scene_id}"


def parse_source_split(value) -> str | None:
    if isinstance(value, list) and value:
        return str(value[0])
    text = str(value or "").strip()
    for ch in "[]'\"":
        text = text.replace(ch, "")
    if "," in text:
        text = text.split(",", 1)[0]
    return text or None


def raw_root_for_npz(npz_path: str) -> Path:
    feature_root = Path(npz_path).parent.parent
    root = RAW_ROOT_BY_FEATURE_ROOT.get(feature_root.name)
    if root is None:
        raise KeyError(f"Unknown HANDAL feature root: {feature_root}")
    return root


def find_source_scene(item: dict) -> tuple[Path, str]:
    raw_root = raw_root_for_npz(item["npz"])
    category = item["category"]
    scene_id = item["scene_id"]
    preferred = parse_source_split(item.get("source_split"))
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["train_seen", "val_seen", "test_seen", "test_unseen_category"])
    seen = set()
    for split in candidates:
        if split in seen:
            continue
        seen.add(split)
        path = raw_root / "subset_export" / split / category / scene_id
        if path.exists():
            return path, split
    raise FileNotFoundError(f"No raw scene found for {category} {scene_id} under {raw_root}")


def load_split_rows(split_manifest: Path, splits: list[str]) -> list[dict]:
    payload = json.loads(split_manifest.read_text())["splits"]
    rows: dict[str, dict] = {}
    for split in splits:
        for item in payload[split]:
            key = item["scene_key"]
            raw_scene, raw_split = find_source_scene(item)
            row = {
                "scene_key": key,
                "model_split": split,
                "raw_split": raw_split,
                "category": item["category"],
                "scene_id": item["scene_id"],
                "instance_id": str(item.get("instance_id", "")),
                "source_npz": item["npz"],
                "source_feature_root": str(Path(item["npz"]).parent.parent),
                "source_scene": str(raw_scene),
                "old_num_gaussians": int(item.get("num_gaussians", 0)),
                "old_num_handle": int(item.get("num_handle", 0)),
            }
            rows[key] = row
    return sorted(rows.values(), key=lambda r: (r["model_split"], r["category"], r["scene_key"]))


def shard_rows(rows: list[dict], num_shards: int) -> list[list[dict]]:
    shards = [[] for _ in range(num_shards)]
    loads = [0 for _ in range(num_shards)]
    for row in sorted(rows, key=lambda r: int(r["old_num_gaussians"]), reverse=True):
        idx = min(range(num_shards), key=lambda i: loads[i])
        shards[idx].append(row)
        loads[idx] += int(row["old_num_gaussians"])
    for shard in shards:
        shard.sort(key=lambda r: (r["model_split"], r["category"], r["scene_key"]))
    return shards


def selected_frame_ids(raw_scene: Path, max_images: int) -> list[int]:
    return pilot.selected_frame_ids(raw_scene, max_images)


def symlink_or_replace(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def prepare_scene(row: dict, args: Namespace) -> Path:
    key = row["scene_key"]
    raw_scene = Path(row["source_scene"])
    dst_scene = args.work_root / "3dgs_scenes" / key
    images_dir = dst_scene / "images"
    sparse_dir = dst_scene / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    scene_camera = pilot.load_json(raw_scene / "scene_camera.json")
    scene_gt = pilot.load_json(raw_scene / "scene_gt.json")
    frame_ids = selected_frame_ids(raw_scene, args.max_images)
    for frame_id in frame_ids:
        cam = scene_camera[str(frame_id)]
        if "width" not in cam or "height" not in cam:
            width, height = pilot.Image.open(raw_scene / "rgb" / f"{frame_id:06d}.jpg").size
            cam["width"] = width
            cam["height"] = height
        symlink_or_replace(raw_scene / "rgb" / f"{frame_id:06d}.jpg", images_dir / f"{frame_id:06d}.jpg")

    obj_id = int(scene_gt[str(frame_ids[0])][0]["obj_id"])
    pilot.write_cameras_txt(sparse_dir / "cameras.txt", scene_camera)
    pilot.write_images_txt(sparse_dir / "images.txt", frame_ids, scene_gt)
    pilot.write_points_ply(sparse_dir / "points3D.ply", raw_scene, obj_id, args.init_points)
    write_json(
        dst_scene / "handal_source.json",
        {
            "dataset": Path(row["source_scene"]).parts[-5] if len(Path(row["source_scene"]).parts) > 5 else "handal",
            "model_split": row["model_split"],
            "raw_split": row["raw_split"],
            "category": row["category"],
            "scene_id": row["scene_id"],
            "scene_key": key,
            "source_scene": row["source_scene"],
            "obj_id": obj_id,
            "num_images": len(frame_ids),
            "unit_scale": SCALE,
        },
    )
    return dst_scene


def trained_ply(args: Namespace, key: str) -> Path:
    return args.model_root / key / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"


def train_3dgs_scene(row: dict, args: Namespace) -> Path:
    key = row["scene_key"]
    out_ply = trained_ply(args, key)
    if out_ply.exists() and not args.force_3dgs:
        return out_ply
    scene_dir = prepare_scene(row, args)
    model_dir = args.model_root / key
    log_path = args.log_root / f"{key}_3dgs_{args.iterations}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        "train.py",
        "-s",
        str(scene_dir),
        "-m",
        str(model_dir),
        "--config_file",
        str(args.config_file),
        "--iterations",
        str(args.iterations),
        "--test_iterations",
        str(args.iterations),
        "--save_iterations",
        str(args.iterations),
        "--resolution",
        str(args.resolution),
        "--quiet",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    started = time.time()
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, cwd=REALM_ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"3DGS training failed for {key}; see {log_path}")
    if not out_ply.exists():
        raise FileNotFoundError(out_ply)
    print(f"[3dgs] {key} finished in {(time.time() - started) / 60.0:.1f} min", flush=True)
    return out_ply


def filter_ply(src_ply: Path, dst_ply: Path, keep: np.ndarray) -> None:
    ply = PlyData.read(src_ply)
    ply["vertex"].data = ply["vertex"].data[keep]
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(dst_ply)


def object_prune_center(row: dict, args: Namespace, src_ply: Path) -> tuple[Path, dict]:
    raw_scene = Path(row["source_scene"])
    scene_camera = pilot.load_json(raw_scene / "scene_camera.json")
    scene_gt = pilot.load_json(raw_scene / "scene_gt.json")
    ply = PlyData.read(src_ply)
    points = np.vstack([ply["vertex"].data["x"], ply["vertex"].data["y"], ply["vertex"].data["z"]]).T.astype(np.float64)
    hits = np.zeros(len(points), dtype=np.uint16)
    visible = np.zeros(len(points), dtype=np.uint16)

    for frame_id in selected_frame_ids(raw_scene, args.max_images):
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * SCALE
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        mask = pilot.load_mask(raw_scene / "mask" / f"{frame_id:06d}_000000.png")
        width, height = pilot.camera_size(cam, raw_scene, frame_id)
        ui, vi, valid = pilot.project_points(points, rot, trans, cam_k, width, height)
        valid_idx = np.flatnonzero(valid)
        visible[valid_idx] += 1
        inside_idx = valid_idx[mask[vi[valid_idx], ui[valid_idx]]]
        hits[inside_idx] += 1

    score = np.divide(hits, visible, out=np.zeros(len(points), dtype=np.float32), where=visible > 0)
    keep = (visible >= args.object_min_visible) & (score >= args.object_threshold)
    out_ply = args.ply_root / "A_center" / f"{row['scene_key']}_object_center.ply"
    filter_ply(src_ply, out_ply, keep)
    stats = {
        "keep": keep,
        "gaussian_indices": np.flatnonzero(keep).astype(np.int64),
        "object_scores": score[keep].astype(np.float32),
        "object_hits": hits[keep],
        "object_visible_hits": visible[keep],
        "input_gaussians": int(len(points)),
    }
    return out_ply, stats


def fibonacci_dirs(n: int) -> np.ndarray:
    dirs = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - (2.0 * (i + 0.5) / n)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        dirs.append([np.cos(theta) * r, np.sin(theta) * r, z])
    return np.asarray(dirs, dtype=np.float64)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((len(q), 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def ellipsoid_sample_points(vertices, sigma: float, num_samples: int) -> np.ndarray:
    centers = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    scales = np.exp(np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T.astype(np.float64))
    quats = np.vstack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).T
    rots = quat_to_rotmat(quats)
    dirs = fibonacci_dirs(num_samples)
    local = dirs[None, :, :] * scales[:, None, :] * float(sigma)
    offsets = np.einsum("nij,nsj->nsi", rots, local)
    return centers[:, None, :] + offsets


def ellipsoid20_strict75_cleanup(row: dict, args: Namespace, object_ply: Path, center_stats: dict) -> tuple[Path, dict]:
    raw_scene = Path(row["source_scene"])
    scene_camera = pilot.load_json(raw_scene / "scene_camera.json")
    scene_gt = pilot.load_json(raw_scene / "scene_gt.json")
    ply = PlyData.read(object_ply)
    vertices = ply["vertex"].data
    centers = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    sample_points = ellipsoid_sample_points(vertices, args.ellipsoid_sigma, args.ellipsoid_samples)

    valid_total = np.zeros(len(vertices), dtype=np.uint32)
    inside_total = np.zeros(len(vertices), dtype=np.uint32)
    center_visible = np.zeros(len(vertices), dtype=np.uint16)
    center_inside = np.zeros(len(vertices), dtype=np.uint16)

    for frame_id in selected_frame_ids(raw_scene, args.max_images):
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * SCALE
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        mask = pilot.load_mask(raw_scene / "mask" / f"{frame_id:06d}_000000.png")
        width, height = pilot.camera_size(cam, raw_scene, frame_id)

        ui, vi, valid = pilot.project_points(centers, rot, trans, cam_k, width, height)
        valid_idx = np.flatnonzero(valid)
        center_visible[valid_idx] += 1
        inside_idx = valid_idx[mask[vi[valid_idx], ui[valid_idx]]]
        center_inside[inside_idx] += 1

        flat = sample_points.reshape(-1, 3)
        sui, svi, svalid = pilot.project_points(flat, rot, trans, cam_k, width, height)
        svalid = svalid.reshape(len(vertices), args.ellipsoid_samples)
        sui = sui.reshape(len(vertices), args.ellipsoid_samples)
        svi = svi.reshape(len(vertices), args.ellipsoid_samples)
        valid_total += svalid.sum(axis=1).astype(np.uint32)
        if np.any(svalid):
            inside = np.zeros_like(svalid, dtype=bool)
            rr, cc = np.nonzero(svalid)
            inside[rr, cc] = mask[svi[rr, cc], sui[rr, cc]]
            inside_total += inside.sum(axis=1).astype(np.uint32)

    sample_ratio = np.divide(
        inside_total,
        valid_total,
        out=np.zeros(len(vertices), dtype=np.float32),
        where=valid_total > 0,
    )
    center_ratio = np.divide(
        center_inside,
        center_visible,
        out=np.zeros(len(vertices), dtype=np.float32),
        where=center_visible > 0,
    )
    keep = (
        (valid_total >= args.ellipsoid_min_valid_samples)
        & (sample_ratio >= args.ellipsoid_inside_threshold)
        & (center_visible >= args.ellipsoid_center_min_visible)
        & (center_ratio >= args.ellipsoid_center_inside_threshold)
    )
    out_ply = args.ply_root / "B_center_ellipsoid20_strict75" / f"{row['scene_key']}_object_center_ell20s75.ply"
    filter_ply(object_ply, out_ply, keep)
    stats = {
        "keep_from_a": keep,
        "gaussian_indices": center_stats["gaussian_indices"][keep],
        "object_scores": center_stats["object_scores"][keep],
        "object_hits": center_stats["object_hits"][keep],
        "object_visible_hits": center_stats["object_visible_hits"][keep],
        "ellipsoid_valid_samples": valid_total[keep],
        "ellipsoid_inside_samples": inside_total[keep],
        "ellipsoid_inside_ratio": sample_ratio[keep].astype(np.float32),
        "center_reproj_visible": center_visible[keep],
        "center_reproj_inside": center_inside[keep],
        "center_reproj_ratio": center_ratio[keep].astype(np.float32),
    }
    return out_ply, stats


def save_compact_npz(
    row: dict,
    args: Namespace,
    variant: str,
    object_ply: Path,
    object_stats: dict,
    handle_scores: np.ndarray,
    handle_labels: np.ndarray,
    handle_visible: np.ndarray,
) -> Path:
    vertices = PlyData.read(object_ply)["vertex"].data
    xyz, scale, rotation, opacity, color, geometry = pilot.geometry_arrays(vertices)
    out_path = args.npz_root / variant / f"{row['scene_key']}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        geometry_features=geometry.astype(np.float32),
        xyz=xyz.astype(np.float32),
        scale=scale.astype(np.float32),
        rotation=rotation.astype(np.float32),
        opacity=opacity.astype(np.float32),
        color=color.astype(np.float32),
        object_scores=object_stats["object_scores"].astype(np.float32),
        object_hits=object_stats["object_hits"],
        object_visible_hits=object_stats["object_visible_hits"],
        handle_scores=handle_scores.astype(np.float32),
        handle_labels_thr0_25=handle_labels.astype(np.uint8),
        handle_visible_views=handle_visible,
        gaussian_indices=object_stats["gaussian_indices"].astype(np.int64),
        scene_key=np.array([row["scene_key"]]),
        scene_id=np.array([row["scene_id"]]),
        category=np.array([row["category"]]),
        model_split=np.array([row["model_split"]]),
        raw_split=np.array([row["raw_split"]]),
        instance_id=np.array([row["instance_id"]]),
        source_scene_path=np.array([row["source_scene"]]),
        source_npz=np.array([row["source_npz"]]),
        source_object_ply=np.array([str(object_ply)]),
        input_gaussians=np.array([object_stats.get("input_gaussians", -1)], dtype=np.int32),
        object_threshold=np.array([args.object_threshold], dtype=np.float32),
        object_min_visible=np.array([args.object_min_visible], dtype=np.int32),
        handle_threshold=np.array([args.handle_threshold], dtype=np.float32),
        handle_min_visible=np.array([args.handle_min_visible], dtype=np.int32),
        dino_model=np.array(["not_extracted_exp40_15k_gt"]),
        dino_dtype=np.array(["not_extracted"]),
        variant=np.array([variant]),
    )
    for key in [
        "ellipsoid_valid_samples",
        "ellipsoid_inside_samples",
        "ellipsoid_inside_ratio",
        "center_reproj_visible",
        "center_reproj_inside",
        "center_reproj_ratio",
    ]:
        if key in object_stats:
            payload[key] = object_stats[key]
    np.savez_compressed(out_path, **payload)
    return out_path


def qa_keys(rows: list[dict], per_category: int) -> set[str]:
    counts = {}
    selected = set()
    for row in rows:
        cat = row["category"]
        if counts.get(cat, 0) >= per_category:
            continue
        selected.add(row["scene_key"])
        counts[cat] = counts.get(cat, 0) + 1
    return selected


def cleanup_success_temporaries(row: dict, args: Namespace) -> None:
    key = row["scene_key"]
    for path in [args.model_root / key, args.work_root / "3dgs_scenes" / key]:
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def process_scene(row: dict, args: Namespace, qa: set[str]) -> list[dict]:
    key = row["scene_key"]
    a_npz = args.npz_root / "A_center" / f"{key}.npz"
    b_npz = args.npz_root / "B_center_ellipsoid20_strict75" / f"{key}.npz"
    if a_npz.exists() and b_npz.exists() and not args.force_features:
        print(f"[skip] {key} already has A and B NPZs", flush=True)
        return []

    src_ply = train_3dgs_scene(row, args)
    a_ply, a_stats = object_prune_center(row, args, src_ply)
    a_handle_scores, a_handle_labels, a_handle_visible = pilot.handle_labels(row, args, a_ply)
    a_stats["input_gaussians"] = a_stats["input_gaussians"]
    a_npz = save_compact_npz(row, args, "A_center", a_ply, a_stats, a_handle_scores, a_handle_labels, a_handle_visible)

    b_ply, b_stats = ellipsoid20_strict75_cleanup(row, args, a_ply, a_stats)
    b_stats["input_gaussians"] = a_stats["input_gaussians"]
    b_handle_scores, b_handle_labels, b_handle_visible = pilot.handle_labels(row, args, b_ply)
    b_npz = save_compact_npz(
        row,
        args,
        "B_center_ellipsoid20_strict75",
        b_ply,
        b_stats,
        b_handle_scores,
        b_handle_labels,
        b_handle_visible,
    )

    if key in qa:
        pilot.write_debug_ply(a_ply, a_handle_labels, args.ply_root / "QA_handle_red" / "A_center" / f"{key}_handle_red.ply")
        pilot.write_debug_ply(
            b_ply,
            b_handle_labels,
            args.ply_root / "QA_handle_red" / "B_center_ellipsoid20_strict75" / f"{key}_handle_red.ply",
        )

    rows = []
    for variant, npz_path, ply_path, stats, labels in [
        ("A_center", a_npz, a_ply, a_stats, a_handle_labels),
        ("B_center_ellipsoid20_strict75", b_npz, b_ply, b_stats, b_handle_labels),
    ]:
        obj_n = int(len(stats["object_scores"]))
        handle_n = int(labels.sum())
        rows.append(
            {
                "scene_key": key,
                "variant": variant,
                "status": "ok",
                "model_split": row["model_split"],
                "raw_split": row["raw_split"],
                "category": row["category"],
                "scene_id": row["scene_id"],
                "input_gaussians": int(a_stats["input_gaussians"]),
                "object_gaussians": obj_n,
                "handle_gaussians": handle_n,
                "handle_ratio": float(handle_n / max(obj_n, 1)),
                "retention_vs_a": 1.0 if variant == "A_center" else float(obj_n / max(len(a_stats["object_scores"]), 1)),
                "npz": str(npz_path),
                "object_ply": str(ply_path),
            }
        )
    if args.cleanup_success:
        cleanup_success_temporaries(row, args)
    return rows


def write_config(path: Path, densify_until_iter: int) -> Path:
    cfg = {
        "densify_until_iter": densify_until_iter,
        "num_classes": 2,
        "reg3d_interval": 1000000,
        "reg3d_k": 5,
        "reg3d_lambda_val": 0,
        "reg3d_max_points": 300000,
        "reg3d_sample_size": 1000,
        "object_loss_start_iter": 1000000,
    }
    write_json(path, cfg)
    return path


def write_metrics_md(path: Path, rows: list[dict], args: Namespace) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    lines = [
        "# Exp40 15k HANDAL GT Feature Rebuild",
        "",
        f"- iterations: {args.iterations}",
        f"- resolution: {args.resolution}",
        f"- object pruning A: center, visible >= {args.object_min_visible}, score >= {args.object_threshold}",
        f"- object pruning B: A + ellipsoid{args.ellipsoid_samples}_strict{int(args.ellipsoid_inside_threshold * 100)}",
        f"- handle labels: footprint overlap, visible >= {args.handle_min_visible}, score >= {args.handle_threshold}",
        "- DINO: not extracted",
        "",
        "| variant | scenes | mean object gaussians | mean handle ratio |",
        "|---|---:|---:|---:|",
    ]
    for variant in ["A_center", "B_center_ellipsoid20_strict75"]:
        items = [r for r in ok_rows if r["variant"] == variant]
        if not items:
            continue
        lines.append(
            f"| {variant} | {len(items)} | "
            f"{np.mean([float(r['object_gaussians']) for r in items]):.1f} | "
            f"{np.mean([float(r['handle_ratio']) for r in items]):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def aggregate_outputs(args: Namespace) -> None:
    rows = []
    for manifest in sorted(args.out_root.glob("run_manifest_shard*.csv")):
        with manifest.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        return
    fieldnames = [
        "scene_key",
        "variant",
        "status",
        "model_split",
        "raw_split",
        "category",
        "scene_id",
        "input_gaussians",
        "object_gaussians",
        "handle_gaussians",
        "handle_ratio",
        "retention_vs_a",
        "npz",
        "object_ply",
    ]
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    write_csv(args.out_root / "per_scene_summary.csv", ok_rows, fieldnames)

    grouped = {}
    for r in ok_rows:
        grouped.setdefault((r["variant"], r["model_split"], r["category"]), []).append(r)
    cat_rows = []
    for (variant, split, category), items in sorted(grouped.items()):
        cat_rows.append(
            {
                "variant": variant,
                "model_split": split,
                "category": category,
                "num_scenes": len(items),
                "total_object_gaussians": int(sum(int(float(i["object_gaussians"])) for i in items)),
                "total_handle_gaussians": int(sum(int(float(i["handle_gaussians"])) for i in items)),
                "mean_object_gaussians": float(np.mean([float(i["object_gaussians"]) for i in items])),
                "mean_handle_ratio": float(np.mean([float(i["handle_ratio"]) for i in items])),
                "mean_retention_vs_a": float(np.mean([float(i["retention_vs_a"]) for i in items])),
            }
        )
    write_csv(
        args.out_root / "per_category_summary.csv",
        cat_rows,
        [
            "variant",
            "model_split",
            "category",
            "num_scenes",
            "total_object_gaussians",
            "total_handle_gaussians",
            "mean_object_gaussians",
            "mean_handle_ratio",
            "mean_retention_vs_a",
        ],
    )
    write_json(
        args.out_root / "dataset_summary.json",
        {
            "num_ok_variant_rows": len(ok_rows),
            "num_unique_scenes_ok": len({r["scene_key"] for r in ok_rows}),
            "variants": ["A_center", "B_center_ellipsoid20_strict75"],
            "npz_root": str(args.npz_root),
            "ply_root": str(args.ply_root),
        },
    )
    write_metrics_md(args.out_root / "metrics.md", ok_rows, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=BASE_ROOT
        / "outputs/03_handle_generalization/08_17category_mlp_baselines_v1/01_seen_instance_17cat/split_manifest.json",
    )
    parser.add_argument("--out_root", type=Path, default=BASE_ROOT / "data" / "handal_exp40_15k_gt_features_v1")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--write_specs_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--densify_until_iter", type=int, default=9000)
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--init_points", type=int, default=30000)
    parser.add_argument("--object_threshold", type=float, default=0.60)
    parser.add_argument("--object_min_visible", type=int, default=20)
    parser.add_argument("--handle_threshold", type=float, default=0.25)
    parser.add_argument("--handle_min_visible", type=int, default=10)
    parser.add_argument("--sigma_extent", type=float, default=3.0)
    parser.add_argument("--max_radius", type=int, default=24)
    parser.add_argument("--min_var_px", type=float, default=0.25)
    parser.add_argument("--ellipsoid_samples", type=int, default=20)
    parser.add_argument("--ellipsoid_sigma", type=float, default=1.0)
    parser.add_argument("--ellipsoid_min_valid_samples", type=int, default=20)
    parser.add_argument("--ellipsoid_inside_threshold", type=float, default=0.75)
    parser.add_argument("--ellipsoid_center_min_visible", type=int, default=3)
    parser.add_argument("--ellipsoid_center_inside_threshold", type=float, default=0.30)
    parser.add_argument("--qa_per_category", type=int, default=1)
    parser.add_argument("--summary_every", type=int, default=5)
    parser.add_argument("--force_3dgs", action="store_true")
    parser.add_argument("--force_features", action="store_true")
    parser.add_argument("--cleanup_success", action="store_true", default=True)
    args = parser.parse_args()

    args.work_root = args.out_root / "work"
    args.model_root = args.work_root / "3dgs_models"
    args.log_root = args.work_root / "logs"
    args.npz_root = args.out_root / "npz"
    args.ply_root = args.out_root / "ply"
    args.config_file = write_config(args.out_root / "config" / "handal_3dgs_15k_gt.json", args.densify_until_iter)
    for path in [args.out_root, args.work_root, args.model_root, args.log_root, args.npz_root, args.ply_root]:
        path.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_outputs(args)
        return

    rows = load_split_rows(args.split_manifest, args.splits)
    if args.max_scenes is not None:
        rows = rows[: args.max_scenes]
    shards = shard_rows(rows, args.num_shards)
    write_csv(
        args.out_root / "scene_specs.csv",
        rows,
        [
            "scene_key",
            "model_split",
            "raw_split",
            "category",
            "scene_id",
            "instance_id",
            "source_npz",
            "source_feature_root",
            "source_scene",
            "old_num_gaussians",
            "old_num_handle",
        ],
    )
    for idx, shard in enumerate(shards):
        write_csv(args.out_root / f"scene_specs_shard{idx}.csv", shard, list(rows[0].keys()))
    write_json(
        args.out_root / "experiment_config.json",
        {
            "design": "15k HANDAL GT feature rebuild; A=center object prune; B=A+ellipsoid20_strict75; no DINO",
            "num_scenes": len(rows),
            "num_shards": args.num_shards,
            "iterations": args.iterations,
            "resolution": args.resolution,
            "densify_until_iter": args.densify_until_iter,
            "object_threshold": args.object_threshold,
            "object_min_visible": args.object_min_visible,
            "handle_threshold": args.handle_threshold,
            "handle_min_visible": args.handle_min_visible,
        },
    )
    if args.write_specs_only:
        print(f"[specs] wrote {len(rows)} scenes -> {args.out_root}", flush=True)
        return

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"Invalid shard_index={args.shard_index}")
    shard = shards[args.shard_index]
    qa = qa_keys(rows, args.qa_per_category)
    manifest_path = args.out_root / f"run_manifest_shard{args.shard_index}.csv"
    fail_path = args.out_root / f"failed_scenes_shard{args.shard_index}.csv"
    manifest_fields = [
        "scene_key",
        "variant",
        "status",
        "model_split",
        "raw_split",
        "category",
        "scene_id",
        "input_gaussians",
        "object_gaussians",
        "handle_gaussians",
        "handle_ratio",
        "retention_vs_a",
        "npz",
        "object_ply",
    ]
    fail_fields = ["scene_key", "model_split", "category", "scene_id", "error"]
    print(
        f"[start] shard={args.shard_index}/{args.num_shards} gpu={args.gpu} scenes={len(shard)} out={args.out_root}",
        flush=True,
    )
    start = time.time()
    for idx, row in enumerate(shard, start=1):
        key = row["scene_key"]
        print(f"[{idx}/{len(shard)}] {key}", flush=True)
        try:
            summary_rows = process_scene(row, args, qa)
            for item in summary_rows:
                append_csv(manifest_path, item, manifest_fields)
            if idx % args.summary_every == 0:
                aggregate_outputs(args)
            elapsed = (time.time() - start) / 60.0
            print(f"[ok] {key}; elapsed={elapsed:.1f} min", flush=True)
        except Exception as exc:  # keep worker resumable across scene failures
            print(f"[fail] {key}: {exc}", flush=True)
            append_csv(
                fail_path,
                {
                    "scene_key": key,
                    "model_split": row["model_split"],
                    "category": row["category"],
                    "scene_id": row["scene_id"],
                    "error": repr(exc),
                },
                fail_fields,
            )
    aggregate_outputs(args)
    print(f"[done] shard={args.shard_index}; manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
