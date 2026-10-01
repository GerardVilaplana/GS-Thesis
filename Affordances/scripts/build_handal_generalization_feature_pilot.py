import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from build_handal_exp40_supervision import (
    center_zbuffer_filter,
    footprint_overlap,
    gaussian_covariances,
    load_mask,
    project_means_and_covariances,
    project_points,
    sigmoid,
)
from extract_handal_dinov2_gaussian_embeddings import (
    accumulate_weighted_features,
    extract_patch_features,
    load_dinov2,
    make_view_contributions,
    normalize_rows,
    rgb_to_sh,
)
from prepare_handal_3dgs_scenes import rotmat2qvec


C0 = 0.28209479177387814
REALM_ROOT = Path("/home/gvilaplana/GS-Thesis/REALM-Code")
PYTHON = Path("/home/gvilaplana/miniconda3/envs/realm/bin/python")
CONFIG = Path("/home/gvilaplana/GS-Thesis/Affordances/configs/handal_3dgs_pilot.json")
DATASET_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_v1")
OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features")
SCALE = 0.001


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def symlink_or_replace(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def scene_key(category, scene_id):
    return f"{category}__{scene_id}"


def load_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def local_scene_path(dataset_root, row):
    return dataset_root / "subset_export" / row["split"] / row["category"] / row["scene_id"]


def select_pilot_rows(rows, dataset_root):
    categories = sorted({row["category"] for row in rows})
    selected = []
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        splits = {row["split"] for row in category_rows}
        preferred_split = "train_seen" if "train_seen" in splits else "test_unseen_category"
        candidates = [row for row in category_rows if row["split"] == preferred_split]
        candidates.sort(key=lambda row: (row["instance_id"], row["scene_id"]))
        for row in candidates:
            if local_scene_path(dataset_root, row).exists():
                selected.append(row)
                break
        else:
            raise FileNotFoundError(f"No local scene found for category {category}")
    return selected


def selected_frame_ids(scene_dir, max_images):
    frame_ids = sorted(int(p.stem) for p in (scene_dir / "rgb").glob("*.jpg"))
    if len(frame_ids) > max_images:
        idx = np.linspace(0, len(frame_ids) - 1, max_images).round().astype(np.int64)
        frame_ids = [frame_ids[i] for i in idx]
    return frame_ids




def camera_size(cam, raw_scene=None, frame_id=None):
    if "width" in cam and "height" in cam:
        return int(cam["width"]), int(cam["height"])
    if raw_scene is not None and frame_id is not None:
        return Image.open(raw_scene / "rgb" / f"{frame_id:06d}.jpg").size
    raise KeyError("Camera entry has no width/height and no image path fallback was provided.")

def write_cameras_txt(path, scene_camera):
    first = scene_camera[sorted(scene_camera.keys(), key=lambda x: int(x))[0]]
    fx, _, cx, _, fy, cy, _, _, _ = first["cam_K"]
    width = int(first["width"])
    height = int(first["height"])
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}\n")


def write_images_txt(path, frame_ids, scene_gt):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frame_ids)}, mean observations per image: 0\n")
        for image_id, frame_id in enumerate(frame_ids, start=1):
            gt = scene_gt[str(frame_id)][0]
            rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * SCALE
            qvec = rotmat2qvec(rot)
            f.write(
                f"{image_id} "
                f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
                f"{trans[0]:.12f} {trans[1]:.12f} {trans[2]:.12f} "
                f"1 {frame_id:06d}.jpg\n\n"
            )


def write_points_ply(path, raw_scene, obj_id, max_points):
    model_path = raw_scene.parent / "models" / f"obj_{obj_id:06d}.ply"
    ply = PlyData.read(model_path)
    vertex = ply["vertex"]
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32) * SCALE
    if {"red", "green", "blue"}.issubset(vertex.data.dtype.names):
        rgb = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points).round().astype(np.int64)
        xyz = xyz[idx]
        rgb = rgb[idx]

    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    out = np.empty(len(xyz), dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["nx"], out["ny"], out["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    out["red"], out["green"], out["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(path)


def prepare_scene(row, args):
    raw_scene = local_scene_path(args.dataset_root, row)
    key = scene_key(row["category"], row["scene_id"])
    dst_scene = args.work_root / "3dgs_scenes" / key
    images_dir = dst_scene / "images"
    sparse_dir = dst_scene / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")
    frame_ids = selected_frame_ids(raw_scene, args.max_images)
    for frame_id in frame_ids:
        cam = scene_camera[str(frame_id)]
        if "width" not in cam or "height" not in cam:
            width, height = Image.open(raw_scene / "rgb" / f"{frame_id:06d}.jpg").size
            cam["width"] = width
            cam["height"] = height
    obj_id = int(scene_gt[str(frame_ids[0])][0]["obj_id"])

    for frame_id in frame_ids:
        symlink_or_replace(raw_scene / "rgb" / f"{frame_id:06d}.jpg", images_dir / f"{frame_id:06d}.jpg")

    write_cameras_txt(sparse_dir / "cameras.txt", scene_camera)
    write_images_txt(sparse_dir / "images.txt", frame_ids, scene_gt)
    write_points_ply(sparse_dir / "points3D.ply", raw_scene, obj_id, args.init_points)
    write_json(
        dst_scene / "handal_source.json",
        {
            "dataset": "handal_handle_generalization_v1",
            "split": row["split"],
            "category": row["category"],
            "scene_id": row["scene_id"],
            "scene_key": key,
            "source_scene": str(raw_scene),
            "obj_id": obj_id,
            "num_images": len(frame_ids),
            "unit_scale": SCALE,
        },
    )
    return dst_scene


def trained_ply(model_root, key, iteration):
    return model_root / key / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"


def final_raw_ply(args, key):
    return args.work_root / "raw_3dgs_ply" / f"{key}_iter{args.iterations}.ply"


def train_3dgs(rows, args):
    args.model_root.mkdir(parents=True, exist_ok=True)
    (args.work_root / "raw_3dgs_ply").mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    pending = []

    for row in rows:
        key = scene_key(row["category"], row["scene_id"])
        scene_dir = prepare_scene(row, args)
        out_ply = final_raw_ply(args, key)
        if out_ply.exists() and not args.force_3dgs:
            print(f"reuse raw 3DGS PLY: {out_ply}")
            continue
        existing = trained_ply(args.model_root, key, args.iterations)
        if existing.exists() and not args.force_3dgs:
            shutil.copy2(existing, out_ply)
            print(f"copied existing trained PLY: {key}")
            continue
        pending.append((row, scene_dir))

    active = []
    start = time.time()
    completed = 0
    print(f"3DGS pending pilot scenes: {len(pending)}")
    while pending or active:
        while pending and len(active) < len(args.gpus):
            row, scene_dir = pending.pop(0)
            key = scene_key(row["category"], row["scene_id"])
            busy_gpus = {job["gpu"] for job in active}
            free_gpus = [gpu for gpu in args.gpus if gpu not in busy_gpus]
            gpu = free_gpus[0] if free_gpus else args.gpus[len(active) % len(args.gpus)]
            model_dir = args.model_root / key
            log_path = args.log_root / f"{key}.log"
            log_f = open(log_path, "w")
            cmd = [
                str(PYTHON),
                "train.py",
                "-s",
                str(scene_dir),
                "-m",
                str(model_dir),
                "--config_file",
                str(CONFIG),
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
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(cmd, cwd=REALM_ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            active.append({"key": key, "gpu": gpu, "proc": proc, "log": log_f, "log_path": log_path, "start": time.time()})
            print(f"started 3DGS {key} on GPU {gpu}; log={log_path}")

        still_active = []
        for job in active:
            ret = job["proc"].poll()
            if ret is None:
                still_active.append(job)
                continue
            job["log"].close()
            key = job["key"]
            elapsed = (time.time() - job["start"]) / 60.0
            if ret != 0:
                raise RuntimeError(f"3DGS training failed for {key}; see {job['log_path']}")
            src = trained_ply(args.model_root, key, args.iterations)
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, final_raw_ply(args, key))
            completed += 1
            avg = (time.time() - start) / max(completed, 1) / 60.0
            remaining = len(pending) + len(still_active)
            eta = remaining * avg / max(len(args.gpus), 1)
            print(f"finished 3DGS {key} in {elapsed:.1f} min; rough ETA {eta:.1f} min")
        active = still_active
        if pending or active:
            time.sleep(10)


def object_prune(row, args):
    key = scene_key(row["category"], row["scene_id"])
    source = load_json(args.work_root / "3dgs_scenes" / key / "handal_source.json")
    raw_scene = Path(source["source_scene"])
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")
    src_ply = final_raw_ply(args, key)
    ply = PlyData.read(src_ply)
    points = np.vstack([ply["vertex"].data["x"], ply["vertex"].data["y"], ply["vertex"].data["z"]]).T.astype(np.float64)
    hits = np.zeros(len(points), dtype=np.uint16)
    visible = np.zeros(len(points), dtype=np.uint16)

    frame_ids = selected_frame_ids(raw_scene, args.max_images)
    for frame_id in frame_ids:
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        mask = load_mask(raw_scene / "mask" / f"{frame_id:06d}_000000.png")
        ui, vi, valid = project_points(points, rot, trans, cam_k, *camera_size(cam, raw_scene, frame_id))
        valid_idx = np.flatnonzero(valid)
        visible[valid_idx] += 1
        inside_idx = valid_idx[mask[vi[valid_idx], ui[valid_idx]]]
        hits[inside_idx] += 1

    score = np.divide(hits, visible, out=np.zeros(len(points), dtype=np.float32), where=visible > 0)
    keep = (visible >= args.object_min_visible) & (score >= args.object_threshold)
    object_ply = args.work_root / "object_ply" / f"{key}_object_pruned_thr{args.object_threshold:.2f}.ply"
    object_ply.parent.mkdir(parents=True, exist_ok=True)
    filtered = PlyData.read(src_ply)
    filtered["vertex"].data = filtered["vertex"].data[keep]
    filtered.write(object_ply)
    return object_ply, score[keep].astype(np.float32), np.flatnonzero(keep).astype(np.int64), int(len(points))


def handle_labels(row, args, object_ply):
    key = scene_key(row["category"], row["scene_id"])
    source = load_json(args.work_root / "3dgs_scenes" / key / "handal_source.json")
    raw_scene = Path(source["source_scene"])
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")
    ply = PlyData.read(object_ply)
    vertices = ply["vertex"].data
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    cov_world = gaussian_covariances(vertices)
    weighted_hits = np.zeros(len(points), dtype=np.float64)
    weighted_total = np.zeros(len(points), dtype=np.float64)
    visible = np.zeros(len(points), dtype=np.uint16)

    frame_ids = selected_frame_ids(raw_scene, args.max_images)
    for frame_id in frame_ids:
        mask_path = raw_scene / "mask_parts" / f"{frame_id:06d}_000000_handle.png"
        if mask_path.exists():
            mask = load_mask(mask_path)
        else:
            cam0 = scene_camera[str(frame_id)]
            width0, height0 = camera_size(cam0, raw_scene, frame_id)
            mask = np.zeros((height0, width0), dtype=bool)
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        u, v, z, cov2d, valid = project_means_and_covariances(
            points, cov_world, rot, trans, cam_k, *camera_size(cam, raw_scene, frame_id), args.min_var_px
        )
        width, _ = camera_size(cam, raw_scene, frame_id)
        for idx in center_zbuffer_filter(u, v, z, valid, width):
            inside, total = footprint_overlap(mask, u[idx], v[idx], cov2d[idx], args.sigma_extent, args.max_radius)
            if total <= 0:
                continue
            visible[idx] += 1
            weighted_hits[idx] += inside
            weighted_total[idx] += total

    score = np.divide(weighted_hits, weighted_total, out=np.zeros(len(points), dtype=np.float64), where=weighted_total > 0).astype(np.float32)
    labels = ((visible >= args.handle_min_visible) & (score >= args.handle_threshold)).astype(np.uint8)
    return score, labels, visible


def base_color_from_vertices(vertices):
    names = vertices.dtype.names
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        color = np.vstack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).T.astype(np.float32) * C0 + 0.5
        return np.clip(color, 0.0, 1.0).astype(np.float32)
    if {"red", "green", "blue"}.issubset(names):
        return (np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0).astype(np.float32)
    return np.full((len(vertices), 3), 0.5, dtype=np.float32)


def geometry_arrays(vertices):
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    scale = np.exp(np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T.astype(np.float32))
    rotation = np.vstack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).T.astype(np.float32)
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True).clip(min=1e-12)
    opacity = sigmoid(np.asarray(vertices["opacity"], dtype=np.float32))[:, None].astype(np.float32)
    color = base_color_from_vertices(vertices)
    geometry = np.concatenate([xyz, scale, rotation, opacity], axis=1).astype(np.float32)
    return xyz, scale, rotation, opacity, color, geometry


@torch.no_grad()
def dino_embeddings(row, args, object_ply, model, patch_size, hidden_size, device):
    key = scene_key(row["category"], row["scene_id"])
    source = load_json(args.work_root / "3dgs_scenes" / key / "handal_source.json")
    raw_scene = Path(source["source_scene"])
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")
    frame_ids = selected_frame_ids(raw_scene, args.max_images)
    if args.max_dino_views is not None:
        frame_ids = frame_ids[: args.max_dino_views]

    ply = PlyData.read(object_ply)
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
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        image_path = args.work_root / "3dgs_scenes" / key / "images" / f"{frame_id:06d}.jpg"
        patch_features, original_width, original_height = extract_patch_features(
            model,
            image_path,
            args.resize_width,
            args.resize_height,
            patch_size,
            device,
            True,
        )
        feat_h, feat_w = patch_features.shape[:2]
        u, v, z, cov2d, valid = project_means_and_covariances(
            points, cov_world, rot, trans, cam_k, *camera_size(cam, raw_scene, frame_id), args.min_var_px
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
            print(f"{key} DINO view {view_idx:03d}/{len(frame_ids)}: no contributions")
            continue
        before = weight_sum.copy()
        patch_ids, gaussian_ids, weights = contribs
        accumulate_weighted_features(feature_sum, weight_sum, contribution_count, patch_features, patch_ids, gaussian_ids, weights)
        visible_views[(weight_sum - before) > args.min_view_weight] += 1
        if getattr(args, "verbose_views", False):
            print(f"{key} DINO view {view_idx:03d}/{len(frame_ids)}: {len(weights)} weighted contributions")

    embeddings = np.divide(feature_sum, weight_sum[:, None], out=np.zeros_like(feature_sum), where=weight_sum[:, None] > 0)
    valid = weight_sum >= args.min_total_weight
    if np.any(valid):
        embeddings[valid] = normalize_rows(embeddings[valid]).astype(np.float32)
    return embeddings, valid, weight_sum, contribution_count, visible_views


def write_debug_ply(object_ply, handle_labels_arr, out_ply):
    ply = PlyData.read(object_ply)
    vertices = np.array(ply["vertex"].data, copy=True)
    color = base_color_from_vertices(vertices)
    color[handle_labels_arr.astype(bool)] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(vertices.dtype.names):
        sh = rgb_to_sh(color)
        vertices["f_dc_0"] = sh[:, 0]
        vertices["f_dc_1"] = sh[:, 1]
        vertices["f_dc_2"] = sh[:, 2]
    elif {"red", "green", "blue"}.issubset(vertices.dtype.names):
        rgb = np.clip(color * 255.0, 0, 255).astype(np.uint8)
        vertices["red"], vertices["green"], vertices["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(out_ply)


def save_compact_npz(row, args, object_ply, object_scores, gaussian_indices, input_gaussians, handle_scores, handle_labels_arr, handle_visible, dino_pack):
    key = scene_key(row["category"], row["scene_id"])
    vertices = PlyData.read(object_ply)["vertex"].data
    xyz, scale, rotation, opacity, color, geometry = geometry_arrays(vertices)
    embeddings, valid_embeddings, weight_sum, contribution_count, visible_views = dino_pack
    dino_dtype = np.float16 if args.dino_dtype == "float16" else np.float32
    out_path = args.npz_root / f"{key}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        dino_features=embeddings.astype(dino_dtype),
        valid_dino=valid_embeddings.astype(bool),
        dino_weight_sum=weight_sum.astype(np.float32),
        dino_contribution_count=contribution_count,
        dino_visible_views=visible_views,
        geometry_features=geometry.astype(np.float32),
        xyz=xyz.astype(np.float32),
        scale=scale.astype(np.float32),
        rotation=rotation.astype(np.float32),
        opacity=opacity.astype(np.float32),
        color=color.astype(np.float32),
        object_scores=object_scores.astype(np.float32),
        handle_scores=handle_scores.astype(np.float32),
        handle_labels_thr0_25=handle_labels_arr.astype(np.uint8),
        handle_visible_views=handle_visible,
        gaussian_indices=gaussian_indices.astype(np.int64),
        scene_key=np.array([key]),
        scene_id=np.array([row["scene_id"]]),
        category=np.array([row["category"]]),
        split=np.array([row["split"]]),
        instance_id=np.array([row["instance_id"]]),
        source_scene_path=np.array([str(local_scene_path(args.dataset_root, row))]),
        source_object_ply=np.array([str(object_ply)]),
        input_gaussians=np.array([input_gaussians], dtype=np.int32),
        object_threshold=np.array([args.object_threshold], dtype=np.float32),
        handle_threshold=np.array([args.handle_threshold], dtype=np.float32),
        dino_model=np.array([args.model_name]),
        dino_dtype=np.array([args.dino_dtype]),
        resize_width=np.array([args.resize_width], dtype=np.int32),
        resize_height=np.array([args.resize_height], dtype=np.int32),
    )
    return out_path


def build_features(rows, args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, patch_size, hidden_size = load_dinov2(args.model_name, device, args.local_files_only)
    if args.resize_width % patch_size != 0 or args.resize_height % patch_size != 0:
        raise ValueError(f"resize dimensions must be divisible by patch size {patch_size}")
    summaries = []
    for row in rows:
        key = scene_key(row["category"], row["scene_id"])
        print(f"building compact features for {key}")
        object_ply, object_scores, gaussian_indices, input_gaussians = object_prune(row, args)
        handle_scores, handle_labels_arr, handle_visible = handle_labels(row, args, object_ply)
        dino_pack = dino_embeddings(row, args, object_ply, model, patch_size, hidden_size, device)
        npz_path = save_compact_npz(
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
        debug_ply = args.ply_root / f"{key}_object_handle_red.ply"
        write_debug_ply(object_ply, handle_labels_arr, debug_ply)
        summary = {
            "scene_key": key,
            "split": row["split"],
            "category": row["category"],
            "scene_id": row["scene_id"],
            "input_gaussians": input_gaussians,
            "object_gaussians": int(len(object_scores)),
            "handle_gaussians": int(handle_labels_arr.sum()),
            "handle_ratio": float(handle_labels_arr.sum() / max(len(handle_labels_arr), 1)),
            "valid_dino": int(dino_pack[1].sum()),
            "valid_dino_ratio": float(dino_pack[1].sum() / max(len(handle_labels_arr), 1)),
            "npz": str(npz_path),
            "debug_ply": str(debug_ply),
            "object_ply": str(object_ply),
        }
        summaries.append(summary)
        print(
            f"{key}: object {summary['object_gaussians']}/{input_gaussians}, "
            f"handle {summary['handle_gaussians']} ({summary['handle_ratio']:.1%}), "
            f"valid DINO {summary['valid_dino_ratio']:.1%}"
        )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--gpus", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--init_points", type=int, default=30000)
    parser.add_argument("--force_3dgs", action="store_true")
    parser.add_argument("--model_name", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dino_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--resize_width", type=int, default=448)
    parser.add_argument("--resize_height", type=int, default=336)
    parser.add_argument("--max_dino_views", type=int, default=None)
    parser.add_argument("--object_threshold", type=float, default=0.60)
    parser.add_argument("--object_min_visible", type=int, default=20)
    parser.add_argument("--handle_threshold", type=float, default=0.25)
    parser.add_argument("--handle_min_visible", type=int, default=10)
    parser.add_argument("--sigma_extent", type=float, default=3.0)
    parser.add_argument("--max_radius", type=int, default=24)
    parser.add_argument("--min_var_px", type=float, default=0.25)
    parser.add_argument("--max_patch_radius", type=int, default=4)
    parser.add_argument("--min_patch_radius", type=int, default=1)
    parser.add_argument("--min_var_patch", type=float, default=0.25)
    parser.add_argument("--min_alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--alpha_clip", type=float, default=0.99)
    parser.add_argument("--min_total_weight", type=float, default=1e-4)
    parser.add_argument("--min_view_weight", type=float, default=1e-5)
    args = parser.parse_args()

    args.work_root = args.out_root / "work"
    args.model_root = args.work_root / "3dgs_models"
    args.log_root = args.work_root / "3dgs_logs"
    args.npz_root = args.out_root / "npz"
    args.ply_root = args.out_root / "ply"
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.npz_root.mkdir(parents=True, exist_ok=True)
    args.ply_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.dataset_root / "manifests" / "manifest.csv"
    rows = select_pilot_rows(load_manifest(manifest_path), args.dataset_root)
    write_json(
        args.out_root / "pilot_selection.json",
        [
            {
                "scene_key": scene_key(row["category"], row["scene_id"]),
                "split": row["split"],
                "category": row["category"],
                "scene_id": row["scene_id"],
                "instance_id": row["instance_id"],
                "num_rgb_images": row["num_rgb_images"],
            }
            for row in rows
        ],
    )
    print("Pilot scenes:")
    for row in rows:
        print(f"  {scene_key(row['category'], row['scene_id'])} ({row['split']}, instance {row['instance_id']})")

    train_3dgs(rows, args)
    summaries = build_features(rows, args)
    write_json(
        args.out_root / "pilot_summary.json",
        {
            "mode": "pilot_1_scene_per_category",
            "stored_gaussian_set": "object_pruned_only",
            "dino_dtype": args.dino_dtype,
            "geometry_dtype": "float32",
            "object_threshold": args.object_threshold,
            "handle_threshold": args.handle_threshold,
            "npz_root": str(args.npz_root),
            "ply_root": str(args.ply_root),
            "scenes": summaries,
        },
    )
    print(f"Pilot summary -> {args.out_root / 'pilot_summary.json'}")


if __name__ == "__main__":
    main()
