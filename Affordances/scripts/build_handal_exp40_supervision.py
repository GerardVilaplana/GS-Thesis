import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


HANDAL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
SCENE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_3dgs_scenes")
SPLIT_PATH = Path("/home/gvilaplana/GS-Thesis/Affordances/configs/handal_exp40_split.json")
GS_PLY_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_3dgs_exp40_ply")
OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_exp40_supervision")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_mask(path):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def selected_frame_ids(scene_dir):
    return sorted(int(p.stem) for p in (scene_dir / "images").glob("*.jpg"))


def sigmoid(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def quat_to_rotmat(q):
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


def gaussian_covariances(vertices):
    scales = np.exp(
        np.vstack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).T.astype(np.float64)
    )
    quats = np.vstack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).T
    rots = quat_to_rotmat(quats)
    scaled_rots = rots * scales[:, None, :]
    return scaled_rots @ np.transpose(scaled_rots, (0, 2, 1))


def project_points(points, rot, trans, cam_k, width, height):
    cam = points @ rot.T + trans[None, :]
    z = cam[:, 2]
    valid = z > 1e-5
    u = cam_k[0] * cam[:, 0] / np.maximum(z, 1e-8) + cam_k[2]
    v = cam_k[4] * cam[:, 1] / np.maximum(z, 1e-8) + cam_k[5]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    valid &= ui >= 0
    valid &= ui < width
    valid &= vi >= 0
    valid &= vi < height
    return ui, vi, valid


def project_means_and_covariances(points, cov_world, rot, trans, cam_k, width, height, min_var):
    cam = points @ rot.T + trans[None, :]
    z = cam[:, 2]
    valid_z = z > 1e-5
    u = np.full(len(points), -1.0, dtype=np.float64)
    v = np.full(len(points), -1.0, dtype=np.float64)
    u[valid_z] = cam_k[0] * cam[valid_z, 0] / z[valid_z] + cam_k[2]
    v[valid_z] = cam_k[4] * cam[valid_z, 1] / z[valid_z] + cam_k[5]
    valid = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    cov_cam = rot[None, :, :] @ cov_world @ rot.T[None, :, :]
    x = cam[:, 0]
    y = cam[:, 1]
    fx = cam_k[0]
    fy = cam_k[4]
    safe_z = np.maximum(z, 1e-8)
    a = fx / safe_z
    b = -fx * x / (safe_z * safe_z)
    c = fy / safe_z
    d = -fy * y / (safe_z * safe_z)

    sxx = cov_cam[:, 0, 0]
    sxy = cov_cam[:, 0, 1]
    sxz = cov_cam[:, 0, 2]
    syy = cov_cam[:, 1, 1]
    syz = cov_cam[:, 1, 2]
    szz = cov_cam[:, 2, 2]

    cov2d = np.empty((len(points), 2, 2), dtype=np.float64)
    cov2d[:, 0, 0] = a * a * sxx + 2 * a * b * sxz + b * b * szz + min_var
    cov2d[:, 0, 1] = a * c * sxy + a * d * sxz + b * c * syz + b * d * szz
    cov2d[:, 1, 0] = cov2d[:, 0, 1]
    cov2d[:, 1, 1] = c * c * syy + 2 * c * d * syz + d * d * szz + min_var
    return u, v, z, cov2d, valid


def center_zbuffer_filter(u, v, z, valid, width):
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return valid_idx
    ui = np.rint(u[valid_idx]).astype(np.int32)
    vi = np.rint(v[valid_idx]).astype(np.int32)
    pix = vi.astype(np.int64) * int(width) + ui.astype(np.int64)
    order = np.lexsort((z[valid_idx], pix))
    sorted_idx = valid_idx[order]
    sorted_pix = pix[order]
    first = np.r_[True, sorted_pix[1:] != sorted_pix[:-1]]
    return sorted_idx[first]


def footprint_overlap(mask, cx, cy, cov, sigma_extent, max_radius):
    vals = np.linalg.eigvalsh(cov)
    max_sigma = np.sqrt(max(vals[-1], 1e-8))
    radius = max(1, min(int(np.ceil(sigma_extent * max_sigma)), max_radius))
    h, w = mask.shape
    x0 = max(0, int(np.floor(cx)) - radius)
    x1 = min(w - 1, int(np.floor(cx)) + radius)
    y0 = max(0, int(np.floor(cy)) - radius)
    y1 = min(h - 1, int(np.floor(cy)) + radius)
    if x1 < x0 or y1 < y0:
        return 0.0, 0.0
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    if det <= 1e-10:
        return 0.0, 0.0
    inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]], dtype=np.float64) / det
    xs = np.arange(x0, x1 + 1, dtype=np.float64)
    ys = np.arange(y0, y1 + 1, dtype=np.float64)
    dx = xs[None, :] - cx
    dy = ys[:, None] - cy
    quad = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
    support = quad <= sigma_extent * sigma_extent
    if not np.any(support):
        return 0.0, 0.0
    weights = np.exp(-0.5 * quad) * support
    total = float(weights.sum())
    inside = float(weights[mask[y0 : y1 + 1, x0 : x1 + 1]].sum())
    return inside, total


def write_filtered_ply(src_ply, dst_ply, keep):
    ply = PlyData.read(src_ply)
    ply["vertex"].data = ply["vertex"].data[keep]
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(dst_ply)


def scene_paths(scene):
    scene_dir = SCENE_ROOT / scene
    source = load_json(scene_dir / "handal_source.json")
    raw_scene = HANDAL_ROOT / source["split"] / scene
    return scene_dir, source, raw_scene


def object_prune(scene, threshold, min_visible, out_root, gs_ply_root):
    scene_dir, source, raw_scene = scene_paths(scene)
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")
    src_ply = gs_ply_root / f"handal_mug_scene_{scene}_iter3000.ply"
    ply = PlyData.read(src_ply)
    points = np.vstack([ply["vertex"].data["x"], ply["vertex"].data["y"], ply["vertex"].data["z"]]).T.astype(np.float64)
    hits = np.zeros(len(points), dtype=np.uint16)
    visible = np.zeros(len(points), dtype=np.uint16)

    for frame_id in selected_frame_ids(scene_dir):
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        mask = load_mask(raw_scene / "mask" / f"{frame_id:06d}_000000.png")
        ui, vi, valid = project_points(points, rot, trans, cam_k, int(cam["width"]), int(cam["height"]))
        valid_idx = np.flatnonzero(valid)
        visible[valid_idx] += 1
        inside_idx = valid_idx[mask[vi[valid_idx], ui[valid_idx]]]
        hits[inside_idx] += 1

    score = np.divide(hits, visible, out=np.zeros(len(points), dtype=np.float32), where=visible > 0)
    keep = (visible >= min_visible) & (score >= threshold)
    out_ply = out_root / "object_thr0.75_ply" / f"handal_mug_scene_{scene}_object_pruned_thr0.75.ply"
    score_path = out_root / "object_scores" / f"handal_mug_scene_{scene}_object_scores.npz"
    write_filtered_ply(src_ply, out_ply, keep)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_path, object_score=score, object_hits=hits, visible_hits=visible, keep=keep)
    return out_ply, int(keep.sum()), int(len(points))


def handle_labels(scene, object_ply, threshold, min_visible, out_root, sigma_extent, max_radius, min_var):
    scene_dir, source, raw_scene = scene_paths(scene)
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")
    ply = PlyData.read(object_ply)
    vertices = ply["vertex"].data
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    cov_world = gaussian_covariances(vertices)
    weighted_hits = np.zeros(len(points), dtype=np.float64)
    weighted_total = np.zeros(len(points), dtype=np.float64)
    visible = np.zeros(len(points), dtype=np.uint16)

    for frame_id in selected_frame_ids(scene_dir):
        cam = scene_camera[str(frame_id)]
        gt = scene_gt[str(frame_id)][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        mask = load_mask(raw_scene / "mask_parts" / f"{frame_id:06d}_000000_handle.png")
        u, v, z, cov2d, valid = project_means_and_covariances(
            points, cov_world, rot, trans, cam_k, int(cam["width"]), int(cam["height"]), min_var
        )
        for idx in center_zbuffer_filter(u, v, z, valid, int(cam["width"])):
            inside, total = footprint_overlap(mask, u[idx], v[idx], cov2d[idx], sigma_extent, max_radius)
            if total <= 0:
                continue
            visible[idx] += 1
            weighted_hits[idx] += inside
            weighted_total[idx] += total

    score = np.divide(weighted_hits, weighted_total, out=np.zeros(len(points), dtype=np.float64), where=weighted_total > 0).astype(np.float32)
    is_handle = (visible >= min_visible) & (score >= threshold)
    label_path = out_root / "handle_labels_thr0.25" / f"handal_mug_scene_{scene}_handle_labels_splat_overlap_thr0.25.npz"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        label_path,
        handle_score=score,
        weighted_handle_hits=weighted_hits,
        weighted_total=weighted_total,
        visible_hits=visible,
        is_handle=is_handle,
        threshold=np.array([threshold], dtype=np.float32),
        min_visible=np.array([min_visible], dtype=np.int32),
    )
    return label_path, int(is_handle.sum()), int(len(points))


def load_scenes(split_path):
    split = load_json(split_path)
    return split["train"] + split["test"]


def process_item(item, params):
    scene = item["scene"]
    out_root = Path(params["out_root"])
    object_ply, kept, total = object_prune(
        scene,
        params["object_threshold"],
        params["object_min_visible"],
        out_root,
        Path(params["gs_ply_root"]),
    )
    label_path, handle, kept_again = handle_labels(
        scene,
        object_ply,
        params["handle_threshold"],
        params["handle_min_visible"],
        out_root,
        params["sigma_extent"],
        params["max_radius"],
        params["min_var"],
    )
    return {
        "split": item["split"],
        "scene": scene,
        "input_gaussians": total,
        "object_gaussians": kept,
        "handle_gaussians": handle,
        "handle_ratio": float(handle / max(kept_again, 1)),
        "object_ply": str(object_ply),
        "handle_labels": str(label_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--gs_ply_root", type=Path, default=GS_PLY_ROOT)
    parser.add_argument("--object_threshold", type=float, default=0.75)
    parser.add_argument("--object_min_visible", type=int, default=20)
    parser.add_argument("--handle_threshold", type=float, default=0.25)
    parser.add_argument("--handle_min_visible", type=int, default=10)
    parser.add_argument("--sigma_extent", type=float, default=3.0)
    parser.add_argument("--max_radius", type=int, default=24)
    parser.add_argument("--min_var", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    params = {
        "out_root": str(args.out_root),
        "gs_ply_root": str(args.gs_ply_root),
        "object_threshold": args.object_threshold,
        "object_min_visible": args.object_min_visible,
        "handle_threshold": args.handle_threshold,
        "handle_min_visible": args.handle_min_visible,
        "sigma_extent": args.sigma_extent,
        "max_radius": args.max_radius,
        "min_var": args.min_var,
    }
    items = load_scenes(args.split_path)
    summaries = []
    if args.workers <= 1:
        for item in items:
            summary = process_item(item, params)
            summaries.append(summary)
            print(
                f"{summary['scene']}: object {summary['object_gaussians']}/{summary['input_gaussians']}, "
                f"handle {summary['handle_gaussians']}/{summary['object_gaussians']} ({summary['handle_ratio']:.1%})"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_item, item, params) for item in items]
            for done, future in enumerate(as_completed(futures), start=1):
                summary = future.result()
                summaries.append(summary)
                remaining = len(futures) - done
                print(
                    f"[{done}/{len(futures)}; remaining {remaining}] {summary['scene']}: "
                    f"object {summary['object_gaussians']}/{summary['input_gaussians']}, "
                    f"handle {summary['handle_gaussians']}/{summary['object_gaussians']} "
                    f"({summary['handle_ratio']:.1%})"
                )

    order = {item["scene"]: i for i, item in enumerate(items)}
    summaries.sort(key=lambda item: order[item["scene"]])
    summary_path = args.out_root / "summary" / "supervision_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
