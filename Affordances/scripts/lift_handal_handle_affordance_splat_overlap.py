import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


C0 = 0.28209479177387814
HANDAL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
SCENE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_3dgs_scenes")
OBJECT_PLY_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_object_pruned_ply/thr0.75")
OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_handle_affordance_ply/object_thr0.75_splat_overlap")


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


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
    radius = int(np.ceil(sigma_extent * max_sigma))
    radius = max(1, min(radius, max_radius))

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


def color_affordance_ply(src_ply, dst_ply, is_handle):
    ply = PlyData.read(src_ply)
    vertices = np.array(ply["vertex"].data, copy=True)
    non_handle = rgb_to_sh([0.62, 0.62, 0.62])
    handle = rgb_to_sh([1.0, 0.08, 0.0])

    vertices["f_dc_0"] = non_handle[0]
    vertices["f_dc_1"] = non_handle[1]
    vertices["f_dc_2"] = non_handle[2]
    vertices["f_dc_0"][is_handle] = handle[0]
    vertices["f_dc_1"][is_handle] = handle[1]
    vertices["f_dc_2"][is_handle] = handle[2]

    ply["vertex"].data = vertices
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(dst_ply)


def lift_scene(scene_name, threshold, min_visible, sigma_extent, max_radius, min_var, use_center_zbuffer):
    scene_dir = SCENE_ROOT / scene_name
    source = load_json(scene_dir / "handal_source.json")
    raw_scene = HANDAL_ROOT / source["split"] / scene_name
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")

    src_ply = OBJECT_PLY_ROOT / f"handal_mug_scene_{scene_name}_object_pruned_thr0.75.ply"
    ply = PlyData.read(src_ply)
    vertices = ply["vertex"].data
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    cov_world = gaussian_covariances(vertices)

    weighted_hits = np.zeros(len(points), dtype=np.float64)
    weighted_total = np.zeros(len(points), dtype=np.float64)
    visible = np.zeros(len(points), dtype=np.uint16)

    for frame_id in selected_frame_ids(scene_dir):
        key = str(frame_id)
        cam = scene_camera[key]
        gt = scene_gt[key][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        width = int(cam["width"])
        height = int(cam["height"])
        mask = load_mask(raw_scene / "mask_parts" / f"{frame_id:06d}_000000_handle.png")

        u, v, z, cov2d, valid = project_means_and_covariances(
            points, cov_world, rot, trans, cam_k, width, height, min_var
        )
        indices = center_zbuffer_filter(u, v, z, valid, width) if use_center_zbuffer else np.flatnonzero(valid)
        for idx in indices:
            inside, total = footprint_overlap(mask, u[idx], v[idx], cov2d[idx], sigma_extent, max_radius)
            if total <= 0:
                continue
            visible[idx] += 1
            weighted_hits[idx] += inside
            weighted_total[idx] += total

    score = np.divide(
        weighted_hits,
        weighted_total,
        out=np.zeros(len(points), dtype=np.float64),
        where=weighted_total > 0,
    ).astype(np.float32)
    is_handle = (visible >= min_visible) & (score >= threshold)

    method = "center_zbuffer" if use_center_zbuffer else "all_projected"
    out_dir = OUT_ROOT / f"handle_thr{threshold:.2f}_{method}"
    out_ply = out_dir / "ply" / f"handal_mug_scene_{scene_name}_handle_affordance_splat_overlap_thr{threshold:.2f}.ply"
    score_path = out_dir / "scores" / f"handal_mug_scene_{scene_name}_handle_scores_splat_overlap_thr{threshold:.2f}.npz"
    color_affordance_ply(src_ply, out_ply, is_handle)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        score_path,
        handle_score=score,
        weighted_handle_hits=weighted_hits,
        weighted_total=weighted_total,
        visible_hits=visible,
        is_handle=is_handle,
        threshold=np.array([threshold], dtype=np.float32),
        min_visible=np.array([min_visible], dtype=np.int32),
        sigma_extent=np.array([sigma_extent], dtype=np.float32),
        max_radius=np.array([max_radius], dtype=np.int32),
        min_var=np.array([min_var], dtype=np.float32),
        use_center_zbuffer=np.array([use_center_zbuffer], dtype=np.bool_),
    )

    return {
        "scene": scene_name,
        "input_gaussians": int(len(points)),
        "handle_gaussians": int(is_handle.sum()),
        "handle_ratio": float(is_handle.sum() / max(len(points), 1)),
        "threshold": threshold,
        "min_visible": min_visible,
        "method": method,
        "sigma_extent": sigma_extent,
        "max_radius": max_radius,
        "min_var": min_var,
        "ply": str(out_ply),
        "scores": str(score_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min_visible", type=int, default=10)
    parser.add_argument("--sigma_extent", type=float, default=3.0)
    parser.add_argument("--max_radius", type=int, default=24)
    parser.add_argument("--min_var", type=float, default=0.25)
    parser.add_argument("--use_center_zbuffer", action="store_true")
    parser.add_argument("--scenes", nargs="+", default=["001001", "002001", "003001", "005001", "006001"])
    args = parser.parse_args()

    method = "center_zbuffer" if args.use_center_zbuffer else "all_projected"
    out_dir = OUT_ROOT / f"handle_thr{args.threshold:.2f}_{method}"
    (out_dir / "ply").mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)

    summaries = [
        lift_scene(
            scene,
            args.threshold,
            args.min_visible,
            args.sigma_extent,
            args.max_radius,
            args.min_var,
            args.use_center_zbuffer,
        )
        for scene in args.scenes
    ]

    summary_path = out_dir / "summary" / f"handle_affordance_summary_splat_overlap_thr{args.threshold:.2f}.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    for item in summaries:
        print(
            f"{item['scene']}: handle {item['handle_gaussians']}/{item['input_gaussians']} "
            f"({item['handle_ratio']:.1%}) -> {item['ply']}"
        )


if __name__ == "__main__":
    main()
