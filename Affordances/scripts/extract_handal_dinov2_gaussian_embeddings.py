import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData


C0 = 0.28209479177387814
HANDAL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
SCENE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_3dgs_scenes")
OBJECT_PLY_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_object_pruned_ply/thr0.75")
OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_gaussian_embeddings")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def selected_frame_ids(scene_dir):
    return sorted(int(p.stem) for p in (scene_dir / "images").glob("*.jpg"))


def sigmoid(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def normalize_rows(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def safe_name(text):
    return text.replace("/", "_").replace("-", "_")


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


def project_means_and_covariances(points, cov_world, rot, trans, cam_k, width, height, min_var_px):
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
    cov2d[:, 0, 0] = a * a * sxx + 2 * a * b * sxz + b * b * szz + min_var_px
    cov2d[:, 0, 1] = a * c * sxy + a * d * sxz + b * c * syz + b * d * szz
    cov2d[:, 1, 0] = cov2d[:, 0, 1]
    cov2d[:, 1, 1] = c * c * syy + 2 * c * d * syz + d * d * szz + min_var_px
    return u, v, z, cov2d, valid


def load_dinov2(model_name, device, local_files_only):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    patch_size = int(getattr(model.config, "patch_size", 14))
    hidden_size = int(getattr(model.config, "hidden_size"))
    return model, patch_size, hidden_size


def preprocess_image(image_path, resize_width, resize_height, device):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, original_width, original_height


@torch.no_grad()
def extract_patch_features(model, image_path, resize_width, resize_height, patch_size, device, normalize):
    pixel_values, original_width, original_height = preprocess_image(
        image_path, resize_width, resize_height, device
    )
    output = model(pixel_values=pixel_values)
    tokens = output.last_hidden_state[:, 1:, :]
    feat_h = resize_height // patch_size
    feat_w = resize_width // patch_size
    features = tokens.reshape(1, feat_h, feat_w, -1).squeeze(0)
    if normalize:
        features = F.normalize(features, dim=-1)
    return features.float().cpu().numpy(), original_width, original_height


def make_view_contributions(
    u,
    v,
    z,
    cov2d_px,
    valid,
    opacities,
    original_width,
    original_height,
    feat_w,
    feat_h,
    resize_width,
    resize_height,
    patch_size,
    sigma_extent,
    max_patch_radius,
    min_patch_radius,
    min_alpha,
    alpha_clip,
    min_var_patch,
):
    sx = (resize_width / float(original_width)) / float(patch_size)
    sy = (resize_height / float(original_height)) / float(patch_size)
    scale = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)

    patch_ids = []
    gaussian_ids = []
    depths = []
    alphas = []

    for idx in np.flatnonzero(valid):
        cov_grid = scale @ cov2d_px[idx] @ scale.T
        # DINO features live on a coarse patch grid. This floor approximates
        # integration over a patch area, so small splats still vote to the
        # token they visually occupy.
        cov_grid[0, 0] = max(cov_grid[0, 0], min_var_patch)
        cov_grid[1, 1] = max(cov_grid[1, 1], min_var_patch)
        det = cov_grid[0, 0] * cov_grid[1, 1] - cov_grid[0, 1] * cov_grid[1, 0]
        if det <= 1e-10:
            continue

        eigvals = np.linalg.eigvalsh(cov_grid)
        max_sigma = np.sqrt(max(eigvals[-1], 1e-8))
        radius = int(np.ceil(sigma_extent * max_sigma))
        radius = max(min_patch_radius, min(radius, max_patch_radius))

        cx = u[idx] * sx
        cy = v[idx] * sy
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(feat_w - 1, int(np.ceil(cx + radius)))
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(feat_h - 1, int(np.ceil(cy + radius)))
        if x1 < x0 or y1 < y0:
            continue

        inv = np.array(
            [[cov_grid[1, 1], -cov_grid[0, 1]], [-cov_grid[1, 0], cov_grid[0, 0]]],
            dtype=np.float64,
        ) / det

        xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
        dx = xs[None, :] - cx
        dy = ys[:, None] - cy
        quad = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
        support = quad <= sigma_extent * sigma_extent
        if not np.any(support):
            continue

        local_alpha = opacities[idx] * np.exp(-0.5 * quad)
        local_alpha = np.clip(local_alpha, 0.0, alpha_clip)
        support &= local_alpha >= min_alpha
        if not np.any(support):
            continue

        yy, xx = np.nonzero(support)
        px = (xx + x0).astype(np.int32)
        py = (yy + y0).astype(np.int32)
        patch_ids.extend((py * feat_w + px).tolist())
        gaussian_ids.extend([idx] * len(px))
        depths.extend([float(z[idx])] * len(px))
        alphas.extend(local_alpha[yy, xx].astype(np.float32).tolist())

    if not patch_ids:
        return None

    patch_ids = np.asarray(patch_ids, dtype=np.int32)
    gaussian_ids = np.asarray(gaussian_ids, dtype=np.int32)
    depths = np.asarray(depths, dtype=np.float32)
    alphas = np.asarray(alphas, dtype=np.float32)

    order = np.lexsort((depths, patch_ids))
    compositing_weights = np.zeros(len(patch_ids), dtype=np.float32)
    current_patch = -1
    transmittance = 1.0
    blocked = False

    for src in order:
        patch_id = int(patch_ids[src])
        if patch_id != current_patch:
            current_patch = patch_id
            transmittance = 1.0
            blocked = False
        if blocked:
            continue
        alpha = float(alphas[src])
        weight = transmittance * alpha
        compositing_weights[src] = weight
        transmittance *= 1.0 - alpha
        if transmittance < 1e-4:
            blocked = True

    keep = compositing_weights > 0.0
    return patch_ids[keep], gaussian_ids[keep], compositing_weights[keep]


def accumulate_weighted_features(feature_sum, weight_sum, contribution_count, patch_features, patch_ids, gaussian_ids, weights):
    features_flat = patch_features.reshape(-1, patch_features.shape[-1])
    np.add.at(weight_sum, gaussian_ids, weights)
    np.add.at(contribution_count, gaussian_ids, 1)

    chunk = 16384
    for start in range(0, len(weights), chunk):
        end = min(start + chunk, len(weights))
        weighted = features_flat[patch_ids[start:end]] * weights[start:end, None]
        np.add.at(feature_sum, gaussian_ids[start:end], weighted.astype(np.float32))


def extract_scene_embeddings(scene_name, args, method_dir, model, patch_size, hidden_size, device):
    scene_dir = SCENE_ROOT / scene_name
    source = load_json(scene_dir / "handal_source.json")
    raw_scene = HANDAL_ROOT / source["split"] / scene_name
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")
    frame_ids = selected_frame_ids(scene_dir)
    if args.max_views is not None:
        frame_ids = frame_ids[: args.max_views]

    src_ply = args.object_ply_root / f"handal_mug_scene_{scene_name}_object_pruned_thr0.75.ply"
    ply = PlyData.read(src_ply)
    vertices = ply["vertex"].data
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    cov_world = gaussian_covariances(vertices)
    opacities = sigmoid(np.asarray(vertices["opacity"], dtype=np.float64))

    feature_sum = np.zeros((len(points), hidden_size), dtype=np.float32)
    weight_sum = np.zeros(len(points), dtype=np.float32)
    contribution_count = np.zeros(len(points), dtype=np.uint32)
    visible_views = np.zeros(len(points), dtype=np.uint16)

    for view_idx, frame_id in enumerate(frame_ids, start=1):
        key = str(frame_id)
        cam = scene_camera[key]
        gt = scene_gt[key][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        width = int(cam["width"])
        height = int(cam["height"])
        image_path = scene_dir / "images" / f"{frame_id:06d}.jpg"

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
            print(f"{scene_name} view {view_idx:03d}/{len(frame_ids)} frame {frame_id:06d}: no contributions")
            continue

        patch_ids, gaussian_ids, weights = contribs
        before = weight_sum.copy() if args.count_visible_views else None
        accumulate_weighted_features(feature_sum, weight_sum, contribution_count, patch_features, patch_ids, gaussian_ids, weights)
        if args.count_visible_views:
            touched = (weight_sum - before) > args.min_view_weight
            visible_views[touched] += 1

        print(
            f"{scene_name} view {view_idx:03d}/{len(frame_ids)} frame {frame_id:06d}: "
            f"{len(weights)} weighted Gaussian-patch contributions"
        )

    embeddings = np.divide(
        feature_sum,
        weight_sum[:, None],
        out=np.zeros_like(feature_sum),
        where=weight_sum[:, None] > 0,
    )
    valid_embeddings = weight_sum >= args.min_total_weight
    if args.normalize_output_features and np.any(valid_embeddings):
        embeddings[valid_embeddings] = normalize_rows(embeddings[valid_embeddings]).astype(np.float32)

    feature_dir = method_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    out_path = feature_dir / f"handal_mug_scene_{scene_name}_dinov2_render_contrib_features.npz"
    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),
        valid_embeddings=valid_embeddings,
        weight_sum=weight_sum,
        contribution_count=contribution_count,
        visible_views=visible_views,
        scene=np.array([scene_name]),
        source_ply=np.array([str(src_ply)]),
        model_name=np.array([args.model_name]),
        resize_width=np.array([args.resize_width], dtype=np.int32),
        resize_height=np.array([args.resize_height], dtype=np.int32),
        patch_size=np.array([patch_size], dtype=np.int32),
        sigma_extent=np.array([args.sigma_extent], dtype=np.float32),
        min_alpha=np.array([args.min_alpha], dtype=np.float32),
        alpha_clip=np.array([args.alpha_clip], dtype=np.float32),
        min_var_patch=np.array([args.min_var_patch], dtype=np.float32),
    )
    return {
        "scene": scene_name,
        "gaussians": int(len(points)),
        "valid_embeddings": int(valid_embeddings.sum()),
        "valid_ratio": float(valid_embeddings.sum() / max(len(points), 1)),
        "feature_dim": int(hidden_size),
        "views": int(len(frame_ids)),
        "features": str(out_path),
        "source_ply": str(src_ply),
    }


def load_or_extract_scene(scene_name, args, method_dir, model, patch_size, hidden_size, device):
    feature_path = method_dir / "features" / f"handal_mug_scene_{scene_name}_dinov2_render_contrib_features.npz"
    if feature_path.exists() and not args.force:
        data = np.load(feature_path)
        return {
            "scene": scene_name,
            "gaussians": int(data["embeddings"].shape[0]),
            "valid_embeddings": int(data["valid_embeddings"].sum()),
            "valid_ratio": float(data["valid_embeddings"].sum() / max(data["embeddings"].shape[0], 1)),
            "feature_dim": int(data["embeddings"].shape[1]),
            "views": int(len(selected_frame_ids(SCENE_ROOT / scene_name))),
            "features": str(feature_path),
            "source_ply": str(args.object_ply_root / f"handal_mug_scene_{scene_name}_object_pruned_thr0.75.ply"),
            "reused": True,
        }
    return extract_scene_embeddings(scene_name, args, method_dir, model, patch_size, hidden_size, device)


def fit_global_pca(method_dir, summaries, max_samples):
    samples = []
    for item in summaries:
        data = np.load(item["features"])
        embeddings = data["embeddings"]
        valid = data["valid_embeddings"]
        if np.any(valid):
            samples.append(embeddings[valid])
    if not samples:
        raise RuntimeError("No valid embeddings were produced, so PCA coloring cannot be created.")

    x = np.concatenate(samples, axis=0).astype(np.float32)
    if len(x) > max_samples:
        rng = np.random.default_rng(7)
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].astype(np.float32)

    projected_valid = []
    for item in summaries:
        data = np.load(item["features"])
        embeddings = data["embeddings"]
        valid = data["valid_embeddings"]
        if np.any(valid):
            projected_valid.append((embeddings[valid] - mean) @ components.T)
    projected_valid = np.concatenate(projected_valid, axis=0)
    lo = np.percentile(projected_valid, 1.0, axis=0)
    hi = np.percentile(projected_valid, 99.0, axis=0)
    scale = np.maximum(hi - lo, 1e-6)

    pca_path = method_dir / "summary" / "global_pca_projection.npz"
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pca_path,
        mean=mean.astype(np.float32),
        components=components,
        color_low=lo.astype(np.float32),
        color_high=hi.astype(np.float32),
    )
    return mean, components, lo, scale, pca_path


def write_pca_ply(src_ply, feature_path, out_ply, mean, components, lo, scale):
    data = np.load(feature_path)
    embeddings = data["embeddings"]
    valid = data["valid_embeddings"]
    projection = (embeddings - mean) @ components.T
    rgb = np.clip((projection - lo) / scale, 0.0, 1.0).astype(np.float32)
    rgb[~valid] = np.array([0.08, 0.08, 0.08], dtype=np.float32)

    ply = PlyData.read(src_ply)
    vertices = np.array(ply["vertex"].data, copy=True)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]

    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(out_ply)


def write_pca_outputs(method_dir, summaries, max_samples):
    mean, components, lo, scale, pca_path = fit_global_pca(method_dir, summaries, max_samples)
    pca_summaries = []
    for item in summaries:
        scene_name = item["scene"]
        src_ply = Path(item["source_ply"])
        out_ply = method_dir / "pca_ply" / f"handal_mug_scene_{scene_name}_dinov2_render_contrib_pca.ply"
        write_pca_ply(src_ply, item["features"], out_ply, mean, components, lo, scale)
        pca_summaries.append({**item, "pca_ply": str(out_ply)})
    return pca_summaries, str(pca_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/dinov2-small")
    parser.add_argument("--resize_width", type=int, default=448)
    parser.add_argument("--resize_height", type=int, default=336)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_pca", action="store_true")
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--scenes", nargs="+", default=["001001", "002001", "003001", "005001", "006001"])
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
    parser.add_argument("--count_visible_views", action="store_true", default=True)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--object_ply_root", type=Path, default=OBJECT_PLY_ROOT)
    args = parser.parse_args()

    if args.resize_width % 14 != 0 or args.resize_height % 14 != 0:
        raise ValueError("resize_width and resize_height should be divisible by the DINOv2 patch size 14.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, patch_size, hidden_size = load_dinov2(args.model_name, device, args.local_files_only)
    if args.resize_width % patch_size != 0 or args.resize_height % patch_size != 0:
        raise ValueError(f"resize dimensions must be divisible by patch size {patch_size}.")

    method_name = f"{safe_name(args.model_name)}_render_contrib_patch{args.resize_width}x{args.resize_height}"
    if args.max_views is not None:
        method_name += f"_debug{args.max_views}views"
    method_dir = args.out_root / method_name
    (method_dir / "features").mkdir(parents=True, exist_ok=True)
    (method_dir / "pca_ply").mkdir(parents=True, exist_ok=True)
    (method_dir / "summary").mkdir(parents=True, exist_ok=True)

    summaries = []
    for scene_name in args.scenes:
        print(f"Scene {scene_name}: extracting render-contribution DINOv2 embeddings")
        summary = load_or_extract_scene(scene_name, args, method_dir, model, patch_size, hidden_size, device)
        summaries.append(summary)
        print(
            f"{scene_name}: valid embeddings {summary['valid_embeddings']}/{summary['gaussians']} "
            f"({summary['valid_ratio']:.1%}) -> {summary['features']}"
        )

    pca_path = None
    if not args.no_pca:
        summaries, pca_path = write_pca_outputs(method_dir, summaries, args.max_pca_samples)
        for item in summaries:
            print(f"{item['scene']}: PCA PLY -> {item['pca_ply']}")

    summary_path = method_dir / "summary" / "dinov2_render_contrib_embedding_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "method": "DINOv2 full-image patch features lifted to object-pruned Gaussians with front-to-back T*alpha render-contribution weights",
                "model_name": args.model_name,
                "resize_width": args.resize_width,
                "resize_height": args.resize_height,
                "patch_size": patch_size,
                "source_object_ply_root": str(args.object_ply_root),
                "pca_projection": pca_path,
                "scenes": summaries,
            },
            f,
            indent=2,
        )
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
