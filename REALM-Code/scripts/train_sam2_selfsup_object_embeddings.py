import argparse
import csv
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans

from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from gaussian_renderer import render


def load_namespace_cfg(model_path):
    cfg_path = Path(model_path) / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    return eval(cfg_path.read_text(), {"Namespace": Namespace})


def feature_to_rgb_torch(features):
    # features: [C, H, W]
    c, h, w = features.shape
    x = features.detach().permute(1, 2, 0).reshape(-1, c).float().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pca = x @ vt[:3].T
    pca = pca.reshape(h, w, 3)
    lo, hi = np.percentile(pca, [1, 99])
    pca = np.clip((pca - lo) / max(hi - lo, 1e-6), 0, 1)
    return (pca * 255).astype(np.uint8)


def save_image_grid(paths, output_path, tile_width=300):
    images = [Image.open(p).convert("RGB") for p in paths if Path(p).exists()]
    if not images:
        return
    font = ImageFont.load_default()
    tiles = []
    for path, im in zip(paths, images):
        im.thumbnail((tile_width, int(tile_width * 0.75)), Image.BICUBIC)
        tile = Image.new("RGB", (tile_width + 10, int(tile_width * 0.75) + 36), (255, 255, 255))
        tile.paste(im, ((tile.width - im.width) // 2, 28 + (int(tile_width * 0.75) - im.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tile.width, 24], fill=(0, 0, 0))
        draw.text((6, 7), Path(path).stem, fill=(255, 255, 255), font=font)
        tiles.append(tile)
    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tiles[0].width, rows * tiles[0].height), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * tile.width, (i // cols) * tile.height))
    sheet.save(output_path, quality=92)


def id_color(i):
    rng = np.random.default_rng(i + 9001)
    return rng.random(3).astype(np.float32)


def initialize_objects_from_gaussian_properties(gaussians, seed=0):
    with torch.no_grad():
        xyz = gaussians._xyz.detach()
        fdc = gaussians._features_dc.detach().squeeze(1)
        opacity = gaussians.get_opacity.detach()
        scale = gaussians.get_scaling.detach()
        props = torch.cat([xyz, fdc, opacity, scale], dim=1)
        props = (props - props.mean(dim=0, keepdim=True)) / (props.std(dim=0, keepdim=True) + 1e-6)
        gen = torch.Generator(device=props.device)
        gen.manual_seed(seed)
        proj = torch.randn((props.shape[1], gaussians.num_objects), generator=gen, device=props.device)
        obj = torch.tanh(props @ proj / math.sqrt(props.shape[1]))
        obj = F.normalize(obj, dim=1)
        gaussians._objects_dc.data.copy_(obj[:, None, :])
    return props.detach()


def freeze_except_object_features(gaussians):
    for param in [
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ]:
        param.requires_grad_(False)
    gaussians._objects_dc.requires_grad_(True)


def load_sam_mask_record(mask_npz_path, image_hw, args):
    data = np.load(mask_npz_path)
    masks = data["masks"].astype(bool)
    areas = data["areas"] if "areas" in data.files else masks.reshape(masks.shape[0], -1).sum(axis=1)
    pred_ious = data["predicted_ious"] if "predicted_ious" in data.files else np.ones((masks.shape[0],), dtype=np.float32)
    stability = data["stability_scores"] if "stability_scores" in data.files else np.ones((masks.shape[0],), dtype=np.float32)
    h, w = image_hw
    image_area = h * w
    keep = []
    for i, area in enumerate(areas):
        frac = float(area) / float(image_area)
        if frac < args.min_mask_area_frac or frac > args.max_mask_area_frac:
            continue
        keep.append(i)
    if args.max_masks_per_frame > 0:
        keep = keep[: args.max_masks_per_frame]
    masks = masks[keep]
    pred_ious = pred_ious[keep]
    stability = stability[keep]
    areas = areas[keep]
    return {
        "masks": masks,
        "areas": areas.astype(np.float32),
        "predicted_ious": pred_ious.astype(np.float32),
        "stability_scores": stability.astype(np.float32),
        "num_loaded": int(len(keep)),
    }


def mask_area_weight(area_frac):
    if area_frac <= 0:
        return 0.0
    small = min(1.0, math.sqrt(area_frac / 0.02))
    large = min(1.0, math.sqrt(0.25 / area_frac))
    return small * large


def sam_pull_loss(rendered_features, mask_record, args, device):
    # rendered_features: [C, H, W]
    feat = F.normalize(rendered_features.permute(1, 2, 0).reshape(-1, rendered_features.shape[0]), dim=1)
    h, w = rendered_features.shape[1:]
    losses = []
    weights = []
    masks_np = mask_record["masks"]
    image_area = h * w
    for i, mask_np in enumerate(masks_np):
        if mask_np.shape != (h, w):
            mask_t = torch.from_numpy(mask_np.astype(np.float32))[None, None].to(device)
            mask_t = F.interpolate(mask_t, size=(h, w), mode="nearest").squeeze().bool()
        else:
            mask_t = torch.from_numpy(mask_np).to(device).bool()
        idx = torch.nonzero(mask_t.reshape(-1), as_tuple=False).squeeze(1)
        if idx.numel() < args.min_mask_pixels:
            continue
        if idx.numel() > args.pixels_per_mask:
            perm = torch.randperm(idx.numel(), device=device)[: args.pixels_per_mask]
            idx = idx[perm]
        f = feat[idx]
        proto = F.normalize(f.mean(dim=0, keepdim=True), dim=1).detach()
        loss = (1.0 - (f * proto).sum(dim=1)).mean()
        area_frac = float(mask_record["areas"][i]) / float(image_area)
        weight = float(mask_record["predicted_ious"][i]) * float(mask_record["stability_scores"][i]) * mask_area_weight(area_frac)
        losses.append(loss)
        weights.append(weight)
    if not losses:
        return rendered_features.sum() * 0.0, 0
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    weights_t = weights_t / (weights_t.sum() + 1e-6)
    return torch.stack(losses).mul(weights_t).sum(), len(losses)


def variance_covariance_loss(gaussians, args):
    z = gaussians._objects_dc.squeeze(1)
    n = z.shape[0]
    if n > args.vicreg_sample_size:
        idx = torch.randperm(n, device=z.device)[: args.vicreg_sample_size]
        z = z[idx]
    z = (z - z.mean(dim=0, keepdim=True))
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(args.vicreg_std_target - std).mean()
    z = z / (std[None, :] + 1e-6)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / z.shape[1]
    return var_loss, cov_loss


def graph_smoothness_loss(gaussians, prop_features, args):
    n = prop_features.shape[0]
    sample_n = min(args.graph_sample_size, n)
    idx = torch.randperm(n, device=prop_features.device)[:sample_n]
    prop = prop_features[idx]
    obj = F.normalize(gaussians._objects_dc.squeeze(1), dim=1)
    obj_s = obj[idx]
    d = torch.cdist(prop, prop)
    knn = d.topk(k=min(args.graph_k + 1, sample_n), largest=False).indices[:, 1:]
    neigh = obj_s[knn]
    center = obj_s[:, None, :]
    prop_dist = torch.gather(d, 1, knn)
    weight = torch.exp(-prop_dist / (prop_dist.median().detach() + 1e-6))
    loss = (1.0 - (center * neigh).sum(dim=-1)) * weight
    return loss.mean()


def select_camera_mask_pairs(cameras, masks_dir):
    masks_dir = Path(masks_dir)
    pairs = []
    for cam in cameras:
        stem = Path(cam.image_name).stem
        candidates = [
            masks_dir / f"{stem}_raw_sam2_masks.npz",
            masks_dir / f"{stem}_sam2_masks.npz",
            masks_dir / f"{stem}.npz",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is not None:
            pairs.append((cam, found))
    return pairs


def render_feature_snapshots(gaussians, cameras, pipeline, background, out_dir, tag, max_views=6):
    out_dir = Path(out_dir)
    pca_dir = out_dir / "feature_pca"
    cluster_dir = out_dir / "cluster_renders"
    pca_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir.mkdir(parents=True, exist_ok=True)
    pca_paths = []
    cluster_paths = []

    saved_obj = gaussians._objects_dc.data.clone()

    with torch.no_grad():
        for cam in cameras[:max_views]:
            pkg = render(cam, gaussians, pipeline, background)
            rgb = feature_to_rgb_torch(pkg["render_object"])
            path = pca_dir / f"{tag}_{Path(cam.image_name).stem}_feature_pca.jpg"
            Image.fromarray(rgb).save(path, quality=92)
            pca_paths.append(path)

        z = F.normalize(saved_obj.squeeze(1), dim=1).detach().cpu().numpy()
        k = min(64, max(2, int(np.sqrt(z.shape[0] // 20))))
        km = MiniBatchKMeans(n_clusters=k, batch_size=8192, random_state=0, n_init=3, max_iter=100)
        labels = km.fit_predict(z)
        colors = np.stack([id_color(int(i)) for i in labels], axis=0)
        color_features = np.zeros((z.shape[0], gaussians.num_objects), dtype=np.float32)
        color_features[:, :3] = colors
        gaussians._objects_dc.data.copy_(torch.from_numpy(color_features).to(saved_obj.device)[:, None, :])
        for cam in cameras[:max_views]:
            pkg = render(cam, gaussians, pipeline, background)
            rgb = pkg["render_object"][:3].detach().permute(1, 2, 0).cpu().numpy()
            rgb = np.clip(rgb, 0, 1)
            path = cluster_dir / f"{tag}_{Path(cam.image_name).stem}_clusters_k{k}.jpg"
            Image.fromarray((rgb * 255).astype(np.uint8)).save(path, quality=92)
            cluster_paths.append(path)

        gaussians._objects_dc.data.copy_(saved_obj)

    save_image_grid(pca_paths, out_dir / f"{tag}_feature_pca_overview.jpg")
    save_image_grid(cluster_paths, out_dir / f"{tag}_clusters_overview.jpg")


def save_kmeans_classifier(gaussians, output_path, num_classes, seed, batch_size=8192):
    z = F.normalize(gaussians._objects_dc.detach().squeeze(1), dim=1).cpu().numpy()
    num_classes = min(num_classes, z.shape[0])
    km = MiniBatchKMeans(
        n_clusters=num_classes,
        batch_size=batch_size,
        random_state=seed,
        n_init=3,
        max_iter=200,
    )
    labels = km.fit_predict(z)
    centers = km.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6)

    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    with torch.no_grad():
        classifier.weight.copy_(torch.from_numpy(centers)[:, :, None, None])
        classifier.bias.zero_()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), output_path)

    counts = np.bincount(labels, minlength=num_classes)
    metadata = {
        "num_classes": int(num_classes),
        "min_cluster_size": int(counts.min()),
        "median_cluster_size": float(np.median(counts)),
        "max_cluster_size": int(counts.max()),
    }
    output_path.with_name("classifier_kmeans_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def save_checkpoint_with_classifier(gaussians, output_dir, iteration, num_classes, seed):
    ckpt_dir = Path(output_dir) / "point_cloud" / f"iteration_{iteration}"
    gaussians.save_ply(ckpt_dir / "point_cloud.ply")
    return save_kmeans_classifier(gaussians, ckpt_dir / "classifier.pth", num_classes, seed)


def main():
    parser = argparse.ArgumentParser(description="Self-supervised 16D Gaussian object embedding training from SAM2 masks.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--source_path", default="")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--sam_masks_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lambda_sam", type=float, default=1.0)
    parser.add_argument("--lambda_var", type=float, default=0.5)
    parser.add_argument("--lambda_cov", type=float, default=0.05)
    parser.add_argument("--lambda_graph", type=float, default=0.1)
    parser.add_argument("--min_mask_area_frac", type=float, default=0.0005)
    parser.add_argument("--max_mask_area_frac", type=float, default=0.35)
    parser.add_argument("--min_mask_pixels", type=int, default=128)
    parser.add_argument("--pixels_per_mask", type=int, default=2048)
    parser.add_argument("--max_masks_per_frame", type=int, default=80)
    parser.add_argument("--vicreg_sample_size", type=int, default=50000)
    parser.add_argument("--vicreg_std_target", type=float, default=0.25)
    parser.add_argument("--graph_sample_size", type=int, default=2000)
    parser.add_argument("--graph_k", type=int, default=8)
    parser.add_argument("--classifier_num_classes", type=int, default=0)
    parser.add_argument("--snapshot_iterations", nargs="+", type=int, default=[0, 100, 300, 500])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_namespace_cfg(args.model_path)
    cfg.model_path = args.model_path
    if args.source_path:
        cfg.source_path = args.source_path
    cfg.sh_degree = getattr(cfg, "sh_degree", 0)
    classifier_num_classes = args.classifier_num_classes or getattr(cfg, "num_classes", 200)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2))
    render_cfg = Namespace(**vars(cfg).copy())
    render_cfg.model_path = str(out)
    (out / "cfg_args").write_text(str(render_cfg))

    gaussians = GaussianModel(cfg.sh_degree)
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    freeze_except_object_features(gaussians)
    prop_features = initialize_objects_from_gaussian_properties(gaussians, args.seed)
    optimizer = torch.optim.Adam([gaussians._objects_dc], lr=args.lr)

    pipe_parser = argparse.ArgumentParser()
    pp = PipelineParams(pipe_parser)
    pipe = pp.extract(Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False))
    background = torch.tensor([1, 1, 1] if cfg.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    pairs = select_camera_mask_pairs(scene.getTrainCameras(), args.sam_masks_dir)
    if not pairs:
        raise RuntimeError(f"No camera/mask pairs found in {args.sam_masks_dir}")
    print(f"Matched {len(pairs)} camera/mask pairs")
    eval_cams = [p[0] for p in pairs[: min(8, len(pairs))]]
    render_feature_snapshots(gaussians, eval_cams, pipe, background, out / "snapshots", "iter_000000")

    mask_cache = {}
    log_path = out / "loss_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "frame", "loss", "sam", "var", "cov", "graph", "num_masks"])
        writer.writeheader()

    snapshot_set = set(args.snapshot_iterations)
    for iteration in range(1, args.iterations + 1):
        cam, mask_path = pairs[(iteration - 1) % len(pairs)]
        if mask_path not in mask_cache:
            # Need camera-render size, not source image size.
            mask_cache[mask_path] = load_sam_mask_record(mask_path, (cam.image_height, cam.image_width), args)
        mask_record = mask_cache[mask_path]

        pkg = render(cam, gaussians, pipe, background)
        sam_loss, num_masks = sam_pull_loss(pkg["render_object"], mask_record, args, gaussians._objects_dc.device)
        var_loss, cov_loss = variance_covariance_loss(gaussians, args)
        graph_loss = graph_smoothness_loss(gaussians, prop_features, args)
        loss = (
            args.lambda_sam * sam_loss
            + args.lambda_var * var_loss
            + args.lambda_cov * cov_loss
            + args.lambda_graph * graph_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0 or iteration == 1:
            row = {
                "iter": iteration,
                "frame": cam.image_name,
                "loss": float(loss.detach().cpu()),
                "sam": float(sam_loss.detach().cpu()),
                "var": float(var_loss.detach().cpu()),
                "cov": float(cov_loss.detach().cpu()),
                "graph": float(graph_loss.detach().cpu()),
                "num_masks": num_masks,
            }
            with log_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)
            print(json.dumps(row), flush=True)

        if iteration in snapshot_set:
            render_feature_snapshots(gaussians, eval_cams, pipe, background, out / "snapshots", f"iter_{iteration:06d}")
            save_checkpoint_with_classifier(gaussians, out, iteration, classifier_num_classes, args.seed)

    final_classifier_meta = save_checkpoint_with_classifier(gaussians, out, args.iterations, classifier_num_classes, args.seed)
    render_feature_snapshots(gaussians, eval_cams, pipe, background, out / "snapshots", f"iter_{args.iterations:06d}_final")

    summary = {
        "matched_pairs": len(pairs),
        "final_ply": str(out / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"),
        "final_classifier": str(out / "point_cloud" / f"iteration_{args.iterations}" / "classifier.pth"),
        "final_classifier_metadata": final_classifier_meta,
        "loss_log": str(log_path),
        "snapshots": str(out / "snapshots"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
