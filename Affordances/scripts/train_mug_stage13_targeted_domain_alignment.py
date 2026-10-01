#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))
import train_handal_17cat_mlp_baselines as base  # noqa: E402
from train_handal_17cat_pointnet_global import PointNetGlobal  # noqa: E402
import train_mug_synthetic_domain_gap as prev  # noqa: E402
import train_mug_domain_gap_stage7_normalization as stage7  # noqa: E402


SYN_ROOT = BASE_ROOT / "data" / "affordsplat_mug_grasp_features_v1"
HANDAL_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
OUT_ROOT = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "13_mug_targeted_domain_alignment_v1"
)

FEATURE_VARIANTS = [
    "xyz",
    "xyzrgb",
    "xyzrgb_scene_color",
    "geometry_color_no_scale_opacity",
    "geometry_color_scene_norm",
]
FOCUSED_VARIANTS = ["xyz", "xyzrgb", "geometry_color_scene_norm"]

LABEL_TRANSFORMS = {
    "original": {"mode": "none", "radius": 0.0},
    "dilate_small": {"mode": "dilate", "radius": 0.08},
    "dilate_medium": {"mode": "dilate", "radius": 0.14},
    "erode_small": {"mode": "erode", "radius": 0.05},
}

DATA_TRANSFORMS = {
    "clean": {"xyz_jitter": 0.0, "opacity_match": False, "floaters": 0.0},
    "xyz_jitter": {"xyz_jitter": 0.035, "opacity_match": False, "floaters": 0.0},
    "opacity_match": {"xyz_jitter": 0.0, "opacity_match": True, "floaters": 0.0},
    "floaters": {"xyz_jitter": 0.0, "opacity_match": False, "floaters": 0.25},
    "combined": {"xyz_jitter": 0.025, "opacity_match": True, "floaters": 0.25},
}


def load_feature_arrays(item, variant):
    if variant == "xyz":
        with np.load(item["npz"], allow_pickle=False) as z:
            x = base.normalize_xyz(z["xyz"].astype(np.float32))
            y = z["handle_labels_thr0_25"].astype(np.float32)
        return x.astype(np.float32, copy=False), y
    return stage7.load_scene_arrays(item, variant)


def pairwise_min_dist(a, b, chunk=4096):
    if len(b) == 0:
        return np.full(len(a), np.inf, dtype=np.float32)
    if len(a) == 0:
        return np.empty(0, dtype=np.float32)
    tree = cKDTree(b.astype(np.float32, copy=False))
    dist, _ = tree.query(a.astype(np.float32, copy=False), k=1, workers=-1)
    return dist.astype(np.float32, copy=False)


def transform_labels(x, y, label_transform):
    cfg = LABEL_TRANSFORMS[label_transform]
    if cfg["mode"] == "none":
        return y.astype(np.float32, copy=True)
    xyz = x[:, :3]
    y_bool = y.astype(bool)
    if cfg["mode"] == "dilate":
        d = pairwise_min_dist(xyz, xyz[y_bool])
        return np.logical_or(y_bool, d <= cfg["radius"]).astype(np.float32)
    if cfg["mode"] == "erode":
        d = pairwise_min_dist(xyz, xyz[~y_bool])
        return np.logical_and(y_bool, d > cfg["radius"]).astype(np.float32)
    raise ValueError(f"Unknown label transform: {label_transform}")


def quantile_match_column(values, target_sorted):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or target_sorted.size == 0:
        return values
    order = np.argsort(values)
    ranks = np.linspace(0, len(target_sorted) - 1, len(values)).round().astype(np.int64)
    mapped = np.empty_like(values)
    mapped[order] = target_sorted[ranks]
    return mapped


def build_target_stats(handal_train, variants):
    stats = {}
    for variant in variants:
        neg_rows = []
        opacity_values = []
        for item in handal_train:
            x, y = load_feature_arrays(item, variant)
            neg = x[y < 0.5]
            if len(neg):
                take = min(len(neg), 4096)
                rng = np.random.default_rng(base.stable_key(item["scene_key"], 17))
                idx = rng.choice(len(neg), size=take, replace=False)
                neg_rows.append(neg[idx].astype(np.float32))
            if x.shape[1] >= 11 and variant in {"current_geometry_color", "geometry_color_scene_norm"}:
                opacity_values.append(x[:, 10].astype(np.float32))
        stats[variant] = {
            "negative_pool": np.concatenate(neg_rows, axis=0) if neg_rows else None,
            "opacity_sorted": np.sort(np.concatenate(opacity_values)) if opacity_values else None,
        }
    return stats


def apply_data_transform(x, data_transform, variant, rng, target_stats):
    cfg = DATA_TRANSFORMS[data_transform]
    x = x.astype(np.float32, copy=True)
    if cfg["xyz_jitter"] > 0:
        x[:, :3] += rng.normal(0.0, cfg["xyz_jitter"], size=x[:, :3].shape).astype(np.float32)
    if cfg["opacity_match"] and x.shape[1] >= 11 and variant in {"current_geometry_color", "geometry_color_scene_norm"}:
        target = target_stats.get(variant, {}).get("opacity_sorted")
        if target is not None and len(target):
            x[:, 10] = quantile_match_column(x[:, 10], target)
    return x


def make_scene(item, variant, label_transform, data_transform, seed, target_stats, apply_to_source=True):
    x, y = load_feature_arrays(item, variant)
    rng = np.random.default_rng(seed + base.stable_key(item["scene_key"], seed) % 1_000_000)
    if apply_to_source and item["source_domain"] == "affordsplat":
        y = transform_labels(x, y, label_transform)
        x = apply_data_transform(x, data_transform, variant, rng, target_stats)
        floater_ratio = DATA_TRANSFORMS[data_transform]["floaters"]
        pool = target_stats.get(variant, {}).get("negative_pool")
        if floater_ratio > 0 and pool is not None and len(pool):
            n = max(1, int(round(len(x) * floater_ratio)))
            idx = rng.choice(len(pool), size=n, replace=True)
            x = np.concatenate([x, pool[idx].astype(np.float32)], axis=0)
            y = np.concatenate([y.astype(np.float32), np.zeros(n, dtype=np.float32)], axis=0)
    return {"item": item, "x": x.astype(np.float32, copy=False), "y": y.astype(np.float32, copy=False)}


def fit_standardizer(scenes):
    total = 0
    sum_x = None
    sumsq_x = None
    for scene in scenes:
        x = scene["x"].astype(np.float64, copy=False)
        sum_x = x.sum(axis=0) if sum_x is None else sum_x + x.sum(axis=0)
        sumsq_x = np.square(x).sum(axis=0) if sumsq_x is None else sumsq_x + np.square(x).sum(axis=0)
        total += len(x)
    mean64 = sum_x / max(total, 1)
    var64 = (sumsq_x / max(total, 1)) - np.square(mean64)
    mean = mean64.astype(np.float32)[None, :]
    std = np.sqrt(np.maximum(var64, 1e-12)).astype(np.float32)[None, :]
    return mean, np.maximum(std, 1e-6)


def standardize_scenes(scenes, mean, std):
    return [
        {
            "item": scene["item"],
            "x": ((scene["x"] - mean) / std).astype(np.float32, copy=False),
            "y": scene["y"].astype(np.float32, copy=False),
        }
        for scene in scenes
    ]


def predict_pointnet(model, scenes, device):
    model.eval()
    scores, ys, slices = [], [], []
    offset = 0
    with torch.no_grad():
        for scene in scenes:
            logits = model(torch.from_numpy(scene["x"]).to(device))
            score = base.sigmoid_np(logits.detach().cpu().numpy())
            scores.append(score.astype(np.float32))
            ys.append(scene["y"].astype(np.float32))
            end = offset + len(score)
            slices.append((scene["item"], offset, end))
            offset = end
    return np.concatenate(ys), np.concatenate(scores), slices


def train_pointnet(spec, scenes_train, scenes_val, scenes_test, args, device):
    model_name = f"pointnet_{spec['feature_variant']}"
    model_dir = args.out_root / spec["run_name"] / model_name
    if args.skip_existing and (model_dir / "overall_metrics.json").exists():
        print(f"[skip] {spec['run_name']}/{model_name}", flush=True)
        with open(model_dir / "overall_metrics.json", "r") as f:
            return json.load(f)

    model = PointNetGlobal(scenes_train[0]["x"].shape[1], latent_dim=args.latent_dim).to(device)
    train_n = sum(len(s["y"]) for s in scenes_train)
    train_pos = sum(float(s["y"].sum()) for s in scenes_train)
    pos_weight = (train_n - train_pos) / max(train_pos, 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best, best_state, stale = None, None, 0
    history = []
    order = list(range(len(scenes_train)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for idx in order:
            scene = scenes_train[idx]
            x = torch.from_numpy(scene["x"]).to(device)
            y = torch.from_numpy(scene["y"]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        y_val, val_score, val_slices = predict_pointnet(model, scenes_val, device)
        th, _ = base.choose_threshold(y_val, val_score, val_slices)
        val_rows = base.per_scene_metrics(val_slices, y_val, val_score, th["threshold"])
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_scene_iou": base.macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": base.macro_scene_score(val_rows, "f1"),
            "val_threshold": th["threshold"],
        }
        history.append(rec)
        score = (rec["val_macro_scene_iou"], rec["val_macro_scene_f1"])
        if best is None or score > (best["val_macro_scene_iou"], best["val_macro_scene_f1"]):
            best = dict(rec)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[{spec['run_name']}/{model_name}] epoch {epoch:03d} "
                f"loss={rec['loss']:.4f} val_iou={rec['val_macro_scene_iou']:.3f} "
                f"th={rec['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"[{spec['run_name']}/{model_name}] early stopping at {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    y_val, val_score, val_slices = predict_pointnet(model, scenes_val, device)
    th, th_curve = base.choose_threshold(y_val, val_score, val_slices)
    threshold = th["threshold"]
    y_test, test_score, test_slices = predict_pointnet(model, scenes_test, device)
    test_rows = base.per_scene_metrics(test_slices, y_test, test_score, threshold)
    cat_rows = base.per_category_metrics(test_rows)
    micro = base.metrics_from_scores(y_test, test_score, threshold)
    macro = {
        key: base.macro_scene_score(test_rows, key)
        for key in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "iou",
            "gt_handle_ratio",
            "pred_handle_ratio",
            "correct_percent",
            "fp_percent",
            "fn_percent",
        ]
    }
    overall = {
        **spec,
        "model": model_name,
        "input_dim": int(scenes_train[0]["x"].shape[1]),
        "latent_dim": int(args.latent_dim),
        "train_scenes": len(scenes_train),
        "val_scenes": len(scenes_val),
        "test_scenes": len(scenes_test),
        "train_gaussians": int(train_n),
        "train_positive_ratio": float(train_pos / max(train_n, 1)),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(threshold),
        "macro_scene_average": macro,
        "micro": micro,
        "worst_scene_iou": min(test_rows, key=lambda r: r["iou"])["scene_key"],
        "best_scene_iou": max(test_rows, key=lambda r: r["iou"])["scene_key"],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    base.save_csv(model_dir / "history.csv", history)
    base.save_csv(model_dir / "threshold_curve.csv", th_curve)
    base.save_csv(model_dir / "per_scene_metrics.csv", test_rows)
    base.save_csv(model_dir / "per_category_metrics.csv", cat_rows)
    with open(model_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    offsets = np.asarray([[s, e] for _, s, e in test_slices], dtype=np.int64)
    scene_keys = np.asarray([item["scene_key"] for item, _, _ in test_slices])
    np.savez_compressed(
        model_dir / "test_predictions.npz",
        scene_keys=scene_keys,
        offsets=offsets,
        scores=test_score.astype(np.float32),
        labels=y_test.astype(np.uint8),
    )
    torch.save({"model": model.state_dict(), "overall": overall, "threshold": float(threshold)}, model_dir / "model.pt")
    print(
        f"[{spec['run_name']}/{model_name}] TEST IoU={macro['iou']:.3f} "
        f"F1={macro['f1']:.3f} pred_ratio={macro['pred_handle_ratio']:.3f} th={threshold:.2f}",
        flush=True,
    )
    return overall


def write_summary(out_root):
    rows = []
    for p in sorted(out_root.glob("**/overall_metrics.json")):
        with open(p, "r") as f:
            data = json.load(f)
        macro = data["macro_scene_average"]
        micro = data["micro"]
        rows.append(
            {
                "block": data["block"],
                "run_name": data["run_name"],
                "feature_variant": data["feature_variant"],
                "label_transform": data["label_transform"],
                "data_transform": data["data_transform"],
                "train_domain": data["train_domain"],
                "val_domain": data["val_domain"],
                "test_domain": data["test_domain"],
                "model": data["model"],
                "input_dim": data["input_dim"],
                "train_scenes": data["train_scenes"],
                "val_scenes": data["val_scenes"],
                "test_scenes": data["test_scenes"],
                "train_positive_ratio": data["train_positive_ratio"],
                "selected_epoch": data["selected_epoch"],
                "selected_threshold": data["selected_threshold"],
                "macro_iou": macro["iou"],
                "macro_f1": macro["f1"],
                "macro_precision": macro["precision"],
                "macro_recall": macro["recall"],
                "macro_gt_handle_ratio": macro["gt_handle_ratio"],
                "macro_pred_handle_ratio": macro["pred_handle_ratio"],
                "micro_iou": micro["iou"],
                "micro_f1": micro["f1"],
                "micro_correct_percent": micro["correct_percent"],
                "worst_scene_iou": data["worst_scene_iou"],
                "best_scene_iou": data["best_scene_iou"],
            }
        )
    if rows:
        base.save_csv(out_root / "all_stage13_targeted_domain_alignment_summary.csv", rows)
        with open(out_root / "all_stage13_targeted_domain_alignment_summary.json", "w") as f:
            json.dump(rows, f, indent=2)
    print(f"[aggregate] collected {len(rows)} rows", flush=True)


def build_specs(syn, handal, seed):
    specs = []

    def add(block, run_id, feature, train_items, val_items, test_items, label, data, train_domain, val_domain, test_domain):
        specs.append(
            {
                "block": block,
                "run_name": f"{block}__{run_id}__{feature}__label-{label}__data-{data}",
                "feature_variant": feature,
                "label_transform": label,
                "data_transform": data,
                "train_domain": train_domain,
                "val_domain": val_domain,
                "test_domain": test_domain,
                "train_items": train_items,
                "val_items": val_items,
                "test_items": test_items,
            }
        )

    for feature in FEATURE_VARIANTS:
        add(
            "block_a_feature_alignment",
            "synth_to_handal",
            feature,
            syn["train"],
            handal["val"],
            handal["test"],
            "original",
            "clean",
            "affordsplat",
            "handal",
            "handal",
        )

    for feature in FOCUSED_VARIANTS:
        for label in ["dilate_small", "dilate_medium", "erode_small"]:
            add(
                "block_b_label_alignment",
                "synth_to_handal",
                feature,
                syn["train"],
                handal["val"],
                handal["test"],
                label,
                "clean",
                "affordsplat",
                "handal",
                "handal",
            )

    for feature in FOCUSED_VARIANTS:
        for data in ["xyz_jitter", "opacity_match", "floaters", "combined"]:
            add(
                "block_c_stat_noise_alignment",
                "synth_to_handal",
                feature,
                syn["train"],
                handal["val"],
                handal["test"],
                "original",
                data,
                "affordsplat",
                "handal",
                "handal",
            )

    rng = random.Random(seed)
    syn_balanced = list(syn["train"])
    rng.shuffle(syn_balanced)
    syn_balanced = syn_balanced[: len(handal["train"])]
    for feature in FOCUSED_VARIANTS:
        add(
            "block_d_mixed_training",
            "handal_only",
            feature,
            handal["train"],
            handal["val"],
            handal["test"],
            "original",
            "clean",
            "handal",
            "handal",
            "handal",
        )
        add(
            "block_d_mixed_training",
            "handal_plus_clean_synth",
            feature,
            handal["train"] + syn["train"],
            handal["val"],
            handal["test"],
            "original",
            "clean",
            "mixed",
            "handal",
            "handal",
        )
        add(
            "block_d_mixed_training",
            "handal_plus_dilate_small_synth",
            feature,
            handal["train"] + syn["train"],
            handal["val"],
            handal["test"],
            "dilate_small",
            "clean",
            "mixed",
            "handal",
            "handal",
        )
        add(
            "block_d_mixed_training",
            "handal_plus_floaters_synth",
            feature,
            handal["train"] + syn["train"],
            handal["val"],
            handal["test"],
            "original",
            "floaters",
            "mixed",
            "handal",
            "handal",
        )
        add(
            "block_d_mixed_training",
            "handal_plus_combined_synth",
            feature,
            handal["train"] + syn["train"],
            handal["val"],
            handal["test"],
            "dilate_small",
            "combined",
            "mixed",
            "handal",
            "handal",
        )
        add(
            "block_d_mixed_training",
            "handal_plus_balanced_combined_synth",
            feature,
            handal["train"] + syn_balanced,
            handal["val"],
            handal["test"],
            "dilate_small",
            "combined",
            "mixed_balanced",
            "handal",
            "handal",
        )
    return specs


def strip_items(spec):
    return {k: v for k, v in spec.items() if k not in {"train_items", "val_items", "test_items"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_root", type=Path, default=SYN_ROOT)
    parser.add_argument("--handal_root", type=Path, default=HANDAL_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        write_summary(args.out_root)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    syn_items, handal_items = prev.load_items(args.syn_root, args.handal_root)
    syn, handal = prev.split_synthetic(syn_items), prev.split_handal(handal_items)
    print(f"[data] synthetic train/val/test={len(syn['train'])}/{len(syn['val'])}/{len(syn['test'])}", flush=True)
    print(f"[data] HANDAL train/val/test={len(handal['train'])}/{len(handal['val'])}/{len(handal['test'])}", flush=True)
    print(f"[gpu] device={device} shard={args.shard}/{args.num_shards}", flush=True)

    target_stats = build_target_stats(handal["train"], sorted(set(FEATURE_VARIANTS + FOCUSED_VARIANTS)))
    specs = build_specs(syn, handal, args.seed)
    specs = [s for i, s in enumerate(specs) if i % args.num_shards == args.shard]
    manifest = [strip_items(s) for s in build_specs(syn, handal, args.seed)]
    base.save_csv(args.out_root / "experiment_manifest.csv", manifest)
    print(f"[plan] this shard will run {len(specs)} specs", flush=True)

    for spec in specs:
        print(
            f"[run] {spec['run_name']} train={len(spec['train_items'])} "
            f"val={len(spec['val_items'])} test={len(spec['test_items'])}",
            flush=True,
        )
        model_dir = args.out_root / spec["run_name"] / f"pointnet_{spec['feature_variant']}"
        if args.skip_existing and (model_dir / "overall_metrics.json").exists():
            print(f"[skip] {model_dir}", flush=True)
            continue
        train_raw = [
            make_scene(
                item,
                spec["feature_variant"],
                spec["label_transform"],
                spec["data_transform"],
                args.seed,
                target_stats,
                apply_to_source=True,
            )
            for item in spec["train_items"]
        ]
        mean, std = fit_standardizer(train_raw)
        train = standardize_scenes(train_raw, mean, std)
        val = standardize_scenes(
            [
                make_scene(item, spec["feature_variant"], "original", "clean", args.seed, target_stats, apply_to_source=False)
                for item in spec["val_items"]
            ],
            mean,
            std,
        )
        test = standardize_scenes(
            [
                make_scene(item, spec["feature_variant"], "original", "clean", args.seed, target_stats, apply_to_source=False)
                for item in spec["test_items"]
            ],
            mean,
            std,
        )
        train_pointnet(strip_items(spec), train, val, test, args, device)
        write_summary(args.out_root)
    write_summary(args.out_root)


if __name__ == "__main__":
    main()
