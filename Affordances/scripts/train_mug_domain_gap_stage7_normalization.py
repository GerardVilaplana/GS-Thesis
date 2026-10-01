#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))
import train_handal_17cat_mlp_baselines as base  # noqa: E402
from train_handal_17cat_pointnet_global import PointNetGlobal  # noqa: E402
import train_mug_synthetic_domain_gap as prev  # noqa: E402


SYN_ROOT = BASE_ROOT / "data" / "affordsplat_mug_grasp_features_v1"
HANDAL_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "11_mug_stage7_domain_normalization_v1"


FEATURE_VARIANTS = [
    "current_geometry_color",
    "xyzrgb",
    "xyzrgb_scene_color",
    "geometry_color_no_scale_opacity",
    "geometry_color_scene_norm",
]


def scene_zscore(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    return ((x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), eps)).astype(np.float32)


def scene_minmax_unit(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    return ((x - lo) / np.maximum(hi - lo, eps)).astype(np.float32)


def load_scene_arrays(item, feature_variant):
    with np.load(item["npz"], allow_pickle=False) as z:
        xyz = z["xyz"].astype(np.float32)
        xyz_norm = base.normalize_xyz(xyz)
        scale = z["scale"].astype(np.float32)
        rotation = z["rotation"].astype(np.float32)
        opacity = np.asarray(z["opacity"], dtype=np.float32).reshape(-1, 1)
        color = z["color"].astype(np.float32)
        y = z["handle_labels_thr0_25"].astype(np.float32)

    if feature_variant == "current_geometry_color":
        x = np.concatenate([xyz_norm, scale, rotation, opacity, color], axis=1)
    elif feature_variant == "xyzrgb":
        x = np.concatenate([xyz_norm, color], axis=1)
    elif feature_variant == "xyzrgb_scene_color":
        x = np.concatenate([xyz_norm, scene_zscore(color)], axis=1)
    elif feature_variant == "geometry_color_no_scale_opacity":
        x = np.concatenate([xyz_norm, rotation, color], axis=1)
    elif feature_variant == "geometry_color_scene_norm":
        x = np.concatenate(
            [
                xyz_norm,
                scene_zscore(scale),
                rotation,
                scene_minmax_unit(opacity),
                scene_zscore(color),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown feature variant: {feature_variant}")
    return x.astype(np.float32, copy=False), y


def make_scene(item, feature_variant):
    x, y = load_scene_arrays(item, feature_variant)
    return {"item": item, "x": x, "y": y}


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


def prepare_scenes(items, feature_variant, mean=None, std=None):
    scenes = [make_scene(item, feature_variant) for item in items]
    if mean is None or std is None:
        mean, std = fit_standardizer(scenes)
    return standardize_scenes(scenes, mean, std), mean, std


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


def train_pointnet(run_name, feature_variant, scenes_train, scenes_val, scenes_test, args, device, out_root):
    model_name = f"pointnet_{feature_variant}"
    model_dir = out_root / run_name / model_name
    if args.skip_existing and (model_dir / "overall_metrics.json").exists():
        print(f"[skip] {run_name}/{model_name}", flush=True)
        return json.load(open(model_dir / "overall_metrics.json"))

    model = PointNetGlobal(scenes_train[0]["x"].shape[1], latent_dim=args.latent_dim).to(device)
    train_n = sum(len(s["y"]) for s in scenes_train)
    train_pos = sum(float(s["y"].sum()) for s in scenes_train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(train_n - train_pos) / max(train_pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.pointnet_lr, weight_decay=args.weight_decay)

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
                f"[{run_name}/{feature_variant}] epoch {epoch:03d} "
                f"val_iou={rec['val_macro_scene_iou']:.3f} th={rec['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            break

    model.load_state_dict(best_state)
    y_val, val_score, val_slices = predict_pointnet(model, scenes_val, device)
    th, th_curve = base.choose_threshold(y_val, val_score, val_slices)
    threshold = th["threshold"]
    y_test, test_score, test_slices = predict_pointnet(model, scenes_test, device)
    model_dir.mkdir(parents=True, exist_ok=True)
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
        "run_name": run_name,
        "model": model_name,
        "feature_variant": feature_variant,
        "input_dim": int(scenes_train[0]["x"].shape[1]),
        "train_scenes": len(scenes_train),
        "val_scenes": len(scenes_val),
        "test_scenes": len(scenes_test),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(threshold),
        "macro_scene_average": macro,
        "micro": micro,
        "worst_scene_iou": min(test_rows, key=lambda r: r["iou"])["scene_key"],
        "best_scene_iou": max(test_rows, key=lambda r: r["iou"])["scene_key"],
    }
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
    print(f"[{run_name}/{feature_variant}] TEST macro IoU={macro['iou']:.3f} F1={macro['f1']:.3f} th={threshold:.2f}", flush=True)
    return overall


def build_experiments(syn, handal):
    return [
        ("stage7_synthetic_sanity_clean", syn["train"], syn["val"], syn["test"]),
        ("stage7_synth_to_handal_synthval", syn["train"], syn["val"], handal["test"]),
        ("stage7_synth_to_handal_handalval", syn["train"], handal["val"], handal["test"]),
        ("stage7_handal_only", handal["train"], handal["val"], handal["test"]),
        ("stage7_handal_plus_synthetic", handal["train"] + syn["train"], handal["val"], handal["test"]),
    ]


def write_summary(out_root):
    rows = []
    for p in sorted(out_root.glob("**/overall_metrics.json")):
        data = json.load(open(p))
        macro, micro = data["macro_scene_average"], data["micro"]
        rows.append(
            {
                "run_name": data["run_name"],
                "feature_variant": data["feature_variant"],
                "model": data["model"],
                "input_dim": data["input_dim"],
                "train_scenes": data["train_scenes"],
                "val_scenes": data["val_scenes"],
                "test_scenes": data["test_scenes"],
                "selected_epoch": data["selected_epoch"],
                "selected_threshold": data["selected_threshold"],
                "macro_iou": macro["iou"],
                "macro_f1": macro["f1"],
                "macro_precision": macro["precision"],
                "macro_recall": macro["recall"],
                "micro_iou": micro["iou"],
                "micro_f1": micro["f1"],
                "micro_correct_percent": micro["correct_percent"],
                "worst_scene_iou": data["worst_scene_iou"],
                "best_scene_iou": data["best_scene_iou"],
            }
        )
    if rows:
        base.save_csv(out_root / "all_stage7_domain_normalization_summary.csv", rows)
        with open(out_root / "all_stage7_domain_normalization_summary.json", "w") as f:
            json.dump(rows, f, indent=2)
    print(f"[aggregate] wrote {len(rows)} rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_root", type=Path, default=SYN_ROOT)
    parser.add_argument("--handal_root", type=Path, default=HANDAL_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--variants", nargs="+", default=FEATURE_VARIANTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pointnet_lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    syn_items, handal_items = prev.load_items(args.syn_root, args.handal_root)
    syn, handal = prev.split_synthetic(syn_items), prev.split_handal(handal_items)
    print(f"[data] synthetic train/val/test={len(syn['train'])}/{len(syn['val'])}/{len(syn['test'])}", flush=True)
    print(f"[data] HANDAL train/val/test={len(handal['train'])}/{len(handal['val'])}/{len(handal['test'])}", flush=True)
    print(f"[variants] {', '.join(args.variants)}", flush=True)

    manifest_rows = []
    experiments = build_experiments(syn, handal)
    for run_name, train_items, val_items, test_items in experiments:
        for variant in args.variants:
            manifest_rows.append(
                {
                    "run_name": run_name,
                    "feature_variant": variant,
                    "train_scenes": len(train_items),
                    "val_scenes": len(val_items),
                    "test_scenes": len(test_items),
                }
            )
    base.save_csv(args.out_root / "experiment_manifest.csv", manifest_rows)

    for run_name, train_items, val_items, test_items in experiments:
        print(f"[run] {run_name}", flush=True)
        for variant in args.variants:
            print(f"[variant] {variant}", flush=True)
            train_raw = [make_scene(item, variant) for item in train_items]
            mean, std = fit_standardizer(train_raw)
            train = standardize_scenes(train_raw, mean, std)
            val, _, _ = prepare_scenes(val_items, variant, mean, std)
            test, _, _ = prepare_scenes(test_items, variant, mean, std)
            train_pointnet(run_name, variant, train, val, test, args, device, args.out_root)
            write_summary(args.out_root)
    write_summary(args.out_root)


if __name__ == "__main__":
    main()
