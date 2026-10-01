#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))

import train_handal_17cat_mlp_baselines as base  # noqa: E402
import train_mug_synthetic_domain_gap as prev  # noqa: E402
import train_mug_stage13_targeted_domain_alignment as stage13  # noqa: E402
import train_mug_stage17_b0_b6_full_mugs as stage17  # noqa: E402


SYN_ROOT = BASE_ROOT / "data" / "affordsplat_mug_grasp_features_v1"
HANDAL_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "20_mug_capacity_opacity_v1"
FEATURE = "geometry_color_scene_norm"


MODEL_SPECS = {
    "simple": {
        "encoder_hidden": [128],
        "latent_dim": 96,
        "head_hidden": [128, 64],
        "dropout": 0.05,
        "purpose": "Smaller PointNet-Max to test whether B3 is over-parameterized.",
    },
    "wide": {
        "encoder_hidden": [256],
        "latent_dim": 256,
        "head_hidden": [384, 192],
        "dropout": 0.10,
        "purpose": "Wider PointNet-Max to test feature capacity.",
    },
    "deep_wide": {
        "encoder_hidden": [384, 256],
        "latent_dim": 256,
        "head_hidden": [512, 256, 128],
        "dropout": 0.10,
        "purpose": "Deeper/wider PointNet-Max to test whether a stronger local encoder helps.",
    },
    "b3_like": {
        "encoder_hidden": [192],
        "latent_dim": 192,
        "head_hidden": [256, 128],
        "dropout": 0.10,
        "purpose": "B3 architecture, used for opacity-filter comparisons.",
    },
}

FILTERS = {
    "none": {"threshold": None, "purpose": "No opacity filtering."},
    "relaxed": {"threshold": 0.05, "purpose": "Remove only very low-opacity Gaussians."},
    "aggressive": {"threshold": 0.15, "purpose": "Remove a larger low-opacity tail."},
    "very_aggressive": {"threshold": 0.30, "purpose": "Remove most low/mid-opacity Gaussians."},
    "extreme": {"threshold": 0.50, "purpose": "Keep only the highest-opacity half-range."},
}


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def count_domains(items):
    counts = defaultdict(int)
    for item in items:
        counts[item.get("source_domain", "handal")] += 1
    return dict(sorted(counts.items()))


def apply_opacity_filter(scene, mode):
    cfg = FILTERS[mode]
    if cfg["threshold"] is None:
        stats = {
            "raw_gaussians": int(len(scene["y"])),
            "kept_gaussians": int(len(scene["y"])),
            "kept_ratio": 1.0,
            "raw_positive": int(scene["y"].sum()),
            "kept_positive": int(scene["y"].sum()),
        }
        return scene, stats

    x = scene["x"]
    y = scene["y"]
    # For geometry_color_scene_norm, opacity is column 10 after per-scene min-max normalization.
    keep = x[:, 10] >= float(cfg["threshold"])
    if keep.sum() < 10 or y[keep].sum() < 1:
        # Avoid creating unusable empty/negative-only scenes. This is logged in the keep ratio.
        keep = np.ones(len(y), dtype=bool)
    out = {
        "item": scene["item"],
        "x": x[keep].astype(np.float32, copy=False),
        "y": y[keep].astype(np.float32, copy=False),
    }
    stats = {
        "raw_gaussians": int(len(y)),
        "kept_gaussians": int(keep.sum()),
        "kept_ratio": float(keep.mean()),
        "raw_positive": int(y.sum()),
        "kept_positive": int(y[keep].sum()),
    }
    return out, stats


def filter_scenes(scenes, mode):
    out, stats = [], []
    for scene in scenes:
        s, st = apply_opacity_filter(scene, mode)
        out.append(s)
        stats.append(st)
    return out, stats


class FlexiblePointNetMax(nn.Module):
    def __init__(self, in_dim, spec):
        super().__init__()
        enc = []
        last = in_dim
        for hidden in spec["encoder_hidden"]:
            enc.extend([nn.Linear(last, hidden), nn.ReLU(inplace=True), nn.Dropout(spec["dropout"])])
            last = hidden
        enc.extend([nn.Linear(last, spec["latent_dim"]), nn.ReLU(inplace=True), nn.LayerNorm(spec["latent_dim"])])
        self.encoder = nn.Sequential(*enc)

        head = []
        last = spec["latent_dim"] * 2
        for hidden in spec["head_hidden"]:
            head.extend([nn.Linear(last, hidden), nn.ReLU(inplace=True), nn.Dropout(spec["dropout"])])
            last = hidden
        head.append(nn.Linear(last, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        h = self.encoder(x)
        global_max = h.max(dim=0, keepdim=True).values.expand(len(h), -1)
        return self.head(torch.cat([h, global_max], dim=1)).squeeze(-1)


def predict(model, scenes, device):
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


def build_split(args):
    syn_items, handal_items = prev.load_items(args.syn_root, args.handal_root)
    handal_split = stage17.grouped_ratio_split(handal_items, args.seed, train_ratio=0.60, val_ratio=0.10)
    syn_split = prev.split_synthetic(syn_items)
    return handal_split, syn_split


def make_scene(item, label_transform, data_transform, seed, target_stats, apply_to_source=True):
    return stage13.make_scene(
        item,
        FEATURE,
        label_transform,
        data_transform,
        seed,
        target_stats,
        apply_to_source=apply_to_source,
    )


def prepare_scenes(train_items, val_items, test_items, args, target_stats, opacity_filter):
    train_raw = [
        make_scene(item, "original", "opacity_match", args.seed, target_stats, apply_to_source=True)
        for item in train_items
    ]
    val_raw = [make_scene(item, "original", "clean", args.seed, target_stats, apply_to_source=False) for item in val_items]
    test_raw = [
        make_scene(item, "original", "clean", args.seed, target_stats, apply_to_source=False) for item in test_items
    ]

    train_raw, train_filter_stats = filter_scenes(train_raw, opacity_filter)
    val_raw, val_filter_stats = filter_scenes(val_raw, opacity_filter)
    test_raw, test_filter_stats = filter_scenes(test_raw, opacity_filter)

    mean, std = stage13.fit_standardizer(train_raw)
    return (
        stage13.standardize_scenes(train_raw, mean, std),
        stage13.standardize_scenes(val_raw, mean, std),
        stage13.standardize_scenes(test_raw, mean, std),
        {
            "train_mean_keep_ratio": float(np.mean([s["kept_ratio"] for s in train_filter_stats])),
            "val_mean_keep_ratio": float(np.mean([s["kept_ratio"] for s in val_filter_stats])),
            "test_mean_keep_ratio": float(np.mean([s["kept_ratio"] for s in test_filter_stats])),
            "train_kept_gaussians": int(sum(s["kept_gaussians"] for s in train_filter_stats)),
            "val_kept_gaussians": int(sum(s["kept_gaussians"] for s in val_filter_stats)),
            "test_kept_gaussians": int(sum(s["kept_gaussians"] for s in test_filter_stats)),
        },
    )


def train_one(run, args, device, target_stats, handal_split, syn_split):
    model_dir = args.out_root / run["run_name"] / f"pointnet_max_{FEATURE}_{run['model_size']}_{run['opacity_filter']}"
    if args.skip_existing and (model_dir / "overall_metrics.json").exists():
        print(f"[skip] {run['run_id']} {run['run_name']}", flush=True)
        with open(model_dir / "overall_metrics.json") as f:
            return json.load(f)

    train_items = handal_split["train"] + syn_split["train"]
    val_items = handal_split["val"]
    test_items = handal_split["test"]
    scenes_train, scenes_val, scenes_test, filter_stats = prepare_scenes(
        train_items, val_items, test_items, args, target_stats, run["opacity_filter"]
    )

    spec = MODEL_SPECS[run["model_size"]]
    model = FlexiblePointNetMax(scenes_train[0]["x"].shape[1], spec).to(device)
    train_n = sum(len(s["y"]) for s in scenes_train)
    train_pos = sum(float(s["y"].sum()) for s in scenes_train)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(train_n - train_pos) / max(train_pos, 1.0)], device=device)
    )
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

        y_val, score_val, slices_val = predict(model, scenes_val, device)
        th, _ = base.choose_threshold(y_val, score_val, slices_val)
        val_rows = base.per_scene_metrics(slices_val, y_val, score_val, th["threshold"])
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
                f"[{run['run_id']}] epoch={epoch:03d} loss={rec['loss']:.4f} "
                f"val_iou={rec['val_macro_scene_iou']:.3f} th={rec['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"[{run['run_id']}] early_stop epoch={epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    y_val, score_val, slices_val = predict(model, scenes_val, device)
    th, th_curve = base.choose_threshold(y_val, score_val, slices_val)
    y_test, score_test, slices_test = predict(model, scenes_test, device)
    test_rows = base.per_scene_metrics(slices_test, y_test, score_test, th["threshold"])
    macro = {
        k: base.macro_scene_score(test_rows, k)
        for k in [
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
    micro = base.metrics_from_scores(y_test, score_test, th["threshold"])
    overall = {
        "stage": "20_mug_capacity_opacity_v1",
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "baseline_reference": "B3_handal_plus_synth__orig_opacity_match",
        "model": f"pointnet_max_{FEATURE}",
        "model_size": run["model_size"],
        "model_spec": spec,
        "feature_variant": FEATURE,
        "pooling": "max",
        "label_transform": "original",
        "data_transform": "opacity_match",
        "opacity_filter": run["opacity_filter"],
        "opacity_filter_threshold": FILTERS[run["opacity_filter"]]["threshold"],
        "input_dim": int(scenes_train[0]["x"].shape[1]),
        "train_scenes": len(scenes_train),
        "val_scenes": len(scenes_val),
        "test_scenes": len(scenes_test),
        "train_domain_counts": count_domains(train_items),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(th["threshold"]),
        "filter_stats": filter_stats,
        "macro_scene_average": macro,
        "micro": micro,
        "worst_scene_iou": min(test_rows, key=lambda r: r["iou"])["scene_key"],
        "best_scene_iou": max(test_rows, key=lambda r: r["iou"])["scene_key"],
        "purpose": run["purpose"],
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    base.save_csv(model_dir / "history.csv", history)
    base.save_csv(model_dir / "threshold_curve.csv", th_curve)
    base.save_csv(model_dir / "per_scene_metrics.csv", test_rows)
    with open(model_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    offsets = np.asarray([[s, e] for _, s, e in slices_test], dtype=np.int64)
    scene_keys = np.asarray([item["scene_key"] for item, _, _ in slices_test])
    np.savez_compressed(
        model_dir / "test_predictions.npz",
        scene_keys=scene_keys,
        offsets=offsets,
        scores=score_test.astype(np.float32),
        labels=y_test.astype(np.uint8),
    )
    torch.save({"model": model.state_dict(), "overall": overall, "threshold": float(th["threshold"])}, model_dir / "model.pt")
    print(
        f"[{run['run_id']}] TEST IoU={macro['iou']:.3f} F1={macro['f1']:.3f} "
        f"P={macro['precision']:.3f} R={macro['recall']:.3f} th={th['threshold']:.2f}",
        flush=True,
    )
    return overall


def write_summary(out_root):
    rows = []
    for p in sorted(out_root.glob("**/overall_metrics.json")):
        with open(p) as f:
            data = json.load(f)
        macro = data["macro_scene_average"]
        rows.append(
            {
                "run_id": data["run_id"],
                "run_name": data["run_name"],
                "model_size": data["model_size"],
                "opacity_filter": data["opacity_filter"],
                "opacity_filter_threshold": data["opacity_filter_threshold"],
                "train_scenes": data["train_scenes"],
                "val_scenes": data["val_scenes"],
                "test_scenes": data["test_scenes"],
                "selected_epoch": data["selected_epoch"],
                "selected_threshold": data["selected_threshold"],
                "macro_iou": macro["iou"],
                "macro_f1": macro["f1"],
                "macro_precision": macro["precision"],
                "macro_recall": macro["recall"],
                "test_mean_keep_ratio": data["filter_stats"]["test_mean_keep_ratio"],
                "worst_scene_iou": data["worst_scene_iou"],
                "best_scene_iou": data["best_scene_iou"],
                "purpose": data["purpose"],
            }
        )
    save_csv(out_root / "stage20_summary.csv", rows)
    with open(out_root / "stage20_summary.json", "w") as f:
        json.dump(rows, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_root", type=Path, default=SYN_ROOT)
    parser.add_argument("--handal_root", type=Path, default=HANDAL_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        write_summary(args.out_root)
        return

    runs = [
        {
            "run_id": "C1",
            "run_name": "C1_simple_pointnetmax__opacity_match",
            "model_size": "simple",
            "opacity_filter": "none",
            "purpose": MODEL_SPECS["simple"]["purpose"],
        },
        {
            "run_id": "C2",
            "run_name": "C2_wide_pointnetmax__opacity_match",
            "model_size": "wide",
            "opacity_filter": "none",
            "purpose": MODEL_SPECS["wide"]["purpose"],
        },
        {
            "run_id": "C3",
            "run_name": "C3_deep_wide_pointnetmax__opacity_match",
            "model_size": "deep_wide",
            "opacity_filter": "none",
            "purpose": MODEL_SPECS["deep_wide"]["purpose"],
        },
        {
            "run_id": "C7",
            "run_name": "C7_b3_pointnetmax__opacity_match__filter_relaxed",
            "model_size": "b3_like",
            "opacity_filter": "relaxed",
            "purpose": "B3 architecture on current reconstructions with relaxed low-opacity filtering.",
        },
        {
            "run_id": "C8",
            "run_name": "C8_b3_pointnetmax__opacity_match__filter_aggressive",
            "model_size": "b3_like",
            "opacity_filter": "aggressive",
            "purpose": "B3 architecture on current reconstructions with aggressive low-opacity filtering.",
        },
    ]
    runs = [r for i, r in enumerate(runs) if i % args.num_shards == args.shard]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    handal_split, syn_split = build_split(args)
    target_stats = stage13.build_target_stats(handal_split["train"], [FEATURE])

    manifest = []
    for run in runs:
        manifest.append(
            {
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "model_size": run["model_size"],
                "opacity_filter": run["opacity_filter"],
                "opacity_filter_threshold": FILTERS[run["opacity_filter"]]["threshold"],
                "train_scenes": len(handal_split["train"]) + len(syn_split["train"]),
                "val_scenes": len(handal_split["val"]),
                "test_scenes": len(handal_split["test"]),
                "train_domain_counts": json.dumps(
                    count_domains(handal_split["train"] + syn_split["train"]), sort_keys=True
                ),
                "purpose": run["purpose"],
            }
        )
    save_csv(args.out_root / f"experiment_manifest_shard{args.shard}.csv", manifest)

    print(
        f"[data] HANDAL split={ {k: len(v) for k, v in handal_split.items()} } "
        f"AffordSplat={ {k: len(v) for k, v in syn_split.items()} }",
        flush=True,
    )
    print(f"[plan] shard {args.shard}/{args.num_shards} runs={[r['run_id'] for r in runs]} device={device}", flush=True)
    for run in runs:
        print(f"[run] {run['run_id']} {run['run_name']}", flush=True)
        train_one(run, args, device, target_stats, handal_split, syn_split)
        write_summary(args.out_root)
    write_summary(args.out_root)
    print("[done] wrote summary", flush=True)


if __name__ == "__main__":
    main()
