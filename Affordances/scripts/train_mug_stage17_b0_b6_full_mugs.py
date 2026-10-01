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
from train_handal_stage16_context_ablation import PointNetPooling  # noqa: E402
import train_mug_synthetic_domain_gap as prev  # noqa: E402
import train_mug_stage13_targeted_domain_alignment as stage13  # noqa: E402


SYN_ROOT = BASE_ROOT / "data" / "affordsplat_mug_grasp_features_v1"
HANDAL_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "17_mug_b0_b6_full_mugs_v1"
FEATURE = "geometry_color_scene_norm"


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def domain_of(item):
    return item.get("source_domain", "handal")


def grouped_ratio_split(items, seed, train_ratio=0.60, val_ratio=0.10):
    groups = defaultdict(list)
    for item in items:
        groups[str(item["instance_id"])].append(item)
    names = sorted(groups)
    names.sort(key=lambda x: base.stable_key(x, seed))

    total = sum(len(groups[name]) for name in names)
    target_train = int(round(total * train_ratio))
    target_val = int(round(total * val_ratio))
    split = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for name in names:
        group = sorted(groups[name], key=lambda x: x["scene_key"])
        if counts["train"] < target_train:
            dst = "train"
        elif counts["val"] < target_val:
            dst = "val"
        else:
            dst = "test"
        split[dst].extend(group)
        counts[dst] += len(group)
    return {k: sorted(v, key=lambda x: x["scene_key"]) for k, v in split.items()}


def count_domains(items):
    out = defaultdict(int)
    for item in items:
        out[domain_of(item)] += 1
    return dict(sorted(out.items()))


def split_rows(name, split):
    rows = []
    for split_name, items in split.items():
        rows.append(
            {
                "source": name,
                "split": split_name,
                "num_scenes": len(items),
                "num_instances": len({str(x["instance_id"]) for x in items}),
                "num_gaussians": int(sum(x["num_gaussians"] for x in items)),
                "num_positive": int(sum(x["num_handle"] for x in items)),
                "positive_ratio": float(
                    sum(x["num_handle"] for x in items) / max(sum(x["num_gaussians"] for x in items), 1)
                ),
                "scene_keys": " ".join(x["scene_key"] for x in items),
            }
        )
    return rows


def make_specs(syn_train, handal_split):
    h_train = handal_split["train"]
    h_val = handal_split["val"]
    h_test = handal_split["test"]
    return [
        {
            "run_id": "B0",
            "run_name": "B0_handal_only__orig_clean",
            "train_items": h_train,
            "val_items": h_val,
            "test_items": h_test,
            "label_transform": "original",
            "data_transform": "clean",
            "purpose": "Real-domain baseline using only HANDAL mugs.",
        },
        {
            "run_id": "B1",
            "run_name": "B1_handal_plus_synth__orig_clean",
            "train_items": h_train + syn_train,
            "val_items": h_val,
            "test_items": h_test,
            "label_transform": "original",
            "data_transform": "clean",
            "purpose": "Naive synthetic mixing without any adaptation.",
        },
        {
            "run_id": "B2",
            "run_name": "B2_handal_plus_synth__dilate_small_combined",
            "train_items": h_train + syn_train,
            "val_items": h_val,
            "test_items": h_test,
            "label_transform": "dilate_small",
            "data_transform": "combined",
            "purpose": "Previous best manual alignment: small synthetic label dilation plus combined perturbation.",
        },
        {
            "run_id": "B3",
            "run_name": "B3_handal_plus_synth__orig_opacity_match",
            "train_items": h_train + syn_train,
            "val_items": h_val,
            "test_items": h_test,
            "label_transform": "original",
            "data_transform": "opacity_match",
            "purpose": "Statistic alignment alone, using opacity quantile matching.",
        },
        {
            "run_id": "B4",
            "run_name": "B4_handal_plus_synth__dilate_small_opacity_match",
            "train_items": h_train + syn_train,
            "val_items": h_val,
            "test_items": h_test,
            "label_transform": "dilate_small",
            "data_transform": "opacity_match",
            "purpose": "Statistic alignment plus small synthetic label dilation.",
        },
        {
            "run_id": "B5",
            "run_name": "B5_handal_plus_synth__dilate_small_opacity_floaters",
            "train_items": h_train + syn_train,
            "val_items": h_val,
            "test_items": h_test,
            "label_transform": "dilate_small",
            "data_transform": "opacity_light_floaters",
            "purpose": "Statistic alignment plus light HANDAL-like negative floaters.",
        },
        {
            "run_id": "B6",
            "run_name": "B6_synth_only__dilate_small_opacity_floaters",
            "train_items": syn_train,
            "val_items": h_val,
            "test_items": h_test,
            "label_transform": "dilate_small",
            "data_transform": "opacity_light_floaters",
            "purpose": "Transfer test: can adapted synthetic mugs alone predict HANDAL mugs?",
        },
    ]


def make_scene(item, spec, seed, target_stats):
    return stage13.make_scene(
        item,
        FEATURE,
        spec["label_transform"],
        spec["data_transform"],
        seed,
        target_stats,
        apply_to_source=True,
    )


def make_eval_scene(item, seed, target_stats):
    return stage13.make_scene(
        item,
        FEATURE,
        "original",
        "clean",
        seed,
        target_stats,
        apply_to_source=False,
    )


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


def train_one(spec, args, target_stats, device):
    model_dir = args.out_root / spec["run_name"] / f"pointnet_max_{FEATURE}"
    done = model_dir / "overall_metrics.json"
    if args.skip_existing and done.exists():
        with open(done) as f:
            return json.load(f)

    train_raw = [make_scene(item, spec, args.seed, target_stats) for item in spec["train_items"]]
    mean, std = stage13.fit_standardizer(train_raw)
    train = stage13.standardize_scenes(train_raw, mean, std)
    val = stage13.standardize_scenes(
        [make_eval_scene(item, args.seed, target_stats) for item in spec["val_items"]],
        mean,
        std,
    )
    test = stage13.standardize_scenes(
        [make_eval_scene(item, args.seed, target_stats) for item in spec["test_items"]],
        mean,
        std,
    )

    in_dim = train[0]["x"].shape[1]
    model = PointNetPooling(in_dim, latent_dim=args.latent_dim, pooling="max", dropout=args.dropout).to(device)
    train_n = sum(len(s["y"]) for s in train)
    train_pos = sum(float(s["y"].sum()) for s in train)
    pos_weight = (train_n - train_pos) / max(float(train_pos), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best, best_state, stale = None, None, 0
    history = []
    order = list(range(len(train)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for idx in order:
            scene = train[idx]
            x = torch.from_numpy(scene["x"]).to(device)
            y = torch.from_numpy(scene["y"]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        y_val, score_val, slices_val = predict(model, val, device)
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
                f"[{spec['run_id']}] epoch={epoch:03d} loss={rec['loss']:.4f} "
                f"val_iou={rec['val_macro_scene_iou']:.3f} th={rec['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"[{spec['run_id']}] early_stop epoch={epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    y_val, score_val, slices_val = predict(model, val, device)
    th, th_curve = base.choose_threshold(y_val, score_val, slices_val)
    threshold = th["threshold"]
    y_test, score_test, slices_test = predict(model, test, device)
    test_rows = base.per_scene_metrics(slices_test, y_test, score_test, threshold)
    micro = base.metrics_from_scores(y_test, score_test, threshold)
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
        "stage": "17_mug_b0_b6_full_mugs",
        "run_id": spec["run_id"],
        "run_name": spec["run_name"],
        "purpose": spec["purpose"],
        "model": f"pointnet_max_{FEATURE}",
        "feature_variant": FEATURE,
        "pooling": "max",
        "label_transform": spec["label_transform"],
        "data_transform": spec["data_transform"],
        "input_dim": int(in_dim),
        "latent_dim": int(args.latent_dim),
        "train_scenes": len(train),
        "val_scenes": len(val),
        "test_scenes": len(test),
        "train_domain_counts": count_domains(spec["train_items"]),
        "val_domain_counts": count_domains(spec["val_items"]),
        "test_domain_counts": count_domains(spec["test_items"]),
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
    torch.save({"model": model.state_dict(), "overall": overall, "threshold": float(threshold)}, model_dir / "model.pt")
    print(
        f"[{spec['run_id']}] TEST IoU={macro['iou']:.3f} F1={macro['f1']:.3f} "
        f"precision={macro['precision']:.3f} recall={macro['recall']:.3f} th={threshold:.2f}",
        flush=True,
    )
    return overall


def write_summary(out_root):
    rows = []
    for p in sorted(out_root.glob("**/overall_metrics.json")):
        with open(p) as f:
            data = json.load(f)
        macro = data["macro_scene_average"]
        micro = data["micro"]
        rows.append(
            {
                "run_id": data["run_id"],
                "run_name": data["run_name"],
                "model": data["model"],
                "feature_variant": data["feature_variant"],
                "pooling": data["pooling"],
                "label_transform": data["label_transform"],
                "data_transform": data["data_transform"],
                "train_scenes": data["train_scenes"],
                "val_scenes": data["val_scenes"],
                "test_scenes": data["test_scenes"],
                "train_domain_counts": json.dumps(data["train_domain_counts"], sort_keys=True),
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
                "purpose": data["purpose"],
            }
        )
    save_csv(out_root / "all_stage17_b0_b6_summary.csv", rows)
    with open(out_root / "all_stage17_b0_b6_summary.json", "w") as f:
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
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
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

    stage13.DATA_TRANSFORMS["opacity_light_floaters"] = {
        "xyz_jitter": 0.0,
        "opacity_match": True,
        "floaters": 0.15,
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    syn_items, handal_items = prev.load_items(args.syn_root, args.handal_root)
    handal_split = grouped_ratio_split(handal_items, args.seed, train_ratio=0.60, val_ratio=0.10)
    syn_split = prev.split_synthetic(syn_items)
    syn_train = syn_split["train"]
    specs_all = make_specs(syn_train, handal_split)

    save_csv(args.out_root / "split_summary.csv", split_rows("handal", handal_split) + split_rows("affordsplat", syn_split))
    manifest = []
    for spec in specs_all:
        manifest.append(
            {
                "run_id": spec["run_id"],
                "run_name": spec["run_name"],
                "feature_variant": FEATURE,
                "model": f"pointnet_max_{FEATURE}",
                "label_transform": spec["label_transform"],
                "data_transform": spec["data_transform"],
                "train_scenes": len(spec["train_items"]),
                "val_scenes": len(spec["val_items"]),
                "test_scenes": len(spec["test_items"]),
                "train_domain_counts": json.dumps(count_domains(spec["train_items"]), sort_keys=True),
                "purpose": spec["purpose"],
            }
        )
    save_csv(args.out_root / "experiment_manifest.csv", manifest)

    print(
        f"[data] HANDAL mugs total={len(handal_items)} split="
        f"{ {k: len(v) for k, v in handal_split.items()} }",
        flush=True,
    )
    print(
        f"[data] AffordSplat train/val/test="
        f"{len(syn_split['train'])}/{len(syn_split['val'])}/{len(syn_split['test'])}",
        flush=True,
    )
    print(f"[gpu] device={device} shard={args.shard}/{args.num_shards}", flush=True)

    target_stats = stage13.build_target_stats(handal_split["train"], [FEATURE])
    specs = [s for i, s in enumerate(specs_all) if i % args.num_shards == args.shard]
    print(f"[plan] this shard will run {len(specs)} B-runs", flush=True)
    for spec in specs:
        print(
            f"[run] {spec['run_id']} train={len(spec['train_items'])} "
            f"val={len(spec['val_items'])} test={len(spec['test_items'])} "
            f"domains={count_domains(spec['train_items'])}",
            flush=True,
        )
        train_one(spec, args, target_stats, device)
        write_summary(args.out_root)
    write_summary(args.out_root)


if __name__ == "__main__":
    main()
