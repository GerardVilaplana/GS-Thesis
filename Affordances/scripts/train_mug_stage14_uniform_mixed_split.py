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


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))
import train_handal_17cat_mlp_baselines as base  # noqa: E402
import train_mug_synthetic_domain_gap as prev  # noqa: E402
import train_mug_stage13_targeted_domain_alignment as stage13  # noqa: E402


SYN_ROOT = BASE_ROOT / "data" / "affordsplat_mug_grasp_features_v1"
HANDAL_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
OUT_ROOT = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "14_mug_uniform_mixed_60_10_30_v1"
)


def domain_of(item_or_key):
    if isinstance(item_or_key, str):
        return "affordsplat" if item_or_key.startswith("affordsplat_") else "handal"
    return item_or_key.get("source_domain") or domain_of(item_or_key["scene_key"])


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


def build_uniform_split(synthetic, handal, seed):
    syn_split = grouped_ratio_split(synthetic, seed + 11)
    handal_split = grouped_ratio_split(handal, seed + 23)
    mixed = {
        k: sorted(handal_split[k] + syn_split[k], key=lambda x: (domain_of(x), x["scene_key"]))
        for k in ["train", "val", "test"]
    }
    return mixed, syn_split, handal_split


def count_domains(items):
    out = defaultdict(int)
    for item in items:
        out[domain_of(item)] += 1
    return dict(sorted(out.items()))


def make_split_rows(split, syn_split, handal_split):
    rows = []
    for source_name, source_split in [("affordsplat", syn_split), ("handal", handal_split), ("mixed", split)]:
        for split_name in ["train", "val", "test"]:
            items = source_split[split_name]
            rows.append(
                {
                    "source": source_name,
                    "split": split_name,
                    "num_scenes": len(items),
                    "num_instances": len({str(x["instance_id"]) for x in items}),
                    "num_gaussians": sum(int(x["num_gaussians"]) for x in items),
                    "num_positive": sum(int(x["num_handle"]) for x in items),
                    "positive_ratio": sum(int(x["num_handle"]) for x in items)
                    / max(sum(int(x["num_gaussians"]) for x in items), 1),
                }
            )
    return rows


def balanced_train_subset(train_items, seed):
    by_domain = defaultdict(list)
    for item in train_items:
        by_domain[domain_of(item)].append(item)
    if len(by_domain) < 2:
        return sorted(train_items, key=lambda x: x["scene_key"])
    n = min(len(v) for v in by_domain.values())
    out = []
    for domain, items in sorted(by_domain.items()):
        items = list(items)
        random.Random(seed + base.stable_key(domain, seed)).shuffle(items)
        out.extend(items[:n])
    return sorted(out, key=lambda x: (domain_of(x), x["scene_key"]))


def build_specs(split, seed):
    feature = "geometry_color_scene_norm"
    all_train = split["train"]
    balanced_train = balanced_train_subset(all_train, seed)
    return [
        {
            "block": "stage14_uniform_mixed_60_10_30",
            "run_name": "stage14_control_clean_mixed__geometry_color_scene_norm__label-original__data-clean",
            "feature_variant": feature,
            "label_transform": "original",
            "data_transform": "clean",
            "train_domain": "mixed_uniform_all",
            "val_domain": "mixed_uniform",
            "test_domain": "mixed_uniform",
            "train_items": all_train,
            "val_items": split["val"],
            "test_items": split["test"],
        },
        {
            "block": "stage14_uniform_mixed_60_10_30",
            "run_name": "stage14_best2_all_combined__geometry_color_scene_norm__label-dilate_small__data-combined",
            "feature_variant": feature,
            "label_transform": "dilate_small",
            "data_transform": "combined",
            "train_domain": "mixed_uniform_all",
            "val_domain": "mixed_uniform",
            "test_domain": "mixed_uniform",
            "train_items": all_train,
            "val_items": split["val"],
            "test_items": split["test"],
        },
        {
            "block": "stage14_uniform_mixed_60_10_30",
            "run_name": "stage14_best1_balanced_combined__geometry_color_scene_norm__label-dilate_small__data-combined",
            "feature_variant": feature,
            "label_transform": "dilate_small",
            "data_transform": "combined",
            "train_domain": "mixed_uniform_balanced",
            "val_domain": "mixed_uniform",
            "test_domain": "mixed_uniform",
            "train_items": balanced_train,
            "val_items": split["val"],
            "test_items": split["test"],
        },
    ]


def strip_items(spec):
    return {k: v for k, v in spec.items() if k not in {"train_items", "val_items", "test_items"}}


def read_csv_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_domain_metrics(model_dir, test_items):
    scene_to_domain = {item["scene_key"]: domain_of(item) for item in test_items}
    scene_path = model_dir / "per_scene_metrics.csv"
    if not scene_path.exists():
        return {}
    rows = read_csv_rows(scene_path)
    for row in rows:
        row["source_domain"] = scene_to_domain.get(row["scene_key"], domain_of(row["scene_key"]))
    write_csv_rows(scene_path, rows)

    metric_names = [
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
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["source_domain"]].append(row)

    domain_rows = []
    for domain, group in sorted(by_domain.items()):
        out = {
            "source_domain": domain,
            "num_scenes": len(group),
            "num_gaussians": int(sum(float(r["num_gaussians"]) for r in group)),
        }
        for name in metric_names:
            out[f"macro_scene_{name}"] = float(np.nanmean([float(r[name]) for r in group]))
        domain_rows.append(out)
    base.save_csv(model_dir / "per_domain_metrics.csv", domain_rows)

    with open(model_dir / "overall_metrics.json", "r") as f:
        overall = json.load(f)
    overall["domain_macro_scene_average"] = {
        row["source_domain"]: {
            key.replace("macro_scene_", ""): value
            for key, value in row.items()
            if key.startswith("macro_scene_")
        }
        for row in domain_rows
    }
    overall["test_domain_scene_counts"] = count_domains(test_items)
    with open(model_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    return overall["domain_macro_scene_average"]


def write_summary(out_root):
    rows = []
    for p in sorted(out_root.glob("**/overall_metrics.json")):
        with open(p, "r") as f:
            data = json.load(f)
        macro = data["macro_scene_average"]
        micro = data["micro"]
        domains = data.get("domain_macro_scene_average", {})
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
                "handal_macro_iou": domains.get("handal", {}).get("iou", np.nan),
                "handal_macro_f1": domains.get("handal", {}).get("f1", np.nan),
                "affordsplat_macro_iou": domains.get("affordsplat", {}).get("iou", np.nan),
                "affordsplat_macro_f1": domains.get("affordsplat", {}).get("f1", np.nan),
                "micro_iou": micro["iou"],
                "micro_f1": micro["f1"],
                "micro_correct_percent": micro["correct_percent"],
                "worst_scene_iou": data["worst_scene_iou"],
                "best_scene_iou": data["best_scene_iou"],
            }
        )
    if rows:
        base.save_csv(out_root / "all_stage14_uniform_mixed_60_10_30_summary.csv", rows)
        with open(out_root / "all_stage14_uniform_mixed_60_10_30_summary.json", "w") as f:
            json.dump(rows, f, indent=2)
    print(f"[aggregate] collected {len(rows)} rows", flush=True)


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
    split, syn_split, handal_split = build_uniform_split(syn_items, handal_items, args.seed)
    print(f"[data] synthetic total={len(syn_items)} split={ {k: len(v) for k, v in syn_split.items()} }", flush=True)
    print(f"[data] HANDAL total={len(handal_items)} split={ {k: len(v) for k, v in handal_split.items()} }", flush=True)
    print(f"[data] mixed split={ {k: count_domains(v) for k, v in split.items()} }", flush=True)
    print(f"[gpu] device={device} shard={args.shard}/{args.num_shards}", flush=True)

    base.save_csv(args.out_root / "split_summary.csv", make_split_rows(split, syn_split, handal_split))
    specs_all = build_specs(split, args.seed)
    manifest = []
    for spec in specs_all:
        row = strip_items(spec)
        row.update(
            {
                "train_scenes": len(spec["train_items"]),
                "val_scenes": len(spec["val_items"]),
                "test_scenes": len(spec["test_items"]),
                "train_domain_counts": json.dumps(count_domains(spec["train_items"]), sort_keys=True),
                "val_domain_counts": json.dumps(count_domains(spec["val_items"]), sort_keys=True),
                "test_domain_counts": json.dumps(count_domains(spec["test_items"]), sort_keys=True),
            }
        )
        manifest.append(row)
    base.save_csv(args.out_root / "experiment_manifest.csv", manifest)

    target_stats = stage13.build_target_stats(handal_split["train"], ["geometry_color_scene_norm"])
    specs = [s for i, s in enumerate(specs_all) if i % args.num_shards == args.shard]
    print(f"[plan] this shard will run {len(specs)} specs", flush=True)

    for spec in specs:
        print(
            f"[run] {spec['run_name']} train={len(spec['train_items'])} "
            f"val={len(spec['val_items'])} test={len(spec['test_items'])} "
            f"train_domains={count_domains(spec['train_items'])}",
            flush=True,
        )
        model_dir = args.out_root / spec["run_name"] / f"pointnet_{spec['feature_variant']}"
        if args.skip_existing and (model_dir / "overall_metrics.json").exists():
            print(f"[skip] {model_dir}", flush=True)
            add_domain_metrics(model_dir, spec["test_items"])
            write_summary(args.out_root)
            continue

        train_raw = [
            stage13.make_scene(
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
        mean, std = stage13.fit_standardizer(train_raw)
        train = stage13.standardize_scenes(train_raw, mean, std)
        val = stage13.standardize_scenes(
            [
                stage13.make_scene(
                    item,
                    spec["feature_variant"],
                    "original",
                    "clean",
                    args.seed,
                    target_stats,
                    apply_to_source=False,
                )
                for item in spec["val_items"]
            ],
            mean,
            std,
        )
        test = stage13.standardize_scenes(
            [
                stage13.make_scene(
                    item,
                    spec["feature_variant"],
                    "original",
                    "clean",
                    args.seed,
                    target_stats,
                    apply_to_source=False,
                )
                for item in spec["test_items"]
            ],
            mean,
            std,
        )
        stage13.train_pointnet(strip_items(spec), train, val, test, args, device)
        add_domain_metrics(model_dir, spec["test_items"])
        write_summary(args.out_root)
    write_summary(args.out_root)


if __name__ == "__main__":
    main()
