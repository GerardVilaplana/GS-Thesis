#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))
import train_handal_17cat_mlp_baselines as base  # noqa: E402
from train_handal_17cat_pointnet_global import PointNetGlobal  # noqa: E402
import train_mug_synthetic_domain_gap as prev  # noqa: E402
import train_mug_stage13_targeted_domain_alignment as stage13  # noqa: E402
import train_mug_stage14_uniform_mixed_split as stage14  # noqa: E402


STAGE14_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "14_mug_uniform_mixed_60_10_30_v1"
RUNS = [
    "stage14_control_clean_mixed__geometry_color_scene_norm__label-original__data-clean",
    "stage14_best2_all_combined__geometry_color_scene_norm__label-dilate_small__data-combined",
    "stage14_best1_balanced_combined__geometry_color_scene_norm__label-dilate_small__data-combined",
]


def domain_of(item):
    return "affordsplat" if item["scene_key"].startswith("affordsplat_") else "handal"


def filter_by_domain(y, scores, slices, domain):
    keep_y, keep_s, keep_slices = [], [], []
    offset = 0
    for item, start, end in slices:
        if domain_of(item) != domain:
            continue
        yy = y[int(start) : int(end)]
        ss = scores[int(start) : int(end)]
        keep_y.append(yy)
        keep_s.append(ss)
        keep_slices.append((item, offset, offset + len(yy)))
        offset += len(yy)
    if not keep_y:
        return np.empty(0), np.empty(0), []
    return np.concatenate(keep_y), np.concatenate(keep_s), keep_slices


def macro_iou_at(y, scores, slices, threshold):
    if len(slices) == 0:
        return np.nan, np.nan
    rows = base.per_scene_metrics(slices, y, scores, float(threshold))
    return base.macro_scene_score(rows, "iou"), base.macro_scene_score(rows, "f1")


def choose_threshold_for_domain(y, scores, slices, domain):
    yy, ss, sl = filter_by_domain(y, scores, slices, domain)
    if len(sl) == 0:
        return np.nan
    best, _ = base.choose_threshold(yy, ss, sl)
    return float(best["threshold"])


def load_spec_by_run(split, seed):
    return {spec["run_name"]: spec for spec in stage14.build_specs(split, seed)}


def prepare(spec, target_stats, seed):
    train_raw = [
        stage13.make_scene(
            item,
            spec["feature_variant"],
            spec["label_transform"],
            spec["data_transform"],
            seed,
            target_stats,
            apply_to_source=True,
        )
        for item in spec["train_items"]
    ]
    mean, std = stage13.fit_standardizer(train_raw)
    val = stage13.standardize_scenes(
        [
            stage13.make_scene(
                item,
                spec["feature_variant"],
                "original",
                "clean",
                seed,
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
                seed,
                target_stats,
                apply_to_source=False,
            )
            for item in spec["test_items"]
        ],
        mean,
        std,
    )
    return val, test


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage14_root", type=Path, default=STAGE14_ROOT)
    parser.add_argument("--syn_root", type=Path, default=stage14.SYN_ROOT)
    parser.add_argument("--handal_root", type=Path, default=stage14.HANDAL_ROOT)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    syn_items, handal_items = prev.load_items(args.syn_root, args.handal_root)
    split, syn_split, handal_split = stage14.build_uniform_split(syn_items, handal_items, args.seed)
    target_stats = stage13.build_target_stats(handal_split["train"], ["geometry_color_scene_norm"])
    specs = load_spec_by_run(split, args.seed)

    rows = []
    for run_name in RUNS:
        spec = specs[run_name]
        model_dir = args.stage14_root / run_name / "pointnet_geometry_color_scene_norm"
        val, test = prepare(spec, target_stats, args.seed)
        ckpt = torch.load(model_dir / "model.pt", map_location=device)
        model = PointNetGlobal(val[0]["x"].shape[1], latent_dim=int(ckpt["overall"]["latent_dim"])).to(device)
        model.load_state_dict(ckpt["model"])
        y_val, s_val, slices_val = stage13.predict_pointnet(model, val, device)
        y_test, s_test, slices_test = stage13.predict_pointnet(model, test, device)

        mixed_threshold = float(ckpt["threshold"])
        handal_val_threshold = choose_threshold_for_domain(y_val, s_val, slices_val, "handal")
        affordsplat_val_threshold = choose_threshold_for_domain(y_val, s_val, slices_val, "affordsplat")
        handal_test_y, handal_test_s, handal_test_slices = filter_by_domain(y_test, s_test, slices_test, "handal")
        oracle, _ = base.choose_threshold(handal_test_y, handal_test_s, handal_test_slices)
        oracle_threshold = float(oracle["threshold"])

        thresholds = {
            "mixed_val_selected": mixed_threshold,
            "handal_val_selected": handal_val_threshold,
            "affordsplat_val_selected": affordsplat_val_threshold,
            "oracle_handal_test": oracle_threshold,
        }
        for threshold_name, threshold in thresholds.items():
            h_iou, h_f1 = macro_iou_at(handal_test_y, handal_test_s, handal_test_slices, threshold)
            all_iou, all_f1 = macro_iou_at(y_test, s_test, slices_test, threshold)
            rows.append(
                {
                    "run_name": run_name,
                    "threshold_source": threshold_name,
                    "threshold": float(threshold),
                    "handal_test_macro_iou": h_iou,
                    "handal_test_macro_f1": h_f1,
                    "mixed_test_macro_iou": all_iou,
                    "mixed_test_macro_f1": all_f1,
                    "handal_test_scenes": len(handal_test_slices),
                    "mixed_test_scenes": len(slices_test),
                }
            )
        print(f"[done] {run_name}", flush=True)

    out = args.stage14_root / "stage14_threshold_diagnostics.csv"
    write_csv(out, rows)
    print(out)


if __name__ == "__main__":
    main()
