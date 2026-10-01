#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))

import build_handal_generalization_feature_pilot as pilot  # noqa: E402


DATASET_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_v1"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "19_handal_3dgs_quality_pilot_v1"

SELECTED_KEYS = [
    ("mugs", "001001"),
    ("mugs", "007002"),
    ("measuring_cups", "001004"),
    ("measuring_cups", "021011"),
    ("screwdrivers", "001002"),
    ("screwdrivers", "012010"),
    ("power_drills", "001007"),
    ("power_drills", "099010"),
    ("hammers", "003000"),
    ("hammers", "032006"),
    ("whisks", "021003"),
    ("whisks", "042014"),
]

RUNS = [
    {
        "run_id": "B_7000_res2_densify4500",
        "iterations": 7000,
        "resolution": 2,
        "densify_until_iter": 4500,
        "description": "Better cheap HANDAL 3DGS: 7000 iterations, resolution 2, longer densification.",
    },
    {
        "run_id": "C_15000_res2_densify9000",
        "iterations": 15000,
        "resolution": 2,
        "densify_until_iter": 9000,
        "description": "Higher-quality HANDAL 3DGS: 15000 iterations, resolution 2, longer densification.",
    },
]


def load_rows(manifest_path):
    with manifest_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["category"], r["scene_id"]): r for r in rows}
    selected = []
    for key in SELECTED_KEYS:
        if key not in by_key:
            raise KeyError(f"Missing selected scene in manifest: {key}")
        row = by_key[key]
        if not pilot.local_scene_path(DATASET_ROOT, row).exists():
            raise FileNotFoundError(pilot.local_scene_path(DATASET_ROOT, row))
        selected.append(row)
    return selected


def write_config(path, densify_until_iter):
    cfg = {
        "densify_until_iter": densify_until_iter,
        "num_classes": 2,
        "reg3d_interval": 1000000,
        "reg3d_k": 5,
        "reg3d_lambda_val": 0,
        "reg3d_max_points": 300000,
        "reg3d_sample_size": 1000,
        "object_loss_start_iter": 1000000,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cfg, f, indent=2)
    return path


def make_args(run, cli_args):
    out_root = cli_args.out_root / run["run_id"]
    work_root = out_root / "work"
    return Namespace(
        dataset_root=cli_args.dataset_root,
        out_root=out_root,
        work_root=work_root,
        model_root=work_root / "3dgs_models",
        log_root=work_root / "3dgs_logs",
        npz_root=out_root / "npz",
        ply_root=out_root / "ply",
        gpus=cli_args.gpus,
        iterations=run["iterations"],
        resolution=run["resolution"],
        max_images=cli_args.max_images,
        init_points=cli_args.init_points,
        force_3dgs=cli_args.force_3dgs,
        object_threshold=cli_args.object_threshold,
        object_min_visible=cli_args.object_min_visible,
        handle_threshold=0.25,
        handle_min_visible=10,
        sigma_extent=3.0,
        max_radius=24,
        min_var_px=0.25,
    )


def export_plys(rows, args, run_id):
    scene_dir = args.out_root / "ply_exports" / "handal_scenes"
    object_dir = args.out_root / "ply_exports" / "handal_objects"
    scene_dir.mkdir(parents=True, exist_ok=True)
    object_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for row in rows:
        key = pilot.scene_key(row["category"], row["scene_id"])
        raw_ply = pilot.final_raw_ply(args, key)
        object_ply, object_scores, gaussian_indices, input_gaussians = pilot.object_prune(row, args)
        scene_out = scene_dir / f"{key}_{run_id}_full_scene_truecolor.ply"
        object_out = object_dir / f"{key}_{run_id}_object_pruned_truecolor.ply"
        shutil.copy2(raw_ply, scene_out)
        shutil.copy2(object_ply, object_out)
        manifest_rows.append(
            {
                "run_id": run_id,
                "category": row["category"],
                "scene_id": row["scene_id"],
                "scene_key": key,
                "split": row["split"],
                "input_gaussians": input_gaussians,
                "object_gaussians": int(len(gaussian_indices)),
                "object_keep_ratio": float(len(gaussian_indices) / max(input_gaussians, 1)),
                "object_score_mean": float(object_scores.mean()) if len(object_scores) else 0.0,
                "raw_scene_ply": str(raw_ply),
                "object_ply": str(object_ply),
                "export_scene_ply": str(scene_out),
                "export_object_ply": str(object_out),
            }
        )
        print(
            f"[export] {run_id} {key}: scene={scene_out.name} object={len(gaussian_indices)}/{input_gaussians}",
            flush=True,
        )

    manifest = args.out_root / "ply_exports" / "quality_pilot_ply_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[export] wrote {manifest}", flush=True)


def append_run_summary(path, row):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DATASET_ROOT / "manifests" / "manifest.csv")
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--gpus", nargs="+", default=["0", "4"])
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--init_points", type=int, default=30000)
    parser.add_argument("--object_threshold", type=float, default=0.60)
    parser.add_argument("--object_min_visible", type=int, default=20)
    parser.add_argument("--force_3dgs", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest)
    summary_path = args.out_root / "quality_pilot_summary.csv"

    with (args.out_root / "selected_scenes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "scene_id", "split", "instance_id", "num_rgb_images"])
        writer.writeheader()
        writer.writerows(
            {
                "category": r["category"],
                "scene_id": r["scene_id"],
                "split": r["split"],
                "instance_id": r["instance_id"],
                "num_rgb_images": r["num_rgb_images"],
            }
            for r in rows
        )

    for run in RUNS:
        run_args = make_args(run, args)
        for p in [run_args.work_root, run_args.model_root, run_args.log_root, run_args.npz_root, run_args.ply_root]:
            p.mkdir(parents=True, exist_ok=True)
        config_path = write_config(run_args.out_root / "config" / "handal_3dgs_quality.json", run["densify_until_iter"])
        pilot.CONFIG = config_path

        print(
            f"[run] {run['run_id']} scenes={len(rows)} iterations={run_args.iterations} "
            f"resolution={run_args.resolution} densify_until={run['densify_until_iter']} gpus={run_args.gpus}",
            flush=True,
        )
        pilot.train_3dgs(rows, run_args)
        export_plys(rows, run_args, run["run_id"])
        append_run_summary(
            summary_path,
            {
                "run_id": run["run_id"],
                "iterations": run["iterations"],
                "resolution": run["resolution"],
                "densify_until_iter": run["densify_until_iter"],
                "max_images": args.max_images,
                "init_points": args.init_points,
                "num_scenes": len(rows),
                "gpus": " ".join(args.gpus),
                "description": run["description"],
                "run_output": str(run_args.out_root),
            },
        )
        print(f"[run] completed {run['run_id']}", flush=True)

    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
