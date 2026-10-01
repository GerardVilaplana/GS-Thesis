#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))

import build_handal_generalization_feature_pilot as pilot  # noqa: E402


DATASET_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_v1"
OUT_ROOT = BASE_ROOT / "data" / "handal_mugs_b_quality_features_v1"


def load_mug_rows(dataset_root, manifest_path):
    rows = [r for r in pilot.load_manifest(manifest_path) if r["category"] == "mugs"]
    rows = [r for r in rows if pilot.local_scene_path(dataset_root, r).exists()]
    rows.sort(key=lambda r: (r["instance_id"], r["scene_id"], r["split"]))
    if not rows:
        raise FileNotFoundError("No local mug scenes found in HANDAL manifest")
    return rows


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


def make_args(cli_args):
    work_root = cli_args.out_root / "work"
    return Namespace(
        dataset_root=cli_args.dataset_root,
        out_root=cli_args.out_root,
        work_root=work_root,
        model_root=work_root / "3dgs_models",
        log_root=work_root / "3dgs_logs",
        npz_root=cli_args.out_root / "npz",
        ply_root=cli_args.out_root / "ply",
        gpus=cli_args.gpus,
        iterations=cli_args.iterations,
        resolution=cli_args.resolution,
        max_images=cli_args.max_images,
        init_points=cli_args.init_points,
        force_3dgs=cli_args.force_3dgs,
        object_threshold=cli_args.object_threshold,
        object_min_visible=cli_args.object_min_visible,
        handle_threshold=cli_args.handle_threshold,
        handle_min_visible=cli_args.handle_min_visible,
        sigma_extent=3.0,
        max_radius=24,
        min_var_px=0.25,
        model_name="facebook/dinov2-small",
        dino_dtype="not_extracted",
        resize_width=0,
        resize_height=0,
    )


def save_compact_npz_no_dino(row, args, object_ply, object_scores, gaussian_indices, input_gaussians):
    key = pilot.scene_key(row["category"], row["scene_id"])
    handle_scores, handle_labels_arr, handle_visible = pilot.handle_labels(row, args, object_ply)
    vertices = pilot.PlyData.read(object_ply)["vertex"].data
    xyz, scale, rotation, opacity, color, geometry = pilot.geometry_arrays(vertices)
    n = len(xyz)
    out_path = args.npz_root / f"{key}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        dino_features=np.zeros((n, 384), dtype=np.float16),
        valid_dino=np.zeros(n, dtype=bool),
        dino_weight_sum=np.zeros(n, dtype=np.float32),
        dino_contribution_count=np.zeros(n, dtype=np.uint32),
        dino_visible_views=np.zeros(n, dtype=np.uint16),
        geometry_features=geometry.astype(np.float32),
        xyz=xyz.astype(np.float32),
        scale=scale.astype(np.float32),
        rotation=rotation.astype(np.float32),
        opacity=opacity.astype(np.float32),
        color=color.astype(np.float32),
        object_scores=object_scores.astype(np.float32),
        handle_scores=handle_scores.astype(np.float32),
        handle_labels_thr0_25=handle_labels_arr.astype(np.uint8),
        handle_visible_views=handle_visible,
        gaussian_indices=gaussian_indices.astype(np.int64),
        scene_key=np.array([key]),
        scene_id=np.array([row["scene_id"]]),
        category=np.array([row["category"]]),
        split=np.array([row["split"]]),
        instance_id=np.array([row["instance_id"]]),
        source_scene_path=np.array([str(pilot.local_scene_path(args.dataset_root, row))]),
        source_object_ply=np.array([str(object_ply)]),
        input_gaussians=np.array([input_gaussians], dtype=np.int32),
        object_threshold=np.array([args.object_threshold], dtype=np.float32),
        handle_threshold=np.array([args.handle_threshold], dtype=np.float32),
        dino_model=np.array(["not_extracted_b_quality"]),
        dino_dtype=np.array(["not_extracted"]),
        resize_width=np.array([0], dtype=np.int32),
        resize_height=np.array([0], dtype=np.int32),
    )
    return out_path, handle_scores, handle_labels_arr, handle_visible


def cleanup_scene_temporaries(args, key):
    for path in [
        args.model_root / key,
        args.work_root / "3dgs_scenes" / key,
        pilot.final_raw_ply(args, key),
    ]:
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def append_csv(path, row):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def process_chunk(rows, args, summary_path, keep_temporaries):
    pilot.train_3dgs(rows, args)
    for row in rows:
        key = pilot.scene_key(row["category"], row["scene_id"])
        out_npz = args.npz_root / f"{key}.npz"
        if out_npz.exists() and not args.force_features:
            print(f"[skip features] {key}", flush=True)
            if not keep_temporaries:
                cleanup_scene_temporaries(args, key)
            continue
        object_ply, object_scores, gaussian_indices, input_gaussians = pilot.object_prune(row, args)
        npz_path, handle_scores, handle_labels_arr, handle_visible = save_compact_npz_no_dino(
            row, args, object_ply, object_scores, gaussian_indices, input_gaussians
        )
        debug_ply = args.ply_root / f"{key}_b_quality_object_handle_red.ply"
        pilot.write_debug_ply(object_ply, handle_labels_arr, debug_ply)
        append_csv(
            summary_path,
            {
                "scene_key": key,
                "split": row["split"],
                "category": row["category"],
                "scene_id": row["scene_id"],
                "instance_id": row["instance_id"],
                "input_gaussians": input_gaussians,
                "object_gaussians": int(len(object_scores)),
                "handle_gaussians": int(handle_labels_arr.sum()),
                "handle_ratio": float(handle_labels_arr.sum() / max(len(handle_labels_arr), 1)),
                "valid_dino": 0,
                "valid_dino_ratio": 0.0,
                "npz": str(npz_path),
                "object_ply": str(object_ply),
                "debug_ply": str(debug_ply),
                "quality": "B_7000_res2_densify4500",
            },
        )
        print(
            f"[features] {key}: object {len(object_scores)}/{input_gaussians}, "
            f"handle {int(handle_labels_arr.sum())} ({handle_labels_arr.mean():.1%})",
            flush=True,
        )
        if not keep_temporaries:
            cleanup_scene_temporaries(args, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DATASET_ROOT / "manifests" / "manifest.csv")
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--densify_until_iter", type=int, default=4500)
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--init_points", type=int, default=30000)
    parser.add_argument("--object_threshold", type=float, default=0.60)
    parser.add_argument("--object_min_visible", type=int, default=20)
    parser.add_argument("--handle_threshold", type=float, default=0.25)
    parser.add_argument("--handle_min_visible", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=2)
    parser.add_argument("--force_3dgs", action="store_true")
    parser.add_argument("--force_features", action="store_true")
    parser.add_argument("--keep_temporaries", action="store_true")
    parser.add_argument("--max_scenes", type=int, default=None)
    cli_args = parser.parse_args()

    args = make_args(cli_args)
    for p in [args.out_root, args.work_root, args.model_root, args.log_root, args.npz_root, args.ply_root]:
        p.mkdir(parents=True, exist_ok=True)
    pilot.CONFIG = write_config(args.out_root / "config" / "handal_3dgs_b_quality.json", cli_args.densify_until_iter)

    rows = load_mug_rows(args.dataset_root, cli_args.manifest)
    if cli_args.max_scenes is not None:
        rows = rows[: cli_args.max_scenes]
    pending = [r for r in rows if cli_args.force_features or not (args.npz_root / f"{pilot.scene_key(r['category'], r['scene_id'])}.npz").exists()]

    with (args.out_root / "mug_rows_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_key", "split", "category", "scene_id", "instance_id", "num_rgb_images"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scene_key": pilot.scene_key(row["category"], row["scene_id"]),
                    "split": row["split"],
                    "category": row["category"],
                    "scene_id": row["scene_id"],
                    "instance_id": row["instance_id"],
                    "num_rgb_images": row["num_rgb_images"],
                }
            )

    print(
        f"[plan] total_mug_scenes={len(rows)} pending={len(pending)} "
        f"gpus={args.gpus} chunk_size={cli_args.chunk_size} out={args.out_root}",
        flush=True,
    )
    summary_path = args.out_root / "b_quality_mug_feature_summary.csv"
    for i in range(0, len(pending), cli_args.chunk_size):
        chunk = pending[i : i + cli_args.chunk_size]
        print(f"[chunk] {i // cli_args.chunk_size + 1}: {[pilot.scene_key(r['category'], r['scene_id']) for r in chunk]}", flush=True)
        process_chunk(chunk, args, summary_path, cli_args.keep_temporaries)

    print(f"[done] npz={args.npz_root} ply={args.ply_root} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
