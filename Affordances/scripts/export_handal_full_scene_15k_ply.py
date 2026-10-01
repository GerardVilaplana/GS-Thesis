#!/usr/bin/env python3
"""Export a 15k full-scene HANDAL 3DGS PLY for a selected exp40 scene."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from argparse import Namespace
from pathlib import Path

import build_handal_exp40_15k_gt_features as exp40


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")


def load_scene_row(scene_key: str, scene_specs: Path) -> dict:
    with scene_specs.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["scene_key"] == scene_key:
                return row
    raise KeyError(f"Scene not found in {scene_specs}: {scene_key}")


def append_timing(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_key", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--scene_specs", type=Path, default=BASE / "data/handal_exp40_15k_gt_features_v1/scene_specs.csv")
    parser.add_argument("--tmp_root", type=Path, default=BASE / "outputs/thesis_figures/results_plots/projection_quality/full_scene_unavailable_15k/_work")
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--densify_until_iter", type=int, default=9000)
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--init_points", type=int, default=30000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / f"{args.scene_key}_full_scene_15k.ply"
    if dst.exists() and not args.force:
        print(f"[skip] exists: {dst}", flush=True)
        return

    row = load_scene_row(args.scene_key, args.scene_specs)
    run_root = args.tmp_root / args.scene_key
    ns = Namespace(
        out_root=run_root,
        work_root=run_root / "work",
        model_root=run_root / "work/3dgs_models",
        log_root=run_root / "work/logs",
        config_file=exp40.write_config(run_root / "config/handal_3dgs_15k_gt.json", args.densify_until_iter),
        iterations=args.iterations,
        resolution=args.resolution,
        max_images=args.max_images,
        init_points=args.init_points,
        gpu=args.gpu,
        force_3dgs=args.force,
    )
    for path in [ns.out_root, ns.work_root, ns.model_root, ns.log_root]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.time()
    src = exp40.train_3dgs_scene(row, ns)
    shutil.copy2(src, dst)
    elapsed_min = (time.time() - started) / 60.0
    append_timing(
        args.out_dir / "full_scene_15k_export_times.csv",
        {
            "scene_key": args.scene_key,
            "gpu": args.gpu,
            "iterations": args.iterations,
            "resolution": args.resolution,
            "elapsed_min": f"{elapsed_min:.3f}",
            "output_ply": str(dst),
        },
    )
    exp40.cleanup_success_temporaries(row, ns)
    print(f"[done] {args.scene_key} -> {dst} ({elapsed_min:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
