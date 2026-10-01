import argparse
from pathlib import Path

from build_handal_generalization_features_full import (
    DATASET_ROOT,
    OUT_ROOT,
    build_full_features,
    choose_debug_keys,
    compact_npz_path,
    load_manifest,
    load_summary_from_npz,
    select_all_rows,
    write_summaries,
)


def add_common_args(parser):
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DATASET_ROOT / "manifests" / "manifest.csv")
    parser.add_argument("--model_name", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dino_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--resize_width", type=int, default=448)
    parser.add_argument("--resize_height", type=int, default=336)
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--max_dino_views", type=int, default=None)
    parser.add_argument("--object_threshold", type=float, default=0.60)
    parser.add_argument("--object_min_visible", type=int, default=20)
    parser.add_argument("--handle_threshold", type=float, default=0.25)
    parser.add_argument("--handle_min_visible", type=int, default=10)
    parser.add_argument("--sigma_extent", type=float, default=3.0)
    parser.add_argument("--max_radius", type=int, default=24)
    parser.add_argument("--min_var_px", type=float, default=0.25)
    parser.add_argument("--max_patch_radius", type=int, default=4)
    parser.add_argument("--min_patch_radius", type=int, default=1)
    parser.add_argument("--min_var_patch", type=float, default=0.25)
    parser.add_argument("--min_alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--alpha_clip", type=float, default=0.99)
    parser.add_argument("--min_total_weight", type=float, default=1e-4)
    parser.add_argument("--min_view_weight", type=float, default=1e-5)
    parser.add_argument("--debug_extra_per_category", type=int, default=1)
    parser.add_argument("--force_features", action="store_true")


def finalize_args(args):
    args.mode = "full"
    args.work_root = args.out_root / "work"
    args.model_root = args.work_root / "3dgs_models"
    args.log_root = args.work_root / "3dgs_logs"
    args.npz_root = args.out_root / "npz"
    args.ply_root = args.out_root / "ply"
    args.npz_root.mkdir(parents=True, exist_ok=True)
    args.ply_root.mkdir(parents=True, exist_ok=True)
    args.summary_every = 10**9
    return args


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    worker = subparsers.add_parser("worker")
    add_common_args(worker)
    worker.add_argument("--num_shards", type=int, required=True)
    worker.add_argument("--shard", type=int, required=True)

    aggregate = subparsers.add_parser("aggregate")
    add_common_args(aggregate)

    args = finalize_args(parser.parse_args())
    rows = select_all_rows(load_manifest(args.manifest), args.dataset_root)

    if args.cmd == "worker":
        if not (0 <= args.shard < args.num_shards):
            raise ValueError("--shard must be in [0, num_shards)")
        shard_rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard]
        debug_keys = choose_debug_keys(rows, args)
        print(
            f"worker shard {args.shard}/{args.num_shards}: "
            f"{len(shard_rows)} rows, {len(debug_keys)} debug keys",
            flush=True,
        )
        build_full_features(shard_rows, args, debug_keys)
        print(f"worker shard {args.shard}/{args.num_shards}: done", flush=True)
        return

    missing = [row for row in rows if not compact_npz_path(args, row).exists()]
    if missing:
        print(f"missing_npz {len(missing)}")
        for row in missing[:50]:
            print(f"missing {row['category']}__{row['scene_id']}")
        raise SystemExit(1)

    summaries = [load_summary_from_npz(row, args) for row in rows]
    summary_path = write_summaries(args, summaries, "full")
    print(f"wrote {summary_path}")
    print(f"npz_complete {len(summaries)}")


if __name__ == "__main__":
    main()
