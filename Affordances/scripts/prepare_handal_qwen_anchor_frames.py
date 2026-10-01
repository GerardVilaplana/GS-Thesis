#!/usr/bin/env python3
"""Prepare deterministic random anchor frames for Qwen box prompting."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zlib
from pathlib import Path

from PIL import Image

from build_handal_sam2_multimask_realm_scenes import (
    SceneSpec,
    load_json,
    load_scene_specs,
    ordered_scene_frames,
    resize_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--scene_specs_csv", type=Path, required=True)
    parser.add_argument("--only_scene", type=str, default=None)
    parser.add_argument("--num_anchors", type=int, default=3)
    parser.add_argument("--target_width", type=int, default=960)
    parser.add_argument("--seed", type=int, default=2701)
    return parser.parse_args()


def pick_anchor_indices(scene_key: str, n_frames: int, n_anchors: int, seed: int) -> list[int]:
    if n_frames <= 0:
        raise ValueError("Cannot select anchors from an empty frame list")
    n = min(n_anchors, n_frames)
    rng = random.Random(seed + zlib.crc32(scene_key.encode("utf-8")))
    return sorted(rng.sample(range(n_frames), n))


def process_scene(spec: SceneSpec, args: argparse.Namespace) -> dict:
    src_scene_root = spec.feature_root / "work" / "3dgs_scenes" / spec.scene_key
    src_model_root = spec.feature_root / "work" / "3dgs_models" / spec.scene_key
    source_info = load_json(src_scene_root / "handal_source.json")
    frames = ordered_scene_frames(src_model_root, src_scene_root)
    anchor_indices = pick_anchor_indices(spec.scene_key, len(frames), args.num_anchors, args.seed)

    scene_out = args.output_root / spec.scene_key
    anchor_dir = scene_out / "qwen_anchor_images"
    if anchor_dir.exists():
        shutil.rmtree(anchor_dir)
    anchor_dir.mkdir(parents=True, exist_ok=True)

    anchors = []
    for idx in anchor_indices:
        stem, src = frames[idx]
        frame_name = f"frame_{idx + 1:05d}.jpg"
        resize_rgb(Image.open(src), args.target_width).save(
            anchor_dir / frame_name,
            quality=95,
            subsampling=0,
        )
        anchors.append(
            {
                "frame_index": int(idx),
                "frame_name": frame_name,
                "source_stem": stem,
                "source_image": str(src),
            }
        )

    plan = {
        "scene_key": spec.scene_key,
        "query": spec.query,
        "feature_root": str(spec.feature_root),
        "source_scene": source_info.get("source_scene"),
        "num_frames": len(frames),
        "num_anchors": len(anchors),
        "seed": args.seed,
        "target_width": args.target_width,
        "anchor_dir": str(anchor_dir),
        "anchors": anchors,
    }
    (scene_out / "qwen_anchor_plan.json").write_text(json.dumps(plan, indent=2))
    print(f"{spec.scene_key}: prepared {len(anchors)} Qwen anchors in {anchor_dir}", flush=True)
    return plan


def main() -> None:
    args = parse_args()
    specs = load_scene_specs(args.scene_specs_csv)
    if args.only_scene:
        specs = [s for s in specs if s.scene_key == args.only_scene]
        if not specs:
            raise ValueError(f"Scene {args.only_scene!r} not found in {args.scene_specs_csv}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = [process_scene(spec, args) for spec in specs]
    (args.output_root / "qwen_anchor_plan_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
