#!/usr/bin/env python3
"""Refine the corrected scene-5 strainer REALM object using the provided masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
if str(BASE / "scripts") not in sys.path:
    sys.path.insert(0, str(BASE / "scripts"))

from evaluate_exp38_reprojection_strategy_ablation import (  # noqa: E402
    STRATEGIES,
    sample_points,
    strategy_keep,
)


def read_xyz(path: Path) -> tuple[PlyData, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
    return ply, xyz


def write_subset(ply: PlyData, keep: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.array(ply["vertex"].data, copy=True)[keep]
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply.text).write(out)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_ply",
        type=Path,
        default=BASE
        / "outputs/06_own_scenes/scene_5_strainer_corrected_15k_realm_query_pruning"
        / "final_ply/per_id_truecolor/strainer/strainer_realm_id001_truecolor_3dgs.ply",
    )
    parser.add_argument(
        "--model_root",
        type=Path,
        default=Path(
            "/home/gvilaplana/GS-Thesis/REALM-Code/output/own_scenes/"
            "scene_5_strainer_corrected_realm_1600_objpred_15k"
        ),
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=BASE
        / "data/own_scenes/scene_5_strainer_corrected_realm_1600/object_mask",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=BASE
        / "outputs/06_own_scenes/"
        "scene_5_strainer_corrected_15k_realm_query_pruning_correct_mask_refinement",
    )
    parser.add_argument("--mask_id", type=int, default=1)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="ellipsoid20_strict75")
    args = parser.parse_args()

    out_dir = args.out_root / "final_ply/per_id_reprojection_ellipsoid20_strict75"
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = sorted(
        json.loads((args.model_root / "cameras.json").read_text()),
        key=lambda cam: int(cam["id"]),
    )
    frames = []
    frame_rows = []
    for cam in cameras:
        mask_path = args.mask_dir / f"{cam['img_name']}.png"
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        binary = mask == args.mask_id
        frames.append({"frame_name": f"{cam['img_name']}.png", "camera": cam, "mask": binary})
        frame_rows.append(
            {
                "frame_name": f"{cam['img_name']}.png",
                "mask_pixels": int(binary.sum()),
                "mask_id": args.mask_id,
            }
        )

    if not frames:
        raise RuntimeError(f"No masks with id {args.mask_id} found in {args.mask_dir}")

    raw_ply, raw_xyz = read_xyz(args.raw_ply)
    samples = sample_points(raw_ply["vertex"].data, raw_xyz, STRATEGIES[args.strategy]["sample"])
    keep, cleanup, cleanup_frame_records = strategy_keep(
        raw_xyz,
        samples,
        frames,
        STRATEGIES[args.strategy],
    )

    clean_ply = out_dir / f"realm_id001_strainer_correctedmask_{args.strategy}_truecolor_3dgs.ply"
    write_subset(raw_ply, keep, clean_ply)

    row = {
        "class_id": 1,
        "queries": "strainer",
        "raw_ply": str(args.raw_ply),
        "clean_ply": str(clean_ply),
        "matched_frames_used": len(frames),
        "mask_source": str(args.mask_dir),
        "mask_id": args.mask_id,
        **cleanup,
    }
    save_csv(out_dir / "cleanup_summary.csv", [row])
    save_csv(out_dir / "mask_frame_summary.csv", frame_rows)
    save_csv(out_dir / "frame_summary.csv", cleanup_frame_records)

    summary = {
        "settings": jsonable(vars(args)),
        "objects": {
            "1": {
                **row,
                "diagnostics": frame_rows,
            }
        },
    }
    (out_dir / "per_id_reprojection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"raw points: {len(raw_xyz)}")
    print(f"kept points: {int(keep.sum())}")
    print(f"removed points: {int((~keep).sum())}")
    print(clean_ply)


if __name__ == "__main__":
    main()
