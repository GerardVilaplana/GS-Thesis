#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_exp40_b_dino_pilot import fit_pca, save_csv, write_pca_ply  # noqa: E402


DEFAULT_METHOD_DIR = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/"
    "42_exp40_b_dino_pilot_2percat_v1/"
    "facebook_dinov2_small_render_contrib_patch448x336"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_dir", type=Path, default=DEFAULT_METHOD_DIR)
    parser.add_argument("--max_pca_samples", type=int, default=50000)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    summary_csv = args.method_dir / "summary" / "dinov2_pilot_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    with summary_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    feature_paths: list[Path] = []
    for row in rows:
        path = Path(row["features"])
        if path.exists():
            feature_paths.append(path)

    if not feature_paths:
        raise RuntimeError(f"No completed feature files found from {summary_csv}")

    tag = args.tag or f"{len(feature_paths)}done"
    out_dir = args.method_dir / f"pca_ply_interim_{tag}"
    out_csv = args.method_dir / "summary" / f"dinov2_pilot_interim_pca_{tag}.csv"
    pca_npz = args.method_dir / "summary" / f"global_pca_projection_interim_{tag}.npz"

    mean, components, lo, scale = fit_pca(feature_paths, args.max_pca_samples)
    np.savez_compressed(pca_npz, mean=mean, components=components, color_low=lo, color_scale=scale)

    out_rows: list[dict] = []
    for feature_path in feature_paths:
        scene_key = feature_path.name.replace("_dinov2_render_contrib_features.npz", "")
        out_ply = out_dir / f"{scene_key}_dinov2_render_contrib_pca_interim.ply"
        write_pca_ply(feature_path, out_ply, mean, components, lo, scale)
        out_rows.append({"scene_key": scene_key, "features": str(feature_path), "pca_ply": str(out_ply)})

    save_csv(out_csv, out_rows)
    print(f"feature_files={len(feature_paths)}")
    print(f"pca_npz={pca_npz}")
    print(f"pca_dir={out_dir}")
    print(f"pca_plys={len(out_rows)}")
    print(f"summary_csv={out_csv}")


if __name__ == "__main__":
    main()
