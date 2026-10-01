#!/usr/bin/env python3
"""Link corrected scene 5 strainer masks using original frame label names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_scene", required=True, type=Path)
    parser.add_argument("--mask_dir", required=True, type=Path)
    parser.add_argument("--link_dir", required=True, type=Path)
    args = parser.parse_args()

    mapping_path = args.raw_scene / "scene5_tmp_frames_for_sam2" / "frame_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mask_files = sorted(args.mask_dir.glob("object_frame_*.npy"))

    args.link_dir.mkdir(parents=True, exist_ok=True)
    for old in args.link_dir.glob("*_labels.npy"):
        old.unlink()

    for idx, mask_path in enumerate(mask_files):
        stem = Path(mapping[idx]["frame_name"]).stem
        dst = args.link_dir / f"{stem}_labels.npy"
        dst.symlink_to(mask_path)

    print(f"linked {len(mask_files)} corrected masks into {args.link_dir}")


if __name__ == "__main__":
    main()
