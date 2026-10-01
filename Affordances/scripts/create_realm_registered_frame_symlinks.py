#!/usr/bin/env python3
"""Create frame_XXXXX symlinks matching REALM/COLMAP registered camera order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def read_registered_names(images_txt: Path) -> list[str]:
    names: list[str] = []
    expect_image_line = True
    for line in images_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not expect_image_line:
            expect_image_line = True
            continue
        parts = line.split()
        if len(parts) >= 10:
            names.append(parts[9])
            expect_image_line = False
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_txt", required=True, type=Path)
    parser.add_argument("--source_images", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old in args.output_dir.glob("frame_*.jpg"):
        old.unlink()

    names = read_registered_names(args.images_txt)
    missing = []
    for idx, name in enumerate(names, start=1):
        src = args.source_images / name
        if not src.exists():
            missing.append(name)
            continue
        dst = args.output_dir / f"frame_{idx:05d}.jpg"
        os.symlink(os.path.relpath(src, args.output_dir), dst)

    print(f"registered_names={len(names)}")
    print(f"symlinks={len(list(args.output_dir.glob('frame_*.jpg')))}")
    print(f"missing={len(missing)}")
    if missing:
        print("missing_names=" + ",".join(missing[:20]))


if __name__ == "__main__":
    main()
