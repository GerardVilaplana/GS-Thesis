#!/usr/bin/env python3
"""Create frame_XXXXX links matching REALM render camera order."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old in list(args.output_dir.glob("frame_*.jpg")) + list(args.output_dir.glob("frame_*.png")):
        old.unlink()

    cameras = sorted(json.loads(args.cameras_json.read_text()), key=lambda c: int(c["id"]))
    count = 0
    for idx, cam in enumerate(cameras, start=1):
        stem = cam["img_name"]
        src = None
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = args.image_dir / f"{stem}{suffix}"
            if candidate.exists():
                src = candidate
                break
        if src is None:
            raise FileNotFoundError(f"Missing source image for {stem} in {args.image_dir}")
        dst = args.output_dir / f"frame_{idx:05d}.jpg"
        os.symlink(os.path.relpath(src, args.output_dir), dst)
        count += 1

    print(f"linked {count} frames into {args.output_dir}")


if __name__ == "__main__":
    main()
