#!/usr/bin/env python3
"""Prepare mapped JPG frames and propagated label maps for COLMAP/REALM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image


def resize_image(im: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return im
    scale = max_side / float(max(im.size))
    if scale >= 1.0:
        return im
    new_size = (round(im.width * scale), round(im.height * scale))
    return im.resize(new_size, Image.Resampling.LANCZOS)


def resize_labels(labels: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    label_im = Image.fromarray(labels)
    if label_im.size != size_wh:
        label_im = label_im.resize(size_wh, Image.Resampling.NEAREST)
    return np.asarray(label_im, dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True, type=Path)
    parser.add_argument("--labels_dir", required=True, type=Path)
    parser.add_argument("--frame_mapping", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_objects", type=int, default=None)
    parser.add_argument("--max_side", type=int, default=1600)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    image_out = args.output_dir / "images"
    mask_out = args.output_dir / "images_train"
    object_mask_out = args.output_dir / "object_mask"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    object_mask_out.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(args.frame_mapping.read_text())
    converted = 0
    masks_written = 0
    missing = []
    max_label = 0

    for item in mapping:
        original_stem = Path(item["frame_name"]).stem
        jpeg_name = item.get("jpeg_name")
        if jpeg_name is None:
            jpeg_name = PureWindowsPath(item["jpeg_path"]).name
        src_image = args.frames_dir / jpeg_name
        label_path = args.labels_dir / f"{original_stem}_labels.npy"
        dst_image = image_out / f"{original_stem}.jpg"

        if not src_image.exists() or not label_path.exists():
            missing.append(original_stem)
            continue

        if not dst_image.exists():
            with Image.open(src_image) as im:
                im = resize_image(im.convert("RGB"), args.max_side)
                im.save(dst_image, quality=args.quality, subsampling=1)
        converted += 1

        with Image.open(dst_image) as im:
            size_wh = im.size
        labels = resize_labels(np.load(label_path), size_wh)
        max_label = max(max_label, int(labels.max()))
        num_objects = args.num_objects or max_label
        masks = np.stack([(labels == obj_id) for obj_id in range(1, num_objects + 1)], axis=0)
        areas = masks.reshape(num_objects, -1).sum(axis=1).astype(np.float32)
        np.savez_compressed(
            mask_out / f"{original_stem}.npz",
            masks=masks,
            areas=areas,
            predicted_ious=np.ones((num_objects,), dtype=np.float32),
            stability_scores=np.ones((num_objects,), dtype=np.float32),
            labels=labels,
        )
        Image.fromarray(labels.astype(np.uint8)).save(object_mask_out / f"{original_stem}.png")
        masks_written += 1

    summary = args.output_dir / "prepare_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"frames_dir={args.frames_dir}",
                f"labels_dir={args.labels_dir}",
                f"frame_mapping={args.frame_mapping}",
                f"output_dir={args.output_dir}",
                f"max_side={args.max_side}",
                f"num_images_written={converted}",
                f"num_mask_npz_written={masks_written}",
                f"max_label_seen={max_label}",
                f"num_missing={len(missing)}",
                "missing=" + ",".join(missing[:50]),
            ]
        )
        + "\n"
    )
    print(summary.read_text())


if __name__ == "__main__":
    main()
