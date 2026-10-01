#!/usr/bin/env python3
"""Prepare user-captured HEIC scenes and propagated label masks for COLMAP/REALM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def open_heic(path: Path) -> Image.Image:
    import pillow_heif

    pillow_heif.register_heif_opener()
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def resize_image(im: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return im
    scale = max_side / float(max(im.size))
    if scale >= 1.0:
        return im
    new_size = (round(im.width * scale), round(im.height * scale))
    return im.resize(new_size, Image.Resampling.LANCZOS)


def find_label_file(stem: str, masks_root: Path) -> Path | None:
    initial = masks_root / f"{stem}_obj_1_initial.npy"
    if initial.exists():
        return initial
    for split in ("forward", "backward"):
        candidate = masks_root / split / f"{stem}_labels.npy"
        if candidate.exists():
            return candidate
    return None


def load_label_map(stem: str, masks_root: Path, num_objects: int) -> np.ndarray | None:
    initial = masks_root / f"{stem}_obj_1_initial.npy"
    if initial.exists():
        masks = []
        for obj_id in range(1, num_objects + 1):
            obj_path = masks_root / f"{stem}_obj_{obj_id}_initial.npy"
            if not obj_path.exists():
                return None
            masks.append(np.load(obj_path).astype(bool))
        labels = np.zeros(masks[0].shape, dtype=np.uint16)
        for obj_id, mask in enumerate(masks, start=1):
            labels[mask] = obj_id
        return labels

    for split in ("forward", "backward"):
        candidate = masks_root / split / f"{stem}_labels.npy"
        if candidate.exists():
            return np.load(candidate).astype(np.uint16)
    return None


def resize_labels(labels: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    label_im = Image.fromarray(labels)
    if label_im.size != size_wh:
        label_im = label_im.resize(size_wh, Image.Resampling.NEAREST)
    return np.asarray(label_im, dtype=np.uint16)


def labels_to_realm_npz(labels: np.ndarray, size_wh: tuple[int, int], num_objects: int, output_path: Path) -> np.ndarray:
    resized = resize_labels(labels, size_wh)
    masks = np.stack([(resized == obj_id) for obj_id in range(1, num_objects + 1)], axis=0)
    areas = masks.reshape(num_objects, -1).sum(axis=1).astype(np.float32)
    np.savez_compressed(
        output_path,
        masks=masks,
        areas=areas,
        predicted_ious=np.ones((num_objects,), dtype=np.float32),
        stability_scores=np.ones((num_objects,), dtype=np.float32),
        labels=resized,
    )
    return resized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--masks_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_side", type=int, default=1600)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--num_objects", type=int, default=6)
    args = parser.parse_args()

    image_out = args.output_dir / "images"
    mask_out = args.output_dir / "images_train"
    object_mask_out = args.output_dir / "object_mask"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    object_mask_out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.images_dir.glob("*.HEIC"))
    if not image_paths:
        raise RuntimeError(f"No HEIC images found in {args.images_dir}")

    converted = 0
    masks_written = 0
    missing_masks = []
    for image_path in image_paths:
        stem = image_path.stem
        im = resize_image(open_heic(image_path), args.max_side)
        im.save(image_out / f"{stem}.jpg", quality=args.quality, subsampling=1)
        converted += 1

        labels = load_label_map(stem, args.masks_root, args.num_objects)
        if labels is None:
            missing_masks.append(stem)
            continue
        resized_labels = labels_to_realm_npz(labels, im.size, args.num_objects, mask_out / (stem + ".npz"))
        Image.fromarray(resized_labels.astype(np.uint8)).save(object_mask_out / (stem + ".png"))
        masks_written += 1

    summary = args.output_dir / "prepare_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"images_dir={args.images_dir}",
                f"masks_root={args.masks_root}",
                f"output_dir={args.output_dir}",
                f"max_side={args.max_side}",
                f"num_images_converted={converted}",
                f"num_mask_npz_written={masks_written}",
                f"num_missing_masks={len(missing_masks)}",
                "missing_masks=" + ",".join(missing_masks[:50]),
            ]
        )
        + "\n"
    )
    print(summary.read_text())


if __name__ == "__main__":
    main()
