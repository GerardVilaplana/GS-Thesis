#!/usr/bin/env python3
"""Export the supervised power-drill pipeline assets used in the thesis."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCENE_KEY = "power_drills__006006"
SCENE_DIR = (
    BASE
    / "data/handal_handle_generalization_v1/subset_export/test_unseen_category"
    / "power_drills/006006"
)
SOURCE_PLY = (
    BASE
    / "data/handal_exp40_15k_gt_features_v1/ply/B_center_ellipsoid20_strict75"
    / f"{SCENE_KEY}_object_center_ell20s75.ply"
)
MODEL_DIR = (
    BASE
    / "outputs/03_handle_generalization/47_exp46_dino_pointnet_mean_feature_sweep_v1"
    / "object_crop_center_avg/01_seen_instance_17cat"
    / "dino_geometry_color_quality/pointnet_mean"
)
OUT_DIR = BASE / "outputs/thesis_figures/pipeline_supervised_powerdrill"

OBJECT_RGB = np.array([3, 40, 255], dtype=np.float32)
HANDLE_RGB = np.array([5, 215, 35], dtype=np.float32)
ALPHA = 0.70
C0 = 0.28209479177387814


def mask_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def overlay(image: Image.Image, masks_and_colors: list[tuple[np.ndarray, np.ndarray]]) -> Image.Image:
    out = np.asarray(image.convert("RGB"), dtype=np.float32)
    for mask, color in masks_and_colors:
        out[mask] = (1.0 - ALPHA) * out[mask] + ALPHA * color
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), mode="RGB")


def selected_frames() -> list[Path]:
    frames = sorted((SCENE_DIR / "rgb").glob("*.jpg"))
    if len(frames) < 4:
        raise RuntimeError(f"Expected at least four RGB frames, found {len(frames)}")
    indices = [int(np.floor(i * len(frames) / 4)) for i in range(4)]
    return [frames[i] for i in indices]


def prediction_slice() -> tuple[np.ndarray, float]:
    metrics = json.loads((MODEL_DIR / "overall_metrics.json").read_text())
    threshold = float(metrics["selected_threshold"])
    with np.load(MODEL_DIR / "test_predictions.npz", allow_pickle=False) as data:
        keys = [str(value) for value in data["scene_keys"]]
        index = keys.index(SCENE_KEY)
        start, end = (int(value) for value in data["offsets"][index])
        scores = data["scores"][start:end].astype(np.float32)
    return scores, threshold


def sh_to_rgb(sh: np.ndarray) -> np.ndarray:
    return np.clip(sh * C0 + 0.5, 0.0, 1.0)


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / C0


def export_prediction_ply() -> dict:
    scores, threshold = prediction_slice()
    ply = PlyData.read(str(SOURCE_PLY))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(scores):
        raise ValueError(f"PLY/prediction mismatch: {len(vertices)} vs {len(scores)}")

    base_sh = np.column_stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]
    ).astype(np.float32)
    base_rgb = sh_to_rgb(base_sh)
    prediction = scores >= threshold
    mask_rgb = np.empty_like(base_rgb)
    mask_rgb[~prediction] = OBJECT_RGB / 255.0
    mask_rgb[prediction] = HANDLE_RGB / 255.0
    blended_rgb = (1.0 - ALPHA) * base_rgb + ALPHA * mask_rgb
    blended_sh = rgb_to_sh(blended_rgb)
    vertices["f_dc_0"] = blended_sh[:, 0]
    vertices["f_dc_1"] = blended_sh[:, 1]
    vertices["f_dc_2"] = blended_sh[:, 2]

    elements = [
        PlyElement.describe(vertices, "vertex") if element.name == "vertex" else element
        for element in ply.elements
    ]
    out_path = OUT_DIR / f"{SCENE_KEY}_seen_prediction_overlay_a0.70.ply"
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(out_path))
    return {
        "source_ply": str(SOURCE_PLY),
        "output_ply": str(out_path),
        "model_dir": str(MODEL_DIR),
        "threshold": threshold,
        "num_gaussians": int(len(vertices)),
        "predicted_handle_gaussians": int(prediction.sum()),
        "predicted_handle_ratio": float(prediction.mean()),
        "overlay_alpha": ALPHA,
    }


def make_preview(rows: list[list[Image.Image]], out_path: Path) -> None:
    thumb_w = 360
    thumbs: list[list[Image.Image]] = []
    for row in rows:
        resized = []
        for image in row:
            height = int(round(image.height * thumb_w / image.width))
            resized.append(image.resize((thumb_w, height), Image.Resampling.LANCZOS))
        thumbs.append(resized)

    gap = 10
    label_w = 145
    row_h = max(image.height for row in thumbs for image in row)
    canvas = Image.new(
        "RGB",
        (label_w + 4 * thumb_w + 3 * gap, 3 * row_h + 2 * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    labels = ["RGB", "Object mask", "Handle labels"]
    for row_index, (label, images) in enumerate(zip(labels, thumbs)):
        y = row_index * (row_h + gap)
        draw.text((10, y + row_h // 2 - 8), label, fill="black")
        for column, image in enumerate(images):
            x = label_w + column * (thumb_w + gap)
            image_y = y + (row_h - image.height) // 2
            canvas.paste(image, (x, image_y))
            draw.rectangle(
                (x, image_y, x + image.width - 1, image_y + image.height - 1),
                outline="black",
                width=2,
            )
    canvas.save(out_path)


def main() -> None:
    rgb_dir = OUT_DIR / "01_rgb"
    object_dir = OUT_DIR / "02_object_mask_overlay_blue"
    handle_dir = OUT_DIR / "03_object_handle_overlay_blue_green"
    for path in (OUT_DIR, rgb_dir, object_dir, handle_dir):
        path.mkdir(parents=True, exist_ok=True)

    frame_records = []
    preview_rows: list[list[Image.Image]] = [[], [], []]
    for position, rgb_path in enumerate(selected_frames(), start=1):
        frame = rgb_path.stem
        object_mask_path = SCENE_DIR / "mask" / f"{frame}_000000.png"
        handle_mask_path = SCENE_DIR / "mask_parts" / f"{frame}_000000_handle.png"
        if not object_mask_path.exists() or not handle_mask_path.exists():
            raise FileNotFoundError(f"Missing masks for frame {frame}")

        image = Image.open(rgb_path).convert("RGB")
        object_mask = mask_array(object_mask_path)
        handle_mask = mask_array(handle_mask_path)
        if object_mask.shape != (image.height, image.width):
            raise ValueError(f"Object-mask size mismatch for frame {frame}")
        if handle_mask.shape != (image.height, image.width):
            raise ValueError(f"Handle-mask size mismatch for frame {frame}")

        raw_out = rgb_dir / f"{position:02d}_frame_{frame}.jpg"
        object_out = object_dir / f"{position:02d}_frame_{frame}_object_blue_a0.70.png"
        handle_out = handle_dir / f"{position:02d}_frame_{frame}_object_blue_handle_green_a0.70.png"
        shutil.copy2(rgb_path, raw_out)
        object_image = overlay(image, [(object_mask, OBJECT_RGB)])
        combined_image = overlay(
            image,
            [(object_mask, OBJECT_RGB), (handle_mask, HANDLE_RGB)],
        )
        object_image.save(object_out)
        combined_image.save(handle_out)
        preview_rows[0].append(image)
        preview_rows[1].append(object_image)
        preview_rows[2].append(combined_image)

        frame_records.append(
            {
                "position": position,
                "frame": frame,
                "source_rgb": str(rgb_path),
                "source_object_mask": str(object_mask_path),
                "source_handle_mask": str(handle_mask_path),
                "rgb_output": str(raw_out),
                "object_overlay_output": str(object_out),
                "object_handle_overlay_output": str(handle_out),
            }
        )

    preview_path = OUT_DIR / "power_drill_supervised_2d_pipeline_preview.png"
    make_preview(preview_rows, preview_path)
    prediction = export_prediction_ply()
    manifest = {
        "scene_key": SCENE_KEY,
        "source_scene": str(SCENE_DIR),
        "frame_selection": "ordered frame positions at 0%, 25%, 50%, and 75%",
        "object_color_rgb": OBJECT_RGB.astype(int).tolist(),
        "handle_color_rgb": HANDLE_RGB.astype(int).tolist(),
        "overlay_alpha": ALPHA,
        "frames": frame_records,
        "prediction": prediction,
        "preview": str(preview_path),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
