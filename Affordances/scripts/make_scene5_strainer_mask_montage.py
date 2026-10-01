from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    mask_dir = Path(
        "/home/gvilaplana/GS-Thesis/Affordances/data/own_scenes/scene_5/strainer_masks"
    )
    img_dir = Path(
        "/home/gvilaplana/GS-Thesis/Affordances/data/own_scenes/scene_5/scene5_tmp_frames_for_sam2"
    )
    out_dir = Path(
        "/home/gvilaplana/GS-Thesis/Affordances/outputs/06_own_scenes/"
        "scene_5_strainer_corrected_15k_best_dino_handle_predictions"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corrected_strainer_masks_overlay_montage.png"

    frames = [0, 10, 25, 50, 75, 103]
    colors = {
        1: np.array([0, 210, 45], dtype=np.uint8),
        2: np.array([255, 145, 0], dtype=np.uint8),
    }
    labels = {
        1: "ID 1: corrected strainer",
        2: "ID 2: added mask from original labels",
    }
    alpha = 0.68
    thumb_w = 420
    pad = 18
    label_h = 34
    legend_h = 70
    cols = 3
    rows = 2

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    font_small = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
    )

    tiles = []
    for idx in frames:
        mask_path = mask_dir / f"object_frame_{idx:04d}.npy"
        img_path = img_dir / f"{idx:08d}.jpg"
        mask = np.load(mask_path)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32)

        if mask.shape[:2] != img_np.shape[:2]:
            mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
            mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)
            mask = np.array(mask_img)

        overlay = img_np.copy()
        for mask_id, color in colors.items():
            selected = mask == mask_id
            overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * color

        tile = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        scale = thumb_w / tile.width
        tile = tile.resize(
            (thumb_w, int(round(tile.height * scale))), Image.Resampling.LANCZOS
        )

        canvas = Image.new("RGB", (thumb_w, tile.height + label_h), "white")
        canvas.paste(tile, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, thumb_w - 1, tile.height - 1], outline="black", width=2)
        present_labels = sorted(int(x) for x in np.unique(mask) if x != 0)
        text = f"Frame {idx:03d}  labels: {present_labels}"
        bbox = draw.textbbox((0, 0), text, font=font_small)
        draw.text(
            ((thumb_w - (bbox[2] - bbox[0])) // 2, tile.height + 7),
            text,
            fill="black",
            font=font_small,
        )
        tiles.append(canvas)

    max_h = max(tile.height for tile in tiles)
    width = cols * thumb_w + (cols + 1) * pad
    height = rows * max_h + (rows + 1) * pad + legend_h
    montage = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(montage)

    for n, tile in enumerate(tiles):
        row, col = divmod(n, cols)
        x = pad + col * (thumb_w + pad)
        y = pad + row * (max_h + pad)
        montage.paste(tile, (x, y))

    legend_y = pad + rows * (max_h + pad) + 8
    legend_x = pad
    for mask_id in [1, 2]:
        color = tuple(int(v) for v in colors[mask_id])
        draw.rounded_rectangle(
            [legend_x, legend_y, legend_x + 30, legend_y + 30],
            radius=4,
            fill=color,
            outline="black",
            width=1,
        )
        draw.text(
            (legend_x + 40, legend_y + 4),
            labels[mask_id],
            fill="black",
            font=font,
        )
        legend_x += 430

    montage.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
