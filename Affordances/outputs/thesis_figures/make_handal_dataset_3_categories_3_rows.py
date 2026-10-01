from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import make_handal_dataset_6_categories_pair_v2 as pair
import make_handal_dataset_montage as montage


CATEGORIES = ["hammers", "ladles", "mugs"]
ROW_LABELS = ["Image", "Object mask", "Handle mask"]


def overlay(rgb: Image.Image, mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    base = rgb.convert("RGBA")
    layer = Image.new(
        "RGBA",
        base.size,
        (*color, int(round(255 * pair.MASK_ALPHA))),
    )
    return Image.composite(Image.alpha_composite(base, layer), base, mask.convert("L")).convert("RGB")


def category_images(row: dict[str, str], forced_frame: str | None = None) -> tuple[list[Image.Image], dict]:
    scene = Path(row["source_scene"])
    frame = forced_frame or montage.choose_frame(scene)[0]
    rgb_path = montage.find_rgb(scene, frame)
    object_path = scene / "mask" / f"{frame}_000000.png"
    handle_path = scene / "mask_parts" / f"{frame}_000000_handle.png"

    rgb = Image.open(rgb_path).convert("RGB")
    object_mask = Image.open(object_path).convert("L")
    handle_mask = Image.open(handle_path).convert("L")
    box = montage.crop_box(montage.mask_bool(object_path), rgb.size)
    rgb = rgb.crop(box)
    object_mask = object_mask.crop(box)
    handle_mask = handle_mask.crop(box)

    images = [
        pair.fit_square_cover(rgb, 520),
        pair.fit_square_cover(overlay(rgb, object_mask, pair.OBJECT_RGB), 520),
        pair.fit_square_cover(overlay(rgb, handle_mask, pair.HANDLE_RGB), 520),
    ]
    metadata = {
        "category": row["category"],
        "display_name": montage.DISPLAY_NAMES[row["category"]],
        "scene_key": row["scene_key"],
        "source_scene": str(scene),
        "frame": frame,
        "rgb": str(rgb_path),
        "object_mask": str(object_path),
        "handle_mask": str(handle_path),
        "crop_box": [int(value) for value in box],
    }
    return images, metadata


def main() -> None:
    montage.OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = montage.choose_representative_scenes()
    specs = {row["scene_key"]: row for row in montage.read_csv(montage.SCENE_SPECS)}
    mug_rows = [
        row
        for row in montage.read_csv(montage.SUMMARY)
        if row["variant"] == "B_center_ellipsoid20_strict75"
        and row["status"] == "ok"
        and row["scene_key"] == pair.MUG_SCENE
    ]
    mug = dict(mug_rows[0])
    mug.update(specs.get(pair.MUG_SCENE, {}))
    selected["mugs"] = mug

    image_size = 520
    gap_x = 24
    gap_y = 18
    margin_x = 28
    margin_y = 24
    row_label_width = 116
    category_label_height = 62
    border_width = 4
    label_font = ImageFont.truetype(str(montage.FONT_REG), 42)
    row_font = ImageFont.truetype(str(montage.FONT_BOLD), 36)

    grid_width = len(CATEGORIES) * image_size + (len(CATEGORIES) - 1) * gap_x
    grid_height = len(ROW_LABELS) * image_size + (len(ROW_LABELS) - 1) * gap_y
    canvas = Image.new(
        "RGB",
        (
            2 * margin_x + row_label_width + grid_width,
            2 * margin_y + grid_height + category_label_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    grid_x = margin_x + row_label_width

    for index, text in enumerate(ROW_LABELS):
        y = margin_y + index * (image_size + gap_y)
        pair.draw_vertical_label(
            canvas,
            text,
            margin_x + row_label_width // 2,
            y + image_size // 2,
            row_font,
        )

    metadata = []
    for column, category in enumerate(CATEGORIES):
        images, record = category_images(
            selected[category],
            forced_frame=pair.MUG_FRAME if category == "mugs" else None,
        )
        x = grid_x + column * (image_size + gap_x)
        for row, image in enumerate(images):
            y = margin_y + row * (image_size + gap_y)
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + image_size, y + image_size),
                outline="black",
                width=border_width,
            )

        name = montage.DISPLAY_NAMES[category]
        text_box = draw.textbbox((0, 0), name, font=label_font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (x + (image_size - text_width) / 2, margin_y + grid_height + 12),
            name,
            fill="black",
            font=label_font,
        )
        metadata.append(record)

    output = montage.OUT_DIR / "handal_dataset_3_categories_masks_3_rows.png"
    metadata_path = montage.OUT_DIR / "handal_dataset_3_categories_masks_3_rows_metadata.json"
    canvas.save(output, dpi=(300, 300))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)
    print(metadata_path)


if __name__ == "__main__":
    main()
