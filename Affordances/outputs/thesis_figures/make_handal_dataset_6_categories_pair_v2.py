from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import make_handal_dataset_montage as m


CATEGORIES = [
    "hammers",
    "adjustable_wrenches",
    "ladles",
    "spatulas",
    "utensils",
    "mugs",
]

MUG_SCENE = "mugs__022002"
MUG_FRAME = "000238"

OBJECT_RGB = (3, 40, 255)
HANDLE_RGB = (5, 215, 35)
MASK_ALPHA = 0.70


def overlay_blue_green(rgb: Image.Image, obj_mask: Image.Image, handle_mask: Image.Image) -> Image.Image:
    out = rgb.convert("RGBA")
    obj = obj_mask.convert("L")
    handle = handle_mask.convert("L")
    obj_layer = Image.new("RGBA", out.size, (*OBJECT_RGB, int(round(255 * MASK_ALPHA))))
    handle_layer = Image.new("RGBA", out.size, (*HANDLE_RGB, int(round(255 * MASK_ALPHA))))
    out = Image.composite(Image.alpha_composite(out, obj_layer), out, obj)
    out = Image.composite(Image.alpha_composite(out, handle_layer), out, handle)
    return out.convert("RGB")


def fit_square_cover(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    scale = max(size / img.width, size / img.height)
    resized = img.resize(
        (int(round(img.width * scale)), int(round(img.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def forced_images(row: dict[str, str], img_size: int, forced_frame: str | None = None) -> tuple[Image.Image, Image.Image, dict]:
    scene = Path(row["source_scene"])
    frame = forced_frame or m.choose_frame(scene)[0]
    rgb_path = m.find_rgb(scene, frame)
    obj_path = scene / "mask" / f"{frame}_000000.png"
    handle_path = scene / "mask_parts" / f"{frame}_000000_handle.png"

    rgb = Image.open(rgb_path).convert("RGB")
    obj = Image.open(obj_path).convert("L")
    handle = Image.open(handle_path).convert("L")
    box = m.crop_box(m.mask_bool(obj_path), rgb.size)

    rgb_crop = rgb.crop(box)
    obj_crop = obj.crop(box)
    handle_crop = handle.crop(box)
    raw = fit_square_cover(rgb_crop, img_size)
    over = fit_square_cover(overlay_blue_green(rgb_crop, obj_crop, handle_crop), img_size)

    meta = {
        "category": row["category"],
        "display_name": m.DISPLAY_NAMES[row["category"]],
        "scene_key": row["scene_key"],
        "source_scene": str(scene),
        "frame": frame,
        "rgb": str(rgb_path),
        "object_mask": str(obj_path),
        "handle_mask": str(handle_path),
        "crop_box": [int(v) for v in box],
        "object_color_rgb": OBJECT_RGB,
        "handle_color_rgb": HANDLE_RGB,
        "mask_alpha": MASK_ALPHA,
    }
    return raw, over, meta


def draw_vertical_label(canvas: Image.Image, text: str, x_center: int, y_center: int, font: ImageFont.FreeTypeFont) -> None:
    tmp = Image.new("RGBA", (240, 80), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tmp)
    box = draw.textbbox((0, 0), text, font=font)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    draw.text(((tmp.width - text_w) / 2, (tmp.height - text_h) / 2 - 2), text, fill=(0, 0, 0, 255), font=font)
    tmp = tmp.crop(tmp.getbbox()).rotate(90, expand=True)
    canvas.paste(tmp.convert("RGB"), (x_center - tmp.width // 2, y_center - tmp.height // 2), tmp)


def main() -> None:
    m.OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = m.choose_representative_scenes()
    specs = {r["scene_key"]: r for r in m.read_csv(m.SCENE_SPECS)}
    mug_rows = [
        r
        for r in m.read_csv(m.SUMMARY)
        if r["variant"] == "B_center_ellipsoid20_strict75"
        and r["status"] == "ok"
        and r["scene_key"] == MUG_SCENE
    ]
    mug = dict(mug_rows[0])
    mug.update(specs.get(MUG_SCENE, {}))
    selected["mugs"] = mug

    label_font = ImageFont.truetype(str(m.FONT_REG), 38)
    row_font = ImageFont.truetype(str(m.FONT_BOLD), 38)
    legend_font = ImageFont.truetype(str(m.FONT_REG), 30)

    img_size = 355
    gap_x = 18
    gap_y = 18
    row_label_w = 86
    margin_x = 34
    margin_top = 24
    label_h = 82
    legend_h = 66
    margin_bottom = 18
    border_w = 4

    grid_w = len(CATEGORIES) * img_size + (len(CATEGORIES) - 1) * gap_x
    grid_h = img_size * 2 + gap_y
    canvas_w = margin_x * 2 + row_label_w + grid_w
    canvas_h = margin_top + grid_h + label_h + legend_h + margin_bottom
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    grid_x0 = margin_x + row_label_w
    raw_y = margin_top
    mask_y = margin_top + img_size + gap_y

    draw_vertical_label(canvas, "Image", margin_x + row_label_w // 2, raw_y + img_size // 2, row_font)
    draw_vertical_label(canvas, "Masks", margin_x + row_label_w // 2, mask_y + img_size // 2, row_font)

    metas = []
    x = grid_x0
    for cat in CATEGORIES:
        raw, over, meta = forced_images(
            selected[cat],
            img_size,
            forced_frame=MUG_FRAME if cat == "mugs" else None,
        )
        canvas.paste(raw, (x, raw_y))
        canvas.paste(over, (x, mask_y))
        draw.rectangle((x, raw_y, x + img_size, raw_y + img_size), outline=(0, 0, 0), width=border_w)
        draw.rectangle((x, mask_y, x + img_size, mask_y + img_size), outline=(0, 0, 0), width=border_w)

        label = m.wrapped_label(m.DISPLAY_NAMES[cat])
        label_box = draw.multiline_textbbox((0, 0), label, font=label_font, spacing=2, align="center")
        label_w = label_box[2] - label_box[0]
        draw.multiline_text(
            (x + (img_size - label_w) / 2, mask_y + img_size + 12),
            label,
            fill=(0, 0, 0),
            font=label_font,
            spacing=2,
            align="center",
        )
        metas.append(meta)
        x += img_size + gap_x

    swatch = 28
    legend_items = [("Object mask", OBJECT_RGB), ("Handle mask", HANDLE_RGB)]
    item_gap = 64
    item_widths = []
    for text, _ in legend_items:
        tb = draw.textbbox((0, 0), text, font=legend_font)
        item_widths.append(swatch + 12 + (tb[2] - tb[0]))
    legend_w = sum(item_widths) + item_gap * (len(legend_items) - 1) + 44
    legend_box_h = 52
    legend_x = (canvas_w - legend_w) // 2
    legend_y = canvas_h - margin_bottom - legend_box_h
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + legend_w, legend_y + legend_box_h),
        radius=5,
        fill=(255, 255, 255),
        outline=(55, 55, 55),
        width=2,
    )
    item_x = legend_x + 22
    for idx, (text, color) in enumerate(legend_items):
        y0 = legend_y + (legend_box_h - swatch) // 2
        draw.rectangle((item_x, y0, item_x + swatch, y0 + swatch), fill=color, outline=(0, 0, 0), width=1)
        draw.text((item_x + swatch + 12, legend_y + 9), text, fill=(25, 25, 25), font=legend_font)
        item_x += item_widths[idx] + item_gap

    png = m.OUT_DIR / "handal_dataset_6_categories_masks_pair.png"
    jpg = m.OUT_DIR / "handal_dataset_6_categories_masks_pair.jpg"
    meta_path = m.OUT_DIR / "handal_dataset_6_categories_masks_pair_metadata.json"
    v2_png = m.OUT_DIR / "handal_dataset_6_categories_masks_pair_v2.png"
    v2_jpg = m.OUT_DIR / "handal_dataset_6_categories_masks_pair_v2.jpg"

    for path in (png, v2_png):
        canvas.save(path, dpi=(300, 300))
    for path in (jpg, v2_jpg):
        canvas.save(path, quality=98, subsampling=0, dpi=(300, 300))
    meta_path.write_text(json.dumps(metas, indent=2))
    print(png)
    print(jpg)
    print(v2_png)
    print(v2_jpg)
    print(meta_path)


if __name__ == "__main__":
    main()
