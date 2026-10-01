from __future__ import annotations

import json
from pathlib import Path

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


def forced_tile(
    row: dict[str, str],
    tile_w: int,
    tile_h: int,
    img_size: int,
    font: ImageFont.FreeTypeFont,
    forced_frame: str | None = None,
) -> tuple[Image.Image, dict]:
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
    raw = m.fit_square(rgb_crop, img_size)
    over = m.fit_square(m.overlay_masks(rgb_crop, obj_crop, handle_crop), img_size)

    tile = Image.new("RGB", (tile_w, tile_h), "white")
    draw = ImageDraw.Draw(tile)
    label = m.wrapped_label(m.DISPLAY_NAMES[row["category"]])
    label_box = draw.multiline_textbbox((0, 0), label, font=font, spacing=3, align="center")
    label_w = label_box[2] - label_box[0]
    draw.multiline_text(((tile_w - label_w) / 2, 0), label, fill=(25, 25, 25), font=font, spacing=3, align="center")

    y0 = 98
    x0 = (tile_w - img_size) // 2
    tile.paste(raw, (x0, y0))
    tile.paste(over, (x0, y0 + img_size + 14))

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
    }
    return tile, meta


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

    font = ImageFont.truetype(str(m.FONT_REG), 40)
    legend_font = ImageFont.truetype(str(m.FONT_REG), 36)

    tile_w = 520
    img_size = 450
    tile_h = 98 + img_size * 2 + 14
    gap_x = 36
    margin_x = 54
    margin_top = 48
    header_h = 58
    margin_bottom = 0

    canvas_w = margin_x * 2 + len(CATEGORIES) * tile_w + (len(CATEGORIES) - 1) * gap_x
    canvas_h = margin_top + header_h + tile_h + margin_bottom
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    legend_w = 650
    legend_x = (canvas_w - legend_w) // 2
    legend_y = margin_top - 5
    draw.rectangle((legend_x, legend_y, legend_x + 38, legend_y + 24), fill=(0, 205, 255), outline=(0, 95, 190), width=3)
    draw.text((legend_x + 52, legend_y - 7), "Object mask", fill=(35, 35, 35), font=legend_font)
    draw.rectangle((legend_x + 335, legend_y, legend_x + 373, legend_y + 24), fill=(255, 25, 35), outline=(130, 0, 0), width=3)
    draw.text((legend_x + 387, legend_y - 7), "Handle mask", fill=(35, 35, 35), font=legend_font)

    metas = []
    x = margin_x
    y = margin_top + header_h
    for cat in CATEGORIES:
        tile, meta = forced_tile(
            selected[cat],
            tile_w,
            tile_h,
            img_size,
            font,
            forced_frame=MUG_FRAME if cat == "mugs" else None,
        )
        canvas.paste(tile, (x, y))
        metas.append(meta)
        x += tile_w + gap_x

    png = m.OUT_DIR / "handal_dataset_6_categories_masks_pair.png"
    jpg = m.OUT_DIR / "handal_dataset_6_categories_masks_pair.jpg"
    meta_path = m.OUT_DIR / "handal_dataset_6_categories_masks_pair_metadata.json"
    canvas.save(png, dpi=(300, 300))
    canvas.save(jpg, quality=98, subsampling=0, dpi=(300, 300))
    meta_path.write_text(json.dumps(metas, indent=2))
    print(png)
    print(jpg)
    print(meta_path)


if __name__ == "__main__":
    main()
