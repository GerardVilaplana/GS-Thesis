from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
EXP40 = ROOT / "data/handal_exp40_15k_gt_features_v1"
SCENE_SPECS = EXP40 / "scene_specs.csv"
SUMMARY = EXP40 / "per_scene_summary.csv"
OUT_DIR = ROOT / "outputs/thesis_figures"

FONT_REG = Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf")
FONT_BOLD = Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf")

CATEGORY_ORDER = [
    "hammers",
    "fixed_joint_pliers",
    "slip_joint_pliers",
    "locking_pliers",
    "power_drills",
    "ratchets",
    "screwdrivers",
    "adjustable_wrenches",
    "combinational_wrenches",
    "ladles",
    "measuring_cups",
    "mugs",
    "pots_pans",
    "spatulas",
    "strainers",
    "utensils",
    "whisks",
]

BAD_SCENES = {
    "spatulas__032005",
    "slip_joint_pliers__090006",
    "slip_joint_pliers__092005",
    "locking_pliers__040003",
    "locking_pliers__091004",
    "utensils__022007",
    "slip_joint_pliers__093004",
    "locking_pliers__090004",
    "utensils__026007",
    "locking_pliers__041005",
}

REPLACE_SCENES = {
    "power_drills__011010",
    "mugs__002001",
}

DISPLAY_NAMES = {
    "hammers": "Hammer",
    "fixed_joint_pliers": "Fixed joint pliers",
    "slip_joint_pliers": "Slip joint pliers",
    "locking_pliers": "Locking pliers",
    "power_drills": "Power drill",
    "ratchets": "Ratchet",
    "screwdrivers": "Screwdriver",
    "adjustable_wrenches": "Adjustable wrench",
    "combinational_wrenches": "Combination wrench",
    "ladles": "Ladle",
    "measuring_cups": "Measuring cup",
    "mugs": "Mug",
    "pots_pans": "Pot/pan",
    "spatulas": "Spatula",
    "strainers": "Strainer",
    "utensils": "Utensil",
    "whisks": "Whisk",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mask_bool(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def find_rgb(scene: Path, frame: str) -> Path | None:
    for suffix in (".jpg", ".png"):
        path = scene / "rgb" / f"{frame}{suffix}"
        if path.exists():
            return path
    return None


def choose_representative_scenes() -> dict[str, dict[str, str]]:
    specs = {r["scene_key"]: r for r in read_csv(SCENE_SPECS)}
    summaries = [
        r
        for r in read_csv(SUMMARY)
        if r["variant"] == "B_center_ellipsoid20_strict75"
        and r["status"] == "ok"
        and r["scene_key"] not in BAD_SCENES
    ]
    by_cat: dict[str, list[dict[str, str]]] = {}
    for row in summaries:
        row = dict(row)
        row.update(specs.get(row["scene_key"], {}))
        by_cat.setdefault(row["category"], []).append(row)

    selected = {}
    for cat in CATEGORY_ORDER:
        rows = by_cat.get(cat, [])
        if not rows:
            raise RuntimeError(f"No scenes found for category {cat}")
        rows.sort(
            key=lambda r: (
                r.get("model_split") != "train",
                -float(r.get("handle_ratio", 0.0)),
                r["scene_key"],
            )
        )
        for row in rows:
            if row["scene_key"] not in REPLACE_SCENES:
                selected[cat] = row
                break
    return selected


def choose_frame(scene: Path) -> tuple[str, float]:
    best_frame = None
    best_score = -1.0
    for obj_path in sorted((scene / "mask").glob("*_000000.png")):
        frame = obj_path.name.split("_")[0]
        handle_path = scene / "mask_parts" / f"{frame}_000000_handle.png"
        rgb_path = find_rgb(scene, frame)
        if not handle_path.exists() or rgb_path is None:
            continue
        obj = mask_bool(obj_path)
        handle = mask_bool(handle_path)
        obj_area = float(obj.sum())
        handle_area = float(handle.sum())
        if obj_area <= 0 or handle_area <= 0:
            continue
        score = handle_area + 0.15 * obj_area
        if score > best_score:
            best_score = score
            best_frame = frame
    if best_frame is None:
        raise RuntimeError(f"No usable RGB/object/handle frame found in {scene}")
    return best_frame, best_score


def crop_box(mask: np.ndarray, image_size: tuple[int, int], margin: float = 0.22) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, image_size[0], image_size[1])
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    w = x1 - x0
    h = y1 - y0
    pad = int(max(w, h) * margin)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = int(max(w, h) + 2 * pad)
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(image_size[0], left + side)
    bottom = min(image_size[1], top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return (left, top, right, bottom)


def binary_contour(mask: Image.Image, radius: int = 1) -> Image.Image:
    dilated = mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    eroded = mask.filter(ImageFilter.MinFilter(radius * 2 + 1))
    return Image.fromarray((np.asarray(dilated) > np.asarray(eroded)).astype(np.uint8) * 255)


def overlay_masks(rgb: Image.Image, obj_mask: Image.Image, handle_mask: Image.Image) -> Image.Image:
    out = rgb.convert("RGBA")
    obj = obj_mask.convert("L")
    handle = handle_mask.convert("L")

    obj_layer = Image.new("RGBA", out.size, (0, 205, 255, 135))
    handle_layer = Image.new("RGBA", out.size, (255, 25, 35, 190))
    out = Image.composite(Image.alpha_composite(out, obj_layer), out, obj)
    out = Image.composite(Image.alpha_composite(out, handle_layer), out, handle)

    draw = ImageDraw.Draw(out)
    obj_contour = binary_contour(obj, 2)
    handle_contour = binary_contour(handle, 2)
    draw.bitmap((0, 0), obj_contour, fill=(0, 95, 190, 255))
    draw.bitmap((0, 0), handle_contour, fill=(130, 0, 0, 255))
    return out.convert("RGB")


def fit_square(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def wrapped_label(text: str) -> str:
    if text in {"Fixed joint pliers", "Slip joint pliers", "Locking pliers", "Adjustable wrench", "Combination wrench", "Measuring cup"}:
        parts = text.split(" ")
        return " ".join(parts[:-1]) + "\n" + parts[-1]
    return text


def make_tile(row: dict[str, str], tile_w: int, tile_h: int, img_size: int, font: ImageFont.FreeTypeFont) -> tuple[Image.Image, dict]:
    scene = Path(row["source_scene"])
    frame, frame_score = choose_frame(scene)
    rgb_path = find_rgb(scene, frame)
    obj_path = scene / "mask" / f"{frame}_000000.png"
    handle_path = scene / "mask_parts" / f"{frame}_000000_handle.png"

    rgb = Image.open(rgb_path).convert("RGB")
    obj = Image.open(obj_path).convert("L")
    handle = Image.open(handle_path).convert("L")
    box = crop_box(mask_bool(obj_path), rgb.size)

    rgb_crop = rgb.crop(box)
    obj_crop = obj.crop(box)
    handle_crop = handle.crop(box)
    raw = fit_square(rgb_crop, img_size)
    over = fit_square(overlay_masks(rgb_crop, obj_crop, handle_crop), img_size)

    tile = Image.new("RGB", (tile_w, tile_h), "white")
    draw = ImageDraw.Draw(tile)
    label = wrapped_label(DISPLAY_NAMES[row["category"]])
    label_box = draw.multiline_textbbox((0, 0), label, font=font, spacing=2, align="center")
    label_w = label_box[2] - label_box[0]
    draw.multiline_text(((tile_w - label_w) / 2, 0), label, fill=(25, 25, 25), font=font, spacing=2, align="center")
    y0 = 50
    x0 = (tile_w - img_size) // 2
    tile.paste(raw, (x0, y0))
    tile.paste(over, (x0, y0 + img_size + 10))
    draw.rectangle((x0, y0, x0 + img_size, y0 + img_size), outline=(215, 215, 215), width=1)
    draw.rectangle((x0, y0 + img_size + 10, x0 + img_size, y0 + 2 * img_size + 10), outline=(215, 215, 215), width=1)

    meta = {
        "category": row["category"],
        "display_name": DISPLAY_NAMES[row["category"]],
        "scene_key": row["scene_key"],
        "source_scene": str(scene),
        "frame": frame,
        "rgb": str(rgb_path),
        "object_mask": str(obj_path),
        "handle_mask": str(handle_path),
        "frame_score": frame_score,
        "crop_box": [int(v) for v in box],
    }
    return tile, meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT_REG), 30)
    legend_font = ImageFont.truetype(str(FONT_REG), 28)

    tile_w = 330
    img_size = 280
    tile_h = 50 + img_size * 2 + 22
    gap_x = 24
    gap_y = 42
    margin_x = 46
    margin_y = 40
    header_h = 46

    selected = choose_representative_scenes()
    groups = [CATEGORY_ORDER[:6], CATEGORY_ORDER[6:12], CATEGORY_ORDER[12:]]
    canvas_w = margin_x * 2 + 6 * tile_w + 5 * gap_x
    canvas_h = margin_y * 2 + header_h + 3 * tile_h + 2 * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    legend_w = 520
    legend_x = (canvas_w - legend_w) // 2
    legend_y = margin_y - 2
    draw.rectangle((legend_x, legend_y, legend_x + 30, legend_y + 18), fill=(0, 205, 255), outline=(0, 95, 190), width=2)
    draw.text((legend_x + 40, legend_y - 5), "Object mask", fill=(35, 35, 35), font=legend_font)
    draw.rectangle((legend_x + 250, legend_y, legend_x + 280, legend_y + 18), fill=(255, 25, 35), outline=(130, 0, 0), width=2)
    draw.text((legend_x + 290, legend_y - 5), "Handle mask", fill=(35, 35, 35), font=legend_font)

    metas = []
    y = margin_y + header_h
    for group in groups:
        row_w = len(group) * tile_w + (len(group) - 1) * gap_x
        x = margin_x + (6 * tile_w + 5 * gap_x - row_w) // 2
        for cat in group:
            tile, meta = make_tile(selected[cat], tile_w, tile_h, img_size, font)
            canvas.paste(tile, (x, y))
            metas.append(meta)
            x += tile_w + gap_x
        y += tile_h + gap_y

    png = OUT_DIR / "handal_dataset_17_categories_masks_montage.png"
    jpg = OUT_DIR / "handal_dataset_17_categories_masks_montage.jpg"
    meta_path = OUT_DIR / "handal_dataset_17_categories_masks_montage_metadata.json"
    canvas.save(png, dpi=(300, 300))
    canvas.save(jpg, quality=96, subsampling=0, dpi=(300, 300))
    meta_path.write_text(json.dumps(metas, indent=2))
    print(png)
    print(jpg)
    print(meta_path)


if __name__ == "__main__":
    main()
