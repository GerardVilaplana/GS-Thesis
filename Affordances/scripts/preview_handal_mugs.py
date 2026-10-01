from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData


ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
OUT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_previews")


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_mask(path):
    if not path.exists():
        return None
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def overlay_mask(rgb, mask, color, alpha=0.45):
    base = np.array(rgb).astype(np.float32)
    if mask is None:
        return Image.fromarray(base.astype(np.uint8))
    paint = np.zeros_like(base)
    paint[..., 0] = color[0]
    paint[..., 1] = color[1]
    paint[..., 2] = color[2]
    base[mask] = (1.0 - alpha) * base[mask] + alpha * paint[mask]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def tile_label(image, text):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    pad = 5
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * pad
    h = bbox[3] - bbox[1] + 2 * pad
    draw.rectangle((0, 0, w, h), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return image


def select_frames(scene_dir, count=4):
    rgb_files = sorted((scene_dir / "rgb").glob("*.jpg"))
    if len(rgb_files) <= count:
        return rgb_files
    indices = np.linspace(0, len(rgb_files) - 1, count).round().astype(int)
    return [rgb_files[i] for i in indices]


def contact_sheet(scene_rel, out_name):
    scene_dir = ROOT / scene_rel
    frames = select_frames(scene_dir, 4)
    tiles = []
    tile_w = 256
    tile_h = 192
    for rgb_path in frames:
        stem = rgb_path.stem
        rgb = load_rgb(rgb_path).resize((tile_w, tile_h))
        obj_mask = load_mask(scene_dir / "mask" / f"{stem}_000000.png")
        handle_mask = load_mask(scene_dir / "mask_parts" / f"{stem}_000000_handle.png")

        if obj_mask is not None:
            obj_mask = np.array(Image.fromarray(obj_mask.astype(np.uint8) * 255).resize((tile_w, tile_h))) > 0
        if handle_mask is not None:
            handle_mask = np.array(Image.fromarray(handle_mask.astype(np.uint8) * 255).resize((tile_w, tile_h))) > 0

        tiles.append(tile_label(rgb, f"{scene_rel} rgb {stem}"))
        tiles.append(tile_label(overlay_mask(rgb, obj_mask, (255, 45, 45)), "object mask"))
        tiles.append(tile_label(overlay_mask(rgb, handle_mask, (0, 210, 255), 0.60), "handle mask"))

    cols = 3
    rows = len(frames)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (245, 245, 245))
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        sheet.paste(tile, (x, y))
    out_path = OUT / out_name
    sheet.save(out_path)
    return out_path


def read_vertices(path, max_points=5000):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points).round().astype(int)
        xyz = xyz[idx]
    return xyz


def setup_3d_axis(ax, xyz, title):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=18, azim=-55)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def mesh_preview(model_ids):
    fig = plt.figure(figsize=(12, 8), dpi=160)
    for i, model_id in enumerate(model_ids):
        obj_id = f"{model_id:06d}"
        body_path = ROOT / "models_parts" / f"obj_{obj_id}_not.ply"
        handle_path = ROOT / "models_parts" / f"obj_{obj_id}_handle.ply"
        body = read_vertices(body_path)
        handle = read_vertices(handle_path)
        all_xyz = np.vstack([body, handle])

        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.scatter(body[:, 0], body[:, 1], body[:, 2], s=0.25, c="#9a9a9a", depthshade=False)
        ax.scatter(handle[:, 0], handle[:, 1], handle[:, 2], s=0.7, c="#00a6d6", depthshade=False)
        setup_3d_axis(ax, all_xyz, f"obj_{obj_id}: cyan = handle")

    fig.tight_layout(pad=0.2)
    out_path = OUT / "handal_mug_mesh_handle_preview.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    written = [
        contact_sheet(Path("train/001001"), "handal_train_001001_rgb_masks_handles.png"),
        contact_sheet(Path("test/004001"), "handal_test_004001_rgb_masks_handles.png"),
        contact_sheet(Path("dynamic/002999_train"), "handal_dynamic_002999_train_rgb_masks_handles.png"),
        mesh_preview([1, 2, 3, 21, 22, 25]),
    ]
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
