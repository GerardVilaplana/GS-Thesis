import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_contact_sheet(frame_paths, output_path, tile_width=300, max_tiles=12):
    if not frame_paths:
        return
    font = ImageFont.load_default()
    indices = np.linspace(0, len(frame_paths) - 1, num=min(max_tiles, len(frame_paths)), dtype=int)
    tiles = []
    for idx in indices:
        path = frame_paths[int(idx)]
        im = Image.open(path).convert("RGB")
        im.thumbnail((tile_width, int(tile_width * 0.72)), Image.BICUBIC)
        tile = Image.new("RGB", (tile_width + 10, int(tile_width * 0.72) + 36), (255, 255, 255))
        tile.paste(im, ((tile.width - im.width) // 2, 28 + (int(tile_width * 0.72) - im.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tile.width, 24], fill=(0, 0, 0))
        draw.text((6, 7), path.stem, fill=(255, 255, 255), font=font)
        tiles.append(tile)

    cols = min(4, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tiles[0].width, rows * tiles[0].height), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * tile.width, (i // cols) * tile.height))
    sheet.save(output_path, quality=92)


def main():
    parser = argparse.ArgumentParser(description="Create RGB + object_pred overlay frames and videos.")
    parser.add_argument("--render_dir", required=True, help="Path to train/ours_ITER or test/ours_ITER.")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    gt_dir = render_dir / "gt"
    pred_dir = render_dir / "objects_pred"
    overlay_dir = render_dir / "object_pred_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    overlay_video = render_dir / "object_pred_overlay.mp4"
    side_video = render_dir / "object_pred_overlay_side_by_side.mp4"
    contact_sheet = render_dir / "object_pred_overlay_contact_sheet.jpg"

    gt_paths = sorted(gt_dir.glob("*.png"))
    font = ImageFont.load_default()
    writer = None
    side_writer = None
    overlay_paths = []

    for gt_path in gt_paths:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue

        gt = np.array(Image.open(gt_path).convert("RGB"))
        pred = np.array(Image.open(pred_path).convert("RGB"))
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = cv2.addWeighted(gt, 1.0 - args.alpha, pred, args.alpha, 0.0).astype(np.uint8)
        overlay_path = overlay_dir / gt_path.name
        Image.fromarray(overlay).save(overlay_path)
        overlay_paths.append(overlay_path)

        side = np.hstack([gt, overlay, pred]).astype(np.uint8)
        pil_side = Image.fromarray(side)
        draw = ImageDraw.Draw(pil_side)
        labels = [
            ("RGB", 8),
            ("RGB + predicted ID overlay", gt.shape[1] + 8),
            ("predicted IDs", 2 * gt.shape[1] + 8),
        ]
        for text, x in labels:
            draw.rectangle([x - 4, 5, x + len(text) * 7 + 4, 24], fill=(0, 0, 0))
            draw.text((x, 10), text, fill=(255, 255, 255), font=font)
        side = np.array(pil_side)

        if writer is None:
            h, w = overlay.shape[:2]
            writer = cv2.VideoWriter(str(overlay_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            side_writer = cv2.VideoWriter(
                str(side_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (side.shape[1], side.shape[0])
            )

        writer.write(overlay[:, :, ::-1])
        side_writer.write(side[:, :, ::-1])

    if writer is not None:
        writer.release()
    if side_writer is not None:
        side_writer.release()

    write_contact_sheet(overlay_paths, contact_sheet)

    print(f"overlay_frames={len(overlay_paths)}")
    print(f"overlay_dir={overlay_dir}")
    print(f"overlay_video={overlay_video}")
    print(f"side_by_side_video={side_video}")
    print(f"contact_sheet={contact_sheet}")


if __name__ == "__main__":
    main()
