import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sorted_pngs(path):
    return sorted(Path(path).glob('*.png'))


def resize_to_width(image, width):
    if width <= 0 or image.shape[1] == width:
        return image
    scale = width / image.shape[1]
    height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_label(image, text):
    im = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def make_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer for {path}')
    return writer


def main():
    parser = argparse.ArgumentParser(description='Create REALM object-mask diagnostic videos.')
    parser.add_argument('--root', default='output/lerf/figurines/train/ours_30000')
    parser.add_argument('--output_dir', default='output/lerf/figurines/mask_diagnostic_videos')
    parser.add_argument('--fps', type=float, default=12.0)
    parser.add_argument('--max_width', type=int, default=960)
    args = parser.parse_args()

    root = Path(args.root)
    pred_paths = sorted_pngs(root / 'objects_pred')
    render_paths = sorted_pngs(root / 'renders')
    feat_paths = sorted_pngs(root / 'objects_feature16')
    gt_paths = sorted_pngs(root / 'gt_objects_color')

    if not pred_paths:
        raise RuntimeError(f'No objects_pred PNGs found under {root}')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    first_pred = np.array(Image.open(pred_paths[0]).convert('RGB'))
    first_pred = resize_to_width(first_pred, args.max_width)
    h, w = first_pred.shape[:2]

    pred_writer = make_writer(out / 'objects_pred_train.mp4', args.fps, (w, h))
    side_writer = make_writer(out / 'render_vs_objects_pred_train.mp4', args.fps, (w * 2, h))

    full_writer = None
    has_full = len(render_paths) == len(pred_paths) and len(feat_paths) == len(pred_paths) and len(gt_paths) == len(pred_paths)
    if has_full:
        full_writer = make_writer(out / 'render_feature_pred_gt_train.mp4', args.fps, (w * 4, h))

    try:
        for i, pred_path in enumerate(pred_paths):
            pred = np.array(Image.open(pred_path).convert('RGB'))
            pred = resize_to_width(pred, args.max_width)
            pred_labeled = add_label(pred, f'objects_pred | frame {i:05d}')
            pred_writer.write(cv2.cvtColor(pred_labeled, cv2.COLOR_RGB2BGR))

            if i < len(render_paths):
                render = np.array(Image.open(render_paths[i]).convert('RGB'))
                render = resize_to_width(render, args.max_width)
            else:
                render = np.zeros_like(pred)
            side = np.concatenate([
                add_label(render, f'render | frame {i:05d}'),
                pred_labeled,
            ], axis=1)
            side_writer.write(cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

            if full_writer is not None:
                feat = resize_to_width(np.array(Image.open(feat_paths[i]).convert('RGB')), args.max_width)
                gt = resize_to_width(np.array(Image.open(gt_paths[i]).convert('RGB')), args.max_width)
                full = np.concatenate([
                    add_label(render, f'render | frame {i:05d}'),
                    add_label(feat, 'objects_feature16'),
                    pred_labeled,
                    add_label(gt, 'gt_objects_color'),
                ], axis=1)
                full_writer.write(cv2.cvtColor(full, cv2.COLOR_RGB2BGR))

            if (i + 1) % 50 == 0 or i == len(pred_paths) - 1:
                print(f'Processed {i + 1}/{len(pred_paths)} frames', flush=True)
    finally:
        pred_writer.release()
        side_writer.release()
        if full_writer is not None:
            full_writer.release()

    report = out / 'mask_video_report.txt'
    report.write_text(
        f'root: {root}\n'
        f'frames: {len(pred_paths)}\n'
        f'fps: {args.fps}\n'
        f'size: {w}x{h}\n'
        f'has_full_four_panel: {has_full}\n'
    )
    print(f'Saved videos to {out}')


if __name__ == '__main__':
    main()
