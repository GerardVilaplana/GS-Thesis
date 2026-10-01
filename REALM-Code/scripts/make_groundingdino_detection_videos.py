import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from ext.grounded_sam import load_model_hf

FRAME_RE = re.compile(r"frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def frame_number(path):
    m = FRAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1))


def sorted_frame_paths(image_dir):
    paths = [p for p in Path(image_dir).iterdir() if p.is_file() and FRAME_RE.match(p.name)]
    return sorted(paths, key=lambda p: frame_number(p))


def resize_np(image_np, max_width):
    if max_width <= 0 or image_np.shape[1] <= max_width:
        return image_np
    scale = max_width / image_np.shape[1]
    new_h = int(round(image_np.shape[0] * scale))
    return np.array(Image.fromarray(image_np).resize((max_width, new_h), Image.BICUBIC))


def draw_boxes(image_np, boxes_xyxy, phrases, logits, query, frame_name):
    im = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
    draw.text((6, 6), f"{frame_name} | {query} | boxes={len(boxes_xyxy)}", fill=(255, 255, 255), font=font)
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        color = [(255, 50, 50), (50, 210, 255), (255, 210, 50), (80, 255, 120), (200, 80, 255)][i % 5]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        phrase = phrases[i] if i < len(phrases) else "box"
        score = float(logits[i]) if i < len(logits) else 0.0
        label = f"{i}: {phrase} {score:.2f}"
        tx, ty = x1 + 4, max(26, y1 - 15)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color, font=font)
    return np.array(im)


def run_grounding_dino(model, image_np, prompt, box_threshold, text_threshold):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(Image.fromarray(image_np), None)
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    h, w, _ = image_np.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h], dtype=torch.float32)
    return boxes_xyxy.detach().cpu().numpy(), logits.detach().cpu().numpy(), phrases


def main():
    parser = argparse.ArgumentParser(description="Make detector-only GroundingDINO videos for ordered LERF frames.")
    parser.add_argument("--image_dir", default="/home/gvilaplana/GS-Thesis/data/lerf/figurines/images")
    parser.add_argument("--output_dir", default="output/lerf/figurines/groundingdino_detection_videos")
    parser.add_argument("--queries", nargs="+", default=["yellow rubber duck", "red apple", "green apple", "camera", "chairs"])
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--max_width", type=int, default=960)
    args = parser.parse_args()

    image_paths = sorted_frame_paths(args.image_dir)
    if args.stride > 1:
        image_paths = image_paths[::args.stride]
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise RuntimeError(f"No frame_XXXXX images found in {args.image_dir}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    nums = [frame_number(p) for p in image_paths]
    gaps = [b - a for a, b in zip(nums[:-1], nums[1:])]
    order_report = {
        "image_dir": str(Path(args.image_dir).resolve()),
        "num_frames_used": len(image_paths),
        "first_frame": image_paths[0].name,
        "last_frame": image_paths[-1].name,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "is_monotonic_in_filename_order": all(b > a for a, b in zip(nums[:-1], nums[1:])),
        "unique_frame_number_gaps": sorted(set(gaps)),
        "interpretation": "The dataset images are sequential frame_XXXXX files, so this is very likely a video/camera-path order. This script uses that order.",
    }
    (out_root / "frame_order_report.json").write_text(json.dumps(order_report, indent=2))

    model = load_model_hf(
        "ShilongLiu/GroundingDINO",
        "groundingdino_swinb_cogcoor.pth",
        "GroundingDINO_SwinB.cfg.py",
        device="cuda",
    ).cuda().eval()

    summary = {"queries": {}, "settings": vars(args), "frame_order": order_report}
    first_np = np.array(Image.open(image_paths[0]).convert("RGB"))
    first_np = resize_np(first_np, args.max_width)
    frame_h, frame_w = first_np.shape[:2]

    writers = {}
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for query in args.queries:
            slug = query.lower().replace(" ", "_").replace("/", "_")
            video_path = out_root / f"{slug}_groundingdino_boxes.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (frame_w, frame_h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {video_path}")
            writers[query] = writer
            summary["queries"][query] = []

        with torch.no_grad():
            for idx, path in enumerate(image_paths):
                image_np = np.array(Image.open(path).convert("RGB"))
                image_np = resize_np(image_np, args.max_width)
                for query in args.queries:
                    boxes, logits, phrases = run_grounding_dino(model, image_np, query, args.box_threshold, args.text_threshold)
                    overlay = draw_boxes(image_np, boxes, phrases, logits, query, path.name)
                    writers[query].write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    summary["queries"][query].append({
                        "frame_index": idx,
                        "frame_name": path.name,
                        "num_boxes": int(len(boxes)),
                        "phrases": list(map(str, phrases)),
                        "scores": [float(x) for x in logits],
                        "boxes_xyxy": [[float(v) for v in box] for box in boxes],
                    })
                if (idx + 1) % 25 == 0 or idx == len(image_paths) - 1:
                    print(f"Processed {idx + 1}/{len(image_paths)} frames")
    finally:
        for writer in writers.values():
            writer.release()

    (out_root / "groundingdino_detection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved videos and summaries to {out_root}")


if __name__ == "__main__":
    main()
