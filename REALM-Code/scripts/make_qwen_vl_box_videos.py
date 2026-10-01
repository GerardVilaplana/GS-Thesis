import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

FRAME_RE = re.compile(r"frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def sorted_frame_paths(image_dir):
    paths = [p for p in Path(image_dir).iterdir() if p.is_file() and FRAME_RE.match(p.name)]
    return sorted(paths, key=lambda p: int(FRAME_RE.match(p.name).group(1)))


def extract_json(text):
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def resize_np(image_np, max_width):
    if max_width <= 0 or image_np.shape[1] <= max_width:
        return image_np
    scale = max_width / image_np.shape[1]
    new_h = int(round(image_np.shape[0] * scale))
    return np.array(Image.fromarray(image_np).resize((max_width, new_h), Image.BICUBIC))


def add_header(im, text):
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)


def draw_boxes(image_path, boxes, query, frame_name, max_width):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    add_header(im, f"Qwen2.5-VL boxes | {query} | {frame_name} | boxes={len(boxes)}")
    colors = [(255, 40, 40), (40, 200, 255), (255, 220, 40), (80, 255, 120), (220, 80, 255)]
    for i, item in enumerate(boxes):
        box = item.get("bbox_2d", item.get("box", []))
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1 = max(0, min(im.width - 1, x1))
        x2 = max(0, min(im.width - 1, x2))
        y1 = max(0, min(im.height - 1, y1))
        y2 = max(0, min(im.height - 1, y2))
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = str(item.get("label", f"box {i}"))
        if "score" in item:
            label += f" {float(item['score']):.2f}"
        tx, ty = x1 + 4, max(28, y1 - 15)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color, font=font)
    arr = np.array(im)
    return resize_np(arr, max_width)


def make_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def build_prompt(image_path, query):
    im = Image.open(image_path)
    return (
        "Locate the object or objects described by the query in this image. "
        f"Image size is width={im.width}, height={im.height}. "
        "Return ONLY a valid JSON list. Each item must contain "
        "bbox_2d as [x1, y1, x2, y2] in absolute pixel coordinates, "
        "label, and explanation. Do not include markdown or extra text. "
        f"Query: {query}"
    )


def run_qwen(model, processor, image_path, query, max_new_tokens):
    prompt = build_prompt(image_path, query)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    try:
        boxes = extract_json(output_text)
        if not isinstance(boxes, list):
            boxes = []
        parse_error = None
    except Exception as e:
        boxes = []
        parse_error = f"{type(e).__name__}: {e}"
    return output_text, boxes, parse_error


def main():
    parser = argparse.ArgumentParser(description="Generate Qwen2.5-VL bbox videos for ordered LERF frames.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_dir", default="/home/gvilaplana/GS-Thesis/data/lerf/figurines/images")
    parser.add_argument("--output_dir", default="output/lerf/figurines/qwen_vl_box_videos")
    parser.add_argument("--queries", nargs="+", default=["yellow rubber duck", "red apple", "green apple", "camera", "chairs"])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    frames = sorted_frame_paths(args.image_dir)
    if args.stride > 1:
        frames = frames[:: args.stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("No frames selected")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    first_np = resize_np(np.array(Image.open(frames[0]).convert("RGB")), args.max_width)
    h, w = first_np.shape[:2]

    print(f"Loading model {args.model}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    summary = {
        "settings": vars(args),
        "num_frames": len(frames),
        "first_frame": frames[0].name,
        "last_frame": frames[-1].name,
        "queries": {},
    }

    for query in args.queries:
        slug = query.lower().replace(" ", "_").replace("/", "_")
        writer = make_writer(out_root / f"{slug}_qwen_vl_boxes.mp4", args.fps, (w, h))
        summary["queries"][query] = []
        print(f"Processing query: {query}", flush=True)
        try:
            for idx, frame in enumerate(frames):
                raw, boxes, parse_error = run_qwen(model, processor, frame, query, args.max_new_tokens)
                overlay = draw_boxes(frame, boxes, query, frame.name, args.max_width)
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                summary["queries"][query].append({
                    "frame_index": idx,
                    "frame_name": frame.name,
                    "raw_output": raw,
                    "parsed": boxes,
                    "parse_error": parse_error,
                    "num_boxes": len(boxes),
                })
                if (idx + 1) % 10 == 0 or idx == len(frames) - 1:
                    print(f"  {query}: {idx + 1}/{len(frames)}", flush=True)
        finally:
            writer.release()
        (out_root / "qwen_vl_box_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Saved video for {query}", flush=True)

    print(f"Saved outputs to {out_root}", flush=True)


if __name__ == "__main__":
    main()
