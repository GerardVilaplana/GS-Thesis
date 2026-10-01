import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def extract_json(text):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def draw_boxes(image_path, boxes, output_path):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
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
        color = [(255, 40, 40), (40, 200, 255), (255, 220, 40), (80, 255, 120)][i % 4]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = item.get("label", f"box {i}")
        draw.text((x1 + 4, max(0, y1 - 14)), str(label), fill=color, font=font)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image", default="/home/gvilaplana/GS-Thesis/data/lerf/figurines/images/frame_00001.jpg")
    parser.add_argument("--query", default="yellow rubber duck")
    parser.add_argument("--output", default="output/lerf/figurines/qwen_vl_tests/smoke_yellow_rubber_duck.png")
    parser.add_argument("--json_output", default="output/lerf/figurines/qwen_vl_tests/smoke_yellow_rubber_duck.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    im = Image.open(args.image)
    prompt = (
        "Locate the object described by the query in this image. "
        f"Image size is width={im.width}, height={im.height}. "
        "Return ONLY a valid JSON list. Each item must have: "
        "bbox_2d as [x1, y1, x2, y2] in absolute pixel coordinates, "
        "label, and explanation. Do not include markdown. "
        f"Query: {args.query}"
    )

    print(f"Loading model: {args.model}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
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

    print("Generating...", flush=True)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output_text, flush=True)

    try:
        boxes = extract_json(output_text)
    except Exception as e:
        boxes = []
        print(f"JSON parse failed: {type(e).__name__}: {e}", flush=True)

    payload = {"query": args.query, "image": args.image, "raw_output": output_text, "parsed": boxes}
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(payload, indent=2))
    draw_boxes(args.image, boxes, args.output)
    print(f"Saved {args.output}")
    print(f"Saved {args.json_output}")


if __name__ == "__main__":
    main()
