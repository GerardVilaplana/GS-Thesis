import argparse
import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scene import Scene
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from segment_anything import sam_model_registry, SamPredictor

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from ext.grounded_sam import load_model_hf
from utils.camera_utils import sample_reason_cameras_cluster_then_topk


def tensor_to_np_image(t):
    return (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def id_to_rgb(obj_id, max_num_obj=256):
    if not 0 <= obj_id <= max_num_obj:
        return np.zeros((3,), dtype=np.uint8)
    if obj_id == 0:
        return np.zeros((3,), dtype=np.uint8)
    import colorsys

    golden_ratio = 1.6180339887
    h = (obj_id * golden_ratio) % 1
    s = 0.5 + (obj_id % 2) * 0.5
    l = 0.5
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    for obj_id in np.unique(objects):
        rgb_mask[objects == obj_id] = id_to_rgb(int(obj_id))
    return rgb_mask


def draw_boxes(image_np, boxes_xyxy, labels=None, scores=None):
    im = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=4)
        label = labels[i] if labels and i < len(labels) else "box"
        if scores is not None and i < len(scores):
            label = f"{label} {float(scores[i]):.2f}"
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=(255, 40, 40), font=font)
    return im


def masks_to_overlay(image_np, masks):
    base = Image.fromarray(image_np).convert("RGBA")
    if masks is None or len(masks) == 0:
        return base.convert("RGB")
    colors = [
        (255, 0, 0, 120),
        (0, 180, 255, 120),
        (255, 200, 0, 120),
        (0, 255, 120, 120),
        (180, 0, 255, 120),
        (255, 80, 160, 120),
    ]
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for i, mask in enumerate(masks):
        mask_np = mask.detach().bool().cpu().numpy().astype(np.uint8) * 255
        color = Image.new("RGBA", base.size, colors[i % len(colors)])
        overlay.alpha_composite(Image.composite(color, Image.new("RGBA", base.size, (0, 0, 0, 0)), Image.fromarray(mask_np)))
    return Image.alpha_composite(base, overlay).convert("RGB")


def realm_overlay(image_np, pred_obj, masks, ioa_threshold):
    base = Image.fromarray(image_np).convert("RGBA")
    h, w = pred_obj.shape
    pred_cpu = pred_obj.detach().cpu()
    selected_info = []
    union = torch.zeros_like(pred_obj, dtype=torch.bool)

    for det_i, mask in enumerate(masks):
        mask = mask.to(pred_obj.device).bool()
        unique_ids = torch.unique(pred_obj[mask])
        scores = []
        for obj_id in unique_ids:
            obj_mask = pred_obj == obj_id
            obj_area = obj_mask.sum().float()
            if obj_area.item() == 0:
                continue
            inter = (obj_mask & mask).sum().float()
            ioa = inter / obj_area
            iom = inter / mask.sum().float().clamp_min(1.0)
            scores.append((float(ioa.item()), float(iom.item()), int(obj_id.item())))
        scores.sort(reverse=True)
        kept = [(ioa, iom, oid) for ioa, iom, oid in scores if ioa >= ioa_threshold]
        if not kept and scores:
            kept = [scores[0]]
        for ioa, iom, oid in kept:
            union |= pred_obj == oid
            selected_info.append((det_i, oid, ioa, iom))

    if union.any():
        color_mask = visualize_obj(pred_cpu.numpy().astype(np.uint8))
        selected_np = union.detach().cpu().numpy()
        overlay_np = np.zeros((h, w, 4), dtype=np.uint8)
        overlay_np[selected_np, :3] = color_mask[selected_np]
        overlay_np[selected_np, 3] = 150
        overlay = Image.fromarray(overlay_np, mode="RGBA")
        result = Image.alpha_composite(base, overlay)
    else:
        result = base

    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    y = 4
    for det_i, oid, ioa, iom in selected_info[:12]:
        draw.text((4, y), f"d{det_i}->id{oid} ioa={ioa:.2f} iom={iom:.2f}", fill=(255, 255, 255, 255), font=font)
        y += 12
    return result.convert("RGB"), selected_info


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
    return boxes_xyxy, logits, phrases


def run_sam(sam_predictor, image_np, boxes_xyxy, device):
    if len(boxes_xyxy) == 0:
        return []
    sam_predictor.set_image(image_np)
    boxes_xyxy = boxes_xyxy.to(device)
    transformed = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=False,
    )
    return [masks[i, 0].bool() for i in range(masks.shape[0])]


def make_collage(rows, out_path, headers):
    cell_w, cell_h = rows[0][0].size
    header_h = 28
    margin = 8
    cols = len(rows[0])
    width = cols * cell_w + (cols + 1) * margin
    height = header_h + len(rows) * cell_h + (len(rows) + 1) * margin
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for c, h in enumerate(headers):
        x = margin + c * (cell_w + margin)
        draw.text((x, 8), h, fill=(0, 0, 0), font=font)
    for r, row in enumerate(rows):
        y = header_h + margin + r * (cell_h + margin)
        for c, im in enumerate(row):
            x = margin + c * (cell_w + margin)
            canvas.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def resize_cell(im, width=360):
    h = int(im.height * width / im.width)
    return im.resize((width, h), Image.BICUBIC)


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--queries", nargs="+", default=["yellow rubber duck", "red apple", "green apple", "camera", "chairs"])
    parser.add_argument("--num_views", default=5, type=int)
    parser.add_argument("--output_dir", default="output/lerf/figurines/query_diagnostics")
    parser.add_argument("--box_threshold", default=0.25, type=float)
    parser.add_argument("--text_threshold", default=0.25, type=float)
    parser.add_argument("--ioa_threshold", default=0.20, type=float)
    args = get_combined_args(parser)
    args.num_classes = 256
    args.train_split = True

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    classifier = torch.nn.Conv2d(gaussians.num_objects, args.num_classes, kernel_size=1).cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "classifier.pth")))
    classifier.eval()

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    sam_checkpoint = "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).cuda()
    sam_predictor = SamPredictor(sam)

    gdino = load_model_hf(
        "ShilongLiu/GroundingDINO",
        "groundingdino_swinb_cogcoor.pth",
        "GroundingDINO_SwinB.cfg.py",
        device="cuda",
    ).cuda()

    views = sample_reason_cameras_cluster_then_topk(scene.getTrainCameras(), gaussians, pipe, background, classifier)
    views = views[: args.num_views]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_originals = []
    rendered_cache = []
    with torch.no_grad():
        for idx, view in enumerate(views):
            pkg = render(view, gaussians, pipe, background)
            image = pkg["render"]
            pred_obj = torch.argmax(classifier(pkg["render_object"]), dim=0)
            image_np = tensor_to_np_image(image)
            rendered_cache.append((image_np, pred_obj))
            all_originals.append(resize_cell(Image.fromarray(image_np).convert("RGB")))
            Image.fromarray(image_np).save(out_root / f"view_{idx:02d}_original.png")

    make_collage([[im] for im in all_originals], out_root / "original_views.png", ["Original fixed views"])

    summary_lines = []
    for query in args.queries:
        slug = query.lower().replace(" ", "_").replace("/", "_")
        rows = []
        summary_lines.append(f"# {query}")
        for idx, (image_np, pred_obj) in enumerate(rendered_cache):
            boxes, logits, phrases = run_grounding_dino(gdino, image_np, query, args.box_threshold, args.text_threshold)
            masks = run_sam(sam_predictor, image_np, boxes, device="cuda")
            box_im = draw_boxes(image_np, boxes.cpu().numpy(), phrases, logits.detach().cpu().numpy() if torch.is_tensor(logits) else None)
            sam_im = masks_to_overlay(image_np, masks)
            realm_im, selected_info = realm_overlay(image_np, pred_obj, masks, args.ioa_threshold)
            summary_lines.append(f"view {idx}: boxes={len(boxes)} selected={selected_info}")
            rows.append([
                resize_cell(Image.fromarray(image_np).convert("RGB")),
                resize_cell(box_im),
                resize_cell(sam_im),
                resize_cell(realm_im),
            ])
        make_collage(rows, out_root / f"{slug}_diagnostic_collage.png", ["Original", "GroundingDINO boxes", "SAM masks", "REALM object IDs"])
        summary_lines.append("")

    (out_root / "diagnostic_summary.txt").write_text("\n".join(summary_lines))
    print(f"Saved diagnostics to {out_root}")


if __name__ == "__main__":
    main()
