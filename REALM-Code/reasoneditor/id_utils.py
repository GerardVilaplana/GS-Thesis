import ast
import json

import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError


def _strip_markdown_json(text):
    text = str(text).strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def _parse_boxes(reason_output):
    text = _strip_markdown_json(reason_output)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                data = ast.literal_eval(text[start : end + 1])
            else:
                raise
    if isinstance(data, dict):
        data = [data]

    boxes = []
    labels = []
    for item in data:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        box = item["bbox_2d"]
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        boxes.append([x1, y1, x2, y2])
        labels.append(item.get("label", "object"))
    return boxes, labels


def extract_selected_obj_ids(
    reason_output,
    image_tensor,
    pred_obj,
    sam_predictor,
    input_width,
    input_height,
    clip_model=None,
    clip_preprocess=None,
    ioa_thresh=0.20,
):
    """Map a VLM bbox answer to REALM object IDs.

    REALM's public repo imports this from a missing reasoneditor package. This
    replacement keeps the core behavior needed for segmentation: parse bbox_2d,
    run SAM inside those boxes, then select object IDs whose rendered object mask
    overlaps the SAM mask.
    """
    device = pred_obj.device
    boxes, _ = _parse_boxes(reason_output)
    h, w = pred_obj.shape[-2], pred_obj.shape[-1]

    if not boxes:
        empty_ids = torch.empty(0, dtype=torch.long, device=device)
        empty_boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
        empty_mask = torch.zeros((1, h, w), dtype=torch.bool, device=device)
        return empty_ids, empty_boxes, empty_mask, None

    boxes_t = torch.tensor(boxes, dtype=torch.float32, device=device)
    boxes_t[:, [0, 2]] = boxes_t[:, [0, 2]].clamp(0, w - 1)
    boxes_t[:, [1, 3]] = boxes_t[:, [1, 3]].clamp(0, h - 1)

    image_np = (image_tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    sam_predictor.set_image(image_np)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_t, image_np.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    combined_mask = masks[:, 0].any(dim=0)

    ids = torch.unique(pred_obj[combined_mask])
    selected = []
    scores = []
    for obj_id in ids:
        obj_mask = pred_obj == obj_id
        obj_area = obj_mask.sum().float()
        if obj_area.item() == 0:
            continue
        inter = (obj_mask & combined_mask).sum().float()
        ioa = inter / obj_area
        # Keep clear contained components, but always allow the strongest ID below.
        if ioa.item() >= ioa_thresh:
            selected.append(obj_id)
            scores.append(ioa)

    if not selected and ids.numel() > 0:
        best_id = None
        best_score = None
        for obj_id in ids:
            obj_mask = pred_obj == obj_id
            obj_area = obj_mask.sum().float()
            if obj_area.item() == 0:
                continue
            score = (obj_mask & combined_mask).sum().float() / obj_area
            if best_score is None or score > best_score:
                best_id = obj_id
                best_score = score
        if best_id is not None:
            selected = [best_id]
            scores = [best_score]

    if selected:
        selected_ids = torch.stack(selected).long().to(device)
        prob = torch.stack(scores).max().item()
    else:
        selected_ids = torch.empty(0, dtype=torch.long, device=device)
        prob = None

    return selected_ids, boxes_t, combined_mask.unsqueeze(0), prob


def points_inside_convex_hull(points, mask, outlier_factor=1.0):
    """Return points inside the convex hull of currently selected 3D points."""
    device = points.device
    pts = points.detach().cpu().numpy()
    mask_np = mask.detach().bool().cpu().numpy().reshape(-1)
    selected = pts[mask_np]

    if selected.shape[0] < 4:
        return torch.zeros(points.shape[0], dtype=torch.bool, device=device)

    if outlier_factor is not None and outlier_factor < 1.0:
        center = selected.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(selected - center, axis=1)
        keep = dist <= np.quantile(dist, outlier_factor)
        selected = selected[keep]
        if selected.shape[0] < 4:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=device)

    try:
        hull = Delaunay(selected)
        inside = hull.find_simplex(pts) >= 0
    except (QhullError, ValueError):
        inside = mask_np

    return torch.from_numpy(inside).to(device=device, dtype=torch.bool)
