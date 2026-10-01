#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
REALM = Path("/home/gvilaplana/GS-Thesis/REALM-Code")
SCRIPT_DIR = BASE / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from train_handal_17cat_mlp_baselines import normalize_xyz, sigmoid_np  # noqa: E402
from train_handal_17cat_pointnet_global import PointNetGlobal  # noqa: E402


C0 = 0.28209479177387814
MODEL_PATH = (
    BASE
    / "outputs"
    / "03_handle_generalization"
    / "12_17cat_stage9_gaussian_native_pointnet_v1"
    / "01_seen_instance_17cat"
    / "pointnet_xyzrgb"
    / "model.pt"
)
IN_ROOT = REALM / "output" / "lerf" / "requested_object_pruning_plys"
OUT_ROOT = (
    BASE
    / "outputs"
    / "05_realm_pruned_handle_predictions"
    / "01_pointnet_xyzrgb_maxmean_seen17"
)

REQUESTS = [
    {
        "name": "water_pitcher",
        "ply": IN_ROOT / "counter" / "water_pitcher_iou050_multi_id_3dgs.ply",
    },
    {
        "name": "ice_cream",
        "ply": IN_ROOT / "figurines" / "ice_cream_iou050_multi_id_3dgs.ply",
    },
    {
        "name": "chopsticks",
        "ply": IN_ROOT / "ramen" / "chopsticks_iou050_multi_id_3dgs.ply",
    },
]

PUMPKIN_PLY = IN_ROOT / "figurines" / "pumpkin_iou050_multi_id_3dgs.ply"
PUMPKIN_CLASSIFIER = (
    REALM
    / "output"
    / "lerf"
    / "figurines_rgball_objclean145"
    / "point_cloud"
    / "iteration_30000"
    / "classifier.pth"
)
PUMPKIN_REALM_IDS = [122, 36]


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def base_color(vertices):
    names = vertices.dtype.names
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        color = np.vstack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).T
        return np.clip(color.astype(np.float32) * C0 + 0.5, 0.0, 1.0)
    if {"red", "green", "blue"}.issubset(names):
        return np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0
    return np.full((len(vertices), 3), 0.5, dtype=np.float32)


def score_heatmap(scores):
    scores = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
    blue = np.array([0.02, 0.16, 1.0], dtype=np.float32)
    cyan = np.array([0.0, 0.85, 1.0], dtype=np.float32)
    yellow = np.array([1.0, 0.92, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    rgb = np.empty((len(scores), 3), dtype=np.float32)

    low = scores < 0.5
    mid = (scores >= 0.5) & (scores < 0.75)
    high = scores >= 0.75
    if low.any():
        t = (scores[low] / 0.5)[:, None]
        rgb[low] = blue * (1.0 - t) + cyan * t
    if mid.any():
        t = ((scores[mid] - 0.5) / 0.25)[:, None]
        rgb[mid] = cyan * (1.0 - t) + yellow * t
    if high.any():
        t = ((scores[high] - 0.75) / 0.25)[:, None]
        rgb[high] = yellow * (1.0 - t) + red * t
    return np.clip(rgb, 0.0, 1.0)


def red_blue(scores, threshold):
    labels = np.asarray(scores) >= threshold
    rgb = np.full((len(scores), 3), np.array([0.02, 0.16, 1.0], dtype=np.float32))
    rgb[labels] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    return rgb


def write_colored_ply(ply, vertices, out_path, rgb):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_vertices = np.array(vertices, copy=True)
    sh = rgb_to_sh(rgb)
    out_vertices["f_dc_0"] = sh[:, 0]
    out_vertices["f_dc_1"] = sh[:, 1]
    out_vertices["f_dc_2"] = sh[:, 2]
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(out_vertices, "vertex"))
        else:
            elements.append(element)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(out_path))


def load_model(device):
    pack = torch.load(MODEL_PATH, map_location=device)
    model = PointNetGlobal(int(pack["input_dim"]), latent_dim=int(pack["latent_dim"])).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    return model, np.asarray(pack["mean"], dtype=np.float32), np.asarray(pack["std"], dtype=np.float32), float(pack["threshold"]), pack


@torch.no_grad()
def predict_scores(model, vertices, mean, std, device):
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    color = base_color(vertices).astype(np.float32)
    x = np.concatenate([normalize_xyz(xyz), color], axis=1)
    x = ((x - mean) / std).astype(np.float32)
    logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    return sigmoid_np(logits).astype(np.float32)


def predict_realm_ids(vertices, classifier_path, num_classes=256, device="cpu"):
    names = vertices.dtype.names
    obj_names = [f"obj_dc_{i}" for i in range(16)]
    if not set(obj_names).issubset(names):
        raise ValueError("Pumpkin split requires obj_dc_0..obj_dc_15 fields in the PLY.")
    classifier = torch.nn.Conv2d(16, num_classes, kernel_size=1).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    obj = np.vstack([vertices[name] for name in obj_names]).T.astype(np.float32)
    with torch.no_grad():
        logits = classifier(torch.from_numpy(obj.T[:, None, :]).to(device)).squeeze(1)
        return torch.argmax(logits, dim=0).detach().cpu().numpy().astype(np.int32)


def process_object(name, source_ply, vertex_mask, model, mean, std, threshold, device, rows):
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if vertex_mask is not None:
        vertices = vertices[np.asarray(vertex_mask, dtype=bool)]
    if len(vertices) == 0:
        rows.append({"object": name, "status": "empty", "source_ply": str(source_ply)})
        return

    scores = predict_scores(model, vertices, mean, std, device)
    heatmap_ply = OUT_ROOT / "score_heatmap_blue0_red1" / f"{name}_handle_score_heatmap.ply"
    threshold_ply = OUT_ROOT / "threshold_red_blue" / f"{name}_handle_pred_thr{threshold:.2f}.ply"
    write_colored_ply(ply, vertices, heatmap_ply, score_heatmap(scores))
    write_colored_ply(ply, vertices, threshold_ply, red_blue(scores, threshold))

    rows.append(
        {
            "object": name,
            "status": "ok",
            "source_ply": str(source_ply),
            "num_gaussians": int(len(vertices)),
            "threshold": float(threshold),
            "pred_positive_ratio": float((scores >= threshold).mean()),
            "score_min": float(scores.min()),
            "score_mean": float(scores.mean()),
            "score_max": float(scores.max()),
            "heatmap_ply": str(heatmap_ply),
            "threshold_red_blue_ply": str(threshold_ply),
        }
    )
    print(f"[ok] {name}: N={len(vertices)} pred_ratio={rows[-1]['pred_positive_ratio']:.3f}")


def write_manifest(rows, model_pack):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "prediction_manifest.csv"
    keys = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT_ROOT / "prediction_manifest.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "model_path": str(MODEL_PATH),
                "feature_variant": model_pack.get("feature_variant"),
                "architecture": model_pack.get("overall", {}).get("architecture"),
                "threshold": float(model_pack["threshold"]),
                "objects": rows,
            },
            f,
            indent=2,
        )
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, threshold, model_pack = load_model(device)
    rows = []
    for request in REQUESTS:
        process_object(request["name"], request["ply"], None, model, mean, std, threshold, device, rows)

    pumpkin_ply = PlyData.read(str(PUMPKIN_PLY))
    pumpkin_vertices = np.array(pumpkin_ply["vertex"].data, copy=True)
    realm_ids = predict_realm_ids(pumpkin_vertices, PUMPKIN_CLASSIFIER, device=device)
    for idx, realm_id in enumerate(PUMPKIN_REALM_IDS, start=1):
        process_object(
            f"pumpkin_{idx}_realm_id{realm_id}",
            PUMPKIN_PLY,
            realm_ids == realm_id,
            model,
            mean,
            std,
            threshold,
            device,
            rows,
        )

    write_manifest(rows, model_pack)


if __name__ == "__main__":
    main()
