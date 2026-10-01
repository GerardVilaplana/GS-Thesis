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
OUT_ROOT = (
    BASE
    / "outputs"
    / "05_realm_pruned_handle_predictions"
    / "02_pointnet_xyzrgb_maxmean_seen17_object_only"
)

OBJECTS = [
    {
        "name": "water_pitcher",
        "scene": "counter",
        "point_cloud": REALM
        / "output/lerf/counter_rgball_objclean_images2_fast_objpred/point_cloud/iteration_7000/point_cloud.ply",
        "classifier": REALM
        / "output/lerf/counter_rgball_objclean_images2_fast_objpred/point_cloud/iteration_7000/classifier.pth",
        "selected_ids": [39],
    },
    {
        "name": "ice_cream",
        "scene": "figurines",
        "point_cloud": REALM
        / "output/lerf/figurines_rgball_objclean145/point_cloud/iteration_30000/point_cloud.ply",
        "classifier": REALM
        / "output/lerf/figurines_rgball_objclean145/point_cloud/iteration_30000/classifier.pth",
        "selected_ids": [55],
    },
    {
        "name": "pumpkin_1_realm_id122",
        "scene": "figurines",
        "point_cloud": REALM
        / "output/lerf/figurines_rgball_objclean145/point_cloud/iteration_30000/point_cloud.ply",
        "classifier": REALM
        / "output/lerf/figurines_rgball_objclean145/point_cloud/iteration_30000/classifier.pth",
        "selected_ids": [122],
    },
    {
        "name": "pumpkin_2_realm_id36",
        "scene": "figurines",
        "point_cloud": REALM
        / "output/lerf/figurines_rgball_objclean145/point_cloud/iteration_30000/point_cloud.ply",
        "classifier": REALM
        / "output/lerf/figurines_rgball_objclean145/point_cloud/iteration_30000/classifier.pth",
        "selected_ids": [36],
    },
    {
        "name": "chopsticks",
        "scene": "ramen",
        "point_cloud": REALM
        / "output/lerf/ramen_rgball_objclean/point_cloud/iteration_30000/point_cloud.ply",
        "classifier": REALM
        / "output/lerf/ramen_rgball_objclean/point_cloud/iteration_30000/classifier.pth",
        "selected_ids": [119],
    },
    {
        "name": "tea_cup_realm_id103",
        "scene": "teatime",
        "point_cloud": REALM
        / "output/lerf/teatime_rgball_objclean/point_cloud/iteration_30000/point_cloud.ply",
        "classifier": REALM
        / "output/lerf/teatime_rgball_objclean/point_cloud/iteration_30000/classifier.pth",
        "selected_ids": [103],
    },
    {
        "name": "tea_cup_realm_id051",
        "scene": "teatime",
        "point_cloud": REALM
        / "output/lerf/teatime_rgball_objclean/point_cloud/iteration_30000/point_cloud.ply",
        "classifier": REALM
        / "output/lerf/teatime_rgball_objclean/point_cloud/iteration_30000/classifier.pth",
        "selected_ids": [51],
    },
]


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


def binary_red_blue(mask):
    mask = np.asarray(mask, dtype=bool)
    rgb = np.zeros((len(mask), 3), dtype=np.float32)
    rgb[~mask] = np.array([0.02, 0.16, 1.0], dtype=np.float32)
    rgb[mask] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    return rgb


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


def write_vertices_like(ply, vertices, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertices, "vertex"))
        else:
            elements.append(element)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(output_path))


def write_colored_ply(ply, vertices, output_path, rgb):
    out_vertices = np.array(vertices, copy=True)
    sh = rgb_to_sh(rgb)
    out_vertices["f_dc_0"] = sh[:, 0]
    out_vertices["f_dc_1"] = sh[:, 1]
    out_vertices["f_dc_2"] = sh[:, 2]
    write_vertices_like(ply, out_vertices, output_path)


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


@torch.no_grad()
def predict_realm_ids(vertices, classifier_path, num_classes=256, device="cpu"):
    obj_names = [f"obj_dc_{i}" for i in range(16)]
    if not set(obj_names).issubset(vertices.dtype.names):
        raise ValueError(f"{classifier_path}: point cloud has no obj_dc_0..obj_dc_15 fields")
    classifier = torch.nn.Conv2d(16, num_classes, kernel_size=1).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    obj = np.vstack([vertices[name] for name in obj_names]).T.astype(np.float32)
    logits = classifier(torch.from_numpy(obj.T[:, None, :]).to(device)).squeeze(1)
    return torch.argmax(logits, dim=0).detach().cpu().numpy().astype(np.int32)


def process_object(spec, cached_scene, model, mean, std, threshold, device):
    key = (str(spec["point_cloud"]), str(spec["classifier"]))
    if key not in cached_scene:
        ply = PlyData.read(str(spec["point_cloud"]))
        vertices = np.array(ply["vertex"].data, copy=True)
        realm_ids = predict_realm_ids(vertices, spec["classifier"], device=device)
        cached_scene[key] = (ply, vertices, realm_ids)
    ply, vertices, realm_ids = cached_scene[key]

    mask = np.isin(realm_ids, np.asarray(spec["selected_ids"], dtype=np.int32))
    object_vertices = np.array(vertices[mask], copy=True)
    if len(object_vertices) == 0:
        return {
            "object": spec["name"],
            "scene": spec["scene"],
            "status": "empty",
            "selected_ids": ",".join(map(str, spec["selected_ids"])),
            "source_point_cloud": str(spec["point_cloud"]),
        }

    scores = predict_scores(model, object_vertices, mean, std, device)
    true_ply = OUT_ROOT / "object_true_color" / f"{spec['name']}_object_true_color.ply"
    heatmap_ply = OUT_ROOT / "handle_score_heatmap_blue0_red1" / f"{spec['name']}_handle_score_heatmap.ply"
    binary_ply = OUT_ROOT / "threshold_red_blue" / f"{spec['name']}_handle_pred_thr{threshold:.2f}.ply"
    write_vertices_like(ply, object_vertices, true_ply)
    write_colored_ply(ply, object_vertices, heatmap_ply, score_heatmap(scores))
    write_colored_ply(ply, object_vertices, binary_ply, binary_red_blue(scores >= threshold))

    return {
        "object": spec["name"],
        "scene": spec["scene"],
        "status": "ok",
        "selected_ids": ",".join(map(str, spec["selected_ids"])),
        "source_point_cloud": str(spec["point_cloud"]),
        "classifier": str(spec["classifier"]),
        "num_scene_gaussians": int(len(vertices)),
        "num_object_gaussians": int(len(object_vertices)),
        "object_keep_ratio": float(len(object_vertices) / max(len(vertices), 1)),
        "threshold": float(threshold),
        "pred_positive_ratio": float((scores >= threshold).mean()),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "true_color_ply": str(true_ply),
        "heatmap_ply": str(heatmap_ply),
        "binary_ply": str(binary_ply),
    }


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
    return csv_path, json_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, threshold, model_pack = load_model(device)
    cached_scene = {}
    rows = []
    for spec in OBJECTS:
        row = process_object(spec, cached_scene, model, mean, std, threshold, device)
        rows.append(row)
        print(
            f"[{row['status']}] {row['object']}: "
            f"N={row.get('num_object_gaussians', 0)} "
            f"keep={row.get('object_keep_ratio', 0.0):.4f} "
            f"pred_ratio={row.get('pred_positive_ratio', 0.0):.4f}",
            flush=True,
        )
    csv_path, json_path = write_manifest(rows, model_pack)
    print(f"[done] wrote {csv_path}", flush=True)
    print(f"[done] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
