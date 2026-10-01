#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

BASE = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import build_handal_generalization_feature_pilot as pilot  # noqa: E402
import extract_exp45_dino_assignment_visual as d45  # noqa: E402
import train_exp48_gnn_embeddings_pointnet_mean as e48  # noqa: E402
import train_handal_17cat_graph_attention as g15  # noqa: E402
import train_handal_exp44_pointnet_mean_feature_sweep as e44  # noqa: E402
from evaluate_exp38_reprojection_strategy_ablation import project_points  # noqa: E402
from extract_handal_dinov2_gaussian_embeddings import load_dinov2, normalize_rows  # noqa: E402

C0 = 0.28209479177387814
SCENE = BASE / "data/own_scenes/scene_1_realm_1600"
MODEL = Path("/home/gvilaplana/GS-Thesis/REALM-Code/output/own_scenes/scene_1_realm_1600_objpred_15k")
OBJECT_ROOT = BASE / "outputs/06_own_scenes/scene_1_15k_realm_query_pruning"
OUT_ROOT = BASE / "outputs/06_own_scenes/scene_1_15k_best_dino_handle_predictions"
PER_ID_OUT_ROOT = BASE / "outputs/06_own_scenes/scene_1_15k_per_id_best_dino_handle_predictions"
PER_ID_CLEAN_OUT_ROOT = BASE / "outputs/06_own_scenes/scene_1_15k_per_id_reproj_best_dino_handle_predictions"
PN_CKPT = (
    BASE
    / "outputs/03_handle_generalization/47_exp46_dino_pointnet_mean_feature_sweep_v1/object_crop_center_avg"
    / "01_seen_instance_17cat/dino_geometry_color_quality/pointnet_mean/model.pt"
)
GNN_CKPT = (
    BASE
    / "outputs/03_handle_generalization/48_exp46_dino_gnn_embeddings_pointnet_mean_v1/object_crop_best"
    / "01_seen_instance_17cat/object_crop_best/gnn_embeddings_pointnet_mean/gnn_model.pt"
)
SCENE_PREFIX = "scene1"


@dataclass(frozen=True)
class ObjectSpec:
    query: str
    slug: str
    category: str
    source_ply: str = ""
    class_id: int = -1
    query_aliases: str = ""


OBJECTS = [
    ObjectSpec("hammer", "hammer", "hammers"),
    ObjectSpec("screwdriver", "screwdriver", "screwdrivers"),
    ObjectSpec("joint plier", "joint_plier", "slip_joint_pliers"),
    ObjectSpec("cutting plier", "cutting_plier", "fixed_joint_pliers"),
]


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def color_heatmap(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    blue = np.array([0.02, 0.16, 1.0], dtype=np.float32)
    red = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    return blue * (1.0 - scores) + red * scores


def color_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    rgb = np.full((len(scores), 3), np.array([0.02, 0.16, 1.0], dtype=np.float32))
    rgb[np.asarray(scores) >= threshold] = np.array([0.02, 0.82, 0.12], dtype=np.float32)
    return rgb


def write_colored_ply(src: Path, out: Path, rgb: np.ndarray) -> None:
    ply = PlyData.read(str(src))
    vertex = np.array(ply["vertex"].data, copy=True)
    if len(vertex) != len(rgb):
        raise ValueError(f"length mismatch for {src}: {len(vertex)} vs {len(rgb)}")
    sh = rgb_to_sh(rgb)
    vertex["f_dc_0"] = sh[:, 0]
    vertex["f_dc_1"] = sh[:, 1]
    vertex["f_dc_2"] = sh[:, 2]
    elems = [PlyElement.describe(vertex, "vertex") if e.name == "vertex" else e for e in ply.elements]
    out.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elems, text=ply.text, byte_order=ply.byte_order).write(str(out))


def object_ply(spec: ObjectSpec) -> Path:
    if spec.source_ply:
        p = Path(spec.source_ply)
        return p if p.is_absolute() else BASE / p
    return OBJECT_ROOT / "final_ply" / f"{spec.slug}_object_truecolor_3dgs.ply"


def prepare_npz(spec: ObjectSpec, force: bool) -> dict:
    out = OUT_ROOT / "npz" / f"{SCENE_PREFIX}_{spec.slug}.npz"
    ply_path = object_ply(spec)
    if out.exists() and not force:
        with np.load(out, allow_pickle=False) as z:
            return {"scene_key": out.stem, "category": spec.category, "npz": str(out), "source_object_ply": str(ply_path), "gaussians": int(z["xyz"].shape[0]), "reused_npz": True}
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    vertices = PlyData.read(str(ply_path))["vertex"].data
    xyz, scale, rotation, opacity, color, geometry = pilot.geometry_arrays(vertices)
    dummy_y = np.zeros(len(xyz), dtype=np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        geometry_features=geometry.astype(np.float32),
        xyz=xyz.astype(np.float32),
        scale=scale.astype(np.float32),
        rotation=rotation.astype(np.float32),
        opacity=opacity.astype(np.float32),
        color=color.astype(np.float32),
        handle_scores=np.zeros(len(xyz), dtype=np.float32),
        handle_labels_thr0_25=dummy_y,
        handle_visible_views=np.zeros(len(xyz), dtype=np.uint16),
        scene_key=np.array([out.stem]),
        scene_id=np.array([SCENE_PREFIX]),
        category=np.array([spec.category]),
        model_split=np.array(["own_scene"]),
        raw_split=np.array(["own_scene"]),
        instance_id=np.array([spec.slug]),
        source_scene_path=np.array([str(SCENE)]),
        source_object_ply=np.array([str(ply_path)]),
        realm_class_id=np.array([spec.class_id], dtype=np.int32),
        query_aliases=np.array([spec.query_aliases or spec.query]),
        variant=np.array([f"own_{SCENE_PREFIX}_15k_realm_annotation_mask_query_pruned"]),
    )
    return {"scene_key": out.stem, "category": spec.category, "npz": str(out), "source_object_ply": str(ply_path), "gaussians": int(len(xyz)), "reused_npz": False}


def collect_specs(input_mode: str) -> list[ObjectSpec]:
    if input_mode == "query_merged":
        return OBJECTS
    by_id: dict[int, dict] = {}
    if input_mode == "per_id_clean":
        summary_path = OBJECT_ROOT / "final_ply/per_id_reprojection_ellipsoid20_strict75/per_id_reprojection_summary.json"
        summary = json.loads(summary_path.read_text())
        for class_id_text, block in summary.get("objects", {}).items():
            class_id = int(class_id_text)
            queries = [q for q in str(block.get("queries", "")).split(";") if q]
            by_id[class_id] = {"queries": queries, "path": block["clean_ply"]}
    else:
        summary_path = OBJECT_ROOT / "final_ply/per_id_truecolor/per_id_truecolor_summary.json"
        summary = json.loads(summary_path.read_text())
        for query, block in summary.items():
            for part in block.get("parts", []):
                class_id = int(part["class_id"])
                rec = by_id.setdefault(class_id, {"queries": [], "path": part["path"]})
                rec["queries"].append(query)
    specs = []
    for class_id, rec in sorted(by_id.items()):
        aliases = sorted(set(rec["queries"]))
        if aliases == ["hammer"]:
            category = "hammers"
        elif aliases == ["screwdriver"]:
            category = "screwdrivers"
        elif aliases == ["mug"]:
            category = "mugs"
        elif any("spatula" in q for q in aliases):
            category = "spatulas"
        elif any(q in {"fork", "spoon", "kitchen knife", "table knife"} for q in aliases):
            category = "utensils"
        elif any("plier" in q for q in aliases):
            category = "pliers"
        else:
            category = "own_scene_object"
        specs.append(ObjectSpec(query="/".join(aliases), slug=f"realm_id{class_id:03d}", category=category, source_ply=str(rec["path"]), class_id=class_id, query_aliases=";".join(aliases)))
    return specs


def load_image_for_camera(cam: dict) -> Image.Image:
    img_name = str(cam["img_name"])
    for suffix in (".jpg", ".jpeg", ".png"):
        p = SCENE / "images" / f"{img_name}{suffix}"
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(img_name)


@torch.no_grad()
def extract_dino(item: dict, args: argparse.Namespace, model, patch_size: int, hidden_size: int, device) -> dict:
    scene_key = item["scene_key"]
    out = OUT_ROOT / "dino/object_crop_center_avg_patch896x672_views48/features" / f"{scene_key}_object_crop_center_avg_features.npz"
    if out.exists() and not args.force:
        with np.load(out, allow_pickle=False) as z:
            return {"scene_key": scene_key, "valid_dino": int(z["valid_dino"].sum()), "valid_dino_ratio": float(z["valid_dino"].mean()), "features": str(out), "reused_dino": True}

    cameras = sorted(json.loads((MODEL / "cameras.json").read_text()), key=lambda c: int(c["id"]))
    cameras = cameras[: args.max_dino_views]
    ply = PlyData.read(item["source_object_ply"])
    vertices = ply["vertex"].data
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    feature_sum = np.zeros((len(points), hidden_size), dtype=np.float32)
    weight_sum = np.zeros(len(points), dtype=np.float32)
    visible_views = np.zeros(len(points), dtype=np.uint16)
    assigned_centers = np.zeros(len(points), dtype=np.uint32)
    crop_rows = []

    for view_idx, cam in enumerate(cameras, start=1):
        width, height = int(cam["width"]), int(cam["height"])
        u, v, valid = project_points(points[:, None, :], cam, width, height)
        u = u[:, 0]
        v = v[:, 0]
        valid = valid[:, 0]
        crop = d45.object_crop_from_valid(u, v, valid, width, height, args.crop_margin_frac)
        if crop is None:
            continue
        image = load_image_for_camera(cam)
        patch_features = d45.extract_patch_features_pil(
            model,
            image.crop(crop),
            args.resize_width,
            args.resize_height,
            patch_size,
            device,
            args.normalize_patch_features,
        )
        before = weight_sum.copy()
        assigned = d45.add_center_patch_votes(
            feature_sum,
            weight_sum,
            visible_views,
            patch_features,
            valid,
            u,
            v,
            width,
            height,
            args.resize_width,
            args.resize_height,
            patch_size,
            crop,
        )
        assigned_centers[(weight_sum - before) > 0] += 1
        crop_rows.append({"scene_key": scene_key, "view": view_idx, "img_name": cam["img_name"], "x0": crop[0], "y0": crop[1], "x1": crop[2], "y1": crop[3], "assigned": assigned})
        if view_idx == 1 or view_idx == len(cameras) or view_idx % args.log_view_every == 0:
            print(f"{scene_key} dino view {view_idx:03d}/{len(cameras)} {cam['img_name']}: assigned={assigned}", flush=True)

    emb = np.divide(feature_sum, weight_sum[:, None], out=np.zeros_like(feature_sum), where=weight_sum[:, None] > 0)
    valid_dino = weight_sum >= args.min_total_weight
    if args.normalize_output_features and np.any(valid_dino):
        emb[valid_dino] = normalize_rows(emb[valid_dino]).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.dino_dtype == "float16" else np.float32
    np.savez_compressed(
        out,
        dino_features=emb.astype(dtype),
        valid_dino=valid_dino.astype(bool),
        dino_weight_sum=weight_sum.astype(np.float32),
        dino_visible_views=visible_views,
        center_assignments=assigned_centers,
        scene_key=np.array([scene_key]),
        category=np.array([item["category"]]),
        source_object_ply=np.array([item["source_object_ply"]]),
        source_scene_path=np.array([str(SCENE)]),
        method=np.array(["object_crop_center_avg"]),
        dino_model=np.array([args.model_name]),
        resize_width=np.array([args.resize_width], dtype=np.int32),
        resize_height=np.array([args.resize_height], dtype=np.int32),
        patch_size=np.array([patch_size], dtype=np.int32),
        views=np.array([len(cameras)], dtype=np.int32),
    )
    save_csv(OUT_ROOT / "dino/object_crop_center_avg_patch896x672_views48/crop_boxes" / f"{scene_key}_crop_boxes.csv", crop_rows)
    return {"scene_key": scene_key, "valid_dino": int(valid_dino.sum()), "valid_dino_ratio": float(valid_dino.mean()), "features": str(out), "reused_dino": False}


def item_features(item: dict, dino_root: Path) -> tuple[np.ndarray, np.ndarray]:
    e44.ACTIVE_FEATURE_VARIANT = "dino_geometry_color_quality"
    e44.ACTIVE_DINO_ROOT = dino_root
    return e44.load_feature_arrays_variant(item)


@torch.no_grad()
def predict_pn(item: dict, device) -> tuple[np.ndarray, float]:
    ck = torch.load(PN_CKPT, map_location="cpu")
    x, _ = item_features(item, OUT_ROOT / "dino/object_crop_center_avg_patch896x672_views48")
    x = ((x - ck["mean"].astype(np.float32)) / ck["std"].astype(np.float32)).astype(np.float32)
    model = e44.PointNet(int(ck["input_dim"]), "mean", int(ck["latent_dim"]), 0.0).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().astype(np.float32)
    return 1.0 / (1.0 + np.exp(-logits)), float(ck["threshold"])


@torch.no_grad()
def predict_gnn(item: dict, device) -> tuple[np.ndarray, float]:
    ck = torch.load(GNN_CKPT, map_location="cpu")
    x, _ = item_features(item, OUT_ROOT / "dino/object_crop_center_avg_patch896x672_views48")
    x = ((x - ck["mean"].astype(np.float32)) / ck["std"].astype(np.float32)).astype(np.float32)
    with np.load(item["npz"], allow_pickle=False) as z:
        xyz_norm = g15.base.normalize_xyz(z["xyz"].astype(np.float32))
    max_k = max(int(k) for k in ck.get("k_layers", [16, 32]))
    knn = g15.build_knn(xyz_norm, max_k)
    model = e48.GraphEmbeddingNet(int(ck["mean"].shape[1]), int(ck.get("latent_dim", 192)), tuple(ck.get("k_layers", [16, 32])), 0.0).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    scene = {"item": item, "x": x, "y": np.zeros(len(x), dtype=np.float32), "knn": knn}
    _, score, _ = e48.predict_gnn(model, [scene], device)
    return score.astype(np.float32), float(ck["overall"]["selected_threshold"])


def export_predictions(items: list[dict], device, model_names: list[str]) -> list[dict]:
    rows = []
    predictors = {
        "pointnet_mean_seen": predict_pn,
        "direct_gnn_seen": predict_gnn,
    }
    for item in items:
        for model_name in model_names:
            fn = predictors[model_name]
            scores, threshold = fn(item, device)
            ply = Path(item["source_object_ply"])
            prefix = f"{SCENE_PREFIX}_"
            stem = item["scene_key"][len(prefix):] if item["scene_key"].startswith(prefix) else item["scene_key"]
            heat = OUT_ROOT / "prediction_plys" / model_name / "probability_heatmap_blue0_red1" / f"{stem}_{model_name}_prob.ply"
            thr = OUT_ROOT / "prediction_plys" / model_name / "threshold_green_blue" / f"{stem}_{model_name}_thr{threshold:.2f}.ply"
            write_colored_ply(ply, heat, color_heatmap(scores))
            write_colored_ply(ply, thr, color_threshold(scores, threshold))
            rows.append(
                {
                    "scene_key": item["scene_key"],
                    "category": item["category"],
                    "model": model_name,
                    "threshold": threshold,
                    "num_gaussians": len(scores),
                    "pred_handle_ratio": float((scores >= threshold).mean()) if len(scores) else 0.0,
                    "score_mean": float(scores.mean()) if len(scores) else 0.0,
                    "score_max": float(scores.max()) if len(scores) else 0.0,
                    "source_object_ply": str(ply),
                    "probability_ply": str(heat),
                    "threshold_ply": str(thr),
                }
            )
    save_csv(OUT_ROOT / "prediction_manifest.csv", rows)
    (OUT_ROOT / "prediction_manifest.json").write_text(json.dumps(rows, indent=2))
    return rows


def main() -> None:
    global SCENE, MODEL, OBJECT_ROOT, OUT_ROOT, PER_ID_OUT_ROOT, PER_ID_CLEAN_OUT_ROOT, SCENE_PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "predict", "all"], default="all")
    parser.add_argument("--input_mode", choices=["per_id", "per_id_clean", "query_merged"], default="per_id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model_name", default="facebook/dinov2-small")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resize_width", type=int, default=896)
    parser.add_argument("--resize_height", type=int, default=672)
    parser.add_argument("--max_dino_views", type=int, default=48)
    parser.add_argument("--crop_margin_frac", type=float, default=0.20)
    parser.add_argument("--min_total_weight", type=float, default=1.0)
    parser.add_argument("--dino_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--normalize_patch_features", action="store_true", default=True)
    parser.add_argument("--normalize_output_features", action="store_true", default=True)
    parser.add_argument("--log_view_every", type=int, default=16)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--model_root", type=Path, default=MODEL)
    parser.add_argument("--object_root", type=Path, default=OBJECT_ROOT)
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument("--scene_prefix", default=SCENE_PREFIX)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["pointnet_mean_seen", "direct_gnn_seen"],
        default=["pointnet_mean_seen", "direct_gnn_seen"],
    )
    args = parser.parse_args()

    SCENE = args.scene
    MODEL = args.model_root
    OBJECT_ROOT = args.object_root
    SCENE_PREFIX = args.scene_prefix
    if args.out_root is not None:
        OUT_ROOT = args.out_root
        PER_ID_OUT_ROOT = args.out_root
        PER_ID_CLEAN_OUT_ROOT = args.out_root
    if args.input_mode == "per_id":
        OUT_ROOT = PER_ID_OUT_ROOT
    elif args.input_mode == "per_id_clean":
        OUT_ROOT = PER_ID_CLEAN_OUT_ROOT
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    specs = collect_specs(args.input_mode)
    items = [prepare_npz(spec, args.force) for spec in specs]

    if args.mode in {"prepare", "all"}:
        model, patch_size, hidden_size = load_dinov2(args.model_name, device, args.local_files_only)
        rows = []
        for item in items:
            rows.append({**item, **extract_dino(item, args, model, patch_size, hidden_size, device)})
        save_csv(OUT_ROOT / "prepare_summary.csv", rows)
        print(f"[prepare done] objects={len(rows)}", flush=True)

    if args.mode in {"predict", "all"}:
        rows = export_predictions(items, device, args.models)
        print(f"[predict done] outputs={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
