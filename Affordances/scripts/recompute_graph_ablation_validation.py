#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import train_exp48_gnn_embeddings_pointnet_mean as exp48  # noqa: E402
import train_exp50_gnn_variations as exp50  # noqa: E402
import train_handal_exp44_pointnet_mean_feature_sweep as exp44  # noqa: E402


NPZ_ROOT = (
    BASE_ROOT
    / "data/handal_exp40_15k_gt_features_v1/npz/B_center_ellipsoid20_strict75"
)
STANDARD_ROOT = (
    BASE_ROOT
    / "outputs/03_handle_generalization/48_exp46_dino_gnn_embeddings_pointnet_mean_v1"
)
VARIATION_ROOT = (
    BASE_ROOT
    / "outputs/03_handle_generalization/50_exp46_gnn_variations_safe_v1"
)
OUT_DIR = BASE_ROOT / "outputs/03_handle_generalization/graph_ablation_validation"


VARIANTS = {
    "standard_graph_attention": {
        "module": exp48,
        "root": STANDARD_ROOT,
        "subdir": None,
        "gnn_variant": "standard",
    },
    "edge_aware_graph_attention": {
        "module": exp50,
        "root": VARIATION_ROOT,
        "subdir": "edge_aware_xyz_delta",
        "gnn_variant": "edge_aware_xyz_delta",
    },
    "multiscale_graph_attention": {
        "module": exp50,
        "root": VARIATION_ROOT,
        "subdir": "multiscale_k8_16_32_64",
        "gnn_variant": "multiscale_k8_16_32_64",
    },
}


def model_dir(spec: dict, mode: str, category: str | None = None) -> Path:
    root = Path(spec["root"])
    if spec["subdir"]:
        root = root / spec["subdir"]
    root = root / "object_crop_best"
    if mode == "seen":
        return root / "01_seen_instance_17cat/object_crop_best/gnn_embeddings_pointnet_mean"
    assert category is not None
    return (
        root
        / "02_leave_one_category_out_17cat"
        / category
        / "object_crop_best/gnn_embeddings_pointnet_mean"
    )


def load_checkpoints(path: Path) -> tuple[dict, dict]:
    return (
        torch.load(path / "gnn_model.pt", map_location="cpu"),
        torch.load(path / "pointnet_model.pt", map_location="cpu"),
    )


def evaluate(spec: dict, items: list[dict], path: Path, device: torch.device) -> tuple[list[dict], dict]:
    module = spec["module"]
    gnn_ckpt, pointnet_ckpt = load_checkpoints(path)
    cfg = dict(gnn_ckpt["config"])
    cfg["dino_root"] = str(cfg["dino_root"])
    k_layers = tuple(int(k) for k in gnn_ckpt["k_layers"])

    graph_scenes = module.load_graph_scenes(
        items,
        cfg,
        np.asarray(gnn_ckpt["mean"]),
        np.asarray(gnn_ckpt["std"]),
        max(k_layers),
    )
    in_dim = graph_scenes[0]["x"].shape[1]
    if module is exp48:
        gnn = module.GraphEmbeddingNet(
            in_dim,
            int(gnn_ckpt["latent_dim"]),
            k_layers,
            0.1,
        )
    else:
        args = SimpleNamespace(
            gnn_variant=spec["gnn_variant"],
            gnn_latent_dim=int(gnn_ckpt["latent_dim"]),
            k_layers=list(k_layers),
            dropout=0.1,
        )
        gnn = module.make_gnn_model(in_dim, args)
    gnn.load_state_dict(gnn_ckpt["model"])
    gnn.to(device)

    embedding_scenes = module.extract_embedding_scenes(gnn, graph_scenes, device)
    embedding_scenes = module.apply_scene_standardizer(
        embedding_scenes,
        np.asarray(pointnet_ckpt["embedding_mean"]),
        np.asarray(pointnet_ckpt["embedding_std"]),
    )
    pointnet = module.PointNetMean(
        embedding_scenes[0]["x"].shape[1],
        int(pointnet_ckpt["latent_dim"]),
        0.1,
    )
    pointnet.load_state_dict(pointnet_ckpt["model"])
    pointnet.to(device)

    labels, scores, slices = module.predict_pointnet(pointnet, embedding_scenes, device)
    rows, _, macro = module.macro_rows(
        labels,
        scores,
        slices,
        float(pointnet_ckpt["threshold"]),
    )
    return rows, macro


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    items = exp44.load_items(NPZ_ROOT)
    categories = sorted({item["category"] for item in items})
    summary = []
    all_scene_rows = []

    for name, spec in VARIANTS.items():
        seen_items = [item for item in items if item["source_split"] == "val"]
        seen_rows, seen_macro = evaluate(spec, seen_items, model_dir(spec, "seen"), device)
        for row in seen_rows:
            all_scene_rows.append({"variant": name, "setting": "seen", **row})

        loco_category_rows = []
        for category in categories:
            eval_items = [
                item
                for item in items
                if item["category"] == category and item["source_split"] == "val"
            ]
            rows, macro = evaluate(
                spec,
                eval_items,
                model_dir(spec, "loco", category),
                device,
            )
            loco_category_rows.append(
                {
                    "variant": name,
                    "category": category,
                    "num_scenes": len(eval_items),
                    "iou": float(macro["iou"]),
                    "f1": float(macro["f1"]),
                }
            )
            for row in rows:
                all_scene_rows.append(
                    {"variant": name, "setting": "loco", "heldout_category": category, **row}
                )

        summary.append(
            {
                "variant": name,
                "seen_val_miou": float(seen_macro["iou"]),
                "seen_val_f1": float(seen_macro["f1"]),
                "loco_val_miou": float(np.mean([row["iou"] for row in loco_category_rows])),
                "loco_val_f1": float(np.mean([row["f1"] for row in loco_category_rows])),
            }
        )
        write_csv(args.out_dir / f"{name}_loco_per_category.csv", loco_category_rows)
        print(json.dumps(summary[-1], indent=2), flush=True)

    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "per_scene_metrics.csv", all_scene_rows)
    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
