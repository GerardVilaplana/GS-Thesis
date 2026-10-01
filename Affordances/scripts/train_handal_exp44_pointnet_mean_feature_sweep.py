#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import train_handal_17cat_graph_attention as g15  # noqa: E402


DATA_ROOT = BASE_ROOT / "data" / "handal_exp40_15k_gt_features_v1"
NPZ_ROOT = DATA_ROOT / "npz" / "B_center_ellipsoid20_strict75"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "44_exp40_pointnet_mean_feature_sweep_v1"
DINO_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "43_exp40_b_dino_all_v1" / "facebook_dinov2_small_render_contrib_patch448x336"
C0 = 0.28209479177387814

BAD_SCENES = {
    "spatulas__032005",
    "slip_joint_pliers__090006",
    "slip_joint_pliers__092005",
    "locking_pliers__040003",
    "locking_pliers__091004",
    "utensils__022007",
    "slip_joint_pliers__093004",
    "locking_pliers__090004",
    "utensils__026007",
    "locking_pliers__041005",
}

MODEL_CHOICES = ["mlp", "pointnet_max", "pointnet_mean", "pointnet_max_mean"]
FEATURE_CHOICES = [
    "xyz_color",
    "xyz_scale_opacity_color",
    "geometry_color_scene_norm",
    "dino_only",
    "dino_geometry_color",
    "dino_geometry_color_quality",
]
ACTIVE_FEATURE_VARIANT = "geometry_color_scene_norm"
ACTIVE_DINO_ROOT = DINO_ROOT


def scalar_str(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    value = z[key]
    return str(value[0] if getattr(value, "shape", ()) else value)


def read_item(path: Path) -> dict:
    category, scene_id = path.stem.rsplit("__", 1)
    with np.load(path, allow_pickle=False) as z:
        y = z["handle_labels_thr0_25"]
        return {
            "scene_key": path.stem,
            "category": scalar_str(z, "category", category),
            "scene_id": scalar_str(z, "scene_id", scene_id),
            "instance_id": scalar_str(z, "instance_id", scene_id),
            "npz": str(path),
            "source_split": scalar_str(z, "model_split", "unknown"),
            "raw_split": scalar_str(z, "raw_split", "unknown"),
            "source_object_ply": scalar_str(z, "source_object_ply", ""),
            "num_gaussians": int(len(y)),
            "num_handle": int(y.sum()),
            "handle_ratio": float(y.mean()) if len(y) else 0.0,
        }


def load_items(npz_root: Path) -> list[dict]:
    paths = sorted(npz_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ files found in {npz_root}")
    items = [read_item(p) for p in paths]
    found_bad = {x["scene_key"] for x in items} & BAD_SCENES
    missing_bad = sorted(BAD_SCENES - {x["scene_key"] for x in items})
    items = [x for x in items if x["scene_key"] not in BAD_SCENES]
    if missing_bad:
        print(f"[dataset] warning: excluded scene keys not found: {missing_bad}", flush=True)
    print(f"[dataset] excluded {len(found_bad)} bad scenes; remaining={len(items)}", flush=True)
    return sorted(items, key=lambda x: x["scene_key"])


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_seen(items: list[dict]) -> dict[str, list[dict]]:
    split = {"train": [], "val": [], "test": []}
    for item in items:
        name = item["source_split"]
        if name in split:
            split[name].append(item)
    return {k: sorted(v, key=lambda x: x["scene_key"]) for k, v in split.items()}


def split_loco(items: list[dict], heldout_category: str) -> dict[str, list[dict]]:
    return {
        "train": sorted(
            [x for x in items if x["category"] != heldout_category and x["source_split"] == "train"],
            key=lambda x: x["scene_key"],
        ),
        "val": sorted(
            [x for x in items if x["category"] != heldout_category and x["source_split"] == "val"],
            key=lambda x: x["scene_key"],
        ),
        "test": sorted([x for x in items if x["category"] == heldout_category], key=lambda x: x["scene_key"]),
    }


def write_split_summary(split: dict[str, list[dict]], path: Path) -> None:
    rows = []
    for split_name, group in split.items():
        by_category: dict[str, list[dict]] = defaultdict(list)
        for item in group:
            by_category[item["category"]].append(item)
        for category, rows_cat in sorted(by_category.items()):
            n = sum(x["num_gaussians"] for x in rows_cat)
            pos = sum(x["num_handle"] for x in rows_cat)
            rows.append(
                {
                    "split": split_name,
                    "category": category,
                    "num_scenes": len(rows_cat),
                    "num_instances": len({x["instance_id"] for x in rows_cat}),
                    "num_gaussians": n,
                    "num_handle": pos,
                    "handle_ratio": pos / max(n, 1),
                }
            )
    save_csv(path, rows)


def write_dataset_summary(items: list[dict], out_root: Path) -> None:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)
    rows = []
    for category, group in sorted(by_category.items()):
        n = sum(x["num_gaussians"] for x in group)
        pos = sum(x["num_handle"] for x in group)
        rows.append(
            {
                "category": category,
                "num_scenes": len(group),
                "num_instances": len({x["instance_id"] for x in group}),
                "num_gaussians": n,
                "num_handle": pos,
                "handle_ratio": pos / max(n, 1),
            }
        )
    save_csv(out_root / "dataset_summary_by_category.csv", rows)


def resolve_dino_path(scene_key: str) -> Path:
    feature_dir = ACTIVE_DINO_ROOT / "features"
    candidates = [
        feature_dir / f"{scene_key}_dinov2_render_contrib_features.npz",
        feature_dir / f"{scene_key}_full_center_avg_features.npz",
        feature_dir / f"{scene_key}_object_crop_center_avg_features.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(feature_dir.glob(f"{scene_key}_*features.npz"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Missing or ambiguous DINO features for {scene_key} in {feature_dir}: {matches[:5]}")


def load_dino_arrays(item: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = resolve_dino_path(item["scene_key"])
    with np.load(path, allow_pickle=False) as z:
        dino = z["dino_features"].astype(np.float32)
        valid = z["valid_dino"].astype(bool).reshape(-1, 1)
        weight = z["dino_weight_sum"].astype(np.float32).reshape(-1, 1)
        views = z["dino_visible_views"].astype(np.float32).reshape(-1, 1)
    if len(dino) != n:
        raise ValueError(f"DINO length mismatch for {item['scene_key']}: {len(dino)} vs {n}")
    dino = dino.copy()
    dino[~valid[:, 0]] = 0.0
    return dino, valid.astype(np.float32), weight, views


def load_feature_arrays_variant(item: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(item["npz"], allow_pickle=False) as z:
        xyz = z["xyz"].astype(np.float32)
        xyz_norm = g15.base.normalize_xyz(xyz)
        scale = z["scale"].astype(np.float32)
        rotation = z["rotation"].astype(np.float32)
        opacity = np.asarray(z["opacity"], dtype=np.float32).reshape(-1, 1)
        color = z["color"].astype(np.float32)
        y = z["handle_labels_thr0_25"].astype(np.float32)

    geom_color = np.concatenate(
        [xyz_norm, g15.scene_zscore(scale), rotation, g15.scene_minmax_unit(opacity), g15.scene_zscore(color)],
        axis=1,
    )
    if ACTIVE_FEATURE_VARIANT == "geometry_color_scene_norm":
        x = geom_color
    elif ACTIVE_FEATURE_VARIANT == "xyz_color":
        x = np.concatenate([xyz_norm, g15.scene_zscore(color)], axis=1)
    elif ACTIVE_FEATURE_VARIANT == "xyz_scale_opacity_color":
        x = np.concatenate([xyz_norm, g15.scene_zscore(scale), g15.scene_minmax_unit(opacity), g15.scene_zscore(color)], axis=1)
    elif ACTIVE_FEATURE_VARIANT in {"dino_only", "dino_geometry_color", "dino_geometry_color_quality"}:
        dino, valid, weight, views = load_dino_arrays(item, len(y))
        if ACTIVE_FEATURE_VARIANT == "dino_only":
            x = dino
        elif ACTIVE_FEATURE_VARIANT == "dino_geometry_color":
            x = np.concatenate([dino, geom_color], axis=1)
        else:
            quality = np.concatenate([valid, g15.scene_zscore(np.log1p(weight)), views / 96.0], axis=1)
            x = np.concatenate([dino, geom_color, quality], axis=1)
    else:
        raise ValueError(f"Unknown feature variant: {ACTIVE_FEATURE_VARIANT}")
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def load_scene_arrays(item: dict) -> tuple[np.ndarray, np.ndarray]:
    return load_feature_arrays_variant(item)


def fit_standardizer(items: list[dict]) -> tuple[np.ndarray, np.ndarray, int, int]:
    total = 0
    pos = 0
    sum_x = None
    sumsq_x = None
    for item in items:
        x, y = load_scene_arrays(item)
        x64 = x.astype(np.float64, copy=False)
        sum_x = x64.sum(axis=0) if sum_x is None else sum_x + x64.sum(axis=0)
        sumsq_x = np.square(x64).sum(axis=0) if sumsq_x is None else sumsq_x + np.square(x64).sum(axis=0)
        total += len(x)
        pos += int(y.sum())
    mean64 = sum_x / max(total, 1)
    var64 = (sumsq_x / max(total, 1)) - np.square(mean64)
    mean = mean64.astype(np.float32)[None, :]
    std = np.sqrt(np.maximum(var64, 1e-12)).astype(np.float32)[None, :]
    return mean, np.maximum(std, 1e-6), total, pos


def load_scene_list(items: list[dict], mean: np.ndarray, std: np.ndarray) -> list[dict]:
    scenes = []
    for item in items:
        x, y = load_scene_arrays(item)
        scenes.append({"item": item, "x": ((x - mean) / std).astype(np.float32), "y": y})
    return scenes


def load_flat_arrays(items: list[dict], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    xs, ys, slices = [], [], []
    offset = 0
    for item in items:
        x, y = load_scene_arrays(item)
        x = ((x - mean) / std).astype(np.float32, copy=False)
        xs.append(x)
        ys.append(y)
        slices.append((item, offset, offset + len(y)))
        offset += len(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), slices


class MLP(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PointNet(nn.Module):
    def __init__(self, in_dim: int, pooling: str, latent_dim: int, dropout: float):
        super().__init__()
        self.pooling = pooling
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(latent_dim),
        )
        mult = 1 if pooling in {"max", "mean"} else 2
        self.head = nn.Sequential(
            nn.Linear(latent_dim * (1 + mult), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.encoder(x)
        if self.pooling == "max":
            global_feat = local.max(dim=0, keepdim=True).values
        elif self.pooling == "mean":
            global_feat = local.mean(dim=0, keepdim=True)
        elif self.pooling == "max_mean":
            global_feat = torch.cat([local.max(dim=0, keepdim=True).values, local.mean(dim=0, keepdim=True)], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        global_feat = global_feat.expand(len(local), -1)
        return self.head(torch.cat([local, global_feat], dim=1)).squeeze(-1)


def make_model(model_name: str, in_dim: int, args: argparse.Namespace) -> nn.Module:
    if model_name == "mlp":
        return MLP(in_dim, args.dropout)
    if model_name == "pointnet_max":
        return PointNet(in_dim, "max", args.latent_dim, args.dropout)
    if model_name == "pointnet_mean":
        return PointNet(in_dim, "mean", args.latent_dim, args.dropout)
    if model_name == "pointnet_max_mean":
        return PointNet(in_dim, "max_mean", args.latent_dim, args.dropout)
    raise ValueError(f"Unknown model: {model_name}")


def predict_flat(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            out.append(model(xb).detach().cpu().numpy())
    return g15.base.sigmoid_np(np.concatenate(out, axis=0)).astype(np.float32)


def predict_scenes(model: nn.Module, scenes: list[dict], device: torch.device) -> tuple[np.ndarray, np.ndarray, list]:
    model.eval()
    labels, scores, slices = [], [], []
    offset = 0
    with torch.no_grad():
        for scene in scenes:
            x = torch.from_numpy(scene["x"]).to(device)
            score = g15.base.sigmoid_np(model(x).detach().cpu().numpy()).astype(np.float32)
            y = scene["y"].astype(np.float32, copy=False)
            labels.append(y)
            scores.append(score)
            slices.append((scene["item"], offset, offset + len(y)))
            offset += len(y)
    return np.concatenate(labels), np.concatenate(scores), slices


def bce_loss_from_scores(logits: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    return loss_fn(logits, y)


def eval_flat_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, loss_fn: nn.Module, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            yb = torch.from_numpy(y[start : start + batch_size]).to(device)
            losses.append(float(bce_loss_from_scores(model(xb), yb, loss_fn).detach().cpu()))
    return float(np.mean(losses))


def eval_scene_loss(model: nn.Module, scenes: list[dict], loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for scene in scenes:
            x = torch.from_numpy(scene["x"]).to(device)
            y = torch.from_numpy(scene["y"]).to(device)
            losses.append(float(loss_fn(model(x), y).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_epoch_mlp(model, x_train, y_train, loader, loss_fn, opt, device) -> float:
    model.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_epoch_pointnet(model, scenes, loss_fn, opt, device, seed, grad_clip) -> float:
    model.train()
    order = list(range(len(scenes)))
    random.Random(seed).shuffle(order)
    losses = []
    for idx in order:
        scene = scenes[idx]
        x = torch.from_numpy(scene["x"]).to(device)
        y = torch.from_numpy(scene["y"]).to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def plot_curves(model_dir: Path, history: list[dict]) -> None:
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    for kind, columns, ylabel in [
        ("loss_curve", ["train_loss", "val_loss"], "Weighted BCE loss"),
        ("accuracy_curve", ["train_accuracy", "val_accuracy"], "Micro accuracy"),
    ]:
        plt.figure(figsize=(7, 4))
        for col in columns:
            plt.plot(epochs, [r[col] for r in history], label=col)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_dir / f"{kind}.png", dpi=150)
        plt.close()


def rgb_to_sh(rgb_01: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb_01, dtype=np.float32) - 0.5) / C0


def qa_colors(labels: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    labels = labels.astype(bool)
    pred = scores >= threshold
    rgb = np.zeros((len(labels), 3), dtype=np.float32)
    rgb[np.logical_and(~labels, ~pred)] = np.array([0.02, 0.16, 1.0], dtype=np.float32)  # TN blue
    rgb[np.logical_and(labels, pred)] = np.array([0.02, 0.85, 0.12], dtype=np.float32)  # TP green
    rgb[np.logical_and(labels, ~pred)] = np.array([1.0, 0.02, 0.02], dtype=np.float32)  # FN red
    rgb[np.logical_and(~labels, pred)] = np.array([1.0, 0.48, 0.02], dtype=np.float32)  # FP orange
    return rgb


def write_colored_ply(source_ply: Path, out_path: Path, rgb_01: np.ndarray) -> None:
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(rgb_01):
        raise ValueError(f"PLY/color length mismatch for {source_ply}: {len(vertices)} vs {len(rgb_01)}")
    sh = rgb_to_sh(rgb_01)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    elements = []
    for element in ply.elements:
        elements.append(PlyElement.describe(vertices, "vertex") if element.name == "vertex" else element)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(out_path))


def select_qa_rows(test_rows: list[dict]) -> list[dict]:
    by_cat_used = set()
    selected = []

    def add_rows(role: str, candidates: list[dict], count: int) -> None:
        nonlocal selected
        for row in candidates:
            if len([x for x in selected if x["selection_role"] == role]) >= count:
                break
            if row["scene_key"] in {x["scene_key"] for x in selected}:
                continue
            if row["category"] in by_cat_used and len(candidates) >= count:
                continue
            copied = dict(row)
            copied["selection_role"] = role
            selected.append(copied)
            by_cat_used.add(row["category"])
        for row in candidates:
            if len([x for x in selected if x["selection_role"] == role]) >= count:
                break
            if row["scene_key"] in {x["scene_key"] for x in selected}:
                continue
            copied = dict(row)
            copied["selection_role"] = role
            selected.append(copied)

    rows = sorted(test_rows, key=lambda r: float(r["iou"]))
    add_rows("bad", rows, 3)
    lo = max(0, int(len(rows) * 0.15))
    hi = max(lo + 1, int(len(rows) * 0.45))
    add_rows("poor", rows[lo:hi], 2)
    add_rows("good", list(reversed(rows)), 3)
    return selected


def export_qa_plys(model_dir: Path, test_rows: list[dict], test_slices: list, scores: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    slice_by_key = {item["scene_key"]: (item, start, end) for item, start, end in test_slices}
    manifest = []
    for row in select_qa_rows(test_rows):
        item, start, end = slice_by_key[row["scene_key"]]
        source = Path(item["source_object_ply"])
        out = (
            model_dir
            / "prediction_plys"
            / row["selection_role"]
            / f"{row['selection_role']}__{row['scene_key']}__iou{float(row['iou']):.3f}.ply"
        )
        write_colored_ply(source, out, qa_colors(labels[start:end], scores[start:end], threshold))
        manifest.append(
            {
                "selection_role": row["selection_role"],
                "scene_key": row["scene_key"],
                "category": row["category"],
                "iou": row["iou"],
                "f1": row["f1"],
                "source_object_ply": str(source),
                "output_ply": str(out),
            }
        )
    save_csv(model_dir / "prediction_plys" / "prediction_ply_manifest.csv", manifest)


def model_architecture(model_name: str) -> str:
    if model_name == "mlp":
        return "per-Gaussian MLP baseline; no object-level global pooling"
    pooling = model_name[len("pointnet_") :] if model_name.startswith("pointnet_") else model_name
    return f"PointNet shared encoder; {pooling} global pooling concatenated back to each Gaussian"


def train_one(model_name: str, run_name: str, split: dict[str, list[dict]], out_dir: Path, args, device) -> dict:
    model_dir = out_dir / ACTIVE_FEATURE_VARIANT / model_name
    done_path = model_dir / "overall_metrics.json"
    if args.skip_existing and done_path.exists():
        print(f"[{run_name}/{model_name}] skip existing", flush=True)
        with done_path.open() as f:
            return json.load(f)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{run_name}/{model_name}] split train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}",
        flush=True,
    )
    mean, std, train_n, train_pos = fit_standardizer(split["train"])
    pos_weight = (train_n - train_pos) / max(float(train_pos), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    if model_name == "mlp":
        x_train, y_train, train_slices = load_flat_arrays(split["train"], mean, std)
        x_val, y_val, val_slices = load_flat_arrays(split["val"], mean, std)
        x_test, y_test, test_slices = load_flat_arrays(split["test"], mean, std)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        model = make_model(model_name, int(x_train.shape[1]), args).to(device)
    else:
        train_scenes = load_scene_list(split["train"], mean, std)
        val_scenes = load_scene_list(split["val"], mean, std)
        test_scenes = load_scene_list(split["test"], mean, std)
        in_dim = int(train_scenes[0]["x"].shape[1])
        model = make_model(model_name, in_dim, args).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        if model_name == "mlp":
            train_loss = train_epoch_mlp(model, x_train, y_train, loader, loss_fn, opt, device)
            train_score = predict_flat(model, x_train, args.batch_size, device)
            val_score = predict_flat(model, x_val, args.batch_size, device)
            th_best, _ = g15.choose_threshold(y_val, val_score, val_slices)
            train_metrics = g15.metrics_from_scores(y_train, train_score, th_best["threshold"])
            val_rows = g15.per_scene_metrics(val_slices, y_val, val_score, th_best["threshold"])
            val_loss = eval_flat_loss(model, x_val, y_val, loss_fn, args.batch_size, device)
        else:
            train_loss = train_epoch_pointnet(model, train_scenes, loss_fn, opt, device, args.seed + epoch, args.grad_clip)
            y_train, train_score, train_slices = predict_scenes(model, train_scenes, device)
            y_val, val_score, val_slices = predict_scenes(model, val_scenes, device)
            th_best, _ = g15.choose_threshold(y_val, val_score, val_slices)
            train_metrics = g15.metrics_from_scores(y_train, train_score, th_best["threshold"])
            val_rows = g15.per_scene_metrics(val_slices, y_val, val_score, th_best["threshold"])
            val_loss = eval_scene_loss(model, val_scenes, loss_fn, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": g15.macro_scene_score(val_rows, "accuracy"),
            "val_macro_scene_iou": g15.macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": g15.macro_scene_score(val_rows, "f1"),
            "val_macro_scene_auprc": g15.macro_scene_score(val_rows, "auprc"),
            "val_threshold": th_best["threshold"],
        }
        history.append(record)
        score_tuple = (record["val_macro_scene_iou"], record["val_macro_scene_f1"])
        if best is None or score_tuple > (best["val_macro_scene_iou"], best["val_macro_scene_f1"]):
            best = dict(record)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[{run_name}/{model_name}] epoch {epoch:03d}/{args.epochs} "
                f"loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={record['val_macro_scene_iou']:.3f} "
                f"val_f1={record['val_macro_scene_f1']:.3f} val_auprc={record['val_macro_scene_auprc']:.3f} "
                f"th={record['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{run_name}/{model_name}] early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    if model_name == "mlp":
        val_score = predict_flat(model, x_val, args.batch_size, device)
        th_best, threshold_curve = g15.choose_threshold(y_val, val_score, val_slices)
        threshold = th_best["threshold"]
        test_score = predict_flat(model, x_test, args.batch_size, device)
    else:
        y_val, val_score, val_slices = predict_scenes(model, val_scenes, device)
        th_best, threshold_curve = g15.choose_threshold(y_val, val_score, val_slices)
        threshold = th_best["threshold"]
        y_test, test_score, test_slices = predict_scenes(model, test_scenes, device)

    test_rows = g15.per_scene_metrics(test_slices, y_test, test_score, threshold)
    cat_rows = g15.per_category_metrics(test_rows)
    micro = g15.metrics_from_scores(y_test, test_score, threshold)
    macro = {key: g15.macro_scene_score(test_rows, key) for key in g15.METRIC_NAMES}
    in_dim = int(mean.shape[1])
    overall = {
        "run_name": run_name,
        "model": model_name,
        "architecture": model_architecture(model_name),
        "feature_variant": ACTIVE_FEATURE_VARIANT,
        "input_dim": in_dim,
        "latent_dim": int(args.latent_dim),
        "train_scenes": len(split["train"]),
        "val_scenes": len(split["val"]),
        "test_scenes": len(split["test"]),
        "train_gaussians": int(train_n),
        "val_gaussians": int(sum(x["num_gaussians"] for x in split["val"])),
        "test_gaussians": int(len(y_test)),
        "train_handle_ratio": float(train_pos / max(train_n, 1)),
        "val_handle_ratio": float(sum(x["num_handle"] for x in split["val"]) / max(sum(x["num_gaussians"] for x in split["val"]), 1)),
        "test_handle_ratio": float(y_test.mean()),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(threshold),
        "selection_metric": "validation macro scene IoU, tie-broken by validation macro scene F1",
        "loss": "weighted BCEWithLogitsLoss",
        "positive_label": "handle_labels_thr0_25",
        "bad_scenes_excluded": sorted(BAD_SCENES),
        "micro": micro,
        "macro_scene_average": macro,
        "best_scene_iou": max(test_rows, key=lambda r: r["iou"])["scene_key"],
        "worst_scene_iou": min(test_rows, key=lambda r: r["iou"])["scene_key"],
    }

    save_csv(model_dir / "history.csv", history)
    save_csv(model_dir / "threshold_curve.csv", threshold_curve)
    save_csv(model_dir / "per_scene_metrics.csv", test_rows)
    save_csv(model_dir / "per_category_metrics.csv", cat_rows)
    plot_curves(model_dir, history)
    with (model_dir / "overall_metrics.json").open("w") as f:
        json.dump(overall, f, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "model_name": model_name,
            "feature_variant": ACTIVE_FEATURE_VARIANT,
            "input_dim": in_dim,
            "latent_dim": int(args.latent_dim),
            "threshold": float(threshold),
            "overall": overall,
        },
        model_dir / "model.pt",
    )
    offsets = np.asarray([[start, end] for _, start, end in test_slices], dtype=np.int64)
    scene_keys = np.asarray([item["scene_key"] for item, _, _ in test_slices])
    np.savez_compressed(
        model_dir / "test_predictions.npz",
        scene_keys=scene_keys,
        offsets=offsets,
        scores=test_score.astype(np.float32),
        labels=y_test.astype(np.uint8),
    )
    export_qa_plys(model_dir, test_rows, test_slices, test_score, y_test, threshold)
    print(
        f"[{run_name}/{model_name}] TEST macro IoU={macro['iou']:.3f} "
        f"macro F1={macro['f1']:.3f} AUPRC={macro['auprc']:.3f} threshold={threshold:.2f}",
        flush=True,
    )
    return overall


def flat_summary_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        macro = result["macro_scene_average"]
        micro = result["micro"]
        rows.append(
            {
                "run_name": result["run_name"],
                "model": result["model"],
                "feature_variant": result["feature_variant"],
                "input_dim": result["input_dim"],
                "latent_dim": result["latent_dim"],
                "train_scenes": result["train_scenes"],
                "val_scenes": result["val_scenes"],
                "test_scenes": result["test_scenes"],
                "selected_epoch": result["selected_epoch"],
                "selected_threshold": result["selected_threshold"],
                "macro_iou": macro["iou"],
                "macro_f1": macro["f1"],
                "macro_precision": macro["precision"],
                "macro_recall": macro["recall"],
                "macro_auprc": macro["auprc"],
                "macro_sim": macro["sim"],
                "macro_mae": macro["mae"],
                "macro_roc_auc": macro["roc_auc"],
                "macro_gt_handle_ratio": macro["gt_handle_ratio"],
                "macro_pred_handle_ratio": macro["pred_handle_ratio"],
                "micro_iou": micro["iou"],
                "micro_f1": micro["f1"],
                "micro_auprc": micro["auprc"],
                "micro_sim": micro["sim"],
                "micro_mae": micro["mae"],
                "micro_roc_auc": micro["roc_auc"],
                "micro_correct_percent": micro["correct_percent"],
                "worst_scene_iou": result["worst_scene_iou"],
                "best_scene_iou": result["best_scene_iou"],
            }
        )
    return rows


def aggregate_results(out_root: Path) -> None:
    results = []
    for path in sorted(out_root.glob("**/overall_metrics.json")):
        with path.open() as f:
            results.append(json.load(f))
    save_csv(out_root / "all_finished_models_summary.csv", flat_summary_rows(results))
    with (out_root / "all_finished_models_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[aggregate] collected {len(results)} finished model results", flush=True)


def run_seen(items: list[dict], model_name: str, args, device) -> list[dict]:
    split = split_seen(items)
    if args.limit_train_scenes is not None:
        split["train"] = split["train"][: args.limit_train_scenes]
    if args.limit_val_scenes is not None:
        split["val"] = split["val"][: args.limit_val_scenes]
    if args.limit_test_scenes is not None:
        split["test"] = split["test"][: args.limit_test_scenes]
    out_dir = args.out_root / "01_seen_instance_17cat"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "split_manifest.json").open("w") as f:
        json.dump(
            {
                "experiment": "Exp41 seen category / fixed Exp40 model_split",
                "npz_root": str(args.npz_root),
                "bad_scenes_excluded": sorted(BAD_SCENES),
                "splits": split,
            },
            f,
            indent=2,
        )
    write_split_summary(split, out_dir / "split_summary.csv")
    result = train_one(model_name, "01_seen_instance_17cat", split, out_dir, args, device)
    save_csv(out_dir / f"seen_summary_{ACTIVE_FEATURE_VARIANT}_{model_name}.csv", flat_summary_rows([result]))
    return [result]


def run_loco(items: list[dict], model_name: str, args, device) -> list[dict]:
    categories = sorted({x["category"] for x in items})
    if args.heldout_categories:
        categories = [c for c in categories if c in set(args.heldout_categories)]
    results = []
    for category in categories:
        split = split_loco(items, category)
        out_dir = args.out_root / "02_leave_one_category_out_17cat" / category
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "split_manifest.json").open("w") as f:
            json.dump(
                {
                    "heldout_category": category,
                    "split_rule": "train/val use Exp40 model_split train/val from non-heldout categories; test is all heldout category scenes",
                    "npz_root": str(args.npz_root),
                    "bad_scenes_excluded": sorted(BAD_SCENES),
                    "splits": split,
                },
                f,
                indent=2,
            )
        write_split_summary(split, out_dir / "split_summary.csv")
        result = train_one(model_name, category, split, out_dir, args, device)
        save_csv(out_dir / f"category_summary_{ACTIVE_FEATURE_VARIANT}_{model_name}.csv", flat_summary_rows([result]))
        results.append(result)
        save_csv(args.out_root / f"loco_summary_{ACTIVE_FEATURE_VARIANT}_{model_name}.csv", flat_summary_rows(results))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=Path, default=NPZ_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--feature_variant", choices=FEATURE_CHOICES, default="geometry_color_scene_norm")
    parser.add_argument("--dino_root", type=Path, default=DINO_ROOT)
    parser.add_argument("--limit_train_scenes", type=int, default=None)
    parser.add_argument("--limit_val_scenes", type=int, default=None)
    parser.add_argument("--limit_test_scenes", type=int, default=None)
    parser.add_argument("--modes", nargs="+", choices=["seen", "loco"], default=["seen", "loco"])
    parser.add_argument("--heldout_categories", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    global ACTIVE_FEATURE_VARIANT, ACTIVE_DINO_ROOT
    ACTIVE_FEATURE_VARIANT = args.feature_variant
    ACTIVE_DINO_ROOT = args.dino_root
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_results(args.out_root)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    items = load_items(args.npz_root)
    write_dataset_summary(items, args.out_root)
    with (args.out_root / "experiment_config.json").open("w") as f:
        json.dump(
            {
                "experiment": "Exp44 PointNet Mean feature sweep on Exp40 B_center_ellipsoid20_strict75",
                "npz_root": str(args.npz_root),
                "feature_variant": ACTIVE_FEATURE_VARIANT,
                "models": MODEL_CHOICES,
                "dino_root": str(args.dino_root),
                "feature_variants": FEATURE_CHOICES,
                "bad_scenes_excluded": sorted(BAD_SCENES),
                "metrics": g15.METRIC_NAMES,
                "qa_ply_colors": {
                    "TN": "blue",
                    "TP": "green",
                    "FN": "red",
                    "FP": "orange",
                },
            },
            f,
            indent=2,
        )

    results = []
    if "seen" in args.modes:
        results.extend(run_seen(items, args.model, args, device))
    if "loco" in args.modes:
        results.extend(run_loco(items, args.model, args, device))
    save_csv(args.out_root / f"summary_{ACTIVE_FEATURE_VARIANT}_{args.model}.csv", flat_summary_rows(results))
    with (args.out_root / f"summary_{ACTIVE_FEATURE_VARIANT}_{args.model}.json").open("w") as f:
        json.dump(results, f, indent=2)
    aggregate_results(args.out_root)


if __name__ == "__main__":
    main()
