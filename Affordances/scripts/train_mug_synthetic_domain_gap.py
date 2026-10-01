#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))
import train_handal_17cat_mlp_baselines as base  # noqa: E402
from train_handal_17cat_pointnet_global import PointNetGlobal  # noqa: E402


SYN_ROOT = BASE_ROOT / "data" / "affordsplat_mug_grasp_features_v1"
HANDAL_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "10_mug_synthetic_domain_gap_v1"

NOISE_PRESETS = {
    "clean": {"xyz": 0.0, "scale": 0.0, "opacity": 0.0, "color": 0.0, "dropout": 0.0},
    "mild": {"xyz": 0.01, "scale": 0.03, "opacity": 0.03, "color": 0.03, "dropout": 0.05},
    "medium": {"xyz": 0.025, "scale": 0.08, "opacity": 0.08, "color": 0.08, "dropout": 0.15},
    "strong": {"xyz": 0.05, "scale": 0.15, "opacity": 0.15, "color": 0.15, "dropout": 0.30},
}


def str_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.tolist())


def read_item(path, source_domain):
    with np.load(path, allow_pickle=False) as z:
        n = int(len(z["handle_labels_thr0_25"]))
        pos = int(z["handle_labels_thr0_25"].sum())
        split = str_scalar(z["split"]) if "split" in z.files else "unknown"
        scene_id = str_scalar(z["scene_id"]) if "scene_id" in z.files else path.stem
        instance_id = str_scalar(z["instance_id"]) if "instance_id" in z.files else scene_id[:3]
    return {
        "scene_key": path.stem,
        "category": "mugs",
        "scene_id": scene_id,
        "instance_id": instance_id,
        "npz": str(path),
        "source_domain": source_domain,
        "source_split": split,
        "num_gaussians": n,
        "num_handle": pos,
        "handle_ratio": float(pos / max(n, 1)),
    }


def load_items(syn_root, handal_root):
    synthetic = [read_item(p, "affordsplat") for p in sorted((syn_root / "npz").glob("*.npz"))]
    handal = [read_item(p, "handal") for p in sorted((handal_root / "npz").glob("mugs__*.npz"))]
    if not synthetic:
        raise FileNotFoundError(f"No synthetic NPZ files in {syn_root / 'npz'}")
    if not handal:
        raise FileNotFoundError(f"No HANDAL mug NPZ files in {handal_root / 'npz'}")
    return synthetic, handal


def split_synthetic(items):
    split = {"train": [], "val": [], "test": []}
    for item in items:
        split[item["source_split"]].append(item)
    return {k: sorted(v, key=lambda x: x["scene_key"]) for k, v in split.items()}


def split_handal(items):
    mapping = {"train_seen": "train", "val_seen": "val", "test_seen": "test"}
    split = {"train": [], "val": [], "test": []}
    for item in items:
        raw = item["source_split"].strip("[]'\"")
        key = mapping.get(raw, raw)
        if key in split:
            split[key].append(item)
    return {k: sorted(v, key=lambda x: x["scene_key"]) for k, v in split.items()}


def load_scene_arrays(item):
    with np.load(item["npz"], allow_pickle=False) as z:
        geom = z["geometry_features"].astype(np.float32).copy()
        xyz_norm = base.normalize_xyz(z["xyz"].astype(np.float32))
        geom[:, :3] = xyz_norm
        color = z["color"].astype(np.float32)
        y = z["handle_labels_thr0_25"].astype(np.float32)
    x = np.concatenate([geom, color], axis=1).astype(np.float32)
    return x, y


def apply_noise(x, y, preset, rng):
    cfg = NOISE_PRESETS[preset]
    if preset == "clean":
        return x, y
    x = x.copy()
    y = y.copy()
    if cfg["xyz"] > 0:
        x[:, :3] += rng.normal(0.0, cfg["xyz"], size=x[:, :3].shape).astype(np.float32)
    if cfg["scale"] > 0:
        x[:, 3:6] += rng.normal(0.0, cfg["scale"], size=x[:, 3:6].shape).astype(np.float32)
    if cfg["opacity"] > 0:
        x[:, 10] = np.clip(
            x[:, 10] + rng.normal(0.0, cfg["opacity"], size=x[:, 10].shape).astype(np.float32),
            0.0,
            1.0,
        )
    if cfg["color"] > 0:
        x[:, 11:14] = np.clip(
            x[:, 11:14] + rng.normal(0.0, cfg["color"], size=x[:, 11:14].shape).astype(np.float32),
            0.0,
            1.0,
        )
    if cfg["dropout"] > 0 and len(x) > 10:
        keep = rng.random(len(x)) >= cfg["dropout"]
        if keep.any() and keep.sum() >= 10:
            x = x[keep]
            y = y[keep]
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def make_scene(item, noise, seed, noise_sources):
    x, y = load_scene_arrays(item)
    if item["source_domain"] in noise_sources:
        rng = np.random.default_rng(seed + base.stable_key(item["scene_key"], seed) % 1_000_000)
        x, y = apply_noise(x, y, noise, rng)
    return {"item": item, "x": x, "y": y}


def fit_standardizer(scenes):
    total = 0
    sum_x = None
    sumsq_x = None
    pos = 0
    for scene in scenes:
        x = scene["x"].astype(np.float64, copy=False)
        sum_x = x.sum(axis=0) if sum_x is None else sum_x + x.sum(axis=0)
        sumsq_x = np.square(x).sum(axis=0) if sumsq_x is None else sumsq_x + np.square(x).sum(axis=0)
        total += len(x)
        pos += int(scene["y"].sum())
    mean = (sum_x / max(total, 1)).astype(np.float32)[None, :]
    var = (sumsq_x / max(total, 1)) - np.square(sum_x / max(total, 1))
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)[None, :]
    std = np.maximum(std, 1e-6)
    return mean, std, total, pos


def standardize_scenes(scenes, mean, std):
    out = []
    for scene in scenes:
        out.append(
            {
                "item": scene["item"],
                "x": ((scene["x"] - mean) / std).astype(np.float32, copy=False),
                "y": scene["y"].astype(np.float32, copy=False),
            }
        )
    return out


def flatten_scenes(scenes):
    xs, ys, slices = [], [], []
    offset = 0
    for scene in scenes:
        xs.append(scene["x"])
        ys.append(scene["y"])
        end = offset + len(scene["y"])
        slices.append((scene["item"], offset, end))
        offset = end
    return np.concatenate(xs), np.concatenate(ys), slices


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict_mlp(model, x, batch_size, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(torch.from_numpy(x[start : start + batch_size]).to(device))
            scores.append(logits.detach().cpu().numpy())
    return base.sigmoid_np(np.concatenate(scores))


def predict_pointnet(model, scenes, device):
    model.eval()
    scores, ys, slices = [], [], []
    offset = 0
    with torch.no_grad():
        for scene in scenes:
            logits = model(torch.from_numpy(scene["x"]).to(device))
            score = base.sigmoid_np(logits.detach().cpu().numpy())
            scores.append(score.astype(np.float32))
            ys.append(scene["y"].astype(np.float32))
            end = offset + len(score)
            slices.append((scene["item"], offset, end))
            offset = end
    return np.concatenate(ys), np.concatenate(scores), slices


def train_mlp(run_name, scenes_train, scenes_val, scenes_test, args, device, out_dir):
    x_train, y_train, _ = flatten_scenes(scenes_train)
    x_val, y_val, val_slices = flatten_scenes(scenes_val)
    x_test, y_test, test_slices = flatten_scenes(scenes_test)
    model = MLP(x_train.shape[1]).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    best, best_state, stale = None, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_score = predict_mlp(model, x_val, args.batch_size, device)
        th, _ = base.choose_threshold(y_val, val_score, val_slices)
        val_rows = base.per_scene_metrics(val_slices, y_val, val_score, th["threshold"])
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_scene_iou": base.macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": base.macro_scene_score(val_rows, "f1"),
            "val_threshold": th["threshold"],
        }
        history.append(rec)
        score = (rec["val_macro_scene_iou"], rec["val_macro_scene_f1"])
        if best is None or score > (best["val_macro_scene_iou"], best["val_macro_scene_f1"]):
            best = dict(rec)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"[{run_name}/old_mlp] epoch {epoch:03d} val_iou={rec['val_macro_scene_iou']:.3f} th={rec['val_threshold']:.2f}", flush=True)
        if args.patience > 0 and stale >= args.patience:
            break
    model.load_state_dict(best_state)
    val_score = predict_mlp(model, x_val, args.batch_size, device)
    th, th_curve = base.choose_threshold(y_val, val_score, val_slices)
    threshold = th["threshold"]
    test_score = predict_mlp(model, x_test, args.batch_size, device)
    return save_result(run_name, "old_mlp_geometry_color", model, out_dir, history, th_curve, test_slices, y_test, test_score, threshold, best, x_train.shape[1], scenes_train, scenes_val, scenes_test, mean_std=None)


def train_pointnet(run_name, scenes_train, scenes_val, scenes_test, args, device, out_dir):
    model = PointNetGlobal(scenes_train[0]["x"].shape[1], latent_dim=args.latent_dim).to(device)
    train_n = sum(len(s["y"]) for s in scenes_train)
    train_pos = sum(float(s["y"].sum()) for s in scenes_train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(train_n - train_pos) / max(train_pos, 1.0)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.pointnet_lr, weight_decay=args.weight_decay)
    best, best_state, stale = None, None, 0
    history = []
    order = list(range(len(scenes_train)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for idx in order:
            scene = scenes_train[idx]
            x, y = torch.from_numpy(scene["x"]).to(device), torch.from_numpy(scene["y"]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        y_val, val_score, val_slices = predict_pointnet(model, scenes_val, device)
        th, _ = base.choose_threshold(y_val, val_score, val_slices)
        val_rows = base.per_scene_metrics(val_slices, y_val, val_score, th["threshold"])
        rec = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_scene_iou": base.macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": base.macro_scene_score(val_rows, "f1"),
            "val_threshold": th["threshold"],
        }
        history.append(rec)
        score = (rec["val_macro_scene_iou"], rec["val_macro_scene_f1"])
        if best is None or score > (best["val_macro_scene_iou"], best["val_macro_scene_f1"]):
            best = dict(rec)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"[{run_name}/pointnet] epoch {epoch:03d} val_iou={rec['val_macro_scene_iou']:.3f} th={rec['val_threshold']:.2f}", flush=True)
        if args.patience > 0 and stale >= args.patience:
            break
    model.load_state_dict(best_state)
    y_val, val_score, val_slices = predict_pointnet(model, scenes_val, device)
    th, th_curve = base.choose_threshold(y_val, val_score, val_slices)
    threshold = th["threshold"]
    y_test, test_score, test_slices = predict_pointnet(model, scenes_test, device)
    return save_result(run_name, "pointnet_global_geometry_color", model, out_dir, history, th_curve, test_slices, y_test, test_score, threshold, best, scenes_train[0]["x"].shape[1], scenes_train, scenes_val, scenes_test, mean_std=None)


def save_result(run_name, model_name, model, out_dir, history, th_curve, test_slices, y_test, test_score, threshold, best, input_dim, scenes_train, scenes_val, scenes_test, mean_std):
    model_dir = out_dir / run_name / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    test_rows = base.per_scene_metrics(test_slices, y_test, test_score, threshold)
    cat_rows = base.per_category_metrics(test_rows)
    micro = base.metrics_from_scores(y_test, test_score, threshold)
    macro = {
        key: base.macro_scene_score(test_rows, key)
        for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "iou", "gt_handle_ratio", "pred_handle_ratio", "correct_percent", "fp_percent", "fn_percent"]
    }
    overall = {
        "run_name": run_name,
        "model": model_name,
        "input_dim": int(input_dim),
        "train_scenes": len(scenes_train),
        "val_scenes": len(scenes_val),
        "test_scenes": len(scenes_test),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(threshold),
        "macro_scene_average": macro,
        "micro": micro,
        "worst_scene_iou": min(test_rows, key=lambda r: r["iou"])["scene_key"],
        "best_scene_iou": max(test_rows, key=lambda r: r["iou"])["scene_key"],
    }
    base.save_csv(model_dir / "history.csv", history)
    base.save_csv(model_dir / "threshold_curve.csv", th_curve)
    base.save_csv(model_dir / "per_scene_metrics.csv", test_rows)
    base.save_csv(model_dir / "per_category_metrics.csv", cat_rows)
    with open(model_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    offsets = np.asarray([[s, e] for _, s, e in test_slices], dtype=np.int64)
    scene_keys = np.asarray([item["scene_key"] for item, _, _ in test_slices])
    np.savez_compressed(model_dir / "test_predictions.npz", scene_keys=scene_keys, offsets=offsets, scores=test_score.astype(np.float32), labels=y_test.astype(np.uint8))
    torch.save({"model": model.state_dict(), "overall": overall, "threshold": float(threshold)}, model_dir / "model.pt")
    print(f"[{run_name}/{model_name}] TEST macro IoU={macro['iou']:.3f} F1={macro['f1']:.3f} th={threshold:.2f}", flush=True)
    return overall


def prepare_scenes(items, noise, seed, noise_sources, mean=None, std=None):
    scenes = [make_scene(item, noise, seed, noise_sources) for item in items]
    if mean is None or std is None:
        mean, std, _, _ = fit_standardizer(scenes)
    return standardize_scenes(scenes, mean, std), mean, std


def build_experiments(syn, handal):
    return [
        ("stage2_synthetic_sanity_clean", syn["train"], syn["val"], syn["test"], "clean", {"affordsplat"}),
        ("stage3_synth_to_handal_clean_synthval", syn["train"], syn["val"], handal["test"], "clean", {"affordsplat"}),
        ("stage3_synth_to_handal_clean_handalval", syn["train"], handal["val"], handal["test"], "clean", {"affordsplat"}),
        ("stage4_synth_to_handal_mild_handalval", syn["train"], handal["val"], handal["test"], "mild", {"affordsplat"}),
        ("stage4_synth_to_handal_medium_handalval", syn["train"], handal["val"], handal["test"], "medium", {"affordsplat"}),
        ("stage4_synth_to_handal_strong_handalval", syn["train"], handal["val"], handal["test"], "strong", {"affordsplat"}),
        ("stage5_handal_only", handal["train"], handal["val"], handal["test"], "clean", {"affordsplat"}),
        ("stage5_mixed_clean", handal["train"] + syn["train"], handal["val"], handal["test"], "clean", {"affordsplat"}),
        ("stage5_mixed_mild", handal["train"] + syn["train"], handal["val"], handal["test"], "mild", {"affordsplat"}),
        ("stage5_mixed_medium", handal["train"] + syn["train"], handal["val"], handal["test"], "medium", {"affordsplat"}),
        ("stage5_mixed_strong", handal["train"] + syn["train"], handal["val"], handal["test"], "strong", {"affordsplat"}),
    ]


def write_summary(out_root):
    rows = []
    for p in sorted(out_root.glob("**/overall_metrics.json")):
        data = json.load(open(p))
        macro, micro = data["macro_scene_average"], data["micro"]
        rows.append({
            "run_name": data["run_name"],
            "model": data["model"],
            "train_scenes": data["train_scenes"],
            "val_scenes": data["val_scenes"],
            "test_scenes": data["test_scenes"],
            "selected_epoch": data["selected_epoch"],
            "selected_threshold": data["selected_threshold"],
            "macro_iou": macro["iou"],
            "macro_f1": macro["f1"],
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "micro_iou": micro["iou"],
            "micro_f1": micro["f1"],
            "micro_correct_percent": micro["correct_percent"],
            "worst_scene_iou": data["worst_scene_iou"],
            "best_scene_iou": data["best_scene_iou"],
        })
    if rows:
        base.save_csv(out_root / "all_mug_domain_gap_summary.csv", rows)
        json.dump(rows, open(out_root / "all_mug_domain_gap_summary.json", "w"), indent=2)
    print(f"[aggregate] wrote {len(rows)} rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_root", type=Path, default=SYN_ROOT)
    parser.add_argument("--handal_root", type=Path, default=HANDAL_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--models", nargs="+", default=["old_mlp", "pointnet"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pointnet_lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    syn_items, handal_items = load_items(args.syn_root, args.handal_root)
    syn, handal = split_synthetic(syn_items), split_handal(handal_items)
    print(f"[data] synthetic train/val/test={len(syn['train'])}/{len(syn['val'])}/{len(syn['test'])}", flush=True)
    print(f"[data] HANDAL train/val/test={len(handal['train'])}/{len(handal['val'])}/{len(handal['test'])}", flush=True)

    manifest_rows = []
    for run_name, train_items, val_items, test_items, noise, noise_sources in build_experiments(syn, handal):
        manifest_rows.append({"run_name": run_name, "train_scenes": len(train_items), "val_scenes": len(val_items), "test_scenes": len(test_items), "noise": noise, "noise_sources": ",".join(sorted(noise_sources))})
    base.save_csv(args.out_root / "experiment_manifest.csv", manifest_rows)

    for run_name, train_items, val_items, test_items, noise, noise_sources in build_experiments(syn, handal):
        print(f"[run] {run_name} noise={noise}", flush=True)
        run_dir = args.out_root / run_name
        if args.skip_existing and all((run_dir / m / "overall_metrics.json").exists() for m in ["old_mlp_geometry_color", "pointnet_global_geometry_color"] if ("old_mlp" if m.startswith("old") else "pointnet") in args.models):
            print(f"[run] skip existing {run_name}", flush=True)
            continue
        train_raw = [make_scene(item, noise, args.seed, noise_sources) for item in train_items]
        mean, std, _, _ = fit_standardizer(train_raw)
        train = standardize_scenes(train_raw, mean, std)
        val, _, _ = prepare_scenes(val_items, "clean", args.seed, set(), mean, std)
        test, _, _ = prepare_scenes(test_items, "clean", args.seed, set(), mean, std)
        if "old_mlp" in args.models:
            train_mlp(run_name, train, val, test, args, device, args.out_root)
        if "pointnet" in args.models:
            train_pointnet(run_name, train, val, test, args, device, args.out_root)
        write_summary(args.out_root)
    write_summary(args.out_root)


if __name__ == "__main__":
    main()
