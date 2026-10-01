import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


C0 = 0.28209479177387814
BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
FEATURE_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "03_leave_one_category_out_v1"

CATEGORIES = [
    "adjustable_wrenches",
    "hammers",
    "measuring_cups",
    "mugs",
    "pots_pans",
    "power_drills",
    "screwdrivers",
    "spatulas",
    "whisks",
]

MODELS = {
    "old_mlp_dino_geometry_color": {
        "feature_set": {"dino": True, "geometry": True, "color": True},
        "context_mode": "none",
    },
    "knn_geometry_color": {
        "feature_set": {"dino": False, "geometry": True, "color": True},
        "context_mode": "knn",
    },
    "knn_dino_geometry_color": {
        "feature_set": {"dino": True, "geometry": True, "color": True},
        "context_mode": "knn",
    },
}


def stable_key(text, seed):
    h = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def sigmoid_np(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def read_npz_meta(path):
    category, scene_id = path.stem.rsplit("__", 1)
    instance_id = scene_id[:3]
    with np.load(path, allow_pickle=False) as z:
        n = int(len(z["handle_labels_thr0_25"]))
        pos = int(z["handle_labels_thr0_25"].sum())
        valid = int(z["valid_dino"].sum())
    return {
        "scene_key": path.stem,
        "category": category,
        "scene_id": scene_id,
        "instance_id": instance_id,
        "npz": str(path),
        "num_gaussians": n,
        "num_handle": pos,
        "handle_ratio": float(pos / max(n, 1)),
        "valid_dino": valid,
        "valid_dino_ratio": float(valid / max(n, 1)),
    }


def load_all_items(feature_root):
    paths = sorted((feature_root / "npz").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {feature_root / 'npz'}")
    return [read_npz_meta(path) for path in paths]


def split_train_val_by_instance(items, seed):
    by_category = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_category[item["category"]][item["instance_id"]].append(item)

    train, val = [], []
    for category in sorted(by_category):
        groups = by_category[category]
        names = sorted(groups)
        names.sort(key=lambda name: stable_key(f"{category}:{name}", seed))
        total = sum(len(groups[name]) for name in names)
        target_val = max(1, int(round(total * 0.15)))
        val_count = 0
        for name in names:
            group = sorted(groups[name], key=lambda x: x["scene_id"])
            if val_count < target_val:
                val.extend(group)
                val_count += len(group)
            else:
                train.extend(group)
    return sorted(train, key=lambda x: x["scene_key"]), sorted(val, key=lambda x: x["scene_key"])


def build_loco_split(items, heldout_category, seed):
    test = sorted([x for x in items if x["category"] == heldout_category], key=lambda x: x["scene_key"])
    rest = [x for x in items if x["category"] != heldout_category]
    train, val = split_train_val_by_instance(rest, seed + stable_key(heldout_category, seed))
    return {"train": train, "val": val, "test": test}


def normalize_xyz(xyz):
    xyz = xyz.astype(np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    return centered / radius


def load_base_scene(item, feature_set):
    with np.load(item["npz"], allow_pickle=False) as z:
        parts = []
        xyz_norm = normalize_xyz(z["xyz"].astype(np.float32))
        if feature_set["dino"]:
            parts.append(z["dino_features"].astype(np.float32))
        if feature_set["geometry"]:
            geom = z["geometry_features"].astype(np.float32).copy()
            geom[:, :3] = xyz_norm
            parts.append(geom)
        if feature_set["color"]:
            parts.append(z["color"].astype(np.float32))
        x = np.concatenate(parts, axis=1).astype(np.float32)
        y = z["handle_labels_thr0_25"].astype(np.float32)
    return x, y, xyz_norm


def local_mean_features(x, xyz_norm, k):
    n = len(x)
    if n <= 1:
        return x.copy()
    nn = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm="auto").fit(xyz_norm)
    indices = nbrs.kneighbors(xyz_norm, return_distance=False)
    if indices.shape[1] > 1:
        indices = indices[:, 1:]
    return x[indices].mean(axis=1).astype(np.float32)


def augment_scene(x, xyz_norm, context_mode, k_neighbors):
    if context_mode == "none":
        return x
    if context_mode == "knn":
        global_mean = np.repeat(x.mean(axis=0, keepdims=True), len(x), axis=0)
        local_mean = local_mean_features(x, xyz_norm, k_neighbors)
        return np.concatenate([x, global_mean, local_mean], axis=1).astype(np.float32)
    raise ValueError(f"Unknown context mode: {context_mode}")


def load_split_arrays(items, feature_set, context_mode, k_neighbors):
    xs, ys, slices = [], [], []
    offset = 0
    for item in items:
        x, y, xyz_norm = load_base_scene(item, feature_set)
        x = augment_scene(x, xyz_norm, context_mode, k_neighbors)
        xs.append(x)
        ys.append(y)
        slices.append((item, offset, offset + len(y)))
        offset += len(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), slices


def standardize_fit(x):
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        hidden = 384 if in_dim >= 512 else 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(192, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metrics_from_scores(y_true, score, threshold):
    y_true = y_true.astype(bool)
    y_pred = score >= threshold
    tp = int(np.logical_and(y_pred, y_true).sum())
    tn = int(np.logical_and(~y_pred, ~y_true).sum())
    fp = int(np.logical_and(y_pred, ~y_true).sum())
    fn = int(np.logical_and(~y_pred, y_true).sum())
    n = max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "num_gaussians": int(n),
        "gt_handle_ratio": float(y_true.mean()) if n else 0.0,
        "pred_handle_ratio": float(y_pred.mean()) if n else 0.0,
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp_percent": float(tp / n),
        "tn_percent": float(tn / n),
        "fp_percent": float(fp / n),
        "fn_percent": float(fn / n),
        "correct_percent": float((tp + tn) / n),
        "incorrect_percent": float((fp + fn) / n),
    }


def predict_scores(model, x, batch_size, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            logits.append(model(xb).detach().cpu().numpy())
    return sigmoid_np(np.concatenate(logits, axis=0))


def per_scene_metrics(slices, y_all, score_all, threshold):
    rows = []
    for item, start, end in slices:
        row = {
            "scene_key": item["scene_key"],
            "category": item["category"],
            "scene_id": item["scene_id"],
            "instance_id": item["instance_id"],
        }
        row.update(metrics_from_scores(y_all[start:end], score_all[start:end], threshold))
        rows.append(row)
    return rows


def macro_scene_score(scene_rows, metric):
    return float(np.nanmean([row[metric] for row in scene_rows]))


def choose_threshold(y_val, score_val, val_slices):
    best = None
    curve = []
    for threshold in np.linspace(0.05, 0.95, 91):
        rows = per_scene_metrics(val_slices, y_val, score_val, float(threshold))
        entry = {
            "threshold": float(threshold),
            "macro_scene_iou": macro_scene_score(rows, "iou"),
            "macro_scene_f1": macro_scene_score(rows, "f1"),
        }
        curve.append(entry)
        if best is None or (entry["macro_scene_iou"], entry["macro_scene_f1"]) > (
            best["macro_scene_iou"],
            best["macro_scene_f1"],
        ):
            best = entry
    return best, curve


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def color_prediction(scores, threshold):
    scores = np.asarray(scores, dtype=np.float32)
    rgb = np.zeros((len(scores), 3), dtype=np.float32)
    rgb[:, 0] = np.maximum(scores, 0.08)
    rgb[:, 1] = 0.08 * (1.0 - scores)
    rgb[:, 2] = 1.0 - scores
    uncertain = np.abs(scores - threshold) < 0.05
    rgb[uncertain] = np.array([1.0, 0.85, 0.05], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def color_ground_truth(labels):
    labels = np.asarray(labels).astype(bool)
    rgb = np.full((len(labels), 3), np.array([0.12, 0.18, 0.85], dtype=np.float32))
    rgb[labels] = np.array([1.0, 0.03, 0.02], dtype=np.float32)
    return rgb


def write_colored_ply(source_ply, out_ply, rgb):
    ply = PlyData.read(source_ply)
    vertices = np.array(ply["vertex"].data, copy=True)
    if len(vertices) != len(rgb):
        raise ValueError(f"Length mismatch for {source_ply}: ply={len(vertices)} rgb={len(rgb)}")
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(out_ply)


def choose_visual_rows(per_scene, max_count):
    ordered = sorted(per_scene, key=lambda row: row["iou"])
    picks = [
        ordered[0],
        ordered[len(ordered) // 2],
        ordered[-1],
        max(per_scene, key=lambda row: row["fp_percent"]),
        max(per_scene, key=lambda row: row["fn_percent"]),
    ]
    unique = []
    seen = set()
    for row in picks + ordered:
        if row["scene_key"] in seen:
            continue
        unique.append(row)
        seen.add(row["scene_key"])
        if len(unique) >= max_count:
            break
    return unique


def export_selected_plys(out_dir, heldout_category, model_name, test_slices, y_test, test_score, per_scene, threshold, max_count):
    object_ply_root = FEATURE_ROOT / "work" / "object_ply"
    pred_root = out_dir / "selected_prediction_ply"
    rows = []
    slice_by_scene = {item["scene_key"]: (item, start, end) for item, start, end in test_slices}
    for row in choose_visual_rows(per_scene, max_count):
        scene_key = row["scene_key"]
        _, start, end = slice_by_scene[scene_key]
        source_ply = object_ply_root / f"{scene_key}_object_pruned_thr0.60.ply"
        pred_ply = pred_root / model_name / f"{scene_key}_{model_name}_prediction.ply"
        gt_ply = pred_root / "ground_truth_handle_red" / f"{scene_key}_gt_handle_red.ply"
        write_colored_ply(source_ply, pred_ply, color_prediction(test_score[start:end], threshold))
        if not gt_ply.exists():
            write_colored_ply(source_ply, gt_ply, color_ground_truth(y_test[start:end]))
        rows.append(
            {
                "heldout_category": heldout_category,
                "model": model_name,
                "scene_key": scene_key,
                "iou": row["iou"],
                "f1": row["f1"],
                "fp_percent": row["fp_percent"],
                "fn_percent": row["fn_percent"],
                "prediction_ply": str(pred_ply),
                "ground_truth_ply": str(gt_ply),
            }
        )
    return rows


def train_one(heldout_category, model_name, model_cfg, split, args, device):
    out_dir = args.out_root / heldout_category / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_set = model_cfg["feature_set"]
    context_mode = model_cfg["context_mode"]
    print(f"[{heldout_category}/{model_name}] loading arrays")
    x_train, y_train, _ = load_split_arrays(split["train"], feature_set, context_mode, args.k_neighbors)
    x_val, y_val, val_slices = load_split_arrays(split["val"], feature_set, context_mode, args.k_neighbors)
    x_test, y_test, test_slices = load_split_arrays(split["test"], feature_set, context_mode, args.k_neighbors)

    mean, std = standardize_fit(x_train)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

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
        num_workers=0,
        pin_memory=False,
    )

    best = None
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
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

        val_score = predict_scores(model, x_val, args.batch_size, device)
        th_best, _ = choose_threshold(y_val, val_score, val_slices)
        val_rows = per_scene_metrics(val_slices, y_val, val_score, th_best["threshold"])
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_scene_iou": macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": macro_scene_score(val_rows, "f1"),
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
                f"[{heldout_category}/{model_name}] epoch {epoch:03d}/{args.epochs} "
                f"loss={record['loss']:.4f} val_iou={record['val_macro_scene_iou']:.3f} "
                f"val_f1={record['val_macro_scene_f1']:.3f} th={record['val_threshold']:.2f}"
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{heldout_category}/{model_name}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    val_score = predict_scores(model, x_val, args.batch_size, device)
    th_best, threshold_curve = choose_threshold(y_val, val_score, val_slices)
    threshold = th_best["threshold"]
    test_score = predict_scores(model, x_test, args.batch_size, device)
    test_rows = per_scene_metrics(test_slices, y_test, test_score, threshold)
    micro = metrics_from_scores(y_test, test_score, threshold)
    macro = {
        key: macro_scene_score(test_rows, key)
        for key in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "iou",
            "gt_handle_ratio",
            "pred_handle_ratio",
            "correct_percent",
            "fp_percent",
            "fn_percent",
        ]
    }
    overall = {
        "heldout_category": heldout_category,
        "model": model_name,
        "feature_set": feature_set,
        "context_mode": context_mode,
        "k_neighbors": args.k_neighbors if context_mode == "knn" else None,
        "input_dim": int(x_train.shape[1]),
        "train_scenes": len(split["train"]),
        "val_scenes": len(split["val"]),
        "test_scenes": len(split["test"]),
        "train_gaussians": int(len(y_train)),
        "val_gaussians": int(len(y_val)),
        "test_gaussians": int(len(y_test)),
        "train_handle_ratio": float(y_train.mean()),
        "val_handle_ratio": float(y_val.mean()),
        "test_handle_ratio": float(y_test.mean()),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(threshold),
        "selection_metric": "validation macro scene IoU, tie-broken by validation macro scene F1",
        "micro": micro,
        "macro_scene_average": macro,
        "best_scene_iou": max(test_rows, key=lambda r: r["iou"])["scene_key"],
        "worst_scene_iou": min(test_rows, key=lambda r: r["iou"])["scene_key"],
    }

    save_csv(out_dir / "history.csv", history)
    save_csv(out_dir / "threshold_curve.csv", threshold_curve)
    save_csv(out_dir / "per_scene_metrics.csv", test_rows)
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "feature_set": feature_set,
            "context_mode": context_mode,
            "input_dim": int(x_train.shape[1]),
            "threshold": float(threshold),
            "overall": overall,
        },
        out_dir / "model.pt",
    )
    offsets = np.asarray([[start, end] for _, start, end in test_slices], dtype=np.int64)
    scene_keys = np.asarray([item["scene_key"] for item, _, _ in test_slices])
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        scene_keys=scene_keys,
        offsets=offsets,
        scores=test_score.astype(np.float32),
        labels=y_test.astype(np.uint8),
    )
    visual_rows = export_selected_plys(
        out_dir=args.out_root / heldout_category,
        heldout_category=heldout_category,
        model_name=model_name,
        test_slices=test_slices,
        y_test=y_test,
        test_score=test_score,
        per_scene=test_rows,
        threshold=threshold,
        max_count=args.num_visual_plys,
    )
    save_csv(out_dir / "selected_prediction_plys.csv", visual_rows)
    print(
        f"[{heldout_category}/{model_name}] TEST macro IoU={macro['iou']:.3f} "
        f"macro F1={macro['f1']:.3f} pred_ratio={macro['pred_handle_ratio']:.3f} "
        f"gt_ratio={macro['gt_handle_ratio']:.3f} threshold={threshold:.2f}"
    )
    return overall


def write_split_summary(split, path):
    rows = []
    for split_name, items in split.items():
        by_cat = defaultdict(list)
        for item in items:
            by_cat[item["category"]].append(item)
        for category, group in sorted(by_cat.items()):
            rows.append(
                {
                    "split": split_name,
                    "category": category,
                    "num_scenes": len(group),
                    "num_instances": len({x["instance_id"] for x in group}),
                    "num_gaussians": sum(x["num_gaussians"] for x in group),
                    "num_handle": sum(x["num_handle"] for x in group),
                    "handle_ratio": sum(x["num_handle"] for x in group)
                    / max(sum(x["num_gaussians"] for x in group), 1),
                }
            )
    save_csv(path, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--heldout_categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument("--num_visual_plys", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)

    items = load_all_items(args.feature_root)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    all_results = []
    for heldout_category in args.heldout_categories:
        split = build_loco_split(items, heldout_category, args.seed)
        heldout_dir = args.out_root / heldout_category
        heldout_dir.mkdir(parents=True, exist_ok=True)
        with open(heldout_dir / "split_manifest.json", "w") as f:
            json.dump(
                {
                    "heldout_category": heldout_category,
                    "split_rule": "test is all held-out category scenes; remaining categories split 85/15 by instance",
                    "splits": split,
                },
                f,
                indent=2,
            )
        write_split_summary(split, heldout_dir / "split_summary.csv")
        print(
            f"[{heldout_category}] split scenes: train={len(split['train'])} "
            f"val={len(split['val'])} test={len(split['test'])}"
        )
        category_results = {}
        for model_name in args.models:
            overall = train_one(heldout_category, model_name, MODELS[model_name], split, args, device)
            category_results[model_name] = overall
            all_results.append(overall)
        with open(heldout_dir / "all_models_summary.json", "w") as f:
            json.dump(category_results, f, indent=2)

    flat_rows = []
    for result in all_results:
        macro = result["macro_scene_average"]
        micro = result["micro"]
        flat_rows.append(
            {
                "heldout_category": result["heldout_category"],
                "model": result["model"],
                "test_scenes": result["test_scenes"],
                "selected_epoch": result["selected_epoch"],
                "selected_threshold": result["selected_threshold"],
                "macro_iou": macro["iou"],
                "macro_f1": macro["f1"],
                "macro_precision": macro["precision"],
                "macro_recall": macro["recall"],
                "macro_gt_handle_ratio": macro["gt_handle_ratio"],
                "macro_pred_handle_ratio": macro["pred_handle_ratio"],
                "micro_iou": micro["iou"],
                "micro_f1": micro["f1"],
                "micro_correct_percent": micro["correct_percent"],
                "worst_scene_iou": result["worst_scene_iou"],
                "best_scene_iou": result["best_scene_iou"],
            }
        )
    partial_name = "_".join(args.heldout_categories)
    save_csv(args.out_root / f"leave_one_category_summary_partial_{partial_name}.csv", flat_rows)
    with open(args.out_root / f"all_loco_results_partial_{partial_name}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
