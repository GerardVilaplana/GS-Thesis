import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
DEFAULT_BASELINE_ROOT = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "01_per_gaussian_mlp_baseline"
)
DEFAULT_OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "02_context_models_v1"


FEATURE_SETS = {
    "geometry_color": {"dino": False, "geometry": True, "color": True},
    "dino_only": {"dino": True, "geometry": False, "color": False},
    "dino_geometry_color": {"dino": True, "geometry": True, "color": True},
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sigmoid_np(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_xyz(xyz):
    xyz = xyz.astype(np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    radius = np.linalg.norm(centered, axis=1).max()
    radius = max(float(radius), 1e-6)
    return centered / radius


def load_base_scene(item, feature_set):
    with np.load(item["npz"], allow_pickle=False) as z:
        parts = []
        xyz_norm = normalize_xyz(z["xyz"].astype(np.float32))
        if feature_set["dino"]:
            parts.append(z["dino_features"].astype(np.float32))
        if feature_set["geometry"]:
            geom = z["geometry_features"].astype(np.float32)
            geom = geom.copy()
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
    if context_mode == "global":
        global_mean = np.repeat(x.mean(axis=0, keepdims=True), len(x), axis=0)
        return np.concatenate([x, global_mean], axis=1).astype(np.float32)
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


def per_category_metrics(scene_rows):
    by_category = defaultdict(list)
    for row in scene_rows:
        by_category[row["category"]].append(row)
    out = []
    metric_names = [
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
    for category, rows in sorted(by_category.items()):
        merged = {
            "category": category,
            "num_scenes": len(rows),
            "num_gaussians": sum(r["num_gaussians"] for r in rows),
        }
        for name in metric_names:
            merged[f"macro_scene_{name}"] = float(np.nanmean([r[name] for r in rows]))
        out.append(merged)
    return out


def train_one(experiment, context_mode, feature_name, feature_set, split, args, device):
    out_dir = args.out_root / experiment / context_mode / feature_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{experiment}/{context_mode}/{feature_name}] loading arrays")
    x_train, y_train, _ = load_split_arrays(
        split["train"], feature_set, context_mode, args.k_neighbors
    )
    x_val, y_val, val_slices = load_split_arrays(
        split["val"], feature_set, context_mode, args.k_neighbors
    )
    x_test, y_test, test_slices = load_split_arrays(
        split["test"], feature_set, context_mode, args.k_neighbors
    )

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
                f"[{experiment}/{context_mode}/{feature_name}] epoch {epoch:03d}/{args.epochs} "
                f"loss={record['loss']:.4f} val_iou={record['val_macro_scene_iou']:.3f} "
                f"val_f1={record['val_macro_scene_f1']:.3f} th={record['val_threshold']:.2f}"
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{experiment}/{context_mode}/{feature_name}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    val_score = predict_scores(model, x_val, args.batch_size, device)
    th_best, threshold_curve = choose_threshold(y_val, val_score, val_slices)
    threshold = th_best["threshold"]
    test_score = predict_scores(model, x_test, args.batch_size, device)
    test_rows = per_scene_metrics(test_slices, y_test, test_score, threshold)
    cat_rows = per_category_metrics(test_rows)
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
        "experiment": experiment,
        "context_mode": context_mode,
        "feature_name": feature_name,
        "feature_set": feature_set,
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
    save_csv(out_dir / "per_category_metrics.csv", cat_rows)
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
    print(
        f"[{experiment}/{context_mode}/{feature_name}] TEST macro IoU={macro['iou']:.3f} "
        f"macro F1={macro['f1']:.3f} micro correct={100 * micro['correct_percent']:.1f}% "
        f"threshold={threshold:.2f}"
    )
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--experiment",
        choices=["exp1a_seen_instance", "exp1b_holdout_mugs_screwdrivers"],
        required=True,
    )
    parser.add_argument("--context_modes", nargs="+", default=["global", "knn"])
    parser.add_argument("--feature_names", nargs="+", default=list(FEATURE_SETS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split_path = args.baseline_root / args.experiment / "split_manifest.json"
    manifest = load_json(split_path)
    split = manifest["splits"]
    exp_out_dir = args.out_root / args.experiment
    exp_out_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_out_dir / "source_split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    all_results = {}
    for context_mode in args.context_modes:
        all_results[context_mode] = {}
        for feature_name in args.feature_names:
            all_results[context_mode][feature_name] = train_one(
                args.experiment,
                context_mode,
                feature_name,
                FEATURE_SETS[feature_name],
                split,
                args,
                device,
            )
    with open(exp_out_dir / "all_context_baselines_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
