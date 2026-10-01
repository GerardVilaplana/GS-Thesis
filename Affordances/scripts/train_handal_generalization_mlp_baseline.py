import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_ROOT = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features"
)
OUT_ROOT = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/"
    "01_per_gaussian_mlp_baseline"
)


def stable_key(text, seed):
    h = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def read_npz_meta(path):
    stem = path.stem
    category, scene_id = stem.rsplit("__", 1)
    instance_id = scene_id[:3]
    with np.load(path, allow_pickle=False) as z:
        n = int(len(z["handle_labels_thr0_25"]))
        pos = int(z["handle_labels_thr0_25"].sum())
        valid = int(z["valid_dino"].sum())
    return {
        "scene_key": stem,
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
    return [read_npz_meta(p) for p in paths]


def split_groups_by_scene_count(groups, ratios, seed):
    names = sorted(groups)
    names.sort(key=lambda name: stable_key(name, seed))
    total = sum(len(groups[name]) for name in names)
    target_val = int(round(total * ratios["val"]))
    target_test = int(round(total * ratios["test"]))
    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for name in names:
        group = groups[name]
        if ratios["test"] > 0 and counts["test"] < target_test:
            split = "test"
        elif ratios["val"] > 0 and counts["val"] < target_val:
            split = "val"
        else:
            split = "train"
        assigned[split].extend(group)
        counts[split] += len(group)
    return assigned


def build_exp1a_split(items, seed):
    by_category_instance = defaultdict(list)
    for item in items:
        key = f"{item['category']}::{item['instance_id']}"
        by_category_instance[key].append(item)

    by_category = defaultdict(dict)
    for key, group in by_category_instance.items():
        category, _ = key.split("::", 1)
        by_category[category][key] = sorted(group, key=lambda x: x["scene_id"])

    split = {"train": [], "val": [], "test": []}
    for category in sorted(by_category):
        part = split_groups_by_scene_count(
            by_category[category],
            ratios={"train": 0.75, "val": 0.10, "test": 0.15},
            seed=seed + stable_key(category, seed),
        )
        for name in split:
            split[name].extend(part[name])
    return {k: sorted(v, key=lambda x: x["scene_key"]) for k, v in split.items()}


def build_exp1b_split(items, seed, holdout_categories):
    holdout = set(holdout_categories)
    split = {"train": [], "val": [], "test": []}
    remaining_groups = defaultdict(list)
    for item in items:
        if item["category"] in holdout:
            split["test"].append(item)
        else:
            key = f"{item['category']}::{item['instance_id']}"
            remaining_groups[key].append(item)

    by_category = defaultdict(dict)
    for key, group in remaining_groups.items():
        category, _ = key.split("::", 1)
        by_category[category][key] = sorted(group, key=lambda x: x["scene_id"])

    for category in sorted(by_category):
        part = split_groups_by_scene_count(
            by_category[category],
            ratios={"train": 0.85, "val": 0.15, "test": 0.0},
            seed=seed + stable_key(category, seed),
        )
        split["train"].extend(part["train"])
        split["val"].extend(part["val"])
    return {k: sorted(v, key=lambda x: x["scene_key"]) for k, v in split.items()}


def write_split_files(split, out_dir, experiment, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": experiment,
        "seed": seed,
        "split_rule": (
            "1A: deterministic category+instance split with 75/10/15 scene targets"
            if experiment == "exp1a_seen_instance"
            else "1B: mugs+screwdrivers held out; remaining categories split 85/15 by instance"
        ),
        "splits": split,
    }
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    rows = []
    for split_name, items in split.items():
        by_cat = defaultdict(list)
        for item in items:
            by_cat[item["category"]].append(item)
        for category in sorted(by_cat):
            group = by_cat[category]
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
    with open(out_dir / "split_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sigmoid_np(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_scene_arrays(item, feature_set):
    with np.load(item["npz"], allow_pickle=False) as z:
        parts = []
        if feature_set["dino"]:
            parts.append(z["dino_features"].astype(np.float32))
        if feature_set["geometry"]:
            parts.append(z["geometry_features"].astype(np.float32))
        if feature_set["color"]:
            parts.append(z["color"].astype(np.float32))
        x = np.concatenate(parts, axis=1).astype(np.float32)
        y = z["handle_labels_thr0_25"].astype(np.float32)
    return x, y


def load_split_arrays(items, feature_set):
    xs, ys, scene_slices = [], [], []
    offset = 0
    for item in items:
        x, y = load_scene_arrays(item, feature_set)
        xs.append(x)
        ys.append(y)
        scene_slices.append((item, offset, offset + len(y)))
        offset += len(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), scene_slices


def standardize_fit(x):
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


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
    accuracy = (tp + tn) / n
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "num_gaussians": int(n),
        "gt_handle_ratio": float(y_true.mean()) if n else 0.0,
        "pred_handle_ratio": float(y_pred.mean()) if n else 0.0,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
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


def per_scene_metrics(items, slices, y_all, score_all, threshold):
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


def macro_scene_score(scene_rows, metric):
    return float(np.nanmean([row[metric] for row in scene_rows]))


def choose_threshold(y_val, score_val, val_slices):
    best = None
    curve = []
    for threshold in np.linspace(0.05, 0.95, 91):
        rows = per_scene_metrics([], val_slices, y_val, score_val, float(threshold))
        mean_iou = macro_scene_score(rows, "iou")
        mean_f1 = macro_scene_score(rows, "f1")
        entry = {
            "threshold": float(threshold),
            "macro_scene_iou": mean_iou,
            "macro_scene_f1": mean_f1,
        }
        curve.append(entry)
        if best is None or (mean_iou, mean_f1) > (best["macro_scene_iou"], best["macro_scene_f1"]):
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


def train_baseline(name, feature_set, split, args, device):
    out_dir = args.out_root / args.experiment / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.experiment}/{name}] loading arrays")
    x_train, y_train, _ = load_split_arrays(split["train"], feature_set)
    x_val, y_val, val_slices = load_split_arrays(split["val"], feature_set)
    x_test, y_test, test_slices = load_split_arrays(split["test"], feature_set)

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
    history = []
    stale_epochs = 0
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
        threshold_best, _ = choose_threshold(y_val, val_score, val_slices)
        val_rows = per_scene_metrics([], val_slices, y_val, val_score, threshold_best["threshold"])
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_scene_iou": macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": macro_scene_score(val_rows, "f1"),
            "val_threshold": threshold_best["threshold"],
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
                f"[{args.experiment}/{name}] epoch {epoch:03d}/{args.epochs} "
                f"loss={record['loss']:.4f} val_iou={record['val_macro_scene_iou']:.3f} "
                f"val_f1={record['val_macro_scene_f1']:.3f} th={record['val_threshold']:.2f}"
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{args.experiment}/{name}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    val_score = predict_scores(model, x_val, args.batch_size, device)
    threshold_best, threshold_curve = choose_threshold(y_val, val_score, val_slices)
    test_score = predict_scores(model, x_test, args.batch_size, device)
    threshold = threshold_best["threshold"]
    test_rows = per_scene_metrics(split["test"], test_slices, y_test, test_score, threshold)
    category_rows = per_category_metrics(test_rows)
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
        "experiment": args.experiment,
        "baseline": name,
        "feature_set": feature_set,
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
    save_csv(out_dir / "per_category_metrics.csv", category_rows)
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "feature_set": feature_set,
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
        f"[{args.experiment}/{name}] TEST macro IoU={macro['iou']:.3f} "
        f"macro F1={macro['f1']:.3f} micro correct={100 * micro['correct_percent']:.1f}% "
        f"threshold={threshold:.2f}"
    )
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--experiment",
        choices=["exp1a_seen_instance", "exp1b_holdout_mugs_screwdrivers"],
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    items = load_all_items(args.feature_root)
    if args.experiment == "exp1a_seen_instance":
        split = build_exp1a_split(items, args.seed)
    else:
        split = build_exp1b_split(items, args.seed, holdout_categories=["mugs", "screwdrivers"])

    exp_dir = args.out_root / args.experiment
    write_split_files(split, exp_dir, args.experiment, args.seed)
    print(
        f"[{args.experiment}] split scenes: "
        f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    baselines = {
        "geometry_only": {"dino": False, "geometry": True, "color": False},
        "dino_only": {"dino": True, "geometry": False, "color": False},
        "dino_geometry_color": {"dino": True, "geometry": True, "color": True},
    }
    all_results = {}
    for name, feature_set in baselines.items():
        all_results[name] = train_baseline(name, feature_set, split, args, device)
    with open(exp_dir / "all_baselines_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
