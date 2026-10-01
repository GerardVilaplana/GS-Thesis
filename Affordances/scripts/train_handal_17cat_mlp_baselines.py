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


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
OLD_FEATURE_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
NEW_FEATURE_ROOT = BASE_ROOT / "data" / "handal_new_categories_delta_v1_features"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "08_17category_mlp_baselines_v1"

FEATURE_SETS = {
    "xyzrgb": {"xyz": True, "dino": False, "geometry": False, "color": True},
    "geometry_color": {"xyz": False, "dino": False, "geometry": True, "color": True},
    "dino_geometry_color": {"xyz": False, "dino": True, "geometry": True, "color": True},
}


def stable_key(text, seed):
    h = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def sigmoid_np(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_xyz(xyz):
    xyz = xyz.astype(np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    return centered / radius


def read_npz_meta(path):
    category, scene_id = path.stem.rsplit("__", 1)
    with np.load(path, allow_pickle=False) as z:
        instance_id = str(z["instance_id"]) if "instance_id" in z.files else scene_id[:3]
        n = int(len(z["handle_labels_thr0_25"]))
        pos = int(z["handle_labels_thr0_25"].sum())
        valid = int(z["valid_dino"].sum())
        source_split = str(z["split"]) if "split" in z.files else "unknown"
    return {
        "scene_key": path.stem,
        "category": category,
        "scene_id": scene_id,
        "instance_id": instance_id,
        "npz": str(path),
        "source_split": source_split,
        "num_gaussians": n,
        "num_handle": pos,
        "handle_ratio": float(pos / max(n, 1)),
        "valid_dino": valid,
        "valid_dino_ratio": float(valid / max(n, 1)),
    }


def load_all_items(feature_roots):
    items = []
    seen_paths = set()
    for root in feature_roots:
        paths = sorted((root / "npz").glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz files found in {root / 'npz'}")
        for path in paths:
            if path.stem in seen_paths:
                raise ValueError(f"Duplicate scene key across feature roots: {path.stem}")
            seen_paths.add(path.stem)
            items.append(read_npz_meta(path))
    return sorted(items, key=lambda x: x["scene_key"])


def split_groups_by_scene_count(groups, ratios, seed):
    names = sorted(groups)
    names.sort(key=lambda name: stable_key(name, seed))
    total = sum(len(groups[name]) for name in names)
    target_val = int(round(total * ratios["val"]))
    target_test = int(round(total * ratios["test"]))
    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    for name in names:
        group = sorted(groups[name], key=lambda x: x["scene_id"])
        if ratios["test"] > 0 and counts["test"] < target_test:
            split = "test"
        elif ratios["val"] > 0 and counts["val"] < target_val:
            split = "val"
        else:
            split = "train"
        assigned[split].extend(group)
        counts[split] += len(group)
    return assigned


def build_seen_instance_split(items, seed):
    by_category_instance = defaultdict(list)
    for item in items:
        key = f"{item['category']}::{item['instance_id']}"
        by_category_instance[key].append(item)
    by_category = defaultdict(dict)
    for key, group in by_category_instance.items():
        category, _ = key.split("::", 1)
        by_category[category][key] = group
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


def load_scene_arrays(item, feature_set):
    with np.load(item["npz"], allow_pickle=False) as z:
        parts = []
        xyz_norm = normalize_xyz(z["xyz"].astype(np.float32))
        if feature_set["xyz"]:
            parts.append(xyz_norm)
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
    return x, y


def load_split_arrays(items, feature_set):
    xs, ys, slices = [], [], []
    offset = 0
    for item in items:
        x, y = load_scene_arrays(item, feature_set)
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
        hidden = 384 if in_dim >= 256 else 256
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
            "source_split": item["source_split"],
        }
        row.update(metrics_from_scores(y_all[start:end], score_all[start:end], threshold))
        rows.append(row)
    return rows


def per_category_metrics(scene_rows):
    by_category = defaultdict(list)
    for row in scene_rows:
        by_category[row["category"]].append(row)
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
    out = []
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


def train_one(run_name, model_name, feature_set, split, out_dir, args, device):
    model_dir = out_dir / model_name
    done_path = model_dir / "overall_metrics.json"
    if args.skip_existing and done_path.exists():
        print(f"[{run_name}/{model_name}] skip existing {done_path}", flush=True)
        with open(done_path, "r") as f:
            return json.load(f)

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{run_name}/{model_name}] loading arrays", flush=True)
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
                f"[{run_name}/{model_name}] epoch {epoch:03d}/{args.epochs} "
                f"loss={record['loss']:.4f} val_iou={record['val_macro_scene_iou']:.3f} "
                f"val_f1={record['val_macro_scene_f1']:.3f} th={record['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{run_name}/{model_name}] early stopping at epoch {epoch}", flush=True)
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
        "run_name": run_name,
        "model": model_name,
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

    save_csv(model_dir / "history.csv", history)
    save_csv(model_dir / "threshold_curve.csv", threshold_curve)
    save_csv(model_dir / "per_scene_metrics.csv", test_rows)
    save_csv(model_dir / "per_category_metrics.csv", cat_rows)
    with open(model_dir / "overall_metrics.json", "w") as f:
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
    print(
        f"[{run_name}/{model_name}] TEST macro IoU={macro['iou']:.3f} "
        f"macro F1={macro['f1']:.3f} micro correct={100 * micro['correct_percent']:.1f}% "
        f"threshold={threshold:.2f}",
        flush=True,
    )
    return overall


def write_dataset_summary(items, out_root):
    by_category = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)
    rows = []
    for category, group in sorted(by_category.items()):
        rows.append(
            {
                "category": category,
                "num_scenes": len(group),
                "num_instances": len({x["instance_id"] for x in group}),
                "num_gaussians": sum(x["num_gaussians"] for x in group),
                "num_handle": sum(x["num_handle"] for x in group),
                "handle_ratio": sum(x["num_handle"] for x in group)
                / max(sum(x["num_gaussians"] for x in group), 1),
                "mean_valid_dino_ratio": float(np.mean([x["valid_dino_ratio"] for x in group])),
            }
        )
    save_csv(out_root / "dataset_summary_by_category.csv", rows)


def write_flat_summary(path, results):
    rows = []
    for result in results:
        macro = result["macro_scene_average"]
        micro = result["micro"]
        rows.append(
            {
                "run_name": result["run_name"],
                "model": result["model"],
                "input_dim": result["input_dim"],
                "train_scenes": result["train_scenes"],
                "val_scenes": result["val_scenes"],
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
    save_csv(path, rows)


def parse_feature_roots(values):
    return [Path(v) for v in values]


def load_or_build_seen_split(items, args):
    split_dir = args.out_root / "01_seen_instance_17cat"
    manifest_path = split_dir / "split_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)["splits"]
    split = build_seen_instance_split(items, args.seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "experiment": "seen-category unseen-instance split across all 17 categories",
                "seed": args.seed,
                "split_rule": "deterministic category+instance split with 75/10/15 scene targets per category",
                "splits": split,
            },
            f,
            indent=2,
        )
    write_split_summary(split, split_dir / "split_summary.csv")
    return split


def aggregate_results(out_root):
    results = []
    for path in sorted(out_root.glob("**/overall_metrics.json")):
        with open(path, "r") as f:
            results.append(json.load(f))
    write_flat_summary(out_root / "all_finished_models_summary.csv", results)
    with open(out_root / "all_finished_models_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[aggregate] collected {len(results)} finished model results", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_roots",
        nargs="+",
        default=[str(OLD_FEATURE_ROOT), str(NEW_FEATURE_ROOT)],
    )
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--mode", choices=["seen", "loco", "aggregate"], required=True)
    parser.add_argument("--heldout_categories", nargs="+", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=list(FEATURE_SETS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "aggregate":
        aggregate_results(args.out_root)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    feature_roots = parse_feature_roots(args.feature_roots)
    items = load_all_items(feature_roots)
    write_dataset_summary(items, args.out_root)
    categories = sorted({item["category"] for item in items})
    with open(args.out_root / "feature_roots.json", "w") as f:
        json.dump({"feature_roots": [str(x) for x in feature_roots], "categories": categories}, f, indent=2)
    print(
        f"[dataset] scenes={len(items)} categories={len(categories)} "
        f"models={','.join(args.models)}",
        flush=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    all_results = []
    if args.mode == "seen":
        split = load_or_build_seen_split(items, args)
        out_dir = args.out_root / "01_seen_instance_17cat"
        print(
            f"[seen] split scenes: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}",
            flush=True,
        )
        for model_name in args.models:
            all_results.append(
                train_one("01_seen_instance_17cat", model_name, FEATURE_SETS[model_name], split, out_dir, args, device)
            )
        write_flat_summary(out_dir / "seen_instance_summary.csv", all_results)
        with open(out_dir / "seen_instance_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        heldout = args.heldout_categories or categories
        heldout = [c for c in heldout if c in categories]
        heldout = [c for i, c in enumerate(sorted(heldout)) if i % args.num_shards == args.shard]
        print(f"[loco shard {args.shard}/{args.num_shards}] heldout={heldout}", flush=True)
        for heldout_category in heldout:
            split = build_loco_split(items, heldout_category, args.seed)
            out_dir = args.out_root / "02_leave_one_category_out_17cat" / heldout_category
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "split_manifest.json", "w") as f:
                json.dump(
                    {
                        "heldout_category": heldout_category,
                        "seed": args.seed,
                        "split_rule": "test is all held-out category scenes; remaining categories split 85/15 by instance",
                        "splits": split,
                    },
                    f,
                    indent=2,
                )
            write_split_summary(split, out_dir / "split_summary.csv")
            print(
                f"[{heldout_category}] split scenes: "
                f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}",
                flush=True,
            )
            category_results = []
            for model_name in args.models:
                category_results.append(
                    train_one(heldout_category, model_name, FEATURE_SETS[model_name], split, out_dir, args, device)
                )
            write_flat_summary(out_dir / "category_summary.csv", category_results)
            with open(out_dir / "category_summary.json", "w") as f:
                json.dump(category_results, f, indent=2)
            all_results.extend(category_results)
        write_flat_summary(
            args.out_root / f"loco_summary_shard_{args.shard}_of_{args.num_shards}.csv",
            all_results,
        )
        with open(args.out_root / f"loco_summary_shard_{args.shard}_of_{args.num_shards}.json", "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
