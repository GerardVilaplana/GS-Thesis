import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


C0 = 0.28209479177387814
BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SPLIT_PATH = (
    BASE_ROOT
    / "outputs"
    / "03_handle_generalization"
    / "01_per_gaussian_mlp_baseline"
    / "exp1a_seen_instance"
    / "split_manifest.json"
)
FEATURE_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_features"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "04_mug_incremental_category_interference_v1"

CATEGORY_ORDER = [
    "mugs",
    "pots_pans",
    "measuring_cups",
    "whisks",
    "spatulas",
    "hammers",
    "adjustable_wrenches",
    "screwdrivers",
    "power_drills",
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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sigmoid_np(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def normalize_xyz(xyz):
    xyz = xyz.astype(np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    return centered / radius


def with_npz_paths(items):
    fixed = []
    for item in items:
        copied = dict(item)
        copied["npz"] = str(FEATURE_ROOT / "npz" / f"{item['scene_key']}.npz")
        fixed.append(copied)
    return fixed


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


def summarize_rows(rows):
    keys = [
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
    return {key: macro_scene_score(rows, key) for key in keys}


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


def export_selected_plys(out_dir, model_name, test_slices, y_test, test_score, per_scene, threshold, max_count):
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
                "scene_key": scene_key,
                "model": model_name,
                "iou": row["iou"],
                "f1": row["f1"],
                "fp_percent": row["fp_percent"],
                "fn_percent": row["fn_percent"],
                "prediction_ply": str(pred_ply),
                "ground_truth_ply": str(gt_ply),
            }
        )
    return rows


def split_by_categories(split, categories):
    category_set = set(categories)
    return {
        "train": with_npz_paths([x for x in split["train"] if x["category"] in category_set]),
        "val": with_npz_paths([x for x in split["val"] if x["category"] in category_set]),
        "test": with_npz_paths([x for x in split["test"] if x["category"] in category_set]),
    }


def train_one(step_name, included_categories, model_name, model_cfg, split, args, device):
    out_dir = args.out_root / step_name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_set = model_cfg["feature_set"]
    context_mode = model_cfg["context_mode"]
    step_split = split_by_categories(split, included_categories)
    mug_val_items = with_npz_paths([x for x in split["val"] if x["category"] == "mugs"])
    mug_test_items = with_npz_paths([x for x in split["test"] if x["category"] == "mugs"])

    print(f"[{step_name}/{model_name}] loading arrays")
    x_train, y_train, _ = load_split_arrays(step_split["train"], feature_set, context_mode, args.k_neighbors)
    x_val, y_val, val_slices = load_split_arrays(step_split["val"], feature_set, context_mode, args.k_neighbors)
    x_mug_val, y_mug_val, mug_val_slices = load_split_arrays(mug_val_items, feature_set, context_mode, args.k_neighbors)
    x_mug_test, y_mug_test, mug_test_slices = load_split_arrays(mug_test_items, feature_set, context_mode, args.k_neighbors)

    mean, std = standardize_fit(x_train)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_mug_val = (x_mug_val - mean) / std
    x_mug_test = (x_mug_test - mean) / std

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
        mug_val_score = predict_scores(model, x_mug_val, args.batch_size, device)
        mug_val_rows = per_scene_metrics(mug_val_slices, y_mug_val, mug_val_score, th_best["threshold"])
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "mixed_val_macro_scene_iou": macro_scene_score(val_rows, "iou"),
            "mixed_val_macro_scene_f1": macro_scene_score(val_rows, "f1"),
            "mug_val_macro_scene_iou_at_mixed_threshold": macro_scene_score(mug_val_rows, "iou"),
            "mug_val_macro_scene_f1_at_mixed_threshold": macro_scene_score(mug_val_rows, "f1"),
            "mixed_val_threshold": th_best["threshold"],
        }
        history.append(record)
        score_tuple = (record["mixed_val_macro_scene_iou"], record["mixed_val_macro_scene_f1"])
        if best is None or score_tuple > (best["mixed_val_macro_scene_iou"], best["mixed_val_macro_scene_f1"]):
            best = dict(record)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[{step_name}/{model_name}] epoch {epoch:03d}/{args.epochs} "
                f"loss={record['loss']:.4f} mixed_val_iou={record['mixed_val_macro_scene_iou']:.3f} "
                f"mug_val_iou={record['mug_val_macro_scene_iou_at_mixed_threshold']:.3f} "
                f"th={record['mixed_val_threshold']:.2f}"
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{step_name}/{model_name}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    val_score = predict_scores(model, x_val, args.batch_size, device)
    th_best, threshold_curve = choose_threshold(y_val, val_score, val_slices)
    threshold = th_best["threshold"]
    mixed_val_rows = per_scene_metrics(val_slices, y_val, val_score, threshold)

    mug_val_score = predict_scores(model, x_mug_val, args.batch_size, device)
    mug_val_rows = per_scene_metrics(mug_val_slices, y_mug_val, mug_val_score, threshold)
    mug_test_score = predict_scores(model, x_mug_test, args.batch_size, device)
    mug_test_rows = per_scene_metrics(mug_test_slices, y_mug_test, mug_test_score, threshold)

    mixed_val_macro = summarize_rows(mixed_val_rows)
    mug_val_macro = summarize_rows(mug_val_rows)
    mug_test_macro = summarize_rows(mug_test_rows)
    mug_test_micro = metrics_from_scores(y_mug_test, mug_test_score, threshold)

    overall = {
        "step_name": step_name,
        "included_categories": included_categories,
        "num_included_categories": len(included_categories),
        "model": model_name,
        "feature_set": feature_set,
        "context_mode": context_mode,
        "k_neighbors": args.k_neighbors if context_mode == "knn" else None,
        "input_dim": int(x_train.shape[1]),
        "train_scenes": len(step_split["train"]),
        "mixed_val_scenes": len(step_split["val"]),
        "mug_val_scenes": len(mug_val_items),
        "mug_test_scenes": len(mug_test_items),
        "train_gaussians": int(len(y_train)),
        "mixed_val_gaussians": int(len(y_val)),
        "mug_test_gaussians": int(len(y_mug_test)),
        "train_handle_ratio": float(y_train.mean()),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(threshold),
        "selection_metric": "mixed validation macro scene IoU, tie-broken by mixed validation macro scene F1",
        "mixed_val_macro_scene_average": mixed_val_macro,
        "mug_val_macro_scene_average": mug_val_macro,
        "mug_test_macro_scene_average": mug_test_macro,
        "mug_test_micro": mug_test_micro,
        "mug_test_best_scene_iou": max(mug_test_rows, key=lambda r: r["iou"])["scene_key"],
        "mug_test_worst_scene_iou": min(mug_test_rows, key=lambda r: r["iou"])["scene_key"],
    }

    save_csv(out_dir / "history.csv", history)
    save_csv(out_dir / "threshold_curve.csv", threshold_curve)
    save_csv(out_dir / "mixed_val_per_scene_metrics.csv", mixed_val_rows)
    save_csv(out_dir / "mug_val_per_scene_metrics.csv", mug_val_rows)
    save_csv(out_dir / "mug_test_per_scene_metrics.csv", mug_test_rows)
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
    offsets = np.asarray([[start, end] for _, start, end in mug_test_slices], dtype=np.int64)
    scene_keys = np.asarray([item["scene_key"] for item, _, _ in mug_test_slices])
    np.savez_compressed(
        out_dir / "mug_test_predictions.npz",
        scene_keys=scene_keys,
        offsets=offsets,
        scores=mug_test_score.astype(np.float32),
        labels=y_mug_test.astype(np.uint8),
    )
    visual_rows = export_selected_plys(
        out_dir=args.out_root / step_name,
        model_name=model_name,
        test_slices=mug_test_slices,
        y_test=y_mug_test,
        test_score=mug_test_score,
        per_scene=mug_test_rows,
        threshold=threshold,
        max_count=args.num_visual_plys,
    )
    save_csv(out_dir / "selected_prediction_plys.csv", visual_rows)
    print(
        f"[{step_name}/{model_name}] MUG TEST macro IoU={mug_test_macro['iou']:.3f} "
        f"macro F1={mug_test_macro['f1']:.3f} pred_ratio={mug_test_macro['pred_handle_ratio']:.3f} "
        f"gt_ratio={mug_test_macro['gt_handle_ratio']:.3f} threshold={threshold:.2f}"
    )
    return overall


def write_step_manifest(step_dir, step_name, included_categories, split):
    rows = []
    for split_name in ["train", "val", "test"]:
        for category in included_categories:
            group = [x for x in split[split_name] if x["category"] == category]
            rows.append(
                {
                    "step_name": step_name,
                    "split": split_name,
                    "category": category,
                    "num_scenes": len(group),
                    "num_instances": len({x["instance_id"] for x in group}),
                }
            )
    save_csv(step_dir / "step_split_summary.csv", rows)
    with open(step_dir / "step_manifest.json", "w") as f:
        json.dump(
            {
                "step_name": step_name,
                "included_categories": included_categories,
                "category_addition_order": CATEGORY_ORDER,
                "split_source": str(SPLIT_PATH),
                "selection_rule": "mixed validation over all included categories",
                "primary_analysis": "fixed mug test split from source 1A split",
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--step_indices", nargs="+", type=int, default=list(range(len(CATEGORY_ORDER))))
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260715)
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

    manifest = load_json(args.split_path)
    split = manifest["splits"]
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    all_results = []
    for step_idx in args.step_indices:
        included_categories = CATEGORY_ORDER[: step_idx + 1]
        step_name = f"step{step_idx:02d}_" + "__".join(included_categories)
        step_dir = args.out_root / step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        write_step_manifest(step_dir, step_name, included_categories, split)
        print(f"[{step_name}] categories={included_categories}")
        step_results = {}
        for model_name in args.models:
            overall = train_one(step_name, included_categories, model_name, MODELS[model_name], split, args, device)
            step_results[model_name] = overall
            all_results.append(overall)
        with open(step_dir / "all_models_summary.json", "w") as f:
            json.dump(step_results, f, indent=2)

    partial_name = "_".join([f"{i:02d}" for i in args.step_indices])
    flat_rows = []
    for result in all_results:
        mixed = result["mixed_val_macro_scene_average"]
        mug_val = result["mug_val_macro_scene_average"]
        mug_test = result["mug_test_macro_scene_average"]
        micro = result["mug_test_micro"]
        flat_rows.append(
            {
                "step_name": result["step_name"],
                "num_included_categories": result["num_included_categories"],
                "included_categories": "|".join(result["included_categories"]),
                "model": result["model"],
                "selected_epoch": result["selected_epoch"],
                "selected_threshold": result["selected_threshold"],
                "mixed_val_macro_iou": mixed["iou"],
                "mug_val_macro_iou": mug_val["iou"],
                "mug_test_macro_iou": mug_test["iou"],
                "mug_test_macro_f1": mug_test["f1"],
                "mug_test_macro_precision": mug_test["precision"],
                "mug_test_macro_recall": mug_test["recall"],
                "mug_test_gt_handle_ratio": mug_test["gt_handle_ratio"],
                "mug_test_pred_handle_ratio": mug_test["pred_handle_ratio"],
                "mug_test_micro_iou": micro["iou"],
                "mug_test_micro_f1": micro["f1"],
                "mug_test_worst_scene_iou": result["mug_test_worst_scene_iou"],
                "mug_test_best_scene_iou": result["mug_test_best_scene_iou"],
            }
        )
    save_csv(args.out_root / f"mug_incremental_summary_partial_{partial_name}.csv", flat_rows)
    with open(args.out_root / f"mug_incremental_results_partial_{partial_name}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
