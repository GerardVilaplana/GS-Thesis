import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


C0 = 0.28209479177387814
SPLIT_PATH = Path("/home/gvilaplana/GS-Thesis/Affordances/configs/handal_exp40_split.json")
SUP_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_exp40_supervision")
EMB_ROOT = Path(
    "/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_exp40_embeddings/"
    "facebook_dinov2_small_render_contrib_patch448x336"
)
OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_exp40_mlp")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sigmoid_np(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def scene_items(split_path, subset):
    split = load_json(split_path)
    return split[subset]


def feature_path(scene, emb_root):
    return emb_root / "features" / f"handal_mug_scene_{scene}_dinov2_render_contrib_features.npz"


def label_path(scene, sup_root):
    return sup_root / "handle_labels_thr0.25" / f"handal_mug_scene_{scene}_handle_labels_splat_overlap_thr0.25.npz"


def object_ply_path(scene, sup_root):
    return sup_root / "object_thr0.75_ply" / f"handal_mug_scene_{scene}_object_pruned_thr0.75.ply"


def geometry_features(scene, sup_root):
    ply = PlyData.read(object_ply_path(scene, sup_root))
    v = ply["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    scale = np.vstack([v["scale_0"], v["scale_1"], v["scale_2"]]).T.astype(np.float32)
    rot = np.vstack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]).T.astype(np.float32)
    rot = rot / np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8)
    opacity = sigmoid_np(np.asarray(v["opacity"], dtype=np.float32))[:, None].astype(np.float32)
    return np.concatenate([xyz, scale, rot, opacity], axis=1)


def load_scene(scene, sup_root, emb_root, feature_set):
    emb = np.load(feature_path(scene, emb_root))
    labels = np.load(label_path(scene, sup_root))
    dino = emb["embeddings"].astype(np.float32)
    valid = emb["valid_embeddings"].astype(bool)
    y = labels["is_handle"].astype(np.float32)
    if len(y) != len(dino):
        raise ValueError(f"Scene {scene} length mismatch: embeddings={len(dino)} labels={len(y)}")
    parts = []
    if "dino" in feature_set:
        parts.append(dino)
    if "geometry" in feature_set:
        parts.append(geometry_features(scene, sup_root))
    x = np.concatenate(parts, axis=1).astype(np.float32)
    return x[valid], y[valid], valid


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


def metrics_from_scores(y_true, score, threshold=0.5):
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
    try:
        ap = float(average_precision_score(y_true.astype(np.uint8), score))
    except ValueError:
        ap = float("nan")
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
        "average_precision": ap,
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


def standardize_fit(x):
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def train_one(baseline, feature_set, args, train_items, test_items):
    out_dir = args.out_root / baseline
    out_dir.mkdir(parents=True, exist_ok=True)

    train_xs, train_ys = [], []
    for item in train_items:
        x, y, _ = load_scene(item["scene"], args.sup_root, args.emb_root, feature_set)
        train_xs.append(x)
        train_ys.append(y)
    x_train = np.concatenate(train_xs, axis=0)
    y_train = np.concatenate(train_ys, axis=0)
    mean, std = standardize_fit(x_train)
    x_train = (x_train - mean) / std

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = MLP(x_train.shape[1]).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

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
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"{baseline}: epoch {epoch:03d}/{args.epochs}, loss={mean_loss:.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": x_train.shape[1],
            "feature_set": feature_set,
            "mean": mean,
            "std": std,
            "history": history,
        },
        out_dir / "model.pt",
    )

    model.eval()
    per_scene = []
    pooled_y, pooled_score = [], []
    predictions = {}
    with torch.no_grad():
        for item in test_items:
            scene = item["scene"]
            x, y, valid = load_scene(scene, args.sup_root, args.emb_root, feature_set)
            x = (x - mean) / std
            logits = []
            for start in range(0, len(x), args.batch_size):
                xb = torch.from_numpy(x[start : start + args.batch_size]).to(device)
                logits.append(model(xb).detach().cpu().numpy())
            score = sigmoid_np(np.concatenate(logits))
            row = {"scene": scene, **metrics_from_scores(y, score, args.threshold)}
            per_scene.append(row)
            pooled_y.append(y)
            pooled_score.append(score)
            predictions[scene] = {"valid_y": y, "valid_score": score, "valid_mask": valid}

    pooled_y = np.concatenate(pooled_y)
    pooled_score = np.concatenate(pooled_score)
    micro = metrics_from_scores(pooled_y, pooled_score, args.threshold)
    metric_keys = [k for k in per_scene[0].keys() if k != "scene"]
    macro = {
        k: float(np.nanmean([row[k] for row in per_scene]))
        for k in metric_keys
        if isinstance(per_scene[0][k], (int, float, np.integer, np.floating))
    }
    overall = {
        "baseline": baseline,
        "feature_set": feature_set,
        "train_gaussians": int(len(y_train)),
        "train_handle_ratio": float(y_train.mean()),
        "threshold": args.threshold,
        "micro": micro,
        "macro_scene_average": macro,
        "best_scene_iou": max(per_scene, key=lambda r: r["iou"])["scene"],
        "worst_scene_iou": min(per_scene, key=lambda r: r["iou"])["scene"],
    }

    with open(out_dir / "per_scene_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_scene[0].keys()))
        writer.writeheader()
        writer.writerows(per_scene)
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        **{f"{scene}_valid_score": item["valid_score"] for scene, item in predictions.items()},
        **{f"{scene}_valid_y": item["valid_y"] for scene, item in predictions.items()},
        **{f"{scene}_valid_mask": item["valid_mask"] for scene, item in predictions.items()},
    )
    return overall, per_scene, predictions


def write_prediction_ply(scene, sup_root, score_valid, valid_mask, out_ply, threshold):
    ply = PlyData.read(object_ply_path(scene, sup_root))
    vertices = np.array(ply["vertex"].data, copy=True)
    score = np.zeros(len(vertices), dtype=np.float32)
    score[valid_mask] = score_valid.astype(np.float32)
    rgb = np.zeros((len(vertices), 3), dtype=np.float32)
    rgb[:, 0] = np.maximum(score, 0.12)
    rgb[:, 1] = 0.10 * (1.0 - score)
    rgb[:, 2] = 1.0 - score
    uncertain = np.abs(score - threshold) < 0.05
    rgb[uncertain] = np.array([1.0, 0.85, 0.05], dtype=np.float32)
    rgb[~valid_mask] = np.array([0.08, 0.08, 0.08], dtype=np.float32)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(out_ply)


def choose_visual_scenes(per_scene, max_count):
    ordered = sorted(per_scene, key=lambda r: r["iou"])
    choices = [ordered[0], ordered[-1], ordered[len(ordered) // 2]]
    choices.append(max(per_scene, key=lambda r: r["fp_percent"]))
    choices.append(max(per_scene, key=lambda r: r["fn_percent"]))
    unique = []
    seen = set()
    for row in choices + ordered:
        if row["scene"] in seen:
            continue
        unique.append(row)
        seen.add(row["scene"])
        if len(unique) >= max_count:
            break
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--sup_root", type=Path, default=SUP_ROOT)
    parser.add_argument("--emb_root", type=Path, default=EMB_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_visual_plys", type=int, default=5)
    args = parser.parse_args()

    train_items = scene_items(args.split_path, "train")
    test_items = scene_items(args.split_path, "test")
    baselines = {
        "dino_only": ["dino"],
        "geometry_only": ["geometry"],
        "dino_geometry": ["dino", "geometry"],
    }
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_overall = {}
    best_payload = None
    for baseline, feature_set in baselines.items():
        overall, per_scene, predictions = train_one(baseline, feature_set, args, train_items, test_items)
        all_overall[baseline] = overall
        print(
            f"{baseline}: macro IoU={overall['macro_scene_average']['iou']:.3f}, "
            f"macro F1={overall['macro_scene_average']['f1']:.3f}, "
            f"micro correct={100 * overall['micro']['correct_percent']:.1f}%"
        )
        if baseline == "dino_geometry":
            best_payload = (per_scene, predictions)

    with open(args.out_root / "all_baselines_summary.json", "w") as f:
        json.dump(all_overall, f, indent=2)

    per_scene, predictions = best_payload
    selected = choose_visual_scenes(per_scene, args.num_visual_plys)
    visual_rows = []
    for row in selected:
        scene = row["scene"]
        pred = predictions[scene]
        out_ply = args.out_root / "prediction_ply" / f"handal_mug_scene_{scene}_dino_geometry_prediction.ply"
        write_prediction_ply(scene, args.sup_root, pred["valid_score"], pred["valid_mask"], out_ply, args.threshold)
        visual_rows.append({"scene": scene, "reason_metrics": row, "prediction_ply": str(out_ply)})
        print(f"prediction PLY {scene}: {out_ply}")
    with open(args.out_root / "prediction_ply" / "selected_prediction_plys.json", "w") as f:
        json.dump(visual_rows, f, indent=2)


if __name__ == "__main__":
    main()
