import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
SCRIPT_DIR = BASE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import train_handal_17cat_mlp_baselines as base  # noqa: E402


BASELINE_ROOT = (
    BASE_ROOT / "outputs" / "03_handle_generalization" / "08_17category_mlp_baselines_v1"
)
OUT_ROOT = (
    BASE_ROOT / "outputs" / "03_handle_generalization" / "09_pointnet_global_context_v1"
)
MODEL_NAME = "pointnet_global_dino_geometry_color"
FEATURE_SET = base.FEATURE_SETS["dino_geometry_color"]


class PointNetGlobal(nn.Module):
    def __init__(self, in_dim, latent_dim=192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(384, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        local = self.encoder(x)
        global_max = local.max(dim=0, keepdim=True).values
        global_mean = local.mean(dim=0, keepdim=True)
        global_feat = torch.cat([global_max, global_mean], dim=1).expand(len(local), -1)
        return self.head(torch.cat([local, global_feat], dim=1)).squeeze(-1)


def load_split_manifest(path):
    with open(path, "r") as f:
        return json.load(f)["splits"]


def fit_standardizer(items):
    total = 0
    sum_x = None
    sumsq_x = None
    pos = 0
    for item in items:
        x, y = base.load_scene_arrays(item, FEATURE_SET)
        x = x.astype(np.float64, copy=False)
        if sum_x is None:
            sum_x = x.sum(axis=0)
            sumsq_x = np.square(x).sum(axis=0)
        else:
            sum_x += x.sum(axis=0)
            sumsq_x += np.square(x).sum(axis=0)
        total += len(x)
        pos += int(y.sum())
    mean = (sum_x / max(total, 1)).astype(np.float32, copy=False)[None, :]
    var = (sumsq_x / max(total, 1)) - np.square(sum_x / max(total, 1))
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32, copy=False)[None, :]
    std = np.maximum(std, 1e-6)
    return mean, std, total, pos


def load_scene_list(items, mean, std):
    scenes = []
    for item in items:
        x, y = base.load_scene_arrays(item, FEATURE_SET)
        x = ((x - mean) / std).astype(np.float32, copy=False)
        scenes.append({"item": item, "x": x, "y": y.astype(np.float32, copy=False)})
    return scenes


def predict_scene_scores(model, scenes, device):
    model.eval()
    scores = []
    slices = []
    y_all = []
    offset = 0
    with torch.no_grad():
        for scene in scenes:
            x = torch.from_numpy(scene["x"]).to(device)
            logits = model(x).detach().cpu().numpy()
            score = base.sigmoid_np(logits)
            scores.append(score.astype(np.float32, copy=False))
            y_all.append(scene["y"].astype(np.float32, copy=False))
            end = offset + len(score)
            slices.append((scene["item"], offset, end))
            offset = end
    return np.concatenate(y_all), np.concatenate(scores), slices


def train_epoch(model, scenes, loss_fn, opt, device, seed):
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
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def save_csv(path, rows):
    return base.save_csv(path, rows)


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


def train_one(run_name, split, out_dir, args, device):
    model_dir = out_dir / MODEL_NAME
    done_path = model_dir / "overall_metrics.json"
    if args.skip_existing and done_path.exists():
        print(f"[{run_name}/{MODEL_NAME}] skip existing {done_path}", flush=True)
        with open(done_path, "r") as f:
            return json.load(f)

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{run_name}/{MODEL_NAME}] fitting standardizer", flush=True)
    mean, std, train_n, train_pos = fit_standardizer(split["train"])
    pos_weight = (train_n - train_pos) / max(float(train_pos), 1.0)
    print(
        f"[{run_name}/{MODEL_NAME}] loading scenes "
        f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} "
        f"train_gaussians={train_n} pos_weight={pos_weight:.3f}",
        flush=True,
    )
    train_scenes = load_scene_list(split["train"], mean, std)
    val_scenes = load_scene_list(split["val"], mean, std)
    test_scenes = load_scene_list(split["test"], mean, std)

    in_dim = int(train_scenes[0]["x"].shape[1])
    model = PointNetGlobal(in_dim, latent_dim=args.latent_dim).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_scenes, loss_fn, opt, device, args.seed + epoch)
        y_val, val_score, val_slices = predict_scene_scores(model, val_scenes, device)
        th_best, _ = base.choose_threshold(y_val, val_score, val_slices)
        val_rows = base.per_scene_metrics(val_slices, y_val, val_score, th_best["threshold"])
        record = {
            "epoch": epoch,
            "loss": loss,
            "val_macro_scene_iou": base.macro_scene_score(val_rows, "iou"),
            "val_macro_scene_f1": base.macro_scene_score(val_rows, "f1"),
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
                f"[{run_name}/{MODEL_NAME}] epoch {epoch:03d}/{args.epochs} "
                f"loss={record['loss']:.4f} val_iou={record['val_macro_scene_iou']:.3f} "
                f"val_f1={record['val_macro_scene_f1']:.3f} th={record['val_threshold']:.2f}",
                flush=True,
            )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[{run_name}/{MODEL_NAME}] early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    y_val, val_score, val_slices = predict_scene_scores(model, val_scenes, device)
    th_best, threshold_curve = base.choose_threshold(y_val, val_score, val_slices)
    threshold = th_best["threshold"]
    y_test, test_score, test_slices = predict_scene_scores(model, test_scenes, device)
    test_rows = base.per_scene_metrics(test_slices, y_test, test_score, threshold)
    cat_rows = base.per_category_metrics(test_rows)
    micro = base.metrics_from_scores(y_test, test_score, threshold)
    macro = {
        key: base.macro_scene_score(test_rows, key)
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
        "model": MODEL_NAME,
        "architecture": (
            "shared per-Gaussian encoder; global max+mean pooled object embedding; "
            "prediction from local embedding concatenated with global embedding"
        ),
        "feature_set": FEATURE_SET,
        "input_dim": in_dim,
        "latent_dim": args.latent_dim,
        "train_scenes": len(split["train"]),
        "val_scenes": len(split["val"]),
        "test_scenes": len(split["test"]),
        "train_gaussians": int(sum(len(s["y"]) for s in train_scenes)),
        "val_gaussians": int(sum(len(s["y"]) for s in val_scenes)),
        "test_gaussians": int(len(y_test)),
        "train_handle_ratio": float(
            sum(float(s["y"].sum()) for s in train_scenes)
            / max(sum(len(s["y"]) for s in train_scenes), 1)
        ),
        "val_handle_ratio": float(
            sum(float(s["y"].sum()) for s in val_scenes)
            / max(sum(len(s["y"]) for s in val_scenes), 1)
        ),
        "test_handle_ratio": float(y_test.mean()),
        "selected_epoch": int(best["epoch"]),
        "selected_threshold": float(threshold),
        "selection_metric": "validation macro scene IoU, tie-broken by validation macro scene F1",
        "loss": "weighted BCEWithLogitsLoss",
        "positive_label": "handle_labels_thr0_25",
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
            "feature_set": FEATURE_SET,
            "input_dim": in_dim,
            "latent_dim": args.latent_dim,
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
        f"[{run_name}/{MODEL_NAME}] TEST macro IoU={macro['iou']:.3f} "
        f"macro F1={macro['f1']:.3f} micro correct={100 * micro['correct_percent']:.1f}% "
        f"threshold={threshold:.2f}",
        flush=True,
    )
    return overall


def load_seen_split():
    return load_split_manifest(BASELINE_ROOT / "01_seen_instance_17cat" / "split_manifest.json")


def load_loco_split(category):
    return load_split_manifest(
        BASELINE_ROOT / "02_leave_one_category_out_17cat" / category / "split_manifest.json"
    )


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
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--mode", choices=["seen", "loco", "aggregate"], required=True)
    parser.add_argument("--heldout_categories", nargs="+", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "aggregate":
        aggregate_results(args.out_root)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    with open(BASELINE_ROOT / "feature_roots.json", "r") as f:
        feature_meta = json.load(f)
    categories = sorted(feature_meta["categories"])
    with open(args.out_root / "source_baseline_root.json", "w") as f:
        json.dump(
            {
                "baseline_root": str(BASELINE_ROOT),
                "feature_roots": feature_meta["feature_roots"],
                "categories": categories,
                "note": "PointNet-style runs reuse the exact 17-category MLP split manifests.",
            },
            f,
            indent=2,
        )

    results = []
    if args.mode == "seen":
        split = load_seen_split()
        out_dir = args.out_root / "01_seen_instance_17cat"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[seen] split scenes: train={len(split['train'])} "
            f"val={len(split['val'])} test={len(split['test'])}",
            flush=True,
        )
        results.append(train_one("01_seen_instance_17cat", split, out_dir, args, device))
        write_flat_summary(out_dir / "seen_instance_summary.csv", results)
    else:
        heldout = args.heldout_categories or categories
        heldout = [c for c in heldout if c in categories]
        heldout = [c for i, c in enumerate(sorted(heldout)) if i % args.num_shards == args.shard]
        print(f"[loco shard {args.shard}/{args.num_shards}] heldout={heldout}", flush=True)
        for category in heldout:
            split = load_loco_split(category)
            out_dir = args.out_root / "02_leave_one_category_out_17cat" / category
            out_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{category}] split scenes: train={len(split['train'])} "
                f"val={len(split['val'])} test={len(split['test'])}",
                flush=True,
            )
            result = train_one(category, split, out_dir, args, device)
            results.append(result)
            write_flat_summary(out_dir / "category_summary.csv", [result])
        write_flat_summary(args.out_root / f"loco_summary_shard_{args.shard}_of_{args.num_shards}.csv", results)
        with open(args.out_root / f"loco_summary_shard_{args.shard}_of_{args.num_shards}.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
