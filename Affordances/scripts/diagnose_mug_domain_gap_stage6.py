#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plyfile import PlyData


C0 = 0.28209479177387814


RUN_ORDER = [
    "stage2_synthetic_sanity_clean",
    "stage3_synth_to_handal_clean_synthval",
    "stage3_synth_to_handal_clean_handalval",
    "stage4_synth_to_handal_mild_handalval",
    "stage4_synth_to_handal_medium_handalval",
    "stage4_synth_to_handal_strong_handalval",
    "stage5_handal_only",
    "stage5_mixed_clean",
    "stage5_mixed_mild",
    "stage5_mixed_medium",
    "stage5_mixed_strong",
]


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def write_threshold_ply(source_ply, out_ply, scores, threshold):
    ply = PlyData.read(str(source_ply))
    vertices = np.array(ply["vertex"].data, copy=True)
    scores = np.asarray(scores, dtype=np.float32)
    if len(vertices) != len(scores):
        raise ValueError(f"PLY/score length mismatch: {source_ply} ply={len(vertices)} scores={len(scores)}")
    pred = scores >= threshold
    rgb = np.zeros((len(scores), 3), dtype=np.float32)
    rgb[~pred] = np.array([0.05, 0.18, 1.0], dtype=np.float32)
    rgb[pred] = np.array([1.0, 0.02, 0.02], dtype=np.float32)
    sh = rgb_to_sh(rgb)
    vertices["f_dc_0"] = sh[:, 0]
    vertices["f_dc_1"] = sh[:, 1]
    vertices["f_dc_2"] = sh[:, 2]
    ply["vertex"].data = vertices
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(out_ply))


def scalar_str(x):
    a = np.asarray(x)
    if a.shape == ():
        return str(a.item())
    return str(a.tolist())


def read_npz_meta(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        source_ply = scalar_str(z["source_ply"]) if "source_ply" in z.files else ""
        source_object_ply = scalar_str(z["source_object_ply"]) if "source_object_ply" in z.files else ""
    return source_ply or source_object_ply


def resolve_source_ply(scene_key, synthetic_npz_root, handal_npz_root, handal_object_ply_root):
    if scene_key.startswith("affordsplat_"):
        npz_path = synthetic_npz_root / f"{scene_key}.npz"
        source = Path(read_npz_meta(npz_path))
        if source.exists():
            return source
    npz_path = handal_npz_root / f"{scene_key}.npz"
    if npz_path.exists():
        source = read_npz_meta(npz_path)
        if source and Path(source).exists():
            return Path(source)
    return handal_object_ply_root / f"{scene_key}_object_pruned_thr0.60.ply"


def load_overall(model_dir):
    with open(model_dir / "overall_metrics.json", "r") as f:
        return json.load(f)


def export_pointnet_prediction_plys(args):
    out_root = args.out_root / "pointnet_prediction_ply_3_per_run"
    rows = []
    for run_name in RUN_ORDER:
        model_dir = args.exp_root / run_name / "pointnet_global_geometry_color"
        pred_path = model_dir / "test_predictions.npz"
        if not pred_path.exists():
            continue
        overall = load_overall(model_dir)
        threshold = float(overall["selected_threshold"])
        pack = np.load(pred_path)
        scene_keys = [str(x) for x in pack["scene_keys"]]
        offsets = pack["offsets"]
        scores = pack["scores"].astype(np.float32)
        for scene_key, (start, end) in list(zip(scene_keys, offsets))[: args.num_plys_per_run]:
            source_ply = resolve_source_ply(
                scene_key, args.synthetic_npz_root, args.handal_npz_root, args.handal_object_ply_root
            )
            out_ply = out_root / run_name / f"{scene_key}_{run_name}_pointnet_pred_red_blue.ply"
            write_threshold_ply(source_ply, out_ply, scores[int(start) : int(end)], threshold)
            rows.append(
                {
                    "run_name": run_name,
                    "model": "pointnet_global_geometry_color",
                    "scene_key": scene_key,
                    "threshold": threshold,
                    "source_ply": str(source_ply),
                    "output_ply": str(out_ply),
                    "num_gaussians": int(end - start),
                    "pred_positive_ratio": float((scores[int(start) : int(end)] >= threshold).mean()),
                }
            )
            print(f"wrote {out_ply}")
    write_csv(out_root / "exported_pointnet_prediction_plys.csv", rows)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scene_stats(npz_path, domain):
    with np.load(npz_path, allow_pickle=False) as z:
        xyz = z["xyz"].astype(np.float32)
        labels = z["handle_labels_thr0_25"].astype(bool)
        scale = z["scale"].astype(np.float32)
        opacity = z["opacity"].astype(np.float32)
        color = z["color"].astype(np.float32)
        split = scalar_str(z["split"]) if "split" in z.files else "unknown"
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    extent = maxs - mins
    xyz_norm = xyz - xyz.mean(axis=0, keepdims=True)
    radius = max(float(np.linalg.norm(xyz_norm, axis=1).max()), 1e-6)
    xyz_unit = xyz_norm / radius
    if labels.any():
        pos_centroid = xyz_unit[labels].mean(axis=0)
        pos_extent = xyz_unit[labels].max(axis=0) - xyz_unit[labels].min(axis=0)
        pos_radial = np.linalg.norm(xyz_unit[labels], axis=1).mean()
    else:
        pos_centroid = np.full(3, np.nan, dtype=np.float32)
        pos_extent = np.full(3, np.nan, dtype=np.float32)
        pos_radial = np.nan
    return {
        "scene_key": npz_path.stem,
        "domain": domain,
        "split": split,
        "num_gaussians": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_ratio": float(labels.mean()),
        "extent_x": float(extent[0]),
        "extent_y": float(extent[1]),
        "extent_z": float(extent[2]),
        "extent_diag": float(np.linalg.norm(extent)),
        "scale_mean": float(scale.mean()),
        "scale_std": float(scale.std()),
        "opacity_mean": float(opacity.mean()),
        "opacity_std": float(opacity.std()),
        "color_r_mean": float(color[:, 0].mean()),
        "color_g_mean": float(color[:, 1].mean()),
        "color_b_mean": float(color[:, 2].mean()),
        "pos_centroid_x_norm": float(pos_centroid[0]),
        "pos_centroid_y_norm": float(pos_centroid[1]),
        "pos_centroid_z_norm": float(pos_centroid[2]),
        "pos_extent_x_norm": float(pos_extent[0]),
        "pos_extent_y_norm": float(pos_extent[1]),
        "pos_extent_z_norm": float(pos_extent[2]),
        "pos_radial_mean_norm": float(pos_radial),
    }


def sample_gaussian_rows(npz_paths, domain, max_per_scene=1500, seed=20260719):
    rng = np.random.default_rng(seed)
    rows = []
    for p in npz_paths:
        with np.load(p, allow_pickle=False) as z:
            n = len(z["handle_labels_thr0_25"])
            take = min(max_per_scene, n)
            idx = rng.choice(n, size=take, replace=False)
            scale = z["scale"].astype(np.float32)[idx]
            opacity = z["opacity"].astype(np.float32)[idx]
            color = z["color"].astype(np.float32)[idx]
            labels = z["handle_labels_thr0_25"].astype(bool)[idx]
            xyz = z["xyz"].astype(np.float32)
            xyz_norm = xyz - xyz.mean(axis=0, keepdims=True)
            radius = max(float(np.linalg.norm(xyz_norm, axis=1).max()), 1e-6)
            xyz_unit = (xyz_norm / radius)[idx]
        for j in range(take):
            rows.append(
                {
                    "scene_key": p.stem,
                    "domain": domain,
                    "label": "positive" if labels[j] else "negative",
                    "scale_mean": float(scale[j].mean()),
                    "opacity": float(opacity[j]),
                    "color_r": float(color[j, 0]),
                    "color_g": float(color[j, 1]),
                    "color_b": float(color[j, 2]),
                    "x_norm": float(xyz_unit[j, 0]),
                    "y_norm": float(xyz_unit[j, 1]),
                    "z_norm": float(xyz_unit[j, 2]),
                    "radial_norm": float(np.linalg.norm(xyz_unit[j])),
                }
            )
    return rows


def aggregate_scene_stats(df):
    metrics = [
        "num_gaussians",
        "positive_ratio",
        "extent_diag",
        "scale_mean",
        "opacity_mean",
        "color_r_mean",
        "color_g_mean",
        "color_b_mean",
        "pos_radial_mean_norm",
    ]
    rows = []
    for (domain, split), g in df.groupby(["domain", "split"], sort=True):
        row = {"domain": domain, "split": split, "num_scenes": int(len(g))}
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std())
        rows.append(row)
    return rows


def plot_scene_diagnostics(scene_df, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for col, title in [
        ("num_gaussians", "Number of Gaussians"),
        ("positive_ratio", "Handle/Grasp Positive Ratio"),
        ("extent_diag", "XYZ Extent Diagonal"),
        ("scale_mean", "Mean Gaussian Scale"),
        ("opacity_mean", "Mean Opacity"),
    ]:
        plt.figure(figsize=(8, 4.5))
        sns.boxplot(data=scene_df, x="domain", y=col, hue="domain", legend=False)
        sns.stripplot(data=scene_df, x="domain", y=col, color="black", alpha=0.35, size=2)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_dir / f"scene_{col}_domain_boxplot.png", dpi=180)
        plt.close()

    long = scene_df.melt(
        id_vars=["domain", "scene_key"],
        value_vars=["color_r_mean", "color_g_mean", "color_b_mean"],
        var_name="channel",
        value_name="mean_color",
    )
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=long, x="channel", y="mean_color", hue="domain")
    plt.title("Mean Color Distribution by Domain")
    plt.tight_layout()
    plt.savefig(plot_dir / "scene_color_mean_domain_boxplot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=scene_df,
        x="pos_centroid_x_norm",
        y="pos_centroid_y_norm",
        hue="domain",
        style="split",
    )
    plt.title("Positive Label Centroid in Normalized XY")
    plt.tight_layout()
    plt.savefig(plot_dir / "positive_centroid_xy_domain.png", dpi=180)
    plt.close()


def plot_gaussian_diagnostics(gauss_df, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for col, title in [
        ("scale_mean", "Per-Gaussian Scale Mean"),
        ("opacity", "Per-Gaussian Opacity"),
        ("radial_norm", "Per-Gaussian Normalized Radial Position"),
    ]:
        plt.figure(figsize=(8, 4.5))
        sns.kdeplot(data=gauss_df, x=col, hue="domain", common_norm=False, fill=True, alpha=0.25)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_dir / f"gaussian_{col}_domain_kde.png", dpi=180)
        plt.close()

    color_long = gauss_df.melt(
        id_vars=["domain", "label"],
        value_vars=["color_r", "color_g", "color_b"],
        var_name="channel",
        value_name="color",
    )
    plt.figure(figsize=(9, 4.8))
    sns.kdeplot(data=color_long, x="color", hue="domain", common_norm=False)
    plt.title("Per-Gaussian Color Distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "gaussian_color_domain_kde.png", dpi=180)
    plt.close()

    pos = gauss_df[gauss_df["label"] == "positive"]
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=pos, x="x_norm", y="y_norm", hue="domain", alpha=0.4, s=8)
    plt.title("Positive Gaussian Spatial Distribution, Normalized XY")
    plt.tight_layout()
    plt.savefig(plot_dir / "positive_gaussians_xy_domain.png", dpi=180)
    plt.close()


def build_score_histograms(args, plot_dir):
    rows = []
    transfer_runs = [
        "stage3_synth_to_handal_clean_synthval",
        "stage3_synth_to_handal_clean_handalval",
        "stage4_synth_to_handal_mild_handalval",
        "stage4_synth_to_handal_medium_handalval",
        "stage4_synth_to_handal_strong_handalval",
    ]
    for run in transfer_runs:
        for model in ["old_mlp_geometry_color", "pointnet_global_geometry_color"]:
            pred_path = args.exp_root / run / model / "test_predictions.npz"
            if not pred_path.exists():
                continue
            z = np.load(pred_path)
            scores = z["scores"].astype(np.float32)
            labels = z["labels"].astype(bool)
            rng = np.random.default_rng(20260719)
            take = min(args.max_score_samples, len(scores))
            idx = rng.choice(len(scores), size=take, replace=False)
            for score, label in zip(scores[idx], labels[idx]):
                rows.append(
                    {
                        "run_name": run,
                        "model": model,
                        "score": float(score),
                        "label": "positive" if label else "negative",
                    }
                )
    score_df = pd.DataFrame(rows)
    score_df.to_csv(args.out_root / "tables" / "synthetic_trained_on_handal_score_samples.csv", index=False)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for model, g in score_df.groupby("model"):
        grid = sns.FacetGrid(g, col="run_name", col_wrap=2, hue="label", sharex=True, sharey=False, height=3.2)
        grid.map_dataframe(sns.histplot, x="score", bins=40, stat="density", common_norm=False, alpha=0.45)
        grid.add_legend()
        grid.fig.suptitle(f"Scores on HANDAL Test from Synthetic-Trained {model}", y=1.02)
        grid.savefig(plot_dir / f"score_histograms_handal_test_{model}.png", dpi=180, bbox_inches="tight")
        plt.close(grid.fig)
    return score_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_root",
        type=Path,
        default=Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/10_mug_synthetic_domain_gap_v1"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/10_mug_synthetic_domain_gap_v1/stage6_diagnostics"),
    )
    parser.add_argument(
        "--synthetic_npz_root",
        type=Path,
        default=Path("/home/gvilaplana/GS-Thesis/Affordances/data/affordsplat_mug_grasp_features_v1/npz"),
    )
    parser.add_argument(
        "--handal_npz_root",
        type=Path,
        default=Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features/npz"),
    )
    parser.add_argument(
        "--handal_object_ply_root",
        type=Path,
        default=Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_handle_generalization_features/work/object_ply"),
    )
    parser.add_argument("--num_plys_per_run", type=int, default=3)
    parser.add_argument("--max_gaussian_samples_per_scene", type=int, default=1500)
    parser.add_argument("--max_score_samples", type=int, default=25000)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "tables").mkdir(parents=True, exist_ok=True)
    (args.out_root / "plots").mkdir(parents=True, exist_ok=True)

    export_pointnet_prediction_plys(args)

    synthetic_paths = sorted(args.synthetic_npz_root.glob("*.npz"))
    handal_paths = sorted(args.handal_npz_root.glob("mugs__*.npz"))
    scene_rows = [scene_stats(p, "AffordSplat") for p in synthetic_paths]
    scene_rows += [scene_stats(p, "HANDAL") for p in handal_paths]
    scene_df = pd.DataFrame(scene_rows)
    scene_df.to_csv(args.out_root / "tables" / "mug_scene_feature_stats.csv", index=False)
    write_csv(args.out_root / "tables" / "mug_scene_feature_stats_by_domain_split.csv", aggregate_scene_stats(scene_df))

    gauss_rows = sample_gaussian_rows(synthetic_paths, "AffordSplat", args.max_gaussian_samples_per_scene)
    gauss_rows += sample_gaussian_rows(handal_paths, "HANDAL", args.max_gaussian_samples_per_scene)
    gauss_df = pd.DataFrame(gauss_rows)
    gauss_df.to_csv(args.out_root / "tables" / "mug_gaussian_feature_samples.csv", index=False)

    plot_scene_diagnostics(scene_df, args.out_root / "plots")
    plot_gaussian_diagnostics(gauss_df, args.out_root / "plots")
    build_score_histograms(args, args.out_root / "plots")
    print(f"wrote diagnostics to {args.out_root}")


if __name__ == "__main__":
    main()
