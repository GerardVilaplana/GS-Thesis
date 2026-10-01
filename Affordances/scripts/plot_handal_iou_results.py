import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_MLP_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_full_mlp")

BASELINES = {
    "dino_only": "D",
    "geometry_only": "G",
    "dino_geometry": "D+G",
}

PALETTE = {
    "D": "#d62728",
    "G": "#1f77b4",
    "D+G": "#f2c94c",
}


def load_iou_table(mlp_root):
    frames = []
    for folder, label in BASELINES.items():
        path = mlp_root / folder / "per_scene_metrics.csv"
        df = pd.read_csv(path)
        df["model"] = label
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["scene"] = data["scene"].astype(str).str.zfill(6)
    data["model"] = pd.Categorical(data["model"], categories=["G", "D", "D+G"], ordered=True)
    return data


def save_distribution_plot(data, out_path):
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(
        data=data,
        x="iou",
        hue="model",
        hue_order=["G", "D", "D+G"],
        palette=PALETTE,
        common_norm=False,
        linewidth=2.8,
        bw_adjust=0.9,
        clip=(0.0, 1.0),
        ax=ax,
        legend=False,
    )
    sns.rugplot(
        data=data,
        x="iou",
        hue="model",
        hue_order=["G", "D", "D+G"],
        palette=PALETTE,
        height=0.045,
        linewidth=1.2,
        alpha=0.7,
        ax=ax,
        legend=False,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Scene IoU")
    ax.set_ylabel("Density")
    ax.set_title("Test Scene IoU Distribution")
    ax.set_axisbelow(True)
    ax.grid(True, color="#e8e8e8", linewidth=0.8, alpha=0.75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_scene_plot(data, out_path):
    sns.set_theme(style="whitegrid", context="talk")
    scene_order = (
        data[data["model"] == "D+G"]
        .sort_values("iou")["scene"]
        .tolist()
    )
    scene_rank = {scene: idx for idx, scene in enumerate(scene_order)}
    data = data.copy()
    data["scene_rank"] = data["scene"].map(scene_rank)
    data = data.sort_values(["model", "scene_rank"])
    fig, ax = plt.subplots(figsize=(15, 6))

    sns.lineplot(
        data=data,
        x="scene_rank",
        y="iou",
        hue="model",
        hue_order=["G", "D", "D+G"],
        style="model",
        markers=True,
        dashes=False,
        palette=PALETTE,
        sort=False,
        ax=ax,
    )
    ax.set_xticks(range(len(scene_order)))
    ax.set_xticklabels(scene_order, rotation=70, ha="right", fontsize=9)
    ax.set_xlabel("Test scene, sorted by D+G IoU")
    ax.set_ylabel("Scene IoU")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Per-Scene IoU by Model")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_root", type=Path, default=DEFAULT_MLP_ROOT)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.mlp_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_iou_table(args.mlp_root)
    data.to_csv(out_dir / "test_scene_iou_long.csv", index=False)
    save_distribution_plot(data, out_dir / "test_scene_iou_distribution_kde_hist.png")
    save_scene_plot(data, out_dir / "test_scene_iou_by_scene.png")

    summary = (
        data.groupby("model", observed=True)["iou"]
        .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    )
    summary.to_csv(out_dir / "test_scene_iou_summary.csv")

    print(f"Wrote plots and tables to {out_dir}")
    print(summary)


if __name__ == "__main__":
    main()
