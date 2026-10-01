#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
STAGE = ROOT / "outputs" / "03_handle_generalization" / "17_mug_b0_b6_full_mugs_v1"
B0 = STAGE / "B0_handal_only__orig_clean" / "pointnet_max_geometry_color_scene_norm" / "per_scene_metrics.csv"
B3 = STAGE / "B3_handal_plus_synth__orig_opacity_match" / "pointnet_max_geometry_color_scene_norm" / "per_scene_metrics.csv"
OUT = STAGE / "plots" / "b0_vs_b3"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b0 = pd.read_csv(B0)[["scene_key", "iou"]].rename(columns={"iou": "B0_HANDAL_only"})
    b3 = pd.read_csv(B3)[["scene_key", "iou"]].rename(columns={"iou": "B3_HANDAL_plus_opacity_match"})
    df = b0.merge(b3, on="scene_key", validate="one_to_one")
    df = df.sort_values("B3_HANDAL_plus_opacity_match", ascending=True).reset_index(drop=True)
    df["scene_short"] = df["scene_key"].str.replace("mugs__", "", regex=False)
    df["delta_B3_minus_B0"] = df["B3_HANDAL_plus_opacity_match"] - df["B0_HANDAL_only"]
    df.to_csv(OUT / "b0_vs_b3_test_scene_iou.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "B0 HANDAL only": "#2b7bba",
        "B3 HANDAL + opacity-matched synthetic": "#d62728",
    }

    long = df.melt(
        id_vars=["scene_key", "scene_short"],
        value_vars=["B0_HANDAL_only", "B3_HANDAL_plus_opacity_match"],
        var_name="model",
        value_name="IoU",
    )
    long["model"] = long["model"].map(
        {
            "B0_HANDAL_only": "B0 HANDAL only",
            "B3_HANDAL_plus_opacity_match": "B3 HANDAL + opacity-matched synthetic",
        }
    )

    plt.figure(figsize=(14, 6))
    ax = sns.lineplot(
        data=long,
        x="scene_short",
        y="IoU",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        linewidth=2.4,
        markersize=7,
        palette=palette,
    )
    ax.set_title("HANDAL Test Mug IoU: B0 vs B3")
    ax.set_xlabel("Test mug scene, sorted by B3 IoU")
    ax.set_ylabel("Scene IoU")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.65)
    plt.xticks(rotation=60, ha="right")
    plt.legend(title="model", loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "b0_vs_b3_per_scene_iou_lines.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 6))
    ax = sns.kdeplot(
        data=long,
        x="IoU",
        hue="model",
        fill=True,
        common_norm=False,
        alpha=0.22,
        linewidth=2.4,
        palette=palette,
        bw_adjust=0.85,
        clip=(0, 1),
    )
    ax.set_title("HANDAL Test Mug IoU Distribution")
    ax.set_xlabel("Scene IoU")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.65)
    plt.tight_layout()
    plt.savefig(OUT / "b0_vs_b3_iou_kde.png", dpi=220)
    plt.close()

    print(f"wrote {OUT / 'b0_vs_b3_test_scene_iou.csv'}")
    print(f"wrote {OUT / 'b0_vs_b3_per_scene_iou_lines.png'}")
    print(f"wrote {OUT / 'b0_vs_b3_iou_kde.png'}")


if __name__ == "__main__":
    main()
