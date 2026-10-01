#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
STAGE15 = BASE_ROOT / "outputs" / "03_handle_generalization" / "15_graph_attention_gaussian_v1"
STAGE16 = BASE_ROOT / "outputs" / "03_handle_generalization" / "16_context_denoise_ablation_v1"
CRITICAL = ["measuring_cups", "mugs", "power_drills", "screwdrivers", "utensils", "whisks"]


def load_stage15_none():
    df = pd.read_csv(STAGE15 / "all_finished_models_summary.csv")
    df = df[
        (df["feature_variant"] == "geometry_color_scene_norm")
        & (df["model"] == "graph_attention_geometry_color_scene_norm")
    ].copy()
    df["preprocess"] = "none"
    df["model_type"] = "graph"
    df["k_layers"] = "[16, 32]"
    df["pooling"] = ""
    return df


def load_stage16(out_root):
    path = out_root / "all_finished_models_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def choose_preprocess(df):
    rows = []
    for prep in ["none", "density_trim", "largest_component"]:
        part = df[
            (df["preprocess"] == prep)
            & (df["model_type"] == "graph")
            & (df["k_layers"].astype(str) == "[16, 32]")
        ]
        seen = part[part["run_name"] == "01_seen_instance_17cat"]["macro_iou"]
        crit = part[part["run_name"].isin(CRITICAL)]["macro_iou"]
        if len(seen) == 0 or len(crit) < len(CRITICAL):
            continue
        rows.append(
            {
                "preprocess": prep,
                "seen_iou": float(seen.iloc[0]),
                "critical_loco_iou": float(crit.mean()),
                "selection_score": float(0.5 * seen.iloc[0] + 0.5 * crit.mean()),
                "critical_loco_runs": int(len(crit)),
            }
        )
    sel = pd.DataFrame(rows)
    if sel.empty:
        return sel, None
    priority = {"none": 0, "density_trim": 1, "largest_component": 2}
    best_score = sel["selection_score"].max()
    near = sel[sel["selection_score"] >= best_score - 0.01].copy()
    near["priority"] = near["preprocess"].map(priority)
    chosen = near.sort_values(["priority", "selection_score"], ascending=[True, False]).iloc[0].to_dict()
    return sel.sort_values("selection_score", ascending=False), chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, default=STAGE16)
    args = parser.parse_args()

    combined = pd.concat([load_stage15_none(), load_stage16(args.out_root)], ignore_index=True, sort=False)
    args.out_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_root / "stage15_stage16_combined_summary.csv", index=False)

    sel, chosen = choose_preprocess(combined)
    if not sel.empty:
        sel.to_csv(args.out_root / "stage16a_preprocess_selection.csv", index=False)
    with open(args.out_root / "stage16a_chosen_preprocess.json", "w") as f:
        json.dump({"chosen": chosen, "note": "Tie within 0.01 prefers none, then density_trim, then largest_component."}, f, indent=2)
    print("PREPROCESS_SELECTION")
    if sel.empty:
        print("not enough completed Stage 16A rows yet")
    else:
        print(sel.to_string(index=False))
        print("CHOSEN", chosen["preprocess"])

    views = []
    if not combined.empty:
        key_cols = [
            "run_name",
            "model_type",
            "preprocess",
            "pooling",
            "k_layers",
            "macro_iou",
            "macro_f1",
            "macro_auprc",
            "selected_threshold",
            "test_keep_ratio",
        ]
        views = [c for c in key_cols if c in combined.columns]
        combined[views].to_csv(args.out_root / "compact_stage16_summary.csv", index=False)


if __name__ == "__main__":
    main()
