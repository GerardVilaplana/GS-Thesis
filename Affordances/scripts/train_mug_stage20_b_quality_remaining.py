#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
sys.path.insert(0, str(BASE_ROOT / "scripts"))

import train_mug_stage20_capacity_opacity as stage20  # noqa: E402


B_HANDAL_ROOT = BASE_ROOT / "data" / "handal_mugs_b_quality_features_v1"
OUT_ROOT = BASE_ROOT / "outputs" / "03_handle_generalization" / "20_mug_capacity_opacity_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_root", type=Path, default=stage20.SYN_ROOT)
    parser.add_argument("--handal_root", type=Path, default=B_HANDAL_ROOT)
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        stage20.write_summary(args.out_root)
        return

    runs = [
        {
            "run_id": "C4",
            "run_name": "C4_b_quality_pointnetmax__opacity_match",
            "model_size": "b3_like",
            "opacity_filter": "none",
            "purpose": "B3 setup using B-quality HANDAL mug reconstructions.",
        },
        {
            "run_id": "C5",
            "run_name": "C5_b_quality_pointnetmax__opacity_match__filter_relaxed",
            "model_size": "b3_like",
            "opacity_filter": "relaxed",
            "purpose": "B-quality HANDAL mugs with relaxed low-opacity filtering.",
        },
        {
            "run_id": "C6",
            "run_name": "C6_b_quality_pointnetmax__opacity_match__filter_aggressive",
            "model_size": "b3_like",
            "opacity_filter": "aggressive",
            "purpose": "B-quality HANDAL mugs with aggressive low-opacity filtering.",
        },
        {
            "run_id": "C9",
            "run_name": "C9_b_quality_pointnetmax__opacity_match__filter_very_aggressive",
            "model_size": "b3_like",
            "opacity_filter": "very_aggressive",
            "purpose": "B-quality HANDAL mugs with very aggressive low-opacity filtering at 0.30.",
        },
        {
            "run_id": "C10",
            "run_name": "C10_b_quality_pointnetmax__opacity_match__filter_extreme",
            "model_size": "b3_like",
            "opacity_filter": "extreme",
            "purpose": "B-quality HANDAL mugs with extreme low-opacity filtering at 0.50.",
        },
    ]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    handal_split, syn_split = stage20.build_split(args)
    target_stats = stage20.stage13.build_target_stats(handal_split["train"], [stage20.FEATURE])

    print(
        f"[data] B-quality HANDAL split={ {k: len(v) for k, v in handal_split.items()} } "
        f"AffordSplat={ {k: len(v) for k, v in syn_split.items()} }",
        flush=True,
    )
    print(f"[plan] runs={[r['run_id'] for r in runs]} device={device}", flush=True)
    for run in runs:
        print(f"[run] {run['run_id']} {run['run_name']}", flush=True)
        stage20.train_one(run, args, device, target_stats, handal_split, syn_split)
        stage20.write_summary(args.out_root)
    stage20.write_summary(args.out_root)
    print("[done] wrote summary", flush=True)


if __name__ == "__main__":
    main()
