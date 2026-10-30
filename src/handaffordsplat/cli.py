"""Command-line inference for the selected HandAffordSplat model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_predictor
from .features import build_final_features
from .metrics import binary_metrics
from .visualization import write_binary_overlay_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict per-Gaussian handle labels for one refined object."
    )
    parser.add_argument("--object-npz", type=Path, required=True)
    parser.add_argument("--dino-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--source-ply", type=Path)
    parser.add_argument("--output-ply", type=Path)
    parser.add_argument("--overlay-alpha", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.source_ply is None) != (args.output_ply is None):
        raise SystemExit("--source-ply and --output-ply must be provided together")

    bundle = build_final_features(args.object_npz, args.dino_npz)
    predictor = load_predictor(args.checkpoint, args.device)
    scores = predictor.predict(bundle.features)
    threshold = predictor.threshold if args.threshold is None else float(args.threshold)
    prediction = scores >= threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "scores": scores,
        "prediction": prediction.astype(np.uint8),
        "threshold": np.asarray(threshold, dtype=np.float32),
    }
    if bundle.labels is not None:
        output["labels"] = bundle.labels.astype(np.uint8)
    np.savez_compressed(str(args.output), **output)

    summary = {
        "object_npz": str(args.object_npz),
        "dino_npz": str(args.dino_npz),
        "checkpoint": str(args.checkpoint),
        "num_gaussians": int(len(scores)),
        "threshold": threshold,
        "predicted_handle_ratio": float(prediction.mean()),
    }
    if bundle.labels is not None:
        summary["metrics"] = binary_metrics(bundle.labels, scores, threshold)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if args.source_ply is not None:
        write_binary_overlay_ply(
            args.source_ply, args.output_ply, prediction, args.overlay_alpha
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
