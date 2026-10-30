"""Loading and inference for self-contained HandAffordSplat checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .model import PointNetMean

PathLike = Union[str, Path]


@dataclass
class HandAffordSplatPredictor:
    model: PointNetMean
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    device: torch.device
    metadata: dict

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.mean.shape[1]:
            raise ValueError(
                "Expected feature shape (N, {}), found {}".format(
                    self.mean.shape[1], features.shape
                )
            )
        normalized = ((features - self.mean) / self.std).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(normalized).to(self.device)
            scores = torch.sigmoid(self.model(tensor)).cpu().numpy()
        return scores.astype(np.float32)

    def classify(self, features: np.ndarray) -> np.ndarray:
        return self.predict(features) >= self.threshold


def _torch_load(path: PathLike, device: torch.device) -> dict:
    # PyTorch 2.6 changed the default of weights_only. These thesis checkpoints
    # also contain NumPy normalization arrays, so trusted local files need the
    # complete loader.
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def load_predictor(
    checkpoint_path: PathLike, device: Union[str, torch.device] = "cpu"
) -> HandAffordSplatPredictor:
    device = torch.device(device)
    checkpoint = _torch_load(checkpoint_path, device)

    required = {
        "model",
        "mean",
        "std",
        "model_name",
        "feature_variant",
        "input_dim",
        "latent_dim",
        "threshold",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError("Checkpoint is missing keys: {}".format(", ".join(missing)))
    if checkpoint["model_name"] != "pointnet_mean":
        raise ValueError("Unsupported model: {}".format(checkpoint["model_name"]))
    if checkpoint["feature_variant"] != "dino_geometry_color_quality":
        raise ValueError(
            "Unsupported feature variant: {}".format(checkpoint["feature_variant"])
        )

    input_dim = int(checkpoint["input_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    model = PointNetMean(input_dim=input_dim, latent_dim=latent_dim, dropout=0.1)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(1, -1)
    if mean.shape != (1, input_dim) or std.shape != (1, input_dim):
        raise ValueError("Checkpoint normalization shape does not match input_dim")
    std = np.maximum(std, 1e-6)

    return HandAffordSplatPredictor(
        model=model,
        mean=mean,
        std=std,
        threshold=float(checkpoint["threshold"]),
        device=device,
        metadata=dict(checkpoint.get("overall", {})),
    )
