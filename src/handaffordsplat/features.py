"""Feature construction matching the final thesis configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]
LABEL_KEY = "handle_labels_thr0_25"
DEFAULT_VIEW_NORMALIZER = 96.0


@dataclass(frozen=True)
class FeatureBundle:
    features: np.ndarray
    labels: Optional[np.ndarray]
    xyz: np.ndarray


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("xyz must have shape (N, 3) with N > 0")
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    return (centered / radius).astype(np.float32)


def scene_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = np.maximum(values.std(axis=0, keepdims=True), eps)
    return ((values - mean) / std).astype(np.float32)


def scene_minmax_unit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = values.min(axis=0, keepdims=True)
    high = values.max(axis=0, keepdims=True)
    return ((values - low) / np.maximum(high - low, eps)).astype(np.float32)


def _require_rows(name: str, values: np.ndarray, expected: int) -> np.ndarray:
    if len(values) != expected:
        raise ValueError(
            "{} row mismatch: expected {}, found {}".format(name, expected, len(values))
        )
    return values


def build_final_features(
    object_npz: PathLike,
    dino_npz: PathLike,
    view_normalizer: float = DEFAULT_VIEW_NORMALIZER,
) -> FeatureBundle:
    """Build the final 401-D feature vector in the original training order.

    Order: DINOv2 (384), normalized XYZ (3), normalized scale (3), rotation
    (4), normalized opacity (1), normalized RGB (3), and assignment quality
    (3). The historical model divides visible-view counts by 96 even though
    the selected object-crop extraction used at most 48 views.
    """

    if view_normalizer <= 0:
        raise ValueError("view_normalizer must be positive")

    with np.load(str(object_npz), allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        n = len(xyz)
        scale = _require_rows("scale", np.asarray(data["scale"], dtype=np.float32), n)
        rotation = _require_rows(
            "rotation", np.asarray(data["rotation"], dtype=np.float32), n
        )
        opacity = _require_rows(
            "opacity", np.asarray(data["opacity"], dtype=np.float32).reshape(-1, 1), n
        )
        color = _require_rows("color", np.asarray(data["color"], dtype=np.float32), n)
        labels = (
            np.asarray(data[LABEL_KEY], dtype=np.uint8).reshape(-1)
            if LABEL_KEY in data.files
            else None
        )

    if scale.shape[1:] != (3,) or rotation.shape[1:] != (4,) or color.shape[1:] != (3,):
        raise ValueError("Unexpected Gaussian parameter dimensions")

    gaussian = np.concatenate(
        [
            normalize_xyz(xyz),
            scene_zscore(scale),
            rotation,
            scene_minmax_unit(opacity),
            scene_zscore(color),
        ],
        axis=1,
    )

    with np.load(str(dino_npz), allow_pickle=False) as data:
        dino = _require_rows(
            "dino_features", np.asarray(data["dino_features"], dtype=np.float32), n
        ).copy()
        valid = _require_rows(
            "valid_dino", np.asarray(data["valid_dino"], dtype=bool).reshape(-1, 1), n
        )
        weight = _require_rows(
            "dino_weight_sum",
            np.asarray(data["dino_weight_sum"], dtype=np.float32).reshape(-1, 1),
            n,
        )
        views = _require_rows(
            "dino_visible_views",
            np.asarray(data["dino_visible_views"], dtype=np.float32).reshape(-1, 1),
            n,
        )

    dino[~valid[:, 0]] = 0.0
    quality = np.concatenate(
        [
            valid.astype(np.float32),
            scene_zscore(np.log1p(weight)),
            views / float(view_normalizer),
        ],
        axis=1,
    )
    features = np.concatenate([dino, gaussian, quality], axis=1).astype(
        np.float32, copy=False
    )

    if features.shape[1] != 401:
        raise ValueError(
            "Final checkpoint expects 401 features, found {}".format(features.shape[1])
        )
    if labels is not None and len(labels) != n:
        raise ValueError("Label row mismatch")
    return FeatureBundle(features=features, labels=labels, xyz=xyz)
