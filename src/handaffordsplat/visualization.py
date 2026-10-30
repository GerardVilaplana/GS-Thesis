"""PLY visualization helpers using the thesis blue/green overlay convention."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from plyfile import PlyData, PlyElement

PathLike = Union[str, Path]
SH_C0 = 0.28209479177387814
BLUE = np.array([3.0, 40.0, 255.0], dtype=np.float32) / 255.0
GREEN = np.array([5.0, 215.0, 35.0], dtype=np.float32) / 255.0


def write_binary_overlay_ply(
    source_path: PathLike,
    output_path: PathLike,
    handle_prediction: np.ndarray,
    alpha: float = 0.70,
) -> None:
    """Blend binary predictions into the original Gaussian DC colors."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    prediction = np.asarray(handle_prediction, dtype=bool).reshape(-1)
    source = PlyData.read(str(source_path))
    vertices = np.array(source["vertex"].data, copy=True)
    if len(vertices) != len(prediction):
        raise ValueError(
            "PLY/prediction length mismatch: {} vs {}".format(
                len(vertices), len(prediction)
            )
        )

    dc = np.column_stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]
    ).astype(np.float32)
    base_rgb = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0)
    mask_rgb = np.where(prediction[:, None], GREEN, BLUE)
    blended = (1.0 - alpha) * base_rgb + alpha * mask_rgb
    blended_dc = ((blended - 0.5) / SH_C0).astype(np.float32)
    for channel in range(3):
        vertices["f_dc_{}".format(channel)] = blended_dc[:, channel]

    elements = [
        PlyElement.describe(vertices, "vertex")
        if element.name == "vertex"
        else element
        for element in source.elements
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=source.text, byte_order=source.byte_order).write(
        str(output_path)
    )
