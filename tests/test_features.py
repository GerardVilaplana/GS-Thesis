from pathlib import Path

import numpy as np

from handaffordsplat.features import build_final_features


def test_final_feature_shape_and_invalid_dino(tmp_path: Path):
    n = 5
    object_path = tmp_path / "object.npz"
    dino_path = tmp_path / "dino.npz"
    np.savez_compressed(
        object_path,
        xyz=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        scale=np.linspace(0.1, 1.0, n * 3, dtype=np.float32).reshape(n, 3),
        rotation=np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (n, 1)),
        opacity=np.linspace(0.1, 0.9, n, dtype=np.float32),
        color=np.linspace(0, 1, n * 3, dtype=np.float32).reshape(n, 3),
        handle_labels_thr0_25=np.array([0, 1, 0, 1, 0], dtype=np.uint8),
    )
    dino = np.ones((n, 384), dtype=np.float32)
    valid = np.array([1, 1, 0, 1, 1], dtype=np.uint8)
    np.savez_compressed(
        dino_path,
        dino_features=dino,
        valid_dino=valid,
        dino_weight_sum=np.arange(1, n + 1, dtype=np.float32),
        dino_visible_views=np.arange(n, dtype=np.float32),
    )

    bundle = build_final_features(object_path, dino_path)
    assert bundle.features.shape == (n, 401)
    assert bundle.labels.tolist() == [0, 1, 0, 1, 0]
    assert np.all(bundle.features[2, :384] == 0)
    assert np.isclose(bundle.features[4, -1], 4 / 96.0)
