#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

BASE_ROOT = Path('/home/gvilaplana/GS-Thesis/Affordances')
SCRIPT_DIR = BASE_ROOT / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))

import train_handal_exp44_pointnet_mean_feature_sweep as exp44  # noqa: E402

SH_FEATURE_CHOICES = [
    'xyz_sh',
    'xyz_scale_opacity_sh',
    'geometry_sh_scene_norm',
    'dino_geometry_sh',
    'dino_geometry_sh_quality',
]

_sh_cache: dict[str, np.ndarray] = {}


def _field_index(name: str) -> tuple[int, int]:
    if name.startswith('f_dc_'):
        return (0, int(name.rsplit('_', 1)[1]))
    match = re.match(r'f_rest_(\d+)$', name)
    if match:
        return (1, int(match.group(1)))
    return (2, 0)


def load_sh(item: dict, n: int) -> np.ndarray:
    source = item.get('source_object_ply', '')
    if not source:
        raise FileNotFoundError(f'Missing source_object_ply for {item["scene_key"]}')
    if source not in _sh_cache:
        vertices = PlyData.read(source)['vertex'].data
        names = vertices.dtype.names or ()
        fields = sorted([x for x in names if x.startswith('f_dc_') or x.startswith('f_rest_')], key=_field_index)
        if not fields:
            raise ValueError(f'No SH fields found in {source}')
        _sh_cache[source] = np.vstack([vertices[name] for name in fields]).T.astype(np.float32)
    sh = _sh_cache[source]
    if len(sh) != n:
        raise ValueError(f'SH length mismatch for {item["scene_key"]}: {len(sh)} vs {n}')
    return sh


def load_feature_arrays_variant(item: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(item['npz'], allow_pickle=False) as z:
        xyz = z['xyz'].astype(np.float32)
        xyz_norm = exp44.g15.base.normalize_xyz(xyz)
        scale = z['scale'].astype(np.float32)
        rotation = z['rotation'].astype(np.float32)
        opacity = np.asarray(z['opacity'], dtype=np.float32).reshape(-1, 1)
        y = z['handle_labels_thr0_25'].astype(np.float32)

    sh = load_sh(item, len(y))
    sh_norm = exp44.g15.scene_zscore(sh)
    geom_sh = np.concatenate(
        [xyz_norm, exp44.g15.scene_zscore(scale), rotation, exp44.g15.scene_minmax_unit(opacity), sh_norm],
        axis=1,
    )

    variant = exp44.ACTIVE_FEATURE_VARIANT
    if variant == 'xyz_sh':
        x = np.concatenate([xyz_norm, sh_norm], axis=1)
    elif variant == 'xyz_scale_opacity_sh':
        x = np.concatenate([xyz_norm, exp44.g15.scene_zscore(scale), exp44.g15.scene_minmax_unit(opacity), sh_norm], axis=1)
    elif variant == 'geometry_sh_scene_norm':
        x = geom_sh
    elif variant in {'dino_geometry_sh', 'dino_geometry_sh_quality'}:
        dino, valid, weight, views = exp44.load_dino_arrays(item, len(y))
        if variant == 'dino_geometry_sh':
            x = np.concatenate([dino, geom_sh], axis=1)
        else:
            quality = np.concatenate([valid, exp44.g15.scene_zscore(np.log1p(weight)), views / 96.0], axis=1)
            x = np.concatenate([dino, geom_sh, quality], axis=1)
    else:
        raise ValueError(f'Unknown SH feature variant: {variant}')
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def main() -> None:
    exp44.FEATURE_CHOICES = SH_FEATURE_CHOICES
    exp44.OUT_ROOT = BASE_ROOT / 'outputs' / '03_handle_generalization' / '53_exp40_pointnet_mean_sh_ablation_seen_v1'
    exp44.load_feature_arrays_variant = load_feature_arrays_variant
    exp44.export_qa_plys = lambda *args, **kwargs: None
    exp44.main()


if __name__ == '__main__':
    main()
