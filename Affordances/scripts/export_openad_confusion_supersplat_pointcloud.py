#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

DEFAULT_ROOT = Path('/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/51_openad_pointcloud_baseline_v1')


def scene_radius(vertices: np.ndarray, radius_frac: float, min_radius: float, max_radius: float) -> float:
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
    diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    radius = diag * radius_frac
    return max(min_radius, min(max_radius, radius))


def convert_one(src: Path, dst: Path, radius_frac: float, min_radius: float, max_radius: float, opacity: float) -> dict:
    ply = PlyData.read(str(src))
    vertices = np.array(ply['vertex'].data, copy=True)
    names = vertices.dtype.names or ()
    required = ['scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity']
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f'{src} is missing required 3DGS fields: {missing}')

    radius = scene_radius(vertices, radius_frac, min_radius, max_radius)
    log_radius = math.log(radius)
    vertices['scale_0'] = log_radius
    vertices['scale_1'] = log_radius
    vertices['scale_2'] = log_radius
    vertices['rot_0'] = 1.0
    vertices['rot_1'] = 0.0
    vertices['rot_2'] = 0.0
    vertices['rot_3'] = 0.0
    vertices['opacity'] = math.log(opacity / (1.0 - opacity))

    for name in names:
        if name.startswith('f_rest_'):
            vertices[name] = 0.0

    elements = []
    for element in ply.elements:
        elements.append(PlyElement.describe(vertices, 'vertex') if element.name == 'vertex' else element)
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(dst))
    return {'source_ply': str(src), 'output_ply': str(dst), 'num_gaussians': len(vertices), 'sphere_radius': radius}


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert OpenAD confusion 3DGS PLYs to same-size spherical splats for SuperSplat point-cloud style visualization.')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--src_dir_name', default='confusion_plys')
    parser.add_argument('--dst_dir_name', default='confusion_ply_PC')
    parser.add_argument('--radius_frac', type=float, default=0.003)
    parser.add_argument('--min_radius', type=float, default=0.0007)
    parser.add_argument('--max_radius', type=float, default=0.003)
    parser.add_argument('--opacity', type=float, default=0.9)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    src_root = args.root / args.src_dir_name
    dst_root = args.root / args.dst_dir_name
    paths = sorted(src_root.rglob('*.ply'))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f'No PLY files found under {src_root}')

    rows = []
    for idx, src in enumerate(paths, start=1):
        dst = dst_root / src.relative_to(src_root)
        rows.append(convert_one(src, dst, args.radius_frac, args.min_radius, args.max_radius, args.opacity))
        if idx == 1 or idx == len(paths) or idx % 50 == 0:
            print(f'[{idx}/{len(paths)}] {src.name}', flush=True)

    manifest = args.root / 'metrics' / 'confusion_ply_PC_manifest.csv'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_ply', 'output_ply', 'num_gaussians', 'sphere_radius'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'[done] wrote {len(rows)} PLYs to {dst_root}')
    print(f'[done] manifest {manifest}')


if __name__ == '__main__':
    main()
