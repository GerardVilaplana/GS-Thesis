import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


HANDAL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_3dgs_scenes")
SCALE = 0.001


def rotmat2qvec(rot):
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0, 0, 0],
            [ryx + rxy, ryy - rxx - rzz, 0, 0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_scene_json(scene_dir, name):
    with open(scene_dir / name, "r") as f:
        return json.load(f)


def symlink_or_replace(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def write_cameras_txt(path, scene_camera):
    first = scene_camera[sorted(scene_camera.keys(), key=lambda x: int(x))[0]]
    fx, _, cx, _, fy, cy, _, _, _ = first["cam_K"]
    width = int(first["width"])
    height = int(first["height"])
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}\n")


def write_images_txt(path, selected_frames, scene_gt):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(selected_frames)}, mean observations per image: 0\n")
        for image_id, frame_id in enumerate(selected_frames, start=1):
            gt = scene_gt[str(frame_id)][0]
            rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * SCALE
            qvec = rotmat2qvec(rot)
            image_name = f"{frame_id:06d}.jpg"
            values = [image_id, *qvec.tolist(), *trans.tolist(), 1, image_name]
            f.write(
                f"{values[0]} "
                f"{values[1]:.12f} {values[2]:.12f} {values[3]:.12f} {values[4]:.12f} "
                f"{values[5]:.12f} {values[6]:.12f} {values[7]:.12f} "
                f"{values[8]} {values[9]}\n\n"
            )


def write_points_ply(path, obj_id, max_points=30000):
    model_path = HANDAL_ROOT / "models" / f"obj_{obj_id:06d}.ply"
    ply = PlyData.read(model_path)
    vertex = ply["vertex"]
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32) * SCALE
    if {"red", "green", "blue"}.issubset(vertex.data.dtype.names):
        rgb = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points).round().astype(np.int64)
        xyz = xyz[idx]
        rgb = rgb[idx]

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz, dtype=np.float32)
    out = np.empty(len(xyz), dtype=dtype)
    out[:] = list(map(tuple, np.concatenate([xyz, normals, rgb.astype(np.float32)], axis=1)))
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(path)


def prepare_scene(split, scene_name, max_images=96):
    src_scene = HANDAL_ROOT / split / scene_name
    dst_scene = OUT_ROOT / scene_name
    images_dir = dst_scene / "images"
    sparse_dir = dst_scene / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    scene_camera = read_scene_json(src_scene, "scene_camera.json")
    scene_gt = read_scene_json(src_scene, "scene_gt.json")
    frame_ids = sorted(int(p.stem) for p in (src_scene / "rgb").glob("*.jpg"))
    if len(frame_ids) > max_images:
        idx = np.linspace(0, len(frame_ids) - 1, max_images).round().astype(np.int64)
        frame_ids = [frame_ids[i] for i in idx]

    first_gt = scene_gt[str(frame_ids[0])][0]
    obj_id = int(first_gt["obj_id"])

    for frame_id in frame_ids:
        symlink_or_replace(src_scene / "rgb" / f"{frame_id:06d}.jpg", images_dir / f"{frame_id:06d}.jpg")

    write_cameras_txt(sparse_dir / "cameras.txt", scene_camera)
    write_images_txt(sparse_dir / "images.txt", frame_ids, scene_gt)
    write_points_ply(sparse_dir / "points3D.ply", obj_id)

    with open(dst_scene / "handal_source.json", "w") as f:
        json.dump(
            {
                "source_scene": str(src_scene),
                "split": split,
                "scene": scene_name,
                "obj_id": obj_id,
                "num_images": len(frame_ids),
                "unit_scale": SCALE,
            },
            f,
            indent=2,
        )

    return dst_scene, obj_id, len(frame_ids)


def main():
    scenes = [
        ("train", "001001"),
        ("train", "002001"),
        ("train", "003001"),
        ("train", "005001"),
        ("train", "006001"),
    ]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split, scene_name in scenes:
        dst, obj_id, num_images = prepare_scene(split, scene_name)
        print(f"{scene_name}: obj_id={obj_id}, images={num_images}, dst={dst}")


if __name__ == "__main__":
    main()
