import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


HANDAL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
SCENE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_3dgs_scenes")
PLY_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_3dgs_ply")
OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_object_pruned_ply")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_mask(path):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def selected_frame_ids(scene_dir):
    return sorted(int(p.stem) for p in (scene_dir / "images").glob("*.jpg"))


def project_points(points, rot, trans, cam_k, width, height):
    cam = points @ rot.T + trans[None, :]
    z = cam[:, 2]
    valid = z > 1e-5

    u = cam_k[0] * cam[:, 0] / np.maximum(z, 1e-8) + cam_k[2]
    v = cam_k[4] * cam[:, 1] / np.maximum(z, 1e-8) + cam_k[5]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    valid &= ui >= 0
    valid &= ui < width
    valid &= vi >= 0
    valid &= vi < height
    return ui, vi, valid


def write_filtered_ply(src_ply, dst_ply, keep):
    ply = PlyData.read(src_ply)
    vertices = ply["vertex"].data
    filtered = vertices[keep]
    ply["vertex"].data = filtered
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(dst_ply)


def prune_scene(scene_name, threshold, min_visible):
    scene_dir = SCENE_ROOT / scene_name
    source = load_json(scene_dir / "handal_source.json")
    split = source["split"]
    raw_scene = HANDAL_ROOT / split / scene_name
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")

    src_ply = PLY_ROOT / f"handal_mug_scene_{scene_name}_iter3000.ply"
    ply = PlyData.read(src_ply)
    points = np.vstack(
        [ply["vertex"].data["x"], ply["vertex"].data["y"], ply["vertex"].data["z"]]
    ).T.astype(np.float64)

    hits = np.zeros(len(points), dtype=np.uint16)
    visible = np.zeros(len(points), dtype=np.uint16)

    for frame_id in selected_frame_ids(scene_dir):
        frame_key = str(frame_id)
        cam = scene_camera[frame_key]
        gt = scene_gt[frame_key][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        width = int(cam["width"])
        height = int(cam["height"])
        mask = load_mask(raw_scene / "mask" / f"{frame_id:06d}_000000.png")

        ui, vi, valid = project_points(points, rot, trans, cam_k, width, height)
        valid_idx = np.flatnonzero(valid)
        visible[valid_idx] += 1
        inside_idx = valid_idx[mask[vi[valid_idx], ui[valid_idx]]]
        hits[inside_idx] += 1

    score = np.divide(hits, visible, out=np.zeros(len(points), dtype=np.float32), where=visible > 0)
    keep = (visible >= min_visible) & (score >= threshold)

    out_ply = OUT_ROOT / f"handal_mug_scene_{scene_name}_object_pruned_thr{threshold:.2f}.ply"
    write_filtered_ply(src_ply, out_ply, keep)

    score_path = OUT_ROOT / f"handal_mug_scene_{scene_name}_object_scores.npz"
    np.savez_compressed(
        score_path,
        object_score=score,
        object_hits=hits,
        visible_hits=visible,
        keep=keep,
        threshold=np.array([threshold], dtype=np.float32),
        min_visible=np.array([min_visible], dtype=np.int32),
    )

    return {
        "scene": scene_name,
        "input": int(len(points)),
        "kept": int(keep.sum()),
        "threshold": threshold,
        "min_visible": min_visible,
        "ply": str(out_ply),
        "scores": str(score_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min_visible", type=int, default=20)
    parser.add_argument("--scenes", nargs="+", default=["001001", "002001", "003001", "005001", "006001"])
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [prune_scene(scene, args.threshold, args.min_visible) for scene in args.scenes]
    with open(OUT_ROOT / "object_pruning_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    for item in summaries:
        ratio = item["kept"] / max(item["input"], 1)
        print(f"{item['scene']}: kept {item['kept']}/{item['input']} ({ratio:.1%}) -> {item['ply']}")


if __name__ == "__main__":
    main()
