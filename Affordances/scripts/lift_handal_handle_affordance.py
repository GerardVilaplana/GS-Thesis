import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


C0 = 0.28209479177387814
HANDAL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
SCENE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_3dgs_scenes")
OBJECT_PLY_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_object_pruned_ply/thr0.75")
OUT_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_handle_affordance_ply/object_thr0.75")


def rgb_to_sh(rgb):
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


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
    valid_z = z > 1e-5

    u = np.full(len(points), -1.0, dtype=np.float64)
    v = np.full(len(points), -1.0, dtype=np.float64)
    u[valid_z] = cam_k[0] * cam[valid_z, 0] / z[valid_z] + cam_k[2]
    v[valid_z] = cam_k[4] * cam[valid_z, 1] / z[valid_z] + cam_k[5]

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    valid = valid_z
    valid &= ui >= 0
    valid &= ui < width
    valid &= vi >= 0
    valid &= vi < height
    return ui, vi, z, valid


def nearest_visible_indices(ui, vi, z, valid, width):
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return valid_idx

    pix = vi[valid_idx].astype(np.int64) * int(width) + ui[valid_idx].astype(np.int64)
    order = np.lexsort((z[valid_idx], pix))
    sorted_idx = valid_idx[order]
    sorted_pix = pix[order]
    first = np.r_[True, sorted_pix[1:] != sorted_pix[:-1]]
    return sorted_idx[first]


def color_affordance_ply(src_ply, dst_ply, is_handle):
    ply = PlyData.read(src_ply)
    vertices = np.array(ply["vertex"].data, copy=True)

    non_handle = rgb_to_sh([0.62, 0.62, 0.62])
    handle = rgb_to_sh([0.0, 0.9, 1.0])

    vertices["f_dc_0"] = non_handle[0]
    vertices["f_dc_1"] = non_handle[1]
    vertices["f_dc_2"] = non_handle[2]
    vertices["f_dc_0"][is_handle] = handle[0]
    vertices["f_dc_1"][is_handle] = handle[1]
    vertices["f_dc_2"][is_handle] = handle[2]

    ply["vertex"].data = vertices
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write(dst_ply)


def lift_scene(scene_name, threshold, min_visible):
    scene_dir = SCENE_ROOT / scene_name
    source = load_json(scene_dir / "handal_source.json")
    raw_scene = HANDAL_ROOT / source["split"] / scene_name
    scene_camera = load_json(raw_scene / "scene_camera.json")
    scene_gt = load_json(raw_scene / "scene_gt.json")

    src_ply = OBJECT_PLY_ROOT / f"handal_mug_scene_{scene_name}_object_pruned_thr0.75.ply"
    ply = PlyData.read(src_ply)
    points = np.vstack(
        [ply["vertex"].data["x"], ply["vertex"].data["y"], ply["vertex"].data["z"]]
    ).T.astype(np.float64)

    hits = np.zeros(len(points), dtype=np.uint16)
    visible = np.zeros(len(points), dtype=np.uint16)

    for frame_id in selected_frame_ids(scene_dir):
        key = str(frame_id)
        cam = scene_camera[key]
        gt = scene_gt[key][0]
        rot = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        trans = np.array(gt["cam_t_m2c"], dtype=np.float64) * float(source["unit_scale"])
        cam_k = np.array(cam["cam_K"], dtype=np.float64)
        width = int(cam["width"])
        height = int(cam["height"])

        handle_mask = load_mask(raw_scene / "mask_parts" / f"{frame_id:06d}_000000_handle.png")
        ui, vi, z, valid = project_points(points, rot, trans, cam_k, width, height)
        visible_idx = nearest_visible_indices(ui, vi, z, valid, width)
        visible[visible_idx] += 1

        inside = handle_mask[vi[visible_idx], ui[visible_idx]]
        hits[visible_idx[inside]] += 1

    score = np.divide(hits, visible, out=np.zeros(len(points), dtype=np.float32), where=visible > 0)
    is_handle = (visible >= min_visible) & (score >= threshold)

    out_dir = OUT_ROOT / f"handle_thr{threshold:.2f}"
    out_ply = out_dir / f"handal_mug_scene_{scene_name}_handle_affordance_thr{threshold:.2f}.ply"
    score_path = out_dir / f"handal_mug_scene_{scene_name}_handle_scores_thr{threshold:.2f}.npz"
    color_affordance_ply(src_ply, out_ply, is_handle)
    np.savez_compressed(
        score_path,
        handle_score=score,
        handle_hits=hits,
        visible_hits=visible,
        is_handle=is_handle,
        threshold=np.array([threshold], dtype=np.float32),
        min_visible=np.array([min_visible], dtype=np.int32),
    )

    return {
        "scene": scene_name,
        "input_gaussians": int(len(points)),
        "handle_gaussians": int(is_handle.sum()),
        "handle_ratio": float(is_handle.sum() / max(len(points), 1)),
        "threshold": threshold,
        "min_visible": min_visible,
        "ply": str(out_ply),
        "scores": str(score_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min_visible", type=int, default=10)
    parser.add_argument("--scenes", nargs="+", default=["001001", "002001", "003001", "005001", "006001"])
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"handle_thr{args.threshold:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [lift_scene(scene, args.threshold, args.min_visible) for scene in args.scenes]

    with open(out_dir / f"handle_affordance_summary_thr{args.threshold:.2f}.json", "w") as f:
        json.dump(summaries, f, indent=2)

    for item in summaries:
        print(
            f"{item['scene']}: handle {item['handle_gaussians']}/{item['input_gaussians']} "
            f"({item['handle_ratio']:.1%}) -> {item['ply']}"
        )


if __name__ == "__main__":
    main()
