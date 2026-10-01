import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def id2rgb(idx, max_num_obj=256):
    if not 0 <= idx <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    rgb = np.zeros((3,), dtype=np.uint8)
    if idx == 0:
        return rgb
    golden_ratio = 1.6180339887
    h = (idx * golden_ratio) % 1
    s = 0.5 + (idx % 2) * 0.5
    l = 0.5
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)
    return rgb


def rgb_to_sh(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    return (rgb - 0.5) / C0


def color_to_id(rgb, num_classes):
    rgb = np.asarray(rgb, dtype=np.uint8)
    for idx in range(num_classes):
        if np.array_equal(id2rgb(idx, num_classes), rgb):
            return idx
    raise ValueError(f"Could not map RGB color {rgb.tolist()} to an id in [0, {num_classes})")


def slugify(text):
    return text.lower().replace(" ", "_").replace("/", "_")


def load_query_colors(summary_path, query, top_ids):
    summary = json.loads(Path(summary_path).read_text())
    query_info = summary["queries"][query]
    colors = query_info.get("global_selected_colors_rgb", [])
    if len(colors) < top_ids and "color_votes_top20" in query_info:
        colors = [item["rgb"] for item in query_info["color_votes_top20"]]
    return colors[:top_ids]


def predict_gaussian_ids(vertex_data, classifier_path, num_classes):
    obj = np.stack([vertex_data[f"obj_dc_{i}"] for i in range(16)], axis=0).astype(np.float32)
    classifier = torch.nn.Conv2d(16, num_classes, kernel_size=1)
    state = torch.load(classifier_path, map_location="cpu")
    classifier.load_state_dict(state)
    classifier.eval()
    with torch.no_grad():
        x = torch.from_numpy(obj)[None, :, :, None]
        pred = classifier(x).argmax(dim=1).squeeze(0).squeeze(-1).numpy().astype(np.int32)
    return pred


def write_full_highlight_ply(ply_data, selected_mask, output_path, highlight_rgb, dim_factor):
    vertex = ply_data["vertex"].data.copy()
    highlight_sh = rgb_to_sh(highlight_rgb)
    vertex["f_dc_0"][selected_mask] = highlight_sh[0]
    vertex["f_dc_1"][selected_mask] = highlight_sh[1]
    vertex["f_dc_2"][selected_mask] = highlight_sh[2]
    if dim_factor < 1.0:
        vertex["f_dc_0"][~selected_mask] *= dim_factor
        vertex["f_dc_1"][~selected_mask] *= dim_factor
        vertex["f_dc_2"][~selected_mask] *= dim_factor
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply_data.text).write(output_path)


def write_point_mask_ply(vertex_data, selected_mask, output_path, highlight_rgb):
    xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)
    colors = np.full((len(xyz), 3), 185, dtype=np.uint8)
    colors[selected_mask] = np.asarray(highlight_rgb, dtype=np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    out = np.empty(len(xyz), dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["red"], out["green"], out["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(out, "vertex")], text=True).write(output_path)


def write_selected_points_ply(vertex_data, selected_mask, output_path, highlight_rgb):
    xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)
    xyz = xyz[selected_mask]
    colors = np.tile(np.asarray(highlight_rgb, dtype=np.uint8), (len(xyz), 1))
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    out = np.empty(len(xyz), dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["red"], out["green"], out["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(out, "vertex")], text=True).write(output_path)


def main():
    parser = argparse.ArgumentParser(description="Export a REALM text-query result as highlighted Gaussian and point PLY files.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--summary_json", default="output/lerf/figurines/qwen_sam_realm_stage_videos_stride3/qwen_sam_realm_stage_summary.json")
    parser.add_argument("--point_cloud", default="output/lerf/figurines/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--classifier", default="output/lerf/figurines/point_cloud/iteration_30000/classifier.pth")
    parser.add_argument("--output_dir", default="output/lerf/figurines/query_ply")
    parser.add_argument("--num_classes", type=int, default=256)
    parser.add_argument("--top_ids", type=int, default=1)
    parser.add_argument("--highlight_rgb", nargs=3, type=int, default=[255, 0, 0])
    parser.add_argument("--dim_factor", type=float, default=0.25)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.query)

    colors = load_query_colors(args.summary_json, args.query, args.top_ids)
    target_ids = [color_to_id(c, args.num_classes) for c in colors]

    ply_data = PlyData.read(args.point_cloud)
    vertex = ply_data["vertex"].data
    pred_ids = predict_gaussian_ids(vertex, args.classifier, args.num_classes)
    selected = np.isin(pred_ids, target_ids)

    full_path = out_root / f"{slug}_realm_top{args.top_ids}_highlight_3dgs.ply"
    point_path = out_root / f"{slug}_realm_top{args.top_ids}_highlight_points.ply"
    selected_path = out_root / f"{slug}_realm_top{args.top_ids}_selected_points_only.ply"
    meta_path = out_root / f"{slug}_realm_top{args.top_ids}_metadata.json"

    write_full_highlight_ply(ply_data, selected, full_path, args.highlight_rgb, args.dim_factor)
    write_point_mask_ply(vertex, selected, point_path, args.highlight_rgb)
    write_selected_points_ply(vertex, selected, selected_path, args.highlight_rgb)

    metadata = {
        "query": args.query,
        "summary_json": str(args.summary_json),
        "source_point_cloud": str(args.point_cloud),
        "classifier": str(args.classifier),
        "selected_visualization_colors_rgb": colors,
        "target_class_ids": target_ids,
        "num_gaussians": int(len(vertex)),
        "num_selected_gaussians": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "outputs": {
            "full_3dgs_highlight": str(full_path),
            "meshlab_points_highlight": str(point_path),
            "selected_points_only": str(selected_path),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
