#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


BASE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances")
RAW_ROOT = BASE_ROOT / "data" / "handal_dataset_mugs"
DATASET_ROOT = BASE_ROOT / "data" / "handal_handle_generalization_v1"


def read_manifest(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    fieldnames = [
        "split",
        "category",
        "scene_id",
        "source_path",
        "export_path",
        "instance_id",
        "num_rgb_images",
        "has_masks",
        "has_camera_metadata",
        "has_object_model",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["split"], r["category"], r["instance_id"], r["scene_id"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scene_ids_in_curated(dataset_root):
    out = {}
    subset = dataset_root / "subset_export"
    for scene_dir in subset.glob("*/mugs/[0-9][0-9][0-9][0-9][0-9][0-9]"):
        if scene_dir.is_dir():
            out[scene_dir.name] = scene_dir
    return out


def raw_static_scenes(raw_root):
    scenes = []
    for raw_part, target_split in [("train", "train_seen"), ("test", "test_seen")]:
        part_root = raw_root / raw_part
        if not part_root.exists():
            continue
        for scene_dir in sorted(part_root.iterdir(), key=lambda p: p.name):
            if scene_dir.is_dir() and scene_dir.name.isdigit() and len(scene_dir.name) == 6:
                scenes.append((raw_part, target_split, scene_dir))
    return scenes


def has_required_scene_files(scene_dir):
    return (
        (scene_dir / "rgb").is_dir()
        and (scene_dir / "mask").is_dir()
        and (scene_dir / "mask_visib").is_dir()
        and (scene_dir / "mask_parts").is_dir()
        and (scene_dir / "scene_camera.json").exists()
        and (scene_dir / "scene_gt.json").exists()
    )


def symlink_or_keep(src, dst):
    if dst.exists() or dst.is_symlink():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, default=RAW_ROOT)
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()

    manifest_path = args.dataset_root / "manifests" / "manifest.csv"
    rows = read_manifest(manifest_path)
    manifest_keys = {(r["category"], r["scene_id"]) for r in rows}
    curated = scene_ids_in_curated(args.dataset_root)

    added_rows = 0
    added_links = 0
    skipped_existing = 0
    skipped_bad = []

    for raw_part, target_split, scene_dir in raw_static_scenes(args.raw_root):
        scene_id = scene_dir.name
        if not has_required_scene_files(scene_dir):
            skipped_bad.append(scene_id)
            continue

        target_dir = args.dataset_root / "subset_export" / target_split / "mugs" / scene_id
        if scene_id not in curated:
            if symlink_or_keep(scene_dir, target_dir):
                added_links += 1
            curated[scene_id] = target_dir
        else:
            skipped_existing += 1
            target_dir = curated[scene_id]

        if ("mugs", scene_id) not in manifest_keys:
            rows.append(
                {
                    "split": target_split,
                    "category": "mugs",
                    "scene_id": scene_id,
                    "source_path": str(scene_dir),
                    "export_path": str(target_dir),
                    "instance_id": scene_id[:3],
                    "num_rgb_images": str(len(list((scene_dir / "rgb").glob("*.jpg")))),
                    "has_masks": "True",
                    "has_camera_metadata": "True",
                    "has_object_model": str((args.raw_root / "models" / f"obj_{int(scene_id[:3]):06d}.ply").exists()),
                    "notes": f"integrated_from_full_mugs_{raw_part}",
                }
            )
            manifest_keys.add(("mugs", scene_id))
            added_rows += 1

    write_manifest(manifest_path, rows)

    mug_rows = [r for r in rows if r["category"] == "mugs"]
    print(f"raw_static_scenes={len(raw_static_scenes(args.raw_root))}")
    print(f"manifest_total_rows={len(rows)}")
    print(f"manifest_mug_rows={len(mug_rows)}")
    print(f"added_manifest_rows={added_rows}")
    print(f"added_scene_symlinks={added_links}")
    print(f"already_curated_static_scenes={skipped_existing}")
    print(f"skipped_bad_scenes={len(skipped_bad)}")
    if skipped_bad:
        print("bad_scene_ids=" + " ".join(skipped_bad))


if __name__ == "__main__":
    main()
