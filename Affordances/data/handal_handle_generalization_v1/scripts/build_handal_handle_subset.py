#!/usr/bin/env python3
"""Build a curated HANDAL handle/generalization subset.

The script is intentionally conservative:
- original zip files are never modified;
- raw archives are extracted once into raw_unzipped/<category>;
- generated subset/manifests are rebuilt deterministically with seed 42;
- scene ids must be exact six-digit folders, with instance_id = scene_id[:3].
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable


SEED = 42
PROJECT_NAME = "handal_handle_generalization_v1"

SEEN_CATEGORIES = [
    "mugs",
    "measuring_cups",
    "hammers",
    "screwdrivers",
    "adjustable_wrenches",
]

UNSEEN_CATEGORIES = [
    "pots_pans",
    "power_drills",
    "spatulas",
    "whisks",
]

CATEGORY_ZIPS = {
    "adjustable_wrenches": "handal_dataset_adjustable_wrenches_no_depth.zip",
    "hammers": "handal_dataset_hammers_no_depth.zip",
    "measuring_cups": "handal_dataset_measuring_cups_no_depth.zip",
    "mugs": "handal_dataset_mugs_no_depth.zip",
    "pots_pans": "handal_dataset_pots_pans_no_depth.zip",
    "power_drills": "handal_dataset_power_drills_no_depth.zip",
    "screwdrivers": "handal_dataset_screwdrivers_no_depth.zip",
    "spatulas": "handal_dataset_spatulas_no_depth.zip",
    "whisks": "handal_dataset_whisks_no_depth.zip",
}

SCENE_ID_RE = re.compile(r"^\d{6}$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RGB_DIR_NAMES = {"rgb", "image", "images", "color", "color_images"}
MASK_HINTS = ("mask", "annotation", "annotations", "label", "labels", "seg")
CAMERA_HINTS = ("scene_camera", "camera", "cameras", "intrinsic", "intrinsics", "transforms", "pose", "poses")
SCENE_META_NAMES = {"scene_gt.json", "scene_gt_info.json"}
SCENE_CONTAINER_NAMES = {"train", "test", "dynamic"}


@dataclass(frozen=True)
class SceneInfo:
    category: str
    scene_id: str
    source_path: Path
    instance_id: str
    num_rgb_images: int
    has_masks: bool
    has_camera_metadata: bool
    has_object_model: bool
    notes: str = ""


@dataclass
class ExportRecord:
    split: str
    scene: SceneInfo
    export_path: Path
    notes: str = ""


@dataclass
class ValidationResult:
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Workspace containing zips/ and the handal_handle_generalization_v1 project folder.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-unzip", action="store_true", help="Reuse existing raw_unzipped folders only.")
    parser.add_argument(
        "--keep-existing-export",
        action="store_true",
        help="Do not remove existing subset_export/manifests before exporting.",
    )
    return parser.parse_args()


def project_paths(workspace: Path) -> dict[str, Path]:
    project = workspace / PROJECT_NAME
    return {
        "workspace": workspace,
        "zips": workspace / "zips",
        "project": project,
        "raw": project / "raw_unzipped",
        "export": project / "subset_export",
        "manifests": project / "manifests",
        "scripts": project / "scripts",
    }


def ensure_project_dirs(paths: dict[str, Path]) -> None:
    for key in ("raw", "export", "manifests", "scripts"):
        paths[key].mkdir(parents=True, exist_ok=True)


def zip_signature(zip_path: Path) -> dict[str, int | str]:
    stat = zip_path.stat()
    return {"zip_name": zip_path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sentinel_matches(dest: Path, zip_path: Path) -> bool:
    sentinel = dest / ".unzipped_complete.json"
    if not dest.exists() or not sentinel.exists():
        return False
    try:
        current = json.loads(sentinel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return current == zip_signature(zip_path)


def strip_zip_root(member_name: str) -> Path | None:
    parts = PurePosixPath(member_name).parts
    if not parts:
        return None
    stripped = parts[1:] if len(parts) > 1 else ()
    if not stripped:
        return None
    if any(part in ("", ".", "..") for part in stripped):
        raise ValueError(f"Unsafe zip member path: {member_name}")
    return Path(*stripped)


def extract_zip_stripping_root(zip_path: Path, dest: Path) -> None:
    tmp_dest = dest.with_name(f".extracting_{dest.name}")
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest)
    tmp_dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        total = len(members)
        last_report = time.monotonic()
        for idx, info in enumerate(members, start=1):
            rel = strip_zip_root(info.filename)
            if rel is None:
                continue
            target = tmp_dest / rel
            if not str(target.resolve()).startswith(str(tmp_dest.resolve())):
                raise ValueError(f"Unsafe extraction target: {target}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            date_time = time.mktime(info.date_time + (0, 0, -1))
            try:
                os.utime(target, (date_time, date_time))
            except OSError:
                pass
            now = time.monotonic()
            if now - last_report > 20:
                print(f"  extracted {idx:,}/{total:,} entries from {zip_path.name}", flush=True)
                last_report = now

    (tmp_dest / ".unzipped_complete.json").write_text(
        json.dumps(zip_signature(zip_path), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if dest.exists():
        shutil.rmtree(dest)
    last_error: OSError | None = None
    for _ in range(6):
        try:
            tmp_dest.rename(dest)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
            time.sleep(2)
    if last_error is not None:
        raise last_error


def unzip_all(paths: dict[str, Path], skip_unzip: bool) -> None:
    for category, zip_name in CATEGORY_ZIPS.items():
        zip_path = paths["zips"] / zip_name
        dest = paths["raw"] / category
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing zip for {category}: {zip_path}")
        if sentinel_matches(dest, zip_path):
            print(f"[unzip] {category}: already complete")
            continue
        if skip_unzip:
            raise FileNotFoundError(f"{dest} is not marked complete and --skip-unzip was used")
        print(f"[unzip] {category}: extracting {zip_path.name} -> {dest}")
        extract_zip_stripping_root(zip_path, dest)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def count_rgb_images(scene_path: Path) -> int:
    count = 0
    for path in iter_files(scene_path):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_parts = {part.lower() for part in path.relative_to(scene_path).parts[:-1]}
        if rel_parts & RGB_DIR_NAMES:
            count += 1
    return count


def contains_hint_file(scene_path: Path, hints: tuple[str, ...], exact_names: set[str] | None = None) -> bool:
    exact_names = exact_names or set()
    for path in iter_files(scene_path):
        rel_lower = "/".join(part.lower() for part in path.relative_to(scene_path).parts)
        if path.name.lower() in exact_names:
            return True
        if any(hint in rel_lower for hint in hints):
            return True
    return False


def category_has_object_model(category_path: Path) -> bool:
    for model_dir_name in ("models", "models_parts", "model", "meshes", "object_models"):
        model_dir = category_path / model_dir_name
        if model_dir.exists() and any(model_dir.rglob("*")):
            return True
    return False


def discover_scenes(paths: dict[str, Path]) -> tuple[dict[str, list[SceneInfo]], list[str]]:
    scenes_by_category: dict[str, list[SceneInfo]] = {}
    warnings: list[str] = []
    all_categories = SEEN_CATEGORIES + UNSEEN_CATEGORIES

    for category in all_categories:
        category_path = paths["raw"] / category
        if not category_path.exists():
            warnings.append(f"{category}: raw folder missing: {category_path}")
            scenes_by_category[category] = []
            continue

        has_model = category_has_object_model(category_path)
        candidates = sorted(path for path in category_path.rglob("*") if path.is_dir() and SCENE_ID_RE.fullmatch(path.name))
        category_scenes: list[SceneInfo] = []

        for scene_path in candidates:
            scene_id = scene_path.name
            num_rgb = count_rgb_images(scene_path)
            has_masks = contains_hint_file(scene_path, MASK_HINTS, SCENE_META_NAMES)
            has_camera = contains_hint_file(scene_path, CAMERA_HINTS)
            notes: list[str] = []
            if num_rgb == 0:
                notes.append("missing_rgb")
            if not has_masks:
                notes.append("missing_masks_or_annotations")
            if not has_camera:
                notes.append("missing_camera_metadata")
            if not has_model:
                notes.append("missing_category_object_model")

            if num_rgb > 0:
                category_scenes.append(
                    SceneInfo(
                        category=category,
                        scene_id=scene_id,
                        source_path=scene_path,
                        instance_id=scene_id[:3],
                        num_rgb_images=num_rgb,
                        has_masks=has_masks,
                        has_camera_metadata=has_camera,
                        has_object_model=has_model,
                        notes=";".join(notes),
                    )
                )
            else:
                warnings.append(f"{category}/{scene_id}: candidate skipped because no RGB/images were found")

        scenes_by_category[category] = category_scenes
        ignored_dynamic = sorted(
            path.name
            for path in (category_path / "dynamic").glob("*")
            if (category_path / "dynamic").exists() and path.is_dir() and not SCENE_ID_RE.fullmatch(path.name)
        )
        if ignored_dynamic:
            warnings.append(
                f"{category}: ignored {len(ignored_dynamic)} dynamic non-six-digit scene folders "
                f"(examples: {', '.join(ignored_dynamic[:5])})"
            )

    return scenes_by_category, warnings


def group_by_instance(scenes: list[SceneInfo]) -> dict[str, list[SceneInfo]]:
    grouped: dict[str, list[SceneInfo]] = {}
    for scene in sorted(scenes, key=lambda s: s.scene_id):
        grouped.setdefault(scene.instance_id, []).append(scene)
    return grouped


def split_seen_instances(instance_ids: list[str], rng: random.Random) -> dict[str, list[str]]:
    ids = list(instance_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return {"train_seen": [], "val_seen": [], "test_seen": []}
    if n == 1:
        return {"train_seen": ids, "val_seen": [], "test_seen": []}
    if n == 2:
        return {"train_seen": ids[:1], "val_seen": [], "test_seen": ids[1:]}

    n_val = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.20))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    return {
        "train_seen": ids[:n_train],
        "val_seen": ids[n_train : n_train + n_val],
        "test_seen": ids[n_train + n_val :],
    }


def choose_scenes_for_instances(
    grouped: dict[str, list[SceneInfo]],
    instance_ids: list[str],
    max_scenes_per_instance: int,
    rng: random.Random,
) -> list[SceneInfo]:
    ordered_instances = list(instance_ids)
    selected: list[SceneInfo] = []
    shuffled_scenes: dict[str, list[SceneInfo]] = {}
    for instance_id in ordered_instances:
        options = list(grouped[instance_id])
        rng.shuffle(options)
        shuffled_scenes[instance_id] = options[:max_scenes_per_instance]

    for scene_offset in range(max_scenes_per_instance):
        for instance_id in ordered_instances:
            options = shuffled_scenes.get(instance_id, [])
            if scene_offset < len(options):
                selected.append(options[scene_offset])
    return selected


def select_subset(
    scenes_by_category: dict[str, list[SceneInfo]],
    seed: int,
) -> tuple[list[tuple[str, SceneInfo]], dict]:
    rng = random.Random(seed)
    selections: list[tuple[str, SceneInfo]] = []
    config: dict = {
        "seed": seed,
        "scene_id_rule": "scene folders must be exact six-digit ids; instance_id = scene_id[:3]",
        "seen_categories": SEEN_CATEGORIES,
        "unseen_categories": UNSEEN_CATEGORIES,
        "seen_split_targets": {
            "train_seen": {"instance_fraction": 0.70, "max_scenes_per_instance": 3},
            "val_seen": {"instance_fraction": 0.10, "max_scenes_per_instance": 2},
            "test_seen": {"instance_fraction": 0.20, "max_scenes_per_instance": 2},
        },
        "unseen_test_target": {"max_instances_per_category": 20, "max_scenes_per_instance": 2},
        "category_details": {},
    }

    for category in SEEN_CATEGORIES:
        grouped = group_by_instance(scenes_by_category.get(category, []))
        instance_ids = sorted(grouped)
        split_ids = split_seen_instances(instance_ids, rng)
        config["category_details"][category] = {
            "available_instances": len(instance_ids),
            "available_scenes": len(scenes_by_category.get(category, [])),
            "split_instances": {split: ids for split, ids in split_ids.items()},
        }
        for split, max_scenes in (("train_seen", 3), ("val_seen", 2), ("test_seen", 2)):
            chosen = choose_scenes_for_instances(grouped, split_ids[split], max_scenes, rng)
            selections.extend((split, scene) for scene in chosen)

    for category in UNSEEN_CATEGORIES:
        grouped = group_by_instance(scenes_by_category.get(category, []))
        instance_ids = sorted(grouped)
        rng.shuffle(instance_ids)
        chosen_instances = instance_ids[:20]
        config["category_details"][category] = {
            "available_instances": len(instance_ids),
            "available_scenes": len(scenes_by_category.get(category, [])),
            "selected_instances": chosen_instances,
        }
        chosen = choose_scenes_for_instances(grouped, chosen_instances, 2, rng)
        selections.extend(("test_unseen_category", scene) for scene in chosen)

    return selections, config


def remove_generated_outputs(paths: dict[str, Path], keep_existing_export: bool) -> None:
    if keep_existing_export:
        return
    for key in ("export", "manifests"):
        target = paths[key]
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def copy_category_support_files(category_path: Path, export_category_path: Path) -> None:
    export_category_path.mkdir(parents=True, exist_ok=True)
    for child in category_path.iterdir():
        if child.name == ".unzipped_complete.json":
            continue
        if child.name in SCENE_CONTAINER_NAMES:
            continue
        dest = export_category_path / child.name
        if child.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(child, dest)
        elif child.is_file():
            shutil.copy2(child, dest)


def export_subset(paths: dict[str, Path], selections: list[tuple[str, SceneInfo]]) -> list[ExportRecord]:
    records: list[ExportRecord] = []
    copied_support: set[tuple[str, str]] = set()

    for split, scene in selections:
        category_export_root = paths["export"] / split / scene.category
        support_key = (split, scene.category)
        if support_key not in copied_support:
            copy_category_support_files(paths["raw"] / scene.category, category_export_root)
            copied_support.add(support_key)

        export_path = category_export_root / scene.scene_id
        if export_path.exists():
            shutil.rmtree(export_path)
        shutil.copytree(scene.source_path, export_path)
        records.append(ExportRecord(split=split, scene=scene, export_path=export_path))

    return records


def validate_records(records: list[ExportRecord]) -> ValidationResult:
    result = ValidationResult()
    scene_split_seen: dict[tuple[str, str], str] = {}
    instance_split_seen: dict[tuple[str, str], str] = {}

    for record in records:
        scene = record.scene
        rel_scene_key = (scene.category, scene.scene_id)
        previous_split = scene_split_seen.get(rel_scene_key)
        if previous_split and previous_split != record.split:
            result.warn(f"{scene.category}/{scene.scene_id}: appears in both {previous_split} and {record.split}")
        scene_split_seen[rel_scene_key] = record.split

        if record.split in {"train_seen", "val_seen", "test_seen"}:
            inst_key = (scene.category, scene.instance_id)
            previous_inst_split = instance_split_seen.get(inst_key)
            if previous_inst_split and previous_inst_split != record.split:
                result.warn(
                    f"{scene.category} instance {scene.instance_id}: appears in both "
                    f"{previous_inst_split} and {record.split}"
                )
            instance_split_seen[inst_key] = record.split

        if not record.export_path.exists():
            result.warn(f"{record.split}/{scene.category}/{scene.scene_id}: exported scene folder missing")
            continue

        num_rgb = count_rgb_images(record.export_path)
        has_masks = contains_hint_file(record.export_path, MASK_HINTS, SCENE_META_NAMES)
        has_camera = contains_hint_file(record.export_path, CAMERA_HINTS)
        if num_rgb == 0:
            result.warn(f"{record.split}/{scene.category}/{scene.scene_id}: no RGB/images in export")
        if not has_masks:
            result.warn(f"{record.split}/{scene.category}/{scene.scene_id}: no mask/annotation files in export")
        if not has_camera:
            result.warn(f"{record.split}/{scene.category}/{scene.scene_id}: no camera/metadata files in export")

    return result


def write_manifest(paths: dict[str, Path], records: list[ExportRecord]) -> None:
    manifest_path = paths["manifests"] / "manifest.csv"
    fields = [
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
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r.split, r.scene.category, r.scene.scene_id)):
            scene = record.scene
            writer.writerow(
                {
                    "split": record.split,
                    "category": scene.category,
                    "scene_id": scene.scene_id,
                    "source_path": str(scene.source_path),
                    "export_path": str(record.export_path),
                    "instance_id": scene.instance_id,
                    "num_rgb_images": scene.num_rgb_images,
                    "has_masks": scene.has_masks,
                    "has_camera_metadata": scene.has_camera_metadata,
                    "has_object_model": scene.has_object_model,
                    "notes": ";".join(part for part in (scene.notes, record.notes) if part),
                }
            )


def summarize(records: list[ExportRecord]) -> list[dict[str, str | int]]:
    grouped: dict[tuple[str, str], dict[str, set[str] | int]] = {}
    for record in records:
        key = (record.split, record.scene.category)
        if key not in grouped:
            grouped[key] = {"scenes": 0, "instances": set()}
        grouped[key]["scenes"] = int(grouped[key]["scenes"]) + 1
        grouped[key]["instances"].add(record.scene.instance_id)  # type: ignore[union-attr]

    rows: list[dict[str, str | int]] = []
    split_order = ["train_seen", "val_seen", "test_seen", "test_unseen_category"]
    category_order = SEEN_CATEGORIES + UNSEEN_CATEGORIES
    for split in split_order:
        for category in category_order:
            key = (split, category)
            if key not in grouped:
                continue
            rows.append(
                {
                    "split": split,
                    "category": category,
                    "num_scenes": int(grouped[key]["scenes"]),
                    "num_instances": len(grouped[key]["instances"]),  # type: ignore[arg-type]
                }
            )
    return rows


def build_selection_warnings(rows: list[dict[str, str | int]], config: dict) -> list[str]:
    warnings: list[str] = []
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["split"]] = totals.get(row["split"], 0) + int(row["num_scenes"])

    soft_targets = {
        "train_seen": (300, 500),
        "val_seen": (40, 80),
        "test_seen": (80, 150),
        "test_unseen_category": (150, 250),
    }
    for split, (lower, upper) in soft_targets.items():
        total = totals.get(split, 0)
        if total < lower or total > upper:
            warnings.append(
                f"{split}: selected {total} scenes, outside soft target {lower}-{upper}; "
                "instance diversity and requested per-instance caps were prioritized"
            )

    for category in UNSEEN_CATEGORIES:
        details = config["category_details"].get(category, {})
        available_instances = int(details.get("available_instances", 0))
        if available_instances < 20:
            warnings.append(
                f"{category}: only {available_instances} valid six-digit instances available for unseen testing; "
                "all available instances were used"
            )

    return warnings


def write_summary(paths: dict[str, Path], records: list[ExportRecord]) -> list[dict[str, str | int]]:
    rows = summarize(records)
    with (paths["manifests"] / "split_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "category", "num_scenes", "num_instances"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_config(paths: dict[str, Path], config: dict, warnings: list[str], validation: ValidationResult) -> None:
    payload = dict(config)
    payload["warnings"] = warnings
    payload["validation_warnings"] = validation.warnings
    (paths["manifests"] / "split_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def folder_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def write_readme(paths: dict[str, Path], rows: list[dict[str, str | int]], warnings: list[str], validation: ValidationResult) -> None:
    summary_lines = [
        f"- {row['split']} / {row['category']}: {row['num_scenes']} scenes, {row['num_instances']} instances"
        for row in rows
    ]
    warning_lines = [f"- {warning}" for warning in (warnings + validation.warnings)] or ["- None"]
    readme = f"""# HANDAL Handle Generalization Subset v1

This folder contains a curated HANDAL no-depth subset for affordance-aware Gaussian Splatting data preparation.

## Source Categories

Seen-category split categories:
- mugs
- measuring_cups
- hammers
- screwdrivers
- adjustable_wrenches

Unseen-category test categories:
- pots_pans
- power_drills
- spatulas
- whisks

## Scene And Instance Rule

HANDAL scene folders are treated as exact six-digit ids such as `001000` or `002001`.
The object instance id is the first three digits:

```text
instance_id = scene_id[:3]
```

For seen categories, instances are split across `train_seen`, `val_seen`, and `test_seen`.
No seen-category object instance should appear in more than one of those three splits.
Scenes with dynamic suffixes such as `007999_train` are not included because they are not exact six-digit scene ids.

## Export Layout

Selected scenes are copied as real files under:

```text
subset_export/
  train_seen/<category>/<scene_id>/
  val_seen/<category>/<scene_id>/
  test_seen/<category>/<scene_id>/
  test_unseen_category/<category>/<scene_id>/
```

Category-level support files such as `dataset_info.md`, `models`, and `models_parts` are copied into each exported split/category folder.

## Rebuild

From the workspace root:

```powershell
python handal_handle_generalization_v1/scripts/build_handal_handle_subset.py
```

Use `--skip-unzip` to reuse completed `raw_unzipped` folders.

## Manifests

- `manifests/manifest.csv`: one row per exported scene.
- `manifests/split_summary.csv`: scene and instance counts by split/category.
- `manifests/split_config.json`: deterministic split config, selected instances, and warnings.

## Split Summary

{chr(10).join(summary_lines)}

## Warnings

{chr(10).join(warning_lines)}
"""
    (paths["project"] / "README.md").write_text(readme, encoding="utf-8")


def print_final_summary(
    paths: dict[str, Path],
    rows: list[dict[str, str | int]],
    warnings: list[str],
    validation: ValidationResult,
) -> None:
    print("\n=== Split summary ===")
    for row in rows:
        print(
            f"{row['split']:22s} {row['category']:22s} "
            f"{row['num_scenes']:4d} scenes {row['num_instances']:4d} instances"
        )
    size = folder_size(paths["export"])
    print(f"\nsubset_export size: {format_bytes(size)}")
    all_warnings = warnings + validation.warnings
    print(f"warnings: {len(all_warnings)}")
    for warning in all_warnings[:50]:
        print(f"- {warning}")
    if len(all_warnings) > 50:
        print(f"- ... {len(all_warnings) - 50} more warnings in manifests/split_config.json")
    print(f"\nUpload this folder to SSH machine:\n{paths['project'].resolve()}")


def main() -> int:
    args = parse_args()
    paths = project_paths(args.workspace.resolve())
    ensure_project_dirs(paths)

    print(f"Workspace: {paths['workspace']}")
    print(f"Project:   {paths['project']}")

    unzip_all(paths, skip_unzip=args.skip_unzip)
    print("[discover] scanning six-digit scene folders")
    scenes_by_category, discovery_warnings = discover_scenes(paths)
    for category, scenes in scenes_by_category.items():
        instances = {scene.instance_id for scene in scenes}
        print(f"[discover] {category}: {len(scenes)} scenes, {len(instances)} instances")

    print("[select] applying deterministic split")
    selections, config = select_subset(scenes_by_category, args.seed)

    print("[export] copying selected scenes and category support files")
    remove_generated_outputs(paths, keep_existing_export=args.keep_existing_export)
    records = export_subset(paths, selections)

    print("[validate] checking exported scene integrity and split leakage")
    validation = validate_records(records)

    print("[manifest] writing CSV/JSON manifests and README")
    write_manifest(paths, records)
    rows = write_summary(paths, records)
    selection_warnings = build_selection_warnings(rows, config)
    all_warnings = discovery_warnings + selection_warnings
    write_config(paths, config, all_warnings, validation)
    write_readme(paths, rows, all_warnings, validation)

    print_final_summary(paths, rows, all_warnings, validation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
