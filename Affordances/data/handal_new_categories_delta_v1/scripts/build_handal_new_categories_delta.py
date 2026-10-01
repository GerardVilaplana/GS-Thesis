#!/usr/bin/env python3
"""Build a HANDAL delta subset for newly downloaded handle categories.

This creates a separate uploadable subset that does not duplicate the original
9-category handal_handle_generalization_v1 export.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import build_handal_handle_subset as handal


DELTA_PROJECT_NAME = "handal_new_categories_delta_v1"
NEW_CATEGORIES = [
    "combinational_wrenches",
    "fixed_joint_pliers",
    "ladles",
    "locking_pliers",
    "ratchets",
    "slip_joint_pliers",
    "strainers",
    "utensils",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Workspace containing zips/ and handal_handle_generalization_v1/.",
    )
    parser.add_argument("--seed", type=int, default=handal.SEED)
    parser.add_argument("--allow-missing", action="store_true", help="Build a provisional subset from categories whose ZIPs are present.")
    parser.add_argument("--skip-unzip", action="store_true", help="Reuse existing raw_unzipped folders only.")
    return parser.parse_args()


def paths_for(workspace: Path) -> dict[str, Path]:
    project = workspace / DELTA_PROJECT_NAME
    return {
        "workspace": workspace,
        "zips": workspace / "zips",
        "project": project,
        "raw": workspace / "handal_handle_generalization_v1" / "raw_unzipped",
        "export": project / "subset_export",
        "manifests": project / "manifests",
        "scripts": project / "scripts",
    }


def available_categories(paths: dict[str, Path], allow_missing: bool) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for category in NEW_CATEGORIES:
        zip_path = paths["zips"] / f"handal_dataset_{category}_no_depth.zip"
        if zip_path.exists():
            present.append(category)
        else:
            missing.append(category)
    if missing and not allow_missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required new category ZIP(s): {missing_text}")
    return present, missing


def unzip_categories(paths: dict[str, Path], categories: list[str], skip_unzip: bool) -> None:
    paths["raw"].mkdir(parents=True, exist_ok=True)
    for category in categories:
        zip_path = paths["zips"] / f"handal_dataset_{category}_no_depth.zip"
        dest = paths["raw"] / category
        if handal.sentinel_matches(dest, zip_path):
            print(f"[unzip] {category}: already complete")
            continue
        if skip_unzip:
            raise FileNotFoundError(f"{dest} is not marked complete and --skip-unzip was used")
        print(f"[unzip] {category}: extracting {zip_path.name} -> {dest}")
        handal.extract_zip_stripping_root(zip_path, dest)


def copy_builder_script(paths: dict[str, Path]) -> None:
    paths["scripts"].mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    shutil.copy2(src, paths["scripts"] / src.name)


def write_delta_readme(paths: dict[str, Path], categories: list[str], missing: list[str], rows: list[dict[str, str | int]]) -> None:
    lines = [
        "# HANDAL New Categories Delta v1",
        "",
        "This folder contains only the newly added HANDAL categories, so it can be uploaded alongside the original 9-category subset without duplicating it.",
        "",
        "Scene folders are exact six-digit ids. The object instance rule is:",
        "",
        "```text",
        "instance_id = scene_id[:3]",
        "```",
        "",
        "All included categories are split as seen categories into `train_seen`, `val_seen`, and `test_seen` by instance id.",
        "",
        "## Included Categories",
        "",
    ]
    lines.extend(f"- {category}" for category in categories)
    if missing:
        lines.extend(["", "## Missing When Built", ""])
        lines.extend(f"- {category}" for category in missing)
    lines.extend(["", "## Split Summary", ""])
    lines.extend(
        f"- {row['split']} / {row['category']}: {row['num_scenes']} scenes, {row['num_instances']} instances"
        for row in rows
    )
    (paths["project"] / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = paths_for(args.workspace.resolve())
    for key in ("project", "export", "manifests", "scripts"):
        paths[key].mkdir(parents=True, exist_ok=True)

    categories, missing = available_categories(paths, args.allow_missing)
    print(f"Workspace: {paths['workspace']}")
    print(f"Delta project: {paths['project']}")
    print(f"Present new categories: {', '.join(categories) if categories else '(none)'}")
    if missing:
        print(f"Missing new categories: {', '.join(missing)}")

    # Reuse the original pipeline with a narrower category config.
    handal.SEEN_CATEGORIES = categories
    handal.UNSEEN_CATEGORIES = []
    handal.CATEGORY_ZIPS = {category: f"handal_dataset_{category}_no_depth.zip" for category in categories}

    unzip_categories(paths, categories, skip_unzip=args.skip_unzip)

    print("[discover] scanning six-digit scene folders")
    scenes_by_category, discovery_warnings = handal.discover_scenes(paths)
    for category, scenes in scenes_by_category.items():
        instances = {scene.instance_id for scene in scenes}
        print(f"[discover] {category}: {len(scenes)} scenes, {len(instances)} instances")

    print("[select] applying deterministic instance split")
    selections, config = handal.select_subset(scenes_by_category, args.seed)
    config["delta_project_name"] = DELTA_PROJECT_NAME
    config["missing_categories_when_built"] = missing

    print("[export] rebuilding delta subset_export")
    handal.remove_generated_outputs(paths, keep_existing_export=False)
    records = handal.export_subset(paths, selections)

    print("[validate] checking exported scenes")
    validation = handal.validate_records(records)

    print("[manifest] writing outputs")
    handal.write_manifest(paths, records)
    rows = handal.write_summary(paths, records)
    selection_warnings = [
        warning
        for warning in handal.build_selection_warnings(rows, config)
        if not warning.startswith("test_unseen_category:")
    ]
    all_warnings = discovery_warnings + selection_warnings
    handal.write_config(paths, config, all_warnings, validation)
    write_delta_readme(paths, categories, missing, rows)
    copy_builder_script(paths)

    handal.print_final_summary(paths, rows, all_warnings, validation)
    if missing:
        print("\nAfter the last ZIP is downloaded, rerun without --allow-missing to build the final 8-category delta.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
