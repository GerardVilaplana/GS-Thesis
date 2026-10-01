#!/usr/bin/env python3
"""Export fallback PLYs by forcing the top visible REALM query candidate ID."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/03_handle_generalization/24_handal_full_realm_query_pruning_5scenes_v1")
AFF = Path("/home/gvilaplana/GS-Thesis/Affordances")
EXPORT_SCRIPT = AFF / "scripts/export_realm_selected_ids_variants.py"


def main() -> None:
    summary = {}
    with (ROOT / "run_manifest.csv").open() as f:
        for row in csv.DictReader(f):
            scene_key = row["scene_key"]
            query = row["query"]
            report_path = Path(row["query_report"])
            if not report_path.exists():
                summary[scene_key] = {"query": query, "error": "missing query report"}
                continue
            report = json.loads(report_path.read_text())
            query_info = report["queries"][query]
            candidates = query_info.get("candidates", [])
            if not candidates:
                summary[scene_key] = {"query": query, "error": "no candidates"}
                continue

            top_id = int(candidates[0]["class_id"])
            query_info["selected_class_ids"] = [top_id]
            query_info["fallback_reason"] = (
                "Strict SAM/REALM matching selected no IDs; forced top visible candidate "
                "for visual inspection only."
            )

            forced_report = ROOT / scene_key / "query_ply" / "fallback_top_candidate_query_report.json"
            forced_report.write_text(json.dumps(report, indent=2))
            out_dir = ROOT / scene_key / "fallback_top_candidate_ply"

            cmd = [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--point_cloud",
                row["model_root"] + "/point_cloud/iteration_3000/point_cloud.ply",
                "--classifier",
                row["model_root"] + "/point_cloud/iteration_3000/classifier.pth",
                "--query_report",
                str(forced_report),
                "--output_dir",
                str(out_dir),
                "--queries",
                query,
                "--num_classes",
                "2",
            ]
            subprocess.run(cmd, check=True)
            selected_summary = json.loads((out_dir / "selected_id_ply_summary.json").read_text())
            summary[scene_key] = {
                "query": query,
                "forced_id": top_id,
                "candidate": candidates[0],
                "outputs": selected_summary.get(query),
            }

    (ROOT / "fallback_top_candidate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
