#!/usr/bin/env python3
"""Select one Qwen anchor frame/box by maximum reported confidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_item(item: dict) -> float:
    for key in ("confidence", "score", "conf"):
        if key in item:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()

    summary = json.loads(args.input.read_text())
    rows = summary.get("queries", {}).get(args.query, [])
    candidates = []
    for row in rows:
        parsed = row.get("parsed", [])
        for item in parsed:
            box = item.get("bbox_2d", item.get("box", []))
            if len(box) == 4:
                candidates.append((score_item(item), row, item))
    if not candidates:
        raise RuntimeError(f"No valid Qwen boxes for query {args.query!r} in {args.input}")

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    selected = []
    used_frames = set()
    for score, row, item in candidates:
        frame_name = row.get("frame_name")
        if frame_name in used_frames:
            continue
        row = dict(row)
        item = dict(item)
        item.setdefault("confidence", score)
        row["parsed"] = [item]
        row["num_boxes"] = 1
        row["anchor_selection"] = {
            "strategy": "top_qwen_confidence_over_candidate_frames",
            "selected_confidence": float(score),
            "source_summary": str(args.input),
            "source_frame_name": frame_name,
            "rank": len(selected) + 1,
        }
        selected.append(row)
        used_frames.add(frame_name)
        if len(selected) >= args.top_k:
            break
    if not selected:
        raise RuntimeError(f"No selected Qwen boxes for query {args.query!r} in {args.input}")

    out = dict(summary)
    out["num_frames"] = len(selected)
    out["first_frame"] = selected[0].get("frame_name")
    out["last_frame"] = selected[-1].get("frame_name")
    out["anchor_selection"] = {
        "strategy": "top_qwen_confidence_over_candidate_frames",
        "top_k": int(args.top_k),
        "source_summary": str(args.input),
        "selected": [r["anchor_selection"] for r in selected],
    }
    out["queries"] = {args.query: selected}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(
        "selected " + ", ".join(f"{r.get('frame_name')} conf={r['anchor_selection']['selected_confidence']:.4f}" for r in selected) + f" for query={args.query}",
        flush=True,
    )


if __name__ == "__main__":
    main()
