import json
from pathlib import Path


HANDAL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
OUT_PATH = Path("/home/gvilaplana/GS-Thesis/Affordances/configs/handal_full_split.json")


def scene_items(split):
    scenes = sorted(p.name for p in (HANDAL_ROOT / split).iterdir() if p.is_dir())
    return [{"split": split, "scene": scene} for scene in scenes]


def main():
    train = scene_items("train")
    test = scene_items("test")
    data = {
        "name": "handal_full_95train_30test",
        "description": "Full available HANDAL mug split for Gaussian-level affordance MLP.",
        "train": train,
        "test": test,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(f"train={len(train)} test={len(test)} total={len(train) + len(test)}")


if __name__ == "__main__":
    main()
