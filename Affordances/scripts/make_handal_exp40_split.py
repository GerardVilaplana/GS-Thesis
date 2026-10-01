import json
from pathlib import Path


HANDAL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_dataset_mugs")
OUT_PATH = Path("/home/gvilaplana/GS-Thesis/Affordances/configs/handal_exp40_split.json")


def scene_names(split):
    return sorted(p.name for p in (HANDAL_ROOT / split).iterdir() if p.is_dir())


def main():
    reused = ["001001", "002001", "003001", "005001", "006001"]
    train_all = scene_names("train")
    train = reused + [name for name in train_all if name not in reused][:25]
    test = scene_names("test")[:10]
    data = {
        "name": "handal_exp40_30train_10test",
        "description": "40-scene HANDAL mug affordance MLP pilot: reuse the 5 initial train scenes, add 25 train scenes, evaluate on 10 official test scenes.",
        "train": [{"split": "train", "scene": scene} for scene in train],
        "test": [{"split": "test", "scene": scene} for scene in test],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(f"train={len(train)} test={len(test)} total={len(train) + len(test)}")
    print("train:", " ".join(train))
    print("test:", " ".join(test))


if __name__ == "__main__":
    main()
