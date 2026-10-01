import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from prepare_handal_3dgs_scenes import prepare_scene


REALM_ROOT = Path("/home/gvilaplana/GS-Thesis/REALM-Code")
PYTHON = Path("/home/gvilaplana/miniconda3/envs/realm/bin/python")
CONFIG = Path("/home/gvilaplana/GS-Thesis/Affordances/configs/handal_3dgs_pilot.json")
SPLIT_PATH = Path("/home/gvilaplana/GS-Thesis/Affordances/configs/handal_exp40_split.json")
SCENE_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/data/handal_3dgs_scenes")
OLD_MODEL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_3dgs")
EXP40_MODEL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_3dgs_exp40")
MODEL_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_3dgs_exp40")
PLY_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_3dgs_exp40_ply")
LOG_ROOT = Path("/home/gvilaplana/GS-Thesis/Affordances/outputs/handal_3dgs_exp40_logs")


def load_split(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["train"] + data["test"]


def final_ply(scene):
    return final_ply.root / f"handal_mug_scene_{scene}_iter3000.ply"


def trained_ply(model_root, scene):
    return model_root / scene / "point_cloud" / "iteration_3000" / "point_cloud.ply"


def copy_if_available(scene):
    dst = final_ply(scene)
    if dst.exists():
        return True
    for root in (OLD_MODEL_ROOT, EXP40_MODEL_ROOT, copy_if_available.model_root):
        src = trained_ply(root, scene)
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
    return False


def command(scene, gpu):
    scene_dir = SCENE_ROOT / scene
    model_dir = command.model_root / scene
    return [
        "bash",
        "-lc",
        (
            f"CUDA_VISIBLE_DEVICES={gpu} {PYTHON} train.py "
            f"-s {scene_dir} -m {model_dir} "
            f"--config_file {CONFIG} --iterations 3000 "
            f"--test_iterations 3000 --save_iterations 3000 "
            f"--resolution 4 --quiet"
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--gpus", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--max_images", type=int, default=96)
    parser.add_argument("--model_root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--ply_root", type=Path, default=PLY_ROOT)
    parser.add_argument("--log_root", type=Path, default=LOG_ROOT)
    args = parser.parse_args()

    final_ply.root = args.ply_root
    copy_if_available.model_root = args.model_root
    command.model_root = args.model_root
    items = load_split(args.split_path)
    args.model_root.mkdir(parents=True, exist_ok=True)
    args.ply_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    print("Preparing COLMAP-style HANDAL scenes")
    for item in items:
        dst, obj_id, num_images = prepare_scene(item["split"], item["scene"], max_images=args.max_images)
        print(f"prepared {item['split']} {item['scene']}: obj_id={obj_id}, images={num_images}, dst={dst}")

    pending = []
    for item in items:
        scene = item["scene"]
        if copy_if_available(scene):
            print(f"reuse/copy ready PLY for {scene}: {final_ply(scene)}")
        else:
            pending.append(scene)

    active = []
    start = time.time()
    completed = 0
    total = len(pending)
    print(f"3DGS pending scenes: {total}")

    while pending or active:
        while pending and len(active) < len(args.gpus):
            scene = pending.pop(0)
            gpu = args.gpus[len(active) % len(args.gpus)]
            log_path = args.log_root / f"{scene}.log"
            log_f = open(log_path, "w")
            proc = subprocess.Popen(command(scene, gpu), cwd=REALM_ROOT, stdout=log_f, stderr=subprocess.STDOUT)
            active.append({"scene": scene, "gpu": gpu, "proc": proc, "log": log_f, "log_path": log_path, "start": time.time()})
            print(f"started {scene} on GPU {gpu}; log={log_path}")

        still_active = []
        for job in active:
            ret = job["proc"].poll()
            if ret is None:
                still_active.append(job)
                continue
            job["log"].close()
            scene = job["scene"]
            elapsed = (time.time() - job["start"]) / 60.0
            if ret != 0:
                raise RuntimeError(f"3DGS training failed for {scene} with code {ret}; see {job['log_path']}")
            src = trained_ply(args.model_root, scene)
            if not src.exists():
                raise FileNotFoundError(f"Expected trained PLY missing for {scene}: {src}")
            shutil.copy2(src, final_ply(scene))
            completed += 1
            done = completed
            remaining = total - done
            avg = (time.time() - start) / max(done, 1) / 60.0
            eta = remaining * avg / max(len(args.gpus), 1)
            print(
                f"finished {scene} in {elapsed:.1f} min; "
                f"new training progress {done}/{total}; rough ETA {eta:.1f} min"
            )
        active = still_active
        if pending or active:
            time.sleep(10)

    print(f"3DGS stage complete. PLY root: {args.ply_root}")


if __name__ == "__main__":
    main()
