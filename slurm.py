from __future__ import annotations

import sys

from utils.decoupled_utils import breakpoint_on_error

sys.path.insert(0, "/home/aswerdlo/repos/point_odyssey")

import io
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import typer
from tqdm import tqdm

from constants import DATA_DIR, run_command, validation_blender_scenes
from export_unified import RenderArgs, render

node_gpus = {
    "matrix-0-16": "titanx",
    "matrix-0-18": "titanx",
    "matrix-0-24": "P40,volta",
    "matrix-0-26": "titanx",
    "matrix-0-36": "2080Ti",
    "matrix-1-1": "volta",
    "matrix-1-6": "2080Ti",
    "matrix-1-10": "2080Ti",
    "matrix-1-14": "volta",
    "matrix-1-16": "volta",
    "matrix-1-18": "titanx",
    "matrix-1-22": "2080Ti",
    "matrix-1-24": "volta",
    "matrix-2-1": "2080Ti",
    "matrix-2-25": "A100",
    "matrix-2-29": "A100",
    "matrix-3-18": "6000ADA",
    "matrix-3-22": "6000ADA",
    "matrix-0-34": "2080Ti",
    "matrix-0-22": "titanx",
    "matrix-0-28": "titanx",
    "matrix-0-38": "titanx",
    "matrix-1-4": "2080Ti",
    "matrix-1-8": "2080Ti",
    "matrix-1-12": "2080Ti",
    "matrix-1-20": "titanx",
    "matrix-2-3": "2080Ti",
    "matrix-2-5": "2080Ti",
    "matrix-2-7": "2080Ti",
    "matrix-2-9": "2080Ti",
    "matrix-2-11": "2080Ti",
    "matrix-2-13": "2080Ti",
    "matrix-2-15": "2080Ti",
    "matrix-2-17": "2080Ti",
    "matrix-2-19": "2080Ti",
    "matrix-2-21": "2080Ti",
    "matrix-2-23": "2080Ti",
    "matrix-3-13": "1080Ti",
    "matrix-2-33": "3090",
    "matrix-2-37": "3090",
    "matrix-3-26": "A5500",
    "matrix-3-28": "A5500",
}


def get_excluded_nodes(*args):
    return ",".join([x for x in node_gpus.keys() if not any(s in node_gpus[x] for s in args)])


def signal_handler(signum, frame):
    raise KeyboardInterrupt


def random_choice(objects, weights):
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    return random.choices(objects, probabilities)[0]


def train(
    data_path,
    slurm_task_index,
    mode=None,
    local=False,
    existing_output_dir: Optional[Path] = None,
    fast: bool = False,
    num_frames: Optional[int] = None,
    render_premade_scenes: bool = False,
    use_character: Optional[bool] = None,
    use_objaverse: Optional[bool] = None,
):
    assert num_frames is not None
    timestamp = time.time_ns() / 1_000_000_000
    np.random.seed(int(timestamp))
    random.seed(timestamp)
    torch.manual_seed(timestamp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        job_id = os.getenv("SLURM_JOB_ID")
        job_array_id = os.getenv("SLURM_ARRAY_JOB_ID")
        job_index = os.getenv("SLURM_ARRAY_TASK_ID")
        addr = None
        info_str = f"{os.getpid()} {socket.gethostname()} {device} {job_id} {addr}"
        print(f"Starting inference on {info_str}")
    except:
        job_id = None
        job_array_id = None
        job_index = None

    result = subprocess.check_output("nvidia-smi -L", shell=True).decode()
    print(result)

    if mode is None:
        mode_probabilities = {"generated": 1.0, "generated_deformable": 0.0, "premade": 0.0}
        modes = list(mode_probabilities.keys())
        probabilities = list(mode_probabilities.values())
        mode = random.choices(modes, probabilities)[0]
        print(f"Choosing mode {mode} with probability {mode_probabilities[mode]}")

    args = RenderArgs(
        rendering=existing_output_dir is None,
        exr=existing_output_dir is None,
        export_obj=existing_output_dir is None,
        export_tracking=True,
        use_gpu=True,
        samples_per_pixel=64,
        fps=32,
        num_frames=512,
        batch_size=32,
        background_hdr_folder=DATA_DIR / "hdri",
        slurm_task_index=slurm_task_index,
    )

    if render_premade_scenes:
        mode = "premade"

    if existing_output_dir is not None:
        output_dir = existing_output_dir
    else:
        output_dir = data_path / mode / f"{slurm_task_index}"
        if output_dir.exists():
            idx = 0
            while (output_dir := data_path / mode / f"{slurm_task_index}_{idx}").exists():
                idx += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    if local is False:
        args.use_singularity = True

    scene_dir = DATA_DIR / "scenes"

    if "val" in str(data_path):
        args.validation = True

    args.num_frames = num_frames
    args.num_assets = random.randint(3, 15)
    args.add_force = random_choice([True, False], [0.8, 0.2])
    args.fps = random.randint(4, 24)
    args.force_interval = max(args.fps * random.randint(1, 6), args.num_frames // 2)
    args.force_step = max(random.randint(1, 5), args.fps)
    args.force_scale = random.uniform(0.1, 1.0)
    args.custom_scene = DATA_DIR / "blender_assets/hdri_plane.blend"

    if mode == "generated":
        args.add_smoke = False
        args.add_fog = False
        args.use_character = random_choice([True, False], [0.1, 0.9])
        args.use_partnet = False
        args.object_ratio_weights = random_choice([(0.4, 0.0, 0.6), (0.9, 0.1, 0.0)], [0.9, 0.1])
        if args.object_ratio_weights[-1] > 0:
            args.use_objaverse = True

    elif mode == "generated_deformable":
        args.use_animal = True
        args.custom_scene = DATA_DIR / "blender_assets" / "hdri_plane.blend"
        args.material_path = DATA_DIR / "blender_assets" / "animal_material.blend"
        args.add_smoke = random_choice([True, False], [0.5, 0.5])

    if mode == "premade":
        print("Setting premade_scene")
        blend_files = list(scene_dir.glob("*.blend"))
        args.custom_scene = random.choice(blend_files)
        args.premade_scene = True
        args.randomize = True
        args.add_objects = False
        args.add_force = False
        args.fps = 24
        args.add_fog = random_choice([True, False], [0.25, 0.75])

        if render_premade_scenes:
            args.add_fog = False
            chunks = json.load(open(Path(f"data/tmp/scene_chunks{'_validation' if args.validation else ''}.json"), "r")) 
            chunk = chunks[str(slurm_task_index)]
            args.start_frame = chunk["chunk_start"]
            args.end_frame = chunk["chunk_end"]
            args.num_frames = (chunk["chunk_end"] - chunk["chunk_start"]) + 1
            assert args.num_frames == num_frames
            scene = chunk["scene"]
            args.custom_scene = DATA_DIR / "scenes" / f"{scene}.blend"
            if args.validation:
                assert any(scene in s for s in validation_blender_scenes)
            else:
                assert not any(scene in s for s in validation_blender_scenes)

            if "human_in_scene1" in args.custom_scene.name or "dancing.blend" in args.custom_scene.name or "robot.blend" in args.custom_scene.name:
                print(f"Skipping scene: {args.custom_scene}") 
                exit()

    if fast:
        args.samples_per_pixel = 8
        args.num_frames = 4
        args.add_force = True
        args.scene_scale = 1
        args.num_assets = 2
        args.add_objects = True
        args.add_force = False
        args.add_fog = False

    if num_frames is not None:
        args.num_frames = num_frames

    with open(output_dir / "slurm_metadata.txt", "w") as f:
        f.write(f"{os.getpid()} {socket.gethostname()} {device} {job_id} {addr}\n")
        slurm_env_vars = ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_SUBMIT_DIR", "SLURM_CLUSTER_NAME"]
        for var in slurm_env_vars:
            f.write(f"{var} = {os.getenv(var, 'None')}\n")

        f.write("\n")
        for field in args.__dataclass_fields__:
            f.write(f"{field} = {getattr(args, field)}\n")

        f.write("\n")
        for key, value in os.environ.items():
            f.write(f"{key} = {value}\n")

    if job_id is not None:
        initial_log_file = Path("outputs") / f"{job_array_id}_{job_index}_{job_id}.out"

        try:
            os.symlink(initial_log_file.resolve(), output_dir / "log.out")
        except:
            print(f"Failed to symlink {initial_log_file} to log.out")

    render(args, use_tmpfs=True)
    print(f"Finished rendering {output_dir}")


def tail_log_file(log_file_path, glob_str):
    max_retries = 60
    retry_interval = 2

    print(f"Command: tail -f -n +1 {log_file_path}/{glob_str}")

    for _ in range(max_retries):
        try:
            if len(list(log_file_path.glob(glob_str))) > 0:
                run_command(f"tail -f -n +1 {log_file_path}/{glob_str}")
        except:
            print(f"Tried to glob: {log_file_path}, {glob_str}")
        finally:
            time.sleep(retry_interval)

    print(f"File not found: {log_file_path} after {max_retries * retry_interval} seconds...")


def run_slurm(
    data_path,
    num_chunks,
    num_workers,
    partition,
    exclude: bool = False,
    num_frames: Optional[int] = None,
    exclude_large_nodes: bool = True,
    mode: Optional[str] = None,
    export_scene: Optional[Path] = None,
    render_premade_scenes: bool = False,
    refresh_mounts: bool = False,
):
    print(f"Running slurm job with {num_chunks} chunks and {num_workers} workers...")
    from simple_slurm import Slurm

    kwargs = dict()
    if partition == "all" and exclude:
        kwargs["exclude"] = get_excluded_nodes("volta", "2080Ti")

    if partition == "all" and exclude_large_nodes:
        kwargs["exclude"] = ["matrix-0-22", "matrix-1-14", "matrix-1-24", "matrix-2-25", "matrix-2-29", "matrix-3-18", "matrix-3-22", "matrix-3-26", "matrix-3-28"]

    if export_scene is not None:
        num_chunks = 1
        num_workers = 1
        num_frames = 64

    is_val = 'val' in str(data_path)
    if render_premade_scenes:
        with open(Path(f"data/tmp/scene_chunks{'_validation' if is_val else ''}.json"), "r") as f:
            scene_chunks = json.load(f)
        num_chunks = len(scene_chunks)
        print(f"Running {num_chunks} chunks... instead of {len(scene_chunks)}")

        premade_path = data_path / "premade"
        indices_to_remove = set(folder.name for folder in premade_path.iterdir() if folder.is_dir())
        range_to_render = [str(x) for x in list(range(0, min(num_chunks, 1000))) if str(x) not in indices_to_remove]
        range_to_render = range_to_render[:1000]
        range_to_render = ",".join(range_to_render)

    print(kwargs)
    assert num_frames is not None
    mem_dict = {32: "16g", 64: "24g", 128: "50g", 256: "30g"}

    if refresh_mounts:
        run_command(f"{(Path.home() /'bin' / 'cluster-scripts' / 'onallnodes').resolve()} scripts/refresh_mounts.sh", raise_error=False)
        time.sleep(120)

    slurm = Slurm(
        job_name=f"blender_{data_path.name}_{random.randint(0, 255):02x}",
        cpus_per_task=4,
        mem=mem_dict[max(num_frames, 32)],
        export="ALL",
        gres=["gpu:1"],
        output=f"outputs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}_{Slurm.JOB_ID}.out",
        time=timedelta(days=3, hours=0, minutes=0, seconds=0) if "kate" in partition else timedelta(days=0, hours=6, minutes=0, seconds=0),
        array=range_to_render if render_premade_scenes else range(0, min(num_chunks, 1000)),
        partition=partition,
        requeue='',
        **kwargs,
    )
    run_str = f"python slurm.py --data_path={data_path} --is_slurm_task --slurm_task_index=$SLURM_ARRAY_TASK_ID"
    if num_frames is not None:
        run_str += f" --num_frames={num_frames}"
    if mode is not None:
        run_str += f" --mode={mode}"
    if export_scene is not None:
        run_str += f" --export_scene={export_scene}"
    if render_premade_scenes:
        run_str += " --render_premade_scenes"
    
    print(run_str)
    print(slurm)
    job_id = slurm.sbatch(run_str)
    print(f"Submitted job {job_id} with {num_chunks} tasks and {num_workers} workers...")
    tail_log_file(Path(f"outputs"), f"{job_id}*")


def export_scene_func(scene_path: Path):
    args = RenderArgs(use_singularity=True, rendering=False, export_obj=False, export_tracking=True, exr=False, remove_temporary_files=True, output_dir=scene_path)
    render(args, use_tmpfs=True)


typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    data_path: Path = "results",
    num_workers: Optional[int] = None,
    num_to_process: Optional[int] = None,
    is_slurm_task: bool = False,
    slurm_task_index: int = None,
    partition: str = "all",
    existing_output_dir: Optional[Path] = None,
    local: bool = False,
    fast: bool = False,
    num_frames: Optional[int] = None,
    mode: Optional[str] = None,
    export_scene: Optional[Path] = None,
    render_premade_scenes: bool = False,
):
    if num_to_process is not None or num_workers is not None:
        if num_to_process is None:
            num_to_process = num_workers * 2
        run_slurm(data_path, num_to_process, num_workers, partition, num_frames=num_frames, mode=mode, export_scene=export_scene, render_premade_scenes=render_premade_scenes)
    elif export_scene is not None:
        export_scene_func(export_scene)
    elif is_slurm_task:
        print(f"Running slurm task {slurm_task_index} ...")
        train(data_path, slurm_task_index, num_frames=num_frames, mode=mode, render_premade_scenes=render_premade_scenes)
    else:
        with breakpoint_on_error():
            train(
                data_path=data_path,
                slurm_task_index=0,
                local=local,
                existing_output_dir=existing_output_dir,
                fast=fast,
                num_frames=num_frames,
                mode=mode,
                use_character=False,
                use_objaverse=True
            )


if __name__ == "__main__":
    app()

# python slurm.py --data_path='active/train_premade' --num_frames=128 --num_workers=128 --render_premade_scenes
# python slurm.py --data_path='generated/val/val_premade' --num_frames=128 --num_workers=128 --render_premade_scenes
# python slurm.py --data_path='active/train_v6' --num_frames=128 --num_to_process=968
# python slurm.py --data_path='generated/train/v11' --num_frames=128 --num_to_process=968 --mode=generated
# python slurm.py --data_path='generated/val/v2' --num_frames=128 --num_to_process=32 --mode=generated
# python slurm.py --data_path='debug/v0' --num_frames=8 --mode=generated
# python slurm.py --data_path='debug/v2' --num_frames=6 --mode=generated --num_to_process=4
# sb python scripts/check_tmp.py --gpu_count=0 --cpu_count=1 --mem=1 --partition='all' --quick