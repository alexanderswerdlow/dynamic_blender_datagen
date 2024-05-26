from __future__ import annotations

import sys

from utils.decoupled_utils import breakpoint_on_error

sys.path.insert(0, "/home/aswerdlo/repos/point_odyssey")

import io
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
from tqdm import tqdm

from constants import DATA_DIR, run_command
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
    "matrix-3-28": "A5500"
}

def get_excluded_nodes(*args):
    return ",".join([x for x in node_gpus.keys() if not any(s in node_gpus[x] for s in args)])

def signal_handler(signum, frame):
    raise KeyboardInterrupt

def random_choice(objects, weights):
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    return random.choices(objects, probabilities)[0]

def train(data_path, slurm_task_index, mode=None, local=False, existing_output_dir: Optional[Path] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        job_id = os.getenv('SLURM_JOB_ID')
        addr = None
        info_str = f"{os.getpid()} {socket.gethostname()} {device} {job_id} {addr}"
        print(f"Starting inference on {info_str}")
    except:
        pass

    result = subprocess.check_output("nvidia-smi -L", shell=True).decode()
    print(result)

    if mode is None:
        import random
        mode_probabilities = {
            'indoor': 0.0,
            'robot': 0.0,
            'outdoor': 1.00,
        }

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
        samples_per_pixel=4,
        fps=32,
        end_frame=512,
    )

    if local is False:
        args.use_singularity = True

    if mode == 'indoor':
        args.add_fog = True
        args.randomize = True
        args.add_force = True
    elif mode == 'robot':
        args.scene_dir = DATA_DIR / "demo_scene" / "robot.blend"
    elif mode == 'outdoor':
        args.type = "human"
        args.add_smoke = False
        args.add_fog = False
        args.num_assets = random_choice([2, 5, 8, 10, 15], [0.2, 0.2, 0.5, 0.5, 0.05])
        args.add_force = random_choice([True, False], [0.5, 0.5])
        args.force_interval = random_choice(
            [args.end_frame // 1, args.end_frame // 2, args.end_frame // 4, args.end_frame // 8, args.end_frame // 16],
            [0.3, 0.5, 0.2, 0.05, 0.05]
        )
        args.force_step = random_choice([1, 2, 3, 4, 5], [0.6, 0.3, 0.2, 0.05, 0.05])
        args.force_scale = random_choice([0.05, 0.1, 0.25, 0.4, 0.6, 1.0], [0.6, 0.4, 0.2, 0.05, 0.05, 0.05])
        args.randomize = True
        args.scene_root = DATA_DIR / "blender_assets" / random_choice(["hdri.blend", "hdri_plane.blend"], [0.5, 0.5])
    elif mode == 'animal':
        args.type = "animal"
        args.material_path = DATA_DIR / "blender_assets" / "animal_material.blend"

    if existing_output_dir is not None:
        output_dir = existing_output_dir
    else:
        while (output_dir := data_path / mode / f"{slurm_task_index}").exists():
            slurm_task_index += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    
    with open(output_dir / "slurm_metadata.txt", "w") as f:
        f.write(f"{os.getpid()} {socket.gethostname()} {device} {job_id} {addr}\n")
        slurm_env_vars = [
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_NODELIST", "SLURM_SUBMIT_DIR", "SLURM_CLUSTER_NAME"
        ]
        for var in slurm_env_vars:
            f.write(f"{var}={os.getenv(var, 'Not Set')}\n")

        for field in args.__dataclass_fields__:
            f.write(f"{field} = {getattr(args, field)}\n")

    render(args)
    print(f"Finished rendering {output_dir}")
    

def tail_log_file(log_file_path, glob_str):
    max_retries = 60
    retry_interval = 2

    for _ in range(max_retries):
        try:
            if len(list(log_file_path.glob(glob_str))) > 0:
                try:
                    proc = subprocess.Popen(['tail', '-f', "-n", "+1", f"{log_file_path}/{glob_str}"], stdout=subprocess.PIPE)
                    print(['tail', '-f', "-n", "+1", f"{log_file_path}/{glob_str}"])
                    for line in iter(proc.stdout.readline, b''):
                        print(line.decode('utf-8'), end='')
                except:
                    proc.terminate()
        except:
            print(f"Tried to glob: {log_file_path}, {glob_str}")
        finally:
            time.sleep(retry_interval)

    print(f"File not found: {log_file_path} after {max_retries * retry_interval} seconds...")


def run_slurm(data_path, num_chunks, num_workers, partition, exclude: bool = False):
    print(f"Running slurm job with {num_chunks} chunks and {num_workers} workers...")
    from simple_slurm import Slurm

    kwargs = dict()
    if partition == 'all' and exclude:
        kwargs['exclude'] = get_excluded_nodes("volta", "2080Ti")

    print(kwargs)
    slurm = Slurm(
        "--requeue=10",
        job_name='blender',
        cpus_per_task=3,
        mem='16g',
        export='ALL',
        gres=['gpu:1'],
        output=f'outputs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        time=timedelta(days=3, hours=0, minutes=0, seconds=0) if 'kate' in partition else timedelta(days=0, hours=6, minutes=0, seconds=0),
        array=f"0-{num_chunks-1}%{num_workers}",
        partition=partition,
        **kwargs
    )
    job_id = slurm.sbatch(f"python slurm.py --data_path={data_path} --is_slurm_task --slurm_task_index=$SLURM_ARRAY_TASK_ID")
    print(f"Submitted job {job_id} with {num_chunks} tasks and {num_workers} workers...")
    tail_log_file(Path(f"outputs"), f"{job_id}*")

import typer

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    data_path: Path = "results",
    num_workers: int = 10,
    num_to_process: int = 1000,
    use_slurm: bool = False,
    is_slurm_task: bool = False,
    slurm_task_index: int = None,
    partition: str = 'all',
    existing_output_dir: Optional[Path] = None,
    local: bool = False,
):
    if use_slurm:
        run_slurm(data_path, num_to_process, num_workers, partition)
    elif is_slurm_task:
        print(f"Running slurm task {slurm_task_index} ...")
        train(data_path, slurm_task_index)
    else:
        with breakpoint_on_error():
            train(data_path=data_path, slurm_task_index=0, mode='outdoor', local=local, existing_output_dir=existing_output_dir)
    
if __name__ == '__main__':
    app()