from __future__ import annotations

import sys

from utils.decoupled_utils import breakpoint_on_error

sys.path.insert(0, "/home/aswerdlo/repos/point_odyssey")

import io
import os
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
from joblib import Memory
from tqdm import tqdm

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

def train(data_path, slurm_task_index, mode=None, local=False):
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
            'indoor': 0.10,
            'robot': 0.10,
            'outdoor': 0.80,
        }

        modes = list(mode_probabilities.keys())
        probabilities = list(mode_probabilities.values())
        mode = random.choices(modes, probabilities)[0]
        print(f"Choosing mode {mode} with probability {mode_probabilities[mode]}")

    samples_per_pixel = 32
    common_command = [
        "python", "export_unified.py",
        "--rendering",
        "--exr",
        "--export_obj",
        "--export_tracking",
        "--use_gpu",
        "--samples_per_pixel", f"{samples_per_pixel}",
        "--sampling_character_num", "5000",
        "--sampling_scene_points", "2000",
        "--start_frame", "1",
        "--end_frame", "11",
    ]

    if local is False:
        common_command = common_command + [
            "--use_singularity",
        ]

    if mode == 'indoor':
        command = common_command + [
            "--add_fog",
            "--randomize",
            "--add_force",
        ]
    elif mode == 'robot':
        command = common_command + [
            "--scene_dir", "./data/demo_scene/robot.blend",
        ]
    elif mode == 'outdoor':
        command = common_command + [
            "--type", "human",
            "--add_force",
            "--scene_root", "./data/blender_assets/hdri.blend" # TODO: Not sure what the difference is between hdri and hdri_plane
        ]
    elif mode == 'animal':
        command = common_command + [
            "--type", "animal",
            "--material_path", "./data/blender_assets/animal_material.blend",
        ]

    output_dir = data_path / mode / f"{slurm_task_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = command + [
        "--output_dir", output_dir,
    ]

    print(f"SLURM Running command: {command}")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {result.stderr}")
    

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
        job_name='image_folder_parallel',
        cpus_per_task=4,
        mem='8g',
        export='ALL',
        gres=['gpu:1'],
        output=f'outputs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        time=timedelta(days=3, hours=0, minutes=0, seconds=0) if 'kate' in partition else timedelta(days=0, hours=6, minutes=0, seconds=0),
        array=f"0-{num_chunks-1}%{num_workers}",
        partition=partition,
        **kwargs
    )
    job_id = slurm.sbatch(f"python slurm.py {data_path} --is_slurm_task --slurm_task_index=$SLURM_ARRAY_TASK_ID")
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
):
    if use_slurm:
        run_slurm(data_path, num_to_process, num_workers, partition)
    elif is_slurm_task:
        print(f"Running slurm task {slurm_task_index} ...")
        train(data_path, slurm_task_index)
    else:
        with breakpoint_on_error():
            train(data_path, 0, mode='outdoor', local=True)
    
if __name__ == '__main__':
    app()