from pathlib import Path
import time
import shutil


data_dir = Path("/dev/shm/point_odyssey")

if data_dir.exists() is False:
    print("/dev/shm/point_odyssey does not exist")
    exit()

import subprocess

def get_slurm_job_ids(username):
    result = subprocess.run(['squeue', '-u', username, '-h', '-o', '%A'], capture_output=True, text=True)
    job_ids = result.stdout.split()
    return job_ids

job_ids = set(get_slurm_job_ids('aswerdlo'))

def find_value_in_txt(file_path, key):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith(key):
                return line.split("=")[1].strip()

scene_paths = []

def find_folders_with_metadata(path):
    for folder in path.iterdir():
        if folder.is_dir():
            if (folder / 'slurm_metadata.txt').exists():
                scene_paths.append(folder)
            else:
                find_folders_with_metadata(folder)

find_folders_with_metadata(data_dir)

print(f"Found {len(scene_paths)} scenes")
for scene_path in scene_paths:
    if find_value_in_txt(scene_path / 'slurm_metadata.txt', 'SLURM_JOB_ID') in job_ids:
        print(f"Job is running, skipping: {scene_path}")
        continue
    # elif (scene_path / 'slurm_metadata.txt').stat().st_mtime < (time.time() - 8 * 3600):
    if (scene_path / 'track_metadata.npz').exists() is False:
        print(f"Deleting {scene_path}")
        shutil.rmtree(scene_path)

# SLURM_JOB_ID
# sb python scripts/check_tmp.py --gpu_count=0 --cpu_count=1 --mem=1 --partition='all' --quick
# sb python scripts/check_tmp.py --gpu_count=0 --cpu_count=1 --mem=1 --partition='all' --node_name='matrix-0-34'