import os
from pathlib import Path
import numpy as np
import shutil
from constants import validation_blender_scenes

scene_paths = []

def find_value_in_txt(file_path, key):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith(key):
                return line.split("=")[1].strip()

def find_folders_with_metadata(path):
    for folder in path.iterdir():
        if folder.is_dir() and 'generated' not in folder.stem:
            if (folder / 'track_metadata.npz').exists():
                scene_paths.append(folder)
            else:
                find_folders_with_metadata(folder)

find_folders_with_metadata(Path('active'))
find_folders_with_metadata(Path('generated/train'))

for scene_path in scene_paths:
    custom_scene = Path(find_value_in_txt(scene_path / "slurm_metadata.txt", "custom_scene"))
    if custom_scene.name in validation_blender_scenes:
        print(f"mv {scene_path.resolve()} /home/aswerdlo/repos/point_odyssey/generated/val/v0/premade/")
        


scene_paths = []

find_folders_with_metadata(Path('generated/val'))

for scene_path in scene_paths:
    custom_scene = Path(find_value_in_txt(scene_path / "slurm_metadata.txt", "custom_scene"))
    if custom_scene.name not in validation_blender_scenes:
        print(f"mv {scene_path.resolve()} /home/aswerdlo/repos/point_odyssey/generated/train/v4/premade/")
        