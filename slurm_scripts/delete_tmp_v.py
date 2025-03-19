import os
from pathlib import Path
import numpy as np
import shutil
scene_paths = []

def find_folders_with_metadata(path):
    for folder in path.iterdir():
        if folder.is_dir():
            if (folder / 'slurm_metadata.txt').exists():
                scene_paths.append(folder)
            else:
                find_folders_with_metadata(folder)

find_folders_with_metadata(Path('generated'))

for scene_path in sorted(scene_paths):
    if (scene_path / 'track_metadata.npz').exists() is False:
        print(scene_path)
        

