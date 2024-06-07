import os
from pathlib import Path
import numpy as np
import shutil
scene_paths = []

def find_folders_with_metadata(path):
    for folder in path.iterdir():
        if folder.is_dir():
            if (folder / 'track_metadata.npz').exists():
                scene_paths.append(folder)
            else:
                find_folders_with_metadata(folder)

find_folders_with_metadata(Path('active/train_premade'))

most_recent_file = max(
    (scene_path / 'slurm_metadata.txt' for scene_path in scene_paths if (scene_path / 'track_metadata.npz').exists()),
    key=lambda p: p.stat().st_ctime,
    default=None
)
print(most_recent_file)
        

