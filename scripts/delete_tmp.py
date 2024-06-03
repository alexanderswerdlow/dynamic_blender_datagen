import os
from pathlib import Path
from image_utils import Im
import numpy as np
import shutil
scene_paths = []

def remove_file_or_folder(path: Path, raise_error: bool = True):
    if path.exists():
        if path.is_file():
            os.remove(path)
        else:
            shutil.rmtree(path)
    else:
        if raise_error:
            raise ValueError(f"Path {path} does not exist")
        else:
            print(f"Path {path} does not exist")

def find_folders_with_metadata(path):
    for folder in path.iterdir():
        if folder.is_dir():
            if (folder / 'track_metadata.npz').exists():
                scene_paths.append(folder)
            else:
                find_folders_with_metadata(folder)

find_folders_with_metadata(Path('active'))
find_folders_with_metadata(Path('generated'))

for scene_path in scene_paths:
    if (scene_path / 'track_metadata.npz').exists():
        print(scene_path / 'track_metadata.npz')
        remove_file_or_folder(scene_path / 'obj', raise_error=False)
        remove_file_or_folder(scene_path / 'tmp', raise_error=False)
        remove_file_or_folder(scene_path / 'exr', raise_error=False)
        remove_file_or_folder(scene_path / 'scene.blend1', raise_error=False)
        remove_file_or_folder(scene_path / 'scene.blend', raise_error=False)
        
        # remove_file_or_folder(scene_path / 'exr_img')
        # remove_file_or_folder(scene_path / 'images')
        

