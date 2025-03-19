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

find_folders_with_metadata(Path('active'))
find_folders_with_metadata(Path('generated'))

for scene_path in scene_paths:
    if "generated_deformable" in scene_path.parts:
        import random

        rgbs_path = scene_path / 'rgbs'
        jpg_files = list(rgbs_path.glob('*.jpg'))
        
        if jpg_files:
            random_jpg = random.choice(jpg_files)
            # output_filename = str(scene_path).replace(os.sep, '_') + '.jpg'
            # output_path = Path('outputs_viz') / output_filename
            # output_path.parent.mkdir(parents=True, exist_ok=True)
            # shutil.copy(random_jpg, output_path)
            print(f"{random_jpg}")
            print(f"{scene_path}")
            print("")

