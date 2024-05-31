import os
from pathlib import Path
from image_utils import Im
import numpy as np

def find_image_paths(results_path):
    all_image_lists = []
    for root, dirs, files in os.walk(results_path):
        if os.path.basename(root) == "images":
            png_files = [Path(root) / file for file in files if file.endswith(".png")]
            if png_files:
                all_image_lists.append(png_files)
    return all_image_lists

def find_value_in_txt(file_path, key):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith(key):
                return int(line.split("=")[1].strip())

results_path = 'generated/v6'
image_paths = sorted(find_image_paths(results_path), key=len, reverse=True)

for path_list in image_paths:
    fps = find_value_in_txt(path_list[0].parent.parent / "slurm_metadata.txt", "fps")
    Im(np.stack([Im(img_path).np for img_path in sorted(path_list)])).save_video(f'video_{path_list[0].parent.parent.name}.mp4', fps=fps)