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

results_path = 'generated/v3'
image_paths = sorted(find_image_paths(results_path), key=len, reverse=True)

map_chars = {"/": "__", " ": "_"}
def sanitize_filename(filename: str) -> str:
    return "".join(map_chars.get(c, c) for c in filename if c.isalnum() or map_chars.get(c, c) in (" ", "_", "-", "__"))

import json

for path_list in image_paths:
    output_dir = path_list[0].parent.parent
    print(output_dir)
    scene_info = json.load(open(os.path.join(output_dir, "scene_info.json"), "r"))

    assets_keys = scene_info["assets"]
    # scene_info["assets_saved"] = [sanitize_filename(key) for key in assets_keys]
    scene_info["assets_saved"] = [key.replace('.', '_') for key in assets_keys]

    json.dump(scene_info, open(os.path.join(output_dir, "scene_info.json"), "w"), indent=4)