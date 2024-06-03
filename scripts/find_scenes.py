import bpy
import json
import os
from pathlib import Path

directory_path = Path("data/scenes")
output_json = "scene_frames.json"

if os.path.exists(output_json):
    with open(output_json, "r") as f:
        scene_frames = json.load(f)
else:
    scene_frames = {}

    # Iterate through all .blend files in the directory
    for blend_file in directory_path.glob("*.blend"):
        bpy.ops.wm.open_mainfile(filepath=str(blend_file))
        
        for scene in bpy.data.scenes:
            scene_frames[blend_file.stem] = {
                "start_frame": scene.frame_start,
                "end_frame": scene.frame_end
            }

    # Save the data to a JSON file
    with open(output_json, "w") as f:
        json.dump(scene_frames, f, indent=4)

    print(f"Scene frames saved to {output_json}")


final_output_json = "scene_chunks.json"
chunk_size = 128

# Load the data from the JSON file
with open(output_json, "r") as f:
    scene_frames = json.load(f)

scene_chunks = {}
index = 0

# Iterate through the scenes and split them into chunks
for scene_name, frames in scene_frames.items():
    start_frame = frames["start_frame"]
    end_frame = frames["end_frame"]
    
    for chunk_start in range(start_frame, end_frame, chunk_size):
        chunk_end = chunk_start + chunk_size
        if chunk_end <= end_frame:
            scene_chunks[index] = {
                "scene": scene_name,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end - 1
            }
            index += 1

# Save the new dictionary to a JSON file
with open(final_output_json, "w") as f:
    json.dump(scene_chunks, f, indent=4)

print(f"Scene chunks saved to {final_output_json}")