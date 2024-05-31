import bpy

import sys
import argparse
import os
import json

argv = sys.argv

if "--" not in argv:
    argv = []
else:
    argv = argv[argv.index("--") + 1 :]

print("argsv:{0}".format(argv))
parser = argparse.ArgumentParser(description="Export obj data")

parser.add_argument("--scene_root", type=str, default="")
parser.add_argument("--output_dir", type=str, metavar="PATH", default="./", help="img save dir")
parser.add_argument("--indoor", type=bool, default=False)

args = parser.parse_args(argv)
print("args:{0}".format(args))

bpy.ops.wm.open_mainfile(filepath=args.scene_root)
frames = range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)
print(f"Loaded frame range: {frames}")
print(f"Loaded frame FPS: {bpy.context.scene.render.fps}")

if args.indoor:
    collection_set = ['Furniture', 'Wall', 'Floor', 'Ceiling']
    scene_assets_keys = []
    for collection_name in collection_set:
        if not collection_name in bpy.data.collections:
            continue
        collection = bpy.data.collections[collection_name]
        scene_assets_keys += [obj.name for obj in collection.objects if obj.type == 'MESH' and not obj.hide_render and not 'Fire' in obj.name and not 'Smoke' in obj.name]

assets_keys = bpy.data.objects.keys()
assets_keys = [
    key
    for key in assets_keys
    if bpy.data.objects[key].type == "MESH" and key != "Plane" and "Smoke" not in key and not bpy.data.objects[key].hide_render
]

if args.indoor:
    print(f"Started with {len(assets_keys)} assets")
    assets_keys = [s for s in assets_keys if s not in scene_assets_keys]
    print(f"Removed {len(scene_assets_keys)} assets as they are in the background")

scene_info = json.load(open(os.path.join(args.output_dir, "scene_info.json"), "r"))

scene_info["assets"] = ["background"] + assets_keys
obj_save_dir = os.path.join(args.output_dir, "obj")
if not os.path.exists(obj_save_dir):
    os.makedirs(obj_save_dir)

print("assets_keys", assets_keys)
map_chars = {"/": "__", " ": "_"}

def sanitize_filename(filename: str) -> str:
    return "".join(map_chars.get(c, c) for c in filename if c.isalnum() or map_chars.get(c, c) in (" ", "_", "-", "__"))

valid_assets_keys = [sanitize_filename(key) for key in assets_keys]
print("valid_assets_keys", valid_assets_keys)

for frame_nr in frames:
    bpy.context.scene.frame_set(frame_nr)
    for valid_asset, asset in zip(valid_assets_keys, assets_keys):
        bpy.data.objects[asset].select_set(True)
        bpy.ops.export_scene.obj(
            filepath=os.path.join(obj_save_dir, f"{valid_asset}_{frame_nr:04d}.obj"),
            use_selection=True,
            use_mesh_modifiers=True,
            use_normals=False,
            use_uvs=False,
            use_triangles=False,
            keep_vertex_order=True,
            use_materials=False,
        )
        bpy.data.objects[asset].select_set(False)

scene_info["assets"] = ["background"] + assets_keys
scene_info["assets_saved"] = ["background"] + valid_assets_keys

json.dump(scene_info, open(os.path.join(args.output_dir, "scene_info.json"), "w"), indent=4)