import bpy

import sys
import argparse
import os
import json

argv = sys.argv

if "--" not in argv:
    argv = []
else:
    argv = argv[argv.index("--") + 1:]

print("argsv:{0}".format(argv))
parser = argparse.ArgumentParser(description='Export obj data')

parser.add_argument('--scene_root', type=str, default='')
parser.add_argument('--output_dir', type=str, metavar='PATH', default='',
                    help='obj save dir')
parser.add_argument('--export_character', type=bool, default=True)
parser.add_argument('--start_frame', type=int, default=None)

args = parser.parse_args(argv)
print("args:{0}".format(args))

bpy.ops.wm.open_mainfile(filepath=args.scene_root)
assets_name = bpy.context.scene.objects.keys()
assets_name = [name for name in assets_name if bpy.data.objects[name].type == 'MESH']
from pdb import set_trace; set_trace()

scene_info = json.load(open(os.path.join(args.output_dir, 'scene_info.json'), 'r'))
scene_info['assets'] = ['background'] + assets_name
json.dump(scene_info, open(os.path.join(args.output_dir, 'scene_info.json'), 'w'), indent=4)