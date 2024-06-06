import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import bpy
import mathutils
import numpy as np
from tap import to_tap_class

from constants import urban_scenes, validation_animals, sky_scenes
from export_unified import RenderTap

FOCAL_LENGTH = 30
SENSOR_WIDTH = 50
RESULOUTION_X = 960
RESULOUTION_Y = 540

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def read_obj_file(obj_file_path):
    '''
    Load .obj file, return vertices, faces.
    return: vertices: N_v X 3, faces: N_f X 3
    '''
    obj_f = open(obj_file_path, 'r')
    lines = obj_f.readlines()
    vertices = []
    faces = []
    vt = []
    vt_f = []
    for ori_line in lines:
        line = ori_line.split()
        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])  # x, y, z
        elif line[0] == 'f':  # Need to consider / case, // case, etc.
            faces.append([int(line[1].split('/')[0]),
                          int(line[2].split('/')[0]),
                          int(line[3].split('/')[0]) \
                          ])  # Notice! Need to reverse back when using the face since here it would be clock-wise!
            # Convert face order from clockwise to counter-clockwise direction.
            if len(line[1].split('/')) > 1:
                vt_f.append([int(line[1].split('/')[1]),
                           int(line[2].split('/')[1]),
                           int(line[3].split('/')[1]) \
                           ])
        elif line[0] == 'vt':
            vt.append([float(line[1]), float(line[2])])
        obj_f.close()

    return np.asarray(vertices), np.asarray(faces), np.asarray(vt), np.asarray(vt_f)

def save_obj_file(obj_file_path, vertices, faces, f_idx_offset=0, vt=None, vt_f=None):
    with open(obj_file_path, 'w') as f:
        for v in vertices:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        # adding uv coordinates
        if vt is not None:
            for v in vt:
                f.write('vt %f %f\n' % (v[0], v[1]))
        for i, face in enumerate(faces):
            if vt_f is not None and i < vt_f.shape[0]:
                f.write('f %d/%d %d/%d %d/%d\n' % (face[0] + f_idx_offset, vt_f[i][0], face[1] + f_idx_offset, vt_f[i][1], face[2] + f_idx_offset, vt_f[i][2]))
            else:
                f.write('f %d %d %d\n' % (face[0] + f_idx_offset, face[1] + f_idx_offset, face[2] + f_idx_offset))


def copy_obj(data_root, animal_name, num_seq, save_path):
    animal_sequences = [p for p in os.listdir(data_root) if animal_name in p]
    animal_sequences = np.random.choice(animal_sequences, num_seq, replace=False if num_seq < len(animal_sequences) else True)

    os.makedirs(save_path, exist_ok=True)

    idx = 0
    vt = None
    vt_f = None
    for i, animal_sequence in enumerate(animal_sequences):
        print(animal_sequence)
        obj_list = [p for p in os.listdir(os.path.join(data_root, animal_sequence, 'mesh_seq')) if '.obj' in p]
        obj_list = sorted(obj_list)
        # copy obj from the left to the right timeline
        if idx == 0:
            shutil.copy(os.path.join(data_root, animal_sequence, 'mesh_seq', obj_list[0]), os.path.join(save_path, str(idx).zfill(5) + '.obj'))
            # using blender to unwrap the first obj

            # load obj
            bpy.ops.import_scene.obj(filepath=os.path.join(save_path, str(idx).zfill(5) + '.obj'),
                                     use_groups_as_vgroups=True, split_mode='OFF')
            # select the object
            imported_object = bpy.context.selected_objects[0]
            bpy.ops.object.select_all(action='DESELECT')
            imported_object.select_set(True)
            bpy.context.view_layer.objects.active = imported_object

            # edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            # smart uv project the entire object
            bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)

            # scale the uv


            # finish the edit mode
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            # save the obj
            bpy.ops.export_scene.obj(filepath=os.path.join(save_path, str(idx).zfill(5) + '.obj'), use_selection=True,
                                     use_materials=False, use_normals=False, use_uvs=True, use_triangles=False,
                                     keep_vertex_order=True)
            # delete the object
            bpy.ops.object.delete(use_global=False)

            v, f, vt, vt_f = read_obj_file(os.path.join(save_path, str(idx).zfill(5) + '.obj'))
            # scale vt

            # vt -= 0.5
            # vt *= 5
            # vt += 0.5

        for obj in obj_list:
            v, f, _, _ = read_obj_file(os.path.join(data_root, animal_sequence, 'mesh_seq', obj))

            save_obj_file(os.path.join(save_path, str(idx).zfill(5) + '.obj'), v, f, vt=vt, vt_f=vt_f)
            idx += 1
        # copy obj from the right to the left timeline
        for obj in obj_list[::-1]:
            v, f, _, _ = read_obj_file(os.path.join(data_root, animal_sequence, 'mesh_seq', obj))

            save_obj_file(os.path.join(save_path, str(idx).zfill(5) + '.obj'), v, f, vt=vt, vt_f=vt_f)
            idx += 1

        if i < num_seq - 1:
            # interpolate obj
            obj_0 = read_obj_file(os.path.join(data_root, animal_sequence, 'mesh_seq', obj_list[0]))
            obj_list1 = [p for p in os.listdir(os.path.join(data_root, animal_sequences[i+1], 'mesh_seq')) if '.obj' in p]
            obj_list1 = sorted(obj_list1)
            obj_1 = read_obj_file(os.path.join(data_root, animal_sequences[i+1], 'mesh_seq', obj_list1[0]))

            for j in range(0, 10):
                obj_v = (obj_0[0] * (10 - j) + obj_1[0] * j) / 10
                obj_f = obj_0[1]
                save_obj_file(os.path.join(save_path, str(idx).zfill(5) + '.obj'), obj_v, obj_f, vt=vt, vt_f=vt_f)
                idx += 1

def anime2obj(anime_path, save_path):
    print(f"Converting {anime_path} to {save_path}")
    f = open(anime_path, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", anime_path)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))

    obj_v = vert_data
    v_list = []
    for i in range(nf):
        if i == 0:
            obj_v = vert_data
        else:
            obj_v = vert_data + offset_data[i - 1]

        # check if the obj is under the ground
        z_min = np.min(obj_v[:, 2])
        z_max = np.max(obj_v[:, 2])
        z_diff = z_max - z_min
        if z_min < -z_diff * 0.2:
            return
        v_list.append(obj_v.copy())

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"There are {nf} frames in {anime_path}")
    for i in range(nf):
        obj_v = v_list[i]
        obj_f = face_data
        save_obj_file(os.path.join(save_path, str(i).zfill(5) + '.obj'), obj_v, obj_f, f_idx_offset=1)

class Blender_render:
    def __init__(
        self,
        scratch_dir=None,
        partnet_path=None,
        GSO_path=None,
        character_path=None,
        motion_path=None,
        camera_path=None,
        render_engine="BLENDER_EEVEE",
        adaptive_sampling=False,
        use_denoising=True,
        samples_per_pixel=128,
        num_assets=2,
        custom_scene=None,
        use_gpu: bool = False,
        premade_scene: bool = False,
        add_force: bool = False,
        force_step: int = 3,
        force_num: int = 3,
        force_scale: float = 1.0,
        force_interval: int = 200,
        views: int = 1,
        num_frames: int = None,
        fps: Optional[int] = None,
        randomize: bool = False,
        add_fog: bool = False,
        fog_path: Optional[str] = None,
        add_smoke: bool = False,
        material_path: Optional[str] = None,
        scene_scale: Optional[float] = 1.0,
        use_animal: bool = False,
        animal_path: Optional[str] = None,
        animal_name: Optional[str] = None,
        validation: bool = False,
        args: Any = None,
    ):
        
        self.args = args
        self.validation = validation
        self.background_hdr_folder = self.args.background_hdr_folder
        self.background_hdr_path = self.args.background_hdr_path
        self.scale_factor = scene_scale
        self.premade_scene = premade_scene

        def validate_path(path):
            if not(path.suffix in [".hdr", ".exr"]):
                return False
            if self.premade_scene:
                return True
            if 'outdoor' not in str(path):
                return False
            if any(s in str(path) for s in urban_scenes):
                return False
            if not any(s in str(path) for s in sky_scenes):
                return False
            return True

        if self.background_hdr_path is None and self.background_hdr_folder is not None:
            hdr_list = [str(path) for path in Path(self.background_hdr_folder).rglob("*") if validate_path(path)]
            self.background_hdr_path = np.random.choice(hdr_list)
        
        print(f"Background: {self.background_hdr_path}")
        print(f"Premade: {self.premade_scene}")

        if self.premade_scene is False:
            if self.background_hdr_path is not None and any(s in Path(self.background_hdr_path).parent.name for s in ("indoor", "demo_scene")):
                print(f"Using {self.background_hdr_path} which is indoor, setting scale factor by 2")
                self.scale_factor *= 2
            else:
                print(f"Outdoor, setting scale factor by 10")
                self.scale_factor *= 10

        self.use_animal = use_animal
        self.animal_path = animal_path
        self.animal_name = animal_name
        self.blender_scene = bpy.context.scene
        self.render_engine = render_engine
        self.use_gpu = use_gpu
        
        self.force_step = force_step
        self.force_num = force_num
        self.force_scale = force_scale
        self.force_interval = force_interval
        self.add_force = add_force
        self.views = views
        self.num_frames = num_frames
        self.fps = fps
        self.randomize = randomize
        self.add_fog = add_fog
        self.fog_path = fog_path
        self.add_smoke = add_smoke
        self.material_path = material_path
        self.num_assets = num_assets
        self.adaptive_sampling = adaptive_sampling  # speeds up rendering
        self.use_denoising = use_denoising  # improves the output quality
        self.samples_per_pixel = samples_per_pixel

        assert self.force_interval > 0

        self.set_render_engine()

        self.scratch_dir = scratch_dir
        self.GSO_path = GSO_path
        self.partnet_path = partnet_path
        self.character_path = character_path
        self.use_character = premade_scene is False
        self.motion_path = motion_path
        self.GSO_path = GSO_path
        self.camera_path = camera_path
        self.motion_path = motion_path
        self.add_objects = self.args.add_objects

        if self.premade_scene is False and self.use_animal is False:
            self.motion_datasets = [d.name for d in Path(motion_path).iterdir() if d.is_dir()]
            self.motion_speed = {"TotalCapture": 1 / 1.5, "DanceDB": 1.0, "CMU": 1.0, "MoSh": 1.0 / 1.2, "SFU": 1.0 / 1.2}

        custom_scene = Path(custom_scene)
        assert custom_scene is not None
        print("Loading scene from '%s'" % custom_scene)

        if custom_scene.is_dir():
            blend_files = glob.glob(str(custom_scene / "**/*.blend"), recursive=True)
            assert len(blend_files) > 0, "No .blend files found in the specified directory"
            custom_scene = np.random.choice(blend_files)
            print("Randomly selected scene file: '%s'" % custom_scene)

        bpy.ops.wm.open_mainfile(filepath=str(custom_scene))

        self.obj_set = set(bpy.context.scene.objects)
        self.assets_set = []
        self.gso_force = []
        self.setup_renderer()

        if self.premade_scene is False:
            self.setup_scene()

        print(f"Loading assets...")
        self.load_assets()
        print(f"Finished loading assets...")

        self.activate_render_passes(normal=self.args.export_normals, optical_flow=self.args.export_flow, segmentation=self.args.export_segmentation, uv=self.args.export_uv)
        self.exr_output_node = self.set_up_exr_output_node()

        # self.blender_scene.render.resolution_percentage = 10
        if self.background_hdr_path is not None:
            print("loading hdr from:", self.background_hdr_path)
            self.load_background_hdr(str(self.background_hdr_path))
            self.args.background_hdr_path = self.background_hdr_path
            self.args.save(self.scratch_dir / 'config.json')

        if self.randomize and os.path.exists(self.material_path):
            self.randomize_scene()
        if self.add_fog and os.path.exists(self.fog_path):
            self.load_fog()

        try:
            # Ensure you have the correct context
            blender_scene = bpy.context.scene
            print(blender_scene.frame_start)
        except Exception as e:
            print(f"An error occurred: {e}")

        if self.use_animal:
            bpy.ops.outliner.orphans_purge()
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)
            bpy.ops.file.pack_all() # pack external data

        # save blend file
        os.makedirs(scratch_dir, exist_ok=True)
        absolute_path = os.path.abspath(scratch_dir)

        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, "scene.blend"))

    def set_render_engine(self):
        bpy.context.scene.render.engine = self.render_engine
        print("Using render engine: {}".format(self.render_engine))
        if self.use_gpu:
            print("----------------------------------------------")
            print("setting up gpu ......")

            bpy.context.scene.cycles.device = "GPU"
            for scene in bpy.data.scenes:
                print(scene.name)
                scene.cycles.device = "GPU"

            # if cuda arch use cuda, else use metal
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                d.use = True
                print("Device '{}' type {} : {}".format(d.name, d.type, d.use))
            print("setting up gpu done")
            print("----------------------------------------------")

    def load_fog(self):
        print("Loading fog")
        # append the fod file
        bpy.ops.wm.append(directory=os.path.join(self.fog_path, "Collection"), filename="fog")

        # addjust the fog
        fog_material = bpy.data.materials["fog"]
        # randomize the colorRamp
        fog_material.node_tree.nodes["ColorRamp"].color_ramp.elements[0].position = np.random.uniform(0.45, 0.55)
        fog_material.node_tree.nodes["ColorRamp"].color_ramp.elements[1].position = np.random.uniform(0.6, 1.0)

        # randomize the noise texture
        fog_material.node_tree.nodes["Noise Texture"].inputs[3].default_value = np.random.uniform(500, 4000)
        fog_material.node_tree.nodes["Noise Texture"].inputs[4].default_value = np.random.uniform(0.25, 1.0)

        # add keyframes of the noise texture
        mapping = fog_material.node_tree.nodes["Mapping"]
        for i in range(0, bpy.context.scene.frame_end // 200):
            bpy.context.scene.frame_set(i * 200)
            mapping.inputs[1].default_value[0] = np.random.uniform(-3, 3)
            mapping.inputs[1].default_value[1] = np.random.uniform(-3, 3)
            mapping.inputs[1].default_value[2] = np.random.uniform(-3, 3)
            mapping.inputs[2].default_value[0] = np.random.uniform(-np.pi, np.pi)
            mapping.inputs[2].default_value[1] = np.random.uniform(-np.pi, np.pi)
            mapping.inputs[2].default_value[2] = np.random.uniform(-np.pi, np.pi)

            # add keyframes of the mapping
            mapping.inputs[1].keyframe_insert(data_path="default_value", frame=i * 200)
            mapping.inputs[2].keyframe_insert(data_path="default_value", frame=i * 200)

        print("Loading fog done")

    def setup_renderer(self):
        # setup scene
        bpy.context.scene.render.resolution_x = RESULOUTION_X
        bpy.context.scene.render.resolution_y = RESULOUTION_Y
        bpy.context.scene.render.resolution_percentage = 100
        # setup render sampling
        bpy.context.scene.cycles.samples = self.samples_per_pixel
        # setup framerate
        if self.premade_scene is False:
            bpy.context.scene.render.fps = self.fps

    def setup_scene(self):
        bpy.ops.object.camera_add()
        self.camera = bpy.data.objects["Camera"]

        # adjust gravity
        bpy.context.scene.gravity *= self.scale_factor

        # setup camera
        self.cam_loc = (
            mathutils.Vector(
                (
                    np.random.uniform(-3, -3.5) * np.random.choice((-1, 1)),
                    np.random.uniform(-3, -3.5) * np.random.choice((-1, 1)),
                    np.random.uniform(1, 2.5),
                )
            )
            * self.scale_factor
        )

        self.cam_lookat = mathutils.Vector((0, 0, 0.5)) * self.scale_factor
        self.set_cam(self.cam_loc, self.cam_lookat)
        self.camera.data.lens = FOCAL_LENGTH
        self.camera.data.clip_end = 10000
        self.camera.data.sensor_width = SENSOR_WIDTH

        # scale boundingbox object
        print(f"Cube in scene: {'Cube' in bpy.data.objects.keys()}")
        if  "Cube" in bpy.data.objects.keys():
            bpy.data.objects['Cube'].location *= self.scale_factor
            bpy.data.objects['Cube'].scale *= self.scale_factor
            # apply scale
            bpy.context.view_layer.objects.active = bpy.data.objects["Cube"]
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            print(f"Scaling Cube in scene by {self.scale_factor}")

        # setup area light
        bpy.ops.object.light_add(
            type="AREA",
            align="WORLD",
            location=mathutils.Vector((np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(4, 5))) * self.scale_factor,
        )
        self.light = bpy.data.objects["Area"]
        self.light.data.energy = 1000 * self.scale_factor

        # add camera to scene
        bpy.context.scene.camera = self.camera

        # disable gravity
        # bpy.context.scene.gravity = (0, 0, 0)

    def set_cam(self, cam_loc, point):
        self.camera.location = self.cam_loc
        direction = point - cam_loc
        rot_quat = direction.to_track_quat("-Z", "Y")
        self.camera.rotation_euler = rot_quat.to_euler()

    def randomize_scene(self):
        """
        Randomize the scene: textures of floors, walls, ceilings, and strength of light
        """
        print("Randomizing scene ...")
        # randomize light strength
        for light in bpy.data.lights:
            light.energy *= np.random.uniform(0.7, 1.3)

        # append materials
        # Right now this only contains a cube which we don't want
        bpy.ops.wm.append(directory=os.path.join(self.material_path, "Object"), filename="Material")

        # randomize floor material
        if "Floor" in bpy.data.collections:
            floor_collection = bpy.data.collections["Floor"]
            floor_materials = [m for m in bpy.data.materials if "floor" in m.name or "Floor" in m.name]
            if len(floor_materials) == 0:
                print("No floor material found")
            else:
                for obj in floor_collection.objects:
                    if len(obj.data.materials) == 0:
                        # create a new material
                        obj.data.materials.append(np.random.choice(floor_materials))
                    else:
                        obj.data.materials[0] = np.random.choice(floor_materials)

        # randomize wall material
        if "Wall" in bpy.data.collections:
            wall_collection = bpy.data.collections["Wall"]
            wall_materials = [m for m in bpy.data.materials if "wall" in m.name or "Wall" in m.name]

            if len(wall_materials) == 0:
                print("No wall material found")
            else:
                # randomize each 2 walls with the same material
                for i in range(0, len(wall_collection.objects), 2):
                    wall_material = np.random.choice(wall_materials)
                    for j in range(2):
                        if i + j < len(wall_collection.objects):
                            wall_collection.objects[i + j].data.materials.append(wall_material)
                            wall_collection.objects[i + j].data.materials[0] = wall_material

        # randomize ceiling material
        if "Ceiling" in bpy.data.collections:
            ceiling_collection = bpy.data.collections["Ceiling"]
            ceiling_materials = [m for m in bpy.data.materials if "ceiling" in m.name or "Ceiling" in m.name]
            if len(ceiling_materials) == 0:
                print("No ceiling material found")
            else:
                for obj in ceiling_collection.objects:
                    obj.data.materials.append(np.random.choice(ceiling_materials))
                    obj.data.materials[0] = np.random.choice(ceiling_materials)

        print("Scene randomized")

    def activate_render_passes(self, normal: bool = True, optical_flow: bool = True, segmentation: bool = True, uv: bool = True, depth: bool = True):
        # We use two separate view layers
        # 1) the default view layer renders the image and uses many samples per pixel
        # 2) the aux view layer uses only 1 sample per pixel to avoid anti-aliasing

        # Starting in Blender 3.0 the depth-pass must be activated separately
        if depth:
            default_view_layer = bpy.context.scene.view_layers[0]
            default_view_layer.use_pass_z = True

        aux_view_layer = bpy.context.scene.view_layers.new("AuxOutputs")
        aux_view_layer.samples = 1  # only use 1 ray per pixel to disable anti-aliasing
        aux_view_layer.use_pass_z = False  # no need for a separate z-pass
        if hasattr(aux_view_layer, "aovs"):
            object_coords_aov = aux_view_layer.aovs.add()
        else:
            # seems that some versions of blender use this form instead
            object_coords_aov = aux_view_layer.cycles.aovs.add()

        object_coords_aov.name = "ObjectCoordinates"
        aux_view_layer.cycles.use_denoising = False

        # For optical flow, uv, and normals we use the aux view layer
        aux_view_layer.use_pass_vector = optical_flow
        aux_view_layer.use_pass_uv = uv
        aux_view_layer.use_pass_normal = normal  # surface normals
        # We use the default view layer for segmentation, so that we can get
        # anti-aliased crypto-matte
        if bpy.app.version >= (2, 93, 0):
            aux_view_layer.use_pass_cryptomatte_object = segmentation
            if segmentation:
                aux_view_layer.pass_cryptomatte_depth = 2
        else:
            aux_view_layer.cycles.use_pass_crypto_object = segmentation
            if segmentation:
                aux_view_layer.cycles.pass_crypto_depth = 2

    def load_animal(self):
        animal_list = os.listdir(self.animal_path)
        if self.animal_name is None:
            if self.validation:
                animal_list = [s for s in animal_list if any(x in s for x in validation_animals)]
            else:
                animal_list = [s for s in animal_list if not any(x in s for x in validation_animals)]

            animal = np.random.choice(animal_list)
            self.animal_name = animal.split('_')[0]
            self.args.animal_name = self.animal_name
            self.args.save(self.args.output_dir / 'config.json')

        animal_list = [c for c in animal_list if self.animal_name in c]
        animal_list = sorted(animal_list, key=lambda x: os.path.getsize(os.path.join(self.animal_path, x)))[:50] # sort animal_list by file size
        animal_list = np.random.choice(animal_list, 30, replace=False if len(animal_list) > 30 else True)

        print(f"Chose {self.animal_name}, {animal_list}")
        print(f"Saving to {os.path.join(self.scratch_dir, 'tmp')}")

        animal_obj_savedir = Path(self.scratch_dir) / 'tmp'
        animal_obj_savedir.mkdir(parents=True, exist_ok=True)

        for animal_seq in animal_list:
            anime2obj(Path(self.animal_path) / animal_seq / f"{animal_seq}.anime", animal_obj_savedir / animal_seq / 'mesh_seq')

        print(f"Saved to: {animal_obj_savedir}")
        copy_obj(animal_obj_savedir, self.animal_name, 15, Path(self.scratch_dir) / 'tmp' / 'animal_obj')

        bpy.context.scene.frame_end = len(list((Path(self.scratch_dir) / 'tmp' / 'animal_obj').iterdir()))

        # load mesh sequence
        seq_imp_settings = bpy.types.PropertyGroup.bl_rna_get_subclass_py("SequenceImportSettings")
        seq_imp_settings.fileNamePrefix = bpy.props.StringProperty(name='File Name', default='0')
        print('importing mesh sequence')
        bpy.ops.ms.import_sequence(directory=os.path.join(self.scratch_dir, 'tmp', 'animal_obj'))
        print('importing mesh sequence done!')
        self.animal = bpy.context.selected_objects[0]
        # scale and rotate the animal
        dimension = np.max(self.animal.dimensions)
        animal_scale = np.random.uniform(1, 1.4) * self.scale_factor * np.random.uniform(1.5, 2) / dimension
        self.animal.scale = (animal_scale, animal_scale, animal_scale)
        self.animal.rotation_euler = (0, 0, 0)

        # make the animal stand on the ground without penetrating
        z_min = np.min([b[2] for b in self.animal.bound_box], axis=0)
        self.animal.location = (0, 0, -z_min * animal_scale)

        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

        # add random texture to animal
        print('adding texture')
        # append materials

        # Tmp disable cube
        bpy.ops.wm.append(directory=os.path.join(self.material_path, "Object"), filename="Cube")

        try:
            furry_material = [f for f in bpy.data.materials if 'Animal' in f.name]
            furry_material = np.random.choice(furry_material)
            self.animal.data.materials.clear()
            self.animal.data.materials.append(furry_material)
        except:
            print("No furry material found")

        # add physics
        bpy.context.view_layer.objects.active = self.animal
        bpy.ops.rigidbody.object_add()
        self.animal.rigid_body.collision_shape = 'MESH'
        self.animal.rigid_body.type = 'PASSIVE'
        # enable animated
        self.animal.rigid_body.kinematic = True

        print(f"Animal loaded")

    def load_character(self):
        # load character
        character_list = os.listdir(self.character_path)
        character_list = [c for c in character_list if os.path.isdir(os.path.join(self.character_path, c))]
        character = np.random.choice(character_list)
        self.character_name = character
        character_collection_path = os.path.join(self.character_path, character, "{}.blend".format(character), "Collection")

        bpy.ops.wm.append(directory=character_collection_path, filename=character)
        character_collection = bpy.data.collections[character]
        self.character = character_collection
        self.skeleton = bpy.data.objects["skeleton_" + character]

        bone_mapping_path = os.path.join(self.character_path, character, "bone_mapping.json")
        bone_mapping = json.load(open(bone_mapping_path, "r"))
        # add visible mesh objs in character collection to assets_set
        for obj in character_collection.objects:
            if not obj.hide_render and obj.name in bpy.data.objects.keys() and bpy.data.objects[obj.name].type == "MESH":
                self.assets_set.append(obj)

        self.retarget_smplx2skeleton(bone_mapping)

        bpy.ops.object.select_all(action="DESELECT")
        # select all objects in the collection
        for obj in character_collection.objects:
            obj.select_set(True)

        bpy.ops.transform.resize(
            value=(self.scale_factor * 1.2, self.scale_factor * 1.2, self.scale_factor * 1.2),
            orient_type="GLOBAL",
            orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            orient_matrix_type="GLOBAL",
            mirror=False,
            use_proportional_edit=False,
            proportional_edit_falloff="SMOOTH",
            proportional_size=1,
            use_proportional_connected=False,
            use_proportional_projected=False,
        )

    def load_assets(self):
        if self.use_animal:
            self.load_animal()
        elif self.use_character:
            self.load_character()

        if self.add_smoke:
            # add a cube
            bpy.ops.mesh.primitive_cube_add(
                size=2,
                enter_editmode=False,
                align="WORLD",
                location=(np.random.uniform(-1.5, 1.5) * self.scale_factor, np.random.uniform(-1.5, 1.5) * self.scale_factor, 1),
                scale=(self.scale_factor / 5, self.scale_factor / 5, self.scale_factor / 5),
            )
            # change the name of the cube
            bpy.context.object.name = "Smoke_cube"
            # make it not rendered
            bpy.context.object.hide_render = True
            # add smoke
            bpy.ops.object.quick_smoke()
            # scale the smoke
            bpy.context.object.scale = (self.scale_factor / 2, self.scale_factor / 2, self.scale_factor * 3)
            # move z axis
            z_min = bpy.context.object.bound_box[0][2]
            bpy.context.object.location[2] -= -z_min * self.scale_factor / 5
            # change the resulution of the smoke
            bpy.context.object.modifiers["Fluid"].domain_settings.resolution_max = 128

            # enable the adptive domain
            bpy.context.object.modifiers["Fluid"].domain_settings.use_adaptive_domain = True
            bpy.context.object.modifiers["Fluid"].domain_settings.cache_frame_start = 1
            bpy.context.object.modifiers["Fluid"].domain_settings.cache_frame_end = bpy.context.scene.frame_end

        if self.add_objects:
            GSO_assets = sorted(os.listdir(self.GSO_path))
            validation_assets = GSO_assets[::50]
            if self.validation:
                GSO_assets = validation_assets
            else:
                GSO_assets = [asset for asset in GSO_assets if asset not in validation_assets]

            print(f"Validation: {self.validation}, GSO assets: {len(GSO_assets)}")
            GSO_assets = [os.path.join(self.GSO_path, asset) for asset in GSO_assets]
            GSO_assets = [asset for asset in GSO_assets if os.path.isdir(asset)]
            GSO_assets_path = np.random.choice(GSO_assets, size=self.num_assets // 2, replace=False)
            print(f"GSO, {self.GSO_path}, Selected: {GSO_assets_path}")

            partnet_assets = os.listdir(self.partnet_path)
            partnet_assets = [os.path.join(self.partnet_path, asset) for asset in partnet_assets]
            partnet_assets = [asset for asset in partnet_assets if os.path.isdir(asset) and len(os.listdir(os.path.join(asset, "objs"))) < 15]
            validation_partnet_assets = partnet_assets[::50]
            if self.validation:
                partnet_assets = validation_partnet_assets
            else:
                partnet_assets = [p for p in partnet_assets if p not in validation_partnet_assets]

            print(f"Validation: {self.validation}, Partnet assets: {len(partnet_assets)}")
            partnet_assets = np.random.choice(partnet_assets, size=self.num_assets - len(GSO_assets_path), replace=False)
            print(f"Partnet, {self.partnet_path}, Selected: {partnet_assets}")

            # generating location lists for assets, and remove the center area
            location_list = np.random.uniform(np.array([-2.5, -2.5, 0.8]), np.array([-1, -1, 2]), size=(self.num_assets * 50, 3)) * self.scale_factor
            location_list = location_list * np.sign(np.random.uniform(-1, 1, size=(self.num_assets * 50, 3)))
            location_list[:, 2] = np.abs(location_list[:, 2])
            location_list = self.farthest_point_sampling(location_list, self.num_assets + 1)
            for i, asset_path in enumerate(GSO_assets_path):
                bpy.ops.import_scene.obj(filepath=os.path.join(asset_path, "meshes", "model.obj"))
                imported_object = bpy.context.selected_objects[0]
                self.assets_set.append(imported_object)
                self.load_asset_texture(
                    imported_object,
                    mat_name=imported_object.data.name + "mat",
                    texture_path=os.path.join(asset_path, "materials", "textures", "texture.png"),
                )
                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
                # randomize location and translation
                imported_object.location = location_list[i]
                imported_object.rotation_euler = (np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi))

                # set scale
                dimension = np.max(imported_object.dimensions)
                scale = np.random.uniform(1, 6) * self.scale_factor
                if scale * dimension > 0.8 * self.scale_factor:  # max 0.8m
                    scale = 0.8 * self.scale_factor / dimension
                elif scale * dimension < 0.1 * self.scale_factor:
                    scale = 0.1 * self.scale_factor / dimension
                imported_object.scale = (scale, scale, scale)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

                # set obj active
                bpy.context.view_layer.objects.active = imported_object
                # add rigid body
                bpy.ops.rigidbody.object_add()
                imported_object.rigid_body.type = "ACTIVE"
                # imported_object.rigid_body.collision_shape = 'MESH'
                imported_object.rigid_body.collision_shape = "CONVEX_HULL"

                imported_object.rigid_body.mass = 0.5 * scale / self.scale_factor
                # bpy.ops.object.modifier_add(type='COLLISION')
            print("GSO assets loaded")
            print("loading partnet assets")
            print('partnet', partnet_assets)
            for j, obj_path in enumerate(partnet_assets):
                parts = sorted(os.listdir(os.path.join(obj_path, "objs")))
                part_objs = []
                for p in parts:
                    if not "obj" in p:
                        continue
                    bpy.ops.import_scene.obj(filepath=os.path.join(obj_path, "objs", p))
                    imported_object = bpy.context.selected_objects[0]
                    part_objs.append(imported_object)

                    # unwrap obj
                    bpy.ops.object.select_all(action="DESELECT")
                    imported_object.select_set(True)
                    bpy.context.view_layer.objects.active = imported_object
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="SELECT")
                    # smart uv project the entire object
                    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
                    # finish the edit mode
                    bpy.ops.mesh.select_all(action="DESELECT")
                    bpy.ops.object.mode_set(mode="OBJECT")

                    # load random texture from gso
                    gso_random_index = np.random.choice(range(len(GSO_assets)))
                    self.load_asset_texture(
                        imported_object,
                        mat_name=imported_object.data.name + "mat",
                        texture_path=os.path.join(GSO_assets[gso_random_index], "materials", "textures", "texture.png"),
                    )
                # merge parts into one obj
                bpy.ops.object.select_all(action="DESELECT")
                for part in part_objs:
                    part.select_set(True)
                bpy.ops.object.join()
                imported_object = bpy.context.selected_objects[0]
                # randomize location and translation
                imported_object.location = location_list[len(GSO_assets_path) + j]
                imported_object.rotation_euler = (np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi))
                # random scale
                dimension = np.max(imported_object.dimensions)
                scale = np.random.uniform(1, 6) * self.scale_factor
                if scale * dimension > 0.8 * self.scale_factor:
                    scale = 0.8 * self.scale_factor / dimension
                elif scale * dimension < 0.1 * self.scale_factor:
                    scale = 0.1 * self.scale_factor / dimension
                imported_object.scale = (scale, scale, scale)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

                # set obj active
                bpy.context.view_layer.objects.active = imported_object
                # add rigid body
                bpy.ops.rigidbody.object_add()
                imported_object.rigid_body.type = "ACTIVE"
                imported_object.rigid_body.collision_shape = "CONVEX_HULL"
                imported_object.rigid_body.mass = 1 * scale / self.scale_factor
                self.assets_set.append(imported_object)

        # add force
        if self.add_force:
            assert self.add_objects
            for i in range(self.force_num):
                dxyz = np.random.uniform(-4, 4, size=3) * self.scale_factor
                dxyz[2] = -abs(dxyz[2]) * 5
                bpy.ops.object.empty_add(type="PLAIN_AXES", location=dxyz)
                obj_axis = bpy.context.selected_objects[0]
                self.gso_force.append(obj_axis)
                # add force filed to axis
                bpy.ops.object.forcefield_toggle()
                bpy.context.object.field.shape = "POINT"
                bpy.context.object.field.type = "FORCE"
                # set min and max range
                bpy.context.object.field.use_min_distance = True
                bpy.context.object.field.use_max_distance = True
                bpy.context.object.field.distance_max = 1000
                bpy.context.object.field.strength = np.random.uniform(1000, 200)

        print("len of assets_set:", len(self.assets_set))
        print("len of forces:", len(self.gso_force))

    @staticmethod
    def farthest_point_sampling(p, K):
        """
        greedy farthest point sampling
        p: point cloud
        K: number of points to sample
        """

        farthest_point = np.zeros((K, 3))
        max_idx = np.random.randint(0, p.shape[0] - 1)
        farthest_point[0] = p[max_idx]
        for i in range(1, K):
            pairwise_distance = np.linalg.norm(p[:, None, :] - farthest_point[None, :i, :], axis=2)
            distance = np.min(pairwise_distance, axis=1, keepdims=True)
            max_idx = np.argmax(distance)
            farthest_point[i] = p[max_idx]
        return farthest_point

    def load_background_hdr(self, background_hdr_path):
        if self.premade_scene:
            world = bpy.context.scene.world
            for node in world.node_tree.nodes:
                world.node_tree.nodes.remove(node)

            node_background = world.node_tree.nodes.new(type='ShaderNodeBackground')
            node_env = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
            node_output = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
            node_env.image = bpy.data.images.load(background_hdr_path)
            world.node_tree.links.new(node_env.outputs["Color"], node_background.inputs["Color"])
            world.node_tree.links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
        else:

            world = bpy.context.scene.world
            node_env = world.node_tree.nodes["Environment Texture"]
            node_env.image = bpy.data.images.load(background_hdr_path)

    def load_asset_texture(self, obj, mat_name, texture_path, normal_path=None, roughness_path=None):
        print(f"Loading texture for {obj.name}, {texture_path}, {normal_path}, {roughness_path}, {mat_name}")
        mat = bpy.data.materials.new(name=mat_name)

        mat.use_nodes = True

        mat_nodes = mat.node_tree.nodes
        mat_links = mat.node_tree.links

        img_tex_node = mat_nodes.new(type="ShaderNodeTexImage")
        img_tex_node.image = bpy.data.images.load(texture_path)

        mat_links.new(img_tex_node.outputs["Color"], mat_nodes["Principled BSDF"].inputs["Base Color"])

        if normal_path:
            diffuse_tex_node = mat_nodes.new(type="ShaderNodeTexImage")
            diffuse_tex_node.image = bpy.data.images.load(normal_path)
            img_name = diffuse_tex_node.image.name
            bpy.data.images[img_name].colorspace_settings.name = "Raw"
            mat_links.new(diffuse_tex_node.outputs["Color"], mat_nodes["Principled BSDF"].inputs["Normal"])
        if roughness_path:
            roughness_tex_node = mat_nodes.new(type="ShaderNodeTexImage")
            roughness_tex_node.image = bpy.data.images.load(roughness_path)
            bpy.data.images[roughness_tex_node.image.name].colorspace_settings.name = "Raw"
            mat_links.new(roughness_tex_node.outputs["Color"], mat_nodes["Principled BSDF"].inputs["Roughness"])

        # clear all materials
        obj.data.materials.clear()

        # assign to 1st material slot
        obj.data.materials.append(mat)

    def set_up_exr_output_node(self):
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # set exr output
        default_layers = ("Image", "Depth")
        aux_layers = ("UV", "Normal", "CryptoObject00", "ObjectCoordinates")

        # clear existing nodes
        for node in tree.nodes:
            tree.nodes.remove(node)

        # the render node has outputs for all the rendered layers
        render_node = tree.nodes.new(type="CompositorNodeRLayers")
        render_node_aux = tree.nodes.new(type="CompositorNodeRLayers")
        render_node_aux.name = "Render Layers Aux"
        render_node_aux.layer = "AuxOutputs"

        # create a new FileOutput node
        out_node = tree.nodes.new(type="CompositorNodeOutputFile")
        # set the format to EXR (multilayer)
        out_node.format.file_format = "OPEN_EXR_MULTILAYER"

        out_node.file_slots.clear()
        for layer_name in default_layers:
            out_node.file_slots.new(layer_name)
            links.new(render_node.outputs.get(layer_name), out_node.inputs.get(layer_name))

        for layer_name in aux_layers:
            out_node.file_slots.new(layer_name)
            links.new(render_node_aux.outputs.get(layer_name), out_node.inputs.get(layer_name))

        # manually convert to RGBA. See:
        # https://blender.stackexchange.com/questions/175621/incorrect-vector-pass-output-no-alpha-zero-values/175646#175646
        split_rgba = tree.nodes.new(type="CompositorNodeSepRGBA")
        combine_rgba = tree.nodes.new(type="CompositorNodeCombRGBA")
        for channel in "RGBA":
            links.new(split_rgba.outputs.get(channel), combine_rgba.inputs.get(channel))
        out_node.file_slots.new("Vector")
        links.new(render_node_aux.outputs.get("Vector"), split_rgba.inputs.get("Image"))
        links.new(combine_rgba.outputs.get("Image"), out_node.inputs.get("Vector"))
        return out_node

    def set_exr_output_path(self, path_prefix: Optional[str]):
        """Set the target path prefix for EXR output.

        The final filename for a frame will be "{path_prefix}{frame_nr:04d}.exr".
        If path_prefix is None then EXR output is disabled.
        """
        if path_prefix is None:
            self.exr_output_node.mute = True
        else:
            self.exr_output_node.mute = False
            self.exr_output_node.base_path = str(path_prefix)

    def clear_scene(self):
        for k in bpy.data.objects.keys():
            bpy.data.objects[k].select_set(False)

    def retarget_smplx2skeleton(self, mapping):
        # get source motion
        for _ in range(100):
            motion_dataset = np.random.choice(self.motion_datasets)
            # find all the npz file in the folder recursively
            motion_files = glob.glob(os.path.join(self.motion_path, motion_dataset, "**/*.npz"), recursive=True)
            motion_files = [f for f in motion_files]
            print(f"Number of motion before filter: {len(motion_files)}")
            # filter out too small motion
            motion_files = [f for f in motion_files if os.path.getsize(f) > 5e6]
            if len(motion_files) > 0:
                break
        print(f"Number of motion files: {len(motion_files)} in {self.motion_path}{motion_dataset}")
        motion = np.random.choice(motion_files)
        print(f"loading motion {motion}")

        # load smplx motion using smplx addon
        bpy.ops.object.smplx_add_animation(filepath=motion)
        smplx = bpy.context.selected_objects[0]
        smplx_skeleton = smplx.parent

        # slow down the motion
        speed_scale = self.motion_speed[motion_dataset] if motion_dataset in self.motion_speed else 1.0
        bpy.context.scene.frame_end = int(bpy.context.scene.frame_end / speed_scale)
        for fcurve in smplx_skeleton.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.co[0] = keyframe.co[0] / speed_scale

        print(f"Re-targeting...")
        bpy.context.scene.rsl_retargeting_armature_source = smplx_skeleton
        bpy.context.scene.rsl_retargeting_armature_target = self.skeleton
        bpy.ops.rsl.build_bone_list()

        print("mapping skeleton")
        for bone in bpy.context.scene.rsl_retargeting_bone_list:
            if bone.bone_name_source in mapping.keys():
                if mapping[bone.bone_name_source] in self.skeleton.data.bones.keys():
                    bone.bone_name_target = mapping[bone.bone_name_source]
            else:
                bone.bone_name_target = ""

        # retarget motion
        print("retargeting")
        bpy.ops.rsl.retarget_animation()
        print("retargeting done")

        # delete smplx
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.data.objects:
            if "SMPLX" in obj.name:
                obj.select_set(True)
        smplx.select_set(True)
        bpy.ops.object.delete()

        # post processing the motion to make it stand on the ground without interpenetration
        bpy.ops.object.select_all(action="DESELECT")
        # find root bone
        root_bone = None
        for bone in self.skeleton.data.bones:
            if bone.parent is None:
                root_bone = bone
                break
        z_pre = 0
        for f in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            bpy.context.scene.frame_set(f)
            # get min z of the character
            print("resolve interpenetration: frame {}".format(f))
            min_z = 0
            for character_part in self.assets_set:
                character_part.select_set(True)
                bounding_box = [character_part.matrix_world @ mathutils.Vector(corner) for corner in character_part.bound_box]
                min_z = min(min_z, min([v[2] for v in bounding_box]))
                character_part.select_set(False)
            delta_z = z_pre
            if min_z < 0:
                delta_z = min_z
                z_pre = min_z

            bpy.ops.object.select_all(action="DESELECT")
            self.skeleton.select_set(True)
            bpy.ops.object.mode_set(mode="POSE")
            root_bone.select = True
            bpy.context.object.data.bones.active = root_bone
            bpy.context.active_pose_bone.location[1] -= delta_z
            # replace keyframe
            bpy.context.active_pose_bone.keyframe_insert(data_path="location", frame=f)
            bpy.ops.object.mode_set(mode="OBJECT")
        self.skeleton.select_set(False)

    @staticmethod
    def bake_to_keyframes(frame_start, frame_end, step):
        bake = []
        objects = []
        context = bpy.context
        scene = bpy.context.scene
        frame_orig = scene.frame_current
        frames_step = range(frame_start, frame_end + 1, step)
        frames_full = range(frame_start, frame_end + 1)

        # filter objects selection
        for obj in context.selected_objects:
            if not obj.rigid_body or obj.rigid_body.type != "ACTIVE":
                obj.select_set(False)

        objects = context.selected_objects

        if objects:
            # store transformation data
            # need to start at scene start frame so simulation is run from the beginning
            for f in frames_full:
                scene.frame_set(f)
                print("saving transform data for frame ", f)
                if f in frames_step:
                    mat = {}
                    for i, obj in enumerate(objects):
                        mat[i] = obj.matrix_world.copy()
                    bake.append(mat)

            # apply transformations as keyframes
            for i, f in enumerate(frames_step):
                scene.frame_set(f)
                for j, obj in enumerate(objects):
                    mat = bake[i][j]
                    # Convert world space transform to parent space, so parented objects don't get offset after baking.
                    if obj.parent:
                        mat = obj.matrix_parent_inverse.inverted() @ obj.parent.matrix_world.inverted() @ mat

                    obj.location = mat.to_translation()

                    rot_mode = obj.rotation_mode
                    if rot_mode == "QUATERNION":
                        q1 = obj.rotation_quaternion
                        q2 = mat.to_quaternion()
                        # make quaternion compatible with the previous one
                        if q1.dot(q2) < 0.0:
                            obj.rotation_quaternion = -q2
                        else:
                            obj.rotation_quaternion = q2
                        obj.keyframe_insert(data_path="rotation_quaternion", frame=f)
                    elif rot_mode == "AXIS_ANGLE":
                        # this is a little roundabout but there's no better way right now
                        aa = mat.to_quaternion().to_axis_angle()
                        obj.rotation_axis_angle = (aa[1], *aa[0])
                        obj.keyframe_insert(data_path="rotation_axis_angle", frame=f)
                    else:  # euler
                        # make sure euler rotation is compatible to previous frame
                        # NOTE: assume that on first frame, the starting rotation is appropriate
                        obj.rotation_euler = mat.to_euler(rot_mode, obj.rotation_euler)
                        obj.keyframe_insert(data_path="rotation_euler", frame=f)
                    # bake to keyframe
                    obj.keyframe_insert(data_path="location", frame=f)

                print("Baking frame %d" % f)

            # remove baked objects from simulation
            for obj in objects:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_remove()

            # clean up keyframes
            for obj in objects:
                action = obj.animation_data.action
                for fcu in action.fcurves:
                    keyframe_points = fcu.keyframe_points
                    i = 1
                    # remove unneeded keyframes
                    while i < len(keyframe_points) - 1:
                        val_prev = keyframe_points[i - 1].co[1]
                        val_next = keyframe_points[i + 1].co[1]
                        val = keyframe_points[i].co[1]

                        if abs(val - val_prev) + abs(val - val_next) < 0.0001:
                            keyframe_points.remove(keyframe_points[i])
                        else:
                            i += 1
                    # use linear interpolation for better visual results
                    for keyframe in keyframe_points:
                        keyframe.interpolation = "LINEAR"

    def bake_camera(self, camera_rt, frames):
        self.camera_T = -camera_rt[:, :3, :3].transpose((0, 2, 1)) @ camera_rt[:, :3, 3:]

        xy_min = np.min(self.camera_T[:, :2], axis=0)
        xy_max = np.max(self.camera_T[:, :2], axis=0)
        xy_length = np.max(np.abs(xy_max - xy_min))
        scale = 1.5
        if xy_length < 8:
            scale = 8 / xy_length
        elif xy_length > 10:
            scale = 10 / xy_length
        self.camera_T[:, :2] *= scale
        trajectory_vec = (self.camera_T[-1] - self.camera_T[0]).reshape(-1)
        cam_vec = np.array(self.cam_loc)
        cam_sign = np.sign(cam_vec * trajectory_vec)
        cam_sign *= -1
        cam_sign[2] = 1
        self.cam_sign = cam_sign.reshape(-1)
        time_ratio = self.camera_T.shape[0] / (frames[-1] - frames[0])
        initial_r = np.linalg.norm(cam_vec[:2])

        # set camera poses
        for cam_idx in range(0, self.camera_T.shape[0] - 1):
            frame_nr = frames[0] + int(cam_idx / time_ratio)
            bpy.context.scene.frame_set(frame_nr)
            bpy.context.scene.camera.keyframe_insert(data_path="location", frame=frame_nr)
            bpy.context.scene.camera.keyframe_insert(data_path="rotation_euler", frame=frame_nr)

            frame_next = frames[0] + int((cam_idx + 1) / time_ratio)
            bpy.context.scene.frame_set(frame_next)
            delta_T = self.camera_T[cam_idx + 1] - self.camera_T[cam_idx]
            delta_T = delta_T.reshape(-1)
            delta_T = delta_T * self.cam_sign
            delta_T *= self.scale_factor
            print("delta_T", delta_T)
            # delta_T = np.clip(delta_T, -0.2 * 1 / time_ratio * self.scale_factor, 0.2 * 1 / time_ratio * self.scale_factor).reshape(3)
            mean_location = np.mean(
                [obj.matrix_world.translation for obj in self.assets_set if np.max(np.abs(obj.matrix_world.translation)) < 3 * self.scale_factor],
                axis=0,
            )

            # mean_location[:2] *= 0
            self.cam_lookat = self.cam_lookat * 0.95 + mathutils.Vector(mean_location) * 0.05

            self.cam_loc = self.cam_loc + mathutils.Vector([delta_T[0], delta_T[1], delta_T[2]])
            if np.linalg.norm(np.array(self.cam_loc[:2])) < initial_r * 0.75:
                self.cam_loc[:2] = self.cam_loc[:2] / np.linalg.norm(np.array(self.cam_loc[:2])) * initial_r * 0.75
            cam_height_max = 3 * self.scale_factor
            if self.cam_loc[2] > cam_height_max:
                self.cam_loc[2] = cam_height_max
                self.cam_sign[2] *= -1
            if self.cam_loc[2] < 0.5 * self.scale_factor:
                self.cam_loc[2] = 0.5 * self.scale_factor
                self.cam_sign[2] *= -1
            self.set_cam(self.cam_loc, self.cam_lookat)
            print("inserting camera keyframe {}, delta_T:{}".format(frame_next, delta_T))
            # add camera lcoation and rotation keyframe
            bpy.context.scene.camera.keyframe_insert(data_path="location", frame=frame_next)
            bpy.context.scene.camera.keyframe_insert(data_path="rotation_euler", frame=frame_next)

    def render(self):
        print(
            f"Default start/end range: {range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)}, Default FPS: {bpy.context.scene.render.fps}"
        )

        scene_start, scene_end = bpy.context.scene.frame_start, bpy.context.scene.frame_end
        if self.args.start_frame is not None and self.args.end_frame is not None:
            assert self.args.start_frame + self.args.num_frames - 1 == self.args.end_frame
            start_frame = self.args.start_frame
            end_frame = self.args.end_frame
        else:
            start_frame = np.random.randint(scene_start, scene_end - self.num_frames + 2)
            end_frame = start_frame + self.num_frames - 1

        bpy.context.scene.frame_start = start_frame
        bpy.context.scene.frame_end = end_frame

        self.args.start_frame = start_frame
        self.args.end_frame = end_frame
        self.args.save(self.args.output_dir / 'config.json')

        print(
            f"New start/end range: {range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)}, New FPS: {bpy.context.scene.render.fps}",
            flush=True,
        )

        """Renders all frames (or a subset) of the animation.
        """
        print("Using scratch rendering folder: '%s'" % self.scratch_dir)

        if self.add_objects:
            # setup rigid world cache
            bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
            bpy.context.scene.rigidbody_world.point_cache.frame_end = bpy.context.scene.frame_end + 1
            bpy.context.view_layer.objects.active = self.assets_set[0]

        self.set_render_engine()
        self.clear_scene()

        absolute_path = os.path.abspath(self.scratch_dir)
        frames = range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)

        # add forces
        for frame_nr in frames:
            if self.add_force and frame_nr % self.force_interval == 1:
                # add keyframe to force strength
                bpy.context.scene.frame_set(frame_nr)
                if self.use_animal:
                    force_loc_list = np.random.uniform(np.array([-16, -16, -3]), np.array([16, 16, 0]), size=(self.num_assets * 50, 3)) * self.scale_factor
                else:
                    force_loc_list = np.random.uniform(np.array([-4, -4, -5]), np.array([4, 4, -3]), size=(self.num_assets * 50, 3)) * self.scale_factor
                force_loc_list = self.farthest_point_sampling(force_loc_list, self.force_num)
                print("force_loc_list", force_loc_list)
                for i in range(len(self.gso_force)):
                    force_source = self.gso_force[i]
                    # select obj
                    force_source.field.strength = np.random.uniform(500, 1000) * self.force_scale
                    force_source.field.distance_max = 1000
                    force_loc_list[i][2] *= 5
                    force_source.location = force_loc_list[i]
                    force_source.keyframe_insert(data_path="location", frame=frame_nr)
                    force_source.keyframe_insert(data_path="location", frame=frame_nr + self.force_interval - 1)
                    force_source.keyframe_insert(data_path="field.strength", frame=frame_nr)
                    force_source.keyframe_insert(data_path="field.strength", frame=frame_nr + self.force_step - 1)
                    force_source.keyframe_insert(data_path="field.distance_max", frame=frame_nr)
                    force_source.keyframe_insert(data_path="field.distance_max", frame=frame_nr + self.force_step - 1)
                    force_source.field.strength *= 0  # disable force
                    force_source.field.distance_max *= 0
                    force_source.keyframe_insert(data_path="field.strength", frame=frame_nr + self.force_step)
                    force_source.keyframe_insert(data_path="field.strength", frame=frame_nr + self.force_interval - 1)
                    force_source.keyframe_insert(data_path="field.distance_max", frame=frame_nr + self.force_step)
                    force_source.keyframe_insert(data_path="field.distance_max", frame=frame_nr + self.force_interval - 1)

        
        if self.add_objects:
            bpy.ops.object.select_all(action="SELECT")
            bpy.context.view_layer.objects.active = self.assets_set[0]
            print("start baking") # bake rigid body simulation
            self.bake_to_keyframes(frames[0], frames[-1], 1)
            print("baking done")

        if self.premade_scene is False:
            camdata = self.camera.data
        else:
            camdata = bpy.context.scene.camera.data

        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, "scene.blend"))

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        scene_info = {"sensor_width": sensor_width, "sensor_height": sensor_height, "focal_length": focal, "assets": ["background"], "fps": bpy.context.scene.render.fps}

        if self.premade_scene:
            assets_name = bpy.context.scene.objects.keys()
            assets_name = [name for name in assets_name if bpy.data.objects[name].type == 'MESH']
            scene_info["assets"] += assets_name
            if len(self.assets_set) > 0:
                scene_info["assets"] += [x.data.name for x in self.assets_set]
        else:
            scene_info["assets"] += [x.data.name for x in self.assets_set]

        json.dump(scene_info, open(os.path.join(self.scratch_dir, "scene_info.json"), "w"))

        use_multiview = self.views > 1

        self.set_exr_output_path(os.path.join(self.scratch_dir, "exr", "frame_"))

        if not use_multiview:
            camera_save_dir = os.path.join(self.scratch_dir, "cam")
            obj_save_dir = os.path.join(self.scratch_dir, "obj")
            os.makedirs(camera_save_dir, exist_ok=True)
            os.makedirs(obj_save_dir, exist_ok=True)

            if self.premade_scene is False:
                # set camera poses from real camera trajectory
                camera_files = glob.glob(os.path.join(self.camera_path, "*/*.txt"))
                # filter out small files
                camera_files = [c for c in camera_files if os.path.getsize(c) > 5000]
                camera_file = np.random.choice(camera_files)
                print("camera file: ", camera_file)
                camera_rt = np.loadtxt(camera_file, skiprows=1)[:, 7:].reshape(-1, 3, 4)
                self.bake_camera(camera_rt, frames)
                bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, "scene.blend"))

            for frame_nr in frames:
                bpy.context.scene.frame_set(frame_nr)

                bpy.context.scene.render.filepath = os.path.join(self.scratch_dir, "images", f"frame_{frame_nr:04d}.png")

                bpy.ops.render.render(animation=False, write_still=True)

                modelview_matrix = bpy.context.scene.camera.matrix_world.inverted()
                K = get_calibration_matrix_K_from_blender(bpy.context.scene, mode="simple")

                np.savetxt(os.path.join(camera_save_dir, f"RT_{frame_nr:04d}.txt"), modelview_matrix)
                np.savetxt(os.path.join(camera_save_dir, f"K_{frame_nr:04d}.txt"), K)
                print("Rendered frame '%s'" % bpy.context.scene.render.filepath)
        else:
            assert self.premade_scene is False
            # set camera poses from real camera trajectory
            camera_files = glob.glob(os.path.join(self.camera_path, "*/*.txt"))
            # filter out small files
            camera_files = [c for c in camera_files if os.path.getsize(c) > 5000]
            camera_files = np.random.choice(camera_files, self.views, replace=False)
            print("camera files: ", camera_files)

            self.camera_list = []
            for i in range(self.views):
                # create new cameras
                bpy.ops.object.camera_add(enter_editmode=False, align="VIEW", location=(0, 0, 0), rotation=(0, 0, 0))
                self.camera_list.append(bpy.context.object)

                self.camera = self.camera_list[i]
                bpy.context.scene.camera = self.camera

                # setup camera
                extra_camera_scale = np.random.uniform(1.0, 1.4)
                print(f"original scene_scale: {self.scale_factor}")
                print("extra_camera_scale: ", extra_camera_scale)
                self.cam_loc = (
                    mathutils.Vector(
                        (
                            np.random.uniform(-3, -3.5) * np.random.choice((-1, 1)),
                            np.random.uniform(-3, -3.5) * np.random.choice((-1, 1)),
                            np.random.uniform(1, 2.5),
                        )
                    )
                    * self.scale_factor * extra_camera_scale
                )
                self.cam_lookat = mathutils.Vector((0, 0, 0.5)) * self.scale_factor
                self.set_cam(self.cam_loc, self.cam_lookat)
                self.camera.data.lens = FOCAL_LENGTH
                self.camera.data.clip_end = 10000
                self.camera.data.sensor_width = SENSOR_WIDTH

                camera_file = camera_files[i]
                camera_rt = np.loadtxt(camera_file, skiprows=1)[:, 7:].reshape(-1, 3, 4)
                self.bake_camera(camera_rt, frames)

            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, "scene.blend"))
            for frame_nr in frames:
                bpy.context.scene.frame_set(frame_nr)

                for i in range(len(self.camera_list)):
                    camera_save_dir = os.path.join(self.scratch_dir, "cam", "view{}".format(i))
                    if not os.path.exists(camera_save_dir):
                        os.makedirs(camera_save_dir)
                    bpy.context.scene.camera = self.camera_list[i]
                    self.set_exr_output_path(os.path.join(self.scratch_dir, "view{}".format(i), "exr", "frame_"))
                    bpy.context.scene.render.filepath = os.path.join(self.scratch_dir, "view{}".format(i), "images", f"frame_{frame_nr:04d}.png")

                    bpy.ops.render.render(animation=False, write_still=True)

                    modelview_matrix = bpy.context.scene.camera.matrix_world.inverted()
                    K = get_calibration_matrix_K_from_blender(bpy.context.scene, mode="simple")

                    np.savetxt(os.path.join(camera_save_dir, f"RT_{frame_nr:04d}.txt"), modelview_matrix)
                    np.savetxt(os.path.join(camera_save_dir, f"K_{frame_nr:04d}.txt"), K)
                    print("Rendered frame '%s'" % bpy.context.scene.render.filepath)


def get_calibration_matrix_K_from_blender(scene, mode="simple"):
    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    camdata = scene.camera.data
    K = np.zeros((3, 3), dtype=np.float32)

    if mode == "simple":
        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2.0 / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.0
        K[1][2] = height / 2.0
        K[2][2] = 1.0
        K.transpose()

    if mode == "complete":

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if camdata.sensor_fit == "VERTICAL":
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        K = np.array([[alpha_u, skew, u_0], [0, alpha_v, v_0], [0, 0, 1]], dtype=np.float32)

    return K


if __name__ == "__main__":
    import sys
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1 :]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description="Render Motion in 3D Environment for HuMoR Generation.")
    parser.add_argument("--output_dir", type=str, metavar="PATH", default="./", help="img save dir")
    tmp_args = parser.parse_args(argv)
    print("args:{0}".format(tmp_args))
    
    output_dir = Path(tmp_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = RenderTap(description=__doc__)
    args.load(output_dir / 'config.json')

    renderer = Blender_render(
        samples_per_pixel=args.samples_per_pixel,
        scratch_dir=output_dir,
        render_engine=args.render_engine,
        use_gpu=args.use_gpu,
        character_path=args.character_root,
        motion_path=args.motion_root,
        camera_path=args.camera_root,
        GSO_path=args.gso_root,
        num_assets=args.num_assets,
        custom_scene=args.custom_scene,
        premade_scene=args.premade_scene,
        partnet_path=args.partnet_root,
        add_force=args.add_force,
        force_step=args.force_step,
        force_interval=args.force_interval,
        force_num=args.force_num,
        views=args.views,
        num_frames=args.num_frames,
        fps=args.fps,
        randomize=args.randomize,
        add_fog=args.add_fog,
        fog_path=args.fog_path,
        add_smoke=args.add_smoke,
        material_path=args.material_path,
        scene_scale=args.scene_scale,
        force_scale=args.force_scale,
        use_animal=args.use_animal,
        animal_path=args.animal_path,
        animal_name=args.animal_name,
        validation=args.validation,
        args=args
    )

    renderer.render()
