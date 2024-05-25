import argparse
import os
from constants import DATA_DIR

from constants import run_command

from pathlib import Path
from tap import Tap, to_tap_class
from dataclasses import dataclass

@dataclass
class RenderArgs():
    type: str = None
    scene_dir: Path = DATA_DIR / 'demo_scene' / 'robot.blend'
    output_dir: Path = Path('results') / 'robot_demo'
    use_singularity: bool = False

    # rendering settings
    rendering: bool = False
    background_hdr_path: Path = DATA_DIR / 'hdri'
    add_fog: bool = False
    fog_path: Path = DATA_DIR / 'blender_assets' / 'fog.blend'
    end_frame: int = 1100
    samples_per_pixel: int = 1024
    use_gpu: bool = False
    randomize: bool = False
    material_path: Path = DATA_DIR / 'blender_assets' / 'materials.blend'
    fps: Optional[int] = None

    # exr settings
    exr: bool = False
    batch_size: int = 64
    frame_idx: int = 1

    # export obj settings
    export_obj: bool = False
    ignore_character: bool = False

    # export tracking settings
    export_tracking: bool = False
    sampling_scene_points: int = 20000
    sampling_character_num: int = 5000

    # Human
    sampling_points: int = 5000
    character_root: Path = DATA_DIR / 'robots'
    use_character: Path = None
    motion_root: Path = DATA_DIR / 'motions'
    scene_root: Path = DATA_DIR / 'blender_assets' / 'hdri_plane.blend'
    indoor_scale: bool = False
    partnet_root: Path = DATA_DIR / 'partnet'
    gso_root: Path = DATA_DIR / 'GSO'
    render_engine: str = 'CYCLES'
    force_num: int = 5
    add_force: bool = False
    force_step: int = 3
    force_interval: int = 120
    camera_root: Path = DATA_DIR / 'camera_trajectory' / 'MannequinChallenge'
    num_assets: int = 5

    # Animal
    animal_root: Path = DATA_DIR / 'deformingthings4d'
    add_smoke: bool = False
    animal_name: str = None


def render(args: RenderArgs):
    current_path = os.path.dirname(os.path.realpath(__file__))
    print(f"Current path: {current_path}")
    print(f"Running command: {args.type}")
    print("args:{0}".format(args))

    singularity_cmd = 'singularity'

    pwd = os.getcwd()
    blender_path = f'{singularity_cmd} run --nv singularity/blender_binary.sig' if args.use_singularity else 'blender'
    if args.type is None:
        if args.rendering:
            rendering_script = (
                f"{blender_path} --background --python {current_path}/render_single.py -- "
                f"--output_dir {args.output_dir} "
                f"--scene {args.scene_dir} "
                f"--render_engine CYCLES "
                f"--samples_per_pixel {args.samples_per_pixel} "
                f"--background_hdr_path {args.background_hdr_path} "
                f"--end_frame {args.end_frame} "
            )
            if args.use_gpu:
                rendering_script += ' --use_gpu'
            if args.add_fog:
                rendering_script += ' --add_fog'
                rendering_script += f' --fog_path {args.fog_path}'
            if args.randomize:
                rendering_script += ' --randomize'
            if args.material_path is not None:
                rendering_script += f' --material_path {args.material_path}'

            run_command(rendering_script)
        if args.exr:
            exr_script = f'python -m utils.openexr_utils --data_dir {args.output_dir} --output_dir {args.output_dir / exr_img} --batch_size {args.batch_size} --frame_idx {args.frame_idx}'
            run_command(exr_script)
        if args.export_obj:
            obj_script = f'{blender_path} --background --python {current_path}/utils/export_scene.py \
            -- --scene_root {args.scene_dir} --output_dir {args.output_dir} --export_character {not args.ignore_character} --skip_n {args.skip_n}'
            run_command(obj_script)
        if args.export_tracking:
            tracking_script = f'python -m utils.gen_tracking_indoor --data_root {args.output_dir} --cp_root {args.output_dir} --sampling_scene_points {args.sampling_scene_points} --sampling_character_num {args.sampling_character_num}'
            run_command(tracking_script)
    else:
        if args.rendering:
            if args.type == 'animal':
                rendering_script = (
                    f"{blender_path} --background --python {current_path}/render_animal.py -- "
                    f"--output_dir {args.output_dir} --partnet_root {args.partnet_root} "
                    f"--gso_root {args.gso_root} --background_hdr_path {args.background_hdr_path} "
                    f"--animal_root {args.animal_root} --camera_root {args.camera_root} "
                    f"--num_assets {args.num_assets} --render_engine {args.render_engine} "
                    f"--force_num {args.force_num} --force_step {args.force_step} "
                    f"--force_interval {args.force_interval} --material_path {args.material_path} "
                )
                if args.use_gpu:
                    rendering_script += ' --use_gpu'
                if args.add_force:
                    rendering_script += ' --add_force'
                if args.add_smoke:
                    rendering_script += ' --add_smoke'
                if args.animal_name is not None:
                    rendering_script += f' --animal_name {args.animal_name}'
                run_command(rendering_script)
            elif args.type == 'human':
                rendering_script = (
                    f"{blender_path} --background --python render_human.py -- "
                    f"--output_dir {args.output_dir} --character_root {args.character_root} "
                    f"--partnet_root {args.partnet_root} --gso_root {args.gso_root} "
                    f"--background_hdr_path {args.background_hdr_path} --scene_root {args.scene_root} "
                    f"--camera_root {args.camera_root} --num_assets {args.num_assets} "
                    f"--render_engine {args.render_engine} --force_num {args.force_num} "
                    f"--force_step {args.force_step} --force_interval {args.force_interval} "
                    f"--end_frame {args.end_frame} "
                    f"--fps {args.fps}"
                )
                if args.use_gpu:
                    rendering_script += ' --use_gpu'
                if args.indoor_scale:
                    rendering_script += ' --indoor'
                run_command(rendering_script)
            else:
                raise ValueError('Invalid type')
        if args.export_obj:
            obj_script = f'{blender_path} --background --python {current_path}/utils/export_obj.py \
            -- --scene_root {args.output_dir / "scene.blend"} --output_dir {args.output_dir}'
            run_command(obj_script)
        if args.exr:
            exr_script = f'python -m utils.openexr_utils --data_dir {args.output_dir} --output_dir {args.output_dir}/exr_img --batch_size {args.batch_size} --frame_idx {args.frame_idx}'
            run_command(exr_script)

        if args.export_tracking:
            tracking_script = f'python -m utils.gen_tracking --data_root {args.output_dir} --cp_root {args.output_dir} --sampling_points {args.sampling_points} --sampling_scene_points {args.sampling_scene_points}'
            run_command(tracking_script)


if __name__ == '__main__':
    RenderTap = to_tap_class(RenderArgs)
    tap = RenderTap(description=__doc__)  # from the top of this script
    args = tap.parse_args()
    render(RenderArgs(**args.as_dict()))
