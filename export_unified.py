import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

from tap import Tap, to_tap_class

from constants import DATA_DIR, run_command


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
    remove_temporary_files: bool = True
    scene_scale: float = 1.0
    force_scale: float = 1.0

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
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    print(f"Render args: {args}")
    print(f"Current path: {current_path}")
    print(f"Rendering type: {args.type}")
    
    blender_path = f'singularity run --bind {os.getcwd()}/singularity/config:/.config --nv singularity/blender.sif' if args.use_singularity else 'blender'
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
                rendering_script += ' --randomize '
            if args.material_path is not None:
                rendering_script += f' --material_path {args.material_path}'

            run_command(rendering_script)
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
                    f"--fps {args.fps} "
                    f"--samples_per_pixel {args.samples_per_pixel} "
                    f"--scene_scale {args.scene_scale} --force_scale {args.force_scale} "
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
                if args.add_smoke:
                    rendering_script += ' --add_smoke'
                if args.add_force:
                    rendering_script += ' --add_force'
                run_command(rendering_script)
            else:
                raise ValueError('Invalid type')

    if args.export_obj:
        obj_script = f"{blender_path} --background --python {str(current_path / 'utils' / ('export_scene.py' if args.type is None else 'export_obj.py'))} \
        -- --scene_root {args.output_dir / 'scene.blend'} --output_dir {args.output_dir}"
        run_command(obj_script)

    python_path = f"singularity exec --bind {os.getcwd()}/singularity/config:/.config --nv singularity/blender.sif /bin/bash -c '$BLENDERPY" if args.use_singularity else "python"
    postfix = "'" if args.use_singularity else ""
    if args.exr:
        exr_script = f"{python_path} {str(current_path / 'utils' / 'openexr_utils.py')} --data_dir {args.output_dir} --output_dir {args.output_dir}/exr_img --batch_size {args.batch_size} --frame_idx {args.frame_idx}" + postfix
        run_command(exr_script)

    if args.export_tracking:
        tracking_script = f"{python_path} {str(current_path / 'export_tracks.py')} --data_root {args.output_dir} --cp_root {args.output_dir}" + postfix
        run_command(tracking_script)

    if args.remove_temporary_files:
        exr_path = args.output_dir / 'exr'
        if exr_path.exists():
            shutil.rmtree(exr_path)
        else:
            raise ValueError(f"exr_path {exr_path} does not exist")


if __name__ == '__main__':
    RenderTap = to_tap_class(RenderArgs)
    tap = RenderTap(description=__doc__)
    args = tap.parse_args()
    render(RenderArgs(**args.as_dict()))
