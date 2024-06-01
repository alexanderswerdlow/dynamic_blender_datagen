import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional
from tap import Tap, to_tap_class
from constants import DATA_DIR, run_command
import dataclasses

@dataclass
class RenderArgs():
    type: str = None
    output_dir: Path = Path('results') / 'robot_demo'
    use_singularity: bool = False
    validation: bool = False

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

    # Human
    sampling_points: int = 5000
    character_root: Path = DATA_DIR / 'robots'
    motion_root: Path = DATA_DIR / 'motions'
    custom_scene: Path = DATA_DIR / 'blender_assets' / 'hdri_plane.blend'
    partnet_root: Path = DATA_DIR / 'partnet'
    gso_root: Path = DATA_DIR / 'GSO'
    render_engine: str = 'CYCLES'
    force_num: int = 5
    add_force: bool = False
    force_step: int = 3
    force_interval: int = 120
    camera_root: Path = DATA_DIR / 'camera_trajectory' / 'MannequinChallenge'
    num_assets: int = 5
    views: int = 1

    # Animal
    animal_path: Path = DATA_DIR / 'deformingthings4d'
    add_smoke: bool = False
    animal_name: str = None
    use_animal: bool = False
    indoor: bool = False

RenderTap = to_tap_class(RenderArgs)

def render(args: RenderArgs):
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    print(f"Render args: {args}")
    print(f"Current path: {current_path}")
    print(f"Rendering type: {args.type}")

    tap = RenderTap(description=__doc__)
    args = tap.from_dict(dataclasses.asdict(args))
    args.save(args.output_dir / 'config.json')
    
    blender_path = f'singularity run --bind {os.getcwd()}/singularity/config:/.config --nv singularity/blender.sif' if args.use_singularity else 'blender'
    rendering_script = (
        f"{blender_path} --background --python render_human.py -- "
        f"--output_dir {args.output_dir} "
    )

    run_command(rendering_script)

    if args.export_obj:
        obj_script = f"{blender_path} --background --python {str(current_path / 'utils' / 'export_obj.py')} \
        -- --scene_root {args.output_dir / 'scene.blend'} --output_dir {args.output_dir} --indoor {args.indoor}"
        run_command(obj_script)

    python_path = f"singularity exec --bind {os.getcwd()}/singularity/config:/.config --nv singularity/blender.sif /bin/bash -c '$BLENDERPY" if args.use_singularity else "python"
    postfix = "'" if args.use_singularity else ""
    if args.exr:
        exr_script = f"{python_path} {str(current_path / 'utils' / 'openexr_utils.py')} --data_dir {args.output_dir} --output_dir {args.output_dir}/exr_img --batch_size {args.batch_size} --frame_idx {args.frame_idx}" + postfix
        run_command(exr_script)

    if args.end_frame <= 64:
        if args.export_tracking:
            tracking_script = f"{python_path} {str(current_path / 'export_tracks.py')} --data_root {args.output_dir}" + postfix
            run_command(tracking_script)

        if args.remove_temporary_files:
            exr_path = args.output_dir / 'exr'
            if exr_path.exists():
                shutil.rmtree(exr_path)
            else:
                raise ValueError(f"exr_path {exr_path} does not exist")
    else:
        print(f"End frame is {args.end_frame}, skipping exporting tracking and removing temporary files. You must export tracks separately.")


if __name__ == '__main__':
    RenderTap = to_tap_class(RenderArgs)
    tap = RenderTap(description=__doc__)
    args = tap.parse_args()
    render(RenderArgs(**args.as_dict()))
