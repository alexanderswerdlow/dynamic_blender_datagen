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
    background_hdr_folder: Optional[Path] = DATA_DIR / 'hdri'
    background_hdr_path: Optional[Path] = None
    add_fog: bool = False
    fog_path: Path = DATA_DIR / 'blender_assets' / 'fog.blend'
    num_frames: int = 1100
    samples_per_pixel: int = 1024
    use_gpu: bool = False
    randomize: bool = False
    material_path: Path = DATA_DIR / 'blender_assets' / 'materials.blend'
    fps: Optional[int] = None
    remove_temporary_files: bool = True
    scene_scale: float = 1.0
    force_scale: float = 1.0
    export_segmentation: bool = True
    export_uv: bool = False
    export_normals: bool = False
    export_flow: bool = False
    export_object_coordinates: bool = False
    add_objects: bool = True

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
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    # Animal
    animal_path: Path = DATA_DIR / 'deformingthings4d'
    add_smoke: bool = False
    animal_name: str = None
    use_animal: bool = False
    premade_scene: bool = False

    slurm_task_index: Optional[int] = None
    final_output_dir: Optional[Path] = None

RenderTap = to_tap_class(RenderArgs)

def remove_file_or_folder(path: Path, raise_error: bool = True):
    if path.exists():
        if path.is_file():
            os.remove(path)
        else:
            shutil.rmtree(path)
    else:
        if raise_error:
            raise ValueError(f"Path {path} does not exist")

def render(args: RenderArgs, use_tmpfs: bool = False):
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    print(f"Render args: {args}")
    print(f"Current path: {current_path}")

    tap = RenderTap(description=__doc__)
    args = tap.from_dict(dataclasses.asdict(args))

    repo_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    tmp_root = Path('/dev/shm') / repo_dir.stem

    def get_free_space_gb(path: Path) -> float:
        statvfs = os.statvfs(path)
        return (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)

    required_space_gb = 24
    try:
        if use_tmpfs:
            run_command("python scripts/check_tmp.py")
            tmpfs_space = get_free_space_gb(tmp_root.parent)
            print(f"TMPFS space: {tmpfs_space}")
        if use_tmpfs and tmp_root.parent.exists() and tmpfs_space >= required_space_gb:
            args.final_output_dir = args.output_dir
            args.output_dir = (tmp_root / args.output_dir.relative_to(args.output_dir.parent.parent.parent)).resolve()
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(args.final_output_dir, args.output_dir, dirs_exist_ok=True)
            print(f"Using tmpfs for output: {args.output_dir}")
            print(f"Final output dir: {args.final_output_dir}")
        elif use_tmpfs and tmp_root.parent.exists():
            print(f"We do not have enough space on TMPFS. We need at least {required_space_gb} GB, saving directly to disk.")

        if args.rendering:    
            args.save(args.output_dir / 'config.json')
            args.save(args.output_dir / 'render_config.json')
            blender_path = f'singularity run --bind {os.getcwd()}/singularity/config:/.config --nv singularity/blender.sif' if args.use_singularity else 'blender'
            rendering_script = (
                f"{blender_path} --background --python render.py -- "
                f"--output_dir {args.output_dir} "
            )

            run_command(rendering_script)

        if args.remove_temporary_files:
            remove_file_or_folder(args.output_dir / 'tmp', raise_error=False)
            run_command("python scripts/check_tmp.py")

        if args.export_obj:
            obj_script = f"{blender_path} --background --python {str(current_path / 'utils' / 'export_obj.py')} \
            -- --scene_root {args.output_dir / 'scene.blend'} --output_dir {args.output_dir} --premade_scene {args.premade_scene}"
            run_command(obj_script)

        if args.remove_temporary_files:
            remove_file_or_folder(args.output_dir / 'scene.blend')
            remove_file_or_folder(args.output_dir / 'scene.blend1', raise_error=False)

        python_path = f"singularity exec --bind {os.getcwd()}/singularity/config:/.config --nv singularity/blender.sif /bin/bash -c '$BLENDERPY" if args.use_singularity else "python"
        postfix = "'" if args.use_singularity else ""
        if args.exr:
            exr_script = f"{python_path} {str(current_path / 'utils' / 'openexr_utils.py')} --output_dir {args.output_dir}" + postfix
            run_command(exr_script)

        if args.remove_temporary_files:
            remove_file_or_folder(args.output_dir / 'exr')
            run_command("python scripts/check_tmp.py")

        if args.export_tracking:
            tracking_script = f"{python_path} {str(current_path / 'export_tracks.py')} --output_dir {args.output_dir}" + postfix
            run_command(tracking_script)

        if args.remove_temporary_files:
            remove_file_or_folder(args.output_dir / 'obj')
            remove_file_or_folder(args.output_dir / 'exr_img')
            remove_file_or_folder(args.output_dir / 'images')
            
        if args.final_output_dir is not None:
            dir_size = sum(f.stat().st_size for f in args.output_dir.glob('**/*') if f.is_file()) / (1024 ** 3)
            print(f"Rendered firectory size: {dir_size:.2f} GB")

            if args.final_output_dir.exists():
                dir_size = sum(f.stat().st_size for f in args.final_output_dir.glob('**/*') if f.is_file()) / (1024 ** 3)
                print(f"Final output directory size: {dir_size:.2f} GB")
                shutil.rmtree(args.final_output_dir)

            print(f"Moving {args.output_dir} to {args.final_output_dir.parent}")
            shutil.move(str(args.output_dir), str(args.final_output_dir.parent))
            print(f"Move to final output dir: {args.final_output_dir}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        print(f"Exception: {e}")

        if args.final_output_dir is not None and args.final_output_dir.exists():
            print(f"Removing final output dir: {args.final_output_dir}")
            shutil.rmtree(args.final_output_dir)

        raise e

if __name__ == '__main__':
    RenderTap = to_tap_class(RenderArgs)
    tap = RenderTap(description=__doc__)
    args = tap.parse_args()
    render(RenderArgs(**args.as_dict()))
