import argparse
import os
import subprocess

def run_command(command):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {command}\n{result.stderr}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default=None)
    parser.add_argument('--scene_dir', type=str, default='./data/demo_scene/robot.blend')
    parser.add_argument('--output_dir', type=str, default='./results/robot_demo')
    parser.add_argument('--use_singularity', default=False, action='store_true')

    # rendering settings
    parser.add_argument('--rendering',  default=False, action='store_true')
    parser.add_argument('--background_hdr_path', type=str, default='./data/hdri/')
    
    parser.add_argument('--add_fog', default=False, action='store_true')
    parser.add_argument('--fog_path', default='./data/blender_assets/fog.blend', type=str)
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--end_frame', type=int, default=1100)
    parser.add_argument('--samples_per_pixel', type=int, default=1024)
    parser.add_argument('--use_gpu',  default=False, action='store_true')
    parser.add_argument('--randomize', default=False, action='store_true')
    parser.add_argument('--material_path', default='./data/blender_assets/materials.blend', type=str)
    parser.add_argument('--skip_n', default=1, type=int)

    # exr settings
    parser.add_argument('--exr',  default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--frame_idx', type=int, default=1)

    # export obj settings
    parser.add_argument('--export_obj',  default=False, action='store_true')
    parser.add_argument('--ignore_character',  default=False, action='store_true')

    # export tracking settings
    parser.add_argument('--export_tracking',  default=False, action='store_true')
    parser.add_argument('--sampling_scene_points', type=int, default=20000)
    parser.add_argument('--sampling_character_num', type=int, default=5000)

    # Human
    parser.add_argument('--sampling_points', type=int, default=5000)
    parser.add_argument('--character_root', type=str, metavar='PATH', default='./data/robots/')
    parser.add_argument('--use_character', type=str, metavar='PATH', default=None)
    parser.add_argument('--motion_root', type=str, metavar='PATH', default='./data/motions/')
    parser.add_argument('--scene_root', type=str, default='./data/blender_assets/hdri_plane.blend')
    parser.add_argument('--indoor_scale', action='store_true', default=False)
    parser.add_argument('--partnet_root', type=str, metavar='PATH', default='./data/partnet/')
    parser.add_argument('--gso_root', type=str, metavar='PATH', default='./data/GSO/')
    parser.add_argument('--render_engine', type=str, default='CYCLES')
    parser.add_argument('--force_num', type=int, default=5)
    parser.add_argument('--add_force', default=False, action='store_true')
    parser.add_argument('--force_step', type=int, default=3)
    parser.add_argument('--force_interval', type=int, default=120)
    parser.add_argument('--camera_root', type=str, metavar='PATH', default='./data/camera_trajectory/MannequinChallenge')
    parser.add_argument('--num_assets', type=int, default=5)

    # Animal
    parser.add_argument('--animal_root', type=str, default='./data/deformingthings4d')
    parser.add_argument('--add_smoke', default=False, action='store_true')
    parser.add_argument('--animal_name', type=str, metavar='PATH', default=None)

    args = parser.parse_args()
    current_path = os.path.dirname(os.path.realpath(__file__))

    print(f"Current path: {current_path}")
    print(f"Running command: {args.type}")
    print("args:{0}".format(args))

    blender_path = 'singularity run --nv blender_binary.sig' if args.use_singularity else 'blender'
    if args.type is None:
        if args.rendering:
            rendering_script = (
                f"{blender_path} --background --python {current_path}/render_single.py -- "
                f"--output_dir {args.output_dir} "
                f"--scene {args.scene_dir} "
                f"--render_engine CYCLES "
                f"--start_frame {args.start_frame} "
                f"--end_frame {args.end_frame} "
                f"--samples_per_pixel {args.samples_per_pixel} "
                f"--background_hdr_path {args.background_hdr_path} "
                f"--skip_n {args.skip_n}"
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
            exr_script = f'python -m utils.openexr_utils --data_dir {args.output_dir} --output_dir {args.output_dir}/exr_img --batch_size {args.batch_size} --frame_idx {args.frame_idx}'
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
            assert args.skip_n == 1
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
                    f"{blender_path} --background --python {current_path}/render_human.py -- "
                    f"--output_dir {args.output_dir} --character_root {args.character_root} "
                    f"--partnet_root {args.partnet_root} --gso_root {args.gso_root} "
                    f"--background_hdr_path {args.background_hdr_path} --scene_root {args.scene_root} "
                    f"--camera_root {args.camera_root} --num_assets {args.num_assets} "
                    f"--render_engine {args.render_engine} --force_num {args.force_num} "
                    f"--force_step {args.force_step} --force_interval {args.force_interval} "
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
            -- --scene_root {os.path.join(args.output_dir, "scene.blend")} --output_dir {args.output_dir}'
            run_command(obj_script)
        if args.exr:
            exr_script = f'python -m utils.openexr_utils --data_dir {args.output_dir} --output_dir {args.output_dir}/exr_img --batch_size {args.batch_size} --frame_idx {args.frame_idx}'
            run_command(exr_script)

        if args.export_tracking:
            tracking_script = f'python -m utils.gen_tracking --data_root {args.output_dir} --cp_root {args.output_dir} --sampling_points {args.sampling_points} --sampling_scene_points {args.sampling_scene_points}'
            run_command(tracking_script)

