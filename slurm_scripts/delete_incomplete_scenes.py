import shutil
import subprocess
import time
from pathlib import Path

try:
    import typer
    app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=False)
except:
    def dummy_command_decorator(func):
        def wrapper(*args, **kwargs):
            print("This is a dummy command decorator.")
            return func(*args, **kwargs)
        return wrapper

    class DummyApp:
        def command(self, *args, **kwargs):
            return dummy_command_decorator

    app = DummyApp()

def get_slurm_job_ids(username):
    result = subprocess.run(['squeue', '-u', username, '-h', '-o', '%A'], capture_output=True, text=True)
    job_ids = result.stdout.split()
    return job_ids

def find_value_in_txt(file_path, key):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith(key):
                return eval(line.split("=")[1].strip())

def find_folders_with_metadata(path, scene_paths):
    for folder in path.iterdir():
        if folder.is_dir():
            if (folder / 'slurm_metadata.txt').exists():
                scene_paths.append(folder)
            else:
                find_folders_with_metadata(folder, scene_paths)

@app.command()
def delete_incomplete_scenes(data_dir: Path = Path("/dev/shm/point_odyssey"), use_time: bool = False, dry_run: bool = False):
    print(f"Data dir: {data_dir}")
    if data_dir.exists() is False:
        print("/dev/shm/point_odyssey does not exist")
        exit()

    scene_paths = []
    find_folders_with_metadata(data_dir, scene_paths)
    print(f"Found {len(scene_paths)} scenes")

    job_ids = set(get_slurm_job_ids('aswerdlo'))
    for scene_path in scene_paths:
        try:
            if str(slurm_job_id := find_value_in_txt(scene_path / 'slurm_metadata.txt', 'SLURM_JOB_ID')) in job_ids:
                print(f"Job is running, skipping: {scene_path}")
                continue

            if slurm_job_id is None:
                print(f"Job is not using slurm, skipping: {scene_path}")
                continue

            if (scene_path / 'track_metadata.npz').exists() is False:
                if use_time is False or (scene_path / 'slurm_metadata.txt').stat().st_mtime < (time.time() - 8 * 3600):
                    file_age = time.time() - (scene_path / 'slurm_metadata.txt').stat().st_mtime
                    print(f"{'Deleting' if dry_run is False else 'Would delete'} {scene_path}, File {scene_path / 'slurm_metadata.txt'} is {file_age / 3600:.2f} hours old")
                    if dry_run is False:
                        shutil.rmtree(scene_path)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app()

