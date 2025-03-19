from pathlib import Path
import shutil



folders = ['generated/14', 'generated/53', 'generated/46', 'generated/43', 'generated/7', 'generated/58', 'generated/60', 'generated/312', 'generated/16', 'generated/5', 'generated_deformable/62', 'generated_deformable/15', 'generated_deformable/13', 'generated_deformable/59', 'generated_deformable/50', 'generated_deformable/49', 'generated_deformable/6', 'premade/0', 'premade/45', 'premade/42', 'premade/12', 'premade/51', 'premade/44', 'premade/8', 'premade/52', 'premade/9', 'premade/17', 'premade/10', 'premade/61']

root = Path('generated/train/v2')

for folder_path in folders:
    path = root / Path(folder_path)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)