import multiprocessing
from functools import partial
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import objaverse
import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str
from objaverse.xl import downloaders


def sanitize_filename(filename: str) -> str:
    return "".join(map_chars.get(c, c) for c in filename if c.isalnum() or map_chars.get(c, c) in (" ", "_", "-", "__"))

map_chars = {"/": "__", " ": "_"}

def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    save_object_dir: Path,
) -> bool:
    """Called when an object is successfully found and downloaded.

    Here, the object has the same sha256 as the one that was downloaded with
    Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
    it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        render_dir (str): Directory where the objects will be rendered.

    Returns: True if the object was rendered successfully, False otherwise.
    """
    save_uid = get_uid_from_str(file_identifier)

    save_object_dir = Path(save_object_dir)
    save_object_dir.mkdir(parents=True, exist_ok=True)

    local_path = Path(local_path)

    destination = (save_object_dir / sanitize_filename(local_path.stem)).with_suffix(local_path.suffix)
    copy2(local_path, destination)

    print(f"Downloaded {local_path} to {destination} with save_uid {save_uid}.")

def download_objects(
    save_object_dir: Optional[Path] = "debug",
    download_dir: Optional[str] = None,
    num_to_download: int = 8,
    should_break: bool = False,
) -> None:
    processes = min(multiprocessing.cpu_count(), num_to_download * 2)

    sources = list(downloaders.keys())
    sources.remove('thingiverse')
    
    selected_source = np.random.choice(sources)
    print(f"Using source: {selected_source}.")

    # get the objects to render
    objects = oxl.get_annotations()
    objects = objects.query('fileType == "glb" or fileType == "fbx" or fileType == "usd" or fileType == "blend"')
    objects = objects.query(f'source == "{selected_source}"')

    if should_break:
        breakpoint()
        
    print(f"Provided {len(objects)} objects total.")

    # shuffle the objects
    objects = objects.sample(n=num_to_download * 2).reset_index(drop=True)
    print(f"Downloading {len(objects)} objects.")

    oxl.download_objects(
        objects=objects,
        processes=processes,
        download_dir=download_dir,
        handle_found_object=partial(
            handle_found_object,
            save_object_dir=save_object_dir,
        ),
    )

    print(f"Downloaded {len(objects)} objects to {save_object_dir}.")


if __name__ == "__main__":
    import fire
    fire.Fire(download_objects)
