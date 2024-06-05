import objaverse.xl as oxl
from pathlib import Path
print('imported')

download_dir = Path('data') / 'objaverse'
annotations = oxl.get_annotations(
    download_dir=download_dir
)

to_download = annotations.sample(5)


from typing import Any, Dict, Hashable

def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    print("\n\n\n---HANDLE_FOUND_OBJECT CALLED---\n",
          f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")
    
oxl.download_objects(objects=to_download, download_dir=download_dir, handle_found_object=handle_found_object)

breakpoint()
# oxl.download_objects(
#     # Base parameters:
#     objects: pd.DataFrame,
#     download_dir: str = "~/.objaverse",
#     processes: Optional[int] = None,  # None => multiprocessing.cpu_count()

#     # optional callback functions:
#     handle_found_object: Optional[Callable] = None,
#     handle_modified_object: Optional[Callable] = None,
#     handle_missing_object: Optional[Callable] = None,

#     # GitHub specific:
#     save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
#     handle_new_object: Optional[Callable] = None,
# )