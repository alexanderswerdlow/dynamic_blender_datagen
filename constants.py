from pathlib import Path
import os

DATA_DIR = Path.home() / "data" / "point_odyssey" / "data"
if DATA_DIR.exists() is False:
    DATA_DIR = Path("data")

DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DIR))