from pathlib import Path
import os

DATA_DIR = Path.home() / "data" / "point_odyssey" / "data"
DATA_DIR = os.getenv("DATA_DIR", DATA_DIR)