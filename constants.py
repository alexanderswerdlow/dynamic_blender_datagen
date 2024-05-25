from pathlib import Path
import os
import sys
import subprocess

DATA_DIR = Path.home() / "data" / "point_odyssey" / "data"
if DATA_DIR.exists() is False:
    DATA_DIR = Path("data")

DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DIR))

def run_command(command):
    print(f"Running command: {command}")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True) as proc:
        with os.fdopen(sys.stdout.fileno(), 'wb', closefd=False) as stdout:
            for line in proc.stdout:
                stdout.write(line)
                stdout.flush()

        return_code = proc.wait()
        if return_code != 0:
            raise Exception(f"Command failed: {command}")

