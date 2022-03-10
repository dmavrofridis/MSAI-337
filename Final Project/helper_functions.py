from pathlib import Path
import os


def generate_dir_if_not_exists(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def file_exists(path):
    return os.path.exists(path)