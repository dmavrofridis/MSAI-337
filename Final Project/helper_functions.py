from pathlib import Path
import os

import torch


def generate_dir_if_not_exists(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def file_exists(path):
    return os.path.exists(path)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print("Device is -> " + str(device) + " and number of devices available -> " + str(available_gpus))
