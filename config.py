import os
import yaml
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import torch
import random
import numpy as np


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False


def load_cfg(data_name, image_size, mode='cycle', curr_type='metrics'):
    
    mode = "latent" if ['celeb', 'afhq', 'imagenet'] else "pixel"
    assert data_name in ['celeb', 'afhq', 'cifar', 'imagenet'], f"Unknown dataset {data_name}."
    assert image_size in {256, 512, 32, 64}, f"Unknow model image size. Currently GAF is not supported for {image_size} size images."

    res_suffix = f"_{image_size}"
    path = Path("configs") / f"{data_name}{res_suffix}.yaml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if 'data' not in data:
        raise KeyError(f"Config is missing 'data' key. Please provide the location in the config.")
    
    curr_data = data['data']
    gen_dir = data['gen_dir']
    main_dir = os.path.join(gen_dir, f"gaf_{curr_data}{image_size}_{curr_type}")
    timestamp = datetime.now().strftime("%d_%m_%Y%H_%M_%S")
    final_path = os.path.join(main_dir, f"{timestamp}_{mode}_gen")
    
    data["gen_dir"] = final_path

    return SimpleNamespace(**data)


def save_to_cfg(cfg, path):
    with open(path, "w") as f:
        yaml.dump(vars(cfg), f, sort_keys=False)