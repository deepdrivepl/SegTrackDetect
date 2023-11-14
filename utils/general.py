import os
import json

from glob import glob

import torch


def create_directory(out_dir):
    # increment directory (never overwrite)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "000")):
        out_dir = os.path.join(out_dir, "000")
    else:
        if not os.listdir(out_dir):
            out_dir = os.path.join(out_dir, "000")
        else:
            subdir = sorted([int(x.split(os.sep)[-1]) for x in glob(os.path.join(out_dir, "*")) if os.path.isdir(x)])[-1]
            out_dir = os.path.join(out_dir, "%03d" % (subdir+1))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_args(out_dir, args, fname="args.json"):
    # save args to json
    with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
        info = {**vars(args)}
        json.dump(info, f, ensure_ascii=False, indent=4)
        
        
def load_model(config, device, weights=None):
    weights_path = config['weights'] if weights is None else weights
    print(f"Loading weights: {os.path.basename(weights_path)}")
    net = torch.jit.load(weights_path)
    net.to(device)
    net.eval()
    return net