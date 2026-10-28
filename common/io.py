import hashlib
import json
import os
import platform
import subprocess

import torch
import transformers
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(base_path, method_path, smoke):
    #base recipe, then the method file, then the smoke overrides on top
    cfg= load_yaml(base_path)
    cfg.update(load_yaml(method_path))

    smoke_cfg= cfg.pop("smoke")
    if smoke:
        cfg.update(smoke_cfg)

    cfg["smoke"]= smoke

    return cfg


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def git_commit():
    try:
        commit= subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty= subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    return commit + ("-dirty" if dirty else "")


def run_metadata(cfg, device):
    gpu= torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    return {
        "git_commit": git_commit(),
        "seed": cfg["seed"],
        "config": cfg,
        "gpu": gpu,
        "host": platform.node(),
        "torch": torch.__version__,
        "transformers": transformers.__version__
    }


def write_json(path, obj):
    #write then rename so a crash never leaves half a json file
    tmp= path + ".tmp"

    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)

    os.replace(tmp, path)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    if not os.path.exists(path):
        return []

    rows= []
    with open(path) as f:
        for line in f:
            #a crash can leave one half-written last line
            try:
                rows.append(json.loads(line))
            except ValueError:
                break

    return rows


def truncate_jsonl(path, max_update):
    #drop rows written after the checkpoint we resume from, so no update is logged twice
    rows= [r for r in read_jsonl(path) if r["update"] <= max_update]

    tmp= path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    os.replace(tmp, path)

    return rows
