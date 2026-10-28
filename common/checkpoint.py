import hashlib
import json
import os
import re
import shutil

import torch

CKPT_PATTERN= re.compile(r"^ckpt_(\d+)$")
KINDS= ["rolling", "milestones"]


def sha256_file(path):
    h= hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def fsync_path(path):
    fd= os.open(path, os.O_RDONLY)

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ckpt_name(update):
    return f"ckpt_{update:05d}"


def save_checkpoint(run_dir, kind, update, files, meta):
    #write everything into ckpt_tmp/, fsync, manifest last, then one rename -> a half-written ckpt never looks valid
    parent= os.path.join(run_dir, kind)
    os.makedirs(parent, exist_ok=True)

    tmp= os.path.join(parent, "ckpt_tmp")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    entries= {}
    for name, obj in files.items():
        path= os.path.join(tmp, name)

        torch.save(obj, path)
        fsync_path(path)

        entries[name]= {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}

    manifest= dict(meta)
    manifest["update"]= update
    manifest["files"]= entries

    manifest_path= os.path.join(tmp, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    final= os.path.join(parent, ckpt_name(update))
    if os.path.exists(final):
        shutil.rmtree(final)

    os.rename(tmp, final)
    fsync_path(parent)

    return final


def verify_checkpoint(ckpt_dir):
    manifest_path= os.path.join(ckpt_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        return False, "no manifest"

    try:
        with open(manifest_path) as f:
            manifest= json.load(f)
    except ValueError:
        return False, "unreadable manifest"

    for name, entry in manifest["files"].items():
        path= os.path.join(ckpt_dir, name)

        if not os.path.exists(path):
            return False, f"missing {name}"
        if os.path.getsize(path) != entry["bytes"]:
            return False, f"wrong size for {name}"
        if sha256_file(path) != entry["sha256"]:
            return False, f"sha256 mismatch for {name}"

    return True, "ok"


def list_checkpoints(run_dir):
    #newest first, rolling and milestone checkpoints together
    found= []

    for kind in KINDS:
        parent= os.path.join(run_dir, kind)
        if not os.path.isdir(parent):
            continue

        for name in os.listdir(parent):
            match= CKPT_PATTERN.match(name)
            if match:
                found.append((int(match.group(1)), os.path.join(parent, name)))

    found.sort(reverse=True)

    return [path for _, path in found]


def latest_valid_checkpoint(run_dir):
    for path in list_checkpoints(run_dir):
        ok, reason= verify_checkpoint(path)

        if ok:
            return path

        print(f"{os.path.basename(path)} is corrupt ({reason}), falling back to the previous checkpoint")

    return None


def prune_rolling(run_dir, keep):
    parent= os.path.join(run_dir, "rolling")
    if not os.path.isdir(parent):
        return

    names= sorted(n for n in os.listdir(parent) if CKPT_PATTERN.match(n))

    for name in names[:-keep]:
        shutil.rmtree(os.path.join(parent, name))
