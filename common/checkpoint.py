import hashlib
import json
import os
import re
import shutil
import time

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


def prune_checkpoints(run_dir, keep, keep_milestones):
    #keep the newest `keep` checkpoints; milestone ones are kept forever only if keep_milestones
    #(adapters: tiny, keep all; full ft: ~7 GB each, only the fp16 exports are kept for evaluation)
    removed= []
    kept= 0

    for path in list_checkpoints(run_dir):
        is_milestone= os.path.basename(os.path.dirname(path)) == "milestones"

        if is_milestone and keep_milestones:
            continue

        if kept < keep:
            kept+= 1
            continue

        shutil.rmtree(path)
        removed.append(path)

    return removed


def dir_bytes(path):
    total= 0

    for root, _, files in os.walk(path):
        for name in files:
            full= os.path.join(root, name)
            if os.path.isfile(full):
                total+= os.path.getsize(full)

    return total


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    return True


def reserve_disk(output_dir, run_name, run_dir, need_bytes, is_full_ft):
    #registry of running jobs on this disk: refuse to start if the jobs together would overflow it,
    #or if another full-ft job is running (never full_ft and l2sp in the same session)
    os.makedirs(output_dir, exist_ok=True)
    registry_path= os.path.join(output_dir, ".disk_reservations.json")
    lock_path= registry_path + ".lock"

    #crude lock so two jobs starting together don't both read an empty registry
    start= time.time()
    while True:
        try:
            fd= os.open(lock_path, os.O_CREAT | os.O_EXCL)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > 60:
                os.remove(lock_path)
            time.sleep(0.2)

    try:
        registry= {}
        if os.path.exists(registry_path):
            with open(registry_path) as f:
                registry= json.load(f)

        others= {name: r for name, r in registry.items() if name != run_name and pid_alive(r["pid"])}

        if is_full_ft:
            for name, r in others.items():
                if r["is_full_ft"]:
                    return False, f"full-FT job {name} is already running on this machine"

        #space the others will still take = their reservation minus what they already wrote
        others_pending= sum(max(0, r["need_bytes"] - dir_bytes(r["run_dir"])) for r in others.values())
        own_pending= max(0, need_bytes - dir_bytes(run_dir))
        free= shutil.disk_usage(output_dir).free

        print(
            f"disk: free={free / 1e9:.1f} GB, "
            f"this job needs {own_pending / 1e9:.1f} GB more, "
            f"other running jobs {others_pending / 1e9:.1f} GB more"
        )

        if own_pending + others_pending > free:
            return False, "not enough disk for this job plus the running ones"

        others[run_name]= {
            "pid": os.getpid(),
            "need_bytes": need_bytes,
            "run_dir": run_dir,
            "is_full_ft": is_full_ft
        }

        with open(registry_path, "w") as f:
            json.dump(others, f, indent=2)
    finally:
        os.remove(lock_path)

    return True, "ok"
