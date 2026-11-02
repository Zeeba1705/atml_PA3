import argparse
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionConfig, SiglipVisionModel

from common.io import load_yaml, run_metadata, write_json
from task2.build_data import SPLIT_DIR

BASE_CONFIG= "configs/task2/base.yaml"
DATA_CONFIG= "configs/task2/data.yaml"
SHARD_SIZE= 1000
#parallel image downloads (network bound)
N_THREADS= 16


def default_output_dir():
    return "/kaggle/working/outputs" if os.path.exists("/kaggle") else "outputs"


def feature_dir(output_dir, smoke):
    return os.path.join(output_dir, "task2", "features_smoke" if smoke else "features")


def shard_path(feat_dir, split, k):
    return os.path.join(feat_dir, f"{split}_{k:03d}.pt")


def load_image(image_id, split, image_root, url_template):
    #local copy if there is one, else the official coco url (3 tries)
    name= f"COCO_{SPLIT_DIR[split]}_{image_id:012d}.jpg"
    if image_root is not None:
        path= os.path.join(image_root, SPLIT_DIR[split], name)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")

    url= url_template.format(split=SPLIT_DIR[split], image_id=image_id)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def smoke_image(image_id):
    #fixed random rgb image per id
    rng= np.random.default_rng(image_id)

    return Image.fromarray(rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8))


def vision_model(cfg, smoke, device):
    if smoke:
        #one-layer random siglip with the real patch grid (14 x 14) and width (768)
        config= SiglipVisionConfig(hidden_size=cfg["vision_dim"], intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=224, patch_size=16)
        torch.manual_seed(cfg["seed"])
        return SiglipVisionModel(config).to(device).eval()

    dtype= torch.float16 if device.type == "cuda" else torch.float32

    return SiglipVisionModel.from_pretrained(cfg["vision_model"], dtype=dtype).to(device).eval()


def encode(model, processor, imgs, device):
    #(B, 196, 768). hf's last_hidden_state already includes post_layernorm: it is applied exactly once, here
    pixels= processor(images=imgs, return_tensors="pt")["pixel_values"].to(device, model.dtype)

    with torch.no_grad():
        out= model(pixel_values=pixels).last_hidden_state

    return out.half().cpu()


def main(args):
    cfg= load_yaml(BASE_CONFIG)
    cfg.update({k: v for k, v in load_yaml(DATA_CONFIG).items() if k != "smoke"})

    device= torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)

    with open(os.path.join(args.split_dir, "images.json")) as f:
        images= json.load(f)

    feat_dir= feature_dir(args.output_dir, args.smoke)
    os.makedirs(feat_dir, exist_ok=True)

    #deterministic preprocessing: the model's own processor (resize to 224, rescale, normalize), no augmentation
    processor= AutoImageProcessor.from_pretrained(cfg["vision_model"])
    model= vision_model(cfg, args.smoke, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0= time.time()
    load_seconds, n_done= 0.0, 0

    pool= ThreadPoolExecutor(N_THREADS)
    for split in ["train", "val"]:
        ids= images[split]
        for k in range(0, len(ids), SHARD_SIZE):
            path= shard_path(feat_dir, split, k // SHARD_SIZE)
            if os.path.exists(path):
                continue

            chunk= ids[k:k + SHARD_SIZE]

            t1= time.time()
            if args.smoke:
                imgs= [smoke_image(i) for i in chunk]
            else:
                imgs= list(pool.map(lambda i: load_image(i, split, args.image_root, cfg["image_url"]), chunk))
            load_seconds+= time.time() - t1

            feats= torch.cat([encode(model, processor, imgs[j:j + args.batch_size], device) for j in range(0, len(imgs), args.batch_size)])
            assert feats.shape == (len(chunk), cfg["n_patches"], cfg["vision_dim"]), feats.shape

            #shard written atomically: a rerun skips it
            torch.save({"ids": chunk, "feats": feats}, path + ".tmp")
            os.replace(path + ".tmp", path)

            n_done+= len(chunk)
            rate= n_done / (time.time() - t0)
            print(f"{split} {k + len(chunk)}/{len(ids)} ({rate:.1f} img/s)", flush=True)

    meta= run_metadata(cfg, device)
    meta.update({
        "n_images": {s: len(v) for s, v in images.items()},
        "seconds_this_session": time.time() - t0,
        "image_load_seconds_this_session": load_seconds,
        "peak_mem_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "dtype": "float16",
        "post_layernorm": "applied once (hf last_hidden_state)"
    })
    os.makedirs(args.results_dir, exist_ok=True)
    write_json(os.path.join(args.results_dir, "features_smoke.json" if args.smoke else "features.json"), meta)

    #free the vision encoder before anything loads qwen
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return 0


def load_features(output_dir, smoke):
    #{(split, image_id): row} index plus one fp16 tensor per split, all on cpu
    feat_dir= feature_dir(output_dir, smoke)
    out= {}
    for split in ["train", "val"]:
        shards= sorted(f for f in os.listdir(feat_dir) if f.startswith(split + "_") and f.endswith(".pt"))
        parts= [torch.load(os.path.join(feat_dir, f)) for f in shards]

        ids= [i for p in parts for i in p["ids"]]
        out[split]= {"index": {i: r for r, i in enumerate(ids)}, "feats": torch.cat([p["feats"] for p in parts]) if parts else None}

    return out


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits/task2"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir()
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/task2"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    raise SystemExit(main(parser.parse_args()))
