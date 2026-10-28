import argparse
import functools
import json
import math
import os
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from common.amp import make_scaler
from common.checkpoint import latest_valid_checkpoint, prune_rolling, save_checkpoint
from common.io import (
    append_jsonl,
    config_hash,
    load_config,
    read_jsonl,
    run_metadata,
    truncate_jsonl,
    write_json
)
from common.seed import get_rng_state, seed_everything, set_rng_state
from task1.data import collate
from task1.methods.full_ft import full_ft_train
from task1.methods.lora import add_lora, lora_train

BASE_CONFIG= "configs/task1/base.yaml"
ADAPTER_METHODS= ["lora", "oplora"]


def default_output_dir():
    return "/kaggle/working/outputs" if os.path.exists("/kaggle") else "outputs"


def lr_factor(update, warmup, total):
    #update is 0-based: linear warmup to peak, then cosine to 0 (docs/decisions.md)
    if update < warmup:
        return (update + 1) / warmup

    progress= (update - warmup) / max(1, total - warmup)

    return 0.5 * (1.0 + math.cos(math.pi * progress))


def smoke_examples(cfg):
    #synthetic pre-tokenized examples: random ids, a masked prompt prefix, the rest supervised
    gen= torch.Generator().manual_seed(cfg["seed"])
    vocab= cfg["smoke_model"]["vocab_size"]

    examples= []
    for _ in range(cfg["n_examples"]):
        length= int(torch.randint(cfg["min_len"], cfg["max_len"] + 1, (1,), generator=gen))
        n_prompt= int(torch.randint(2, length - 1, (1,), generator=gen))

        ids= torch.randint(1, vocab, (length,), generator=gen)
        labels= ids.clone()
        labels[:n_prompt]= -100

        examples.append({"input_ids": ids, "labels": labels, "length": length})

    return examples, 0


def load_examples(cfg, output_dir):
    if cfg["smoke"]:
        return smoke_examples(cfg)

    #built by `python -m task1.data`, must match the committed selection token for token
    data= torch.load(os.path.join(output_dir, "task1", "data", "train_tokenized.pt"))

    with open(cfg["selection_path"]) as f:
        selection= json.load(f)

    assert data["fingerprint"] == selection["fingerprint"], "tokenized data differs from splits/task1 selection"
    assert len(data["examples"]) == cfg["n_examples"]

    return data["examples"], data["pad_id"]


def build_model(cfg, device):
    if cfg["smoke"]:
        config= Qwen3Config(**cfg["smoke_model"], tie_word_embeddings=True)

        #fixed seed so every process starts from the same random init (stands in for the hf checkpoint)
        torch.manual_seed(cfg["seed"])
        model= Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa", dtype=torch.float32)
    else:
        #full ft keeps fp32 master params, adapter runs freeze an fp16 backbone
        dtype= torch.float16 if cfg["method"] in ADAPTER_METHODS else torch.float32
        model= AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=dtype, attn_implementation="sdpa")

    model.to(device)

    if cfg["method"] == "lora":
        add_lora(
            model,
            cfg["lora_targets"],
            rank=cfg["rank"],
            alpha=cfg["alpha"],
            dropout=cfg["lora_dropout"],
            seed=cfg["seed"]
        )

    if cfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache= False

    return model


def make_optimizer(model, cfg, device):
    #no weight decay on biases and on params with fewer than 2 dims (norms)
    decay= []
    no_decay= []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer= torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0}
        ],
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
        fused=device.type == "cuda"
    )

    return optimizer, decay + no_decay


def trainable_state(model):
    #full ft: every param (fp32 master copy), adapters: only the adapters, never the frozen backbone
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def write_checkpoint(run_dir, kind, model, optimizer, scheduler, scaler, state, cfg_hash):
    files= {
        "trainable.pt": trainable_state(model),
        "state.pt": {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": get_rng_state(),
            "progress": dict(state)
        }
    }

    meta= {
        "examples_consumed": state["examples_consumed"],
        "config_hash": cfg_hash
    }

    return save_checkpoint(run_dir, kind, state["update"], files, meta)


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, device):
    trainable= torch.load(os.path.join(ckpt_dir, "trainable.pt"), map_location=device)

    params= dict(model.named_parameters())
    with torch.no_grad():
        for name, value in trainable.items():
            params[name].copy_(value)

    #our own file: holds python/numpy rng states, so not weights_only
    saved= torch.load(os.path.join(ckpt_dir, "state.pt"), map_location="cpu", weights_only=False)

    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    set_rng_state(saved["rng"])

    return saved["progress"]


def main(args):
    cfg= load_config(BASE_CONFIG, args.config, args.smoke)

    if args.rank is not None:
        cfg["rank"]= args.rank
    if args.alpha is not None:
        cfg["alpha"]= args.alpha
    if args.ckpt_every is not None:
        cfg["ckpt_every"]= args.ckpt_every

    cfg_hash= config_hash(cfg)

    seed_everything(cfg["seed"])

    device= torch.device(args.device)
    print("Using device:", device)

    #fp16 autocast + grad scaling on gpu, plain fp32 for cpu smoke runs
    use_fp16= device.type == "cuda" and not cfg["smoke"]

    run_dir= os.path.join(args.output_dir, "task1", args.run_name)
    results_dir= os.path.join(args.results_dir, "task1", args.run_name)
    log_path= os.path.join(results_dir, "train_log.jsonl")

    ckpt_path= latest_valid_checkpoint(run_dir)

    if args.fresh and ckpt_path is not None:
        if not args.force:
            print(f"refusing --fresh: {ckpt_path} exists. pass --fresh --force to delete it and start over")
            sys.exit(1)

        print("--fresh --force: deleting", run_dir, "and", log_path)
        shutil.rmtree(run_dir)
        if os.path.exists(log_path):
            os.remove(log_path)
        ckpt_path= None

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    examples, pad_id= load_examples(cfg, args.output_dir)

    examples_per_update= cfg["micro_batch"] * cfg["grad_accum"]
    assert len(examples) >= cfg["num_updates"] * examples_per_update

    model= build_model(cfg, device)
    optimizer, params= make_optimizer(model, cfg, device)

    scheduler= torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda u: lr_factor(u, cfg["warmup_updates"], cfg["num_updates"])
    )
    scaler= make_scaler(use_fp16)

    n_trainable= sum(p.numel() for p in params)
    print("trainable params:", n_trainable)

    state= {
        "update": 0,
        "examples_consumed": 0,
        "skipped_updates": 0,
        "train_seconds": 0.0,
        "ckpt_seconds": 0.0
    }

    if ckpt_path is None:
        #new run: start the log from scratch
        if os.path.exists(log_path):
            os.remove(log_path)
    else:
        with open(os.path.join(ckpt_path, "manifest.json")) as f:
            saved_hash= json.load(f)["config_hash"]

        if saved_hash != cfg_hash:
            print("config changed since this checkpoint was written, refusing to resume:", ckpt_path)
            sys.exit(1)

        state= load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
        truncate_jsonl(log_path, state["update"])

        print(f"resumed from update {state['update']} ({ckpt_path})")

    meta= run_metadata(cfg, device)
    meta["run_name"]= args.run_name
    meta["trainable_params"]= n_trainable
    meta["dtype"]= "fp16 autocast" if use_fp16 else "fp32"
    write_json(os.path.join(results_dir, "config.json"), meta)

    #data order is fixed: resume by skipping exactly the examples already consumed
    #own generator so creating the loader doesn't draw from the global torch rng (would break bitwise resume)
    loader= DataLoader(
        examples[state["examples_consumed"]:],
        batch_size=cfg["micro_batch"],
        shuffle=False,
        collate_fn=functools.partial(collate, pad_id=pad_id),
        num_workers=0 if cfg["smoke"] else args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg["seed"])
    )
    batches= iter(loader)

    last_ckpt_time= time.time()
    session_start= time.time()
    session_start_update= state["update"]

    while state["update"] < cfg["num_updates"]:
        microbatches= [next(batches) for _ in range(cfg["grad_accum"])]

        t0= time.time()
        lr= optimizer.param_groups[0]["lr"]

        if cfg["method"] == "full_ft":
            loss, grad_norm, n_tokens, skipped= full_ft_train(
                model=model,
                params=params,
                microbatches=microbatches,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_fp16=use_fp16,
                max_grad_norm=cfg["max_grad_norm"]
            )
        elif cfg["method"] == "lora":
            loss, grad_norm, n_tokens, skipped= lora_train(
                model=model,
                params=params,
                microbatches=microbatches,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_fp16=use_fp16,
                max_grad_norm=cfg["max_grad_norm"]
            )
        else:
            raise ValueError(f"method {cfg['method']} not wired into the loop yet")

        #schedule advances on every update, skipped or not
        scheduler.step()

        #one host sync per update
        loss_val, grad_norm_val= torch.stack([loss.float(), grad_norm.float()]).tolist()

        state["update"]+= 1
        state["examples_consumed"]+= examples_per_update
        state["skipped_updates"]+= int(skipped)
        state["train_seconds"]+= time.time() - t0

        row= {
            "update": state["update"],
            "loss": loss_val,
            "lr": lr,
            "grad_norm": grad_norm_val,
            "scale": scaler.get_scale(),
            "skipped": bool(skipped),
            "n_tokens": n_tokens,
            "examples_consumed": state["examples_consumed"]
        }
        append_jsonl(log_path, row)

        if skipped:
            print(f"update {state['update']}: skipped by grad scaler (inf/nan grads)")

        done= state["update"] - session_start_update
        if done == 5 or state["update"] % 25 == 0:
            sec_per_update= (time.time() - session_start) / done
            remaining= (cfg["num_updates"] - state["update"]) * sec_per_update

            print(
                f"update {state['update']}/{cfg['num_updates']}: "
                f"loss={loss_val:.4f}, "
                f"lr={lr:.2e}, "
                f"{sec_per_update:.2f}s/update, "
                f"eta={remaining / 3600:.2f}h"
            )

        #testing hook: simulate a session dying without warning
        if args.crash_at_update == state["update"]:
            print(f"crash_at_update: killing the process after update {state['update']}")
            sys.stdout.flush()
            os._exit(1)

        is_milestone= state["update"] in cfg["milestones"]
        is_rolling= state["update"] % cfg["ckpt_every"] == 0
        overdue= time.time() - last_ckpt_time > cfg["ckpt_every_minutes"] * 60

        if is_milestone or is_rolling or overdue:
            t1= time.time()
            kind= "milestones" if is_milestone else "rolling"

            path= write_checkpoint(run_dir, kind, model, optimizer, scheduler, scaler, state, cfg_hash)
            prune_rolling(run_dir, cfg["keep_rolling"])

            write_json(os.path.join(results_dir, f"{args.run_name}_history.json"), read_jsonl(log_path))

            state["ckpt_seconds"]+= time.time() - t1
            last_ckpt_time= time.time()
            print("saved", path)

    print(
        f"finished {args.run_name}: "
        f"updates={state['update']}, "
        f"skipped={state['skipped_updates']}, "
        f"train_time={state['train_seconds']:.1f}s"
    )


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )
    parser.add_argument(
        "--fresh",
        action="store_true"
    )
    parser.add_argument(
        "--force",
        action="store_true"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir()
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=None
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2
    )
    parser.add_argument(
        "--crash_at_update",
        type=int,
        default=None,
        help="testing only: hard-exit after this update to simulate a dead session"
    )

    main(parser.parse_args())
