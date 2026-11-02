import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from common.amp import make_scaler, scaler_step
from common.checkpoint import latest_valid_checkpoint, save_checkpoint
from common.io import append_jsonl, load_yaml, read_jsonl, run_metadata, write_json
from common.seed import get_rng_state, seed_everything, set_rng_state
from common.sync import make_api, pull_latest_checkpoint, start_sync, sync_put, sync_wait
from task1.methods.lora import LoRALinear, add_lora
from task2.connector import Connector, calibrate_scale
from task2.features import default_output_dir, load_features
from task2.vlm import build_example, collate, example_nll, prompt_segments, vlm_loss

BASE_CONFIG= "configs/task2/base.yaml"
#training split files (built by task2/build_data.py)
TRAIN_FILES= {"caption": "captions_train", "vqa": "vqa_train", "presence": "presence_pool_train", "counting": "counting_pool_train"}
DEV_FILES= {"captions": "captions_dev", "vqa": "vqa_dev", "grounding": "grounding_dev"}
#calibration sample sizes (training data only)
CALIB_IMAGES= 256
CALIB_TEXTS= 2000


def load_config(smoke):
    cfg= load_yaml(BASE_CONFIG)
    smoke_cfg= cfg.pop("smoke", None)
    if smoke and smoke_cfg:
        cfg.update(smoke_cfg)
    cfg["smoke"]= smoke

    return cfg


def load_splits(split_dir, names):
    out= {}
    for key, name in names.items():
        with open(os.path.join(split_dir, f"{name}.json")) as f:
            out[key]= json.load(f)

    return out


def build_llm(cfg, device):
    #frozen fp16 qwen (fp32 tiny random qwen for --smoke), gradient checkpointing without reentrancy
    if cfg["smoke"]:
        config= Qwen3Config(vocab_size=cfg["smoke_vocab"], tie_word_embeddings=True, max_position_embeddings=512, **cfg["smoke_llm"])
        torch.manual_seed(cfg["seed"])
        llm= Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa", dtype=torch.float32)
    else:
        llm= AutoModelForCausalLM.from_pretrained(cfg["llm_name"], dtype=torch.float16, attn_implementation="sdpa")

    llm.to(device)
    for p in llm.parameters():
        p.requires_grad= False

    llm.config.use_cache= False
    if cfg["grad_checkpointing"]:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return llm


def lr_factor(step, total, cfg):
    #per stage: linear warmup over 5% of its updates, then cosine from the peak down to 10% of it
    warm= max(1, round(cfg["warmup_frac"] * total))
    if step < warm:
        return (step + 1) / warm

    progress= (step - warm) / max(1, total - warm)

    return cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * progress))


def stage_optimizer(cfg, stage, connector, llm):
    groups= [{"params": list(connector.parameters()), "lr": stage["connector_lr"], "name": "connector"}]
    lora= [p for m in llm.modules() if isinstance(m, LoRALinear) for p in [m.lora_A, m.lora_B]]
    if stage["lora_lr"] is not None:
        groups.append({"params": lora, "lr": stage["lora_lr"], "name": "lora"})

    optimizer= torch.optim.AdamW(groups, betas=tuple(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    scheduler= torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, stage["updates"], cfg))

    return optimizer, scheduler


def trainable_params(connector, llm):
    return list(connector.parameters()) + [p for p in llm.parameters() if p.requires_grad]


def caption_item(c, cfg):
    return {"image": (c["split"], c["image_id"]), "question": cfg["caption_prompt"], "target": c["caption"], "max_target": cfg["max_caption_tokens"]}


def pair_items(pair, cfg):
    return [{"image": (it["split"], it["image_id"]), "question": it["question"], "target": it["target"], "max_target": cfg["max_answer_tokens"]} for it in pair["items"]]


def draw(sampler, data, stage, cfg):
    #one draw = two examples: a full question pair, or two captions. stage 1 walks shuffled passes over the
    #captions; later stages pick a component by the stage mixture (seeded, with replacement, no quotas)
    rng= sampler["rng"]
    if stage["name"] == "stage1":
        out= []
        for _ in range(cfg["examples_per_draw"]):
            if sampler["pos"] >= len(sampler["perm"]):
                sampler["perm"]= rng.permutation(len(data["caption"])).tolist()
                sampler["pos"]= 0
            out.append(caption_item(data["caption"][sampler["perm"][sampler["pos"]]], cfg))
            sampler["pos"]+= 1

        return "caption", out

    names= sorted(stage["mixture"])
    kind= names[rng.choice(len(names), p=[stage["mixture"][n] for n in names])]
    if kind == "caption":
        idx= rng.integers(0, len(data["caption"]), size=cfg["examples_per_draw"])
        return kind, [caption_item(data["caption"][i], cfg) for i in idx]

    return kind, pair_items(data[kind][rng.integers(0, len(data[kind]))], cfg)


def feats_for(items, feats, device):
    #stacked features of the items that have an image, in item order
    rows= [feats[s]["feats"][feats[s]["index"][i]] for s, i in (it["image"] for it in items if it["image"] is not None)]

    return torch.stack(rows).to(device) if rows else None


def make_batch(items, tokenizer, segments, cfg, supervise_eot, left_pad=False):
    examples= [
        build_example(
            segments,
            tokenizer,
            it["question"],
            it["target"],
            it["image"] is not None,
            cfg["n_patches"],
            cfg["max_question_tokens"],
            it["max_target"],
            supervise_eot
        )
        for it in items
    ]

    return collate(examples, tokenizer.pad_token_id, left_pad)


def nll_items(llm, connector, tokenizer, segments, cfg, items, feats, device, supervise_eot, batch_size):
    #teacher-forced per-example mean nll (eval mode, no grad), in item order
    out= []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for k in range(0, len(items), batch_size):
            chunk= items[k:k + batch_size]
            batch= {n: t.to(device) for n, t in make_batch(chunk, tokenizer, segments, cfg, supervise_eot).items()}
            out+= example_nll(llm, connector, batch, feats_for(chunk, feats, device)).float().cpu().tolist()

    return out


def pair_condition_items(pairs, condition, cfg):
    #both questions of every pair under one image condition; references stay the original ones
    items= []
    for p in pairs:
        a, b= p["items"]
        for it, partner in [(a, b), (b, a)]:
            image= {"correct": (it["split"], it["image_id"]), "complementary": (partner["split"], partner["image_id"]), "no_image": None}[condition]
            items.append({"image": image, "question": it["question"], "target": it["target"], "max_target": cfg["max_answer_tokens"]})

    return items


def dev_eval(llm, connector, tokenizer, segments, cfg, dev, feats, device):
    #caption reference loss (training loss definition) + answer nll without the termination token, per suite and
    #image condition, and the image-swap penalty dNLL = nll(complementary) - nll(correct)
    llm.eval()
    connector.eval()
    bs= cfg["eval_batch_size"]

    caps= [caption_item(c, cfg) for c in dev["captions"]]
    metrics= {"caption_loss": float(np.mean(nll_items(llm, connector, tokenizer, segments, cfg, caps, feats, device, True, bs)))}

    for suite in ["vqa", "grounding"]:
        nll= {c: nll_items(llm, connector, tokenizer, segments, cfg, pair_condition_items(dev[suite], c, cfg), feats, device, False, bs) for c in ["correct", "complementary", "no_image"]}
        for c, v in nll.items():
            metrics[f"{suite}_nll_{c}"]= float(np.mean(v))
        metrics[f"{suite}_delta_nll"]= float(np.mean(np.array(nll["complementary"]) - np.array(nll["correct"])))

    llm.train()
    connector.train()

    return metrics


def calibration_texts(data, tokenizer, cfg, rng):
    #token embeddings of training captions and questions only
    texts= [c["caption"] for c in data["caption"]] + [it["question"] for kind in ["vqa", "presence", "counting"] for p in data[kind] for it in p["items"]]
    pick= rng.choice(len(texts), size=min(CALIB_TEXTS, len(texts)), replace=False)

    return [i for j in pick for i in tokenizer(texts[j], add_special_tokens=False)["input_ids"]]


def calibrate(connector, llm, tokenizer, data, feats, cfg, device):
    rng= np.random.default_rng(cfg["seed"])
    ids= torch.tensor(calibration_texts(data, tokenizer, cfg, rng), device=device)

    train= feats["train"]
    rows= rng.choice(train["feats"].shape[0], size=min(CALIB_IMAGES, train["feats"].shape[0]), replace=False)

    with torch.no_grad():
        text= llm.get_input_embeddings()(ids).float()

    return calibrate_scale(connector, train["feats"][torch.tensor(sorted(rows))].to(device).float(), text)


def stage_at(cfg, update):
    #(stage index, updates done inside that stage) for a global update count
    done= 0
    for k, stage in enumerate(cfg["stages"]):
        if update < done + stage["updates"]:
            return k, update - done
        done+= stage["updates"]

    return len(cfg["stages"]) - 1, cfg["stages"][-1]["updates"]


def dev_points(cfg):
    total= sum(s["updates"] for s in cfg["stages"])
    ends= list(np.cumsum([s["updates"] for s in cfg["stages"]]))

    return sorted(set(range(0, total + 1, cfg["dev_eval_every"])) | {int(e) for e in ends})


def trainable_state(connector, llm):
    lora= {n: p.detach().cpu() for n, p in llm.named_parameters() if p.requires_grad}

    return {"connector": {k: v.detach().cpu() for k, v in connector.state_dict().items()}, "lora": lora}


def load_trainable(state, connector, llm):
    connector.load_state_dict(state["connector"])
    params= dict(llm.named_parameters())
    with torch.no_grad():
        for n, v in state["lora"].items():
            params[n].copy_(v)


def main(args):
    cfg= load_config(args.smoke)
    seed_everything(cfg["seed"])

    device= torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_fp16= device.type == "cuda"
    print("Using device:", device)

    run_dir= os.path.join(args.output_dir, "task2", args.run_name)
    results_dir= os.path.join(args.results_dir, args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    run_prefix= f"outputs/task2/{args.run_name}"
    results_prefix= "results/" + os.path.relpath(results_dir, os.path.dirname(os.path.dirname(results_dir)))

    api= make_api() if args.sync_repo else None
    sync= start_sync(args.sync_repo, f"task2-{args.run_name}", api=api) if args.sync_repo else None

    #newest checkpoint: local, or pulled from the hub if newer
    ckpt= latest_valid_checkpoint(run_dir)
    if api is not None:
        local_update= -1 if ckpt is None else int(os.path.basename(ckpt).split("_")[1])
        pulled= pull_latest_checkpoint(api, args.sync_repo, f"task2-{args.run_name}", run_prefix, run_dir, local_update, os.path.join(args.output_dir, "task2", "staging"))
        ckpt= pulled or ckpt

    data= load_splits(args.split_dir, TRAIN_FILES)
    dev= load_splits(args.split_dir, DEV_FILES)
    feats= load_features(args.output_dir, args.smoke)

    tokenizer= AutoTokenizer.from_pretrained(cfg["llm_name"])
    if args.smoke:
        cfg["smoke_vocab"]= len(tokenizer)
    segments= prompt_segments(tokenizer, cfg)

    llm= build_llm(cfg, device)
    llm_dim= llm.config.hidden_size
    assert args.smoke or llm_dim == cfg["llm_dim"]
    connector= Connector(cfg["vision_dim"], cfg["connector_hidden"], llm_dim).to(device)

    #train mode: lora dropout applies and gradient checkpointing is active (hf loads models in eval mode)
    llm.train()
    connector.train()

    sampler= {"rng": np.random.default_rng(cfg["seed"]), "perm": [], "pos": 0}
    progress= {"update": 0, "counts": {}, "calibration": None, "train_seconds": 0.0, "sessions": 0}
    saved= None
    if ckpt is not None:
        saved= torch.load(os.path.join(ckpt, "state.pt"), map_location="cpu", weights_only=False)
        progress= saved["progress"]
        sampler["rng"].bit_generator.state= saved["sampler"]["rng"]
        sampler["perm"], sampler["pos"]= saved["sampler"]["perm"], saved["sampler"]["pos"]
        print("resuming from", ckpt)
    else:
        progress["calibration"]= calibrate(connector, llm, tokenizer, data, feats, cfg, device)
        print("connector calibration:", progress["calibration"])
        write_json(os.path.join(results_dir, "config.json"), {**cfg, "calibration": progress["calibration"], "run_name": args.run_name})
    progress["sessions"]+= 1

    total= sum(s["updates"] for s in cfg["stages"])
    points= dev_points(cfg)
    log_path= os.path.join(results_dir, "train_log.jsonl")
    dev_path= os.path.join(results_dir, "dev_metrics.jsonl")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start= time.time()
    session_updates= 0
    stage_idx= None
    optimizer= scheduler= None
    scaler= make_scaler(use_fp16)

    def checkpoint(kind):
        files= {
            "trainable.pt": trainable_state(connector, llm),
            "state.pt": {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": get_rng_state(),
                "sampler": {"rng": sampler["rng"].bit_generator.state, "perm": sampler["perm"], "pos": sampler["pos"]},
                "progress": dict(progress, train_seconds=progress["train_seconds"] + time.time() - t_start)
            }
        }
        path= save_checkpoint(run_dir, kind, progress["update"], files, {"stage": cfg["stages"][stage_idx]["name"]})
        if sync is not None:
            sync_put(sync, "upload_folder", path, f"{run_prefix}/{kind}/{os.path.basename(path)}")
            sync_put(sync, "upload_folder", results_dir, results_prefix)

        return path

    while progress["update"] < total:
        k, done_in_stage= stage_at(cfg, progress["update"])
        stage= cfg["stages"][k]

        if k != stage_idx:
            #stage start (or resume inside it): lora appears from stage 2 on; a fresh optimizer and schedule per stage
            if stage["lora_lr"] is not None and not any(isinstance(m, LoRALinear) for m in llm.modules()):
                add_lora(llm, cfg["lora_targets"], cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"], cfg["seed"])
                llm.to(device)
            stage_idx= k
            optimizer, scheduler= stage_optimizer(cfg, stage, connector, llm)

            if saved is not None:
                load_trainable(torch.load(os.path.join(ckpt, "trainable.pt"), map_location=device), connector, llm)
                scaler.load_state_dict(saved["scaler"])
                set_rng_state(saved["rng"])

                #a checkpoint exactly at a stage boundary belongs to the finished stage: the new stage keeps its
                #fresh optimizer and schedule
                if done_in_stage > 0:
                    optimizer.load_state_dict(saved["optimizer"])
                    scheduler.load_state_dict(saved["scheduler"])
                saved= None

            n_train= sum(p.numel() for p in trainable_params(connector, llm))
            print(f"=== {stage['name']}: updates {done_in_stage}/{stage['updates']}, trainable params {n_train:,} ===")
            progress.setdefault("trainable_params", {})[stage["name"]]= n_train

        if progress["update"] in points and not any(r["update"] == progress["update"] for r in read_jsonl(dev_path)):
            m= dev_eval(llm, connector, tokenizer, segments, cfg, dev, feats, device)
            append_jsonl(dev_path, {"update": progress["update"], "stage": stage["name"], **m})
            print("dev", progress["update"], {n: round(v, 4) for n, v in m.items()})
            if sync is not None:
                sync_put(sync, "upload_folder", results_dir, results_prefix)

        #one optimizer update = grad_accum microbatches of micro_batch examples
        optimizer.zero_grad(set_to_none=True)
        losses= []
        for _ in range(cfg["grad_accum"]):
            items= []
            while len(items) < cfg["micro_batch"]:
                kind, new= draw(sampler, data, stage, cfg)
                progress["counts"][f"{stage['name']}/{kind}"]= progress["counts"].get(f"{stage['name']}/{kind}", 0) + 1
                items+= new

            batch= {n: t.to(device) for n, t in make_batch(items, tokenizer, segments, cfg, True).items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
                loss= vlm_loss(llm, connector, batch, feats_for(items, feats, device))

            #equal microbatch sizes: dividing by grad_accum gives the mean over the effective batch
            scaler.scale(loss / cfg["grad_accum"]).backward()
            losses.append(loss.item())

        scaler.unscale_(optimizer)
        grad_norm= torch.nn.utils.clip_grad_norm_(trainable_params(connector, llm), cfg["max_grad_norm"])
        lrs= {g["name"]: g["lr"] for g in optimizer.param_groups}
        skipped= scaler_step(optimizer, scaler)
        scheduler.step()
        progress["update"]+= 1

        append_jsonl(log_path, {
            "update": progress["update"],
            "stage": stage["name"],
            "loss": float(np.mean(losses)),
            "lr_connector": lrs["connector"],
            "lr_lora": lrs.get("lora"),
            "grad_norm": float(grad_norm),
            "scale": scaler.get_scale(),
            "skipped": skipped,
            "counts": dict(progress["counts"])
        })

        elapsed= time.time() - t_start
        session_updates+= 1
        if session_updates == 5 or progress["update"] % 25 == 0:
            per= elapsed / session_updates
            print(f"update {progress['update']}/{total} loss {np.mean(losses):.4f} | {per:.2f} s/update, ~{per * (total - progress['update']) / 3600:.2f} h left", flush=True)

        #small result files go up often, so progress is visible on the hub between checkpoints
        if sync is not None and progress["update"] % 50 == 0:
            sync_put(sync, "upload_folder", results_dir, results_prefix)

        stage_end= progress["update"] == sum(s["updates"] for s in cfg["stages"][:k + 1])
        if progress["update"] % cfg["ckpt_every"] == 0 or stage_end:
            checkpoint("milestones" if stage_end else "rolling")
        if stage_end:
            #connector/adapter weights of the finished stage
            stage_dir= os.path.join(run_dir, stage["name"])
            os.makedirs(stage_dir, exist_ok=True)
            torch.save(trainable_state(connector, llm), os.path.join(stage_dir, "weights.pt"))
            if sync is not None:
                sync_put(sync, "upload_folder", stage_dir, f"{run_prefix}/{stage['name']}")

        if args.max_hours is not None and elapsed > args.max_hours * 3600 and progress["update"] < total:
            checkpoint("rolling")
            print(f"stopping at update {progress['update']}: session time limit")
            if sync is not None:
                sync_wait(sync)
            return 0

    #final dev point (end of stage 3) if it was not logged yet
    if not any(r["update"] == total for r in read_jsonl(dev_path)):
        m= dev_eval(llm, connector, tokenizer, segments, cfg, dev, feats, device)
        append_jsonl(dev_path, {"update": total, "stage": cfg["stages"][-1]["name"], **m})
        print("dev", total, {n: round(v, 4) for n, v in m.items()})

    summary= run_metadata(cfg, device)
    summary.update({
        "run_name": args.run_name,
        "progress": dict(progress, train_seconds=progress["train_seconds"] + time.time() - t_start),
        "peak_mem_bytes_last_session": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "dtype": "float16 backbone, fp32 trainable" if use_fp16 else "float32"
    })
    write_json(os.path.join(results_dir, "train_summary.json"), summary)

    if sync is not None:
        sync_put(sync, "upload_folder", results_dir, results_prefix)
        if not sync_wait(sync):
            print("WARNING not everything was synced:", sync["failures"])

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        type=str,
        default="main"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits/task2"
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
        "--sync_repo",
        type=str,
        default=None
    )
    parser.add_argument(
        "--max_hours",
        type=float,
        default=None
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
