import argparse
import importlib.metadata
import json
import os
import time

import lm_eval
import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from common.io import load_config, load_yaml, run_metadata, write_json
from common.sync import make_api, pull_file, start_sync, sync_put, sync_wait
from task1.data import load_gsm8k_test
from task1.merge import merge_adapters
from task1.methods.lora import LoRALinear, add_lora
from task1.train import FULL_METHODS, default_output_dir

BASE_CONFIG= "configs/task1/base.yaml"
EVAL_CONFIG= "configs/task1/eval.yaml"
LORA_CONFIG= "configs/task1/lora.yaml"

#what lm-eval's flexible-extract filter returns when it finds no number
INVALID= "[invalid]"


def eval_root(results_dir, smoke):
    #smoke writes next to the real results, never over them
    return os.path.join(results_dir, "task1", "eval_smoke" if smoke else "eval")


def bench_paths(cond_dir, benchmark):
    #metrics json is written last, so it marks the benchmark as complete
    return {
        "metrics": os.path.join(cond_dir, f"{benchmark}.json"),
        "samples": os.path.join(cond_dir, f"{benchmark}_samples.jsonl"),
        "table": os.path.join(cond_dir, f"{benchmark}_table.jsonl")
    }


def sync_branch(condition, smoke):
    return f"task1-eval{'_smoke' if smoke else ''}-{condition}"


def check_revisions(api, cfg):
    #lm-eval loads each dataset's main branch: refuse to run if it moved since the revision we pinned
    for name, b in cfg["benchmarks"].items():
        sha= api.dataset_info(b["dataset"]).sha
        assert sha == b["revision"], f"{name}: {b['dataset']} is now at {sha}, expected {b['revision']}"


def pull_results(api, sync_repo, branch, cond_dir, prefix, staging):
    #results from earlier sessions, so finished benchmarks are skipped
    try:
        files= api.list_repo_files(sync_repo, revision=branch)
    except Exception:
        return

    for path in files:
        if not path.startswith(prefix + "/"):
            continue

        local= os.path.join(cond_dir, os.path.relpath(path, prefix))
        if not os.path.exists(local):
            pull_file(api, sync_repo, branch, path, local, staging)


def pull_export(api, args, spec, staging):
    #fp16 model (full ft) or adapter factors, as exported by task1.train and synced to the run's branch
    rel= os.path.join("task1", spec["run"], "export", f"update_{spec['update']:05d}")
    local= os.path.join(args.output_dir, rel)

    if os.path.exists(os.path.join(local, "export_info.json")):
        return local

    assert api is not None, f"{local} is missing and there is no --sync_repo to pull it from"

    branch= f"task1-{spec['run']}"
    files= [f for f in api.list_repo_files(args.sync_repo, revision=branch) if f.startswith(f"outputs/{rel}/")]
    assert files, f"no export at outputs/{rel} on {args.sync_repo}@{branch}"

    #export_info.json last: it marks the local copy as complete
    files.sort(key=lambda f: f.endswith("export_info.json"))
    for path in files:
        ok= pull_file(api, args.sync_repo, branch, path, os.path.join(args.output_dir, os.path.relpath(path, "outputs")), staging)
        assert ok, f"download failed: {path}"

    return local


def smoke_model(cfg, tokenizer):
    #tiny random qwen3 with the real vocab, so lm-eval's real prompts tokenize
    config= Qwen3Config(**cfg["smoke_model"], vocab_size=len(tokenizer), tie_word_embeddings=True)

    torch.manual_seed(cfg["seed"])
    return Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa", dtype=torch.float32)


def load_adapters(model, path):
    tensors= load_file(path)

    layers= [(name, m) for name, m in model.named_modules() if isinstance(m, LoRALinear)]
    assert 2 * len(layers) == len(tensors), f"{len(tensors)} adapter tensors for {len(layers)} layers"

    with torch.no_grad():
        for name, m in layers:
            m.lora_A.copy_(tensors[name + ".lora_A"])
            m.lora_B.copy_(tensors[name + ".lora_B"])


def merge_check(model, tokenizer, questions, enable_thinking, device):
    #logits on the same prompts before and after folding the adapters into the fp16 weights
    prompts= [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        for q in questions
    ]
    batch= tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    mask= batch["attention_mask"].bool()

    with torch.no_grad():
        before= model(**batch).logits.float()[mask]

        merge_adapters(model)

        after= model(**batch).logits.float()[mask]

    diff= (before - after).abs()

    return {
        "n_prompts": len(prompts),
        "n_positions": int(mask.sum()),
        "max_abs_logit_diff": diff.max().item(),
        "mean_abs_logit_diff": diff.mean().item(),
        "max_abs_logit": before.abs().max().item()
    }


def load_condition(cfg, condition, args, api, tokenizer, device, staging):
    #the model lm-eval scores, in eval mode (no dropout), plus a record of what it was built from
    spec= cfg["conditions"][condition]
    dtype= torch.float32 if cfg["smoke"] else torch.float16
    info= {"condition": condition, "dtype": str(dtype)}
    has_adapters= False

    if cfg["smoke"]:
        model= smoke_model(cfg, tokenizer)
        info["source"]= "smoke random init"

        if spec is not None:
            add_lora(model, load_yaml(LORA_CONFIG)["lora_targets"], spec["rank"], spec["alpha"], 0.0, cfg["seed"])

            #random B so the merge actually changes the weights
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, LoRALinear):
                        m.lora_B.normal_(std=0.02)

            has_adapters= True
    elif spec is None:
        model= AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=dtype, attn_implementation="sdpa")
        info["source"]= cfg["model_name"]
    else:
        export= pull_export(api, args, spec, staging)
        with open(os.path.join(export, "export_info.json")) as f:
            export_info= json.load(f)

        info["source"]= export
        info["export_info"]= export_info

        if export_info["method"] in FULL_METHODS:
            model= AutoModelForCausalLM.from_pretrained(export, dtype=dtype, attn_implementation="sdpa")
        else:
            #fresh backbone copy + the exported factors (projected ones for oplora, so a plain LoRALinear is exact)
            model= AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=dtype, attn_implementation="sdpa")
            add_lora(model, export_info["targets"], export_info["rank"], export_info["alpha"], 0.0, cfg["seed"])
            load_adapters(model, os.path.join(export, "adapters.safetensors"))
            has_adapters= True

    model.to(device)
    model.eval()

    if has_adapters:
        questions= load_gsm8k_test(cfg)[:cfg["merge_check_questions"]]
        info["merge_check"]= merge_check(model, tokenizer, questions, cfg["enable_thinking"], device)
        print("merge check:", info["merge_check"])

    assert not any(isinstance(m, LoRALinear) for m in model.modules())

    return model, info


def record_generation_lengths(model, tokenizer, lengths):
    #lm-eval only returns decoded text, and re-tokenizing that can't tell whether generation hit the cap
    #(a run of newline tokens re-tokenizes to a handful). so wrap generate and keep, per row, the number of
    #new tokens before the first eos/pad, keyed by the prompt text. lm-eval calls model.generate(input_ids=...,
    #attention_mask=..., ...); undo with `del model.generate`
    generate= model.generate

    eos= model.generation_config.eos_token_id
    eos= [] if eos is None else eos if isinstance(eos, list) else [eos]
    stop_ids= set(eos) | {tokenizer.pad_token_id}

    def wrapped(input_ids, attention_mask, **kwargs):
        out= generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        width= input_ids.shape[1]

        for row, mask, full in zip(input_ids, attention_mask, out):
            prompt= tokenizer.decode(row[mask.bool()], skip_special_tokens=False)
            new= full[width:].tolist()
            lengths[prompt]= next((i for i, t in enumerate(new) if t in stop_ids), len(new))

        return out

    model.generate= wrapped


def gsm8k_table(samples, lengths, max_gen_toks):
    #one row per question, from lm-eval's flexible-extract samples. lengths: prompt -> new tokens generated
    rows= []

    for s in samples:
        if s["filter"] != "flexible-extract":
            continue

        generation= s["resps"][0][0]
        extracted= s["filtered_resps"][0]
        n_tokens= lengths[s["arguments"][0][0]]

        rows.append({
            "id": s["doc_id"],
            "question": s["doc"]["question"],
            "target": s["target"],
            "generation": generation,
            "extracted": extracted,
            "correct": bool(s["exact_match"]),
            "n_tokens": n_tokens,
            "truncated": n_tokens >= max_gen_toks,
            #unextractable answers are scored wrong by exact match
            "extract_failed": extracted == INVALID
        })

    return sorted(rows, key=lambda r: r["id"])


def metric_value(res, name):
    #lm-eval keys look like "acc_norm,none" with the stderr at "acc_norm_stderr,none"
    metric, filt= name.split(",")

    return {
        "name": name,
        "value": res[name],
        "stderr": res.get(f"{metric}_stderr,{filt}")
    }


def plain(obj):
    #lm-eval's task config carries the model and tokenizer objects in its metadata: keep only their class names
    return json.loads(json.dumps(obj, default=lambda o: type(o).__name__))


def write_jsonl(path, rows):
    tmp= path + ".tmp"

    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")

    os.replace(tmp, path)


def run_benchmark(cfg, benchmark, model, tokenizer):
    b= cfg["benchmarks"][benchmark]
    is_gsm8k= benchmark == "gsm8k"

    batch_size= b["batch_size"]
    max_gen_toks= b.get("max_gen_toks")
    if cfg["smoke"]:
        batch_size= cfg["gsm8k_batch_size"] if is_gsm8k else 2
        max_gen_toks= cfg["max_gen_toks"] if is_gsm8k else None

    lengths= {}
    if is_gsm8k:
        record_generation_lengths(model, tokenizer, lengths)

    out= lm_eval.simple_evaluate(
        model="hf",
        model_args={
            "pretrained": model,
            "tokenizer": tokenizer,
            "enable_thinking": cfg["enable_thinking"],
            "max_length": cfg["max_length"]
        },
        tasks=[b["task"]],
        num_fewshot=0,
        batch_size=batch_size,
        limit=cfg.get("limit"),
        log_samples=True,
        apply_chat_template=b["chat_template"],
        gen_kwargs={"max_gen_toks": max_gen_toks} if is_gsm8k else None,
        random_seed=cfg["seed"],
        numpy_random_seed=cfg["seed"],
        torch_random_seed=cfg["seed"],
        fewshot_random_seed=cfg["seed"]
    )

    if is_gsm8k:
        del model.generate

    res= out["results"][b["task"]]

    metrics= {
        "benchmark": benchmark,
        "task": b["task"],
        "main": metric_value(res, b["metric"]),
        "n_samples": out.get("n-samples", {}).get(b["task"]),
        "batch_size": batch_size,
        "chat_template": b["chat_template"],
        "max_gen_toks": max_gen_toks,
        "dataset": b["dataset"],
        "dataset_revision": b["revision"],
        "lm_eval_config": plain(out.get("configs", {}).get(b["task"]))
    }
    if "raw_metric" in b:
        metrics["raw"]= metric_value(res, b["raw_metric"])

    samples= out["samples"][b["task"]]
    table= gsm8k_table(samples, lengths, max_gen_toks) if is_gsm8k else None

    if table is not None:
        metrics["n_truncated"]= sum(r["truncated"] for r in table)
        metrics["n_extract_failed"]= sum(r["extract_failed"] for r in table)

    return metrics, samples, table


def main(args):
    cfg= load_config(BASE_CONFIG, EVAL_CONFIG, args.smoke)

    device= torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)

    conditions= [args.condition] if args.condition else list(cfg["conditions"])
    for condition in conditions:
        assert condition in cfg["conditions"], f"unknown condition {condition}"

    api= make_api() if args.sync_repo else None
    if not args.smoke:
        check_revisions(api or HfApi(), cfg)

    tokenizer= AutoTokenizer.from_pretrained(cfg["model_name"])
    root= eval_root(args.results_dir, args.smoke)
    staging= os.path.join(args.output_dir, "task1", "eval_staging")
    lm_eval_version= importlib.metadata.version("lm_eval")

    for condition in conditions:
        cond_dir= os.path.join(root, condition)
        os.makedirs(cond_dir, exist_ok=True)

        branch= sync_branch(condition, args.smoke)
        prefix= os.path.relpath(cond_dir, args.results_dir)
        sync= start_sync(args.sync_repo, branch, api=api) if args.sync_repo else None
        if sync is not None:
            pull_results(api, args.sync_repo, branch, cond_dir, f"results/{prefix}", staging)

        todo= [b for b in cfg["benchmarks"] if not os.path.exists(bench_paths(cond_dir, b)["metrics"])]
        print(f"=== {condition}: {len(cfg['benchmarks']) - len(todo)}/{len(cfg['benchmarks'])} benchmarks done ===")
        if not todo:
            continue

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0= time.time()
        model, info= load_condition(cfg, condition, args, api, tokenizer, device, staging)
        info["load_seconds"]= time.time() - t0

        write_json(os.path.join(cond_dir, "model_info.json"), info)

        for benchmark in todo:
            t1= time.time()
            metrics, samples, table= run_benchmark(cfg, benchmark, model, tokenizer)

            paths= bench_paths(cond_dir, benchmark)
            write_jsonl(paths["samples"], samples)
            if table is not None:
                write_jsonl(paths["table"], table)

            metrics.update(run_metadata(cfg, device))
            metrics["condition"]= condition
            metrics["model"]= info
            metrics["lm_eval"]= lm_eval_version
            metrics["seconds"]= time.time() - t1
            if device.type == "cuda":
                metrics["peak_mem_bytes"]= torch.cuda.max_memory_allocated(device)

            write_json(paths["metrics"], metrics)
            print(f"{condition} {benchmark}: {metrics['main']['name']}={metrics['main']['value']:.4f} ({metrics['seconds']:.0f}s)")

            #each finished benchmark is synced right away, so a dead session loses at most the current one
            sync_put(sync, "upload_folder", cond_dir, f"results/{prefix}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if sync is not None and not sync_wait(sync):
            print("WARNING not everything was synced:", sync["failures"])

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        type=str,
        default=None
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )
    parser.add_argument(
        "--device",
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
        default="results"
    )
    parser.add_argument(
        "--sync_repo",
        type=str,
        default=None
    )

    raise SystemExit(main(parser.parse_args()))
