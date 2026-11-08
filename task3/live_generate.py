import argparse
import json
import os
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from common.io import load_yaml, read_jsonl, run_metadata, write_json
from common.sync import make_api, start_sync, sync_put, sync_wait
from task3.answers import NO_ANSWER, group_math, grade_math
from task3.load import FOLDERS
from task3.selectors import self_consistency

LIVE_CONFIG= "configs/task3/live.yaml"


def last_boxed(text):
    #README rule: last literal \boxed{, balanced braces, nested latex kept, whitespace stripped; absent/empty/unclosed -> None
    start= text.rfind("\\boxed{")
    if start < 0:
        return None

    i= start + len("\\boxed{")
    depth= 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth+= 1
        elif text[j] == "}":
            depth-= 1
            if depth == 0:
                ans= text[i:j].strip()
                return ans or None

    return None


def load_problems(cfg, split_dir):
    #the 50 live questions (first 50 math eval ids in split order) with their original problem text and gold answer;
    #saved once from the course data so the gpu job only needs the repo
    path= os.path.join(split_dir, "live_problems.json")
    with open(os.path.join(split_dir, "math_split.json")) as f:
        live= json.load(f)["live"]
    if os.path.exists(path):
        with open(path) as f:
            probs= json.load(f)
        assert [p["sample_id"] for p in probs] == live, f"{path} does not match the live split"
        return probs

    t= pq.read_table(os.path.join(cfg["data_dir"], FOLDERS["math"], "generations.parquet"), columns=["sample_id", "rollout_id", "question_text", "gold_answer"]).to_pylist()
    first= {r["sample_id"]: r for r in t if r["rollout_id"] == 0}

    probs= [{"sample_id": q, "problem": first[q]["question_text"], "gold": first[q]["gold_answer"]} for q in live]
    write_json(path, probs)

    return probs


def generate(model, tokenizer, messages, max_new_tokens, seed, cfg, device):
    #one independently seeded sampled call (batch size 1); returns text, token counts, truncation and time
    text= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc= tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
    t0= time.time()

    with torch.no_grad():
        out= model.generate(
            **enc,
            do_sample=True,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            repetition_penalty=cfg["repetition_penalty"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds= time.time() - t0

    new= out[0, enc["input_ids"].shape[1]:].tolist()
    eos= set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])
    finished= bool(new) and new[-1] in eos
    response= tokenizer.decode(new, skip_special_tokens=True)

    return {
        "prompt": text,
        "response": response,
        "seed": seed,
        "input_tokens": int(enc["input_ids"].shape[1]),
        "output_tokens": len(new),
        "truncated": not finished and len(new) >= max_new_tokens,
        "seconds": seconds,
        "extracted": last_boxed(response)
    }


def run_question(model, tokenizer, prob, qi, cfg, device):
    #single (1 x 3072), parallel (3 x 1024, self-consistency), refine (parallel #1 + one 2048 critique-and-revision)
    user= prob["problem"] + "\n\n" + cfg["instruction"]
    base= [{"role": "system", "content": cfg["system"]}, {"role": "user", "content": user}]
    seed= lambda k: cfg["seed"] * 1000 + qi * 10 + k

    calls= {"single": generate(model, tokenizer, base, cfg["single_tokens"], seed(0), cfg, device)}
    calls["parallel"]= [generate(model, tokenizer, base, cfg["parallel_tokens"], seed(1 + k), cfg, device) for k in range(cfg["n_parallel"])]

    #refinement reuses the first parallel response as its initial answer
    initial= calls["parallel"][0]
    review= base + [{"role": "assistant", "content": initial["response"]}, {"role": "user", "content": cfg["review_instruction"]}]
    calls["refine"]= generate(model, tokenizer, review, cfg["refine_tokens"], seed(9), cfg, device)

    return calls


def grade(prob, calls):
    #answers grouped with the same math-verify rule as the offline analysis; no gold in any selection
    keys, reps= group_math([calls["single"]["extracted"]])
    single= {"answer": keys[0], "correct": keys[0] != NO_ANSWER and keys[0] == grade_math(reps, prob["gold"])}

    keys, reps= group_math([c["extracted"] for c in calls["parallel"]])
    sc= self_consistency(keys)
    gold= grade_math(reps, prob["gold"])
    parallel= {"answer": sc, "correct": sc != NO_ANSWER and sc == gold, "initial_answer": keys[0], "initial_correct": keys[0] != NO_ANSWER and keys[0] == gold}

    keys, reps= group_math([calls["refine"]["extracted"]])
    refine= {"answer": keys[0], "correct": keys[0] != NO_ANSWER and keys[0] == grade_math(reps, prob["gold"])}

    return {"single": single, "parallel": parallel, "refine": refine}


def main(args):
    cfg= load_yaml(LIVE_CONFIG)
    cfg.update({k: v for k, v in load_yaml("configs/task3/base.yaml").items() if k not in cfg})

    device= torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    assert args.smoke or device.type == "cuda" or args.device == "cpu", "no GPU found: enable the accelerator (or pass --device cpu on purpose)"
    print("Using device:", device)

    problems= load_problems(cfg, args.split_dir)
    mine= [(qi, p) for qi, p in enumerate(problems) if qi % args.n_shards == args.shard]
    if args.smoke:
        mine= mine[:1]
        cfg.update(cfg["smoke"])

    out_dir= os.path.join(args.results_dir, "live_smoke" if args.smoke else "live")
    os.makedirs(out_dir, exist_ok=True)
    path= os.path.join(out_dir, f"shard{args.shard}of{args.n_shards}.jsonl")
    done= {r["sample_id"] for r in read_jsonl(path)}

    tokenizer= AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    if args.smoke:
        #tiny random qwen2 with the real tokenizer and chat template
        config= Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192)
        torch.manual_seed(cfg["seed"])
        model= Qwen2ForCausalLM(config).to(device).eval()
        model.generation_config.eos_token_id= [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
    else:
        model= AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.float16).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sync= start_sync(args.sync_repo, "task3-live") if args.sync_repo else None
    prefix= "results/" + os.path.relpath(out_dir, os.path.dirname(args.results_dir))

    for qi, prob in mine:
        if prob["sample_id"] in done:
            continue

        t0= time.time()
        calls= run_question(model, tokenizer, prob, qi, cfg, device)
        row= {
            "sample_id": prob["sample_id"],
            "question_index": qi,
            "gold": prob["gold"],
            "calls": calls,
            "grades": grade(prob, calls),
            "question_seconds": time.time() - t0,
            "peak_mem_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

        g= row["grades"]
        print(f"{qi} {prob['sample_id']}: single {g['single']['correct']} parallel {g['parallel']['correct']} refine {g['refine']['correct']} ({row['question_seconds']:.0f}s)", flush=True)
        if sync is not None:
            sync_put(sync, "upload_folder", out_dir, prefix)

    meta= run_metadata(cfg, device)
    meta.update({"model": cfg["model"], "revision": cfg["revision"], "dtype": "float16", "shard": args.shard, "n_shards": args.n_shards})
    write_json(os.path.join(out_dir, f"shard{args.shard}of{args.n_shards}_info.json"), meta)
    if sync is not None:
        sync_put(sync, "upload_folder", out_dir, prefix)
        if not sync_wait(sync):
            print("WARNING not everything was synced:", sync["failures"])
            return 1

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--shard",
        type=int,
        default=0
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=1
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits/task3"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/task3"
    )
    parser.add_argument(
        "--sync_repo",
        type=str,
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
