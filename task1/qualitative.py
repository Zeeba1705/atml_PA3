import argparse
import csv
import hashlib
import json
import os
import time

import torch
from transformers import AutoTokenizer

from common.io import load_config, load_yaml, read_jsonl, run_metadata, write_json
from common.sync import make_api, pull_file, start_sync, sync_put, sync_wait
from task1.evaluate import BASE_CONFIG, EVAL_CONFIG, load_condition, write_jsonl
from task1.train import default_output_dir

QUAL_CONFIG= "configs/task1/qualitative.yaml"
#written last, after every condition: marks step 7 as complete
GROUPED_FILE= "responses_by_question.json"
SHEET_FILE= "annotation_sheet.csv"
IDS_FILE= "annotation_prompt_ids.json"


def qual_root(results_dir, smoke):
    #smoke writes next to the real results, never over them (and never over the real annotation sheet)
    return os.path.join(results_dir, "task1", "qualitative_smoke" if smoke else "qualitative")


def qual_branch(smoke):
    return "task1-qualitative_smoke" if smoke else "task1-qualitative"


def load_qual_config(smoke):
    #base + eval (conditions, model loading, merge check) + the step 7 settings on top
    cfg= load_config(BASE_CONFIG, EVAL_CONFIG, smoke)

    qual= load_yaml(QUAL_CONFIG)
    qual_smoke= qual.pop("smoke")
    cfg.update(qual)
    if smoke:
        cfg.update(qual_smoke)

    return cfg


def load_prompts(cfg):
    #every entry's question text and optional system prompt; sampling counts, judges etc. are ignored (handout:
    #one greedy response per prompt, annotated by hand)
    with open(cfg["questions_path"], "rb") as f:
        raw= f.read()

    digest= hashlib.sha256(raw).hexdigest()
    assert digest == cfg["questions_sha256"], f"{cfg['questions_path']} differs from the pinned file ({digest})"

    prompts= []
    for entry in load_yaml(cfg["questions_path"]):
        assert len(entry["paraphrases"]) == 1, f"{entry['id']}: expected one paraphrase"

        fmt, question= "plain", entry["id"]
        for name, suffix in cfg["format_suffixes"].items():
            if entry["id"].endswith(suffix):
                fmt, question= name, entry["id"][:-len(suffix)]

        prompts.append({
            "prompt_id": entry["id"],
            "question_id": question,
            "format": fmt,
            "system": entry.get("system"),
            "prompt": entry["paraphrases"][0]
        })

    #handout: 24 prompts = 8 questions x plain, json, template
    formats= ["plain"] + list(cfg["format_suffixes"])
    questions= [p["question_id"] for p in prompts if p["format"] == "plain"]
    assert len(questions) == cfg["n_questions"], f"{len(questions)} plain questions"
    assert sorted((p["question_id"], p["format"]) for p in prompts) == sorted((q, f) for q in questions for f in formats)

    return prompts, digest


def annotation_prompt_ids(prompts):
    #handout: the first and last underlying questions in the file's plain-question order, all three formats
    plain= [p["question_id"] for p in prompts if p["format"] == "plain"]
    chosen= [plain[0], plain[-1]]

    return [p["prompt_id"] for q in chosen for p in prompts if p["question_id"] == q]


def chat_text(tokenizer, prompt, enable_thinking):
    messages= []
    if prompt["system"] is not None:
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({"role": "user", "content": prompt["prompt"]})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def generate_all(model, tokenizer, prompts, cfg, device):
    #greedy, left padding, fixed batches. truncated = no end-of-turn token within max_new_tokens
    eos= model.generation_config.eos_token_id
    eos= [] if eos is None else eos if isinstance(eos, list) else [eos]
    stop_ids= set(eos) | {tokenizer.pad_token_id}

    rows= []
    for start in range(0, len(prompts), cfg["batch_size"]):
        batch= prompts[start:start + cfg["batch_size"]]
        texts= [chat_text(tokenizer, p, cfg["enable_thinking"]) for p in batch]
        enc= tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            out= model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=cfg["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                temperature=None,
                top_p=None,
                top_k=None
            )

        width= enc["input_ids"].shape[1]
        for p, text, full in zip(batch, texts, out):
            new= full[width:].tolist()
            stop= next((i for i, t in enumerate(new) if t in stop_ids), None)
            n_tokens= len(new) if stop is None else stop

            rows.append({
                **p,
                "chat_text": text,
                "response": tokenizer.decode(new[:n_tokens], skip_special_tokens=True),
                "n_tokens": n_tokens,
                "truncated": stop is None
            })

    return rows


def condition_path(root, condition):
    return os.path.join(root, f"{condition}.jsonl")


def pull_condition(api, args, cfg, root, condition, staging):
    #a condition finished in an earlier session is reused, not regenerated
    path= condition_path(root, condition)
    if os.path.exists(path) or api is None:
        return os.path.exists(path)

    rel= "results/" + os.path.relpath(path, args.results_dir)
    return pull_file(api, args.sync_repo, qual_branch(cfg["smoke"]), rel, path, staging)


def build_outputs(cfg, root, prompts, conditions):
    #all responses grouped by question, then format, then condition; plus the empty annotation sheet
    by_condition= {c: {r["prompt_id"]: r for r in read_jsonl(condition_path(root, c))} for c in conditions}
    for c, rows in by_condition.items():
        assert len(rows) == len(prompts), f"{c}: {len(rows)} of {len(prompts)} responses"

    formats= ["plain"] + list(cfg["format_suffixes"])
    questions= [p["question_id"] for p in prompts if p["format"] == "plain"]
    ids= {(p["question_id"], p["format"]): p["prompt_id"] for p in prompts}

    grouped= []
    for q in questions:
        entry= {"question_id": q, "formats": {}}
        for fmt in formats:
            pid= ids[q, fmt]
            first= by_condition[conditions[0]][pid]
            entry["formats"][fmt]= {
                "prompt_id": pid,
                "system": first["system"],
                "prompt": first["prompt"],
                "responses": {c: {k: by_condition[c][pid][k] for k in ["response", "n_tokens", "truncated"]} for c in conditions}
            }
        grouped.append(entry)

    #readable copy of the same thing
    lines= [f"# Step 7 responses ({len(prompts)} prompts x {len(conditions)} conditions, greedy, {cfg['max_new_tokens']} new tokens)", ""]
    for entry in grouped:
        lines.append(f"## {entry['question_id']}")
        for fmt, block in entry["formats"].items():
            lines+= ["", f"### {fmt} (`{block['prompt_id']}`)", ""]
            if block["system"] is not None:
                lines.append(f"**system:** {block['system']}")
                lines.append("")
            lines+= [f"**prompt:** {block['prompt']}", ""]
            for c, r in block["responses"].items():
                flag= ", TRUNCATED" if r["truncated"] else ""
                lines+= [f"#### {c} ({r['n_tokens']} tokens{flag})", "", "```", r["response"], "```", ""]

    with open(os.path.join(root, "responses_by_question.md"), "w") as f:
        f.write("\n".join(lines))

    chosen= annotation_prompt_ids(prompts)
    write_json(os.path.join(root, IDS_FILE), {"rule": "first and last plain-order questions, all formats", "prompt_ids": chosen})

    #never overwrite: once it exists it may hold hand-written annotations
    sheet= os.path.join(os.path.dirname(root), SHEET_FILE) if not cfg["smoke"] else os.path.join(root, SHEET_FILE)
    if os.path.exists(sheet):
        print("keeping existing", sheet, "(not overwritten)")
    else:
        with open(sheet, "w", newline="") as f:
            writer= csv.writer(f)
            writer.writerow(["prompt_id", "question_id", "format", "condition", "n_tokens", "truncated", "response"] + cfg["annotation_columns"])
            for pid in chosen:
                for c in conditions:
                    r= by_condition[c][pid]
                    writer.writerow([pid, r["question_id"], r["format"], c, r["n_tokens"], "yes" if r["truncated"] else "no", r["response"]] + [""] * len(cfg["annotation_columns"]))
        print("saved", sheet, f"({len(chosen) * len(conditions)} rows, annotation columns empty)")

    #last: marks the step as complete
    write_json(os.path.join(root, GROUPED_FILE), grouped)
    print("saved", os.path.join(root, GROUPED_FILE))


def main(args):
    cfg= load_qual_config(args.smoke)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device= torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)

    prompts, digest= load_prompts(cfg)
    conditions= list(cfg["conditions"])

    api= make_api() if args.sync_repo else None
    root= qual_root(args.results_dir, args.smoke)
    os.makedirs(root, exist_ok=True)
    staging= os.path.join(args.output_dir, "task1", "qual_staging")
    sync= start_sync(args.sync_repo, qual_branch(args.smoke), api=api) if args.sync_repo else None
    prefix= "results/" + os.path.relpath(root, args.results_dir)

    tokenizer= AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.padding_side= "left"

    for condition in conditions:
        if pull_condition(api, args, cfg, root, condition, staging):
            print(f"=== {condition}: already generated ===")
            continue

        if args.collect:
            print(f"=== {condition}: missing, not generating (--collect) ===")
            return 1

        t0= time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model, info= load_condition(cfg, condition, args, api, tokenizer, device, staging)
        rows= generate_all(model, tokenizer, prompts, cfg, device)

        meta= run_metadata(cfg, device)
        meta.update({"condition": condition, "model": info, "questions_sha256": digest, "seconds": time.time() - t0})
        if device.type == "cuda":
            meta["peak_mem_bytes"]= torch.cuda.max_memory_allocated(device)
        write_json(os.path.join(root, f"{condition}_info.json"), meta)

        #the responses file last: it marks the condition as done
        write_jsonl(condition_path(root, condition), [{"condition": condition, **r} for r in rows])
        n_trunc= sum(r["truncated"] for r in rows)
        print(f"=== {condition}: {len(rows)} responses, {n_trunc} truncated ({meta['seconds']:.0f}s) ===")

        if sync is not None:
            sync_put(sync, "upload_folder", root, prefix)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    build_outputs(cfg, root, prompts, conditions)

    if sync is not None and not args.collect:
        sync_put(sync, "upload_folder", root, prefix)
        if not sync_wait(sync):
            print("WARNING not everything was synced:", sync["failures"])
            return 1

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true"
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="only pull finished conditions and rebuild the grouped files / annotation sheet (no generation)"
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
