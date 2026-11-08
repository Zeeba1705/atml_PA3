import argparse
import hashlib
import json
import os
import re
import time
import unicodedata

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from common.io import git_commit, load_yaml, write_json

BASE_CONFIG= "configs/task1/base.yaml"
END_OF_TURN= "<|im_end|>"
EXCLUSION_REASONS= ["empty", "gsm8k_overlap", "too_long"]


def normalize_question(text):
    #NFKC, lowercase, drop punctuation, collapse whitespace (docs/decisions.md)
    text= unicodedata.normalize("NFKC", text).lower()
    text= re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def build_example(tokenizer, question, response):
    #non-thinking generation prompt ends with an empty <think>\n\n</think>\n\n block, which counts as prompt
    prompt= tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    #tokenize prompt and response separately: ~2% of responses start with "\n", which would merge with the
    #prompt's final "\n\n" into one "\n\n\n" token. separate ids keep the exact generation-time prompt tokens
    prompt_ids= tokenizer(prompt, add_special_tokens=False)["input_ids"]

    #template would add "\n" after <|im_end|>, we stop at the end-of-turn token
    response_ids= tokenizer(response + END_OF_TURN, add_special_tokens=False)["input_ids"]

    full_ids= prompt_ids + response_ids
    input_ids= torch.tensor(full_ids, dtype=torch.long)

    labels= input_ids.clone()
    labels[:len(prompt_ids)]= -100

    return {"input_ids": input_ids, "labels": labels, "length": len(full_ids)}


def collate(examples, pad_id):
    #dynamic padding: pad to the longest sequence in this microbatch, on the right
    max_len= max(len(ex["input_ids"]) for ex in examples)

    input_ids= torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels= torch.full((len(examples), max_len), -100, dtype=torch.long)
    attention_mask= torch.zeros((len(examples), max_len), dtype=torch.long)

    for i, ex in enumerate(examples):
        n= len(ex["input_ids"])

        input_ids[i, :n]= ex["input_ids"]
        labels[i, :n]= ex["labels"]
        attention_mask[i, :n]= 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def select_examples(records, tokenizer, gsm8k_questions, n_examples, max_length):
    #walk the source in order, first failing check is the reason, stop once n_examples are kept
    gsm8k_norm= set(normalize_question(q) for q in gsm8k_questions)

    kept_indices= []
    examples= []
    exclusions= {reason: [] for reason in EXCLUSION_REASONS}

    for idx, rec in enumerate(records):
        if len(kept_indices) == n_examples:
            break

        query= rec["query"]
        response= rec["response"]
        original= rec.get("original_question") or ""

        if not query.strip() or not response.strip():
            exclusions["empty"].append(idx)
            continue

        if normalize_question(query) in gsm8k_norm or normalize_question(original) in gsm8k_norm:
            exclusions["gsm8k_overlap"].append(idx)
            continue

        ex= build_example(tokenizer, query, response)

        #length exactly max_length is kept, only longer ones go
        if ex["length"] > max_length:
            exclusions["too_long"].append(idx)
            continue

        kept_indices.append(idx)
        examples.append(ex)

    if len(kept_indices) < n_examples:
        raise ValueError(f"only {len(kept_indices)} eligible examples, need {n_examples}")

    return kept_indices, examples, exclusions


def fingerprint(examples):
    #hash of every token and label, so a re-tokenized file on kaggle must match the committed selection exactly
    h= hashlib.sha256()

    for ex in examples:
        h.update(ex["input_ids"].numpy().tobytes())
        h.update(ex["labels"].numpy().tobytes())

    return h.hexdigest()


def load_metamath(cfg):
    path= hf_hub_download(
        cfg["dataset_name"],
        cfg["dataset_file"],
        repo_type="dataset",
        revision=cfg["dataset_revision"]
    )

    with open(path) as f:
        return json.load(f)


def load_gsm8k_test(cfg):
    path= hf_hub_download(
        cfg["gsm8k_name"],
        cfg["gsm8k_test_file"],
        repo_type="dataset",
        revision=cfg["gsm8k_revision"]
    )

    return [row["question"] for row in pq.read_table(path).to_pylist()]


def main(args):
    cfg= load_yaml(BASE_CONFIG)

    n_examples= cfg["smoke"]["n_examples"] if args.smoke else cfg["n_examples"]

    #smoke writes next to the real files, never over them
    selection_path= cfg["selection_path"]
    tokenized_path= os.path.join(args.output_dir, "task1", "data", "train_tokenized.pt")

    if args.smoke:
        selection_path= os.path.join(args.output_dir, "task1", "data", "smoke_selection.json")
        tokenized_path= os.path.join(args.output_dir, "task1", "data", "smoke_tokenized.pt")

    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    os.makedirs(os.path.dirname(tokenized_path), exist_ok=True)

    t0= time.time()

    tokenizer= AutoTokenizer.from_pretrained(cfg["model_name"])
    records= load_metamath(cfg)
    gsm8k_questions= load_gsm8k_test(cfg)

    print("metamath records:", len(records))
    print("gsm8k test questions:", len(gsm8k_questions))

    kept_indices, examples, exclusions= select_examples(
        records,
        tokenizer,
        gsm8k_questions,
        n_examples,
        cfg["max_length"]
    )

    lengths= [ex["length"] for ex in examples]
    n_supervised= sum(int((ex["labels"] != -100).sum()) for ex in examples)
    digest= fingerprint(examples)

    print("kept:", len(kept_indices), "last kept source index:", kept_indices[-1])
    for reason in EXCLUSION_REASONS:
        print(f"excluded {reason}:", len(exclusions[reason]))
    print("max length:", max(lengths), "mean length:", round(sum(lengths) / len(lengths), 1))

    selection= {
        "dataset": cfg["dataset_name"],
        "dataset_file": cfg["dataset_file"],
        "dataset_revision": cfg["dataset_revision"],
        "gsm8k": cfg["gsm8k_name"],
        "gsm8k_revision": cfg["gsm8k_revision"],
        "tokenizer": cfg["model_name"],
        "normalization": "NFKC, lowercase, punctuation -> space, collapse whitespace; exact match on query or original_question",
        "max_length": cfg["max_length"],
        "n_kept": len(kept_indices),
        "n_scanned": kept_indices[-1] + 1,
        "exclusion_counts": {reason: len(exclusions[reason]) for reason in EXCLUSION_REASONS},
        "excluded_indices": exclusions,
        "total_tokens": sum(lengths),
        "supervised_tokens": n_supervised,
        "max_tokens": max(lengths),
        "fingerprint": digest,
        "git_commit": git_commit(),
        "kept_indices": kept_indices
    }
    write_json(selection_path, selection)

    torch.save(
        {
            "examples": examples,
            "pad_id": tokenizer.pad_token_id,
            "kept_indices": kept_indices,
            "fingerprint": digest
        },
        tokenized_path
    )

    print("saved", selection_path)
    print("saved", tokenized_path)
    print("data prep time:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kaggle/working/outputs" if os.path.exists("/kaggle") else "outputs"
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    main(parser.parse_args())
