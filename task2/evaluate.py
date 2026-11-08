import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from common.bootstrap import bootstrap_mean_ci, paired_difference_ci, resample_indices
from common.io import read_jsonl, run_metadata, write_json
from common.sync import make_api, pull_file, start_sync, sync_put, sync_wait
from task1.methods.lora import add_lora
from task2.connector import Connector
from task2.features import default_output_dir, load_features
from task2.train import build_llm, feats_for, load_config, load_trainable, make_batch, nll_items
from task2.vlm import input_embeds, prompt_segments
from task2.vqa import exact_match, normalize_answer, soft_accuracy

SUITES= {"vqa": "vqa_test", "grounding": "grounding_test"}
CONDITIONS= ["correct", "complementary", "mismatched", "no_image"]


def mismatch_map(pairs, seed):
    #pair k borrows the images of pair perm[k] (same item slot), perm a derangement: never its own pair's images
    rng= np.random.default_rng(seed)
    while True:
        perm= rng.permutation(len(pairs))
        if not np.any(perm == np.arange(len(pairs))):
            break

    return {pairs[k]["pair_id"]: pairs[int(perm[k])]["pair_id"] for k in range(len(pairs))}


def condition_items(pairs, condition, mismatch, cfg):
    #both questions of every pair under one image condition; references and question text never change
    by_id= {p["pair_id"]: p for p in pairs}
    items= []
    for p in pairs:
        for j, it in enumerate(p["items"]):
            partner= p["items"][1 - j]
            other= by_id[mismatch[p["pair_id"]]]["items"][j] if condition == "mismatched" else None

            image= {
                "correct": (it["split"], it["image_id"]),
                "complementary": (partner["split"], partner["image_id"]),
                "mismatched": None if other is None else (other["split"], other["image_id"]),
                "no_image": None
            }[condition]

            items.append({
                "pair_id": p["pair_id"],
                "slot": j,
                "type": p["type"],
                "image": image,
                "question": it["question"],
                "target": it["target"],
                "answers": it.get("answers"),
                "partner_target": partner["target"],
                "max_target": cfg["max_answer_tokens"]
            })

    return items


def generate(llm, connector, tokenizer, segments, cfg, items, feats, device):
    #greedy, thinking off, 16 new tokens, left-padded batches of 8; seconds per answer = batch time / batch size
    out= []
    for k in range(0, len(items), cfg["eval_batch_size"]):
        chunk= items[k:k + cfg["eval_batch_size"]]
        prompts= [dict(it, target=None) for it in chunk]
        batch= {n: t.to(device) for n, t in make_batch(prompts, tokenizer, segments, cfg, False, left_pad=True).items()}

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0= time.time()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            embeds= input_embeds(llm, connector, batch, feats_for(chunk, feats, device))
            gen= llm.generate(
                inputs_embeds=embeds,
                attention_mask=batch["attention_mask"],
                max_new_tokens=cfg["eval_max_new_tokens"],
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=segments["eot"]
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        per= (time.time() - t0) / len(chunk)

        #with inputs_embeds, generate returns only the new tokens
        for it, ids in zip(chunk, gen):
            out.append({"prediction": tokenizer.decode(ids, skip_special_tokens=True).strip(), "gen_seconds": per})

    return out


def score_item(suite, it, pred):
    #grounding: normalized exact match; vqa: soft accuracy over the ten human answers
    acc= exact_match(pred, it["target"]) if suite == "grounding" else soft_accuracy(pred, it["answers"])

    return {
        "accuracy": acc,
        "exact": exact_match(pred, it["target"]),
        "follows_partner": exact_match(pred, it["partner_target"]),
        "empty": float(normalize_answer(pred) == "")
    }


def pair_rows(rows):
    #rows of one condition grouped into pairs (slot 0, slot 1), in file order
    pairs= {}
    for r in rows:
        pairs.setdefault(r["pair_id"], [None, None])[r["slot"]]= r

    return pairs


def suite_metrics(rows):
    pairs= pair_rows(rows)

    def summary(sub, sub_pairs):
        return {
            "n_questions": len(sub),
            "n_pairs": len(sub_pairs),
            "accuracy": float(np.mean([r["accuracy"] for r in sub])),
            "pair_accuracy": float(np.mean([a["exact"] * b["exact"] for a, b in sub_pairs])),
            "identical_answer_rate": float(np.mean([normalize_answer(a["prediction"]) == normalize_answer(b["prediction"]) for a, b in sub_pairs])),
            "empty_answer_rate": float(np.mean([r["empty"] for r in sub])),
            "follows_partner_rate": float(np.mean([r["follows_partner"] for r in sub])),
            "mean_reference_nll": float(np.mean([r["nll"] for r in sub])),
            "gen_seconds_per_answer": float(np.mean([r["gen_seconds"] for r in sub]))
        }

    out= {"all": summary(rows, list(pairs.values()))}
    for t in sorted({r["type"] for r in rows}):
        sub_pairs= [p for p in pairs.values() if p[0]["type"] == t]
        out[t]= summary([r for p in sub_pairs for r in p], sub_pairs)

    return out


def per_pair(rows, key):
    pairs= pair_rows(rows)

    return {pid: (a[key] + b[key]) / 2 for pid, (a, b) in pairs.items()}


def advantages(by_cond, cfg):
    #correct-image advantage over complementary swaps (and the other controls), per pair, with shared resamples
    out= {}
    pair_ids= sorted(per_pair(by_cond["correct"], "accuracy"))
    idx= resample_indices(len(pair_ids), cfg["bootstrap_resamples"], cfg["seed"])

    for key, sign in [("accuracy", 1), ("nll", -1)]:
        vals= {c: per_pair(by_cond[c], key) for c in by_cond}
        out[key]= {"correct": bootstrap_mean_ci([vals["correct"][p] for p in pair_ids], idx)}
        for c in [c for c in CONDITIONS if c != "correct"]:
            #accuracy: correct - control; nll: control - correct (positive = correct image helps either way)
            a= [vals["correct"][p] for p in pair_ids]
            b= [vals[c][p] for p in pair_ids]
            out[key][f"correct_minus_{c}" if sign > 0 else f"{c}_minus_correct"]= paired_difference_ci(a, b, idx) if sign > 0 else paired_difference_ci(b, a, idx)

    return out


def find_examples(by_cond):
    #fixed rules over pairs: right on both images; under the swap, follows the partner image's answer on both
    #questions; same answer for both images under the correct images
    correct= pair_rows(by_cond["correct"])
    swapped= pair_rows(by_cond["complementary"])

    def view(pid):
        a, b= correct[pid]
        sa, sb= swapped[pid]
        return {
            "pair_id": pid,
            "question": a["question"],
            "references": [a["target"], b["target"]],
            "pred_correct_images": [a["prediction"], b["prediction"]],
            "pred_swapped_images": [sa["prediction"], sb["prediction"]]
        }

    rules= {
        "both_right": [pid for pid, (a, b) in correct.items() if a["exact"] and b["exact"]],
        "follows_partner_when_swapped": [pid for pid, (a, b) in swapped.items() if a["follows_partner"] and b["follows_partner"]],
        "same_answer_both_images": [pid for pid, (a, b) in correct.items() if normalize_answer(a["prediction"]) == normalize_answer(b["prediction"])]
    }

    return {rule: [view(pid) for pid in sorted(pids)] for rule, pids in rules.items()}


def load_model(cfg, args, device, tokenizer):
    llm= build_llm(cfg, device)
    llm.config.use_cache= True
    connector= Connector(cfg["vision_dim"], cfg["connector_hidden"], llm.config.hidden_size).to(device)
    add_lora(llm, cfg["lora_targets"], cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"], cfg["seed"])
    llm.to(device)

    #final model = the completed stage 3 weights (pulled from the run's hub branch if not on disk)
    path= os.path.join(args.output_dir, "task2", args.run_name, "stage3", "weights.pt")
    if not os.path.exists(path):
        assert args.sync_repo, f"{path} missing and no --sync_repo"
        rel= f"outputs/task2/{args.run_name}/stage3/weights.pt"
        assert pull_file(make_api(), args.sync_repo, f"task2-{args.run_name}", rel, path, os.path.join(args.output_dir, "task2", "staging")), rel

    load_trainable(torch.load(path, map_location=device), connector, llm)
    llm.eval()
    connector.eval()

    return llm, connector, path


def main(args):
    cfg= load_config(args.smoke)

    device= torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    #a kaggle session without an accelerator silently fell back to cpu once (66 min for the first dev eval):
    #real runs need a gpu unless cpu is asked for explicitly
    assert args.smoke or device.type == "cuda" or args.device == "cpu", "no GPU found: enable the T4 accelerator (or pass --device cpu on purpose)"
    print("Using device:", device)

    eval_dir= os.path.join(args.results_dir, args.run_name, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    tokenizer= AutoTokenizer.from_pretrained(cfg["llm_name"])
    if args.smoke:
        cfg["smoke_vocab"]= len(tokenizer)
    segments= prompt_segments(tokenizer, cfg)

    feats= load_features(args.output_dir, args.smoke)
    llm, connector, weights= load_model(cfg, args, device, tokenizer)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metrics, adv, examples= {}, {}, {}
    for suite, name in SUITES.items():
        with open(os.path.join(args.split_dir, f"{name}.json")) as f:
            pairs= json.load(f)

        #fixed before any generation and reused for every comparison
        map_path= os.path.join(eval_dir, f"mismatch_{suite}.json")
        mismatch= mismatch_map(pairs, cfg["seed"])
        if os.path.exists(map_path):
            with open(map_path) as f:
                assert json.load(f) == mismatch, f"{map_path} differs from the seeded mapping"
        write_json(map_path, mismatch)

        by_cond= {}
        for condition in CONDITIONS:
            path= os.path.join(eval_dir, f"generations_{suite}_{condition}.jsonl")
            items= condition_items(pairs, condition, mismatch, cfg)

            rows= read_jsonl(path)
            if len(rows) != len(items):
                #reference nll (no termination token) and greedy answers, written once per condition
                nll= nll_items(llm, connector, tokenizer, segments, cfg, items, feats, device, False, cfg["eval_batch_size"])
                gens= generate(llm, connector, tokenizer, segments, cfg, items, feats, device)

                rows= []
                for it, n, g in zip(items, nll, gens):
                    row= {k: v for k, v in it.items() if k != "max_target"}
                    row.update({"condition": condition, "image": list(it["image"]) if it["image"] else None, "nll": n, **g})
                    row.update(score_item(suite, it, g["prediction"]))
                    rows.append(row)

                with open(path, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                print(f"{suite} {condition}: {len(rows)} answers")

            by_cond[condition]= rows

        metrics[suite]= {c: suite_metrics(by_cond[c]) for c in CONDITIONS}
        adv[suite]= advantages(by_cond, cfg)
        examples[suite]= find_examples(by_cond)
        print(suite, {c: round(metrics[suite][c]["all"]["accuracy"], 4) for c in CONDITIONS})

    info= run_metadata(cfg, device)
    info.update({"weights": weights, "peak_mem_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
    write_json(os.path.join(eval_dir, "metrics.json"), {"metrics": metrics, "advantages": adv, "info": info})
    write_json(os.path.join(eval_dir, "examples.json"), examples)

    if args.sync_repo:
        sync= start_sync(args.sync_repo, f"task2-{args.run_name}")
        sync_put(sync, "upload_folder", eval_dir, f"results/task2/{args.run_name}/eval")
        if not sync_wait(sync):
            print("WARNING not everything was synced:", sync["failures"])
            return 1

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
        "--device",
        type=str,
        default=None
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    raise SystemExit(main(parser.parse_args()))
