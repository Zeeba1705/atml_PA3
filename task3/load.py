import argparse
import json
import os

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common.io import load_yaml, write_json

#dataset name -> folder of ATML-PA3-Data (see its README.MD)
FOLDERS= {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
KEYS= ["dataset", "sample_id", "rollout_id"]
N_ROLLOUTS= 32


def read_sorted(path, columns):
    #one row per rollout, sorted by the join key so the three files line up row by row
    t= pq.read_table(path, columns=KEYS + columns)

    return t.take(pc.sort_indices(t, sort_keys=[("sample_id", "ascending"), ("rollout_id", "ascending")]))


def ragged(table, column):
    #list column -> (flat numpy values, row offsets); row i is values[off[i]:off[i + 1]]
    col= table.column(column).combine_chunks()

    return col.values.to_numpy(zero_copy_only=False), col.offsets.to_numpy()


def split_rows(values, offsets):
    return [values[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def load_dataset(data_dir, name):
    #per-rollout fields of one dataset, rows sorted by (sample_id, rollout_id), token lists checked for alignment
    folder= os.path.join(data_dir, FOLDERS[name])

    gen= read_sorted(os.path.join(folder, "generations.parquet"), ["question_text", "generated_token_ids", "n_generated_text_tokens", "hit_max_new_tokens", "gold_answer", "extracted_answer"])
    prob= read_sorted(os.path.join(folder, "probability_metrics.parquet"), ["logp", "kl_u_p"])
    tables= [gen, prob]

    if name == "mathvista":
        att= read_sorted(os.path.join(folder, "attention_metrics.parquet"), ["n_generated_text_tokens", "img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"])
        tables.append(att)

    #same join keys in the same order in every file
    for t in tables[1:]:
        for k in KEYS:
            assert t.column(k).equals(gen.column(k)), f"{name}: {k} differs between files"

    n_tok= gen.column("n_generated_text_tokens").to_numpy()
    _, tok_off= ragged(gen, "generated_token_ids")
    assert np.array_equal(np.diff(tok_off), n_tok), f"{name}: token ids vs n_generated_text_tokens"

    logp, off= ragged(prob, "logp")
    assert np.array_equal(off, tok_off), f"{name}: logp not aligned with generated tokens"
    kl, off= ragged(prob, "kl_u_p")
    assert np.array_equal(off, tok_off), f"{name}: kl_u_p not aligned with generated tokens"

    out= {
        "sample_id": gen.column("sample_id").to_pylist(),
        "rollout_id": gen.column("rollout_id").to_numpy(),
        "question_text": gen.column("question_text").to_pylist(),
        "gold": gen.column("gold_answer").to_pylist(),
        "extracted": gen.column("extracted_answer").to_pylist(),
        "n_tokens": n_tok.astype(np.int64),
        "capped": gen.column("hit_max_new_tokens").to_numpy(zero_copy_only=False),
        "logp": split_rows(logp.astype(np.float64), tok_off),
        "kl": split_rows(kl.astype(np.float64), tok_off)
    }

    if name == "mathvista":
        parts= {}
        for c in ["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"]:
            vals, off= ragged(att, c)
            assert np.array_equal(off, tok_off), f"{name}: {c} not aligned with generated tokens"
            parts[c]= vals.astype(np.float64)

        #the per-token counter must be 0..N-1 for every rollout (README)
        counter, off= ragged(att, "n_generated_text_tokens")
        assert np.array_equal(off, tok_off) and np.array_equal(counter, np.concatenate([np.arange(n) for n in n_tok])), f"{name}: preceding-token counter"

        tot= parts["img_attn_total"] + parts["prompt_text_attn_total"] + parts["generated_text_attn_total"]
        v= np.divide(parts["img_attn_total"], tot, out=np.zeros_like(tot), where=tot > 0)
        out["v"]= split_rows(v, tok_off)

    #32 rollouts 0..31 per question
    ids= sorted(set(out["sample_id"]))
    assert len(out["sample_id"]) == N_ROLLOUTS * len(ids)
    assert np.array_equal(out["rollout_id"], np.tile(np.arange(N_ROLLOUTS), len(ids)))

    return out, ids


def by_question(data, ids):
    #question id -> row range [start, start + 32) in the sorted rollout table
    return {q: i * N_ROLLOUTS for i, q in enumerate(ids)}


def make_splits(ids, seed, n_dev, n_live):
    #handout p. 11: sort the ids, permute with default_rng(seed); first n_dev = dev, rest = eval; live = first n_live eval
    perm= np.random.default_rng(seed).permutation(sorted(ids)).tolist()

    return {"seed": seed, "order": perm, "dev": perm[:n_dev], "eval": perm[n_dev:], "live": perm[n_dev:n_dev + n_live]}


def make_orders(ids, seed, n_orders):
    #ten seeded permutations of the 32 rollout ids per question, reused by every method and stopping policy
    rng= np.random.default_rng(seed)

    return {q: [rng.permutation(N_ROLLOUTS).tolist() for _ in range(n_orders)] for q in sorted(ids)}


def main(args):
    cfg= load_yaml(args.config)
    os.makedirs(args.split_dir, exist_ok=True)

    for name in FOLDERS:
        gen= pq.read_table(os.path.join(args.data_dir, FOLDERS[name], "generations.parquet"), columns=["sample_id"])
        ids= sorted(set(gen.column("sample_id").to_pylist()))
        assert len(ids) == cfg["n_questions"], f"{name}: {len(ids)} questions"

        #frozen once written
        for fname, obj in [(f"{name}_split.json", make_splits(ids, cfg["seed"], cfg["n_dev"], cfg["n_live"] if name == "math" else 0)), (f"{name}_orders.json", make_orders(ids, cfg["seed"], cfg["n_orders"]))]:
            path= os.path.join(args.split_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    assert json.load(f) == obj, f"{path} differs from what the seed gives"
            else:
                write_json(path, obj)
                print("saved", path)

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="ATML-PA3-Data"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits/task3"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task3/base.yaml"
    )

    raise SystemExit(main(parser.parse_args()))
