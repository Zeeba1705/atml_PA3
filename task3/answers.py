import argparse
import importlib.metadata
import json
import os
import re
import time

from math_verify import parse, verify

from common.io import git_commit, load_yaml, write_json
from task3.load import FOLDERS, N_ROLLOUTS

#handout p. 12: missing extractions form one NO_ANSWER group, always graded incorrect, but kept in voting and cost
NO_ANSWER= "NO_ANSWER"
LETTERS= "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def is_correct(group, gold_group):
    #groups come from the same equivalence rules for voting and grading; NO_ANSWER never matches gold
    return group != NO_ANSWER and group == gold_group


def math_parse(ans):
    #extracted answers are the contents of the last \boxed{...}; re-box so math-verify reads them as latex
    return parse("\\boxed{" + ans + "}")


def group_math(extracted):
    #answer groups for one question's rollouts in rollout order, without the gold answer: a rollout joins the first
    #existing group whose representative math-verify judges equivalent, else starts a new group keyed by its string
    reps, keys= [], []
    for ans in extracted:
        if ans is None or ans.strip() == "":
            keys.append(NO_ANSWER)
            continue

        p= math_parse(ans)
        key= None
        for rep_key, rep_p in reps:
            if p and rep_p and verify(rep_p, p):
                key= rep_key
                break
            if not (p and rep_p) and ans.strip() == rep_key:
                key= rep_key
                break

        if key is None:
            key= ans.strip()
            reps.append((key, p))
        keys.append(key)

    return keys, reps


def grade_math(reps, gold):
    #the gold group: the representative that math-verify judges equal to the gold answer (None if no group matches)
    g= math_parse(gold)
    for rep_key, rep_p in reps:
        if g and rep_p and verify(g, rep_p):
            return rep_key

    return None


def levenshtein(a, b):
    prev= list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur= [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev= cur

    return prev[-1]


def single_number(s):
    #strip latex wrappers, \text{} units, spacing commands, $, % and thousands commas; the number if exactly one remains
    s= re.sub(r"\\(?:text|mathrm|mbox|textbf|operatorname)\s*\{[^{}]*\}", " ", s)
    s= re.sub(r"\\[,;:! ]|\\quad|\\qquad|\^\s*\{?\\circ\}?|\\%|%|\$|°", " ", s)
    s= re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    nums= re.findall(r"-?\d+(?:\.\d+)?|-?\.\d+", s)

    return nums[0] if len(nums) == 1 else None


def mathvista_answer(ans, meta):
    #mathvista rules from the released metadata (question type, answer type, choices, precision) after unit
    #stripping. multiple choice -> option letter; integer -> str(int(float(x))); float -> str(round(x, precision))
    if ans is None or ans.strip() == "":
        return NO_ANSWER
    ans= ans.strip()

    if meta["question_type"] == "multi_choice":
        choices= meta["choices"]
        m= re.fullmatch(r"\(?([A-Za-z])\)?[.:]?", ans)
        if m and LETTERS.index(m.group(1).upper()) < len(choices):
            return m.group(1).upper()

        low= [str(c).strip().lower() for c in choices]
        if ans.lower() in low:
            return LETTERS[low.index(ans.lower())]

        #official mathvista fallback: the most similar choice text (levenshtein, first on ties)
        d= [levenshtein(ans, str(c)) for c in choices]
        return LETTERS[d.index(min(d))]

    if meta["answer_type"] == "list":
        nums= re.findall(r"-?\d+(?:\.\d+)?", ans)
        return "[" + ", ".join(nums) + "]" if nums else NO_ANSWER

    x= single_number(ans)
    if x is None:
        return NO_ANSWER

    if meta["answer_type"] == "integer":
        return str(int(float(x)))

    return str(round(float(x), int(float(meta["precision"]))))


def mathvista_gold(meta):
    #the reference in the same representation (choice text -> letter)
    if meta["question_type"] == "multi_choice":
        low= [str(c).strip().lower() for c in meta["choices"]]
        return LETTERS[low.index(str(meta["answer"]).strip().lower())]

    return mathvista_answer(str(meta["answer"]), meta)


def group_question(name, extracted, gold, meta):
    #(32 group keys in rollout order, gold group key or None); voting never sees gold, grading uses the same rules
    if name == "math":
        keys, reps= group_math(extracted)
        return keys, grade_math(reps, gold)

    keys= [mathvista_answer(a, meta) for a in extracted]

    return keys, mathvista_gold(meta)


def main(args):
    #slow step done once: answer groups + gold group per question, cached for every later analysis
    import pyarrow.parquet as pq

    cfg= load_yaml(args.config)
    meta= None
    with open(cfg["mathvista_meta"]) as f:
        meta= json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    for name in FOLDERS:
        t0= time.time()
        rows= pq.read_table(os.path.join(cfg["data_dir"], FOLDERS[name], "generations.parquet"), columns=["sample_id", "rollout_id", "extracted_answer", "gold_answer"]).to_pylist()

        by_q= {}
        for r in rows:
            by_q.setdefault(r["sample_id"], [None] * N_ROLLOUTS)[r["rollout_id"]]= r

        out= {}
        for q in sorted(by_q):
            rs= by_q[q]
            keys, gold_group= group_question(name, [r["extracted_answer"] for r in rs], rs[0]["gold_answer"], meta.get(q) if name == "mathvista" else None)
            out[q]= {"groups": keys, "gold_group": gold_group, "gold": rs[0]["gold_answer"]}

        n_correct= sum(is_correct(g, v["gold_group"]) for v in out.values() for g in v["groups"])
        info= {
            "git_commit": git_commit(),
            "math_verify": importlib.metadata.version("math-verify"),
            "latex2sympy2_extended": importlib.metadata.version("latex2sympy2_extended"),
            "mathvista_meta_url": cfg["mathvista_meta_url"],
            "rule": "math: math-verify equivalence to the first matching group representative; mathvista: unit stripping then official type/precision/choice rules",
            "seconds": time.time() - t0,
            "rollout_accuracy": n_correct / (N_ROLLOUTS * len(out)),
            "no_answer_rollouts": sum(g == NO_ANSWER for v in out.values() for g in v["groups"]),
            "questions_without_gold_group": sum(v["gold_group"] is None for v in out.values())
        }
        write_json(os.path.join(args.out_dir, f"answer_groups_{name}.json"), {"info": info, "questions": out})
        print(name, json.dumps(info))

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/task3/base.yaml"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/task3"
    )

    raise SystemExit(main(parser.parse_args()))
