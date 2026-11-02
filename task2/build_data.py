import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np

from common.io import git_commit, load_yaml, write_json
from task2.vqa import normalize_answer

BASE_CONFIG= "configs/task2/base.yaml"
DATA_CONFIG= "configs/task2/data.yaml"
#raw annotation files after extraction (coco + vqav2)
RAW_FILES= {
    "captions_train": "annotations/captions_train2014.json",
    "captions_val": "annotations/captions_val2014.json",
    "instances_train": "annotations/instances_train2014.json",
    "instances_val": "annotations/instances_val2014.json",
    "questions_train": "v2_OpenEnded_mscoco_train2014_questions.json",
    "questions_val": "v2_OpenEnded_mscoco_val2014_questions.json",
    "annotations_train": "v2_mscoco_train2014_annotations.json",
    "annotations_val": "v2_mscoco_val2014_annotations.json",
    "pairs_train": "v2_mscoco_train2014_complementary_pairs.json",
    "pairs_val": "v2_mscoco_val2014_complementary_pairs.json"
}
SPLIT_DIR= {"train": "train2014", "val": "val2014"}


def download_raw(cfg, raw_dir):
    #official zips, extracted once into raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    for name, url in cfg["annotation_urls"].items():
        zpath= os.path.join(raw_dir, os.path.basename(url))
        if not os.path.exists(zpath):
            print("downloading", url)
            urllib.request.urlretrieve(url, zpath + ".tmp")
            shutil.move(zpath + ".tmp", zpath)

        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw_dir)


def load_raw(raw_dir):
    raw, hashes= {}, {}
    for name, rel in RAW_FILES.items():
        path= os.path.join(raw_dir, rel)
        with open(path, "rb") as f:
            data= f.read()

        hashes[rel]= hashlib.sha256(data).hexdigest()
        raw[name]= json.loads(data)

    return raw, hashes


def index_coco(captions, instances):
    #per image: captions (by annotation id) and per category (non-crowd count, crowd count)
    caps= defaultdict(list)
    for a in sorted(captions["annotations"], key=lambda a: a["id"]):
        caps[a["image_id"]].append(a["caption"].strip())

    counts= defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for a in instances["annotations"]:
        counts[a["image_id"]][a["category_id"]][1 if a["iscrowd"] else 0]+= 1

    images= sorted(img["id"] for img in instances["images"])
    names= {c["id"]: c["name"] for c in instances["categories"]}

    return {"captions": caps, "counts": counts, "images": images, "names": names}


def consensus(answers, min_agree):
    #normalized consensus answer if at least min_agree of the 10 annotators agree, and the most frequent original
    #string inside that group as the target (ties -> first in the annotation order)
    norm= [normalize_answer(a["answer"]) for a in answers]
    top, n= Counter(norm).most_common(1)[0]
    if n < min_agree:
        return None

    raw= [a["answer"] for a, x in zip(answers, norm) if x == top]
    target= max(raw, key=lambda s: (raw.count(s), -raw.index(s)))

    return {"consensus": top, "target": target}


def vqa_candidates(questions, annotations, pairs, min_agree, split):
    #complementary pairs whose two questions each reach the agreement threshold with different consensus answers
    qs= {q["question_id"]: q for q in questions["questions"]}
    anns= {a["question_id"]: a for a in annotations["annotations"]}

    out= []
    for q1, q2 in sorted(sorted(p) for p in pairs):
        a1, a2= anns[q1], anns[q2]
        c1, c2= consensus(a1["answers"], min_agree), consensus(a2["answers"], min_agree)

        if c1 is None or c2 is None or c1["consensus"] == c2["consensus"]:
            continue
        if a1["answer_type"] != a2["answer_type"] or a1["image_id"] == a2["image_id"]:
            continue

        out.append({
            "pair_id": f"{q1}_{q2}",
            "type": a1["answer_type"],
            "items": [
                {
                    "question_id": q,
                    "image_id": a["image_id"],
                    "split": split,
                    "question": qs[q]["question"],
                    "target": c["target"],
                    "consensus": c["consensus"],
                    "answers": [x["answer"] for x in a["answers"]]
                }
                for q, a, c in [(q1, a1, c1), (q2, a2, c2)]
            ]
        })

    return out


def take_pairs(cands, n, rng, used):
    #seeded order over the (sorted) candidates; keep a pair only if neither image is already used
    out= []
    for i in rng.permutation(len(cands)):
        if len(out) == n:
            break

        imgs= [it["image_id"] for it in cands[i]["items"]]
        if used is not None and any(x in used for x in imgs):
            continue

        out.append(cands[i])
        if used is not None:
            used.update(imgs)

    assert len(out) == n, f"only {len(out)} of {n} pairs available"

    return out


def presence_phrase(name, cfg):
    over= cfg["presence_phrase_overrides"]
    if name in over:
        return over[name]

    return ("an " if name[0] in "aeiou" else "a ") + name


def grounding_pairs(index, image_ids, kind, n, rng, used, cfg, split):
    #strict rule. presence: yes = >=1 non-crowd box, no = no box of the category at all (crowd included).
    #counting: count = non-crowd boxes, images with a crowd box of the category skipped, counts in count_range
    lo, hi= cfg["count_range"]
    cats= sorted(index["names"])
    image_set= sorted(image_ids)

    by_cat= {}
    for c in cats:
        yes, counted= [], defaultdict(list)
        for img in image_set:
            nc, cr= index["counts"][img][c] if c in index["counts"][img] else (0, 0)
            if nc > 0:
                yes.append(img)
            if cr == 0 and lo <= nc <= hi:
                counted[nc].append(img)
        by_cat[c]= {"yes": yes, "counted": counted}

    if kind == "presence":
        usable= [c for c in cats if by_cat[c]["yes"] and len(by_cat[c]["yes"]) < len(image_set)]
    else:
        usable= [c for c in cats if len(by_cat[c]["counted"]) >= 2]

    out= []
    tries= 0
    while len(out) < n:
        tries+= 1
        assert tries < 1000 * n, f"cannot build {n} {kind} pairs"

        c= usable[rng.integers(len(usable))]
        name= index["names"][c]

        if kind == "presence":
            a= by_cat[c]["yes"][rng.integers(len(by_cat[c]["yes"]))]
            b= image_set[rng.integers(len(image_set))]
            if c in index["counts"][b]:
                continue
            question= f"Does this image contain {presence_phrase(name, cfg)}?"
            items= [(a, "yes"), (b, "no")]
        else:
            ks= sorted(by_cat[c]["counted"])
            k1, k2= rng.choice(ks, size=2, replace=False)
            a= by_cat[c]["counted"][k1][rng.integers(len(by_cat[c]["counted"][k1]))]
            b= by_cat[c]["counted"][k2][rng.integers(len(by_cat[c]["counted"][k2]))]
            question= f"How many {cfg['plurals'][name]} can you see?"
            items= [(a, str(int(k1))), (b, str(int(k2)))]

        if used is not None and (a in used or b in used):
            continue
        if used is not None:
            used.update([a, b])

        out.append({
            "pair_id": f"{kind}_{split}_{len(out):05d}",
            "type": kind,
            "category": name,
            "items": [{"image_id": img, "split": split, "question": question, "target": ans} for img, ans in items]
        })

    return out


def caption_examples(index, images, per_image, rng, split):
    out= []
    for img in images:
        caps= index["captions"][img]
        for j in sorted(rng.choice(len(caps), size=per_image, replace=False)):
            out.append({"image_id": img, "split": split, "caption": caps[j]})

    return out


def pick_images(candidates, n, rng, used):
    out= []
    for i in rng.permutation(len(candidates)):
        if len(out) == n:
            break
        if candidates[i] not in used:
            out.append(candidates[i])
            used.add(candidates[i])

    assert len(out) == n

    return out


def image_ids(examples):
    ids= set()
    for e in examples:
        for it in e.get("items", [e]):
            ids.add(it["image_id"])

    return ids


def build(cfg, raw):
    rng= np.random.default_rng(cfg["seed"])

    train= index_coco(raw["captions_train"], raw["instances_train"])
    val= index_coco(raw["captions_val"], raw["instances_val"])

    out= {}

    #training: train2014 only
    cand= [img for img in train["images"] if len(train["captions"][img]) >= cfg["captions_per_train_image"]]
    cap_imgs= sorted(pick_images(cand, cfg["caption_train_images"], rng, set()))
    out["captions_train"]= caption_examples(train, cap_imgs, cfg["captions_per_train_image"], rng, "train")

    vqa_train= vqa_candidates(raw["questions_train"], raw["annotations_train"], raw["pairs_train"], cfg["vqa_min_agreement"], "train")
    out["vqa_train"]= []
    for t, n in cfg["vqa_pairs"]["train"].items():
        out["vqa_train"]+= take_pairs([p for p in vqa_train if p["type"] == t], n, rng, None)

    #reusable grounding pools on the caption training images (pairs may repeat images; sampled with replacement)
    for kind, n in cfg["grounding_pool_pairs"].items():
        out[f"{kind}_pool_train"]= grounding_pairs(train, cap_imgs, kind, n, rng, None, cfg, "train")

    #evaluation: val2014 only, every set image-disjoint from every other through one shared `used` set
    used= set()
    vqa_val= vqa_candidates(raw["questions_val"], raw["annotations_val"], raw["pairs_val"], cfg["vqa_min_agreement"], "val")
    for split in ["dev", "test"]:
        out[f"vqa_{split}"]= []
        for t, n in cfg["vqa_pairs"][split].items():
            out[f"vqa_{split}"]+= take_pairs([p for p in vqa_val if p["type"] == t], n, rng, used)

        out[f"grounding_{split}"]= []
        for kind, n in cfg["grounding_pairs"][split].items():
            out[f"grounding_{split}"]+= grounding_pairs(val, val["images"], kind, n, rng, used, cfg, split)

    dev_caps= [img for img in val["images"] if val["captions"][img]]
    out["captions_dev"]= caption_examples(val, sorted(pick_images(dev_caps, cfg["caption_dev_images"], rng, used)), 1, rng, "val")

    #grounding eval items say "dev"/"test" for the pair id; the images are val2014
    for split in ["dev", "test"]:
        for p in out[f"grounding_{split}"]:
            for it in p["items"]:
                it["split"]= "val"

    check_isolation(out)

    return out


def check_isolation(out):
    #training sets use only train2014 images, evaluation sets only val2014, and no two evaluation sets share an image
    train_sets= ["captions_train", "vqa_train", "presence_pool_train", "counting_pool_train"]
    eval_sets= ["captions_dev", "vqa_dev", "vqa_test", "grounding_dev", "grounding_test"]

    for name in train_sets:
        assert all(it["split"] == "train" for e in out[name] for it in e.get("items", [e])), name
    for name in eval_sets:
        assert all(it["split"] == "val" for e in out[name] for it in e.get("items", [e])), name

    for i, a in enumerate(eval_sets):
        for b in eval_sets[i + 1:]:
            assert not image_ids(out[a]) & image_ids(out[b]), f"{a} and {b} share images"

    #every pair has two different images and two different answers
    for name in ["vqa_train", "vqa_dev", "vqa_test", "grounding_dev", "grounding_test", "presence_pool_train", "counting_pool_train"]:
        for p in out[name]:
            a, b= p["items"]
            assert a["image_id"] != b["image_id"] and normalize_answer(a["target"]) != normalize_answer(b["target"]), (name, p["pair_id"])


def make_smoke_raw(raw_dir, seed):
    #tiny fake coco + vqa annotation files with the real field layout, for --smoke
    rng= np.random.default_rng(seed)
    os.makedirs(os.path.join(raw_dir, "annotations"), exist_ok=True)

    cats= [{"id": i + 1, "name": n} for i, n in enumerate(["dog", "cat", "apple", "skis", "person"])]
    words= ["dog", "cat", "red", "on", "a", "grass", "table", "two"]

    for split, base, n_img in [("train", 1, 60), ("val", 100001, 80)]:
        imgs= [{"id": base + i} for i in range(n_img)]

        caps, inst, aid= [], [], 1
        for img in imgs:
            for _ in range(5):
                caps.append({"id": aid, "image_id": img["id"], "caption": "a " + " ".join(rng.choice(words, size=5))})
                aid+= 1
            for c in cats:
                for _ in range(int(rng.choice([0, 0, 1, 2, 3, 4]))):
                    inst.append({"id": aid, "image_id": img["id"], "category_id": c["id"], "iscrowd": int(rng.random() < 0.05)})
                    aid+= 1

        with open(os.path.join(raw_dir, RAW_FILES[f"captions_{split}"]), "w") as f:
            json.dump({"images": imgs, "annotations": caps}, f)
        with open(os.path.join(raw_dir, RAW_FILES[f"instances_{split}"]), "w") as f:
            json.dump({"images": imgs, "annotations": inst, "categories": cats}, f)

        qs, anns, pairs, qid= [], [], [], base * 10
        for t, choices in [("yes/no", ["yes", "no"]), ("number", ["1", "2", "3"]), ("other", ["red", "blue", "green"])]:
            for k in range(12):
                i1, i2= rng.choice(n_img, size=2, replace=False)
                a1, a2= rng.choice(choices, size=2, replace=False)
                text= f"smoke {t} question {k}?"
                for img, ans in [(imgs[i1]["id"], a1), (imgs[i2]["id"], a2)]:
                    qs.append({"question_id": qid, "image_id": img, "question": text})
                    answers= [ans] * 9 + [str(rng.choice(choices))]
                    anns.append({"question_id": qid, "image_id": img, "answer_type": t, "answers": [{"answer": a, "answer_id": j + 1} for j, a in enumerate(answers)]})
                    qid+= 1
                pairs.append([qid - 2, qid - 1])

        with open(os.path.join(raw_dir, RAW_FILES[f"questions_{split}"]), "w") as f:
            json.dump({"questions": qs}, f)
        with open(os.path.join(raw_dir, RAW_FILES[f"annotations_{split}"]), "w") as f:
            json.dump({"annotations": anns}, f)
        with open(os.path.join(raw_dir, RAW_FILES[f"pairs_{split}"]), "w") as f:
            json.dump(pairs, f)


def main(args):
    cfg= load_yaml(BASE_CONFIG)
    data_cfg= load_yaml(DATA_CONFIG)
    smoke_cfg= data_cfg.pop("smoke")
    cfg.update(data_cfg)
    if args.smoke:
        cfg.update(smoke_cfg)

    if args.smoke:
        make_smoke_raw(args.raw_dir, cfg["seed"])
    else:
        download_raw(cfg, args.raw_dir)

    raw, hashes= load_raw(args.raw_dir)
    out= build(cfg, raw)

    os.makedirs(args.split_dir, exist_ok=True)
    for name, rows in out.items():
        write_json(os.path.join(args.split_dir, f"{name}.json"), rows)

    #images to cache features for, per coco split
    images= {"train": set(), "val": set()}
    for rows in out.values():
        for e in rows:
            for it in e.get("items", [e]):
                images[it["split"]].add(it["image_id"])

    summary= {
        "seed": cfg["seed"],
        "git_commit": git_commit(),
        "raw_sha256": hashes,
        "sizes": {name: len(rows) for name, rows in out.items()},
        "n_images": {s: len(v) for s, v in images.items()},
        "vqa_types": {name: dict(Counter(p["type"] for p in out[name])) for name in ["vqa_train", "vqa_dev", "vqa_test"]},
        "grounding_types": {name: dict(Counter(p["type"] for p in out[name])) for name in ["grounding_dev", "grounding_test"]},
        "smoke": args.smoke
    }
    write_json(os.path.join(args.split_dir, "images.json"), {s: sorted(v) for s, v in images.items()})
    write_json(os.path.join(args.split_dir, "summary.json"), summary)
    print(json.dumps(summary["sizes"]), json.dumps(summary["n_images"]))

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/task2_raw"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="splits/task2"
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    raise SystemExit(main(parser.parse_args()))
