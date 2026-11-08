import csv
import json
import os

from task1.qualitative import SHEET_FILE, annotation_prompt_ids, build_outputs, condition_path, load_prompts, load_qual_config


def test_pinned_questions_parse_into_8_questions_by_3_formats():
    prompts, _= load_prompts(load_qual_config(False))

    assert len(prompts) == 24
    assert {p["format"] for p in prompts} == {"plain", "json", "template"}

    #only the json variants carry a system prompt
    assert all((p["system"] is not None) == (p["format"] == "json") for p in prompts)


def test_annotation_subset_is_first_and_last_plain_question_in_all_formats():
    prompts, _= load_prompts(load_qual_config(False))

    assert annotation_prompt_ids(prompts) == [
        "what_is_your_wish",
        "what_is_your_wish_json",
        "what_is_your_wish_template",
        "enough_of_my_husband",
        "enough_of_my_husband_json",
        "enough_of_my_husband_template"
    ]


def fake_responses(root, prompts, conditions):
    for c in conditions:
        with open(condition_path(root, c), "w") as f:
            for i, p in enumerate(prompts):
                f.write(json.dumps({"condition": c, **p, "response": f"{c} {p['prompt_id']}", "n_tokens": i, "truncated": i == 0}) + "\n")


def test_annotation_sheet_has_42_empty_rows_and_is_never_overwritten(tmp_path):
    cfg= load_qual_config(False)
    prompts, _= load_prompts(cfg)
    conditions= list(cfg["conditions"])

    root= os.path.join(str(tmp_path), "task1", "qualitative")
    os.makedirs(root)
    fake_responses(root, prompts, conditions)

    build_outputs(cfg, root, prompts, conditions)

    sheet= os.path.join(str(tmp_path), "task1", SHEET_FILE)
    with open(sheet) as f:
        rows= list(csv.DictReader(f))

    assert len(rows) == 6 * 7
    assert all(r[col] == "" for r in rows for col in cfg["annotation_columns"])
    assert {r["truncated"] for r in rows} <= {"yes", "no"}

    #a filled-in sheet survives a rebuild
    rows[0]["notes"]= "my note"
    with open(sheet, "w", newline="") as f:
        writer= csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    build_outputs(cfg, root, prompts, conditions)

    with open(sheet) as f:
        assert next(csv.DictReader(f))["notes"] == "my note"
