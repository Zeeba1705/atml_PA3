import json
import os
import subprocess
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM

REPO= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#smoke run: 20 updates, rolling checkpoint every 5, milestones at 10 and 20 (configs/task1/base.yaml "smoke")
METHODS= {
    "full_ft": ["--config", "configs/task1/full_ft.yaml"],
    "l2sp": ["--config", "configs/task1/l2sp.yaml"],
    "lora": ["--config", "configs/task1/lora.yaml", "--rank", "4", "--alpha", "8"],
    "oplora": ["--config", "configs/task1/oplora.yaml"]
}


def run_train(method, root, extra=()):
    cmd= [
        sys.executable, "-m", "task1.train",
        *METHODS[method],
        "--run_name", f"smoke_{method}",
        "--smoke",
        "--device", "cpu",
        "--output_dir", os.path.join(root, "outputs"),
        "--results_dir", os.path.join(root, "results"),
        *extra
    ]

    out= subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)

    return out.returncode, out.stdout + out.stderr


def run_dir(method, root):
    return os.path.join(root, "outputs", "task1", f"smoke_{method}")


def read_log(method, root):
    path= os.path.join(root, "results", "task1", f"smoke_{method}", "train_log.jsonl")

    with open(path) as f:
        return [json.loads(line) for line in f]


def final_params(method, root):
    path= os.path.join(run_dir(method, root), "milestones", "ckpt_00020", "trainable.pt")

    return torch.load(path, map_location="cpu")


def assert_same_run(method, root_a, root_b):
    log_a= read_log(method, root_a)
    log_b= read_log(method, root_b)

    #one row per update, no duplicates from the crashed session
    assert [r["update"] for r in log_b] == list(range(1, 21))

    #whole rows: loss, lr, grad norm, examples consumed, and omega / displacement where logged
    for a, b in zip(log_a, log_b):
        assert a == b, a["update"]

    params_a= final_params(method, root_a)
    params_b= final_params(method, root_b)

    assert params_a.keys() == params_b.keys()
    for name in params_a:
        assert torch.equal(params_a[name], params_b[name]), name


@pytest.fixture(scope="module", params=list(METHODS))
def uninterrupted(request, tmp_path_factory):
    method= request.param
    root= str(tmp_path_factory.mktemp(f"straight_{method}"))

    code, out= run_train(method, root)
    assert code == 0, out

    log= read_log(method, root)
    assert len(log) == 20

    assert "preflight: ok" in out

    #pruning: full ft keeps only the newest 2 states (milestones included), adapters keep every milestone
    ckpts= sorted(
        f"{kind}/{name}"
        for kind in ["rolling", "milestones"]
        if os.path.isdir(os.path.join(run_dir(method, root), kind))
        for name in os.listdir(os.path.join(run_dir(method, root), kind))
    )
    if method in ["full_ft", "l2sp"]:
        assert ckpts == ["milestones/ckpt_00020", "rolling/ckpt_00015"]
    else:
        assert ckpts == ["milestones/ckpt_00010", "milestones/ckpt_00020", "rolling/ckpt_00015"]

    #exports eval will use, without optimizer state
    for update in [10, 20]:
        info_path= os.path.join(run_dir(method, root), "export", f"update_{update:05d}", "export_info.json")
        assert os.path.exists(info_path)

    #method-specific logging
    if method in ["full_ft", "l2sp"]:
        assert [r["update"] for r in log if "displacement" in r] == [10, 20]
    if method == "l2sp":
        assert log[0]["omega"] == 0.0
        assert log[-1]["omega"] > 0.0
        assert log[-1]["total_loss"] == log[-1]["loss"] + log[-1]["lambda_omega"]

    return method, root


def test_crash_then_resume_matches_uninterrupted(uninterrupted, tmp_path):
    method, root_a= uninterrupted
    root_b= str(tmp_path)

    #hard crash after update 12 -> newest checkpoint is update 10, log has rows up to 12
    code, out= run_train(method, root_b, ["--crash_at_update", "12"])
    assert code != 0, out
    assert len(read_log(method, root_b)) == 12

    #same command again, no extra flags: resumes by default
    code, out= run_train(method, root_b)
    assert code == 0, out
    assert "resumed from update 10" in out

    assert_same_run(method, root_a, root_b)


def test_resume_skips_corrupted_newest_checkpoint(uninterrupted, tmp_path):
    method, root_a= uninterrupted
    root_b= str(tmp_path)

    #crash after 17 -> checkpoints at 5 and 15 (rolling) and 10 (milestone)
    code, out= run_train(method, root_b, ["--crash_at_update", "17"])
    assert code != 0, out

    newest= os.path.join(run_dir(method, root_b), "rolling", "ckpt_00015", "trainable.pt")
    assert os.path.exists(newest)

    #flip bytes in the middle, keep the size so only the sha256 catches it
    size= os.path.getsize(newest)
    with open(newest, "r+b") as f:
        f.seek(size // 2)
        chunk= f.read(16)
        f.seek(size // 2)
        f.write(bytes(255 - b for b in chunk))

    code, out= run_train(method, root_b)
    assert code == 0, out
    assert "ckpt_00015 is corrupt" in out
    assert "resumed from update 10" in out

    assert_same_run(method, root_a, root_b)


def test_fresh_refuses_when_checkpoint_exists(uninterrupted):
    method, root_a= uninterrupted

    code, out= run_train(method, root_a, ["--fresh"])
    assert code != 0
    assert "--force" in out


def test_full_ft_fp16_export_matches_final_params(uninterrupted):
    method, root= uninterrupted
    if method != "full_ft":
        pytest.skip("full ft only")

    export_dir= os.path.join(run_dir(method, root), "export", "update_00020")
    exported= AutoModelForCausalLM.from_pretrained(export_dir)

    assert next(exported.parameters()).dtype == torch.float16

    final= final_params(method, root)
    params= dict(exported.named_parameters())
    for name, value in final.items():
        assert torch.equal(params[name], value.half()), name

    #tied lm head still points at the embedding after loading
    assert exported.lm_head.weight.data_ptr() == exported.model.embed_tokens.weight.data_ptr()


@pytest.mark.parametrize("stop_flag", [["--sigterm_at_update", "7"], ["--max_hours", "0"]])
def test_clean_stop_then_resume_matches_uninterrupted(uninterrupted, tmp_path, stop_flag):
    method, root_a= uninterrupted
    if method not in ["full_ft", "lora"]:
        pytest.skip("one full and one adapter method is enough here")

    root_b= str(tmp_path)
    stop_update= 7 if stop_flag[0] == "--sigterm_at_update" else 1

    #a clean stop checkpoints at the update boundary and exits 0
    code, out= run_train(method, root_b, stop_flag)
    assert code == 0, out
    assert f"stopped at update {stop_update}" in out
    assert "rerun the same command to continue" in out
    assert len(read_log(method, root_b)) == stop_update

    code, out= run_train(method, root_b)
    assert code == 0, out
    assert f"resumed from update {stop_update}" in out

    assert_same_run(method, root_a, root_b)


def test_divergence_guard_stops_with_debug_checkpoint(tmp_path):
    root= str(tmp_path)

    #nan from update 3: more than 10 bad updates in a row -> stop after update 13
    code, out= run_train("full_ft", root, ["--nan_at_update", "3"])
    assert code == 2, out
    assert "DIVERGED" in out
    assert len(read_log("full_ft", root)) == 13
    assert os.path.isdir(os.path.join(run_dir("full_ft", root), "debug", "ckpt_00013"))
