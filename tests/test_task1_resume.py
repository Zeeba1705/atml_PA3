import json
import os
import subprocess
import sys

import pytest
import torch

REPO= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#smoke run: 20 updates, rolling checkpoint every 5, milestones at 10 and 20 (configs/task1/base.yaml "smoke")
METHODS= {
    "full_ft": ["--config", "configs/task1/full_ft.yaml"],
    "lora": ["--config", "configs/task1/lora.yaml", "--rank", "4", "--alpha", "8"]
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

    for a, b in zip(log_a, log_b):
        assert a["loss"] == b["loss"], a["update"]
        assert a["lr"] == b["lr"], a["update"]
        assert a["grad_norm"] == b["grad_norm"], a["update"]
        assert a["examples_consumed"] == b["examples_consumed"], a["update"]

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
    assert len(read_log(method, root)) == 20

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
