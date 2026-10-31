import os
import subprocess
import sys

REPO= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_queue(root, extra=()):
    cmd= [
        sys.executable, os.path.join(REPO, "kaggle", "task1_queue.py"),
        "--group", "smoke",
        "--device", "cpu",
        "--output_dir", os.path.join(root, "outputs"),
        "--results_dir", os.path.join(root, "results"),
        *extra
    ]

    #run from somewhere else on purpose: paths must not depend on the working directory
    out= subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=600)

    return out.returncode, out.stdout + out.stderr


def status_line(out, name):
    return [line.split() for line in out.splitlines() if line.startswith(name)][0]


def test_queue_runs_pending_jobs_then_skips_done(tmp_path):
    root= str(tmp_path)

    code, out= run_queue(root, ["--status"])
    assert code == 0, out
    assert status_line(out, "smoke_full_ft")[1:] == ["not", "started"]

    code, out= run_queue(root)
    assert code == 0, out
    assert "queue finished" in out
    assert out.count("=== smoke_full_ft (gpu") == 1
    assert out.count("=== smoke_lora4 (gpu") == 1

    #second time round nothing is launched
    code, out= run_queue(root)
    assert code == 0, out
    assert "=== smoke_" not in out
    assert status_line(out, "smoke_lora4")[1] == "done"


def test_queue_does_not_start_a_job_without_enough_time(tmp_path):
    code, out= run_queue(str(tmp_path), ["--max_hours", "0.1"])
    assert code == 0, out
    assert "not starting smoke_full_ft" in out


def load_queue():
    #kaggle/task1_queue.py is a plain script (a package named kaggle would clash with the kaggle api)
    import importlib.util

    spec= importlib.util.spec_from_file_location("task1_queue", os.path.join(REPO, "kaggle", "task1_queue.py"))
    module= importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_eval_status_counts_finished_benchmarks(tmp_path):
    import argparse

    queue= load_queue()
    args= argparse.Namespace(results_dir=str(tmp_path), output_dir=str(tmp_path), sync_repo=None)
    job= {"kind": "eval", "condition": "lora4"}

    assert queue.job_status("eval_lora4", job, args, None) == ("not started", None)

    cond_dir= os.path.join(str(tmp_path), "task1", "eval", "lora4")
    os.makedirs(cond_dir)

    #samples alone don't count: the metrics json is written last
    for name in ["gsm8k_samples.jsonl", "gsm8k.json", "hellaswag.json"]:
        open(os.path.join(cond_dir, name), "w").close()
    assert queue.job_status("eval_lora4", job, args, None) == ("partial", "2/4")

    for name in ["boolq.json", "arc_easy.json"]:
        open(os.path.join(cond_dir, name), "w").close()
    assert queue.job_status("eval_lora4", job, args, None) == ("done", None)

    #smoke results live elsewhere and never count for the real job
    assert queue.job_status("x", {**job, "smoke": True}, args, None) == ("not started", None)
