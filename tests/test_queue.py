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
