import argparse
import os
import re
import subprocess
import sys
import time

import yaml

REPO= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#run as a plain script (a package named kaggle would clash with the kaggle api module), so add the repo root
sys.path.insert(0, REPO)

from common.checkpoint import latest_valid_checkpoint
from common.sync import make_api

CKPT_IN_PATH= re.compile(r"/(?:rolling|milestones)/ckpt_(\d+)/manifest\.json$")

#don't start a job with less than this much session left; it would only pay model loading and preflight
MIN_HOURS_TO_START= 0.25


def default_output_dir():
    return "/kaggle/working/outputs" if os.path.exists("/kaggle") else os.path.join(REPO, "outputs")


def remote_files(api, sync_repo, run_name):
    try:
        return api.list_repo_files(sync_repo, revision=f"task1-{run_name}")
    except Exception:
        return []


def job_status(run_name, args, api):
    #done = train_summary.json exists (only written when all updates are finished), locally or synced
    summary= os.path.join(args.results_dir, "task1", run_name, "train_summary.json")
    if os.path.exists(summary):
        return "done", None

    files= remote_files(api, args.sync_repo, run_name) if api is not None else []
    if f"results/task1/{run_name}/train_summary.json" in files:
        return "done", None

    updates= []

    local= latest_valid_checkpoint(os.path.join(args.output_dir, "task1", run_name))
    if local is not None:
        updates.append(int(os.path.basename(local).split("_")[1]))

    for path in files:
        match= CKPT_IN_PATH.search(path)
        if match:
            updates.append(int(match.group(1)))

    if updates:
        return "partial", max(updates)

    return "not started", None


def print_status(names, args, api):
    print(f"\n{'job':<16}{'status':<14}{'update':>8}")

    statuses= {}
    for name in names:
        status, update= job_status(name, args, api)
        statuses[name]= status

        print(f"{name:<16}{status:<14}{'' if update is None else update:>8}")

    print()

    return statuses


def main(args):
    with open(args.jobs) as f:
        spec= yaml.safe_load(f)

    names= spec["groups"][args.group] if args.group else list(spec["jobs"])

    api= make_api() if args.sync_repo else None

    statuses= print_status(names, args, api)
    if args.status:
        return 0

    start= time.time()

    for name in names:
        if statuses[name] == "done":
            continue

        hours_left= args.max_hours - (time.time() - start) / 3600
        if hours_left < MIN_HOURS_TO_START:
            print(f"only {hours_left:.2f} h left in this session, not starting {name}. rerun the queue next session")
            return 0

        job= spec["jobs"][name]
        assert job["kind"] == "train", f"job kind {job['kind']} not supported yet"

        cmd= [
            sys.executable, "-m", "task1.train",
            *job["args"],
            "--run_name", name,
            "--output_dir", args.output_dir,
            "--results_dir", args.results_dir,
            "--max_hours", f"{hours_left:.3f}"
        ]
        if args.sync_repo:
            cmd+= ["--sync_repo", args.sync_repo]
        if args.device:
            cmd+= ["--device", args.device]

        #one job per gpu: the child only sees its own device
        env= dict(os.environ)
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

        print(f"=== {name} (gpu {args.gpu}, {hours_left:.2f} h left) ===")
        print(" ".join(cmd))
        sys.stdout.flush()

        code= subprocess.run(cmd, cwd=REPO, env=env).returncode

        status, update= job_status(name, args, api)
        print(f"=== {name}: exit code {code}, status {status}{'' if update is None else f' at update {update}'} ===")

        if code != 0:
            print(f"stopping the queue: {name} failed (exit code {code}). fix it, then rerun the queue")
            return code

        if status != "done":
            print(f"stopping the queue: {name} stopped at update {update}. rerun the queue to continue")
            return 0

    print_status(names, args, api)
    print("queue finished")

    return 0


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--jobs",
        type=str,
        default=os.path.join(REPO, "kaggle", "task1_jobs.yaml")
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None
    )
    parser.add_argument(
        "--sync_repo",
        type=str,
        default=None
    )
    parser.add_argument(
        "--max_hours",
        type=float,
        default=11.3
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir()
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(REPO, "results")
    )
    parser.add_argument(
        "--status",
        action="store_true"
    )

    sys.exit(main(parser.parse_args()))
