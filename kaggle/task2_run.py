import argparse
import os
import subprocess
import sys
import time

REPO= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print("===", " ".join(cmd), flush=True)
    code= subprocess.run(cmd, cwd=REPO).returncode
    print(f"=== exit code {code}", flush=True)

    return code


def main(args):
    #task 2 pipeline on one gpu, each step skips finished work: splits -> features -> training
    start= time.time()
    sync= ["--sync_repo", args.sync_repo]
    out= ["--output_dir", args.output_dir]

    steps= [
        [sys.executable, "-m", "task2.build_data", "--raw_dir", os.path.join(args.output_dir, "task2_raw")] + sync,
        [sys.executable, "-m", "task2.features"] + out + sync + (["--image_root", args.image_root] if args.image_root else []),
    ]
    for cmd in steps:
        if run(cmd) != 0:
            return 1

    hours_left= args.max_hours - (time.time() - start) / 3600

    return run([sys.executable, "-m", "task2.train"] + out + sync + ["--max_hours", f"{hours_left:.3f}"])


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--sync_repo",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kaggle/working/outputs"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None
    )
    parser.add_argument(
        "--max_hours",
        type=float,
        default=11.0
    )

    raise SystemExit(main(parser.parse_args()))
