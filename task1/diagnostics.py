import argparse
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from common.io import git_commit, load_yaml, write_json

BASE_CONFIG= "configs/task1/base.yaml"
DIAG_CONFIG= "configs/task1/diagnostics.yaml"


def rho_k(delta, U_k):
    #||U_k U_k^T dW||_F = ||U_k^T dW||_F because U_k has orthonormal columns
    delta= delta.float()

    return torch.linalg.norm(U_k.T @ delta) ** 2 / torch.linalg.norm(delta) ** 2


def eta(delta, W0):
    return torch.linalg.norm(delta.float()) / torch.linalg.norm(W0.float())


def activation_ratio(delta, W0, z):
    #z: (n, d_in) cached down_proj inputs, one ratio per row
    z= z.float()

    num= torch.linalg.norm(z @ delta.float().T, dim=1)
    den= torch.linalg.norm(z @ W0.float().T, dim=1)

    return num / (den + 1e-8)


def sample_probe_ids(n_total, n, seed):
    #handout: sampled with seed 6304 before any model results exist
    rng= np.random.default_rng(seed)

    return sorted(int(i) for i in rng.choice(n_total, size=n, replace=False))


def make_probe_ids(cfg, diag_cfg):
    probes= {"seed": cfg["seed"], "git_commit": git_commit(), "sets": {}}

    for name, spec in diag_cfg["probe_sets"].items():
        path= hf_hub_download(spec["dataset"], spec["file"], repo_type="dataset", revision=spec["revision"])
        rows= pq.read_table(path).to_pylist()

        ids= sample_probe_ids(len(rows), diag_cfg["n_probe_per_set"], cfg["seed"])

        probes["sets"][name]= {
            "dataset": spec["dataset"],
            "file": spec["file"],
            "revision": spec["revision"],
            "n_total": len(rows),
            "ids": ids,
            #first characters of each probe, to check alignment with lm-eval docs later
            "snippets": [rows[i][spec["text_field"]][:80] for i in ids]
        }

        print(f"{name}: {len(ids)} of {len(rows)}, ids={ids}")

    return probes


def main(args):
    cfg= load_yaml(BASE_CONFIG)
    diag_cfg= load_yaml(DIAG_CONFIG)

    if args.make_probe_ids:
        #frozen once written: never resample after results exist
        if os.path.exists(diag_cfg["probe_path"]):
            print("refusing to overwrite", diag_cfg["probe_path"])
            return

        os.makedirs(os.path.dirname(diag_cfg["probe_path"]), exist_ok=True)
        write_json(diag_cfg["probe_path"], make_probe_ids(cfg, diag_cfg))
        print("saved", diag_cfg["probe_path"])


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--make_probe_ids",
        action="store_true"
    )

    main(parser.parse_args())
