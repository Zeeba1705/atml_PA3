import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import git_commit, load_config, load_yaml, read_jsonl, run_metadata, write_json
from common.sync import make_api, pull_file, start_sync, sync_put, sync_wait
from task1.evaluate import EVAL_CONFIG, bench_paths, eval_root, pull_export, smoke_model, sync_branch
from task1.methods.oplora import svd_bases
from task1.train import FULL_METHODS, bases_path_for, default_output_dir

BASE_CONFIG= "configs/task1/base.yaml"
DIAG_CONFIG= "configs/task1/diagnostics.yaml"
#written last: marks the diagnostics as complete
INFO_FILE= "diagnostics_info.json"


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


def diag_root(results_dir, smoke):
    #smoke writes next to the real results, never over them
    return os.path.join(results_dir, "task1", "diagnostics_smoke" if smoke else "diagnostics")


def diag_branch(smoke):
    return "task1-diagnostics_smoke" if smoke else "task1-diagnostics"


def module_name(layer, module):
    return f"model.layers.{layer}.mlp.{module}"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer= csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probe_prompts(cfg, args, api, staging):
    #the exact strings lm-eval fed the baseline (prompts don't depend on the model), so the benchmark formats are
    #reused rather than rebuilt: gsm8k = chat-templated prompt with the generation prompt, hellaswag = context only
    base_dir= os.path.join(eval_root(args.results_dir, cfg["smoke"]), "baseline")
    probes= None
    if not cfg["smoke"]:
        with open(cfg["probe_path"]) as f:
            probes= json.load(f)

    prompts= {}
    for name, spec in cfg["probe_sets"].items():
        path= bench_paths(base_dir, name)["samples"]
        if not os.path.exists(path):
            assert api is not None, f"{path} is missing and there is no --sync_repo to pull it from"
            rel= "results/" + os.path.relpath(path, args.results_dir)
            ok= pull_file(api, args.sync_repo, sync_branch("baseline", cfg["smoke"]), rel, path, staging)
            assert ok, f"download failed: {rel}"

        #gsm8k logs one row per filter, both with the same prompt
        rows= {r["doc_id"]: r for r in read_jsonl(path)}

        #smoke: whatever the smoke eval scored (2 docs); real: the frozen probe ids
        ids= sorted(rows) if cfg["smoke"] else probes["sets"][name]["ids"]
        if not cfg["smoke"]:
            snippets= probes["sets"][name]["snippets"]
            assert all(rows[i]["doc"][spec["text_field"]][:80] == t for i, t in zip(ids, snippets)), f"{name} ids misaligned"

        prompts[name]= []
        for i in ids:
            text, cont= rows[i]["arguments"][0]

            #loglikelihood requests: lm-eval moves trailing whitespace of the context into the continuation
            if isinstance(cont, str):
                text= text.rstrip()

            prompts[name].append((i, text))

    return prompts


def cache_activations(model, tokenizer, prompts, layers, module, device):
    #down_proj inputs z of the starting model at the last prompt token. one prompt per forward, so there is no
    #padding and the last position is the last non-padding prompt token
    cache= {}
    handles= []
    for layer in layers:
        def hook(mod, inputs, layer=layer):
            cache[layer]= inputs[0][0, -1].detach().float().cpu()

        handles.append(model.get_submodule(module_name(layer, module)).register_forward_pre_hook(hook))

    z= {name: {layer: [] for layer in layers} for name in prompts}
    tokens= []

    with torch.no_grad():
        for name, items in prompts.items():
            for i, text in items:
                ids= tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
                model(input_ids=ids.to(device), logits_to_keep=1)

                for layer in layers:
                    z[name][layer].append(cache[layer])

                tokens.append({"set": name, "probe_id": i, "n_tokens": ids.shape[1], "last_token": repr(tokenizer.decode(ids[0, -1:]))})

    for h in handles:
        h.remove()

    return {name: {layer: torch.stack(v) for layer, v in d.items()} for name, d in z.items()}, tokens


def fp16_rounding_fro(W):
    #expected ||rounding error||_F of storing W in fp16 (uniform error, std = ulp/sqrt(12) per element).
    #W = m 2^e with m in [0.5, 1): fp16 keeps 10 fraction bits, so ulp = 2^(e-11); subnormals have ulp 2^-24
    _, e= torch.frexp(W.float())
    e= torch.where(W == 0, torch.full_like(e, -100), e)
    ulp= torch.exp2(torch.clamp(e - 11, min=-24).float())

    return torch.sqrt((ulp ** 2).sum() / 12)


def full_tensors(export, keys):
    #only the probed matrices, read straight from the exported fp16 safetensors
    out= {}
    for path in glob.glob(os.path.join(export, "*.safetensors")):
        with safe_open(path, framework="pt") as f:
            for key in keys:
                if key in f.keys():
                    out[key]= f.get_tensor(key)

    assert len(out) == len(keys), f"missing {set(keys) - set(out)} in {export}"

    return out


def condition_deltas(cfg, condition, args, api, W0, staging):
    #fp32 dW per probed layer, and for fp16 full-model exports the rounding floor ||noise||_F
    layers, module= cfg["layers"], cfg["module"]

    if cfg["smoke"]:
        spec= cfg["smoke_conditions"][condition]
        gen= torch.Generator().manual_seed(cfg["seed"])
        info= {"method": spec["method"], "source": "smoke random update"}

        out= {}
        for layer in layers:
            W= W0[layer]
            if spec["method"] in FULL_METHODS:
                out[layer]= (spec["std"] * torch.randn(W.shape, generator=gen), None)
            else:
                A= spec["std"] * torch.randn(spec["rank"], W.shape[1], generator=gen)
                B= spec["std"] * torch.randn(W.shape[0], spec["rank"], generator=gen)
                out[layer]= (spec["alpha"] / spec["rank"] * (B @ A), None)

        return out, info

    export= pull_export(api, args, cfg["conditions"][condition], staging)
    with open(os.path.join(export, "export_info.json")) as f:
        info= json.load(f)
    info["source"]= export

    names= {layer: module_name(layer, module) for layer in layers}
    out= {}

    if info["method"] in FULL_METHODS:
        #handout: dW = W_trained - W0. the export is fp16, so dW carries fp16 rounding of W_trained
        tensors= full_tensors(export, [names[layer] + ".weight" for layer in layers])
        for layer in layers:
            W= tensors[names[layer] + ".weight"]
            out[layer]= (W.float() - W0[layer], fp16_rounding_fro(W))
    else:
        #dW = (alpha/r) B A from the exported fp32 factors (projected ones for oplora), exact
        tensors= load_file(os.path.join(export, "adapters.safetensors"))
        scale= info["alpha"] / info["rank"]
        for layer in layers:
            A= tensors[names[layer] + ".lora_A"].float()
            B= tensors[names[layer] + ".lora_B"].float()
            out[layer]= (scale * (B @ A), None)

    return out, info


def load_bases(cfg, args, api, W0, staging):
    #U_k per probed layer. real: the bases the oplora16 run trained with; smoke: svd of the random W0
    names= {layer: module_name(layer, cfg["module"]) for layer in cfg["layers"]}

    if cfg["smoke"]:
        return {layer: svd_bases(W0[layer], cfg["protect_k"])[0] for layer in cfg["layers"]}, {"k": cfg["protect_k"]}

    path= bases_path_for(cfg, args.output_dir)
    if not os.path.exists(path):
        assert api is not None, f"{path} is missing and there is no --sync_repo to pull it from"
        rel= "outputs/" + os.path.relpath(path, args.output_dir)
        assert pull_file(api, args.sync_repo, f"task1-{cfg['bases_run']}", rel, path, staging), f"download failed: {rel}"

    saved= torch.load(path, map_location="cpu")
    assert saved["model_id"] == cfg["model_name"], f"{path} was made for {saved['model_id']}"

    bases= {layer: saved["bases"][names[layer]][0].float() for layer in cfg["layers"]}

    #sanity: the saved U_k span the top-k left singular directions of the W0 used here (projector difference)
    check= {}
    for layer in cfg["layers"]:
        U= bases[layer]
        U_new= svd_bases(W0[layer], saved["k"])[0]
        check[layer]= float((U @ U.T - U_new @ U_new.T).abs().max())

    return bases, {"k": saved["k"], "path": path, "projector_max_abs_diff": check}


def run_diagnostics(cfg, args):
    device= torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)
    t0= time.time()

    api= make_api() if args.sync_repo else None
    staging= os.path.join(args.output_dir, "task1", "diag_staging")
    root= diag_root(args.results_dir, cfg["smoke"])
    os.makedirs(root, exist_ok=True)

    layers, module= cfg["layers"], cfg["module"]
    conditions= list(cfg["smoke_conditions"]) if cfg["smoke"] else [c for c, spec in cfg["conditions"].items() if spec is not None]

    #starting model in fp32 (handout: fp32 for the diagnostics); bf16 checkpoint values are exact in fp32
    tokenizer= AutoTokenizer.from_pretrained(cfg["model_name"])
    if cfg["smoke"]:
        model= smoke_model(load_config(BASE_CONFIG, EVAL_CONFIG, True), tokenizer)
    else:
        model= AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)
    model.eval()

    W0= {layer: model.get_submodule(module_name(layer, module)).weight.detach().float().cpu() for layer in layers}

    prompts= probe_prompts(cfg, args, api, staging)
    z, tokens= cache_activations(model, tokenizer, prompts, layers, module, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    #cached features are outputs, not results
    z_dir= os.path.join(args.output_dir, "task1", os.path.basename(root))
    os.makedirs(z_dir, exist_ok=True)
    torch.save({"z": z, "prompts": prompts}, os.path.join(z_dir, "z.pt"))

    bases, bases_info= load_bases(cfg, args, api, W0, staging)

    weight_rows, act_rows, summary_rows, sources= [], [], [], {}
    for condition in conditions:
        deltas, info= condition_deltas(cfg, condition, args, api, W0, staging)
        sources[condition]= info

        for layer in layers:
            dW, noise= deltas[layer]
            w0_fro= torch.linalg.norm(W0[layer])

            weight_rows.append({
                "condition": condition,
                "method": info["method"],
                "update": info.get("update"),
                "layer": layer,
                "module": module,
                "k": bases_info["k"],
                "rho_k": float(rho_k(dW, bases[layer])),
                "eta": float(eta(dW, W0[layer])),
                "delta_fro": float(torch.linalg.norm(dW)),
                "w0_fro": float(w0_fro),
                #eta of the fp16 storage noise alone (full-model exports only); compare with eta
                "fp16_noise_eta": None if noise is None else float(noise / w0_fro)
            })

            for name, spec in cfg["probe_sets"].items():
                R= activation_ratio(dW, W0[layer], z[name][layer])

                for (i, _), r in zip(prompts[name], R.tolist()):
                    act_rows.append({"condition": condition, "layer": layer, "set": name, "kind": spec["kind"], "probe_id": i, "R": r})

                summary_rows.append({
                    "condition": condition,
                    "layer": layer,
                    "set": name,
                    "kind": spec["kind"],
                    "n": len(R),
                    "mean": float(R.mean()),
                    "median": float(R.median()),
                    "std": float(R.std()) if len(R) > 1 else 0.0,
                    "min": float(R.min()),
                    "max": float(R.max())
                })

        print(condition, [(r["layer"], round(r["rho_k"], 4), round(r["eta"], 5)) for r in weight_rows if r["condition"] == condition])

    write_csv(os.path.join(root, "weight_metrics.csv"), weight_rows)
    write_csv(os.path.join(root, "activation_ratios.csv"), act_rows)
    write_csv(os.path.join(root, "activation_summary.csv"), summary_rows)
    write_csv(os.path.join(root, "probe_tokens.csv"), tokens)

    info= run_metadata(cfg, device)
    info["conditions"]= sources
    info["bases"]= bases_info
    info["probe_path"]= None if cfg["smoke"] else cfg["probe_path"]
    info["seconds"]= time.time() - t0
    write_json(os.path.join(root, INFO_FILE), info)
    print("saved", root, f"({info['seconds']:.0f}s)")

    if args.sync_repo:
        sync= start_sync(args.sync_repo, diag_branch(cfg["smoke"]), api=api)
        sync_put(sync, "upload_folder", root, "results/" + os.path.relpath(root, args.results_dir))
        if not sync_wait(sync):
            print("WARNING not everything was synced:", sync["failures"])
            return 1

    return 0


def main(args):
    if args.make_probe_ids:
        cfg= load_yaml(BASE_CONFIG)
        diag_cfg= load_yaml(DIAG_CONFIG)

        #frozen once written: never resample after results exist
        if os.path.exists(diag_cfg["probe_path"]):
            print("refusing to overwrite", diag_cfg["probe_path"])
            return 0

        os.makedirs(os.path.dirname(diag_cfg["probe_path"]), exist_ok=True)
        write_json(diag_cfg["probe_path"], make_probe_ids(cfg, diag_cfg))
        print("saved", diag_cfg["probe_path"])

        return 0

    cfg= load_config(BASE_CONFIG, DIAG_CONFIG, args.smoke)
    cfg["conditions"]= load_yaml(EVAL_CONFIG)["conditions"]

    return run_diagnostics(cfg, args)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--make_probe_ids",
        action="store_true"
    )
    parser.add_argument(
        "--smoke",
        action="store_true"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir()
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )
    parser.add_argument(
        "--sync_repo",
        type=str,
        default=None
    )

    raise SystemExit(main(parser.parse_args()))
