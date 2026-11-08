import json
import os
import shutil
import time

import torch
from safetensors.torch import save_file

from common.checkpoint import dir_bytes
from task1.methods.lora import LoRALinear


def merge_adapters(model):
    #W = W0 + (alpha/r) B A, using the projected factors for oplora (factors() already projects)
    layers= [(name, m) for name, m in model.named_modules() if isinstance(m, LoRALinear)]

    for name, m in layers:
        A, B= m.factors()

        with torch.no_grad():
            merged= m.base.weight.float() + m.scale * (B.float() @ A.float())
            m.base.weight.copy_(merged.to(m.base.weight.dtype))

        parent_name, child_name= name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, m.base)

    return model


def adapter_tensors(model):
    #effective factors: projected A~, B~ for oplora, plain A, B for lora
    tensors= {}

    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, LoRALinear):
                A, B= m.factors()
                tensors[name + ".lora_A"]= A.detach().float().cpu().contiguous()
                tensors[name + ".lora_B"]= B.detach().float().cpu().contiguous()

    return tensors


def export_weights(model, cfg, out_dir, update):
    #weights eval needs, without optimizer state: fp16 model for full ft, fp32 factors for adapters
    t0= time.time()

    tmp= out_dir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    if cfg["method"] in ["full_ft", "l2sp"]:
        state= {k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items()}

        #tied lm_head is the embedding, save it once
        if model.config.tie_word_embeddings:
            state.pop("lm_head.weight", None)

        model.save_pretrained(tmp, state_dict=state, safe_serialization=True)

        #config still says float32 (the training copy), but the saved weights are fp16 and eval loads them as such
        config_path= os.path.join(tmp, "config.json")
        with open(config_path) as f:
            config= json.load(f)

        config["dtype"]= "float16"
        config["use_cache"]= True

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        save_file(adapter_tensors(model), os.path.join(tmp, "adapters.safetensors"))

    info= {
        "method": cfg["method"],
        "update": update,
        "seconds": time.time() - t0,
        "bytes": dir_bytes(tmp)
    }
    if cfg["method"] in ["lora", "oplora"]:
        info["rank"]= cfg["rank"]
        info["alpha"]= cfg["alpha"]
        info["scale"]= cfg["alpha"] / cfg["rank"]
        info["targets"]= cfg["lora_targets"]
        info["projected"]= cfg["method"] == "oplora"

    with open(os.path.join(tmp, "export_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)

    return info
