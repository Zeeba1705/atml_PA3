import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.amp import scaler_step
from task1.losses import accumulate_sft_grads


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()

        self.base= base
        self.rank= rank
        self.scale= alpha / rank

        #adapters are fp32 even when the frozen backbone is fp16
        device= base.weight.device
        self.lora_A= nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=device))
        self.lora_B= nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=device))

        #random A (same init as PEFT), zero B so training starts from the base model
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout= nn.Dropout(dropout)

    def factors(self):
        return self.lora_A, self.lora_B

    def forward(self, x):
        A, B= self.factors()

        out= self.base(x)

        #x -> A (r features) -> B (out features), never forming the dense B @ A
        z= self.dropout(x).to(A.dtype)
        update= F.linear(F.linear(z, A), B) * self.scale

        return out + update.to(out.dtype)


def add_adapters(model, targets, make_layer, seed):
    #backbone frozen, only adapters train (no biases)
    for p in model.parameters():
        p.requires_grad= False

    names= [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and name.split(".")[-1] in targets
    ]

    #same seed + same module order -> identical A for lora16 and oplora16
    torch.manual_seed(seed)

    for name in names:
        parent_name, child_name= name.rsplit(".", 1)
        parent= model.get_submodule(parent_name)

        setattr(parent, child_name, make_layer(name, getattr(parent, child_name)))

    return names


def add_lora(model, targets, rank, alpha, dropout, seed):
    return add_adapters(
        model,
        targets,
        lambda name, base: LoRALinear(base, rank, alpha, dropout),
        seed
    )


def lora_train(model, params, microbatches, optimizer, scaler, device, use_fp16, max_grad_norm):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    loss, n_tokens= accumulate_sft_grads(
        model=model,
        microbatches=microbatches,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16
    )

    scaler.unscale_(optimizer)
    grad_norm= torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

    skipped= scaler_step(optimizer, scaler)

    return (loss, grad_norm, n_tokens, skipped)
