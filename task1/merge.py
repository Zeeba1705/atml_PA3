import torch

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
