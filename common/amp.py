import torch


def make_scaler(use_fp16):
    #disabled scaler is a pass-through (cpu / fp32 smoke runs), so training code is the same either way
    return torch.amp.GradScaler("cuda", enabled=use_fp16)


def scaler_step(optimizer, scaler):
    #the scaler skips the step when it finds inf/nan grads and then lowers its scale, that's how we detect a skip
    scale_before= scaler.get_scale()

    scaler.step(optimizer)
    scaler.update()

    return scaler.get_scale() < scale_before
