import torch

from common.amp import scaler_step
from task1.losses import accumulate_sft_grads

#params per foreach call: a whole-model theta - theta0 temp would be another 2.4 GB in fp32
CHUNK= 16


def unique_params(model):
    #tied weights (embed_tokens / lm_head) are one tensor, keep it once by storage pointer
    params= []
    seen= set()

    for p in model.parameters():
        if not p.requires_grad or p.data_ptr() in seen:
            continue

        seen.add(p.data_ptr())
        params.append(p)

    return params


def snapshot_params(params, device=None):
    #theta0: fixed fp32 copy, never trained. device=cpu keeps it off the gpu (pinned for fast copies)
    theta0= []

    for p in params:
        #copied straight to the target device: no temporary on the gpu, which may be nearly full
        t= p.detach().to(device=device or p.device, dtype=torch.float32, copy=True)

        if device is not None and device.type == "cpu" and torch.cuda.is_available():
            t= t.pin_memory()

        theta0.append(t)

    return theta0


def l2sp_omega(params, theta0):
    #omega = 0.5 * sum_j (theta_j - theta0_j)^2, summed over every entry, not averaged
    with torch.no_grad():
        total= torch.zeros((), device=params[0].device)

        for i in range(0, len(params), CHUNK):
            chunk= params[i:i + CHUNK]
            ref= [t.to(p.device, non_blocking=True) for t, p in zip(theta0[i:i + CHUNK], chunk)]

            diffs= torch._foreach_sub([p.float() for p in chunk], ref)
            total+= torch.stack(torch._foreach_norm(diffs)).pow(2).sum()

        return 0.5 * total


def add_l2sp_grad(params, theta0, lambda_l2sp):
    #grad += lambda*(theta - theta0), as two in-place adds so no full-size temp is needed
    #called after scaler.unscale_ so this large term never goes through fp16 loss scaling
    with torch.no_grad():
        for i in range(0, len(params), CHUNK):
            chunk= params[i:i + CHUNK]
            grads= [p.grad for p in chunk]
            ref= [t.to(p.device, non_blocking=True) for t, p in zip(theta0[i:i + CHUNK], chunk)]

            torch._foreach_add_(grads, chunk, alpha=lambda_l2sp)
            torch._foreach_add_(grads, ref, alpha=-lambda_l2sp)


def l2sp_train(model, params, l2sp_params, theta0, microbatches, optimizer, scaler, device, use_fp16, max_grad_norm,
               lambda_l2sp):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    loss, n_tokens= accumulate_sft_grads(
        model=model,
        microbatches=microbatches,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16
    )

    #omega at the params the loss was computed on, i.e. before this step
    omega= l2sp_omega(l2sp_params, theta0)

    scaler.unscale_(optimizer)
    add_l2sp_grad(l2sp_params, theta0, lambda_l2sp)

    #clip the total gradient (sft + penalty)
    grad_norm= torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

    skipped= scaler_step(optimizer, scaler)

    return (loss, grad_norm, n_tokens, skipped, omega)
