import torch


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


def snapshot_params(params):
    #theta0: fixed fp32 copy, never trained
    return [p.detach().float().clone() for p in params]


def l2sp_omega(params, theta0):
    #omega = 0.5 * sum_j (theta_j - theta0_j)^2, summed over every entry, not averaged
    with torch.no_grad():
        diffs= torch._foreach_sub([p.float() for p in params], theta0)
        norms= torch._foreach_norm(diffs)

        return 0.5 * torch.stack(norms).pow(2).sum()


def add_l2sp_grad(params, theta0, lambda_l2sp):
    #gradient of lambda*omega is lambda*(theta - theta0); added after scaler.unscale_ so it never goes through fp16 scaling
    with torch.no_grad():
        diffs= torch._foreach_sub([p.float() for p in params], theta0)
        torch._foreach_add_([p.grad for p in params], diffs, alpha=lambda_l2sp)
