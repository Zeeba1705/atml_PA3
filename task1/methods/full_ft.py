import torch

from common.amp import scaler_step
from task1.losses import accumulate_sft_grads


def full_ft_train(model, params, microbatches, optimizer, scaler, device, use_fp16, max_grad_norm):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    loss, n_tokens= accumulate_sft_grads(
        model=model,
        microbatches=microbatches,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16
    )

    #unscale first so clipping sees the true gradient norm
    scaler.unscale_(optimizer)
    grad_norm= torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

    skipped= scaler_step(optimizer, scaler)

    return (loss, grad_norm, n_tokens, skipped)
