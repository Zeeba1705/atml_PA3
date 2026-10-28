import torch
import torch.nn.functional as F


def count_tokens(labels):
    #valid targets after the one-token shift
    return int((labels[:, 1:] != -100).sum())


def ce_sum(logits, labels):
    #reference version on full logits: token t predicts token t+1, summed (not averaged) over valid targets
    logits= logits[:, :-1]
    targets= labels[:, 1:]

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum"
    )


def supervised_ce_sum(model, input_ids, attention_mask, labels):
    hidden= model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state

    hidden= hidden[:, :-1]
    targets= labels[:, 1:]
    keep= targets != -100

    #lm head only where there is a target: vocab is ~152k so full logits would be huge
    logits= model.lm_head(hidden[keep])

    return F.cross_entropy(logits.float(), targets[keep], reduction="sum")


def accumulate_sft_grads(model, microbatches, scaler, device, use_fp16):
    #count tokens of the whole update first so every microbatch is divided by the same total
    n_tokens= sum(count_tokens(mb["labels"]) for mb in microbatches)

    total= torch.zeros((), device=device)

    for mb in microbatches:
        input_ids= mb["input_ids"].to(device, non_blocking=True)
        attention_mask= mb["attention_mask"].to(device, non_blocking=True)
        labels= mb["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            loss= supervised_ce_sum(model, input_ids, attention_mask, labels) / n_tokens

        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        total= total + loss.detach()

    return total, n_tokens
