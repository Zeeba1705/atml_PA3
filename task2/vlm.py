import torch
import torch.nn.functional as F

IGNORE= -100


def prompt_segments(tokenizer, cfg):
    #qwen's own chat template with the image marker as the user message's first line, split at the marker:
    #"<|im_start|>user\n" [visual tokens] "\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    marker= cfg["image_marker"]
    text= tokenizer.apply_chat_template(
        [{"role": "user", "content": marker + "\nQUESTION"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg["enable_thinking"]
    )

    before, after= text.split(marker)
    sep, tail= after.split("QUESTION")

    def ids(s):
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    return {"before": ids(before), "sep": ids(sep), "tail": ids(tail), "eot": tokenizer.convert_tokens_to_ids("<|im_end|>")}


def build_example(segments, tokenizer, question, target, has_image, n_visual, max_question, max_target, supervise_eot):
    #token ids with a placeholder (pad) at visual positions, labels on target tokens (+ end-of-turn if supervised).
    #no image: the visual positions are simply absent, the text is unchanged
    q_ids= tokenizer(question, add_special_tokens=False)["input_ids"][:max_question]
    prompt= segments["before"] + [tokenizer.pad_token_id] * (n_visual if has_image else 0) + segments["sep"] + q_ids + segments["tail"]

    #target None = generation input (prompt only)
    answer, labels= [], [IGNORE] * len(prompt)
    if target is not None:
        t_ids= tokenizer(target, add_special_tokens=False)["input_ids"][:max_target]
        answer= t_ids + [segments["eot"]]

        #training supervises end-of-turn; answer nll (dev/eval) excludes it
        labels+= t_ids + [segments["eot"] if supervise_eot else IGNORE]

    is_visual= [False] * len(segments["before"]) + [True] * (n_visual if has_image else 0)
    is_visual+= [False] * (len(prompt) + len(answer) - len(is_visual))

    return {"input_ids": prompt + answer, "labels": labels, "is_visual": is_visual, "has_image": has_image}


def collate(examples, pad_id, left_pad):
    #pad to the longest; positions count only real tokens, so padding (either side) never shifts them
    width= max(len(e["input_ids"]) for e in examples)

    ids, labels, mask, vis= [], [], [], []
    for e in examples:
        n= width - len(e["input_ids"])
        pad= lambda x, v: [v] * n + x if left_pad else x + [v] * n

        ids.append(pad(e["input_ids"], pad_id))
        labels.append(pad(e["labels"], IGNORE))
        mask.append(pad([1] * len(e["input_ids"]), 0))
        vis.append(pad(e["is_visual"], False))

    mask= torch.tensor(mask)
    positions= (mask.cumsum(dim=1) - 1).clamp(min=0)

    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": mask,
        "position_ids": positions,
        "is_visual": torch.tensor(vis)
    }


def input_embeds(llm, connector, batch, feats):
    #text embeddings everywhere, then the connector's visual tokens written into the visual positions.
    #feats: (n_with_image, 196, 768), in the batch order of the examples that have an image
    embeds= llm.get_input_embeddings()(batch["input_ids"])

    if batch["is_visual"].any():
        visual= connector(feats).to(embeds.dtype)
        embeds= embeds.masked_scatter(batch["is_visual"].unsqueeze(-1), visual.reshape(-1, visual.shape[-1]))

    return embeds


def example_nll(llm, connector, batch, feats):
    #mean token nll within each example over its labelled tokens -> (B,). logits only at labelled positions
    embeds= input_embeds(llm, connector, batch, feats)

    hidden= llm.model(
        inputs_embeds=embeds,
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"]
    ).last_hidden_state

    #the token at position t is predicted from position t-1
    labels= batch["labels"][:, 1:]
    sel= labels != IGNORE
    logits= llm.lm_head(hidden[:, :-1][sel]).float()

    tok_nll= F.cross_entropy(logits, labels[sel], reduction="none")

    rows= sel.nonzero()[:, 0]
    per_ex= torch.zeros(labels.shape[0], device=tok_nll.device).index_add_(0, rows, tok_nll)
    counts= sel.sum(dim=1)

    return per_ex / counts.clamp(min=1)


def vlm_loss(llm, connector, batch, feats):
    #handout loss: average supervised-token nll within each example, then across the batch
    return example_nll(llm, connector, batch, feats).mean()
