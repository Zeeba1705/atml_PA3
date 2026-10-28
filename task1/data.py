import re
import unicodedata

import torch

END_OF_TURN= "<|im_end|>"


def normalize_question(text):
    #NFKC, lowercase, drop punctuation, collapse whitespace (docs/decisions.md)
    text= unicodedata.normalize("NFKC", text).lower()
    text= re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def build_example(tokenizer, question, response):
    #non-thinking generation prompt ends with an empty <think>\n\n</think>\n\n block, which counts as prompt
    prompt= tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    #template would add "\n" after <|im_end|>, we stop at the end-of-turn token
    full= prompt + response + END_OF_TURN

    prompt_ids= tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids= tokenizer(full, add_special_tokens=False)["input_ids"]

    #tokenizing the whole string must not merge tokens across the prompt/response boundary
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokens change when the response is appended")

    input_ids= torch.tensor(full_ids, dtype=torch.long)

    labels= input_ids.clone()
    labels[:len(prompt_ids)]= -100

    return {"input_ids": input_ids, "labels": labels, "length": len(full_ids)}


def collate(examples, pad_id):
    #dynamic padding: pad to the longest sequence in this microbatch, on the right
    max_len= max(len(ex["input_ids"]) for ex in examples)

    input_ids= torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels= torch.full((len(examples), max_len), -100, dtype=torch.long)
    attention_mask= torch.zeros((len(examples), max_len), dtype=torch.long)

    for i, ex in enumerate(examples):
        n= len(ex["input_ids"])

        input_ids[i, :n]= ex["input_ids"]
        labels[i, :n]= ex["labels"]
        attention_mask[i, :n]= 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
