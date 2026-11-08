# Task 2 — results summary (SigLIP–Qwen VLM)

Numbers only, generated from saved result files (source named under each table). No interpretation.

## Setup

- Vision `google/siglip-base-patch16-224` (frozen, 196 patch features, cached FP16); connector Linear 768→1024, GELU, Linear 1024→1024 (FP32); `Qwen/Qwen3-0.6B` frozen FP16, thinking off; LoRA r16/α32/dropout 0.05 on q,k,v,o in stages 2–3.
- Curriculum: stage 1 captions 2,250 updates (connector lr 1e-3); stage 2 presence/counting/caption 0.5/0.25/0.25, 1,500 updates (connector 1e-4, LoRA 2e-5); stage 3 VQA/presence/counting/caption 0.7/0.1/0.1/0.1, 750 updates (5e-5, 1e-5). Micro-batch 8 × accumulation 2; seed 6304.
- Final evaluation: greedy, 16 new tokens, batch 8; grounding = normalized exact match; VQAv2 = local soft accuracy over 10 answers; 95% CI from 4,000 pair bootstrap resamples.

## Data (selected with seed 6304)

| set | size |
|---|---|
| captions_train | 18000 |
| vqa_train | 1536 |
| presence_pool_train | 8000 |
| counting_pool_train | 8000 |
| vqa_dev | 32 |
| grounding_dev | 24 |
| vqa_test | 96 |
| grounding_test | 64 |
| captions_dev | 96 |

Images: 8698 train2014, 528 val2014 (all evaluation sets image-disjoint). VQA pair types: {"vqa_train": {"yes/no": 640, "number": 256, "other": 640}, "vqa_dev": {"yes/no": 12, "number": 8, "other": 12}, "vqa_test": {"yes/no": 36, "number": 24, "other": 36}}. Grounding: {"grounding_dev": {"presence": 12, "counting": 12}, "grounding_test": {"presence": 32, "counting": 32}}.

Source: `splits/task2/summary.json`.

## Development metrics (learning curve points)

| update | model_stage | caption_loss | vqa_nll_correct | vqa_nll_complementary | vqa_nll_no_image | vqa_delta_nll | grounding_nll_correct | grounding_nll_complementary | grounding_nll_no_image | grounding_delta_nll |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | init | 5.4421 | 13.4093 | 13.1587 | 13.2286 | -0.2507 | 11.9809 | 12.0444 | 11.0919 | 0.0634 |
| 500 | stage1 | 2.5154 | 9.16 | 9.2108 | 13.2286 | 0.0509 | 8.4408 | 8.3768 | 11.0919 | -0.064 |
| 1000 | stage1 | 2.383 | 9.0427 | 9.1482 | 13.2286 | 0.1055 | 8.8176 | 8.7578 | 11.0919 | -0.0597 |
| 1500 | stage1 | 2.3155 | 9.2601 | 9.4381 | 13.2286 | 0.178 | 9.6991 | 9.6272 | 11.0919 | -0.0719 |
| 2000 | stage1 | 2.2997 | 9.2772 | 9.3956 | 13.2286 | 0.1184 | 9.5554 | 9.5231 | 11.0919 | -0.0323 |
| 2250 | stage1 | 2.2938 | 9.3198 | 9.4273 | 13.2286 | 0.1075 | 9.7404 | 9.71 | 11.0919 | -0.0304 |
| 2500 | stage2 | 2.3111 | 1.9235 | 2.8061 | 11.2239 | 0.8826 | 1.0141 | 2.9282 | 7.1392 | 1.9141 |
| 3000 | stage2 | 2.2963 | 1.9726 | 2.9558 | 11.0966 | 0.9832 | 0.9408 | 3.0723 | 6.5918 | 2.1315 |
| 3500 | stage2 | 2.2882 | 2.131 | 3.3125 | 10.9823 | 1.1815 | 1.137 | 3.7366 | 6.2811 | 2.5996 |
| 3750 | stage2 | 2.2867 | 2.1647 | 3.4063 | 11.0766 | 1.2415 | 1.1934 | 3.8529 | 6.3557 | 2.6595 |
| 4000 | stage3 | 2.2829 | 1.5412 | 2.632 | 8.4615 | 1.0909 | 1.0104 | 2.5595 | 4.2204 | 1.5492 |
| 4500 | stage3 | 2.2908 | 1.5753 | 3.0342 | 8.2389 | 1.4588 | 1.0373 | 3.01 | 3.9259 | 1.9727 |

Rows are the model after that many updates (`model_stage`). NLL excludes the end-of-turn token; ΔNLL = NLL(partner image) − NLL(correct image). Source: `results/task2/dev_metrics.csv`; figure `results/task2/figures/learning_curves.png`.

## Final test — VQAv2 test (96 pairs / 192 questions)

| condition | accuracy | pair_accuracy | identical_answer_rate | empty_answer_rate | follows_partner_rate | mean_reference_nll | gen_seconds_per_answer |
|---|---|---|---|---|---|---|---|
| correct | 0.6281 | 0.3333 | 0.4896 | 0.0 | 0.2083 | 1.2172 | 0.0371 |
| complementary | 0.2599 | 0.0 | 0.4896 | 0.0 | 0.5938 | 2.4347 | 0.038 |
| mismatched | 0.3396 | 0.0312 | 0.7083 | 0.0 | 0.3542 | 2.1779 | 0.0397 |
| no_image | 0.0052 | 0.0 | 1.0 | 0.0 | 0.0052 | 7.7038 | 0.1221 |

Accuracy by question type:

| condition | number | other | yes/no |
|---|---|---|---|
| correct | 0.6792 | 0.4819 | 0.7403 |
| complementary | 0.0813 | 0.2375 | 0.4014 |
| mismatched | 0.1583 | 0.2931 | 0.5069 |
| no_image | 0.0208 | 0.0 | 0.0 |

## Final test — Grounding test (64 pairs / 128 questions)

| condition | accuracy | pair_accuracy | identical_answer_rate | empty_answer_rate | follows_partner_rate | mean_reference_nll | gen_seconds_per_answer |
|---|---|---|---|---|---|---|---|
| correct | 0.6875 | 0.5156 | 0.1406 | 0.0 | 0.1094 | 0.7349 | 0.0344 |
| complementary | 0.1094 | 0.0312 | 0.1406 | 0.0 | 0.6875 | 2.9729 | 0.0354 |
| mismatched | 0.3906 | 0.0781 | 0.6094 | 0.0 | 0.3672 | 1.8196 | 0.0362 |
| no_image | 0.0547 | 0.0 | 1.0 | 0.0 | 0.0547 | 3.6497 | 0.1084 |

Accuracy by question type:

| condition | counting | presence |
|---|---|---|
| correct | 0.4375 | 0.9375 |
| complementary | 0.1562 | 0.0625 |
| mismatched | 0.25 | 0.5312 |
| no_image | 0.1094 | 0.0 |

`follows_partner_rate` = prediction matches the partner image's reference. Source: `results/task2/final_results.csv`; figure `results/task2/figures/final_accuracy_by_condition.png`.

## Correct-image advantages (mean per pair, 95% pair-bootstrap CI)

| suite | metric | comparison | mean | ci95_lo | ci95_hi |
|---|---|---|---|---|---|
| vqa | accuracy | correct | 0.6281 | 0.5615 | 0.6937 |
| vqa | accuracy | correct_minus_complementary | 0.3682 | 0.2781 | 0.4568 |
| vqa | accuracy | correct_minus_mismatched | 0.2885 | 0.2068 | 0.3708 |
| vqa | accuracy | correct_minus_no_image | 0.6229 | 0.5563 | 0.6891 |
| vqa | nll | correct | 1.2172 | 0.975 | 1.4748 |
| vqa | nll | complementary_minus_correct | 1.2175 | 0.8926 | 1.5826 |
| vqa | nll | mismatched_minus_correct | 0.9607 | 0.6476 | 1.3215 |
| vqa | nll | no_image_minus_correct | 6.4866 | 5.8835 | 7.0965 |
| grounding | accuracy | correct | 0.6875 | 0.6016 | 0.7734 |
| grounding | accuracy | correct_minus_complementary | 0.5781 | 0.4453 | 0.7031 |
| grounding | accuracy | correct_minus_mismatched | 0.2969 | 0.2031 | 0.3828 |
| grounding | accuracy | correct_minus_no_image | 0.6328 | 0.5312 | 0.7344 |
| grounding | nll | correct | 0.7349 | 0.5654 | 0.9144 |
| grounding | nll | complementary_minus_correct | 2.2379 | 1.7717 | 2.7101 |
| grounding | nll | mismatched_minus_correct | 1.0847 | 0.8322 | 1.3539 |
| grounding | nll | no_image_minus_correct | 2.9148 | 2.5906 | 3.2497 |

Accuracy: correct − control; NLL: control − correct. Source: `results/task2/advantages.csv`.

## Resources

| item | value |
|---|---|
| trainable params, stage 1 (connector) | 1837056 |
| trainable params, stages 2-3 (connector + LoRA) | 6424576 |
| training time (h), all 4,500 updates incl. dev evals | 1.718 |
| training peak GPU memory (GB) | 1.67 |
| training GPU | Tesla T4 |
| eval peak GPU memory (GB) | 1.6 |
| generation time per answer, vqa / grounding (s, batch 8) | 0.0371 / 0.0344 |
| feature caching (images / seconds / device) | 9226 / 3360 / cpu (kaggle session without accelerator) |
| connector output scale (calibration) | 0.05499 |
| draws per component | {"stage1/caption": 18000, "stage2/caption": 2973, "stage2/counting": 3027, "stage2/presence": 6000, "stage3/presence": 563, "stage3/vqa": 4223, "stage3/caption": 587, "stage3/counting": 627} |

Source: `results/task2/resources.csv`.

## Example lists (fixed rules, for you to choose from)

| suite | rule | pairs |
|---|---|---|
| vqa | both_right | 32 |
| vqa | follows_partner_when_swapped | 32 |
| vqa | same_answer_both_images | 47 |
| grounding | both_right | 33 |
| grounding | follows_partner_when_swapped | 33 |
| grounding | same_answer_both_images | 9 |

Each entry has the question, both references and the predictions under correct and swapped images. Source: `results/task2/main/eval/examples.json`; all generations `results/task2/main/eval/generations_<suite>_<condition>.jsonl` (on Hugging Face branch `task2-main`).
