# Task 1 — results summary (LLM fine-tuning)

Numbers only, generated from saved result files (source named under each table). No interpretation.

## Setup

- Model `Qwen/Qwen3-0.6B`; 40,000 MetaMathQA examples; 1,250 optimizer updates (micro-batch 2 × accumulation 16 = 32); seed 6304; FP16 autocast on T4.
- Methods: Full FT (lr 2e-5), Full FT + L2-SP (λ = 0.001, lr 2e-5), LoRA r4/α8 and r16/α32 (lr 2e-4), OPLoRA r16/α32, k = 128 (lr 2e-4); LoRA targets q,k,v,o,gate,up,down.
- Evaluation: lm-eval 0.4.13, zero-shot, FP16, full splits: GSM8K (gsm8k_cot, chat template, thinking off, greedy, 1,024 new tokens, flexible-extract), HellaSwag (acc_norm), BoolQ (acc), ARC-Easy (acc_norm).
- Conditions: baseline, Full FT at update 625 (midpoint), and all five methods at update 1,250.

## Benchmarks (score % ± stderr; Δ = percentage points vs baseline)

| condition | gsm8k (%) | gsm8k Δpp | hellaswag (%) | hellaswag Δpp | boolq (%) | boolq Δpp | arc_easy (%) | arc_easy Δpp |
|---|---|---|---|---|---|---|---|---|
| Baseline | 43.14 ± 1.36 | +0.00 | 47.27 ± 0.50 | +0.00 | 64.43 ± 0.84 | +0.00 | 56.02 ± 1.02 | +0.00 |
| Full FT (625) | 56.41 ± 1.37 | +13.27 | 45.83 ± 0.50 | -1.44 | 61.38 ± 0.85 | -3.06 | 62.21 ± 0.99 | +6.19 |
| Full FT | 64.90 ± 1.31 | +21.76 | 46.01 ± 0.50 | -1.26 | 61.25 ± 0.85 | -3.18 | 63.68 ± 0.99 | +7.66 |
| Full FT + L2-SP | 64.29 ± 1.32 | +21.15 | 46.51 ± 0.50 | -0.76 | 66.36 ± 0.83 | +1.93 | 63.43 ± 0.99 | +7.41 |
| LoRA-4 | 59.82 ± 1.35 | +16.68 | 47.32 ± 0.50 | +0.05 | 69.11 ± 0.81 | +4.68 | 63.59 ± 0.99 | +7.58 |
| LoRA-16 | 62.02 ± 1.34 | +18.88 | 47.13 ± 0.50 | -0.14 | 71.19 ± 0.79 | +6.76 | 63.26 ± 0.99 | +7.24 |
| OPLoRA-16 | 61.56 ± 1.34 | +18.42 | 47.20 ± 0.50 | -0.07 | 70.58 ± 0.80 | +6.15 | 62.96 ± 0.99 | +6.94 |

Source: `results/task1/summary_wide.csv` / `summary.csv`.

### Raw accuracy (no length normalization)

| condition | benchmark | raw acc % | Δ pp |
|---|---|---|---|
| baseline | hellaswag | 37.55 | 0.0 |
| baseline | arc_easy | 60.82 | 0.0 |
| full_ft_mid | hellaswag | 37.14 | -0.41 |
| full_ft_mid | arc_easy | 65.87 | 5.05 |
| full_ft | hellaswag | 37.33 | -0.22 |
| full_ft | arc_easy | 66.46 | 5.64 |
| l2sp | hellaswag | 37.49 | -0.06 |
| l2sp | arc_easy | 66.37 | 5.56 |
| lora4 | hellaswag | 37.81 | 0.26 |
| lora4 | arc_easy | 65.61 | 4.8 |
| lora16 | hellaswag | 37.89 | 0.34 |
| lora16 | arc_easy | 65.78 | 4.97 |
| oplora16 | hellaswag | 37.7 | 0.15 |
| oplora16 | arc_easy | 65.7 | 4.88 |

### GSM8K generation (1,319 questions)

| condition | truncated at 1,024 | no number extracted |
|---|---|---|
| baseline | 10 | 0 |
| full_ft_mid | 14 | 0 |
| full_ft | 5 | 0 |
| l2sp | 12 | 0 |
| lora4 | 9 | 1 |
| lora16 | 8 | 0 |
| oplora16 | 4 | 1 |

Source: `results/task1/summary.csv`, per-question table `results/task1/eval/<condition>/gsm8k_table.jsonl`.

## Resources

| run | method | trainable_params | trainable_pct | train_hours | checkpoint_hours | peak_vram_gb | checkpoint_trainable_mb | checkpoint_total_mb | export_mb | svd_seconds | bases_mb | theta0_mb | skipped_updates | sessions | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_ft | full_ft | 596049920 | 100.0 | 2.322 | 0.24 | 12.55 | 2384.3 | 7153.0 | 1192.1 |  |  | 2384.2 | 0 | 1 | Tesla T4 |
| l2sp | l2sp | 596049920 | 100.0 | 2.451 | 0.239 | 12.55 | 2384.3 | 7153.0 | 1192.1 |  |  | 2384.2 | 0 | 1 | Tesla T4 |
| lora4 | lora | 2523136 | 0.423 | 2.634 | 0.003 | 3.37 | 10.2 | 30.7 | 10.1 |  |  |  | 0 | 1 | Tesla T4 |
| lora16 | lora | 10092544 | 1.693 | 2.727 | 0.008 | 3.5 | 40.5 | 121.5 | 40.4 |  |  |  | 0 | 1 | Tesla T4 |
| oplora16 | oplora | 10092544 | 1.693 | 3.341 | 0.008 | 3.82 | 40.5 | 121.5 | 40.4 | 31.5 | 323.0 |  | 0 | 1 | Tesla T4 |

Source: `results/task1/resources.csv` (from each run's `train_summary.json`).

## L2-SP penalty and displacement

| run | λ | displacement ‖θ−θ0‖ at 625 | at 1,250 | final Ω | final λΩ | final SFT loss |
|---|---|---|---|---|---|---|
| full_ft | 0.0 | 10.023 | 10.660 |  |  | 0.1645 |
| l2sp | 0.001 | 6.436 | 6.271 | 19.66 | 0.0197 | 0.1646 |

Source: `results/task1/{full_ft,l2sp}/train_summary.json`, `train_log.jsonl`. Figures: `figures/sft_loss.png`, `figures/l2sp_terms.png`.

## Weight-space and activation diagnostics (down_proj, layers 0/13/27, FP32)

| condition | layer | k | rho_k | eta | fp16_noise_eta | R_math_mean | R_math_median | R_math_std | R_non_math_mean | R_non_math_median | R_non_math_std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_ft_mid | 0 | 128 | 0.1336 | 0.01322 | 0.00021 | 0.05155 | 0.05151 | 0.00034 | 0.02483 | 0.02542 | 0.00302 |
| full_ft_mid | 13 | 128 | 0.1484 | 0.01541 | 0.00021 | 0.05489 | 0.05485 | 0.00092 | 0.03059 | 0.02977 | 0.00302 |
| full_ft_mid | 27 | 128 | 0.116 | 0.01246 | 0.00021 | 0.02011 | 0.01971 | 0.00177 | 0.02859 | 0.02848 | 0.00167 |
| full_ft | 0 | 128 | 0.1333 | 0.01365 | 0.00021 | 0.05306 | 0.053 | 0.00035 | 0.02548 | 0.02606 | 0.00309 |
| full_ft | 13 | 128 | 0.1483 | 0.01604 | 0.00021 | 0.05533 | 0.05527 | 0.00094 | 0.03116 | 0.03037 | 0.00303 |
| full_ft | 27 | 128 | 0.1154 | 0.01295 | 0.00021 | 0.01998 | 0.01953 | 0.00179 | 0.02884 | 0.02865 | 0.00192 |
| l2sp | 0 | 128 | 0.1313 | 0.01121 | 0.00021 | 0.04523 | 0.04523 | 0.00031 | 0.02235 | 0.0231 | 0.00291 |
| l2sp | 13 | 128 | 0.1495 | 0.01369 | 0.00021 | 0.04593 | 0.04588 | 0.00086 | 0.02813 | 0.02735 | 0.0028 |
| l2sp | 27 | 128 | 0.1067 | 0.00787 | 0.00021 | 0.01888 | 0.01863 | 0.00149 | 0.02568 | 0.02566 | 0.00126 |
| lora4 | 0 | 128 | 0.1372 | 0.0084 |  | 0.05459 | 0.05465 | 0.00045 | 0.01139 | 0.0126 | 0.00338 |
| lora4 | 13 | 128 | 0.1341 | 0.01311 |  | 0.08592 | 0.08619 | 0.00194 | 0.02145 | 0.02125 | 0.00685 |
| lora4 | 27 | 128 | 0.0988 | 0.00679 |  | 0.00513 | 0.005 | 0.00114 | 0.00596 | 0.00592 | 0.00121 |
| lora16 | 0 | 128 | 0.1303 | 0.01412 |  | 0.05962 | 0.05961 | 0.00072 | 0.01919 | 0.02106 | 0.00435 |
| lora16 | 13 | 128 | 0.151 | 0.0192 |  | 0.08211 | 0.08208 | 0.00151 | 0.02769 | 0.0274 | 0.00579 |
| lora16 | 27 | 128 | 0.1068 | 0.01137 |  | 0.01177 | 0.01165 | 0.00082 | 0.01305 | 0.01316 | 0.0023 |
| oplora16 | 0 | 128 | 0.0 | 0.01445 |  | 0.07531 | 0.0751 | 0.00093 | 0.01921 | 0.02041 | 0.0047 |
| oplora16 | 13 | 128 | 0.0 | 0.01926 |  | 0.09082 | 0.09087 | 0.00156 | 0.02842 | 0.02757 | 0.0059 |
| oplora16 | 27 | 128 | 0.0 | 0.00916 |  | 0.01192 | 0.01176 | 0.0009 | 0.01225 | 0.01211 | 0.00177 |

ρ_k with U_k = top-128 left singular vectors of W0 (the OPLoRA bases); η = ‖ΔW‖/‖W0‖; R(z) on 32 GSM8K (math) and 32 HellaSwag (non-math) probe prompts. `fp16_noise_eta` = expected η of FP16 storage rounding alone (Full FT/L2-SP exports are FP16).

Source: `results/task1/diagnostics.csv`, per-probe values `results/task1/diagnostics/`.

## Qualitative generations (24 prompts × 7 conditions, greedy, 512 new tokens)

| condition | responses | truncated (512) | mean tokens |
|---|---|---|---|
| baseline | 24 | 1 | 101 |
| full_ft_mid | 24 | 3 | 164 |
| full_ft | 24 | 4 | 198 |
| l2sp | 24 | 4 | 176 |
| lora4 | 24 | 2 | 118 |
| lora16 | 24 | 3 | 145 |
| oplora16 | 24 | 1 | 132 |

Annotation subset: `what_is_your_wish` and `enough_of_my_husband`, plain/JSON/template × 7 conditions = 42 rows in `results/task1/annotation_sheet.csv` (your annotations). All responses grouped by question: `results/task1/qualitative/responses_by_question.md`.

## Figures

- `results/task1/figures/sft_loss.png` — training loss, all five runs (whole run + zoom)
- `results/task1/figures/l2sp_terms.png` — L2-SP SFT/total loss, λΩ, Ω
- `results/task1/figures/benchmark_scores.png` — scores with 95% CI, dashed = baseline

## Step 7 annotation tallies (42 responses)

Counts of the hand annotations in `annotation_sheet.csv`, per condition (6 responses each: 2 questions x 3 formats). Format compliance counts the 4 json/template rows only (plain = N/A).

| condition | format yes | format partial | format no | coherent_relevant yes | repetition yes | unnecessary math framing yes | truncated |
|---|---|---|---|---|---|---|---|
| baseline | 4 | 0 | 0 | 3/6 | 1/6 | 0/6 | 0/6 |
| full_ft_mid | 3 | 1 | 0 | 3/6 | 1/6 | 3/6 | 1/6 |
| full_ft | 3 | 1 | 0 | 3/6 | 2/6 | 4/6 | 1/6 |
| l2sp | 3 | 1 | 0 | 3/6 | 1/6 | 3/6 | 1/6 |
| lora4 | 3 | 0 | 1 | 5/6 | 1/6 | 0/6 | 0/6 |
| lora16 | 3 | 0 | 1 | 5/6 | 2/6 | 1/6 | 2/6 |
| oplora16 | 3 | 1 | 0 | 3/6 | 0/6 | 2/6 | 0/6 |
| all | 22 | 4 | 2 | 25/42 | 8/42 | 13/42 | 5/42 |

By format:

| format | coherent_relevant yes | repetition yes | unnecessary math framing yes | truncated |
|---|---|---|---|---|
| plain | 7/14 | 4/14 | 5/14 | 4/14 |
| json | 9/14 | 2/14 | 4/14 | 1/14 |
| template | 9/14 | 2/14 | 4/14 | 0/14 |

inappropriate_content: no on all 42 rows. Source: `results/task1/annotation_sheet.csv`.
