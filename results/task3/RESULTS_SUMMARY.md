# Task 3 — results summary (test-time compute)

Numbers only, generated from saved result files (source named under each table). No interpretation.

## Setup

- Supplied rollouts: MATH (Qwen2.5-3B-Instruct) and MathVista testmini (Qwen2.5-VL-3B-Instruct), 1,000 questions × 32 rollouts each.
- Splits (seed 6304): 200 dev / 800 held-out per dataset; all tables below are held-out unless stated. 10 seeded rollout orders per question, nested prefixes, N ∈ {1, 2, 4, 8, 16, 32}; averages over orders within a question, then over questions.
- Answers: math-verify 0.9.0 (MATH); MathVista official type/precision/choice rules after unit stripping; missing extraction = NO_ANSWER (always wrong).

## Answer groups

| dataset | rollout accuracy | NO_ANSWER rollouts | questions with no correct rollout |
|---|---|---|---|
| math | 0.6632 | 521 | 81 |
| mathvista | 0.5422 | 4278 | 0 |

Source: `results/task3/answer_groups_<dataset>.json`.

## Methods at N = 32 — math

| method | accuracy | Δ vs SC (pp) | CI lo | CI hi | mean tokens |
|---|---|---|---|---|---|
| first_ans | 0.6435 | -9.44 | -11.01 | -7.98 | 669.8 |
| max_logprob | 0.64 | -9.79 | -12.39 | -7.4 | 21347.0 |
| max_self_certainty | 0.7063 | -3.16 | -5.51 | -0.83 | 21347.0 |
| self_certainty_rwc | 0.7559 | 1.8 | 0.72 | 2.96 | 21347.0 |
| self_consistency | 0.7379 | 0.0 | 0.0 | 0.0 | 21347.0 |
| shortest | 0.6238 | -11.41 | -14.13 | -8.99 | 21347.0 |

Oracle coverage 0.9137; distinct answer groups 5.905. CI: 1,000 paired question-level bootstrap resamples.

### Accuracy vs N — math

| N | coverage | distinct groups | SC tokens | first_ans | self_consistency | max_logprob | max_self_certainty | self_certainty_rwc | shortest |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.6435 | 1.0 | 669.8 | 0.6435 | 0.6435 | 0.6435 | 0.6435 | 0.6435 | 0.6435 |
| 2 | 0.7249 | 1.371 | 1339.9 | 0.6435 | 0.6435 | 0.658 | 0.6676 | 0.6676 | 0.6529 |
| 4 | 0.7908 | 1.928 | 2679.7 | 0.6435 | 0.692 | 0.6724 | 0.6878 | 0.697 | 0.652 |
| 8 | 0.8435 | 2.741 | 5344.1 | 0.6435 | 0.7179 | 0.6754 | 0.7014 | 0.7289 | 0.6456 |
| 16 | 0.8845 | 4.004 | 10677.7 | 0.6435 | 0.7311 | 0.6639 | 0.707 | 0.7464 | 0.6316 |
| 32 | 0.9137 | 5.905 | 21347.0 | 0.6435 | 0.7379 | 0.64 | 0.7063 | 0.7559 | 0.6238 |

### Failure decomposition — math (fraction of question-orders)

| N | method | no correct candidate | correct available but missed |
|---|---|---|---|
| 1 | first_ans | 0.3565 | 0.0 |
| 2 | first_ans | 0.2751 | 0.0814 |
| 4 | first_ans | 0.2093 | 0.1473 |
| 8 | first_ans | 0.1565 | 0.2 |
| 16 | first_ans | 0.1155 | 0.241 |
| 32 | first_ans | 0.0862 | 0.2702 |
| 1 | self_consistency | 0.3565 | 0.0 |
| 2 | self_consistency | 0.2751 | 0.0814 |
| 4 | self_consistency | 0.2093 | 0.0988 |
| 8 | self_consistency | 0.1565 | 0.1256 |
| 16 | self_consistency | 0.1155 | 0.1534 |
| 32 | self_consistency | 0.0862 | 0.1759 |

## Methods at N = 32 — mathvista

| method | accuracy | Δ vs SC (pp) | CI lo | CI hi | mean tokens |
|---|---|---|---|---|---|
| first_ans | 0.5419 | -8.34 | -10.04 | -6.51 | 277.8 |
| max_logprob | 0.49 | -13.53 | -16.55 | -10.46 | 8897.8 |
| max_self_certainty | 0.5563 | -6.9 | -9.75 | -3.72 | 8897.8 |
| max_visconf | 0.5813 | -4.4 | -7.01 | -1.54 | 8897.8 |
| max_visconf_token | 0.5675 | -5.77 | -8.35 | -2.66 | 8897.8 |
| max_vision | 0.5413 | -8.4 | -11.45 | -5.24 | 8897.8 |
| self_certainty_rwc | 0.6362 | 1.1 | -0.2 | 2.52 | 8897.8 |
| self_consistency | 0.6252 | 0.0 | 0.0 | 0.0 | 8897.8 |
| shortest | 0.541 | -8.43 | -11.45 | -5.54 | 8897.8 |
| visconf_rwc | 0.6412 | 1.6 | 0.21 | 3.13 | 8897.8 |
| visconf_token_rwc | 0.6362 | 1.1 | -0.28 | 2.58 | 8897.8 |
| vision_rwc | 0.6362 | 1.1 | -0.49 | 2.7 | 8897.8 |

Oracle coverage 0.8988; distinct answer groups 4.104. CI: 1,000 paired question-level bootstrap resamples.

### Accuracy vs N — mathvista

| N | coverage | distinct groups | SC tokens | first_ans | self_consistency | max_logprob | max_self_certainty | self_certainty_rwc | max_vision | vision_rwc | max_visconf | visconf_rwc | max_visconf_token | visconf_token_rwc | shortest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5419 | 1.0 | 277.8 | 0.5419 | 0.5419 | 0.5419 | 0.5419 | 0.5419 | 0.5419 | 0.5419 | 0.5419 | 0.5419 | 0.5419 | 0.5419 | 0.5419 |
| 2 | 0.6664 | 1.423 | 551.3 | 0.5419 | 0.5419 | 0.5501 | 0.5544 | 0.5544 | 0.5493 | 0.5493 | 0.5595 | 0.5595 | 0.5599 | 0.5599 | 0.5547 |
| 4 | 0.7632 | 1.937 | 1111.6 | 0.5419 | 0.5853 | 0.541 | 0.5575 | 0.5747 | 0.55 | 0.5703 | 0.5687 | 0.5829 | 0.564 | 0.5809 | 0.5547 |
| 8 | 0.8315 | 2.548 | 2229.9 | 0.5419 | 0.6059 | 0.5274 | 0.5561 | 0.6076 | 0.5484 | 0.6066 | 0.568 | 0.6144 | 0.5619 | 0.6158 | 0.5491 |
| 16 | 0.8755 | 3.265 | 4461.8 | 0.5419 | 0.6168 | 0.5139 | 0.5591 | 0.6209 | 0.5451 | 0.6199 | 0.5732 | 0.6258 | 0.5624 | 0.6266 | 0.546 |
| 32 | 0.8988 | 4.104 | 8897.8 | 0.5419 | 0.6252 | 0.49 | 0.5563 | 0.6362 | 0.5413 | 0.6362 | 0.5813 | 0.6412 | 0.5675 | 0.6362 | 0.541 |

### Failure decomposition — mathvista (fraction of question-orders)

| N | method | no correct candidate | correct available but missed |
|---|---|---|---|
| 1 | first_ans | 0.4581 | 0.0 |
| 2 | first_ans | 0.3336 | 0.1245 |
| 4 | first_ans | 0.2367 | 0.2214 |
| 8 | first_ans | 0.1685 | 0.2896 |
| 16 | first_ans | 0.1245 | 0.3336 |
| 32 | first_ans | 0.1013 | 0.3569 |
| 1 | self_consistency | 0.4581 | 0.0 |
| 2 | self_consistency | 0.3336 | 0.1245 |
| 4 | self_consistency | 0.2367 | 0.178 |
| 8 | self_consistency | 0.1685 | 0.2256 |
| 16 | self_consistency | 0.1245 | 0.2587 |
| 32 | self_consistency | 0.1013 | 0.2735 |

Source: `results/task3/summary.csv`, `methods_n32.csv`; per-question `results/task3/offline/<dataset>_per_question.parquet`. Figures: `figures/accuracy_vs_n.png`, `figures/accuracy_vs_tokens.png`.

## Signal analysis (N = 32, held-out)

| dataset | score | top−bottom quartile correct (all q) | (mixed q) | # mixed q | Spearman score–correct | Spearman length–score |
|---|---|---|---|---|---|---|
| math | logprob | 0.0783 | 0.1335 | 469 | 0.134 | 0.2245 |
| math | self_certainty | 0.1198 | 0.2044 | 469 | 0.198 | -0.1611 |
| math | neg_length | 0.0456 | 0.0778 | 469 | 0.0763 |  |
| mathvista | logprob | 0.0195 | 0.0246 | 635 | 0.0339 | 0.329 |
| mathvista | self_certainty | 0.0362 | 0.0457 | 635 | 0.0497 | 0.3493 |
| mathvista | neg_length | 0.0369 | 0.0465 | 635 | 0.0444 |  |
| mathvista | vision | 0.0353 | 0.0445 | 635 | 0.0371 | -0.6858 |
| mathvista | visconf | 0.072 | 0.0907 | 635 | 0.0871 | -0.275 |
| mathvista | visconf_token | 0.0706 | 0.089 | 635 | 0.078 | -0.3218 |

`neg_length` = shortest-response control. Spearman length–correct (mixed questions): MATH -0.0763, MathVista -0.0444.

MathVista within-rollout confidence–vision correlation (per-question mean over rollouts): mean -0.2110, median -0.2115, IQR [-0.3055, -0.0939], fraction of questions > 0: 0.1062. Check Z_token − Z = Cov_t(c, v): max error 0.0.

Source: `results/task3/signals.csv`, `signals.json` (also has the fixed-rule illustrative trajectories).

## Adaptive stopping (settings chosen on dev, frozen, then held-out)

| dataset | policy | tau | gamma | dev_accuracy | accuracy | mean_tokens | mean_rollouts | stop_4 | stop_8 | stop_16 | stop_32 | incorrect_early_stops | recoverable_by_32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| math | fixed SC N=1 |  |  |  | 0.6435 | 669.8 | 1 |  |  |  |  |  |  |
| math | fixed SC N=2 |  |  |  | 0.6435 | 1339.9 | 2 |  |  |  |  |  |  |
| math | fixed SC N=4 |  |  |  | 0.692 | 2679.7 | 4 |  |  |  |  |  |  |
| math | fixed SC N=8 |  |  |  | 0.7179 | 5344.1 | 8 |  |  |  |  |  |  |
| math | fixed SC N=16 |  |  |  | 0.7311 | 10677.7 | 16 |  |  |  |  |  |  |
| math | fixed SC N=32 |  |  |  | 0.7379 | 21347.0 | 32 |  |  |  |  |  |  |
| math | adaptive agreement | 0.625 |  | 0.826 | 0.7364 | 10140.0 | 11.82 | 0.6627 | 0.0586 | 0.0132 | 0.2654 | 0.0825 | 0.0064 |
| math | adaptive quality | 0.5 | 0 | 0.8215 | 0.7317 | 7663.1 | 9.01 | 0.7636 | 0.0581 | 0.0135 | 0.1648 | 0.144 | 0.0164 |
| mathvista | fixed SC N=1 |  |  |  | 0.5419 | 277.8 | 1 |  |  |  |  |  |  |
| mathvista | fixed SC N=2 |  |  |  | 0.5419 | 551.3 | 2 |  |  |  |  |  |  |
| mathvista | fixed SC N=4 |  |  |  | 0.5853 | 1111.6 | 4 |  |  |  |  |  |  |
| mathvista | fixed SC N=8 |  |  |  | 0.6059 | 2229.9 | 8 |  |  |  |  |  |  |
| mathvista | fixed SC N=16 |  |  |  | 0.6168 | 4461.8 | 16 |  |  |  |  |  |  |
| mathvista | fixed SC N=32 |  |  |  | 0.6252 | 8897.8 | 32 |  |  |  |  |  |  |
| mathvista | adaptive agreement | 0.5 |  | 0.6365 | 0.6199 | 3134.7 | 9.39 | 0.7319 | 0.075 | 0.0201 | 0.173 | 0.2278 | 0.039 |
| mathvista | adaptive quality | 0.5 | 0 | 0.64 | 0.6229 | 3476.7 | 10.46 | 0.6696 | 0.0968 | 0.0291 | 0.2045 | 0.203 | 0.0312 |

`stop_n` = fraction of question-orders stopping after n candidates; `recoverable_by_32` = early stops that were wrong while fixed-32 SC was right. Source: `results/task3/adaptive_vs_fixed.csv`, `adaptive_selected.json` (frozen choice + dev grid), `adaptive_eval.json`; figure `figures/adaptive_vs_fixed.png`.

## Live generation (50 MATH held-out questions, Qwen2.5-3B-Instruct FP16, T4, batch 1)

| strategy | n_questions | accuracy | mean_input_tokens | mean_output_tokens | mean_seconds | truncation_rate | extraction_failure_rate | peak_gpu_gb |
|---|---|---|---|---|---|---|---|---|
| single | 50 | 0.62 | 108.4 | 761.3 | 39.2 | 0.04 | 0.04 | 6.63 |
| parallel | 50 | 0.54 | 325.1 | 1859.5 | 94.68 | 0.1667 | 0.1667 | 6.63 |
| refine | 50 | 0.58 | 889.6 | 1239.5 | 63.81 | 0.11 | 0.02 | 6.63 |

Shared initial answer (parallel response #1) accuracy 0.56; refinement transitions: wrong→correct 1, correct→wrong 0.

Refine costs include the shared initial response. Truncation and extraction rates for parallel are per response. Source: `results/task3/live_summary.csv`, raw prompts/responses/seeds `results/task3/live/shard*of2.jsonl`.

## Case lists (fixed rules, for you to choose from)

| list | entries |
|---|---|
| math: selection disagreement | 20 |
| mathvista: selection disagreement | 20 |
| math: premature stopping (agreement) | 20 |
| math: premature stopping (quality) | 20 |
| mathvista: premature stopping (agreement) | 20 |
| mathvista: premature stopping (quality) | 20 |
| live: revision changes | 1 |

Source: `results/task3/cases.json` (each list states its selection rule).
