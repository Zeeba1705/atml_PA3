# PA3 — methodology and results reference (Tasks 1–3)

Reference notes for writing the report: what was run (settings, rules, choices) and the main numbers. Facts only,
no interpretation. Full tables: `results/task{1,2,3}/RESULTS_SUMMARY.md`. Every handout interpretation, with its
handout page: `docs/decisions.md`. All runs use seed 6304.

---

# Task 1 — Math fine-tuning of an LLM and retention

## Methodology

**Model and data**
- `Qwen/Qwen3-0.6B`, chat template with thinking off (empty `<think></think>` block is part of the prompt).
- 40,000 MetaMathQA examples (pinned revision), taken in Hub order after removing: empty question/response,
  GSM8K-test overlap (exact match after NFKC + lowercase + punctuation removal + whitespace collapse, on `query` or
  `original_question`), and sequences > 2,048 tokens. Selection saved in `splits/task1/`.
- Loss on response tokens + `<|im_end|>` only (prompt masked).

**Shared training recipe**
- 1,250 updates; micro-batch 2 × grad accumulation 16 = 32 examples/update (one pass over 40,000).
- AdamW (β 0.9/0.999, ε 1e-8, weight decay 0.01), linear warmup 38 updates then cosine to 0, grad clip 1.0.
- FP16 autocast with GradScaler, FP32 master weights / optimizer states; gradient checkpointing; Tesla T4.
- Checkpoints at updates 312 / 625 / 937 / 1,250 (+ rolling recovery checkpoints, never evaluated).

**Methods (5 runs)**

| run | what is trained | peak LR | specifics |
|---|---|---|---|
| Full FT | all parameters | 2e-5 | λ = 0 |
| Full FT + L2-SP | all parameters | 2e-5 | penalty λ/2·‖θ−θ0‖², λ = 0.001; gradient λ(θ−θ0) added after unscaling, before clipping |
| LoRA-4 | LoRA r4, α8 | 2e-4 | targets q,k,v,o,gate,up,down; A Kaiming-uniform, B = 0; frozen FP16 backbone, FP32 adapters |
| LoRA-16 | LoRA r16, α32 | 2e-4 | same as LoRA-4 |
| OPLoRA-16 | LoRA r16, α32 + projection | 2e-4 | update projected orthogonal to the top-k = 128 singular subspaces of W0 (left and right); same init as LoRA-16 (hash checked) |

**Evaluation (7 conditions):** baseline, Full FT midpoint (update 625), and all 5 methods at update 1,250.
- lm-eval 0.4.13, HF backend, FP16, zero-shot, full splits.
- GSM8K test (1,319): `gsm8k_cot`, chat template, thinking off, greedy, 1,024 new tokens, flexible-extract exact
  match, batch 64 fixed for all conditions. Truncation = hit 1,024 tokens without EOS.
- HellaSwag val (10,042) acc_norm, BoolQ val (3,270) acc, ARC-Easy test (2,376) acc_norm; raw acc also reported.
- Adapters merged in memory (W0 + (α/r)BA in FP32, stored FP16); merge checked on logits.
- Δ = percentage points vs baseline; error bars = 1.96 × lm-eval stderr.

**Diagnostics (down_proj, layers 0 / 13 / 27, FP32)**
- ρ_k = ‖U_kU_kᵀ ΔW‖_F² / ‖ΔW‖_F² with U_k = top-128 left singular vectors of W0 (the OPLoRA bases).
- η = ‖ΔW‖_F / ‖W0‖_F (Full FT / L2-SP ΔW from FP16 exports; FP16 rounding level reported as `fp16_noise_eta`).
- R(z) = ‖ΔW z‖ / ‖W0 z‖ for z = down_proj input at the last prompt token, on 32 GSM8K (math) and 32 HellaSwag
  (non-math) probe prompts sampled with seed 6304 before any results (`splits/task1/probe_ids.json`).

**Qualitative (Step 7)**
- 24 prompts from `first_plot_questions.yaml` (Emergent Misalignment repo, commit e40daef): 8 questions × plain /
  JSON (system prompt) / template. One greedy response per prompt per condition, thinking off, 512 new tokens.
- Hand annotation of 42 responses: first and last questions (`what_is_your_wish`, `enough_of_my_husband`) × 3
  formats × 7 conditions. Columns: format_compliant, coherent_relevant, repetition, unnecessary_math_framing,
  inappropriate_content, excerpt, notes; truncated filled automatically.
- Annotation rules used: format_compliant yes / partial / no (yes = only a valid JSON object or exactly-filled
  template, a ```json fence allowed; partial = valid object/template present with extra text; no = absent; plain
  rows N/A). coherent_relevant yes / no judges only whether the content makes sense and fits the question;
  repetition and truncation are scored in their own columns.

## Results

**Benchmarks (score %, Δ pp vs baseline)**

| condition | GSM8K | Δ | HellaSwag | Δ | BoolQ | Δ | ARC-Easy | Δ |
|---|---|---|---|---|---|---|---|---|
| Baseline | 43.14 | — | 47.27 | — | 64.43 | — | 56.02 | — |
| Full FT (625) | 56.41 | +13.27 | 45.83 | −1.44 | 61.38 | −3.06 | 62.21 | +6.19 |
| Full FT | 64.90 | +21.76 | 46.01 | −1.26 | 61.25 | −3.18 | 63.68 | +7.66 |
| Full FT + L2-SP | 64.29 | +21.15 | 46.51 | −0.76 | 66.36 | +1.93 | 63.43 | +7.41 |
| LoRA-4 | 59.82 | +16.68 | 47.32 | +0.05 | 69.11 | +4.68 | 63.59 | +7.58 |
| LoRA-16 | 62.02 | +18.88 | 47.13 | −0.14 | 71.19 | +6.76 | 63.26 | +7.24 |
| OPLoRA-16 | 61.56 | +18.42 | 47.20 | −0.07 | 70.58 | +6.15 | 62.96 | +6.94 |

Stderr ≈ 1.3 (GSM8K), 0.5 (HellaSwag), 0.8 (BoolQ), 1.0 (ARC-Easy). GSM8K truncations at 1,024: 10 / 14 / 5 / 12 /
9 / 8 / 4 (same row order); no-number extractions ≤ 1 everywhere.

**Resources**

| run | trainable params (%) | train h | peak VRAM GB | trainable ckpt MB |
|---|---|---|---|---|
| Full FT | 596.0 M (100) | 2.32 | 12.55 | 2,384 |
| L2-SP | 596.0 M (100) | 2.45 | 12.55 | 2,384 (+2,384 MB θ0 in CPU memory) |
| LoRA-4 | 2.52 M (0.42) | 2.63 | 3.37 | 10.2 |
| LoRA-16 | 10.09 M (1.69) | 2.73 | 3.50 | 40.5 |
| OPLoRA-16 | 10.09 M (1.69) | 3.34 | 3.82 | 40.5 (+31.5 s SVD, 323 MB bases) |

**L2-SP:** displacement ‖θ−θ0‖₂ Full FT 10.02 (625) / 10.66 (1,250) vs L2-SP 6.44 / 6.27; final Ω 19.66,
λΩ 0.0197; final SFT loss 0.1645 (Full FT) vs 0.1646 (L2-SP).

**Diagnostics (layer 13; all layers in the full summary)**

| condition | ρ_128 | η | R math (mean) | R non-math (mean) |
|---|---|---|---|---|
| Full FT (625) | 0.148 | 0.0154 | 0.0549 | 0.0306 |
| Full FT | 0.148 | 0.0160 | 0.0553 | 0.0312 |
| L2-SP | 0.150 | 0.0137 | 0.0459 | 0.0281 |
| LoRA-4 | 0.134 | 0.0131 | 0.0859 | 0.0215 |
| LoRA-16 | 0.151 | 0.0192 | 0.0821 | 0.0277 |
| OPLoRA-16 | 0.000 | 0.0193 | 0.0908 | 0.0284 |

FP16 rounding level for η: 0.00021. Random-direction reference for ρ_128: 128/1024 = 0.125.

**Qualitative annotation tallies (6 responses per condition; format over the 4 JSON/template rows)**

| condition | format yes/partial/no | coherent_relevant yes | repetition yes | math framing yes | truncated |
|---|---|---|---|---|---|
| baseline | 4/0/0 | 3/6 | 1/6 | 0/6 | 0/6 |
| full_ft_mid | 3/1/0 | 3/6 | 1/6 | 3/6 | 1/6 |
| full_ft | 3/1/0 | 3/6 | 2/6 | 4/6 | 1/6 |
| l2sp | 3/1/0 | 3/6 | 1/6 | 3/6 | 1/6 |
| lora4 | 3/0/1 | 5/6 | 1/6 | 0/6 | 0/6 |
| lora16 | 3/0/1 | 5/6 | 2/6 | 1/6 | 2/6 |
| oplora16 | 3/1/0 | 3/6 | 0/6 | 2/6 | 0/6 |

inappropriate_content: no on all 42. Across all 168 generations, truncated at 512: 1 / 3 / 4 / 4 / 2 / 3 / 1, mean
tokens 101 / 164 / 198 / 176 / 118 / 145 / 132 (baseline → oplora16). Math markers seen in outputs:
"The answer is: …" and "#### …" (MetaMathQA's answer format: 240,000 of its 395,000 responses contain `####`).

Figures: `results/task1/figures/{sft_loss,l2sp_terms,benchmark_scores}.png`.

---

# Task 2 — Vision-language model and visual grounding

## Methodology

**Architecture**
- Vision: `google/siglip-base-patch16-224`, frozen; 196 patch features (`last_hidden_state`, post-layernorm),
  cached once in FP16.
- Connector: Linear 768→1024, GELU, Linear 1024→1024 (FP32, trainable) with a fixed output scale 0.05499 =
  mean text-embedding norm / mean initial connector-output norm (training data only).
- LLM: `Qwen/Qwen3-0.6B` frozen FP16, thinking off; LoRA r16/α32/dropout 0.05 on q,k,v,o from stage 2.
- Input: Qwen3 chat template, 196 visual embeddings placed at the start of the user turn, then the question.
  No-image condition drops the 196 positions and keeps all text tokens.
- Loss: per-example mean NLL over target tokens (+ end-of-turn), then batch mean. Caps: question 80, caption 64,
  answer 16 tokens.

**Data (COCO 2014 + VQAv2, seed 6304, all evaluation sets image-disjoint)**
- Captions train 18,000 (6,000 images); VQAv2 train 1,536 pairs; grounding pools 8,000 presence + 8,000 counting
  pairs (built on the caption training images).
- Evaluation: VQA dev 32 / test 96 pairs; grounding dev 24 / test 64 pairs (presence + counting split evenly);
  captions dev 96.
- Grounding labels (strict rule, my choice): presence yes = ≥ 1 non-crowd annotation, no = zero annotations incl.
  crowd; counting = number of non-crowd annotations, images with any crowd annotation of that category excluded,
  counts 1–4. Wording: "Does this image contain a/an {name}?", "How many {plural} can you see?".
- VQAv2 complementary pairs: consensus answer with ≥ 8/10 support on each image, consensus answers differ, same
  answer type.

**Curriculum (fresh AdamW + schedule per stage, 5% warmup, cosine to 10%; micro-batch 8 × accumulation 2)**

| stage | updates | mixture | LR connector / LoRA |
|---|---|---|---|
| 1 | 2,250 (2 passes over captions) | captions | 1e-3 / — |
| 2 | 1,500 | presence 0.5, counting 0.25, captions 0.25 | 1e-4 / 2e-5 |
| 3 | 750 | VQA 0.7, presence 0.1, counting 0.1, captions 0.1 | 5e-5 / 1e-5 |

- Dev evaluation at update 0, every 500, and at stage ends: caption loss, answer NLL under correct /
  complementary / no-image, ΔNLL = NLL(partner image) − NLL(correct image). No checkpoint selection (final model
  evaluated).

**Final evaluation (test sets, final model)**
- Image conditions: correct, complementary (the pair partner's image), mismatched (a fixed derangement of other
  test images), no image. Greedy, 16 new tokens, batch 8.
- Grounding: normalized exact match. VQAv2: soft accuracy over the 10 human answers (handout formula; 1.0 for ≥ 4
  matches, 0.9 for 3, 0.6 for 2, 0.3 for 1), official VQA answer normalization.
- Pair metrics: pair accuracy (both questions of a pair right), identical-answer rate, follows-partner rate
  (prediction equals the partner image's reference), mean reference NLL.
- Correct-image advantages with 95% CIs from 4,000 pair-bootstrap resamples (shared indices across conditions).

## Results

**Dev learning curve (selected points)**

| update | stage | caption loss | VQA NLL correct | VQA ΔNLL | grounding NLL correct | grounding ΔNLL |
|---|---|---|---|---|---|---|
| 0 | init | 5.442 | 13.409 | −0.251 | 11.981 | 0.063 |
| 2,250 | end stage 1 | 2.294 | 9.320 | 0.108 | 9.740 | −0.030 |
| 3,750 | end stage 2 | 2.287 | 2.165 | 1.242 | 1.193 | 2.660 |
| 4,500 | end stage 3 | 2.291 | 1.575 | 1.459 | 1.037 | 1.973 |

**Test accuracy by image condition**

| suite | correct | complementary | mismatched | no image |
|---|---|---|---|---|
| VQAv2 (192 q) | 0.628 | 0.260 | 0.340 | 0.005 |
| Grounding (128 q) | 0.688 | 0.109 | 0.391 | 0.055 |

| suite | metric (correct image) | value |
|---|---|---|
| VQAv2 | pair accuracy / identical-answer / follows-partner | 0.333 / 0.490 / 0.208 |
| VQAv2 | by type: number / other / yes-no | 0.679 / 0.482 / 0.740 |
| Grounding | pair accuracy / identical-answer / follows-partner | 0.516 / 0.141 / 0.109 |
| Grounding | by type: counting / presence | 0.438 / 0.938 |
| VQAv2 / grounding | follows-partner under complementary image | 0.594 / 0.688 |

**Correct-image advantages (mean [95% CI])**

| suite | acc: correct − complementary | − mismatched | − no image | NLL: complementary − correct | no image − correct |
|---|---|---|---|---|---|
| VQAv2 | 0.368 [0.278, 0.457] | 0.289 [0.207, 0.371] | 0.623 [0.556, 0.689] | 1.218 [0.893, 1.583] | 6.487 [5.884, 7.097] |
| Grounding | 0.578 [0.445, 0.703] | 0.297 [0.203, 0.383] | 0.633 [0.531, 0.734] | 2.238 [1.772, 2.710] | 2.915 [2.591, 3.250] |

**Resources:** trainable params 1.84 M (stage 1) / 6.42 M (stages 2–3); training 1.72 h for all 4,500 updates
incl. dev evals, peak 1.67 GB, Tesla T4; eval peak 1.6 GB, ~0.035–0.04 s per answer (batch 8); feature caching
9,226 images in 3,360 s on CPU (session had no GPU; features stored FP16 as required).

Figures: `results/task2/figures/{learning_curves,final_accuracy_by_condition}.png`. Example lists:
`results/task2/main/eval/examples.json`.

---

# Task 3 — Test-time compute: sampling, selection, adaptive inference

## Methodology

**Offline data:** supplied rollouts, 1,000 questions × 32 rollouts each: MATH (Qwen2.5-3B-Instruct) and MathVista
testmini (Qwen2.5-VL-3B-Instruct), with per-token log-probs, KL(U‖p) and (MathVista) attention totals.
- Splits (seed 6304): sorted ids permuted; 200 dev / 800 held-out per dataset. All reported numbers are held-out
  unless marked dev.
- 10 seeded rollout orders per question; budgets N ∈ {1, 2, 4, 8, 16, 32} as nested prefixes of each order.
  Average over orders within a question, then over questions. Cost = sum of generated tokens of consumed rollouts.
- Answer grouping without gold: MATH = math-verify 0.9.0 equivalence to the first matching group; MathVista =
  strip units, then official integer / float-precision / multiple-choice rules (Levenshtein fallback). Missing
  extraction = NO_ANSWER (votes, always wrong).

**Scores (computed from revealed candidates only)**
- Mean log-prob L = mean_t log p(y_t); self-certainty K = mean_t KL(U‖p_t).
- MathVista vision share v_t = A_img / (A_img + A_prompt + A_gen); V = mean_t v_t.
- Confidence c_t = clip((K_t − Q05)/(Q95 − Q05)), percentiles pooled over the revealed candidates only.
  VisConf Z = V × C (rollout means multiplied); tokenwise ablation Z_tok = mean_t c_t v_t.
- Length-only control: shortest response.

**Selectors:** FirstAns; self-consistency (majority vote); highest score; RWC = rank-weighted vote with weight
(N − r)/(N − 1), r = average rank (1 = best). Ties → earliest supporting rollout.

**Adaptive stopping (checks after 4, 8, 16 candidates, else 32; final answer = current SC winner)**
- Agreement: stop if a unique non-NO_ANSWER leader has support ≥ τ·n.
- Quality: agreement AND SC winner = RWC winner AND RWC margin ≥ γ (self-certainty for MATH, VisConf for MathVista).
- τ ∈ {0.5, 0.625, 0.75, 0.875, 1.0}, γ ∈ {0, 0.1, 0.25, 0.5}, chosen on dev only: lowest cost with dev accuracy
  ≥ fixed-32 SC − 1 pp. Choice frozen to `results/task3/adaptive_selected.json` before the held-out run.

**Live generation (50 held-out MATH questions)**
- Qwen2.5-3B-Instruct (revision aa8e725), FP16, T4, batch 1; temperature 0.7, top-p 0.8, top-k 20, no repetition
  penalty; system "You are a helpful assistant."; instruction to think in `<think>` tags and box the answer.
- Strategies (≈ equal 3,072-token budget): single = 1 × 3,072; parallel = 3 × 1,024 + self-consistency;
  refine = parallel response #1 + one 2,048-token review turn (review instruction from the README).
- Seeds per call: 6304·1000 + 10·question + k. Answer = last `\boxed{}`, graded with the same math-verify rule.

## Results

**Methods at N = 32 (held-out accuracy; Δ vs SC in pp with 95% paired-bootstrap CI)**

| method | MATH | Δ [CI] | MathVista | Δ [CI] |
|---|---|---|---|---|
| FirstAns (N = 1 cost) | 0.644 | −9.4 [−11.0, −8.0] | 0.542 | −8.3 [−10.0, −6.5] |
| Self-consistency | 0.738 | — | 0.625 | — |
| Max log-prob | 0.640 | −9.8 [−12.4, −7.4] | 0.490 | −13.5 [−16.6, −10.5] |
| Max self-certainty | 0.706 | −3.2 [−5.5, −0.8] | 0.556 | −6.9 [−9.8, −3.7] |
| Self-certainty + RWC | 0.756 | +1.8 [0.7, 3.0] | 0.636 | +1.1 [−0.2, 2.5] |
| Shortest (control) | 0.624 | −11.4 [−14.1, −9.0] | 0.541 | −8.4 [−11.5, −5.5] |
| Max vision | — | — | 0.541 | −8.4 [−11.5, −5.2] |
| Vision + RWC | — | — | 0.636 | +1.1 [−0.5, 2.7] |
| Max VisConf | — | — | 0.581 | −4.4 [−7.0, −1.5] |
| VisConf + RWC | — | — | 0.641 | +1.6 [0.2, 3.1] |
| Max VisConf-token | — | — | 0.568 | −5.8 [−8.4, −2.7] |
| VisConf-token + RWC | — | — | 0.636 | +1.1 [−0.3, 2.6] |

Oracle coverage at 32: MATH 0.914, MathVista 0.899. Mean tokens at 32: MATH 21,347, MathVista 8,898. The three equal
MathVista RWC values (0.6362) are a coincidence: the variants pick different rollouts in 1,084 of 10,000
question-orders.

**Coverage vs selection (SC, fraction of question-orders)**

| N | MATH coverage | MATH SC | MATH missed-with-correct-available | MathVista coverage | MathVista SC | MathVista missed |
|---|---|---|---|---|---|---|
| 1 | 0.644 | 0.644 | 0.000 | 0.542 | 0.542 | 0.000 |
| 4 | 0.791 | 0.692 | 0.099 | 0.763 | 0.585 | 0.178 |
| 8 | 0.844 | 0.718 | 0.126 | 0.832 | 0.606 | 0.226 |
| 16 | 0.885 | 0.731 | 0.153 | 0.876 | 0.617 | 0.259 |
| 32 | 0.914 | 0.738 | 0.176 | 0.899 | 0.625 | 0.274 |

**Signals (N = 32, Spearman score–correct within mixed questions)**
- MATH: log-prob 0.134, self-certainty 0.198, −length 0.076.
- MathVista: log-prob 0.034, self-certainty 0.050, −length 0.044, vision 0.037, VisConf 0.087, VisConf-token 0.078.
- MathVista within-rollout confidence–vision correlation: mean −0.211, median −0.212, 10.6% of questions > 0.

**Adaptive vs fixed (held-out)**

| dataset | policy | dev-chosen setting | accuracy | mean tokens | mean rollouts | stop at 4 | wrong early stops recoverable by 32 |
|---|---|---|---|---|---|---|---|
| MATH | fixed SC 32 | — | 0.738 | 21,347 | 32 | — | — |
| MATH | fixed SC 16 | — | 0.731 | 10,678 | 16 | — | — |
| MATH | agreement | τ = 0.625 | 0.736 | 10,140 | 11.8 | 0.663 | 0.006 |
| MATH | quality | τ = 0.5, γ = 0 | 0.732 | 7,663 | 9.0 | 0.764 | 0.016 |
| MathVista | fixed SC 32 | — | 0.625 | 8,898 | 32 | — | — |
| MathVista | fixed SC 8 | — | 0.606 | 2,230 | 8 | — | — |
| MathVista | agreement | τ = 0.5 | 0.620 | 3,135 | 9.4 | 0.732 | 0.039 |
| MathVista | quality | τ = 0.5, γ = 0 | 0.623 | 3,477 | 10.5 | 0.670 | 0.031 |

**Live generation (50 MATH questions)**

| strategy | accuracy | mean output tokens | mean seconds | truncation | extraction failure |
|---|---|---|---|---|---|
| single (1 × 3,072) | 0.62 | 761 | 39.2 | 0.04 | 0.04 |
| parallel (3 × 1,024) | 0.54 | 1,860 | 94.7 | 0.167 per response | 0.167 per response |
| refine (1,024 + 2,048) | 0.58 | 1,240 | 63.8 | 0.11 | 0.02 |

Initial answer (parallel #1) accuracy 0.56; refinement changed wrong→correct 1, correct→wrong 0. Peak GPU 6.63 GB.

Figures: `results/task3/figures/{accuracy_vs_n,accuracy_vs_tokens,adaptive_vs_fixed}.png`. Case lists:
`results/task3/cases.json`.
