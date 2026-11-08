# CLAUDE.md — ATML PA3 (fine-tuning, VLM grounding, test-time compute), Tasks 1–3

Ask me which task we're on if it isn't clear from my message. Task 1 work lives on branch `task1-finetune`,
Task 2 on `task2-vlm`, Task 3 on `task3-ttc`.

Course assignment, individual, due 06 Nov 2026. Public GitHub repo. I am responsible for and must understand every line,
so explain non-trivial changes before making them. Code style: follow CODE_STYLE2.md for all new code (its section 12
"additions" — commit hash/config/seed/GPU in results, `--smoke`, `--resume`, unit tests — apply to PA3 too).

## Hard rules (never break these)
- NEVER write, edit, or draft anything in `report/` or report prose/analysis. The PDF report must be my own words.
  You may produce plots, tables (CSV/markdown of numbers), and scripts that generate them.
- Do not modify fixed course assets: `ATML-PA3-Data/`, `first_plot_questions.yaml`, the supplied `README.MD`.
  Never commit weights, datasets, COCO images, cached features, or `outputs/`.
- Do not change seed (6304), decoding settings, token caps, splits, probe IDs, or prompts between conditions being
  compared. Read constants from `configs/`; don't hardcode duplicates or silently "improve" hyperparameters.
- No leakage: evaluation/final-test data never enters training, replay, checkpoint choice, or hyperparameter choice.
  Task 1 uses no old-task data or replay. Task 2 dev/test images never appear in training or replay.
  Task 3 selection, stopping and calibration never touch gold answers or unrevealed rollouts. Add cheap asserts.
- Don't pick a setting after looking at held-out results. The only tuned thing in PA3 is Task 3 adaptive stopping,
  and that is selected on the dev split only and frozen (saved to JSON) before held-out evaluation.
- No early stopping and no "best checkpoint" selection — every run uses its fixed update budget.
- If you reuse external code (e.g. VQA answer normalization, OPLoRA reference code, lm-eval snippets), tell me and add
  it to the README "Attribution" section.
- Don't run `git push --force`, `git reset --hard`, or delete branches without asking.

## Compute setup
- No local GPU. I write/test locally with you, then run on **Kaggle** via thin notebooks in `kaggle/`.
- Kaggle GPU = **T4 ×2** (16 GB each, **no bf16** — the PDF requires FP16 anyway), ~30 GPU-h/week, sessions die at 12 h.
  Kaggle CPU sessions are free: all Task 3 offline analysis and all data prep must run on CPU.
- Everything must be checkable on CPU first:
  - unit tests in `tests/` on small hand-made tensors (`python -m pytest tests -q`)
  - every train/eval script takes `--smoke` (a few steps, tiny batch, few examples, FP32 on CPU; may use a tiny
    random-init Qwen config or a handful of layers) so the pipeline is tested end to end without a GPU.
- Sessions die: train scripts save a full resumable checkpoint to `outputs/<task>/<run_name>/` (model/adapter,
  optimizer, scheduler, GradScaler, python/numpy/torch/cuda RNG, data or sampler position, stage progress) and
  support `--resume`. Eval and generation scripts write per-example results incrementally and skip finished items on rerun.
- Every GPU script takes `--device` (default `cuda` if available) so I can run two jobs at once, one per T4
  (`CUDA_VISIBLE_DEVICES=0` / `=1`).
- `/kaggle/working` is ~20 GB. Keep full-FT checkpoints only where the spec needs them; adapters are small.
  Default `--output_dir` to `/kaggle/working/outputs` if `/kaggle` exists, else `outputs`.
- `kaggle/*.ipynb` are thin launchers only: clone/pull, pip install, attach datasets, call scripts, copy outputs to a
  Kaggle dataset for persistence. No experiment logic.
- After a few steps, every long job prints its measured speed and estimated total time.

## Before writing code: explore
There is no starter repo for PA3 — create the layout below. The handout is at `handout/PA3.pdf` (gitignored):
Task 1 pages 2–6, Task 2 pages 6–10, Task 3 pages 10–15, report/deliverables page 15.
Read `ATML-PA3-Data/README.MD` before touching Task 3 (field definitions, model revision, boxed-answer extraction rule).
Check library versions before relying on an API (lm-eval task names/args, PEFT LoRA init, Qwen3 chat template kwargs).
Pin versions in `requirements.txt` once smoke tests pass and record them; don't upgrade afterwards.

Layout:
```
configs/{task1,task2,task3}/*.yaml
common/{seed,checkpoint,amp,bootstrap,io}.py
task1/{data,train,evaluate,merge,diagnostics,qualitative,make_tables}.py
task1/methods/{full_ft,l2sp,lora,oplora}.py          #one method per file, one <method>_train step fn each
task2/{features,connector,vlm,build_data,train,evaluate,make_tables}.py
task3/{load,answers,scores,selectors,offline_eval,signals,live_generate,adaptive,make_tables}.py
splits/{task1,task2,task3}/   results/{task1,task2,task3}/   tests/   kaggle/   docs/decisions.md
```

## Task 1 spec — LLM fine-tuning (handout pages 2–6)
Model `Qwen/Qwen3-0.6B`, `enable_thinking=False` for training and generation. Every run starts independently from
this checkpoint. Seed 6304, same examples/order/tokenization/eval prompts throughout.

Data: first 40,000 **eligible** MetaMathQA examples in source order. Exclude empty query/response, formatted length
> 2,048 tokens, or `query`/`original_question` matching a GSM8K **test** question after text normalization (define
it, record it in docs/decisions.md). No truncation, no shuffling. Save kept indices + exclusion counts to `splits/task1/`.
Format: user = question, assistant = response. Labels = response tokens + end-of-turn token `<|im_end|>`; prompt and
padding = -100. Check what the non-thinking template inserts (e.g. an empty `<think></think>` block) — those are prompt.

Objective (handout):
  min_θ L_SFT(θ) + λ Ω(θ, θ0),  θ − θ0 ∈ S,   Ω = ½ Σ_j (θ_j − θ0,j)²
Loss normalization: sum token CE over all 16 microbatches, divide by the total valid response tokens in the
effective batch (count tokens for the whole update first). Not a mean of microbatch means.

Shared recipe (configs/task1/base.yaml): 1 epoch, 1,250 updates, microbatch 2 × grad accum 16 = 32, AdamW
betas (0.9, 0.999) eps 1e-8, weight decay 0.01 excluding biases and params with ndim < 2, cosine with 38 warmup
updates, clip global norm 1.0, max length 2,048, gradient checkpointing. FP16 autocast + dynamic GradScaler,
unscale before clipping, log every skipped update. Full FT: FP32 master params. Adapters: frozen FP16 backbone,
FP32 adapters. Optimizer states FP32. Checkpoints at updates 312, 625, 937, 1250 (312/937 are recovery only).

### Step 0 — tests first (nothing else until done)
`tests/test_task1.py` on hand-made tensors: label mask decodes to response + `<|im_end|>`; token-normalized loss over
an accumulated batch equals the hand-computed value; L2-SP Ω on a toy model with a tied weight counts it once;
OPLoRA projections give ‖U_kᵀ B̃‖ ≈ 0 and ‖Ã V_k‖ ≈ 0 on random matrices; ρ_k and η vs hand-computed values;
merged vs unmerged LoRA logits match. Show me the tests, then implement.

### Step 1 — Full FT (`methods/full_ft.py`)
Peak LR 2e-5, λ = 0. Record ‖θ − θ0‖₂ at updates 625 and 1250.
Logging for every run: SFT loss + LR per update, training time, peak GPU memory, trainable params, checkpoint
sizes, method-specific overhead.

### Step 2 — Full FT + L2-SP (`methods/l2sp.py`)
Same as Full FT plus λ = 1e-3. Ω over all unique params incl. biases and norms, tied params once (dedupe by
`data_ptr`), θ0 a fixed FP32 copy, summed not averaged. Add the penalty gradient λ(θ − θ0) to `.grad` after
`scaler.unscale_` and before clipping (keeps the large term out of FP16 loss scaling); compute Ω for logging under
`no_grad`. Log Ω, λΩ, SFT loss, total loss per update; displacement at final; θ0 storage + transfer overhead.
Tell me before deviating from this choice; record it in docs/decisions.md.

### Step 3 — LoRA r=4 and r=16 (`methods/lora.py`)
(r, α) = (4, 8) and (16, 32), peak LR 2e-4, dropout 0.05, random A, zero B, no trainable biases, targets
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj in every layer. Record total adapter params.

### Step 4 — OPLoRA (`methods/oplora.py`)
r=16, α=32, k=128, same LR/targets/dropout and **identical initial A, B as LoRA-16** (assert it). For each W0 = UΣVᵀ,
SVD once in FP32, freeze U_k, V_k:
  B̃ = B − U_k(U_kᵀ B),  Ã = A − (A V_k)V_kᵀ,  ΔW = (α/r) B̃ Ã
Use effective factors, never dense projection matrices. Export projected factors for inference. Record basis storage,
SVD preprocessing time, export overhead. Re-verify the constraints on the trained adapters.

### Step 5 — evaluation (`task1/evaluate.py`, `task1/merge.py`)
Seven conditions: `baseline`, `full_ft_mid` (625), `full_ft`, `l2sp`, `lora4`, `lora16`, `oplora16` (all 1250).
Merge adapters into a fresh backbone copy (OPLoRA: projected factors); log max |merged − unmerged| logits on a small batch.
lm-evaluation-harness, zero-shot, FP16, dropout off, complete splits:
- `gsm8k_cot`, num_fewshot 0, non-thinking chat template, greedy, 1,024 new tokens, stock stops/filters,
  metric flexible-extract exact match; unextractable = wrong.
- `hellaswag` (acc_norm), `boolq` (acc), `arc_easy` (acc_norm), stock prompts, no chat template; report raw acc too.
Use lm-eval's vLLM backend for GSM8K only after checking it against HF on ~100 baseline questions (show me agreement).
Always log samples. Per-example table: id, generation, extracted answer, correct, truncated (hit 1,024), extract_failed.
Δ vs baseline in percentage points.

### Step 6 — diagnostics (`task1/diagnostics.py`, FP32)
Probe set: 32 GSM8K test + 32 HellaSwag val, sampled with seed 6304 BEFORE any results exist →
`splits/task1/probe_ids.json`. For the 5 finals + `full_ft_mid`, `down_proj` in layers 0, 13, 27:
  ρ_k = ‖U_k U_kᵀ ΔW‖²_F / ‖ΔW‖²_F,  η = ‖ΔW‖_F / ‖W0‖_F   (Full FT: ΔW = W_trained − W0)
Forward hooks on the STARTING model cache down_proj inputs z at the last non-padding prompt token (benchmark prompt
formats, before any answer). R(z) = ‖ΔW z‖₂ / (‖W0 z‖₂ + 1e-8), math vs non-math.

### Step 7 — qualitative (`task1/qualitative.py`)
All 24 prompts in `first_plot_questions.yaml` × 7 conditions = 168 greedy responses, thinking off, 512 new tokens,
saved grouped by question and format with truncation flags.
HARD RULE: the annotations are MINE. Build `results/task1/annotation_sheet.csv` for the first and last questions
in plain-question order × 3 formats × 7 conditions = 42 rows, with empty yes/no columns format_compliant,
coherent_relevant, repetition, unnecessary_math_framing, inappropriate_content, an auto-filled truncated column,
and empty excerpt and notes columns (see docs/decisions.md). Never fill, suggest or pre-fill them.

Run names: `full_ft`, `l2sp`, `lora4`, `lora16`, `oplora16`. `task1/make_tables.py` → `results/task1/summary.csv`,
resources table, diagnostics table, figures in `results/task1/figures/` (combined SFT-loss curves, L2-SP Ω/λΩ/total).

## Task 1 GPU runs — never lose progress, run as fast as possible (HARD REQUIREMENT)
Kaggle can kill a session at any moment: quota runs out, 12 h limit, idle timeout, OOM, a crash.
Design Task 1 so that **any interruption costs at most ~15 minutes of GPU time**, and the resumed run is
**identical** to an uninterrupted one. Build and test this before the first real GPU run.

### Where progress lives
- Files in `/kaggle/working` are NOT reliably kept when an interactive session dies. So every checkpoint and every
  results file is also **synced off the machine** to a private Hugging Face Hub repo (`--sync_repo <user>/atml-pa3`,
  token from Kaggle Secrets as `HF_TOKEN`). Fallback flag `--sync_kaggle_dataset` using the Kaggle API.
  Upload in a background thread so training keeps going; log upload success/failure; never let an upload error
  crash training (retry, then warn).
- On start, every script first pulls the latest synced state for its run name, then resumes from it.
  So the launcher is always the same command — I never edit flags to resume. `--resume` is the default; `--fresh`
  must be passed explicitly to start over (and must refuse if a synced checkpoint exists, unless `--force`).

### Checkpoints
- Keep the required milestones (updates 312, 625, 937, 1250). In addition, write **rolling recovery checkpoints**
  every `--ckpt_every` updates (default 50 for adapters, 100 for full FT) AND whenever 20 min of wall clock have
  passed since the last one. These extra ones are recovery-only and don't change the evaluation schedule;
  record that in docs/decisions.md.
- A checkpoint contains everything needed for a bit-identical resume: trainable params (FP32 master params for
  full FT, FP32 adapters for LoRA/OPLoRA — never the frozen backbone), optimizer, LR scheduler, GradScaler,
  python/numpy/torch/cuda RNG states, update counter, examples consumed, skipped-update count, accumulated
  timing/memory stats, and for L2-SP the running Ω log. θ0 is NOT saved — reload it from the HF checkpoint.
  OPLoRA bases U_k, V_k are computed once, saved to `outputs/task1/oplora_bases.pt`, synced, and reloaded on resume.
- Writes are **atomic**: save to `ckpt_tmp/`, fsync, then rename; write a small `manifest.json` last (update number,
  file sizes, sha256). On resume, verify the manifest; if the newest checkpoint is incomplete or corrupt, fall back
  to the previous one. Keep only the newest 2 rolling checkpoints (+ milestones as needed) to fit the ~20 GB disk.
- Milestone model weights needed for evaluation (full FT at 625 and 1250, every run's final) are also exported as
  FP16 safetensors (adapters as their own small files) so eval never needs optimizer state.
- Disk budget: a full-FT recovery checkpoint is ~7 GB (2.4 GB FP32 params + 4.8 GB Adam states). Print the projected
  disk use at startup and refuse to start if two parallel jobs would overflow `/kaggle/working`.
  Never run the two full-FT runs (full_ft, l2sp) in the same session; pair each with an adapter run instead.

### Stopping cleanly
- `--max_hours` (default 11.3): when reached, finish the current update, checkpoint, sync, wait for the upload,
  then exit 0 with a clear "stopped at update N, rerun the same command to continue" message.
- Catch SIGTERM / SIGINT / KeyboardInterrupt: checkpoint at the next update boundary, sync, exit.
- Divergence guard: if the loss is NaN/inf after unscaling for more than 10 consecutive updates, save a debug
  checkpoint and stop with a message (scaler skips are normal; log them, don't stop on a single one).

### Resume correctness
- Data order is fixed (no shuffle), so position = examples consumed; resume by skipping exactly that many
  pre-tokenized examples — never re-iterate a dataloader from the start to get there.
- On resume, truncate `train_log.jsonl` to the checkpoint's update before appending, so no duplicate rows.
- Test (CPU, `--smoke`): run 0→20 updates in one go vs 0→10, kill, resume →20. Losses, LR, and final params must
  match (bitwise on CPU). Also test resume from a deliberately corrupted newest checkpoint. Show me both passing.

### Evaluation and generation progress
- Run each (condition × benchmark) as its own job writing its own output file; skip any that already exists and is
  complete. A crash then loses at most one benchmark for one condition. Sync each file as soon as it's written.
- Qualitative generation and diagnostics write per-item results incrementally and skip finished items on rerun.
- A single `kaggle/task1_queue.py` reads a list of jobs (train / eval / qualitative / diagnostics) and runs whatever
  isn't done yet, in order, on the given GPU. I just rerun it after any interruption. It prints a status table
  (done / partial at update N / not started) at start.

### Speed (all must be mathematically identical to the spec)
- Pre-tokenize the 40k examples once (with labels and lengths), save to disk, sync, reuse for all five runs.
- Dynamic padding per microbatch (pad to the longer of the 2 sequences, never to 2,048). Order stays fixed.
- **Compute the LM head only at supervised positions**: take the hidden states at positions whose shifted label
  != -100, then project and take CE. Qwen's vocab is ~152k, so full logits for 2×2,048 tokens are ~2.5 GB in FP32;
  this is the biggest single memory/speed win. Unit-test that the loss equals the full-logits loss.
- `attn_implementation="sdpa"` (FlashAttention-2 doesn't support T4). Gradient checkpointing with
  `use_reentrant=False`.
- `torch.optim.AdamW(..., fused=True)` where supported; L2-SP penalty gradient and Ω via `torch._foreach_*`
  ops, not a Python loop over parameters. Keep θ0 on GPU if it fits (check peak memory with a preflight),
  otherwise pinned CPU memory with non_blocking copies — measure and report the overhead either way.
- No evaluation inside the training loop. Log scalars once per update (one `.item()` sync per update, not per microbatch).
- DataLoader with `num_workers=2`, `pin_memory=True`, collate from pre-tokenized tensors.
- **Preflight before every run** (≈1 min): one forward/backward/optimizer step on the two longest examples
  (worst-case memory), print peak VRAM and est. time per update, and abort with a clear message if it would OOM —
  instead of crashing hours in. Don't actually update weights in the preflight (or restore them after).
- No torch.compile by default (fragile on T4 + checkpointing); allow `--compile` as an opt-in I can test.
- OPLoRA: SVDs once on GPU in FP32 (~minutes), cached; time recorded as preprocessing overhead.
- Evaluation: vLLM for GSM8K generation (after the HF agreement check), HF for likelihood tasks with the largest
  batch size that fits (`batch_size=auto`); merged FP16 weights loaded once per condition for all four benchmarks.

## Task 2 spec — SigLIP–Qwen VLM (handout pages 6–10)
Vision `google/siglip-base-patch16-224` (always frozen), connector Linear 768→1024, GELU, Linear 1024→1024
(trainable throughout, FP32), LLM `Qwen/Qwen3-0.6B` original weights frozen FP16, LoRA only in Stages 2–3,
`enable_thinking=False`. One model continues through all three stages.

Features (`task2/features.py`): 196 final patch features in 14×14 order, SigLIP final layer norm applied exactly once
(HF `last_hidden_state` already includes `post_layernorm` — check and don't reapply). Deterministic preprocessing,
cache FP16, then free the vision model before loading Qwen. Time/memory of caching recorded separately.

Data (`task2/build_data.py`, seed 6304, save all image/question/pair IDs to `splits/task2/`): train from train2014,
dev and final test from val2014, all evaluation sets image-disjoint from each other and from training (assert).
- COCO captions: 6,000 train images × 3 captions = 18,000; dev 96 images × 1 caption.
- Grounding (script-generated): presence "Does this image contain [object]?" (image with the category vs image with
  no annotation of it → yes/no); counting "How many [objects] can you see?" (pairs with different counts in 1–4).
  Dev 24 pairs, test 64 pairs; reusable training pools sampled with replacement. Crowd annotations and plurals:
  propose a rule to me, then record it.
- VQAv2 complementary pairs: normalized consensus answers differ, ≥ 8/10 agreement per image; target = original
  answer string from the consensus group; keep all 10 answers. yes-no/number/other: train 640/256/640,
  dev 12/8/12, test 36/24/36. One shared answer normalizer (official VQA eval rules, attributed).
Caps: question 80, caption target 64, short-answer target 16.

Loss (handout): L = −(1/|B|) Σ_i (1/|R_i|) Σ_{t∈R_i} log p(y_i,t | I_i, q_i, y_i,<t)
Supervised = caption/answer tokens + end-of-turn; visual tokens, prompt, padding masked.

### Step 0 — tests first
`tests/test_task2.py`: per-example-then-batch loss vs hand-computed; visual tokens sit inside the user message before
the question and get label -100; no-image input rebuilds mask and positions; after one backward connector grads
are nonzero and backbone grads are None; VQA soft accuracy vs hand-computed (e.g. 3 matches → 1.0, 2 → 0.6 by the
leave-one-out formula); pair bootstrap resamples whole pairs with matched IDs across conditions.

### Step 1 — visual interface (`task2/connector.py`, `task2/vlm.py`)
Insert visual embeddings into the user message before the question, keep Qwen's chat structure. Don't wrap the LLM
forward in `no_grad`; gradient checkpointing with `use_reentrant=False`. Calibrate connector output magnitude vs text
embeddings on training data only; record the rule and the scale.

### Steps 2–4 — curriculum (`task2/train.py`)
Shared: microbatch 8 × accum 2 = 16, AdamW wd 0.01, clip 1.0, per-stage 5% warmup then cosine to 10% of peak,
FP16 autocast + GradScaler, unscale before clipping.
| Stage | Updates | Connector LR | LoRA LR | Draw mixture |
|---|---|---|---|---|
| 1 caption alignment | 2,250 | 1e-3 | — | captions, 2 passes over 18,000 |
| 2 visual instruction | 1,500 | 1e-4 | 2e-5 | presence 0.5 / counting 0.25 / caption 0.25 |
| 3 natural VQA | 750 | 5e-5 | 1e-5 | VQAv2 0.7 / presence 0.1 / counting 0.1 / caption 0.1 |
LoRA r16, α32, dropout 0.05, no biases, targets q,k,v,o. Each draw = 2 examples (a full pair, or 2 captions); seeded
sampler, no quotas. Log mixture counts. Checkpoint every 250 updates + connector/adapter weights at each stage end.
Dev eval at 0, every 500, and 2250/3750/4500: caption reference loss; answer NLL (no termination token,
per-example mean then mean) on both VQA dev suites under correct/complementary/no-image;
  ΔNLL = NLL(y | q, I_partner) − NLL(y | q, I_correct)
Never touch final test during training.

### Step 5 — final evaluation (`task2/evaluate.py`)
Greedy, thinking off, 16 new tokens, batch 8 (left padding). Conditions: `correct`, `complementary`, `mismatched`
(fixed mapping excluding both images of the original pair, saved before generation), `no_image`.
Grounding: normalized exact match, presence vs counting. VQAv2: local soft accuracy
  s(â) = (1/10) Σ_i min(1, (1/3) Σ_{j≠i} 1[â = a_j]),  by yes/no, number, other.
Per suite × condition: accuracy, pair accuracy, identical-answer rate, empty-answer rate (empties stay in
denominators), mean reference NLL, generation time per answer. Correct-vs-complementary advantage in accuracy and
NLL, 95% CI from 4,000 bootstrap resamples of complete pairs. Save generations.jsonl per condition.
Example finder: list complementary pairs by a fixed rule (e.g. both right / follows partner / same answer both images);
I choose and interpret them.

### Step 6 — optional 49-token run
Only if I ask. 2×2 average pooling before a new connector + adapters; everything else identical. Run name `tok49`.

Run names: `main` (stages saved as `stage1`, `stage2`, `stage3`). `task2/make_tables.py` → tables and learning curves
with stage boundaries in `results/task2/`.

## Task 3 spec — test-time compute (handout pages 10–15)
Offline on `ATML-PA3-Data` (32 rollouts/question; MATH with Qwen2.5-3B-Instruct, MathVista testmini with
Qwen2.5-VL-3B-Instruct). Join `generations.parquet`, `probability_metrics.parquet` (+ `attention_metrics.parquet` for
VLM) on (dataset, sample_id, rollout_id); assert token lists align. All offline work on CPU, vectorized numpy.

Splits: per dataset sort the 1,000 IDs, `np.random.default_rng(6304).permutation(...)`, first 200 dev, remaining 800 eval;
live subset = first 50 MATH eval IDs in that order. Save to `splits/task3/`.
Answers: math-verify for MATH, MathVista answer-type/choice/precision rules from question metadata, multiple choice
as option letters. Group without gold; missing extraction → one `NO_ANSWER` group, always wrong, kept everywhere.
Record scorer versions. Cache groups and correctness labels once (slow).

Protocol: N ∈ {1,2,4,8,16,32}, 10 seeded permutations of rollout IDs per question (saved), nested prefixes, same
orders for all methods. Average over orders within a question, then over questions. Ties → earliest supporting rollout
in current order; RWC ranks use average ranks. N=1 → the only candidate. Metrics: accuracy, oracle coverage,
generated-token cost (FirstAns consumes 1 candidate). 95% CIs: 1,000 paired question-level bootstrap resamples
after averaging over orders.

Scores (handout):
  L_i = (1/T_i) Σ_t log p_i,t(y_i,t),   K_i = (1/T_i) Σ_t k_i,t,  k_i,t = D_KL(U‖p_i,t)  (from `logp`, `kl_u_p`)
  v_i,t = A_img / (A_img + A_prompt + A_gen),   V_i = mean_t v_i,t
  c_i,t = clip_[0,1]((k_i,t − Q.05(G)) / max(Q.95(G) − Q.05(G), 1e-8)),  C_i = mean_t c_i,t,  Z_i = V_i C_i
  (G = pooled token KL of the CURRENT candidate prefix only, linear-interpolation percentiles, recomputed per prefix)
  RWC: w_i = (N − r_i)/(N − 1), W(a) = Σ_{i: a_i = a} w_i, â = argmax_a W(a)

### Step 0 — tests first
`tests/test_task3.py`: RWC weights and average-rank ties; tie-break by earliest rollout; VisConf calibration uses
only the prefix (adding a 33rd fake rollout must not change prefix scores); Z_token − Z = Cov_t(c, v);
NO_ANSWER is never correct and still votes; adaptive margin with one answer group = 1.

### Step 1 — repeated sampling (`task3/offline_eval.py`)
FirstAns accuracy, oracle coverage, distinct answer groups per N. Failure decomposition: no correct candidate vs
correct available but missed. Save per-question results, candidate IDs, selected answers, token costs.

### Step 2 — selectors (`task3/scores.py`, `task3/selectors.py`)
Both datasets: FirstAns, self-consistency, max mean log-prob, max self-certainty, self-certainty + RWC.
MathVista also: vision-only and VisConf, each with highest-score and RWC.

### Step 3 — signal analysis (`task3/signals.py`, N = 32)
Within-question length/score/correctness associations; top-vs-bottom quartile correctness gap per score averaged
over questions; shortest-rollout control; per-rollout confidence–vision correlation summarized across questions;
tokenwise fusion Z_token = mean_t c·v under both aggregations. Fixed-rule illustrative trajectories (I interpret them).

### Step 4 — live generation (`task3/live_generate.py`, GPU, 50 MATH questions)
`Qwen/Qwen2.5-3B-Instruct` at the README revision, FP16, no quantization, batch size 1, temp 0.7, top-p 0.8,
top-k 20, repetition penalty 1.0, native chat template, system "You are a helpful assistant.", problem + exactly:
  `Think step by step inside <think> </think> tags, then give your final answer inside \boxed{}.`
Strategies (3,072 total output allowance): `single` (1 × 3,072); `parallel` (3 × 1,024, self-consistency);
`refine` (reuse parallel response #1, append as assistant message, then the exact review instruction from the handout
as a user message, one critique+revision response of 2,048; return revised answer). Independently seeded calls.
Last `\boxed{}` with the README balanced-brace rule. Save prompts, responses, seeds, input/output tokens, truncation,
end-to-end time excluding model load, peak memory. Charge the shared initial response to both parallel and refine.
Report wrong→correct / correct→wrong transitions. Resumable per question.

### Step 5 — adaptive sampling (`task3/adaptive.py`, CPU)
Checks at n ∈ {4, 8, 16}, else 32; both return the current self-consistency answer.
Agreement: unique non-NO_ANSWER leader with support ≥ τ·n. Quality-aware: agreement AND SC winner == RWC winner AND
RWC margin ≥ γ (margin = (top − second)/total weight; second = 0 with one group); self-certainty for MATH, VisConf for
MathVista, recomputed from revealed candidates only.
Grid on dev ONLY: τ ∈ {0.5, 0.625, 0.75, 0.875, 1.0}, γ ∈ {0, 0.1, 0.25, 0.5}, plus always-32. Per controller family pick
lowest mean token cost with dev accuracy ≥ fixed-32 SC − 1 pp; ties → higher accuracy, smaller τ, smaller γ.
Save to `results/task3/adaptive_selected.json` BEFORE running eval. Report accuracy, mean tokens, mean rollouts,
stopping-count distribution vs fixed-N SC, and list incorrect early stops fixed-32 would have recovered.

### Step 6 — compute allocation (`task3/make_tables.py`)
Accuracy vs N and vs token cost (oracle coverage vs selection accuracy), method comparison table, adaptive vs fixed
budgets at similar cost. Case finder for selection disagreement, revision and premature stopping by a fixed rule;
I choose and interpret.

## Results and logging
- `results/<task>/<run_name>/` per run: `config.json` (full resolved config), `train_log.jsonl` (per optimizer update:
  step, loss terms, lr, grad norm, scaler scale, skipped), `<run_name>_history.json` (rewritten as per CODE_STYLE2.md),
  eval metrics JSON, per-example outputs (jsonl/parquet).
- Every results file also records: git commit hash, seed, config, GPU name, dtype, wall-clock, peak VRAM
  (`torch.cuda.max_memory_allocated`), library versions where relevant (lm-eval, math-verify).
- `make_tables.py` per task collects runs into `results/<task>/summary.csv` + `figures/` (I copy what I need into the report).
- Every interpretation choice or deviation from the handout goes in `docs/decisions.md` with the handout page.

## README
Top-level README.md with one section per task: environment, data download steps, exact commands for each experiment
(smoke + full), which Kaggle notebook runs what, expected outputs, and "Attribution". Flag names use underscores
(CODE_STYLE), e.g. `python -m task1.train --config configs/task1/lora.yaml --run_name lora16 --rank 16 --alpha 32`.

## Workflow
- One coherent change per commit (tests / data prep / one method / training loop / eval / tables). No giant commits.
- Order (fits the Kaggle GPU quota):
  1. CPU: tests + Task 1 data prep + `--smoke` of every Task 1 method; Task 3 offline pipeline end to end.
  2. GPU: Task 1 baseline eval + HF-vs-vLLM check; trainings two at a time, never both full-FT runs together
     (full_ft ∥ lora16, l2sp ∥ lora4, oplora16 ∥ evals). Only after the resume tests pass.
  3. CPU in parallel: Task 2 data building; then one short GPU session for SigLIP feature caching.
  4. GPU: remaining Task 1 evals + qualitative; Task 2 training + eval; Task 3 live generation.
  5. CPU: Task 1 diagnostics, Task 3 adaptive, all tables/figures, README.
- Before any full-run command, show me the command, the GPU it goes on, expected runtime, and expected outputs.
  After I paste a Kaggle error, fix the root cause and add a smoke check that would have caught it.
- When unsure about a spec detail, ask me instead of guessing; quote the handout line you're unsure about.
