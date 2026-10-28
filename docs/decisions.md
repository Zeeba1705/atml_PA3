# Decisions and interpretations

Every interpretation of the handout or deviation from it, with the handout page it refers to.

## Task 1 — LLM fine-tuning

### Data and formatting (handout p. 3 "Training data")

- Source: `meta-math/MetaMathQA` train split in Hub order, pinned to one revision hash (recorded in `splits/task1/`).
  "Empty question/response" = `query.strip() == ""` or `response.strip() == ""`.
- GSM8K overlap: a MetaMathQA example is excluded if its normalized `query` or normalized `original_question` equals
  a normalized GSM8K test question. Normalization: Unicode NFKC, lowercase, remove punctuation, collapse whitespace.
  Exact equality of the normalized strings, no fuzzy matching.
- Chat format (checked with the Qwen3-0.6B tokenizer): with `enable_thinking=False` the template renders
  `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{response}<|im_end|>\n`, and the
  generation prompt is the same text up to and including `<think>\n\n</think>\n\n`. Everything up to there is prompt
  (label -100), including the empty think block.
- Training sequence = prompt + response + `<|im_end|>`. The template's trailing `\n` after `<|im_end|>` is dropped
  (it is not response and not the end-of-turn token). Supervised = response tokens + `<|im_end|>`.
- "Formatted length above 2,048 tokens" is measured on that exact training sequence; length 2,048 is kept.
- Tokenization: prompt and `response + <|im_end|>` are tokenized separately and the ids concatenated. About 1.7% of
  MetaMathQA responses start with `\n`; tokenizing the whole string merges the prompt's final `\n\n` with it into one
  `\n\n\n` token that is half prompt, half response. Separate tokenization keeps the prompt tokens identical to the
  generation-time prompt (generation also starts right after `\n\n`) and leaves every response unchanged.
- Each skipped example gets one reason, checked in this order: empty, GSM8K overlap, too long. Selection stops at
  the 40,000th kept example; counts and indices cover everything scanned up to it.

### Training recipe (handout p. 3 "Shared training settings")

- LR schedule: linear warmup over the first 38 updates (update u, 0-based, uses (u+1)/38 × peak), then cosine decay
  from peak to 0 over the remaining 1,212 updates (update u uses 0.5·(1+cos(π·(u−38)/1212)) × peak). The schedule
  advances on every update, including updates skipped by the GradScaler, so the budget stays fixed.
- LoRA "random A" = Kaiming-uniform with a=√5 (the PEFT default), B = 0. LoRA is implemented as a small module in
  `task1/methods/lora.py` instead of PEFT, so LoRA-16 and OPLoRA-16 share the same init code and the same seed.
- Training time = wall-clock of the training loop summed over resumed sessions. Model load, preflight,
  checkpoint writing and syncing are timed separately.

### Step 2 — L2-SP (handout p. 4)

- The penalty gradient λ(θ − θ0) is added to `.grad` after `scaler.unscale_` and before clipping, so the large
  penalty term never passes through FP16 loss scaling. Clipping applies to the total (SFT + penalty) gradient.
  It is added as two in-place adds (λθ, then −λθ0) in chunks of 16 tensors, so there is no 2.4 GB temporary.
- θ0 is an FP32 copy of every unique parameter (tied embedding/LM head once), taken from the freshly loaded model
  before any checkpoint is restored, and never saved. For L2-SP it stays on the GPU; for Full FT, which only needs
  it for displacement, it stays in CPU memory.
- Logged Ω is computed at the parameters the update's loss was computed on (before the optimizer step), so
  total_loss = SFT loss + λΩ refers to one point. Displacement ‖θ − θ0‖₂ = √(2Ω) is logged after the step at
  updates 625 and 1250, for both Full FT and L2-SP.

### Step 4 — OPLoRA (handout p. 4)

- The projection B̃, Ã is computed in FP32 inside each forward with autocast disabled; under FP16 autocast it would
  otherwise only be orthogonal to FP16 precision.
- SVD bases are computed once from the loaded W0 (FP16 copy of the original BF16 weights, upcast to FP32; BF16
  values in the normal range are exact in FP16), saved to `outputs/task1/oplora_bases.pt`, and reused on resume and
  by the diagnostics.
- Identical init: every adapter run records a sha256 of its initial A matrices; `lora16` and `oplora16` must match.

### Recovery checkpoints (handout p. 3 "Checkpoint and evaluation schedule")

- Besides the required checkpoints at updates 312/625/937/1250, rolling recovery checkpoints are written every
  `ckpt_every` updates (50 for adapters, 100 for full FT) and whenever 20 min have passed since the last one.
  They are only for resuming after an interrupted session: they don't change the evaluation schedule and are
  never evaluated.

### Step 6 — diagnostics (handout p. 3, p. 5)

- Probe set: `np.random.default_rng(6304).choice(n, 32, replace=False)`, sorted, separately for GSM8K test and
  HellaSwag validation; saved to `splits/task1/probe_ids.json` before any model results exist.
- "Benchmark prompt formats": GSM8K = the chat-templated lm-eval `gsm8k_cot` zero-shot prompt with the generation
  prompt; HellaSwag = the lm-eval context only, with no ending appended. z = the `down_proj` input at the last
  non-padding token of that prompt.

### Step 7 — qualitative comparison (handout p. 3 "Diagnostic and qualitative inputs", p. 5 step 7)

Status: `first_plot_questions.yaml` is not in the repo yet. It is only needed for Step 7.

- Prompt source: from `first_plot_questions.yaml` use only the question text and any system prompt / format
  instruction an entry provides. Ignore its sampling counts, temperature and judge prompts. The handout overrides
  them: one greedy response per prompt, annotated manually.
- System prompt: none by default. One is added only if the YAML entry supplies it.
- Generation settings, identical for all 7 conditions (`baseline`, `full_ft_mid`, `full_ft`, `l2sp`, `lora4`,
  `lora16`, `oplora16`): same tokenizer, same chat template with `enable_thinking=False`, FP16, one fixed batch
  size, left padding, greedy decoding, 512 new tokens.
- Truncated = the response hit 512 new tokens without producing an end-of-turn token. Computed automatically.
- Annotation subset: the first and last questions in the order the plain-format entries appear in the file, plus
  their JSON and template variants (6 prompts), × 7 conditions = 42 rows. The chosen prompt IDs are saved to a file.
- Annotation sheet `results/task1/annotation_sheet.csv`: yes/no columns `format_compliant`, `coherent_relevant`,
  `repetition`, `unnecessary_math_framing`, `inappropriate_content`; an auto-filled `truncated` column; empty
  `excerpt` and `notes` columns. All yes/no columns, `excerpt` and `notes` are left EMPTY — the annotations are
  filled in by hand by me, never by code or an assistant.
