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
- Loss computation (memory only, same math): the LM head is applied only at supervised positions, and the
  cross-entropy over them is summed in checkpointed chunks of 1,024 positions, so at most one chunk of
  152k-vocab logits exists at a time.
- Preflight: before training, two forward/backward/optimizer steps on the two longest examples measure worst-case
  peak memory (two, because AdamW's state exists during forward/backward only from the second step on; a single
  step measured 9.6 GB for full FT on a T4 against a real peak of 12.6 GB). Weights don't change (gradients are
  zeroed before each throwaway lr=0 AdamW step) and the RNG state is restored. L2-SP's θ0 is created in pinned CPU memory and the preflight runs with it there; θ0 moves to the GPU only
  if the measured peak plus θ0's size (2.4 GB) stays under 93% of GPU memory. Either way the same values are added
  to the gradient; only speed differs. (The first version put θ0 on the GPU and moved it after a failed preflight;
  on a T4 that move itself ran out of memory, so the order was reversed.)

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
- Retention on disk: a Full FT resumable state is ~7 GB and `/kaggle/working` is ~20 GB, so Full FT and L2-SP keep
  at most the newest 2 resumable states, milestone ones included (older ones are deleted before a new one is
  written). What evaluation needs is kept separately as FP16 exports: Full FT at 625 and 1250, L2-SP at 1250.
  Adapter runs (a few MB per state) keep every milestone checkpoint plus the newest 2 rolling ones.
- Off-machine copies: every checkpoint, export and results file is uploaded in a background thread to a private
  HF Hub repo, one branch per run (`task1-<run>`). The remote mirrors local pruning (new checkpoint uploaded before
  old ones are deleted) and the branch history is squashed after deletions so storage doesn't grow.
- Clean stops: SIGTERM/SIGINT or `--max_hours` → checkpoint at the end of the current update, wait for uploads,
  exit 0. Divergence guard: more than 10 consecutive updates with non-finite loss or gradient norm → debug
  checkpoint, exit 2.

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
