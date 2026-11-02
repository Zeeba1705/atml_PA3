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
- Preflight: before training, two forward/backward/optimizer steps, each accumulating two microbatches of the two
  longest examples, measure worst-case peak memory. Two steps because AdamW's state exists during forward/backward
  only from the second step on; two microbatches because backward then builds new gradients while the accumulated
  ones exist. (A single step, single microbatch measured 9.6 GB for full FT on a T4.) Weights don't change
  (gradients are zeroed before each throwaway lr=0 AdamW step) and the RNG state is restored.
- L2-SP's θ0 (FP32, 2.4 GB) always lives in pinned CPU memory and is copied to the GPU in chunks of 16 tensors after
  backward. On a T4 it doesn't fit next to the training state: an earlier version moved it to the GPU when the
  preflight estimate left room, and real training then ran out of memory in backward. Placement doesn't change the
  values added to the gradient, only speed.

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

### Step 5 — evaluation (handout p. 4)

- Backend: lm-eval 0.4.13 with the **HF backend for all four benchmarks**, no vLLM. Every vLLM release needs a
  newer transformers than the pinned 5.5.0 (the ones built for torch 2.11 exclude 5.5.0 explicitly), and the
  pins are not upgraded after the smoke tests. So there is no HF-vs-vLLM agreement check: there is no second backend.
- GSM8K batch size is fixed at 64 for every condition instead of `auto`, because in FP16 the batch composition can
  change greedy outputs. The likelihood tasks use `auto` (same model size and GPU for every condition).
- Evaluation runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The first baseline GSM8K run ran out of
  memory on a T4 with 9.8 GB allocated and 4.4 GB reserved but fragmented (the adapter conditions peaked at 10.2 GB
  and fit). Kept batch 64 instead of lowering it, so all conditions keep the same batch. The setting only changes
  how the allocator places memory, not any computation, so lora4/lora16/oplora16 (evaluated before the setting)
  stay comparable. Each metrics file records `cuda_alloc_conf`.
- lm-eval loads each benchmark from the dataset's main branch; `task1/evaluate.py` refuses to run if that commit
  differs from the one recorded in `configs/task1/eval.yaml` (gsm8k and hellaswag match the revisions pinned for
  training data and the probe set).
- Splits are lm-eval's stock evaluation splits, complete: gsm8k test (1,319), hellaswag validation (10,042), boolq
  validation (3,270), arc_easy test (2,376). Seeds passed to lm-eval are all 6304 (nothing is sampled: zero-shot,
  greedy).
- "truncated (hit 1,024)" is the number of tokens the model actually generated before its first EOS/pad, recorded by
  wrapping `generate` during the run. Re-tokenizing the returned text is not used: it undercounts degenerate
  outputs (a run of newline tokens re-tokenizes to a few tokens). "extract_failed" = flexible-extract found no
  number (`[invalid]`), which exact match scores as wrong.
- Merging: adapter runs are rebuilt as a fresh FP16 backbone plus the exported FP32 factors (OPLoRA: the projected
  factors, so a plain LoRA layer reproduces the trained model exactly; unit-tested), then merged in memory
  (W0 + (α/r)BA computed in FP32, stored FP16) and handed to lm-eval directly, so no merged copy is written to disk.
  The merge check compares logits before and after merging on the first 4 GSM8K test questions (chat-templated, the
  same prompt format as the GSM8K eval) and logs the max and mean |Δ| over all non-padding positions.
- One process per condition loads the model once and runs the four benchmarks in order; each benchmark's metrics,
  samples (and the GSM8K per-question table) are written and synced as soon as it finishes, and finished benchmarks
  are skipped on rerun.

### Step 6 — diagnostics (handout p. 3, p. 5)

- Probe set: `np.random.default_rng(6304).choice(n, 32, replace=False)`, sorted, separately for GSM8K test and
  HellaSwag validation; saved to `splits/task1/probe_ids.json` before any model results exist.
- "Benchmark prompt formats": GSM8K = the chat-templated lm-eval `gsm8k_cot` zero-shot prompt with the generation
  prompt; HellaSwag = the lm-eval context only, with no ending appended. z = the `down_proj` input at the last
  non-padding token of that prompt.
- The prompt strings are taken from the baseline's logged lm-eval samples (prompts don't depend on the model), so
  they are exactly what the benchmarks scored; each probe's first 80 characters are checked against
  `probe_ids.json`. HellaSwag contexts get lm-eval's trailing-whitespace move (whitespace goes to the continuation).
  Each prompt is run alone (batch 1, no padding), so the last position is the last prompt token
  (`probe_tokens.csv` lists it: GSM8K ends on the `\n\n` after the empty think block).
- W0 and z come from the starting model loaded in FP32 (BF16 checkpoint values are exact in FP32).
- U_k = the bases the oplora16 run trained with (`outputs/task1/oplora_bases.pt`, k = 128), not recomputed;
  `diagnostics_info.json` records the max difference between U_kU_kᵀ and a fresh SVD's projector as a check.
- Adapter runs: ΔW = (α/r)·B·A from the exported FP32 factors (projected ones for OPLoRA), exact.
- Full FT and L2-SP: ΔW = W_trained − W0 with W_trained from the FP16 export, because the FP32 training states
  were deleted after export to stay within the HF storage limit. ΔW therefore includes FP16 storage rounding.
  `weight_metrics.csv` reports `fp16_noise_eta` (expected ‖rounding error‖_F / ‖W0‖_F, ulp/√12 per element) next to
  η, so its size relative to the update is visible. Rounding noise alone has no preferred direction: its ρ_k
  would be ≈ k/d_out = 128/1024 = 0.125.
- Outputs (`results/task1/diagnostics/`): `weight_metrics.csv` (ρ_k, η per condition × layer),
  `activation_ratios.csv` (R(z) per probe), `activation_summary.csv` (mean/median/std/min/max per condition ×
  layer × set), `probe_tokens.csv`, `diagnostics_info.json` (written last). Cached z: `outputs/task1/diagnostics/z.pt`.

### Step 7 — qualitative comparison (handout p. 3 "Diagnostic and qualitative inputs", p. 5 step 7)

- Source: the course did not supply `first_plot_questions.yaml`; the handout's description (24 prompts, 8 questions in
  plain, JSON and template formats, unrelated to mathematics) matches the Emergent Misalignment paper's file of that
  name, used unchanged from its repo at commit `e40daef` (sha256 checked at load; see README Attribution).
- Formats come from the entry ids: `_json` entries carry the file's JSON system prompt, `_template` entries put the
  template in the user message, plain entries have neither.

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
  filled in by hand by me, never by code or an assistant. The sheet also carries the response text and `n_tokens` so it
  can be filled in without another file. It is built locally (`--collect`) and never overwritten once it exists.
- One batch of all 24 prompts per condition (fixed for every condition), left padding, `do_sample=False`.

### Tables and figures (`task1/make_tables.py`)

- Δ vs baseline in percentage points = 100 × (condition − baseline) on the main metric (raw acc also reported).
- Error bars = 95% CI as 1.96 × lm-eval's stderr for each condition alone (not a paired test between conditions).
- Loss curves: trailing 25-update moving average of the per-update SFT loss (the raw values are in each run's
  `train_log.jsonl`); a second panel zooms to updates 100–1250.
- Benchmark figure uses dots + intervals, not bars, because each panel's y-axis is zoomed to the data.
- L2-SP terms are drawn in separate panels (SFT/total, λΩ, Ω) since their scales differ by ~1000×; no dual axes.
- Fixed colours per run (full_ft_mid = a lighter step of Full FT's blue, baseline = gray), same in every figure.


## Task 2 — SigLIP–Qwen VLM

### Visual interface (handout p. 7–9)

- Prompt layout: Qwen3's own chat template (thinking off) for a user message whose first line is the image, split at
  a text marker that is never tokenized: `<|im_start|>user\n` + 196 visual tokens + `\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
  + target + `<|im_end|>`. No-image inputs drop the 196 positions and keep every text token; masks and positions are
  rebuilt (positions = cumulative count of real tokens, so padding on either side never shifts them).
- Caption prompt (not fixed by the handout): "Describe this image in one sentence." (`configs/task2/base.yaml`).
- Caps: question tokens truncated to 80, caption target to 64, short answers to 16 content tokens; end-of-turn is
  appended after the cap and supervised in training. Answer NLL for dev/eval excludes it (handout p. 8).
- Loss: per-example mean NLL over labelled tokens, then the batch mean; logits are computed only at labelled positions.
- Connector magnitude calibration (training data only): a fixed, non-trainable output scale
  s = mean ‖text embedding‖ / mean ‖initial connector output‖, the text side measured over the training captions' and
  questions' token embeddings, the visual side over training-image features. The value is recorded in the run config.
- VQA soft accuracy follows the handout formula; with 10 answers it gives 1.0 for ≥ 4 matches, 0.9 for 3, 0.6 for 2,
  0.3 for 1 (CLAUDE2.md's "3 matches → 1.0" example was off; the test uses the formula).
- One normalizer (official VQA eval rules, adapted; see README Attribution) for VQA answers, grounding targets and
  model outputs. The official code's `re.UNICODE` passed as a replacement count is not copied (all periods matching
  the pattern are stripped).

### Grounding labels (Strict rule, chosen by me)

- Presence "yes" image: ≥ 1 non-crowd annotation of the category; "no" image: zero annotations of it, crowd included.
- Counting: count = number of non-crowd annotations; an image with any crowd annotation of that category is not used
  for counting it. Counts 1–4 only; answers are digits.
- Wording: "Does this image contain a/an {name}?" and "How many {plural} can you see?" with a fixed hand-written plural
  map for the 80 COCO category names (`configs/task2/`).

### Data selection (`task2/build_data.py`, handout p. 7–8)

- Grounding evaluation pairs split evenly between presence and counting: dev 12 + 12, test 32 + 32 (the handout gives
  only the combined 24 / 64).
- Grounding training pools (8,000 presence + 8,000 counting pairs, sampled with replacement during training) are built
  on the 6,000 caption training images, so features are cached once for training.
- Pair generation: category uniform among the usable ones, then images uniform within it (counting: two different
  counts drawn first). VQAv2: candidates sorted by pair id, seeded permutation, taken per answer type.
- VQAv2 consensus = most common normalized answer with ≥ 8/10 support on each image, the two consensus answers
  different, both questions of the same answer type; target = the most frequent original string in the consensus
  group (ties → first in annotation order).
- Order of evaluation selection (one shared "used images" set, so every evaluation set is image-disjoint from every
  other): VQA dev, grounding dev, VQA test, grounding test, caption dev. Asserted in `check_isolation`.
