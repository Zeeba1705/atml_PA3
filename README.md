# atml_PA3

ATML PA3: LLM fine-tuning (Task 1), VLM grounding (Task 2), test-time compute (Task 3).
Interpretation choices and deviations from the handout are in [docs/decisions.md](docs/decisions.md).

## Task 1 — LLM fine-tuning

### Environment

```bash
pip install -r requirements.txt   #torch: use your CUDA build (Kaggle's image as-is); not pinned
python -m pytest tests -q          #CPU only, ~3 min
```

### Data

```bash
python -m task1.data --smoke   #160 examples -> outputs/task1/data/smoke_*
python -m task1.data           #first 40,000 eligible MetaMathQA examples -> outputs/task1/data/train_tokenized.pt
python -m task1.diagnostics --make_probe_ids   #already done: splits/task1/probe_ids.json (refuses to overwrite)
```

`splits/task1/train_selection.json` (committed) holds the dataset revisions, kept indices, exclusion counts and a
token fingerprint; training refuses to start if the tokenized file doesn't match it.

### Training

Smoke (CPU, tiny random-init Qwen3, synthetic data, 20 updates):

```bash
python -m task1.train --config configs/task1/full_ft.yaml --run_name full_ft --smoke
python -m task1.train --config configs/task1/l2sp.yaml --run_name l2sp --smoke
python -m task1.train --config configs/task1/lora.yaml --run_name lora4 --rank 4 --alpha 8 --smoke
python -m task1.train --config configs/task1/lora.yaml --run_name lora16 --rank 16 --alpha 32 --smoke
python -m task1.train --config configs/task1/oplora.yaml --run_name oplora16 --smoke
```

Full runs (one T4 each, 1,250 updates). The same command always resumes from the newest local or synced checkpoint;
`--fresh --force` starts over and deletes the synced state.

```bash
python -m task1.train --config configs/task1/full_ft.yaml --run_name full_ft --sync_repo <user>/atml-pa3
python -m task1.train --config configs/task1/l2sp.yaml --run_name l2sp --sync_repo <user>/atml-pa3
python -m task1.train --config configs/task1/lora.yaml --run_name lora4 --rank 4 --alpha 8 --sync_repo <user>/atml-pa3
python -m task1.train --config configs/task1/lora.yaml --run_name lora16 --rank 16 --alpha 32 --sync_repo <user>/atml-pa3
python -m task1.train --config configs/task1/oplora.yaml --run_name oplora16 --sync_repo <user>/atml-pa3
```

On Kaggle these are run by `kaggle/task1.ipynb` (thin launcher: clone, install, data prep, then
`kaggle/task1_queue.py` once per GPU). The queue runs whatever isn't finished, in order:
`gpu0`: full_ft → l2sp → evals of baseline, full_ft_mid; `gpu1`: lora16 → lora4 → oplora16 → evals
of lora4, lora16, oplora16, full_ft, l2sp (see `kaggle/task1_jobs.yaml`).

```bash
python kaggle/task1_queue.py --status --sync_repo <user>/atml-pa3
python kaggle/task1_queue.py --group gpu0 --gpu 0 --sync_repo <user>/atml-pa3
```

### Evaluation (Step 5)

lm-evaluation-harness 0.4.13, HF backend, zero-shot FP16 on the complete splits: `gsm8k_cot` (chat template,
thinking off, greedy, 1,024 new tokens), `hellaswag`, `boolq`, `arc_easy`. Settings in `configs/task1/eval.yaml`,
choices in [docs/decisions.md](docs/decisions.md). One process per condition loads (and for adapters, merges) the
model once and runs the four benchmarks; finished benchmarks are skipped on rerun.

```bash
python -m task1.evaluate --smoke --device cpu      #tiny random model + random adapters, 2 docs per benchmark (~2 min)
python kaggle/task1_queue.py --group smoke_eval --device cpu   #same, through the queue
python -m task1.evaluate --condition lora16 --sync_repo <user>/atml-pa3   #one condition, all four benchmarks
```

Conditions: `baseline`, `full_ft_mid` (update 625), `full_ft`, `l2sp`, `lora4`, `lora16`, `oplora16` (update 1250).
Exports are pulled from the training branches when they aren't on disk.

### Diagnostics (Step 6)

FP32 weight-space (ρ_k, η) and activation (R(z)) diagnostics for `down_proj` in layers 0, 13, 27, for `full_ft_mid`
and the five finals. U_k = the OPLoRA run's saved bases; probes = `splits/task1/probe_ids.json`, with the exact
prompts from the baseline's logged lm-eval samples (so `eval_baseline` must have run). Choices in
[docs/decisions.md](docs/decisions.md).

```bash
python -m task1.diagnostics --smoke --device cpu      #tiny random model + random updates; needs the eval smoke first
python -m task1.diagnostics --sync_repo <user>/atml-pa3   #real run (queue job `diagnostics`, gpu0 group)
```

### Qualitative comparison (Step 7)

All 24 prompts of `first_plot_questions.yaml` × 7 conditions, greedy, thinking off, 512 new tokens, one left-padded
batch per condition, same merged FP16 models as Step 5. Choices in [docs/decisions.md](docs/decisions.md).

```bash
python -m task1.qualitative --smoke --device cpu      #tiny random model, 8 new tokens
python -m task1.qualitative --sync_repo <user>/atml-pa3   #real run (queue job `qualitative`, gpu1 group)
python -m task1.qualitative --collect --sync_repo <user>/atml-pa3   #locally: pull responses, build grouped files + sheet
```

Outputs in `results/task1/qualitative/`: `<condition>.jsonl` (prompt, response, n_tokens, truncated),
`responses_by_question.{json,md}` (grouped by question, then format, then condition), `annotation_prompt_ids.json`; and
`results/task1/annotation_sheet.csv` (42 rows: first and last question × 3 formats × 7 conditions, with the response and
an automatic `truncated` column; the annotation, excerpt and notes columns are left empty and an existing sheet is
never overwritten).

### Tables and figures

Collects every Task 1 result (pulling only small result files from the HF branches when they aren't on disk) into
`results/task1/`: `summary.csv` (condition × benchmark: score, stderr, Δ pp vs baseline, raw acc, GSM8K truncated /
extract-failed counts, eval time, peak VRAM), `summary_wide.{csv,md}`, `resources.{csv,md}`, `diagnostics.{csv,md}`,
and `figures/` (`sft_loss`, `l2sp_terms`, `benchmark_scores`, each PNG + PDF). Runs on CPU in seconds.

```bash
python -m task1.make_tables --sync_repo <user>/atml-pa3
```

### Outputs

- `outputs/task1/<run>/` (not committed, synced to the HF repo branch `task1-<run>`): resumable checkpoints
  (`rolling/`, `milestones/`, each with a sha256 `manifest.json`), and `export/update_XXXXX/` with what evaluation
  needs (FP16 HF model for full FT at 625/1250, adapter factors for LoRA/OPLoRA at 1250; OPLoRA exports the projected
  factors). `outputs/task1/oplora_bases.pt`: the frozen SVD bases.
- `results/task1/eval/<condition>/` (synced to branch `task1-eval-<condition>`): `<benchmark>.json` (metric, stderr,
  raw acc where it exists, n, settings, lm-eval config, versions, time, peak VRAM), `<benchmark>_samples.jsonl` (every
  lm-eval sample), `gsm8k_table.jsonl` (id, question, target, generation, extracted, correct, n_tokens, truncated,
  extract_failed), `model_info.json` (source export, merge check for adapter runs).
- `results/task1/diagnostics/` (synced to branch `task1-diagnostics`): `weight_metrics.csv`, `activation_ratios.csv`,
  `activation_summary.csv`, `probe_tokens.csv`, `diagnostics_info.json`. Cached z in `outputs/task1/diagnostics/`.
- `results/task1/<run>/`: `config.json`, `train_log.jsonl` (one row per update: SFT loss, LR, grad norm, scaler
  scale, skipped, tokens; L2-SP also Ω, λΩ, total loss; displacement at 625/1250), `<run>_history.json`,
  `train_summary.json` (time, peak VRAM, trainable params, checkpoint sizes, method overhead).

### Attribution

No external code is copied so far. Evaluation runs EleutherAI's lm-evaluation-harness 0.4.13 as a library with its
stock task configs. The LoRA initialization (Kaiming-uniform A with a=√5, zero B) follows the PEFT
default, reimplemented in `task1/methods/lora.py`. `first_plot_questions.yaml` (Step 7 prompts) is copied unchanged
from [emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment)
(`evaluation/first_plot_questions.yaml` at commit `e40daef5791139a2417e22034ab0a0e72900524b`, MIT licence; Betley et
al., 2025, "Emergent Misalignment"). The handout names the file but the course did not ship it; its sha256 is pinned in
`configs/task1/qualitative.yaml`. Task 2's answer normalization (`task2/vqa.py`) is adapted from the official VQA
evaluation code ([GT-Vision-Lab/VQA](https://github.com/GT-Vision-Lab/VQA), `vqaEval.py`: punctuation, number-word,
article and contraction rules).
