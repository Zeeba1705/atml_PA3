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
`gpu0`: full_ft → l2sp, `gpu1`: lora16 → lora4 → oplora16 (see `kaggle/task1_jobs.yaml`).

```bash
python kaggle/task1_queue.py --status --sync_repo <user>/atml-pa3
python kaggle/task1_queue.py --group gpu0 --gpu 0 --sync_repo <user>/atml-pa3
```

### Outputs

- `outputs/task1/<run>/` (not committed, synced to the HF repo branch `task1-<run>`): resumable checkpoints
  (`rolling/`, `milestones/`, each with a sha256 `manifest.json`), and `export/update_XXXXX/` with what evaluation
  needs (FP16 HF model for full FT at 625/1250, adapter factors for LoRA/OPLoRA at 1250; OPLoRA exports the projected
  factors). `outputs/task1/oplora_bases.pt`: the frozen SVD bases.
- `results/task1/<run>/`: `config.json`, `train_log.jsonl` (one row per update: SFT loss, LR, grad norm, scaler
  scale, skipped, tokens; L2-SP also Ω, λΩ, total loss; displacement at 625/1250), `<run>_history.json`,
  `train_summary.json` (time, peak VRAM, trainable params, checkpoint sizes, method overhead).

### Attribution

No external code is copied so far. The LoRA initialization (Kaiming-uniform A with a=√5, zero B) follows the PEFT
default, reimplemented in `task1/methods/lora.py`.
