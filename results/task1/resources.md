| run | method | trainable_params | trainable_pct | train_hours | checkpoint_hours | peak_vram_gb | checkpoint_trainable_mb | checkpoint_total_mb | export_mb | svd_seconds | bases_mb | theta0_mb | skipped_updates | sessions | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_ft | full_ft | 596049920 | 100.0 | 2.322 | 0.24 | 12.55 | 2384.3 | 7153.0 | 1192.1 |  |  | 2384.2 | 0 | 1 | Tesla T4 |
| l2sp | l2sp | 596049920 | 100.0 | 2.451 | 0.239 | 12.55 | 2384.3 | 7153.0 | 1192.1 |  |  | 2384.2 | 0 | 1 | Tesla T4 |
| lora4 | lora | 2523136 | 0.423 | 2.634 | 0.003 | 3.37 | 10.2 | 30.7 | 10.1 |  |  |  | 0 | 1 | Tesla T4 |
| lora16 | lora | 10092544 | 1.693 | 2.727 | 0.008 | 3.5 | 40.5 | 121.5 | 40.4 |  |  |  | 0 | 1 | Tesla T4 |
| oplora16 | oplora | 10092544 | 1.693 | 3.341 | 0.008 | 3.82 | 40.5 | 121.5 | 40.4 | 31.5 | 323.0 |  | 0 | 1 | Tesla T4 |
