# T=16-only vs T-free: 6-scene benchmark comparison

## Background

Mark (12:57) requested a T=16-only DD-MADUN as the upper bound for "T-free's cost":
same architecture, same epoch count, same total training steps, but trained on a
single T=16 instead of the multi-T list [8,16,24,32].

## Training summary

| | T-free | T=16-only |
|---|---|---|
| T_list | [8,16,24,32] | [16] |
| epochs | 30 | 30 |
| n_train | 200 | 100 |
| steps/epoch | 50 | 50 |
| total steps | 1500 | 1500 |
| warm start | checkpoints/best_model.pth (T=8 legacy) | same |
| vgg_weight | 0.02 | 0.02 |
| AMP | yes | yes |
| **Training data** | **DAVIS 140 real + 60 synth = 200** | **0 real + 100 synth = 100** |
| training time | ~7.7 h | ~2.1 h |
| best EMA PSNR (internal) | 34.66 dB | 19.78 dB |

> **Critical confound**: T=16-only training could NOT find DAVIS
> ("DAVIS sequences: 0" in log) and fell back to synthetic-only data (100 clips).
> T-free training used 140 real DAVIS clips + 60 synthetic.
> The comparison is **not apples-to-apples** due to training data difference.

## 6-scene benchmark results (cell=1, noise=0.01, seed=0, disjoint)

### 6-scene mean PSNR (dB)

| T | T-free | T=16-only | Δ (T-free − T16) | T=16 seen? |
|---|---|---|---|---|
| 8 | **28.04** | 23.62 | +4.42 | T-free: yes, T16: no |
| 16 | **25.08** | 21.15 | +3.93 | both: yes |
| 24 | **23.49** | 19.03 | +4.46 | T-free: yes, T16: no |

### Per-scene PSNR (dB)

| Scene | T | T-free | T=16-only | Δ |
|---|---|---|---|---|
| aerial | 8 | 24.79 | 23.19 | +1.60 |
| aerial | 16 | 23.25 | 21.45 | +1.80 |
| aerial | 24 | 22.21 | 20.52 | +1.69 |
| crash | 8 | 25.53 | 23.48 | +2.05 |
| crash | 16 | 23.85 | 21.83 | +2.02 |
| crash | 24 | 23.27 | 19.63 | +3.64 |
| drop | 8 | 36.87 | 27.71 | +9.16 |
| drop | 16 | 33.13 | 25.47 | +7.66 |
| drop | 24 | 30.77 | 20.66 | +10.11 |
| kobe | 8 | 27.98 | 23.21 | +4.77 |
| kobe | 16 | 25.08 | 20.69 | +4.39 |
| kobe | 24 | 23.91 | 19.98 | +3.93 |
| runner | 8 | 31.03 | 25.61 | +5.42 |
| runner | 16 | 27.90 | 22.00 | +5.90 |
| runner | 24 | 26.12 | 20.32 | +5.80 |
| traffic | 8 | 22.06 | 19.20 | +2.86 |
| traffic | 16 | 19.86 | 17.35 | +2.51 |
| traffic | 24 | 19.07 | 16.03 | +3.04 |

## Key findings

1. **T-free dominates T=16-only at every T**, including T=16 (the T=16-only model's
   trained T). The gap is +3.93 dB at T=16, +4.42 dB at T=8, +4.46 dB at T=24.

2. **This is NOT a fair comparison** because of the training data confound:
   - T-free: 140 real DAVIS clips + 60 synthetic = 200 training samples
   - T=16-only: 0 real DAVIS + 100 synthetic = 100 training samples
   - The T=16-only model trained on half the data, all synthetic, no real video.

3. **T=16-only model generalizes poorly to unseen T** (T=8, T=24):
   - At T=8: 23.62 dB (vs T-free 28.04, a 4.42 dB gap)
   - At T=24: 19.03 dB (vs T-free 23.49, a 4.46 dB gap)
   - Even at its trained T=16: only 21.15 dB (vs T-free 25.08)

4. **The T-free model shows no "cost" from multi-T training** — it beats the
   single-T model at the single-T model's own trained T. This is consistent with
   multi-T training acting as data augmentation (more diverse temporal compression
   ratios → better generalization).

5. **Caveat**: The T=16-only result is a lower bound on the true T=16 upper bound.
   A fair comparison requires retraining with DAVIS found (same 200 training samples,
   same real/synthetic ratio as T-free). The DAVIS path issue must be fixed first.

## Next steps

1. **Fix DAVIS path for T=16-only retraining** — the training log shows
   "DAVIS sequences: 0" because DAVIS-2017 was not found from the optics_main
   worktree. Need to point `--davis_root` to the correct path.

2. **Retrain T=16-only with DAVIS** (same n_train=200, 30 epochs, same data as T-free)
   to get a fair upper bound.

3. **Cell fine-tune experiment** (Mark's request): from T-free best_model,
   `--cells 1 2 4 8`, to test whether random cell augmentation closes the gap
   with ADMM-TV at large cells.

## Files

- `tfree_vs_t16_tfree.csv` — T-free model, 6 scenes × T=8/16/24
- `t16_only_vs_tfree.csv` — T=16-only model, 6 scenes × T=8/16/24
