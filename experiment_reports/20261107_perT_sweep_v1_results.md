# RESULTS — T-free vs per-T specialists (5-way sweep), 2026-11-07

STATUS: **COMPLETE, measured**. All 5 models trained (exit 0, 30 epochs each),
12 weight variants evaluated on 6 real benchmark scenes at T=8/16/24/32.
648 eval rows, **0 errors**. Raw data: `perT_sweep/eval_*.csv`,
`perT_sweep/matrix_psnr.csv`, `perT_sweep/matrix_ssim.csv`,
`perT_sweep/manifest.json`, `perT_sweep/supervisor_stdout.log`.

All numbers below are 6-scene **flat-window means** (mean over all disjoint
windows; scene-first aggregation gives different values — always name it).
Primary comparison uses **non-EMA `checkpoint.pth`** weights; EMA rows are
reported but are systematically depressed (see "EMA audit").

---

## 1. Answer to claim (B)

> "Does ONE T-free weight set lose accuracy at each T versus a specialist
> trained ONLY on that T with identical compute?"

**At strictly equal compute (all five models retrained with one command
template, 1500 steps each): essentially no.**

| T | specialist (own T) | fresh T-free | delta (spec − tfree) | specialist wins |
|---|---|---|---|---|
| 8  | 28.42 | 26.85 | **+1.57 dB** | 6/6 scenes |
| 16 | 24.85 | 24.39 | **+0.46 dB** | 6/6 scenes |
| 24 | 22.80 | 22.93 | **−0.13 dB** (T-free wins) | 1/6 scenes |
| 32 | 22.36 | 22.21 | **+0.15 dB** | 1/6 scenes |

Mean absolute diagonal gap **0.58 dB**. SSIM gaps are +0.0076 / +0.0040 /
−0.0014 / −0.0009 — negligible. Using EMA weights instead, the diagonal gaps
shrink to +0.26 / +0.07 / −0.17 / +0.16 dB, i.e. the same conclusion.

The only non-trivial loss is at T=8 (+1.57 dB), where the T=8 specialist also
had **4× as many distinct training clips** (200 vs 50 per T for T-free), so
compute/diversity are confounded (see §6). At T=24/32 T-free is at parity or
ahead despite that disadvantage.

**One T-free weight set replaces four specialists at ~0.6 dB mean cost.**

## 2. Best-of-4-specialists vs one T-free model (PSNR dB, non-EMA)

| T | best specialist | best spec | fresh tfree | existing tfree |
|---|---|---|---|---|
| 8  | T8  | 28.42 | 26.85 (−1.57) | 28.10 (−0.32) |
| 16 | T16 | 24.85 | 24.39 (−0.46) | 25.11 (**+0.26**) |
| 24 | T24 | 22.80 | 22.93 (**+0.13**) | 23.52 (**+0.72**) |
| 32 | T32 | 22.36 | 22.21 (−0.15) | 23.13 (**+0.77**) |

`existing_tfree` = `output_tfree/*`, the published T-free weights used in the
7-T real-scene experiment (commit `144b955`). **It had 4× the compute — see
§5 — so this row is an upper bound, NOT an equal-compute comparison.**

## 3. Specialists collapse off-diagonal; T-free does not

PSNR of each specialist at *other* T values, with its gap vs fresh T-free:

```
T8  specialist -> T=16: 21.59 (-2.80)   T=24: 17.91 (-5.02)   T=32: 16.79 (-5.41)
T16 specialist -> T=8 : 27.12 (+0.27)   T=24: 21.97 (-0.96)   T=32: 19.80 (-2.41)
T24 specialist -> T=8 : 25.34 (-1.51)   T=16: 24.00 (-0.39)   T=32: 21.63 (-0.58)
T32 specialist -> T=8 : 24.56 (-2.28)   T=16: 23.27 (-1.12)   T=24: 22.52 (-0.41)
```

T=8→T=32 PSNR drop: T8 specialist **11.62 dB**, fresh T-free **4.64 dB**,
existing T-free **4.97 dB**. Low-T specialists are strongly T-bound; the
T-free models degrade only with the intrinsic difficulty of larger T.

## 4. Full 12 × 4 matrices (6-scene mean)

PSNR (dB), non-EMA / EMA:

| weight | T=8 | T=16 | T=24 | T=32 |
|---|---|---|---|---|
| T8_nonEMA | 28.42 | 21.59 | 17.91 | 16.79 |
| T8_EMA | 24.94 | 20.95 | 18.37 | 17.44 |
| T16_nonEMA | 27.12 | 24.85 | 21.97 | 19.80 |
| T16_EMA | 25.25 | 22.12 | 19.86 | 18.68 |
| T24_nonEMA | 25.34 | 24.00 | 22.80 | 21.63 |
| T24_EMA | 24.41 | 21.87 | 20.13 | 19.27 |
| T32_nonEMA | 24.56 | 23.27 | 22.52 | 22.36 |
| T32_EMA | 24.16 | 21.79 | 20.25 | 19.64 |
| **tfree_nonEMA** (fresh, 1500 steps) | 26.85 | 24.39 | 22.93 | 22.21 |
| tfree_EMA (fresh) | 24.68 | 22.05 | 20.30 | 19.48 |
| **existing_tfree_nonEMA** (6000 steps) | 28.10 | 25.11 | 23.52 | 23.13 |
| existing_tfree_EMA (6000 steps) | 28.04 | 25.08 | 23.49 | 23.08 |

SSIM:

| weight | T=8 | T=16 | T=24 | T=32 |
|---|---|---|---|---|
| T8_nonEMA | 0.9750 | 0.9175 | 0.8153 | 0.7704 |
| T8_EMA | 0.9512 | 0.8862 | 0.8145 | 0.7734 |
| T16_nonEMA | 0.9692 | 0.9500 | 0.9213 | 0.8792 |
| T16_EMA | 0.9551 | 0.9053 | 0.8553 | 0.8134 |
| T24_nonEMA | 0.9576 | 0.9428 | 0.9264 | 0.9130 |
| T24_EMA | 0.9469 | 0.9000 | 0.8568 | 0.8241 |
| T32_nonEMA | 0.9506 | 0.9351 | 0.9229 | 0.9168 |
| T32_EMA | 0.9444 | 0.8986 | 0.8580 | 0.8296 |
| tfree_nonEMA (fresh) | 0.9674 | 0.9460 | 0.9278 | 0.9178 |
| tfree_EMA (fresh) | 0.9500 | 0.9042 | 0.8620 | 0.8309 |
| existing_tfree_nonEMA | 0.9734 | 0.9522 | 0.9357 | 0.9297 |
| existing_tfree_EMA | 0.9730 | 0.9517 | 0.9349 | 0.9285 |

## 5. CRITICAL CORRECTION — the original T-free command WAS recoverable

Project memory previously recorded "no original command, no stdout log;
n_train=200 INFERRED". That inference is **wrong**. Older workspace memory
(`2026-11-02-21-49-24/.workbuddy-ai/memory/2026-11-02.md` lines 194-204, 278-286
and `2026-11-03.md` lines 27-35, 67-71) records the launch verbatim:

```
python dd_madun_v3_train_tfree.py --warm_start checkpoints/best_model.pth \
  --legacy_T 8 --davis_root DAVIS --out_dir output_tfree --amp \
  --ckpt --ckpt_blocks --epochs 30 --n_train 800 \
  --T_list 8 16 24 32 --warmup_epochs 3 --log_every 20
```

Evidence that this is exactly the run producing `output_tfree/*`:
- The 11-02 memory quotes epoch 1 of that run: `L=0.2725 P=25.80/0.9617
  EMA=16.71/0.5961`. `output_tfree/history.json` epoch 1 is
  `loss=0.272545, psnr=25.8027, ssim=0.9617, ema_psnr=16.7113,
  ema_ssim=0.5961` — an exact match.
- File timestamps: `output_tfree/*` = 2026-11-03 13:07; launch was 11-03
  05:26, matching the recorded "~7 h" ETA for n_train=800.
- `n_batches = 800 // (32//8) = 200` steps/epoch × 30 = **6000 steps**
  (explicitly computed in that memory), not 1500.
- Original `checkpoint.pth` lacks `optimizer_state_dict` / `scheduler_state_dict`
  / `history` / `best_ema`, i.e. saved by an older script version.

**Consequences:**
1. The 1500-step "step alignment" used in this sweep was aligned to a wrong
   inference. The published T-free model had **4× the gradient steps and 4×
   the training clips** of the fresh T-free / specialists here.
2. Therefore §1 (fresh tfree vs specialists) is the only strictly controlled
   claim-(B) answer. §2's `existing_tfree` column is confounded by 4× compute
   and must NEVER be presented as an equal-compute victory.
3. Other config differences vs the original: `--warmup_epochs 3` (ours used
   the default 10), `--ckpt_blocks` (ours off), `--davis_root DAVIS` from the
   root worktree (ours `../DAVIS` from `optics_main`). Same DAVIS data, though.
4. Retraining T-free with the shared template was essential — without it there
   would be no controlled T-free row at all.

## 6. EMA audit — why non-EMA is primary

`update_ema` runs once per optimizer step with decay 0.999. Residual weight on
the initial warm-start after N steps is `0.999^N`: 1500 steps → 0.223,
6000 steps → 0.0025. Measured relative L2 distance `|model−EMA|/|model|`:

| weights | distance | `best_model == ckpt.ema` |
|---|---|---|
| original `output_tfree` | **0.0615** | yes (0.000000) |
| fresh `perT_sweep/tfree` | **0.4023** | yes (0.000000) |
| fresh `perT_sweep/T16` | **0.4059** | yes (0.000000) |

All five 1500-step retrains have an unconverged EMA (epoch-30 non-EMA 27–35 dB
vs EMA 18–24 dB on the script's synthetic validation); the original 6000-step
run converged (non-EMA 34.88 / EMA 34.66). **The training script's own
`per_T` block evaluates `ema_model`, so those numbers are unusable for any
30-epoch/1500-step run.** Both weight variants are reported here; non-EMA is
the fair primary.

## 7. Reproducibility audit

- All 12 `eval_*.csv`: 54 rows each, **648 total, 0 `err`, 0 NaN**, columns
  include `sec` runtime and `group_start`.
- Pipeline cross-check: `eval_existing_tfree_EMA.csv` vs the INDEPENDENT
  `results/ablation/tfree_realT_20261107.csv` (commit `144b955`, run 11-07
  00:26). Per-T PSNR means agree to **+0.0006 / +0.0008 / +0.0001 / +0.0013 dB**
  at T=8/16/24/32. The new sweep reproduces the earlier measurement.
- Training: each `perT_sweep/{T8,T16,T24,T32,tfree}/train.log` shows
  `DAVIS sequences: 90` and real clips (`140`/`70`/`35`/`35`/`35` real).
  All exit 0 and wrote `history.json` with `per_T`.
- Durations (manifest): T16 3.37 h (manual), T8 2.64 h, T24 1.53 h,
  T32 3.93 h, tfree 2.10 h. Supervisor ran 00:55:39 → 14:25:34 (13.5 h),
  exit 0, chained all models then ran `eval` automatically.

## 8. Confounds / scope limits (MUST accompany any claim)

- **Real video FRAMES, SIMULATED measurements.** `ExperimentCUP.measure(gt)` +
  Gaussian noise; `mat['meas']`/`mat['mask']` never read. No hardware CUP data.
- **Equal compute ≠ equal data.** `n_per_T = n_train // len(T_LIST)`: fresh
  T-free sees 50 distinct clips per T; T8 specialist 200, T16 100, T24/T32 50.
  Specialists are an **upper-bound control** at T=8/16. If T-free is close
  anyway (it is), the conclusion is strong; the +1.57 dB at T=8 cannot be
  attributed purely to T-generality.
- **Published T-free is NOT compute-matched** (§5): 6000 vs 1500 steps.
- **EMA is systematically depressed in every retrained model** (§6). EMA rows
  are informational; diagonal conclusions are unchanged either way.
- Only 4 T values, 6 scenes, cell=1, noise=0.01, shear=1, disjoint windows,
  seed=0. Unseen-T generalization (12/20/28) is established separately in
  `20261107_tfree_realT.md`, not here.
- Aggregation is flat-window mean; scene-first means differ.
- **No fabricated values.** Every cell came from an `eval_*.csv`; missing data
  would be reported as "not measured".
