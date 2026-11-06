# Protocol: T-free vs per-T specialists (5-way fair comparison) — 2026-11-07

STATUS: **IN PROGRESS**. Only the T=16 specialist is training (started
2026-11-07 00:40:39). NO comparison numbers exist yet. The matrix files are
deliberately NOT written until `eval_*.csv` files contain real measurements.

## Goal
Answer claim (B): does ONE T-free weight set lose accuracy at each T versus a
specialist trained ONLY on that T with identical compute? Needs a 5 x 4 PSNR/SSIM
matrix on REAL benchmark scenes.

Claim (A) ("one weight set reconstructs multiple T") is already answered by a
separate real experiment: `results/ablation/README_tfree_realT_20261107.md`,
commit `144b955`. That experiment does NOT answer claim (B).

## Scope decision
The user was offered strict / fast / pilot options and skipped the question.
The assistant proceeded with the STRICTEST option: retrain T-free too, so all
five rows are produced by one recorded command template. No numbers are
asserted until each `history.json` / `eval_*.csv` actually exists.

## Why every existing specialist weight is unusable
- `output_e2e_T8/T16/best_model.pth` (1.3 MB): different architecture
  (nf=48, nb=2, E2E network from `sci_ablation/train_e2e.py`), not DD-MADUN v3.
- `output_t16_only/best_model.pth`: correct architecture but trained on
  **synthetic-only** data (`DAVIS sequences: 0`), EMA never converged -> invalid.
- `checkpoints/best_model.pth`: legacy T-bound model, epoch 145, different
  data/loss/epoch; only supports T=8.
- No T=24 / T=32 specialist weights exist anywhere.
=> All 4 specialists must be trained from scratch with one recorded command template.

## Compute alignment (frames_per_batch=32, 50 step/epoch x 30 = 1500 steps)
`n_batches = n_train // (frames_per_batch // T_LIST[0])`
| model | T_list | n_train | n_batches | frames/step |
|-------|--------|---------|-----------|-------------|
| T-free | 8 16 24 32 | 200 | 200//4 = 50 | 32 (T=24 step: 24) |
| spec T=8  | 8  | 200 | 200//4 = 50 | 32 |
| spec T=16 | 16 | 100 | 100//2 = 50 | 32 |
| spec T=24 | 24 | 50  | 50//1  = 50 | 24 |
| spec T=32 | 32 | 50  | 50//1  = 50 | 32 |
Note: T=24 gets 24 frames/step because B=32//24=1. The T-free run has the
same property for its T=24 steps, so both sides stay consistent.

## Single recorded training command template (all 5 models identical except T_list/n_train)
```
python dd_madun_v3_train_tfree.py \
  --T_list <T> --epochs 30 --n_train <N> \
  --warm_start checkpoints/best_model.pth --legacy_T 8 \
  --davis_root ../DAVIS --real_ratio 0.7 \
  --amp --ckpt --vgg_weight 0.02 \
  --out_dir <out_dir> --log_every 10
```
- cwd = `optics_main/`. `--davis_root ../DAVIS` VERIFIED 2026-11-07: 90 sequences
  found, 90/90 >= 24 frames, 89/90 >= 32 frames, real clips loaded at all four T.
- VERIFIED in `perT_sweep/T16/train.log` (the run actually in progress):
  `DAVIS sequences: 90`, `Train: 70 real + 30 synth = 100`,
  `Loaded 70 real video clips`. This fixes the 2026-11-06 `0 real + 100 synth` bug.
- Same warm_start, same seed, same optimizer/scheduler, same AMP, same vgg_weight.
- Each run writes its own command.txt, train.log, history.json, checkpoint.pth.

## Evaluation design (removes the two flaws found on 2026-11-06)
1. Same data for every cell: git-tracked `benchmark/*.mat` 6 real scenes,
   `sci_ablation/run_ablation.py --exp T --Ts 8 16 24 32`, cell=1, noise=0.01,
   shear=1, disjoint, seed=0, cuda. NO synthetic test_clips anywhere.
2. Same weight choice for every model. Evaluate BOTH and report both:
   - `checkpoint.pth` -> `model_state_dict` = **non-EMA final model** (primary)
   - `best_model.pth` -> `model_state_dict` = **EMA weights** (secondary)
   This directly fixes the EMA-convergence mismatch (the invalid T=16 run had
   model P=31.06 but EMA=19.78, while T-free had EMA converged at 34.66).
   NOTE: `dd_madun_v3_train_tfree.py`'s own final `per_T` block evaluates
   `ema_model` only, so those numbers are NOT used for the comparison.
3. All weights are T-free-architecture (trained by `dd_madun_v3_train_tfree.py`),
   so `TFreeMethod` + `resolve_model_cfg` loads every one with the same code path,
   including off-diagonal T, which measures each specialist's T-generalization.
   Verified: `checkpoint.pth` contains `cfg` (K/K1/ng/nf/nb/temporal) and the
   T-free head has cin=2, so `legacy_T` resolves to None.

Output: `perT_sweep/matrix_psnr.csv` and `perT_sweep/matrix_ssim.csv`,
one row per weight variant, columns T=8/16/24/32, 6-scene means.
Diagonal = specialist at its own T vs T-free at that T -> the claim-(B) answer.
Off-diagonal = how badly specialists collapse at other T.
`report()` refuses to write all-NaN placeholder matrices.

## Reference row (already MEASURED, not estimated)
From `results/ablation/tfree_realT_20261107.csv` (85 rows, no `err`), weight
`output_tfree/best_model.pth`, sha256 `48d69188...`, same 6 real scenes:

| T | 8 | 16 | 24 | 32 |
|---|---|----|----|----|
| tfree PSNR (dB) | 28.0392 | 25.0768 | 23.4877 | 23.0745 |
| tfree SSIM | 0.973024 | 0.951739 | 0.934869 | 0.928518 |

These were re-parsed and validated on 2026-11-07 by `perT_sweep._means()`.
They are the numbers the specialists must be compared against, NOT a prediction.

## Measured cost (real numbers from the running job, 2026-11-07 00:50)
`perT_sweep/T16/train.log` with DAVIS real data:
- step 0 at 10s, step 40 at 405s, epoch 1 finished at 495s (incl. eval),
  step 60 at 590s, step 100 at 895s -> steady state ~7.6-8.8 s/step
  (first epoch is slower: cuDNN autotune / data warmup).
- 1500 steps -> ~3.2-3.7 h/model + 30 epoch evals -> **~3.3-3.8 h**.
- 5 models serial, GPU exclusive -> **~16-19 h total** (not the 10-14 h estimated
  before DAVIS was enabled; the earlier synthetic-only T=16 run was 2.12 h).
- 2026-11-07 00:55 state: epoch 3/30, step 100, `checkpoint.pth` at epoch 2,
  GPU 100% / 32086 MiB.

## Execution mechanism (`perT_sweep.py`)
Background bash tasks in this environment are killed around ~2 h, so the sweep
is chained ONE MODEL PER INVOCATION:
```
python perT_sweep.py status          # epoch-level progress, no fabrication
python perT_sweep.py next --dry-run  # what would run
python perT_sweep.py next            # start first unfinished model, or refuse if busy
python perT_sweep.py supervise       # OWNS the chain: wait -> train -> ... -> eval
python perT_sweep.py eval            # after all 5 weights exist
python perT_sweep.py report          # reprint matrices from existing CSVs
```
- A supervisor was launched 2026-11-07 00:55 (background task `OF5FRi`,
  `--poll 120 --budget-hours 24`). It waits while T=16 trains, then
  automatically starts T8 -> T24 -> T32 -> tfree, then runs `eval` and writes
  the matrices. It terminates when the matrix is produced. If its task is
  killed, re-invoke `python perT_sweep.py supervise` (or `next`) and the
  stale lock is ignored automatically.
- `supervisor.lock` ensures only ONE owner of the chain; `train.lock` marks the
  active trainer. `next` refuses if either a supervisor or a trainer is alive.
- `next` refuses to start if any live `dd_madun_v3_train_tfree.py` process exists.
- Every epoch writes `checkpoint.pth` (model + EMA + optimizer + scheduler +
  best_ema + history), so a killed run resumes losslessly via `--resume`.
- `is_done()` uses `history.json.per_T` (written once, at the very end), NOT
  `best_epoch >= 30` -- EMA can peak earlier, which would falsely mark a
  finished run as unfinished and overwrite its history.
- Code fix 2026-11-07: `--resume` now restores `history` and `checkpoint.pth`
  stores it, so a resumed run still writes the FULL 30-epoch training curve.
  The running T=16 job started before this fix; if it is killed and resumed,
  its first 1..N epochs may be missing from `history.json` (the weights and
  optimizer state are still exact). This will be noted if it happens.

## Scope limits / known confounds (must be stated with any result)
- Real video FRAMES; measurements still SIMULATED (`ExperimentCUP.measure` +
  Gaussian noise). No hardware CUP capture, and `mat['meas']`/`mat['mask']`
  are never read.
- Step-count alignment follows Mark's rule. It does NOT equalize the number of
  DISTINCT training clips: `n_per_T = n_train // len(T_LIST)` gives the T-free
  run 50 clips per T, while the T=8 specialist has 200 and T=16 has 100
  (T=24/T=32 specialists have 50). The specialists therefore have a data-
  diversity advantage at equal gradient steps. They are an UPPER-BOUND control,
  not a perfectly matched control; if T-free is close, the conclusion is strong,
  if specialists win, compute and data diversity cannot be separated.
- Even a clean result only answers claim (B) at T=8/16/24/32 on these 6 scenes.
- NEVER fill a missing measurement with a plausible number. If a comparison
  was not run, the cell stays `nan` and is reported as "not measured".
