# T-free real-scene evaluation — 2026-11-07

## Purpose
Demonstrate T-free validity on REAL video frames (not synthetic test clips) with a
single, recorded, reproducible protocol. This replaces the fabricated "T=16 ~19.2 dB"
comparison sent to Slack on 2026-11-06, which had no measurement behind it.

Question answered: can ONE set of T-free weights reconstruct video at multiple T,
including T values NEVER seen during training?

Question NOT answered here: whether T-free matches per-T specialist models.
That still needs a fair T=16 retrain with `--davis_root ../DAVIS`.

## Frozen inputs (recorded BEFORE running)
- Weight: `optics_main/output_tfree/best_model.pth`
  - sha256 `48d691888465ae113651dab3562a5468bc9470f4cda8a4713b19385d298b164a`
  - epoch 30, ema_psnr 34.6583, `T_list=[8,16,24,32]`, temporal=none,
    cfg K=8 K1=3 ng=6 nf=96 nb=8 H=W=256
  - identical copy at `optics_main/checkpoints/tfree_trained/best_model.pth` (same sha256)
- Data: `benchmark/{aerial,crash,drop,kobe,runner,traffic}.mat`, `orig` real video frames
  (256x256x32/40/48). These 6 .mat ARE tracked in git -> reproducible by Mark.
  NOTE: `mat['meas']` / `mat['mask']` are NOT used.
- Code: `sci_ablation/run_ablation.py` + `sci_ablation/data.py` + `sci_ablation/methods.py`
  at commit `e659232`.
- Device: cuda (never CPU).

## Fixed protocol (identical for every row)
`--exp T`, cell=1, noise=0.01, shear=1, `--group disjoint`, `--seed 0`,
H=W=256, `--device cuda`. Measurements E = `ExperimentCUP.measure(gt)`
(i.e. SIMULATED forward measurement + Gaussian noise, not real CUP camera data).

T grid: `8 12 16 20 24 28 32`
- 8/16/24/32 = T values seen during T-free training
- 12/20/28 = T values NEVER seen during training (the generalization test)

## Commands (executed 2026-11-07, cwd = optics_main)
```
python -m sci_ablation.run_ablation --exp T --Ts 8 12 16 20 24 28 32 \
  --methods tfree --ckpt tfree=output_tfree/best_model.pth --device cuda \
  --out results/ablation/tfree_realT_20261107.csv

python -m sci_ablation.run_ablation --exp T --Ts 8 12 16 20 24 28 32 \
  --methods admm_tv --device cuda \
  --out results/ablation/admmtv_realT_20261107.csv

python -m sci_ablation.run_ablation --exp T --Ts 8 \
  --methods ddmadun --device cuda \
  --out results/ablation/ddmadun_T8_20261107.csv
```
`ddmadun` is the legacy T-bound model (checkpoints/best_model.pth); it only supports
T=8 and error-s out on other T, which is itself the motivation for T-free.

## Results (measured 2026-11-07 00:26-00:28 CST, 1m56s total, GPU)
All three CSVs: `err` column empty for every row (tfree 85, admm_tv 85, ddmadun 28 rows).

6-scene mean PSNR (dB), cell=1, noise=0.01, disjoint windows:

| T | tfree | admm_tv | delta | windows | trained T? |
|---|-------|---------|-------|---------|------------|
| 8  | 28.04 | 25.87 | +2.17 | 28 | yes |
| 12 | 26.54 | 24.23 | +2.31 | 16 | **NO (unseen)** |
| 16 | 25.08 | 22.72 | +2.36 | 13 | yes |
| 20 | 25.26 | 22.28 | +2.98 | 9  | **NO (unseen)** |
| 24 | 23.49 | 20.90 | +2.59 | 7  | yes |
| 28 | 23.60 | 20.89 | +2.72 | 6  | **NO (unseen)** |
| 32 | 23.07 | 20.29 | +2.79 | 6  | yes |

6-scene mean SSIM: tfree 0.9730/0.9618/0.9517/0.9502/0.9349/0.9364/0.9285
vs admm_tv 0.9571/0.9378/0.9158/0.8972/0.8720/0.8653/0.8422 for the same T grid.

T=8 three-way, identical scenes and windows (mean): ddmadun (T-bound) **30.13**,
tfree **28.04**, admm_tv **25.87**. Per scene the T-bound model beats tfree by
~1.5-3 dB at every one of the 6 scenes. NOTE: ddmadun is a legacy model from a
completely different training run (epoch 145, different data/loss), so this gap
is NOT a clean measure of the cost of T-generality.

T-bound model cannot run at T=16 at all (verified):
`tfree -> construct OK (TFreeMethod)`;
`ddmadun -> RuntimeError: size mismatch for dm: shape torch.Size([8...])`.

Also verified: these T=8 numbers do NOT match the earlier Nyquist `method=tfree`
rows (e.g. aerial 24.79 here vs 20.54 there), confirming the earlier scan used a
different, unrecorded weight. This run's weight is pinned by sha256.

## Scope limits (do not overclaim)
- Real video FRAMES, but SIMULATED measurements. Not a hardware CUP experiment.
- Proves claim (A): one weight set reconstructs many T, incl. unseen T, beating
  the iterative baseline on real scenes. Does NOT prove claim (B): that T-free
  matches per-T specialists. The T=8 three-way result actually points the other way.
- A fair claim-(B) test still needs the T=16 retrain with `--davis_root ../DAVIS`
  and a single shared weight/EMA protocol for both sides.
