# perT sweep v2：T-free 与 4 个专训模型的对比（等步数、等数据）

日期：2026-11-16 至 11-17（UTC）。所有数字取自 `perT_sweep_v2/eval/*.csv`，
汇总方式为：对 6 个评估场景的所有不重叠窗口直接取平均（与 v1 报告的口径相同）。

## 1. 结论

1. **通用模型的代价随 T 增大而单调减小，到 T≥24 时为零。** 专训模型在自己的 T 上减去 T-free 的 PSNR（非 EMA 权重）：
   T8 +1.17 dB、T16 +0.38 dB、T24 −0.04 dB、T32 −0.01 dB。EMA 权重下方向相同（+1.25 / +0.36 / −0.06 / −0.03）。
2. **T=8 处的差距真实存在，但比 v1 小。** v1 是 +1.57 dB，当时专训和 T-free 的数据量不对等；v2 是 +1.17 dB，专训模型在 6 个场景上全部胜出。
3. **专训模型一离开自己的 T 就明显变差，T-free 在 7 个 T 上都稳定。** 例如 T8 专训在 T=32 上只有 16.37 dB（比 T-free 低 6.8 dB）。
   在 T=12/20/28 这三个没有任何模型训练过的 T 上，T-free 只有一处被超过：T=12 上 T16 专训 26.92 dB，T-free 26.59 dB。
4. **v2 T-free 复现了已发布的 T-free。** 两者配方相同，7 个 T 上的 PSNR 差别都在 ±0.03 dB 以内。

## 2. 结果

PSNR (dB) / SSIM，非 EMA 权重（`checkpoint.pth`，训练结束时的权重）：

| 权重 | T=8 | T=12 | T=16 | T=20 | T=24 | T=28 | T=32 |
|---|---|---|---|---|---|---|---|
| T-free | 28.13 / 0.9735 | 26.59 / 0.9623 | 25.09 / 0.9521 | 25.27 / 0.9505 | 23.50 / 0.9353 | 23.62 / 0.9371 | 23.13 / 0.9299 |
| T8 | **29.30** / 0.9792 | 25.17 / 0.9609 | 21.41 / 0.9157 | 19.15 / 0.8573 | 17.50 / 0.8002 | 17.06 / 0.7830 | 16.37 / 0.7479 |
| T16 | 27.55 / 0.9726 | 26.92 / 0.9664 | **25.46** / 0.9570 | 24.99 / 0.9519 | 22.31 / 0.9294 | 21.30 / 0.9147 | 19.97 / 0.8840 |
| T24 | 25.27 / 0.9588 | 25.66 / 0.9572 | 24.80 / 0.9504 | 25.18 / 0.9503 | **23.46** / 0.9352 | 23.45 / 0.9358 | 22.36 / 0.9247 |
| T32 | 24.26 / 0.9508 | 24.77 / 0.9508 | 24.20 / 0.9452 | 24.73 / 0.9465 | 23.25 / 0.9333 | 23.50 / 0.9363 | **23.12** / 0.9299 |
| 已发布 T-free（参考） | 28.10 / 0.9734 | 26.58 / 0.9623 | 25.11 / 0.9522 | 25.30 / 0.9507 | 23.52 / 0.9357 | 23.65 / 0.9374 | 23.13 / 0.9298 |

专训模型在自己的 T 上减去 T-free：

| | T8 | T16 | T24 | T32 |
|---|---|---|---|---|
| v2 非 EMA，PSNR dB / SSIM | +1.17 / +0.0057 | +0.38 / +0.0050 | −0.04 / −0.0002 | −0.01 / +0.0000 |
| v2 EMA，PSNR dB / SSIM | +1.25 / +0.0061 | +0.36 / +0.0046 | −0.06 / −0.0004 | −0.03 / −0.0002 |
| 专训胜出的场景数（非 EMA / EMA） | 6/6 / 6/6 | 5/6 / 6/6 | 2/6 / 2/6 | 2/6 / 1/6 |
| v1 非 EMA（参考：1.5k 步，数据不对等） | +1.57 | +0.46 | −0.13 | +0.15 |

参考点：ADMM-TV 在 T=8 上为 25.87 dB，在 T=32 上为 20.29 dB；旧版 T=8 专用模型（P4，第 145 个 epoch）在 T=8 上为 30.13 dB。
完整的 EMA 表见 `report/figures/perT_v2_EMA_table.md`，各版本汇总见 `report/figures/compare_all_table.md`。

图：
- `report/figures/perT_v2.png`：非 EMA 权重，PSNR/SSIM 随 T 的曲线，以及差距柱状图
- `report/figures/perT_v2_EMA.png`：同上，EMA 权重
- `report/figures/compare_all.png`：ADMM-TV、旧版模型、v1、v2 放在一起比较

## 3. 实验设置

- 5 个模型都从旧版 T=8 权重热启动（`checkpoints/best_model.pth`，sha256 `1d0602a6…2b21`）：
  head 层按 T 通道取平均变为 2 个输入通道，tail 层取平均变为 1 个输出通道。
- 每个模型训练 6000 步：`--epochs 30 --steps_per_epoch 200 --warmup_epochs 3`。每步 B×T≈32 帧（T8 时 B=4，T16 时 B=2，T24/T32 时 B=1）。
  T-free 每一步从 {8,16,24,32} 中随机抽一个 T，所以每个 T 平均约 1500 步。
- 数据：DAVIS 2017 trainval 480p（90 段序列）。每个 T 200 段片段（140 段真实视频、60 段合成）；T-free 用 `n_train=800`，即每个 T 200 段。
  序列的抽取顺序由 seed+T 决定；起始帧和裁剪位置来自全局随机数，所以只有 T=8 时，专训和 T-free 用的片段完全一样。
- 每一步重新生成随机二值编码掩模，测量噪声 σ=0.01，bf16 混合精度，开启梯度检查点，VGG 损失权重 0.02，EMA 衰减 0.999。
- 评估：`sci_ablation.run_ablation --exp T --Ts 8 12 16 20 24 28 32`，6 个场景，每个 CSV 85 行，`err` 列全部为空。

命令（由 `perT_sweep/run_v2.sh` 生成，见各模型目录下的 `command.txt`）：

```
python -u dd_madun_v3_train_tfree.py --T_list <8 16 24 32 | T> --n_train <800 | 200> \
  --epochs 30 --steps_per_epoch 200 --warmup_epochs 3 \
  --warm_start checkpoints/best_model.pth --legacy_T 8 --davis_root <DAVIS> \
  --real_ratio 0.7 --seed 42 --amp --ckpt --ckpt_blocks --vgg_weight 0.02 --log_every 20 \
  --out_dir perT_sweep_v2/<model>
python -m sci_ablation.run_ablation --exp T --Ts 8 12 16 20 24 28 32 --methods tfree \
  --ckpt tfree=<weights> --device cuda --out perT_sweep_v2/eval/<model>_<nonEMA|EMA>.csv
```

运行环境：vast.ai RTX 5090，NGC PyTorch 2.14.0a0（CUDA 13）。
- 4 卡机：T-free、T8、T16、T24，2026-11-16 15:14 UTC 开始，代码版本 commit `443e0e6`。
- 单卡机：T32，2026-11-16 16:54 UTC 开始。

训练和评估代码（`dd_madun_v3_train_tfree.py`、`dd_madun_v3_tfree.py`、`video_dataset.py`、`warm_start.py`、`sci_ablation/`）
从 `443e0e6` 到 `a63d64e` 之间没有任何改动。速度约 3.7–4.9 秒/步，每个模型约 7–8 小时。

开始时的检查（来自 `train.log`）：

```
[MixedDataset] DAVIS sequences: 90 (89 with >= 32 frames)
[MixedDataset] Train: 140 real + 60 synth = 200
T=32: train=(200, 32, 256, 256)  (B=1)
```

权重 sha256（`checkpoint.pth` = 非 EMA，`best_model.pth` = EMA）：

| 模型 | checkpoint.pth | best_model.pth |
|---|---|---|
| tfree | 9e2f40bb158af264… | 1128bcbc6419b260… |
| T8 | 5bb806f4c822c54f… | ca69e9289ee06dd6… |
| T16 | 2987a2698e2cd84d… | 0673196467d46632… |
| T24 | 65201579fe8f5a3d… | 2b6174e2e98feab6… |
| T32 | 63959d5603f817b3… | 12c6be9128f8db93… |

权重保存在两台服务器的 `/root/Optics_git/perT_sweep_v2/<model>/` 下，A 阶段从这里接着训。

## 4. 局限和待解答的问题

- **训练量远少于旧版。** 旧版 P4 模型按代码默认值约训了 7.25 万步（2000 段片段 × 145 个 epoch）；v2 只训了 6000 步，每个 T 200 段片段。
  旧版在 T=8 上比 v2 T8 专训高 0.83 dB。
- **旧版和 v2 的掩模设置不同。** 旧版整个训练只用一个固定的随机掩模；v2 每一步都换新掩模。
  所以旧版和 v2 之间的差距，不能只归因于训练量。
- **训练时的验证集是 20 段合成场景**（`make_complex_scene`），不含真实视频，不能用来判断模型在真实视频上是否过拟合。
- 下一步是 A/B 阶段（`perT_sweep/run_scale.sh`）：5 个模型都从各自的 v2 权重出发，训练数据换成 DAVIS 全部 150 段序列加 YouTube-VOS 全帧数据，每个 T 2000 段片段，
  再分别加训 6k 步和 12k 步，检验 T=8/16 处的差距会不会随数据量和训练量缩小。
