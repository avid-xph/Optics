# DD-MADUN v3 (P4) — CUP 重建复现包

本目录是 **DD-MADUN v3（P4 改进版）** 的独立复现包，用于复现论文/项目中的
压缩超快摄影（CUP）重建模型与 6 个 benchmark 场景的重建结果。

包含内容：
- 训练好的模型权重 `checkpoints/best_model.pth`（epoch145，约 9.4M 参数）
- 推理/评估脚本，可复现 6 个 benchmark 场景的重建结果
- 6 个 benchmark 场景数据 `benchmark/*.mat`
- 已生成的重建结果对照 `results/`（图 + 数值日志）

## 模型
- 名称：DD_MADUNv3（Dual-Domain MADUN v3）
- 结构：T=8, H=W=256, K=8, K1=3, ng=6（梯度步）, nf=96, nb=8
- 参数量：9,387,010
- 训练设置：real_cup 前向，150 epoch，DAVIS 真实视频 70% + 合成数据，
  FFL 频域损失 + VGG 感知损失 + EMA
- 泛化本质：网络学习“对给定编码算子迭代反演”的算法（DC 步用传入算子保
  测量一致性 + Denoiser 提供 mask 无关图像先验）；测试时换 mask 只换算子、
  权重不动，故对任意同分布随机 mask 都能高质量重建

## 目录结构
```
cup_simulator.py           CUP 前向 / 伴随算子（推理必需）
baselines.py               PSNR / SSIM 等指标（推理必需）
dd_madun_v3.py             模型定义（推理 + 训练必需）
benchmark_reconstruct.py   6 场景重建评估（推理入口）
dataset.py                 合成训练数据生成（重训必需）
video_dataset.py          DAVIS 真实视频数据加载（重训必需）
dd_madun_v3_train_p4.py    训练脚本（重训模型用）
checkpoints/best_model.pth 模型权重
benchmark/*.mat            6 个基准场景（aerial/crash/drop/kobe/runner/traffic）
results/                    已生成的重建图 + 数值日志
requirements.txt           依赖
```

## 环境
```bash
pip install -r requirements.txt
```
需要 PyTorch（CUDA 版用于 GPU 加速；仅 CPU 也能跑，只是较慢）。
推理依赖：torch / numpy / scipy / matplotlib；训练额外需要 torchvision / Pillow。

## 复现重建结果（推理）
```bash
python benchmark_reconstruct.py
```
- 默认使用 `checkpoints/best_model.pth`，对 `benchmark/` 下 6 个场景做 real_cup 重建
- 终端打印每场景 PSNR/SSIM，重建图存 `results/bench_<scene>_realcup.png`，
  汇总数值日志存 `results/benchmark_P4_log.txt`
- 预期（MEAN）：**PSNR = 29.99 dB，SSIM = 0.9832**

| 场景 | PSNR (dB) | SSIM |
|---|---|---|
| aerial  | 26.38 | 0.9706 |
| crash   | 26.40 | 0.9858 |
| drop    | 39.51 | 0.9992 |
| kobe    | 29.77 | 0.9845 |
| runner  | 33.55 | 0.9916 |
| traffic | 24.35 | 0.9676 |
| **MEAN**| **29.99** | **0.9832** |

可用 `--ckpt <path>` 指定其他权重。

## 重训模型（可选）
```bash
python dd_madun_v3_train_p4.py --davis_root <DAVIS2017_480p 路径> --out_dir checkpoints
```
- 训练数据：DAVIS 2017 480p（自行下载）放到 `--davis_root`；合成数据由脚本生成
- 关键超参见 `dd_madun_v3_train_p4.py` 顶部 argparse（默认 150 epoch,
  batch_size=2, lr=1e-3, warmup + cosine + EMA）
- 显存提示：256 分辨率 + nf=96/nb=8 约需 23–31 GB 显存

## 备注
- 本包源码原位于 `CUP_COSUP_unfolding_NN/MADUN/`，此处为独立抽取版，
  已将路径理顺为相对结构（模型/数据/结果分别在 checkpoints、benchmark、results）。
- 重建前向为 real_cup（shear=1, noise=0.01），与训练一致。
