# Optics / #working 状态汇报（2026-11-06 05:56 运行）

> 本文件同时是**待发给 Mark 的 Slack 消息草稿**。Slack 在本轮仍不可用，见第 0 节。

## 0. ⚠️ 本轮仍无法发 Slack
- 第 5 次确认：`im_list_conversations` 与 `im_get_messages(C0CEU6CCMNC)` 均返回
  `IM Channels are not available in headless mode`；系统内无已安装 connector。
- 后果：**读不到 #working 的新消息，也无法回复 Mark**。注意是「读不到」，不是「没有新消息」。
- 本轮也未启动任何新训练（遵守「长训练需 Mark 批准」）。

## 1. GitHub 状态
- `git fetch origin`（SSH key `id_ed25519_personal_Sda`）成功，但**无新 refs**。
- `main` = `5f36b0b`（2026-11-03 05:51），与上一轮相同；全仓库无任何 2026-11-04 之后的提交。
- 未合入 main 的远端分支只剩 2 个，都是我此前推的：
  `diag/dc-init-ab` (`3d93983`)、`exp/e2e-trained-baselines` (`d5221e3`)。
- ⚠️ **standing context 已过期**：里面写的 `main = 6f40246` 和「T-free 训练发散 BLOCKER」
  都已不成立 —— PR #7/#8/#9 早已合入，DC 步长 ×(8/T) 的修复已生效并验证。
- PR #2 仍需 Mark 在网页端关闭。

## 2. 🔑 重大发现：30-epoch T-free 训练其实已经跑完，结果可用
前几轮一直盯着 `github_optics/output_tfree/`（空），但训练输出实际写在
**`optics_main/output_tfree/`**（同一仓库的另一个 worktree）：

| 项 | 值 |
|---|---|
| 文件 | `best_model.pth` / `checkpoint.pth` / `final_ema.pth` / `history.json`，2026-11-03 13:07 |
| epoch | 30（完整跑完） |
| loss | 0.2725 → 0.1407，单调下降，**T=24/32 全程无发散** → BLOCKER 已解除 |
| best EMA | 34.66 dB / SSIM 0.9953（训练脚本自带测试集） |
| ckpt 元数据 | `T_list=[8,16,24,32]`, `K=8,K1=3,ng=6,nf=96,nb=8`, `H=W=256`, `temporal=none` |
| history `per_T` | T=8/16/24/32 = 34.11 / 31.57 / 28.83 / 26.76 dB |

⚠️ **不确定项（不猜，请 Mark 确认）**
1. 无法从文件还原完整命令行（`n_train` / `seed` / `--amp` 等），也无法 100% 确认这就是
   11-03 05:27 那条 PID 2792 训练 —— 30 epoch + `T_list` 完全吻合，**推定**是它。
2. 速度结论需更正：05:27 → 13:07 约 **7.7 h** 跑完，远快于 standing context 的
   「62 s/step、4.3 天」估算。原因未查明（可能当时 GPU 被两个任务共享，或 `n_train` 不同），
   **不要据此推定新训练也只要 7.7 h**。
3. 这 4 个文件在 `optics_main` 里是 **untracked，尚未进 git**，约 150 MB，有丢失风险。
   是否提交进仓库请 Mark 定；本轮**只提交了评测结果，未提交权重**。

## 3. 本轮短 GPU 评测（无训练，RTX 5090 / `pnpsci` 环境）
用 `sci_ablation.run_ablation` 评测 `optics_main/output_tfree/best_model.pth`（EMA @ epoch 30）：
6 个 benchmark 场景、`cell=1`、`noise=0.01`、`seed=0`、disjoint 分组，全部 0 error。

### 6 场景平均 PSNR（dB，先对每场景窗口取均值、再对 6 场景取均值）

| 方法 | T=8 | T=16 | T=24 | T=32 |
|---|---|---|---|---|
| **T-free（单套权重，本轮）** | **28.05** | **25.51** | **24.22** | **23.07** |
| E2ECubeNet（按 T 单独训练） | 26.32 | 22.92 | 不适用 | 不适用 |
| DD-MADUN（T-bound，仅 T=8） | 30.06 | 不适用 | 不适用 | 不适用 |

- 同一 harness 复现的锚点 E2E T=8 = **26.32**、T=16 = **22.92**，与 standing context
  **逐位一致** → 上表数字可直接对比。
- 结论：单套 T-free 权重在 **T=16 比「专门为 T=16 训练的 E2E」高 +2.59 dB**；
  T=24/32 仍能出有效结果，而 T-bound 模型根本跑不了。T=8 比 T=8 专用 DD-MADUN 低 ~2.0 dB
  （专用模型占优，属预期）。
- 若按窗口数加权平均，T=16 / T=24 分别为 25.08 / 23.49 dB（T=8、T=32 基本一致）。

### 逐场景 PSNR（dB）

| 场景 | T=8 | T=16 | T=24 | T=32 |
|---|---|---|---|---|
| aerial | 24.80 | 23.26 | 22.21 | 21.77 |
| crash | 25.53 | 23.85 | 23.27 | 22.15 |
| drop | 36.88 | 33.12 | 30.78 | 28.79 |
| kobe | 27.98 | 25.07 | 23.91 | 22.62 |
| runner | 31.04 | 27.90 | 26.12 | 24.63 |
| traffic | 22.06 | 19.86 | 19.06 | 18.49 |

### 噪声鲁棒性（T=8，6 场景均值）
`noise 0 → 28.08`、`0.01 → 28.05`、`0.05 → 27.14`、`0.10 → 25.29` dB。
对比 E2E 的 26.34→24.77：T-free 起点更高，但 0→0.1 降幅 2.79 dB 略大于 E2E 的 1.57 dB。

## 4. Caveat（务必带上）
- 评测用 `ExperimentCUP` 的 `seed=0` 随机二值 mask；模型训练用的是
  `create_mixed_dataset` 的 mask，**两者不同**。所有方法（含锚点）共用同一评测 mask，
  横向对比公平；但这本质是**跨 mask 泛化**测试，不是训练分布的 in-domain 分数。
- `history.json` 里的 34.66 dB 是训练脚本自带测试集，**不能**与上面 6-scene 数字混用
  （也不要与 PR #6 发布权重的 26.32/22.92 混用）。
- 30-epoch 训练**不是本轮启动**；本轮只做了前向评测，未跑 CPU 实验、未抢 GPU 长任务。

## 5. 待 Mark 决策 / 待办
1. 确认 `optics_main/output_tfree/` 那条训练是否他授权、完整命令行是什么。
2. 是否把 trained 权重（~150 MB，untracked）提交进 git。
3. 开/合 `diag/dc-init-ab`、`exp/e2e-trained-baselines`；关闭 PR #2。
4. Slack 恢复后把本条转给 Mark（我发不出去）。
5. 是否要再跑 T=10/12/20/28/36/40 等**训练未见 T** 的泛化测试（短、GPU 可跑），
   以及把上表写进论文/PPT。

---
产物：`tfree_bench_20261106.csv`、`tfree_noise_20261106.csv`、
`anchor_T8_20261106.csv`、`anchor_e2e_T16_20261106.csv` 及同名 `.log`。
