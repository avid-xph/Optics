# Optics — T-free CUP 视频重建

本文件给在这个仓库里工作的人和 AI agent（Claude Code 等）看：先读完再动手。

## 项目是什么

- **任务**：压缩超快成像（CUP）视频重建。一张二维快照编码 T 帧视频（T = 压缩比）。
- **模型**：T-free DD-MADUN（[dd_madun_v3_tfree.py](dd_madun_v3_tfree.py)）。同一套权重可以重建任意 T。
  它基于组里的 DD-MADUN v3（P4 版本，双域深度展开网络，只能处理 T=8，权重 `checkpoints/best_model.pth`）改造而来。
  关键改动：去噪器逐帧处理、权重在帧之间共享（输入通道从 T+1 变为 2）；数据一致性的步长按 8/T 缩放；
  DCT 矩阵按当前 T 现场生成。
- **核心问题**：A. 一个模型能否覆盖所有 T（已验证，7 个 T 都优于 ADMM-TV，包括没训过的 12/20/28）；
  B. 和每个 T 单独训的专训模型相比，通用要付出多少精度（v2 实验进行中）。
- **合作方式**：代码在 GitHub `avid-xph/Optics`；GPU 实验在租用的服务器上跑。

## 目录约定（必须遵守）

| 目录 | 放什么 |
|---|---|
| `dataset/` | 所有训练数据集，例如 `dataset/DAVIS/JPEGImages/480p/<序列>/*.jpg`。**不进 git** |
| `data_prep/` | 下载、整理数据集的脚本（例如 `prepare_youtubevos.sh`）；输出写到 `dataset/` |
| `benchmark/` | 6 个评估场景（`*.mat`），进 git，不要改 |
| `checkpoints/` | 已发布的权重：legacy T=8（`best_model.pth`）、published T-free（`tfree_trained/`） |
| `perT_sweep_v2/` | v2 实验的输出：每个模型一个子目录（权重、日志），评估 CSV 在 `eval/` |
| `experiment_reports/` | **所有实验报告**（md），命名格式为 `YYYYMMDD_主题.md` |
| `handoff/` | 交给其他机器或其他人执行的任务说明（自包含，照做即可） |
| `results/` | 实验的原始数据（CSV），报告里引用这里的路径 |
| `report/` | 画图和分析脚本；生成的图放在 `report/figures/` |
| `pic/` | 汇报 PPT 用的图，文件名以页码开头（`p04_...png`） |
| `sci_ablation/` | 统一的评估入口（`python -m sci_ablation.run_ablation`） |

## 工作准则

1. **按功能建文件夹，文件及时归位。** 新类型的文件先看上表；没有合适的目录就新建一个以功能命名的目录
   （例如 `dataset/`、`benchmark/`、`experiment_reports/`），不要堆在仓库根目录。
   实验结束后，报告立刻写进 `experiment_reports/`，原始 CSV 放进 `results/`，图放进 `report/figures/` 或 `pic/`。
2. **数字必须是测出来的。** 报告里每个数字都要能追溯到一个 CSV 或日志；没测过就写"未测"，绝不估算后当作结果。
   报告要记录：完整的命令、数据集、权重的 sha256、git commit、聚合方式（默认对 6 个场景的所有窗口取平均）。
3. **公平对比。** 对比的各方使用相同的步数、相同的数据（每个 T 相同的片段）、相同的评估协议，
   并且同时报告非 EMA 和 EMA 两种权重。
4. **Git。** 在 feature 分支上工作（`exp/...`、`feat/...`）；push、开 PR、merge 之前先问用户。
   权重和小数据可以进 git；日志、密钥、大数据集不进。提交邮箱用 dorima06@yahoo.com。
5. **直接写代码。** 不加臆测性的防御代码，也不搞冒烟测试、dry-run 之类的走过场流程。
6. **长任务用 `nohup setsid ... < /dev/null &` 或 tmux 启动**，SSH 断开不能影响训练；中断后重跑同一命令会接着训。
7. **拿不准就问。** 不确定的地方，在 Slack #working 或直接问用户。

## 常用命令

```bash
# v2：T-free + 4 个专训模型，每个 6000 步，每个 T 都是 200 段视频（来自同一批视频序列）
MODELS="tfree T8 T16 T24" GPUS="0 1 2 3" bash perT_sweep/run_v2.sh all
MODELS="T32" GPUS="0" bash perT_sweep/run_v2.sh train      # 拆到另一台机器上跑
bash perT_sweep/run_v2.sh eval && bash perT_sweep/run_v2.sh plot

# 画汇报用的图
python report/plot_perT.py --eval_dir perT_sweep_v2/eval --out report/figures/perT_v2
python report/make_mentor_figs.py [--recon]
```
