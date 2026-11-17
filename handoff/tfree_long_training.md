# 交接任务：T-free 模型长训练（7.5 万步）与阶段评估

本文件交给导师那边的单卡 GPU 机器执行，执行者可以是人，也可以是 AI agent。内容**完全自包含**，照着做即可。
遇到任何不确定的地方，先问发来本文件的人（He Xingpeng），不要自行改参数。

## 1. 背景

T-free DD-MADUN 是一个模型：同一套权重可以重建任意压缩比 T 的 CUP 视频。
它已经训练了 1.2 万步：v2 阶段 6k 步，A 阶段在扩大的数据集上再训 6k 步。
在 6 个评估场景上：
- T=24、32 时，它和逐个 T 单独训练的专用模型持平；
- T=8、16 时，它分别低 1.2 dB 和 0.4 dB。

参照原版 T=8 模型的训练量（约 7.25 万步），这次从 A 阶段的权重出发，**再训练 7.5 万步**，看 T-free 的上限在哪里。
训练过程中每 1.5 万步存一份快照，训练结束后统一评估，得到"效果随训练量变化"的曲线。

## 2. 硬件和环境要求

| 项目 | 要求 | 依据 |
|---|---|---|
| GPU 显存 | **≥ 24 GB** | RTX 5090 上实测峰值 19.2 GB |
| 内存（RAM） | **≥ 64 GB**，最好 ≥ 96 GB | 训练数据启动时全部读进内存：8000 段片段，float32，约 42 GB，构建期间还会更高（这是按数据形状算的，不是实测） |
| 硬盘 | **≥ 45 GB 可用空间** | 数据集最终约 17.5 GB，准备 YouTube-VOS 时临时文件约 20 GB，权重和快照约 1 GB |
| 系统 | Linux（推荐 Ubuntu）。Windows 请在 WSL2 的 Ubuntu 里执行全部步骤 | 数据准备脚本是 bash，会用到 apt-get、aria2c、7z、unzip |
| 软件 | PyTorch ≥ 2.0 且 GPU 可用（RTX 50 系列需要 CUDA ≥ 12.8、PyTorch ≥ 2.7），另需 `scipy numpy matplotlib Pillow torchvision` | **已经装好的 torch 不要重装** |
| 网络 | 能访问 GitHub、ethz.ch、huggingface.co | 下载约 15 GB；第一次训练还会自动下载 VGG16 权重，约 528 MB |
| 时长 | RTX 5090 上约 4.4 秒/步，7.5 万步约 **92 小时（约 4 天）** | 其他 GPU 按开跑后的实际速度换算（见 3.4） |

## 3. 步骤

### 3.1 获取代码和起始权重（公开仓库，不需要账号）

```bash
git clone -b exp/perT-sweep-v2 https://github.com/avid-xph/Optics.git Optics_git
cd Optics_git
git log --oneline -1          # 记下 commit 号，最后要回报
sha256sum checkpoints/tfree_scaleA/checkpoint.pth
# 必须是 31dad5da39dc27b299ff59e5c7f72b7f3cbf9bf0176935a76cec8c92a21fd1f2
grep -n snapshot_every dd_madun_v3_train_tfree.py   # 必须有输出（快照功能）
```

### 3.2 准备数据集（约 1–2 小时，大部分时间在下载）

三条命令依次执行。每个脚本都可以重复运行：已经完成的部分会自动跳过，中断后重跑就会接着做。

```bash
bash data_prep/prepare_davis.sh                          # DAVIS 2017 三个部分，约 1.4 GB
SPLITS="valid test" bash data_prep/prepare_youtubevos.sh # YouTube-VOS 2019 全帧，下载后缩放到 480p
bash data_prep/build_merged.sh                           # 用符号链接合并成 dataset/merged
```

`build_merged.sh` 最后几行**必须**是下面这样，否则停下来回报：

```
[linked] davis: 90 sequences ...
[linked] davisdev: 30 sequences ...
[linked] davischal: 30 sequences ...
[linked] yt: 1048 sequences ...
[total] 1198 sequences in dataset/merged
```

`prepare_youtubevos.sh` 会核对每个下载分卷的 sha256；如果打印 `[FAIL]`，停下来回报，不要绕过。

### 3.3 启动训练（用 tmux 或 nohup，SSH 断开也不能中断）

```bash
mkdir -p perT_sweep_long/tfree
nohup setsid python -u dd_madun_v3_train_tfree.py \
  --T_list 8 16 24 32 --n_train 8000 \
  --warm_start checkpoints/tfree_scaleA/checkpoint.pth --legacy_T 8 \
  --epochs 150 --steps_per_epoch 500 --warmup_epochs 10 --snapshot_every 30 \
  --davis_root dataset/merged --real_ratio 0.7 --seed 42 \
  --amp --ckpt --ckpt_blocks --vgg_weight 0.02 --log_every 50 \
  --out_dir perT_sweep_long/tfree \
  >> perT_sweep_long/tfree/train.log 2>&1 < /dev/null &
```

各参数的含义：
- 150 个 epoch × 每个 epoch 500 步 = 7.5 万步，前 10 个 epoch（5000 步）是学习率预热，之后余弦下降。
- 每一步从 {8, 16, 24, 32} 里随机抽一个 T，每步处理约 32 帧。
- 每个 T 有 2000 段训练片段（70% 真实视频，30% 合成）。
- 每 30 个 epoch（1.5 万步）存一份快照。

**开跑后检查 `train.log`，必须看到下面这些行**，否则停止并回报：

```
[warm start] checkpoints/tfree_scaleA/checkpoint.pth
  adapted head x0, tail x0; missing=0 unexpected=0
[MixedDataset] DAVIS sequences: 1198 (1192 with >= 32 frames)     ← T=32 那一组
[MixedDataset] Train: 1400 real + 600 synth = 2000                ← 共 4 次，每个 T 一次
T=32: train=(2000, 32, 256, 256)  (B=1)
```

构建数据集的时间因机器而异，在我们的两台机器上分别用了 8 分钟和约 90 分钟。之后会开始出现 `[step 50] ...` 这样的行。

### 3.4 估算完成时间

出现 `[step 200]` 那一行以后，用这一行的 `elapsed` 除以 200 得到每步秒数，再算：
剩余时间 ≈ (75000 − 当前步数) × 每步秒数。把这个估计回报给 He Xingpeng。

### 3.5 中途断了怎么办

每个 epoch 结束都会存一次 `perT_sweep_long/tfree/checkpoint.pth`。用**完全相同的命令**，末尾加上
`--resume perT_sweep_long/tfree/checkpoint.pth` 重新运行，训练会从断点接着跑：优化器、学习率、EMA、训练曲线都会恢复。
已经存好的快照不受影响。

### 3.6 什么时候算跑完

出现 `perT_sweep_long/tfree/history.json`，而且其中包含 `"per_T"` 字段。
`perT_sweep_long/tfree/snapshots/` 下应该有 10 个文件：`e030`、`e060`、`e090`、`e120`、`e150` 五个时间点，每个时间点有 `_nonEMA.pth` 和 `_EMA.pth` 两份。

### 3.7 评估（6 个评估场景 × 7 个 T，每份权重约 2 分钟）

**训练结束后**再做评估，不要和训练同时跑，否则会抢占显存：

```bash
mkdir -p perT_sweep_long/eval
for f in perT_sweep_long/tfree/snapshots/*.pth perT_sweep_long/tfree/best_model.pth; do
  n=$(basename "$f" .pth)
  python -m sci_ablation.run_ablation --exp T --Ts 8 12 16 20 24 28 32 --methods tfree \
    --ckpt "tfree=$f" --device cuda --out "perT_sweep_long/eval/tfree_$n.csv"
done
```

每个 CSV 应该有 85 行数据，`err` 列全部为空。

## 4. 需要交回的东西

1. **小文件（必须交回）**，打包成一个 zip 发给 He Xingpeng：
   - `perT_sweep_long/eval/*.csv`（共 11 个）
   - `perT_sweep_long/tfree/history.json`、`perT_sweep_long/tfree/train.log`、`perT_sweep_long/tfree/command.txt`（如果有的话）
2. **权重（必须保留，最好也交回）**：`perT_sweep_long/tfree/snapshots/`（10 个文件，每个约 37 MB）、`best_model.pth`、`checkpoint.pth`。
   可以用网盘发送，或者先留在这台机器上，**不要删除**。
3. **信息**：
   - GPU 型号、PyTorch 版本、内存大小；
   - 开始时间、结束时间、每步秒数；
   - 3.1 记下的 commit 号；
   - `snapshots/e150_nonEMA.pth` 和 `best_model.pth` 的 sha256。

## 5. 规则

- **参数、数据、代码都不能改。**
  - 如果显存不够（OOM）或内存不够（进程被系统杀掉，日志里出现 `Killed`），停下来回报。不要自行减小 batch、片段数或改其他参数。
  - 数据集的序列数必须是 1198。少了说明下载或缩放有问题，回报，不要用不完整的数据开训。
- 只报告实际测到的数字，不要估算或补数据。
- 不要往 GitHub 推送任何东西（没有写权限，也不需要）。
- 更多项目背景和工作准则见仓库根目录的 `CLAUDE.md`。
