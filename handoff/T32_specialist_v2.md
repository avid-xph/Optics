# 交接任务：训练 v2 的 T=32 专训模型并评估

本文件交给导师那边的 GPU 机器（由人或 AI agent 执行），**完全自包含**，照着做即可。
有任何不确定的地方，先问发来本文件的人（He Xingpeng），不要自行改参数。

## 1. 背景（一段话）

我们在比较"一个 T-free 模型（能处理任意压缩比 T）"和"每个 T 单独训练的专训模型"。
实验 v2 一共训练 5 个模型：T-free、T8、T16、T24、T32。它们的**训练步数相同（6000 步），
每个 T 的训练片段数相同、取自同一批视频序列**。前 4 个已经在另一台机器上跑着，**T32 需要在这里跑**。
为了保证对比公平，下面的命令和参数**一个字都不能改**。

## 2. 硬件和环境要求

- NVIDIA GPU，显存 ≥ 24 GB（T24 在 RTX 5090 上占用约 17 GB，T32 预计约 20 GB）。
- PyTorch ≥ 2.0，GPU 可用。RTX 50 系列需要 CUDA ≥ 12.8、PyTorch ≥ 2.7。
- 另外需要：`scipy numpy matplotlib Pillow torchvision`。**已经装好的 torch 不要重装**。
- 硬盘空间 ≥ 5 GB；需要联网（第一次训练会自动下载 VGG16 预训练权重，约 528 MB）。
- 时长：RTX 5090 上预计约 8 小时；更慢的卡按比例增加（T24 实测约 3.8 秒/步，T32 会再慢一些）。

## 3. 步骤

### 3.1 获取代码（公开仓库，不需要账号）

```bash
git clone -b exp/perT-sweep-v2 https://github.com/avid-xph/Optics.git Optics_git
cd Optics_git
git log --oneline -1          # 记下 commit 号，最后要回报
```

核对初始化权重（必须完全一致）：

```bash
sha256sum checkpoints/best_model.pth
# 应为 1d0602a60e26769483dc2803e7ef83d1b35cf49d7ab1218ddb0823cb26af2b21
```

Windows PowerShell 用 `Get-FileHash checkpoints\best_model.pth`。

### 3.2 下载数据集 DAVIS 2017（trainval，480p，约 800 MB）

```bash
mkdir -p dataset && cd dataset
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip -q DAVIS-2017-trainval-480p.zip && rm DAVIS-2017-trainval-480p.zip
ls DAVIS/JPEGImages/480p | wc -l     # 必须是 90
cd ..
```

### 3.3 训练（用 tmux 或 nohup 启动，SSH 断开也不能中断）

```bash
mkdir -p perT_sweep_v2/T32
python -u dd_madun_v3_train_tfree.py \
  --T_list 32 --n_train 200 \
  --epochs 30 --steps_per_epoch 200 --warmup_epochs 3 \
  --warm_start checkpoints/best_model.pth --legacy_T 8 \
  --davis_root dataset/DAVIS --real_ratio 0.7 --seed 42 \
  --amp --ckpt --ckpt_blocks --vgg_weight 0.02 --log_every 20 \
  --out_dir perT_sweep_v2/T32 2>&1 | tee -a perT_sweep_v2/T32/train.log
```

Linux 上也可以直接用仓库里的脚本，效果完全相同：
`MODELS="T32" GPUS="0" bash perT_sweep/run_v2.sh train`。

**开跑后几分钟内检查 `train.log`，必须看到下面三行**，否则立刻停止并回报：

```
[MixedDataset] DAVIS sequences: 90 (89 with >= 32 frames)   ← 第一个数是 90
[MixedDataset] Train: 140 real + 60 synth = 200
T=32: train=(200, 32, 256, 256)  (B=1)
```

**中途断了怎么办**：每个 epoch 都会存 `checkpoint.pth`。在原命令后面加上
`--resume perT_sweep_v2/T32/checkpoint.pth` 重新运行，就会接着训。

**怎么算跑完**：出现 `perT_sweep_v2/T32/history.json`，并且其中包含 `"per_T"` 字段。

### 3.4 评估（6 个真实场景，7 个 T，几分钟）

```bash
mkdir -p perT_sweep_v2/eval
python -m sci_ablation.run_ablation --exp T --Ts 8 12 16 20 24 28 32 --methods tfree \
  --ckpt tfree=perT_sweep_v2/T32/checkpoint.pth --device cuda --out perT_sweep_v2/eval/T32_nonEMA.csv
python -m sci_ablation.run_ablation --exp T --Ts 8 12 16 20 24 28 32 --methods tfree \
  --ckpt tfree=perT_sweep_v2/T32/best_model.pth --device cuda --out perT_sweep_v2/eval/T32_EMA.csv
```

每个 CSV 应该有 85 行数据，`err` 列全部为空。

## 4. 需要交回的东西

1. **小文件（必须交回）**：打包成一个 zip 发给 He Xingpeng。
   - `perT_sweep_v2/eval/T32_nonEMA.csv`、`perT_sweep_v2/eval/T32_EMA.csv`
   - `perT_sweep_v2/T32/history.json`、`perT_sweep_v2/T32/train.log`
2. **权重（必须保留，最好也交回）**：`perT_sweep_v2/T32/checkpoint.pth` 和 `best_model.pth`，共约 150 MB。
   下一阶段要在这两个权重的基础上加数据继续训练，**不要删除**。可以通过网盘发送，或者先留在这台机器上。
3. **信息**：GPU 型号、PyTorch 版本、开始和结束时间、第 3.1 步记下的 commit 号，以及两个权重文件的 sha256。

## 5. 规则

- 参数、数据、代码都不能改。如果显存不够（OOM），停下来回报，**不要自行减小 batch 或改参数**。
- 只报告实际测到的数字，不要估算或补数据。
- 不要往 GitHub 推送任何东西（没有写权限也不需要）。
- 更多项目背景和工作准则见仓库根目录的 `CLAUDE.md`。
