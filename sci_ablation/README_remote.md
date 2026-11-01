# 远程 GPU 机部署说明（SCI 消融实验）

本机（D:\w-xianglei）无 NVIDIA GPU，已确认：`nvidia-smi` 无输出。
因此**训练和大矩阵评测必须在带显卡的机器上跑**。本文件说明拷什么、装什么、跑什么。

---

## 1. 需要拷贝的文件清单

### 必需（实验框架本体，6 个文件）
```
sci_ablation/
  __init__.py       实验矩阵说明（无逻辑）
  forward.py        块状编码 mask + 变 T + 噪声注入的前向模型 ExperimentCUP
  data.py           benchmark 场景加载、变 T 窗口分组
  e2e_model.py      End-to-end 网络 E2ECubeNet + 输入构造
  methods.py        三类方法注册表（iterative / ddmadun / e2e）
  run_ablation.py   统一评测入口（→ CSV）
```

### 必需（框架依赖的根目录模块，4 个文件）
```
cup_simulator.py     CUP 前向/伴随算子，框架直接包装它，不改动
baselines.py         TwIST / ADMM-L1 / ADMM-TV / BP + PSNR / SSIM 计算
dd_madun_v3.py       DD_MADUNv3 模型定义与损失
dd_madun_v3_train_p4.py   原版训练脚本（混合 T 训练脚本的改造基线）
```

### 必需（数据 / 权重）
```
benchmark/*.mat            6 个评测场景（256×256, T_total = 32~48）
checkpoints/best_model.pth 现有 DD-MADUN P4 权重（T=8, 29.99 dB）
```

### 训练时需要（额外）
```
dataset.py / video_dataset.py   训练数据生成与读取
video_dataset.py 依赖 DAVIS：  DAVIS/JPEGImages/480p/<seq>/*.jpg
  若服务器上没有 DAVIS，可用 dataset.py 的合成数据（create_complex_dataset）代替
```

### 不需要拷贝
```
results/*.png        旧的基准可视化图
xingpeng.pptx        任务说明原件
benchmark_reconstruct.py   旧的单点复现脚本（已被 run_ablation.py 覆盖，
                           想核对 29.99 dB 基线时可带上）
```

---

## 2. 服务器环境

```bash
python -m venv venv && source venv/bin/activate
# CUDA 版 torch（按服务器 CUDA 版本选，例：cu121）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib Pillow
```

验证：`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

---

## 3. 跑什么

### 3.1 评测（现在就能跑，框架已验证）

```bash
# 冒烟：先确认环境 OK（BP 反投影，秒级）
python -m sci_ablation.run_ablation --exp T --Ts 8 --scenes traffic \
    --methods bp --out results/ablation/smoke.csv

# 实验③ 序列深度 T（迭代法，任意 T 即插即用）
python -m sci_ablation.run_ablation --exp T \
    --Ts 8 12 16 20 24 28 32 \
    --methods twist admm_l1 admm_tv \
    --out results/ablation/exp_T.csv

# 实验② 编码单元大小
python -m sci_ablation.run_ablation --exp cell \
    --cells 1 2 3 4 5 6 8 10 --methods twist admm_l1 \
    --out results/ablation/exp_cell.csv

# 实验④ 编码单元 × T 联合
python -m sci_ablation.run_ablation --exp joint \
    --Ts 8 16 24 32 --cells 1 2 4 8 \
    --methods twist admm_l1 --out results/ablation/exp_joint.csv

# 实验⑤ 噪声
python -m sci_ablation.run_ablation --exp noise \
    --noise 0 0.005 0.01 0.02 0.05 0.1 \
    --methods twist admm_l1 ddmadun --out results/ablation/exp_noise.csv
```

两个常用开关：
- `--group disjoint|overlap`：T 不整除 T_total 时，不重叠分组 / 半重叠滑窗
- `--iters N`：迭代法迭代数（本机默认 200/50，调小可快速出趋势）

DD-MADUN 只能评 T=8。对其他 T 会自动记录失败原因并跳过，**不会中断整个实验**——
这本身就是任务①要暴露的问题。

### 3.2 训练（脚本待补，见第 5 节）

| 训练任务 | 目的 | 状态 |
|---|---|---|
| E2E 网络（每个 T 一份权重） | 补齐 PPT 要求的第三类方法 | 待写 `train_e2e.py` |
| DD-MADUN 混合 T 训练 | 任务①路线 B：一套权重适配 T∈{8..32} | 待写 `train_ddmadun_mixedT.py` |
| DD-MADUN 架构级 T 无关 | 任务①路线 A：改掉 DCT 矩阵与去噪器通道绑定 | 待写 `dd_madun_v3_tfree.py` |

---

## 4. 算力预算（和负责人沟通用）

| 项目 | 显存 | 时长量级 |
|---|---|---|
| DD-MADUN 重训 256×256（README 原值） | 23–31 GB | 150 epoch |
| E2ECubeNet（约 32 万参数，远小于 DD-MADUN 的 938 万） | 预计 < 12 GB | 明显更快 |
| 混合 T 训练 | 与原版同量级 | 与原版同量级 |
| DD-MADUN 推理（256×256, T=8） | 小 | 本机 CPU 约 90 s/窗口，GPU 上秒级 |

最低需求：**4 个训练任务**（E2E 一个 + 混合 T 一个 + T 无关架构一个 + 余量）。

---

## 5. 当前缺口

- 三个训练脚本尚未编写（E2E / 混合 T / T 无关架构改造）。
- E2E 网络结构已完成并通过前向测试，但**未训练**，所以 `--methods e2e` 目前无法产出数字。
- 奈奎斯特理论分析（任务⑥）尚未动笔，属报告侧内容，不需要 GPU。

---

## 6. 结果回传

把服务器上生成的 `results/ablation/*.csv` 拷回本机 `D:\w-xianglei\results\ablation\`，
后续绘图、汇总、写报告全在本机做（不需要 GPU）。
CSV 字段：`exp, method, scene, T, cell, noise, group_start, psnr, ssim, sec, err`
