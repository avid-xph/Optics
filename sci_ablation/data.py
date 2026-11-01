"""Benchmark 场景加载与变 T 分组.

mat["orig"] 形状为 (H, W, T_total), 值域 [0, 255]。归一化到 [0,1] 并转为
(T_total, H, W)。变 T 评测时 T 不整除 T_total 的两种策略:
  stride="disjoint": 只取不重叠完整窗口（剩余尾帧丢弃）
  stride="overlap": 步长 = T//2 的重叠滑窗（尾帧不足则再补一个末尾窗口）
"""
import os
import numpy as np
import scipy.io
import torch

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark")
FILES = ["aerial", "crash", "drop", "kobe", "runner", "traffic"]


def load_scene(name, bench_dir=None):
    """返回 (T_total, H, W) float tensor, 值域 [0,1]."""
    d = bench_dir or BENCH_DIR
    mat = scipy.io.loadmat(os.path.join(d, f"{name}.mat"))
    orig = mat["orig"].astype(np.float32) / 255.0   # (H, W, T_total)
    return torch.from_numpy(orig).permute(2, 0, 1).contiguous()


def group_starts(T_total, T, mode="disjoint"):
    """返回窗口起点列表。

    disjoint: [0, T, 2T, ...]，最多 floor(T_total/T) 个
    overlap : 步长 max(T//2,1)，尾部不足 T 时追加末尾窗口保证尾帧被覆盖
    """
    if mode == "disjoint":
        return list(range(0, T_total - T + 1, T))
    starts = list(range(0, T_total - T + 1, max(T // 2, 1)))
    if starts and starts[-1] + T < T_total:
        starts.append(T_total - T)
    return starts


def iter_groups(scene, T, mode="disjoint"):
    """yield (start, gt) 其中 gt: (1, T, H, W)."""
    for s in group_starts(scene.shape[0], T, mode):
        yield s, scene[s:s + T].unsqueeze(0)
