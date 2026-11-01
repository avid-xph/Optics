"""End-to-end 重建网络: 测量图 (B,H,Wm) 直接映射数据立方体 (B,T,H,W).

设计要点（对照 PPT 任务③, E2E「对 T 敏感」的结构性根源）:
  - 输入: 对每帧 t 取测量切片 E[:, :, t*shear : t*shear+W] 得到 (B,T,H,W),
    再与 mask 先验 adjoint(E) 拼成 2T 通道 —— 通道数与 T 绑定, 换 T 必须重训。
  - 输出: T 帧立方体。
  - 结构: 紧凑残差 U-Net, 参数量适中, 便于 24GB GPU 上 256 分辨率训练。

训练入口见 train_e2e.py（本地 CPU 冒烟 / 远程 GPU 正式训练共用）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.c1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.c2 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x)))


class E2ECubeNet(nn.Module):
    """残差 U-Net: (B, 2T, H, W) → (B, T, H, W)."""

    def __init__(self, T=8, nf=48, nb=2):
        super().__init__()
        self.T = T
        self.head = nn.Conv2d(2 * T, nf, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(nf) for _ in range(nb)])
        self.down = nn.Conv2d(nf, nf * 2, 2, 2, 0)
        self.mid = ResBlock(nf * 2)
        self.up = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.tail = nn.Conv2d(nf, T, 3, 1, 1)

    def forward(self, x):
        """x: (B, 2T, H, W) 已拼接好的输入张量."""
        if x.shape[1] != 2 * self.T:
            raise ValueError(f"E2ECubeNet trained for T={self.T}, got {x.shape[1]} channels")
        h = F.relu(self.head(x))
        h = self.body(h)
        lo = F.relu(self.down(h))
        lo = self.mid(lo)
        lo = F.interpolate(lo, size=h.shape[2:], mode="bilinear", align_corners=False)
        h = h + self.up(lo)
        out = self.tail(h)
        return out + x[:, self.T:]      # 残差: 网络输出叠加 mask 先验


def build_e2e_input(cup, E):
    """E: (B,H,Wm) 测量 → (B, 2T, H, W) 网络输入.

    前一半: 每帧 t 的测量切片（无 mask, 含全部编码信息）
    后一半: adjoint(E) 的 mask 先验（即去噪起点）
    """
    B, H, Wm = E.shape
    T, shear, W = cup.T, cup.shear, cup.W
    slices = []
    for t in range(T):
        s = t * shear
        if s + W <= Wm:
            slices.append(E[:, :, s:s + W].unsqueeze(1))   # (B,1,H,W)
        else:                            # shear 大于画布余量时的防御分支
            pad = s + W - Wm
            slices.append(F.pad(E[:, :, s:], (0, pad)).unsqueeze(1))
    m_prior = cup.adjoint(E)             # (B,T,H,W)
    return torch.cat(slices + [m_prior], dim=1)
