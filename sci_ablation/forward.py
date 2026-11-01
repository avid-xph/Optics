"""前向模型扩展: 块状编码 mask（编码单元 c×c）+ 变 T + 测量噪声.

不改动原始 cup_simulator.py（保持复现包完整），在此包装:
  - build_block_mask: c×c 像元共享同一随机 0/1 编码值, 开孔率 p=0.5
    （GT 尺寸保持不变: 粗网格 ceil(H/c)×ceil(W/c) 最近邻上采样, 边缘单元可
      能不足 c, 属边界效应, 保证各实验对比同一 GT）
  - ExperimentCUP: 组合 前向算子 + 噪声注入, 统一实验入口
"""
import torch

from cup_simulator import CUPForwardModel


def build_block_mask(H, W, cell=1, p=0.5, generator=None, device="cpu"):
    """生成编码单元为 cell×cell 的随机二值 mask.

    cell=1 即像素级 mask（与原版一致）。cell>1 时粗网格上采样,
    每个编码单元内所有像元共享同一个随机值, 开孔率期望仍为 p。
    返回 (H, W) float tensor。
    """
    if cell <= 1:
        coarse = torch.rand(H, W, generator=generator)
        return (coarse < p).float().to(device)

    Hc, Wc = -(-H // cell), -(-W // cell)  # ceil division
    coarse = torch.rand(Hc, Wc, generator=generator)
    m = (coarse < p).float()
    m = m.repeat_interleave(cell, dim=0).repeat_interleave(cell, dim=1)
    return m[:H, :W].to(device)


class ExperimentCUP:
    """消融实验用前向模型: 真实 CUP 前向 + 块状 mask + 高斯测量噪声."""

    def __init__(self, H=256, W=256, T=8, shear=1, cell=1, noise_std=0.01,
                 p=0.5, device="cpu", seed=None):
        self.H, self.W, self.T = H, W, T
        self.cell = cell
        self.noise_std = noise_std
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        mask = build_block_mask(H, W, cell, p, generator=g, device=device)
        self.cup = CUPForwardModel(H=H, W=W, T=T, shear_per_frame=shear,
                                   real_cup=True, mask=mask)
        self.cup.to(device)

    # ---- 前向/伴随（直通给迭代法与 DD-MADUN 用）----
    @property
    def forward(self):
        return self.cup.forward

    @property
    def adjoint(self):
        return self.cup.adjoint

    @property
    def mask(self):
        return self.cup.mask

    @property
    def shear(self):
        return self.cup.shear

    @property
    def Wm(self):
        return self.W + (self.T - 1) * self.cup.shear

    def measure(self, frames, generator=None):
        """frames: (B,T,H,W) GT → (B,H,Wm) 含噪测量."""
        E = self.cup.forward(frames)
        if self.noise_std > 0:
            noise = torch.randn(E.shape, generator=generator, device=E.device)
            E = E + self.noise_std * noise
        return E

    def measure_groups(self, scene_tensor, T, generator=None):
        """把整段场景按窗口 T 切组生成测量（由 data.group_starts 决定窗口）.

        scene_tensor: (T_total, H, W). 返回 [(start, E, gt)] 列表, E: (1,H,Wm)。
        """
        outs = []
        starts = range(0, scene_tensor.shape[0] - T + 1, T)  # 默认不重叠
        for s in starts:
            gt = scene_tensor[s:s + T].unsqueeze(0)  # (1,T,H,W)
            E = self.measure(gt, generator=generator)
            outs.append((s, E, gt))
        return outs
