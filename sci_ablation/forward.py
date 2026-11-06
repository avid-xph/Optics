"""前向模型扩展: 块状编码 mask（编码单元 c×c）+ 变 T + 测量噪声.

不改动原始 cup_simulator.py（保持复现包完整），在此包装:
  - build_block_mask: c×c 像元共享同一随机 0/1 编码值, 开孔率 p=0.5
    （GT 尺寸保持不变: 粗网格 ceil(H/c)×ceil(W/c) 最近邻上采样, 边缘单元可
      能不足 c, 属边界效应, 保证各实验对比同一 GT）
  - ExperimentCUP: 组合 前向算子 + 噪声注入, 统一实验入口
"""
import math

import torch
import torch.nn.functional as F

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


def image_mask(mask, psf_sigma=0.0, shift=0.0):
    """mask 经中继光学成像到 CCD 上的实际样子（任务⑥）.

    psf_sigma: 高斯 PSF 标准差（CCD 像元）; shift: mask 与像元网格的亚像元错位
    （两个方向相同, 像元）。编码单元在 CCD 上只有 cell 个像元宽时, 模糊和错位会把
    相邻单元混在一起——这就是奈奎斯特条件（单元成像尺寸 ≥ 2 像元）要避免的。
    """
    m = mask[None, None]
    if psf_sigma > 0:
        r = int(math.ceil(3 * psf_sigma))
        x = torch.arange(-r, r + 1, dtype=m.dtype, device=m.device)
        k = torch.exp(-x ** 2 / (2 * psf_sigma ** 2))
        k = k / k.sum()
        m = F.pad(m, (r, r, r, r), mode="reflect")
        m = F.conv2d(m, k.view(1, 1, 1, -1))
        m = F.conv2d(m, k.view(1, 1, -1, 1))
    if shift:
        n, f = int(math.floor(shift)), shift - math.floor(shift)
        m = torch.roll(m, (n, n), dims=(2, 3))
        m = (1 - f) * m + f * torch.roll(m, 1, dims=3)      # 线性插值实现亚像元平移
        m = (1 - f) * m + f * torch.roll(m, 1, dims=2)
    return m[0, 0]


class ExperimentCUP:
    """消融实验用前向模型: 真实 CUP 前向 + 块状 mask + 高斯测量噪声.

    psf_sigma / mask_shift 非 0 时, 测量用成像后的物理 mask, 重建仍用名义 mask
    （等价于按设计图标定的真实系统）, 用来研究编码单元与奈奎斯特采样（任务⑥）。
    """

    def __init__(self, H=256, W=256, T=8, shear=1, cell=1, noise_std=0.01,
                 p=0.5, device="cpu", seed=None, psf_sigma=0.0, mask_shift=0.0):
        self.H, self.W, self.T = H, W, T
        self.cell = cell
        self.noise_std = noise_std
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        mask = build_block_mask(H, W, cell, p, generator=g, device=device)
        self.cup = CUPForwardModel(H=H, W=W, T=T, shear_per_frame=shear,
                                   real_cup=True, mask=mask)
        self.cup.to(device)
        self.true_cup = self.cup
        if psf_sigma > 0 or mask_shift:
            self.true_cup = CUPForwardModel(H=H, W=W, T=T, shear_per_frame=shear,
                                            real_cup=True,
                                            mask=image_mask(mask, psf_sigma, mask_shift))
            self.true_cup.to(device)

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
        """frames: (B,T,H,W) GT → (B,H,Wm) 含噪测量（用物理 mask）."""
        E = self.true_cup.forward(frames)
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
