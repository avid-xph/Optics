"""三类重建方法的统一注册表.

每个方法实现为类, 统一接口:
    reconstruct(E, cup, T) -> (B,T,H,W) in [0,1] 附近
  E: (B,1,H,Wm) 含噪测量; cup: ExperimentCUP（自带 mask/cell/noise 设置）。

  iterative : TwIST / ADMM-L1 / ADMM-TV / BP —— 对 H,W,T 不敏感（即插即用）
  ddmadun   : 现有 P4 权重（仅 T=8）, 用于固定 T=8 的消融点
  tfree     : dd_madun_v3_tfree.DD_MADUNv3, 同一套权重可评任意 T（任务①）
  e2e       : E2ECubeNet, 按训练时的 T 加载权重
"""
import os
import torch

from baselines import bp_reconstruct
from sci_ablation import iterative_torch
from sci_ablation.e2e_model import E2ECubeNet, build_e2e_input

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IterativeMethod:
    """TwIST / ADMM-L1 / ADMM-TV / BP, 任意 T 即插即用（torch 实现, 跟随 cup 所在 device）."""

    name = "iterative"

    def __init__(self, algo="twist", **kwargs):
        assert algo in ("twist", "admm_l1", "admm_tv", "bp")
        self.algo = algo
        self.kw = kwargs

    def reconstruct(self, E, cup, T):
        if self.algo == "bp":
            return bp_reconstruct(E, cup.adjoint)
        fn = getattr(iterative_torch, self.algo)
        return fn(E, cup.forward, cup.adjoint, T, **self.kw)


class DDMADUNMethod:
    """DD-MADUN v3 deep unfolding. 现有权重仅 T=8, 其他 T 报错跳过."""

    name = "ddmadun"

    def __init__(self, ckpt=None, T=8, device="cpu"):
        from dd_madun_v3 import DD_MADUNv3
        self.T = T
        ckpt = ckpt or os.path.join(_BASE, "checkpoints", "best_model.pth")
        self.model = DD_MADUNv3(T=T, H=None, W=None, K=8, K1=3, ng=6, nf=96, nb=8).to(device)
        data = torch.load(ckpt, map_location=device, weights_only=False)
        sd = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
        self.model.load_state_dict(sd)
        self.model.eval()
        self.device = device

    def reconstruct(self, E, cup, T):
        if T != self.T:
            raise RuntimeError(f"DD-MADUN 权重绑定 T={self.T}, 无法评测 T={T}"
                               "（这正是任务①要解决的依赖）")
        with torch.no_grad():
            I, _ = self.model(E.to(self.device), cup.forward, cup.adjoint)
        return torch.clamp(I, 0, 1)


class TFreeMethod:
    """DD-MADUN T-free (dd_madun_v3_tfree), 一套权重跑任意 T.
    结构 (K, K1, nf, nb) 从权重推断, ng / temporal 从 ckpt config 读."""

    name = "tfree"

    def __init__(self, ckpt, ng=None, final_prox=True, device="cpu"):
        from dd_madun_v3_tfree import DD_MADUNv3
        from ckpt_utils import resolve_model_cfg
        raw = torch.load(ckpt, map_location=device, weights_only=False)
        sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
        c = resolve_model_cfg(raw, overrides={"ng": ng})
        if c.get("legacy_T"):
            raise RuntimeError(f"{ckpt} 是 T-bound 权重 (T={c['legacy_T']}), 不是 T-free")
        self.model = DD_MADUNv3(K=c["K"], K1=c["K1"], ng=c["ng"], nf=c["nf"], nb=c["nb"],
                                temporal=c["temporal"], final_prox=final_prox).to(device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.device = device

    def reconstruct(self, E, cup, T):
        with torch.no_grad():
            I, _ = self.model(E.to(self.device), cup.forward, cup.adjoint)
        return torch.clamp(I, 0, 1)


class E2EMethod:
    """End-to-end 网络, 按训练时的 T 加载权重, 其他 T 报错跳过."""

    name = "e2e"

    def __init__(self, ckpt, T=8, nf=48, nb=2, device="cpu"):
        self.T = T
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            cfg = sd.get("config") or {}       # train_e2e.py 保存的结构参数优先
            nf, nb = cfg.get("nf", nf), cfg.get("nb", nb)
            sd = sd["model_state_dict"]
        self.model = E2ECubeNet(T=T, nf=nf, nb=nb).to(device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.device = device

    def reconstruct(self, E, cup, T):
        if T != self.T:
            raise RuntimeError(f"E2E 网络绑定 T={self.T}, 无法评测 T={T}")
        E = E.to(self.device)
        with torch.no_grad():
            x = build_e2e_input(cup, E)
            out = self.model(x)
        return torch.clamp(out, 0, 1)


REGISTRY = {
    "twist":     lambda cfg: IterativeMethod("twist", **cfg.get("iter_kw", {})),
    "admm_l1":   lambda cfg: IterativeMethod("admm_l1", **cfg.get("iter_kw", {})),
    "admm_tv":   lambda cfg: IterativeMethod("admm_tv", **cfg.get("iter_kw", {})),
    "bp":        lambda cfg: IterativeMethod("bp"),
    "ddmadun":   lambda cfg: DDMADUNMethod(ckpt=cfg.get("ckpt"), T=cfg.get("T", 8),
                                           device=cfg.get("device", "cpu")),
    "tfree":     lambda cfg: TFreeMethod(ckpt=cfg["ckpt"], ng=cfg.get("ng"),
                                         final_prox=cfg.get("final_prox", True),
                                         device=cfg.get("device", "cpu")),
    "e2e":      lambda cfg: E2EMethod(ckpt=cfg["ckpt"], T=cfg.get("T", 8),
                                       nf=cfg.get("nf", 48), nb=cfg.get("nb", 2),
                                       device=cfg.get("device", "cpu")),
}


def build_method(key, cfg=None):
    cfg = cfg or {}
    return REGISTRY[key](cfg)
