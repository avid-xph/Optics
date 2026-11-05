"""统一消融评测入口.

实验（对应 PPT 第 4 页）:
  exp=T     : T ∈ 列表, 编码单元=1, 噪声=基准
  exp=cell  : cell ∈ 列表, T=基准
  exp=joint : (T, cell) 网格
  exp=noise : noise_std ∈ 列表, T/固定

用法示例（本地 CPU 冒烟）:
  python -m sci_ablation.run_ablation --exp T --Ts 8 --scenes traffic \
      --methods twist bp --iters 30 --out results/ablation/smoke.csv
  python -m sci_ablation.run_ablation --exp noise --noise 0 0.01 0.05 \
      --scenes drop runner --methods admm_l1 --iters 30

完整矩阵（远程 GPU / 长时）:
  python -m sci_ablation.run_ablation --exp joint \
      --Ts 8 12 16 20 24 28 32 --cells 1 2 3 4 5 6 8 10 \
      --methods twist admm_l1 admm_tv ddmadun tfree e2e \
      --ckpt tfree=output_tfree_mix/best_model.pth e2e=output_e2e_T{T}/best_model.pth
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import compute_psnr, compute_ssim
from sci_ablation import data
from sci_ablation.forward import ExperimentCUP
from sci_ablation.methods import build_method

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_group(method, cup, gt, T, device):
    """一个测量窗口的评测: 返回 (psnr, ssim, 耗时秒)."""
    E = cup.measure(gt).to(device)          # (1,H,Wm)
    t0 = time.time()
    rec = method.reconstruct(E, cup, T)
    dt = time.time() - t0
    if rec.shape[-1] != gt.shape[-1]:       # 迭代法可能返回 (B,T,H,Wm)? 防御性裁剪
        rec = rec[..., :gt.shape[-1]]
    psnr = float(compute_psnr(torch.clamp(rec, 0, 1), gt))
    ssim = float(compute_ssim(torch.clamp(rec, 0, 1), gt))
    return psnr, ssim, dt


def evaluate(method, scene, T, cell, noise_std, H, W, shear,
             seed=0, device="cpu", mode="disjoint"):
    """一个 (方法, 场景, T, cell, noise) 组合 → 每窗口一行记录."""
    rows = []
    cup = ExperimentCUP(H=H, W=W, T=T, shear=shear, cell=cell,
                        noise_std=noise_std, device=device, seed=seed)
    for s, gt in data.iter_groups(scene, T, mode):
        gt = gt.to(device)
        try:
            psnr, ssim, dt = run_group(method, cup, gt, T, device)
            err = ""
        except Exception as e:              # T 不匹配等方法级失败 → 记录原因继续
            psnr = ssim = float("nan"); dt = 0.0
            err = f"{type(e).__name__}: {e}"[:120]
        rows.append(dict(psnr=psnr, ssim=ssim, sec=dt, err=err))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["T", "cell", "joint", "noise"])
    ap.add_argument("--Ts", type=int, nargs="+", default=[8])
    ap.add_argument("--cells", type=int, nargs="+", default=[1])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.01])
    ap.add_argument("--scenes", nargs="+", default=data.FILES)
    ap.add_argument("--methods", nargs="+", default=["twist"])
    ap.add_argument("--iters", type=int, default=None, help="迭代法迭代数（本地小跑用）")
    ap.add_argument("--ckpt", nargs="+", default=[], metavar="METHOD=PATH",
                    help="网络方法的权重, 如 tfree=output_tfree_mix/best_model.pth "
                         "e2e=output_e2e_T{T}/best_model.pth（{T} 按当前 T 替换）")
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--shear", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", choices=["disjoint", "overlap"], default="disjoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    args = ap.parse_args()
    ckpts = dict(kv.split("=", 1) for kv in args.ckpt)

    # 实验矩阵展开
    if args.exp == "T":
        grid = [(t, 1, 0.01) for t in args.Ts]
    elif args.exp == "cell":
        grid = [(args.Ts[0], c, 0.01) for c in args.cells]
    elif args.exp == "joint":
        grid = [(t, c, 0.01) for t in args.Ts for c in args.cells]
    else:
        grid = [(args.Ts[0], args.cells[0], n) for n in args.noise]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fieldnames = ["exp", "method", "scene", "T", "cell", "noise", "group_start",
                  "psnr", "ssim", "sec", "err"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for name in args.scenes:
            scene = data.load_scene(name)
            H, W = scene.shape[1], scene.shape[2]
            if (H, W) != (args.H, args.W):
                print(f"[skip] {name}: {H}x{W} != {args.H}x{args.W}")
                continue
            for T, cell, noise in grid:
                if scene.shape[0] < T:
                    print(f"[skip] {name}: {scene.shape[0]} frames < T={T}")
                    continue
                for mkey in args.methods:
                    cfg = {"device": args.device}
                    if args.iters is not None:
                        cfg["iter_kw"] = {"n_iters": args.iters}
                    if mkey in ("ddmadun", "e2e"):
                        cfg["T"] = T
                    if mkey in ckpts:
                        cfg["ckpt"] = ckpts[mkey].replace("{T}", str(T))
                    method = build_method(mkey, cfg)
                    rows = evaluate(method, scene, T, cell, noise,
                                    H, W, args.shear, args.seed, args.device,
                                    args.group)
                    for (s, _), r in zip(data.iter_groups(scene, T, args.group), rows):
                        wcsv.writerow(dict(exp=args.exp, method=mkey, scene=name,
                                           T=T, cell=cell, noise=noise,
                                           group_start=s, **r))
                    ok = [r for r in rows if not r["err"]]
                    if ok:
                        mp = sum(r["psnr"] for r in ok) / len(ok)
                        print(f"[{name}] T={T} cell={cell} noise={noise} {mkey}: "
                              f"PSNR={mp:.2f} ({len(ok)}/{len(rows)} groups)")
                    elif rows:
                        print(f"[{name}] T={T} cell={cell} noise={noise} {mkey}: "
                              f"all failed ({rows[0]['err'][:60]})")
                    else:
                        print(f"[{name}] T={T} cell={cell} noise={noise} {mkey}: "
                              f"no groups (scene too short for T={T})")
                    f.flush()
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
